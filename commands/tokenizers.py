import json
import logging
import shutil
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, logging as hf_logging
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from commands.configs import (
    BYTE_LLM_PREDICTION_DATA,
    BYTELEVEL_TOK_FOLDER,
    FINEWEBEDU_REPO_ID,
    HF_USERNAME,
    TOK_REPO_ID,
)
from commands.extract import SUPPORTED_MODELS
from src.utilities import get_logger

app = typer.Typer()

ADD_PREFIX_SPACE = True  # Note that we will add a prefix_space to the pre_tokenizer
PAD_TOKEN = "<|padding|>"
EOS_TOKEN = "<|endoftext|>"
DEFAULT_TOKENIZER_SIZES = [8_064, 16_000, 32_000, 64_000, 128_000, 256_000]

# Create logger
logger = get_logger("tokenizer")
hf_logging.set_verbosity_info()  # or _debug for more info


class InfoTokenizerTrainer:
    VALID_MERGE_TYPES = [
        "min-mean-pre-merge",
        "min-mean-post-merge",
        "frequency",
        "minimise-deviation",
        "frequency-mean-post-merge",
        "mutual-information",
    ]

    def __init__(
        self,
        dataset: Dataset,
        byte_tokenizer: PreTrainedTokenizerFast,
        measure: str,
        merge_type: str,
        avoid_merge_ids: list = None,
        frequency_threshold: int = None,
        logger: logging.Logger = None,
    ) -> None:
        """Class for training a merge-based tokenizer using information measures derived from a byte-level LM.
        To avoid merging with spaces, set invalid_post_tokens to [tokenizer.encode(' ')[0]]

        Args:
            dataset (Dataset): The dataset to use for training containing information measures for each token.
            byte_tokenizer (ByteLevelBPETokenizer): The byte-level tokenizer to use for the initial vocabulary.
            measure (str): The information measure to use ('Entropy', 'Surprisal', etc.).
            merge_type (str): The type of merge to perform (see VALID_MERGE_TYPES).
            avoid_merge_ids (list, optional): List of token ids that should not be the second token in a merge.
            frequency_threshold (int, optional): Frequency threshold for merging tokens.
            logger (logging.Logger, optional): Logger for debugging and information.
        """
        if measure not in dataset.column_names:
            raise ValueError(f"Measure '{measure}' not found in dataset columns.")
        if merge_type not in self.VALID_MERGE_TYPES:
            raise ValueError(f"Merge type '{merge_type}' not valid. Choose from {self.VALID_MERGE_TYPES}.")

        self.merge_type = merge_type
        self.eos_token_id = byte_tokenizer.eos_token_id
        self.pad_token_id = byte_tokenizer.pad_token_id
        if self.eos_token_id is None:
            raise ValueError("Byte tokenizer must have an EOS token.")
        self.frequency_threshold = frequency_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger if logger else get_logger("tokenizer")

        # Initial vocabulary map
        self.id_to_token = {id: token for token, id in byte_tokenizer.vocab.items()}

        # Create ids and signal tensors for processing
        self.logger.info("Creating ids and signal tensors...")
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.int64).to(self.device)
        inf_value = torch.tensor([1e9], dtype=torch.float32).to(self.device)
        self.ids = torch.cat(
            [torch.cat([torch.tensor(x, dtype=torch.int64).to(self.device), eos_token]) for x in dataset["input_ids"]]
        )
        self.signal = torch.cat(
            [torch.cat([torch.tensor(x, dtype=torch.float32).to(self.device), inf_value]) for x in dataset[measure]]
        )
        self.logger.info("Ids and signal tensors created.")

        # Create mask for invalid positions. EOS and PAD can't be the first OR second token in a merge,
        # but invalid_post_tokens are invalid only for the second token in a merge (e.g. the space token
        # can be the first token in a merge, but not the second).
        self.mask = (self.ids == byte_tokenizer.pad_token_id) | (self.ids == byte_tokenizer.eos_token_id)
        self.mask = self.mask | self.mask.roll(1)
        if avoid_merge_ids is not None:
            for token in avoid_merge_ids:
                if token not in self.id_to_token.keys():
                    raise ValueError(f"Token '{token}' not found in tokenizer vocabulary.")
                self.mask = self.mask | (self.ids == token)

        self.merges = []

    def create_merge(self) -> None:
        """Create a merge based on the specified merge type and information measure. Increases
        the vocabulary size by one by merging the two tokens with the lowest score.
        """

        # Create unique ID for each pair of adjacent tokens using the token ID and previous token ID
        num_ids = len(self.id_to_token)
        pairs = self.ids * (num_ids + 1) + self.ids.roll(1)
        unique_pairs, inverse_indices = torch.unique(pairs, return_inverse=True)
        pair_counts = torch.bincount(inverse_indices)

        # Create masked signal
        masked_signal = self.signal.clone()
        masked_signal[self.mask] = 1e9

        pair_score = torch.zeros(unique_pairs.size(0), device=self.device)
        if self.merge_type == "min-mean-pre-merge":
            pair_score.index_add_(0, inverse_indices, masked_signal)
            pair_score /= pair_counts
        elif self.merge_type == "min-mean-post-merge":
            # Instead of the score at the second token, use the sum of the scores of both tokens
            pair_score.index_add_(0, inverse_indices, masked_signal)
            prev_token_index = inverse_indices.roll(-1)
            pair_score.index_add_(0, prev_token_index, masked_signal)
            pair_score /= pair_counts
        elif self.merge_type == "frequency":
            # Use the frequency of the pair, like BPE
            # but negated since the we want the most frequent
            pair_score = -pair_counts
            # Add a large value to any invalid positions
            pair_score.index_add_(0, inverse_indices, (self.mask * 1e9).to(pair_score.dtype))
        elif self.merge_type == "mutual-information":
            # Use the mutual information of the pair, like WordPiece (also negated since we want the max)
            token_counts = torch.bincount(self.ids)
            left_count = token_counts[unique_pairs % (num_ids + 1)]
            right_count = token_counts[unique_pairs // (num_ids + 1)]
            pair_score = - pair_counts / (left_count * right_count)
            # Add a large value to any invalid positions
            pair_score.index_add_(0, inverse_indices, (self.mask * 1e9).to(pair_score.dtype))

        elif self.merge_type == "frequency-mean-post-merge":
            # A combination of frequency and mean, select the most frequent pair
            # where the sum of the scores is below the mean
            mean_post_merge = torch.zeros(unique_pairs.size(0), device=self.device)
            mean_post_merge.index_add_(0, inverse_indices, masked_signal)
            prev_token_index = inverse_indices.roll(-1)
            mean_post_merge.index_add_(0, prev_token_index, masked_signal)
            mean_post_merge /= pair_counts

            # Only keep pairs that are below the mean
            total_positions = masked_signal.size(0)
            total_signal = masked_signal.sum()
            adjusted_means = total_signal / (total_positions - pair_counts)
            pair_score = -pair_counts
            pair_score[mean_post_merge > adjusted_means] = 1e9

        elif self.merge_type == "minimise-deviation":
            # Choose the pair that minimises the deviation from the mean, thus
            # bringing us the closest to a flatter distribution with each merge.
            # unfortunately, this implementation is currently far too slow

            total_positions = masked_signal.size(0)
            total_signal = masked_signal.sum()

            # Pre-allocate
            deviations = torch.zeros_like(masked_signal, dtype=torch.float32).to(self.device)
            merge_positions = torch.zeros_like(masked_signal, dtype=torch.bool).to(self.device)

            for i, pair_id in enumerate(unique_pairs):
                merge_positions[:] = pairs == pair_id
                num_merges = pair_counts[i]
                if num_merges < 30:  # slight efficiency gain
                    pair_score[i] = 1e20
                    continue
                mean = total_signal / (total_positions - num_merges)
                deviations[:] = (
                    masked_signal + masked_signal.roll(-1) * merge_positions.roll(-1) - mean
                ) * ~merge_positions
                total_deviation = (deviations**2).sum() / (total_positions - num_merges)
                pair_score[i] = total_deviation

                # # This is the original implementation, clearer but slower
                # tmp_signal = masked_signal.clone()
                # merge_positions = pairs == pair_id
                # tmp_signal[merge_positions.roll(-1)] += tmp_signal[merge_positions]
                # tmp_signal = tmp_signal[~merge_positions]
                # deviation = ((tmp_signal - tmp_signal.mean()) ** 2) / tmp_signal.size(0)).sum()
                # pair_score[i] = deviation
        else:
            raise ValueError(f"Merge type '{self.merge_type}' not valid. Choose from {self.VALID_MERGE_TYPES}.")

        # Filter out pairs that are below the frequency threshold
        if self.frequency_threshold is not None:
            pair_score[pair_counts < self.frequency_threshold] = 1e9

        # Find the pair with the minimum score
        min_pair = unique_pairs[pair_score.argmin()]
        merge_positions = pairs == min_pair

        # Decode unique pair ID to get left and right phonemes
        left_token = min_pair % (num_ids + 1)
        left_token = self.id_to_token[left_token.item()]
        right_token = min_pair // (num_ids + 1)
        right_token = self.id_to_token[right_token.item()]

        # Create new token and add to vocabulary
        joined = left_token + right_token
        if joined in self.id_to_token.values():
            raise RuntimeError(f"Error: merge already exists:  '{left_token}' + '{right_token}' --> '{joined}'")
            new_id = [k for k, v in self.id_to_token.items() if v == joined][0]
        else:
            new_id = num_ids
            self.id_to_token[new_id] = joined
        num_merges = torch.sum(merge_positions).item()
        self.merges.append((left_token, right_token, joined, num_merges))

        # Apply the merge everywhere
        self.ids[merge_positions.roll(-1)] = new_id
        self.signal[merge_positions.roll(-1)] += self.signal[merge_positions]
        self.signal = self.signal[~merge_positions]
        self.ids = self.ids[~merge_positions]
        self.mask = self.mask[~merge_positions]

        self.logger.debug(f"Merge created: {left_token} + {right_token} -> {joined} with {num_merges} merged tokens.")

    def train(self, vocab_size: int, show_progress=True) -> None:
        """Train the tokenizer by performing merges until the desired vocabulary size is reached.

        Args:
            vocab_size (int): The desired vocabulary size.
        """
        self.logger.info(f"Starting training with vocab size {vocab_size}...")
        if show_progress:
            with tqdm(total=vocab_size) as pbar:
                pbar.update(100 * (len(self.id_to_token) // 100))
                while len(self.id_to_token) < vocab_size:
                    self.create_merge()
                    if len(self.id_to_token) % 100 == 0:
                        pbar.update(100)
        else:
            while len(self.id_to_token) < vocab_size:
                self.create_merge()
                if len(self.id_to_token) % 100 == 0:
                    self.logger.info(f"Current vocab size: {len(self.id_to_token)}")
        self.logger.info("Training complete.")

    def get_vocab(self) -> dict:
        """Get the final vocabulary as a dictionary mapping tokens to IDs.

        Returns:
            dict: The final vocabulary sorted by token ID.
        """
        return {self.id_to_token[k]: k for k in sorted(self.id_to_token)}

    def get_merges(self) -> list:
        """Get the list of merges performed during training.

        Returns:
            list: A list of tuples representing the merges performed.
        """
        return [(merge[0], merge[1]) for merge in self.merges]

    def save_vocab_and_merges_data(self, path: Path):
        """Saves vocab.json, merges.txt and merges_data.csv files to the specified path.

        Note that the merges_data.txt file is not the same as merges.txt used by the
        huggingface tokenizer, as it also includes the number of merges performed
        for each merge, which may be useful for later analysis.

        Args:
            path (Path): The path to save the vocabulary and merges data.

        """
        with open(path / "vocab.json", "w") as f:
            json.dump(self.get_vocab(), f)

        with open(path / "merges_data.csv", "w") as f:
            merges_df = pd.DataFrame(self.merges, columns=["left_token", "right_token", "joined_token", "num_merges"])
            merges_df.to_csv(f, index=False)

        with open(path / "merges.txt", "w") as f:
            f.write("#version: 0.2\n")
            for merge in self.get_merges():
                f.write(f"{merge[0]} {merge[1]}\n")

    def create_tokenizer(self) -> PreTrainedTokenizerFast:
        """Create a BPE tokenizer using the trained vocabulary and merges.
        Returns:
            PreTrainedTokenizerFast: The subword BPE-like tokenizer with custom merges.
        """
        tokenizer = Tokenizer(models.BPE(vocab=self.get_vocab(), merges=self.get_merges()))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFD()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ADD_PREFIX_SPACE, use_regex=True)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.decoder = decoders.ByteLevel()

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token=self.id_to_token[self.pad_token_id],
            unk_token=None,
            bos_token=None,
            eos_token=self.id_to_token[self.eos_token_id],
            add_prefix_space=ADD_PREFIX_SPACE,
        )

        return wrapped_tokenizer


@app.command()
def create_subwordlevel(
    model_type: Annotated[
        str,
        typer.Argument(
            help=f"Type of model whose predictions are used to train subwords. Supported models: {SUPPORTED_MODELS}"
        ),
    ],
    measure: Annotated[str, typer.Argument(help="Measure to use for training the subword tokenizer (e.g. Entropy)")],
    merge_type: Annotated[
        str,
        typer.Argument(help=f"Type of merge to perform. Supported options: {InfoTokenizerTrainer.VALID_MERGE_TYPES}"),
    ],
    merge_spaces: Annotated[bool, typer.Option(help="If True, allow multi-word merges.")] = False,
    frequency_threshold: Annotated[int, typer.Option(help="Frequency threshold for merging tokens.")] = 20,
    num_training_rows: Annotated[int, typer.Option(help="Number of training rows to use.")] = 100000,
    vocab_sizes: Annotated[
        list[int], typer.Option(help="Vocabulary sizes for the tokenizer.")
    ] = DEFAULT_TOKENIZER_SIZES,
) -> None:
    
    if measure == "SpaceProbability":
        measure = "Space Probability"

    tokenizer_name = f"{model_type}_{measure}_{merge_type}"
    folder_path = Path(TOK_REPO_ID) / tokenizer_name
    api = HfApi()

    logger.info(f"💡 Will save the tokenizers locally to to: {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Sort vocab_sizes if not already sorted
    vocab_sizes.sort()
    logger.info(f"Using vocab sizes: {vocab_sizes}")

    logger.info("⚙️ Loading bytelevel tokenizer and byte LLM data")
    byte_tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder=BYTELEVEL_TOK_FOLDER)
    dataset = load_dataset(f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}", name=BYTE_LLM_PREDICTION_DATA, split=model_type)

    # Limit the dataset to the specified number of rows
    if num_training_rows > 0:
        dataset = dataset.select(range(num_training_rows))
    logger.info(f"Using {len(dataset)} rows for training")

    logger.info("⚙️ Creating the InfoTokenizer Trainer")
    avoid_merge_ids = None if merge_spaces else [byte_tokenizer.encode(" ")[0]]
    trainer = InfoTokenizerTrainer(
        dataset=dataset,
        byte_tokenizer=byte_tokenizer,
        measure=measure,
        merge_type=merge_type,
        avoid_merge_ids=avoid_merge_ids,
        frequency_threshold=frequency_threshold,
        logger=logger,
    )

    logger.info("⚙️ Training the InfoTokenizer")
    for vocab_size in vocab_sizes:
        folder_path_specific = folder_path / str(vocab_size)
        folder_path_specific.mkdir(parents=True, exist_ok=True)
        trainer.train(vocab_size=vocab_size)
        tokenizer = trainer.create_tokenizer()
        tokenizer.save_pretrained(str(folder_path_specific))
        trainer.save_vocab_and_merges_data(folder_path_specific)
        logger.info(f"✅ Successfully trained a tokenizer with a vocabulary size of {vocab_size}")

        repo_id = f"{HF_USERNAME}/{folder_path.parent}"
        logger.info(f"🆙 Uploading the tokenizer to {repo_id} on the HF Hub")

        path_in_repo = folder_path.name + f"_{vocab_size}"
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=folder_path_specific,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type="model",
            revision="main",
        )
        logger.info(f"✅ Successfully uploaded the tokenizer to {repo_id}")

    shutil.rmtree(folder_path, ignore_errors=True)


@app.command()
def create_bytelevel() -> None:
    folder_path = Path(TOK_REPO_ID) / BYTELEVEL_TOK_FOLDER

    logger.info(f"💡 Will save the tokenizer locally to to: {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    logger.info("⚙️ Creating the ByteLevel tokenizer")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ADD_PREFIX_SPACE, use_regex=True)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    # We "train" on no data to add the properties that we need
    trainer = trainers.BpeTrainer(
        special_tokens=[PAD_TOKEN, EOS_TOKEN], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator([], trainer=trainer)

    # Load the tokenizer as a transformers-compatible tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        unk_token=None,
        bos_token=None,
        eos_token=EOS_TOKEN,
        add_prefix_space=ADD_PREFIX_SPACE,
    )
    wrapped_tokenizer.save_pretrained(str(folder_path))

    repo_id = f"{HF_USERNAME}/{folder_path.parent}"
    logger.info(f"🆙 Uploading the tokenizer to {repo_id} on the HF Hub")

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=folder_path, repo_id=repo_id, path_in_repo=folder_path.name, repo_type="model", revision="main"
    )

    logger.info(f"✅ Successfully created and uploaded the tokenizer to {repo_id}")

    shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == "__main__":
    app()
