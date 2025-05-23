import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
import typer
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, logging as hf_logging
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # type: ignore

from commands.configs import (
    BYTE_LLM_PREDICTION_DATA,
    BYTELEVEL_TOK_FOLDER,
    COMMONCORPUS_REPO_ID,
    FINEWEBEDU_REPO_ID,
    HF_USERNAME,
    TOK_REPO_ID,
)
from commands.extract import SUPPORTED_CORPORA, SUPPORTED_MODELS
from src.utilities import get_logger

app = typer.Typer()

ADD_PREFIX_SPACE = True  # Note that we will add a prefix_space to the pre_tokenizer
PAD_TOKEN = "<|padding|>"
EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"
DEFAULT_TOKENIZER_SIZES = [8_064, 16_000, 32_000, 64_000, 128_000, 256_000]

# Create logger
logger = get_logger("tokenizer")
hf_logging.set_verbosity_info()  # or _debug for more info


class FrequencyTokenizerTrainer:
    VALID_MERGE_TYPES = [
        "frequency",
        "mutual-information",
    ]

    def __init__(
        self,
        dataset: Dataset,
        byte_tokenizer: PreTrainedTokenizerFast,
        merge_type: str,
        frequency_threshold: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Class for training a merge-based tokenizer using the BPE or WordPiece objective.
        To avoid merging with spaces, set invalid_post_tokens to [tokenizer.encode(' ')[0]]

        Args:
            dataset (Dataset): The dataset to use for training containing information measures for each token.
            byte_tokenizer (ByteLevelBPETokenizer): The byte-level tokenizer to use for the initial vocabulary.
            measure (str): The information measure to use ('Entropy', 'Surprisal', etc.).
            merge_type (str): The type of merge to perform (see VALID_MERGE_TYPES).
            frequency_threshold (int, optional): Frequency threshold for merging tokens.
            logger (logging.Logger, optional): Logger for debugging and information.
        """
        if merge_type not in self.VALID_MERGE_TYPES:
            raise ValueError(f"Merge type '{merge_type}' not valid. Choose from {self.VALID_MERGE_TYPES}.")

        self.merges = []
        self.merge_type = merge_type
        self.frequency_threshold = frequency_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.byte_tokenizer = byte_tokenizer
        self.logger = logger if logger else get_logger("tokenizer")
        if self.byte_tokenizer.eos_token_id is None:
            raise ValueError("Byte tokenizer must have an EOS token.")

        # Initial vocabulary map
        self.id_to_token = {id: token for token, id in byte_tokenizer.vocab.items()}

        # Create ids and signal tensors for processing
        self.logger.info("Creating ids and signal tensors...")
        self.ids = torch.cat(
            torch.tensor(x, dtype=torch.int64).to(self.device) for x in dataset["input_ids"]
        )
        self.pre_token_boundaries = torch.cat(
            torch.tensor(x, dtype=torch.bool).to(self.device) for x in dataset["pre_token_boundaries"]
        )

        self.logger.info("Ids tensor created.")

    def create_merge(self) -> None:
        """Create a merge based on the specified merge type. Increases
        the vocabulary size by one by merging the two tokens with the lowest score.
        """

        # Create unique ID for each pair of adjacent tokens using the token ID and previous token ID
        num_ids = len(self.id_to_token)
        pairs = self.ids * (num_ids + 1) + self.ids.roll(1)
        pairs[self.pre_token_boundaries] = -1 # Don't allow merges across a pre-token boundary
        unique_pairs, inverse_indices = torch.unique(pairs, return_inverse=True)
        pair_counts = torch.bincount(inverse_indices)
        pair_counts[unique_pairs == -1] = 0 # Don't count the removed pairs

        pair_score = torch.zeros(unique_pairs.size(0), device=self.device)

        if self.merge_type == "frequency":
            # Use the frequency of the pair, like BPE
            pair_score = pair_counts
        elif self.merge_type == "mutual-information":
            # Use the mutual information of the pair, like WordPiece
            token_counts = torch.bincount(self.ids)
            left_count = token_counts[unique_pairs % (num_ids + 1)]
            right_count = token_counts[unique_pairs // (num_ids + 1)]
            pair_score = -pair_counts / (left_count * right_count)
        else:
            raise ValueError(f"Merge type '{self.merge_type}' not valid. Choose from {self.VALID_MERGE_TYPES}.")

        # Filter out pairs that are below the frequency threshold
        if self.frequency_threshold is not None:
            pair_score[pair_counts < self.frequency_threshold] = -1

        # Find the pair with the minimum score
        min_pair = unique_pairs[pair_score.argmax()]
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
        else:
            new_id = num_ids
            self.id_to_token[new_id] = joined
        num_merges = torch.sum(merge_positions).item()
        self.merges.append((left_token, right_token, joined, num_merges))

        # Apply the merge everywhere
        self.ids[merge_positions.roll(-1)] = new_id
        self.ids = self.ids[~merge_positions]

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
        tokenizer.pre_tokenizer = self.byte_tokenizer._tokenizer.pre_tokenizer
        tokenizer.post_processor = self.byte_tokenizer._tokenizer.post_processor
        tokenizer.decoder = self.byte_tokenizer._tokenizer.decoder

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token=self.id_to_token[self.byte_tokenizer.pad_token_id],  # type: ignore
            unk_token=None,
            bos_token=None,
            eos_token=self.id_to_token[self.byte_tokenizer.eos_token_id],  # type: ignore
            add_prefix_space=ADD_PREFIX_SPACE,
        )

        return wrapped_tokenizer


class ThresholdTokenizerTrainer:
    def __init__(
        self,
        dataset: Dataset,
        byte_tokenizer: PreTrainedTokenizerFast,
        measure: str,
        include_left_byte: bool = False,
        keep_intermediate_vocab: bool = True,
        frequency_threshold: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Class for training a threshold-based tokenizer using information measures derived from a byte-level LM.

        Args:
            dataset (Dataset): The dataset to use for training containing information measures for each token.
            byte_tokenizer (ByteLevelBPETokenizer): The byte-level tokenizer to use for the initial vocabulary.
            measure (str): The information measure to use ('Entropy', 'Surprisal', etc.).
            include_left_byte (bool, optional): Whether to include the left byte in the vocabulary.
            keep_intermediate_vocab (bool, optional): Whether to keep intermediate vocabularies during training.
            frequency_threshold (int, optional): Frequency threshold for tokens to be included in the vocabulary.
            logger (logging.Logger, optional): Logger for debugging and information.
        """
        if measure not in dataset.column_names:
            raise ValueError(f"Measure '{measure}' not found in dataset columns.")

        self.byte_tokenizer = byte_tokenizer
        self.include_left_byte = include_left_byte
        self.keep_intermediate_vocab = keep_intermediate_vocab
        self.frequency_threshold = frequency_threshold
        self.logger = logger if logger else get_logger("tokenizer")

        self.eos_token_id = byte_tokenizer.eos_token_id
        self.pad_token_id = byte_tokenizer.pad_token_id
        self.space_token_id = byte_tokenizer.encode(" ")[0]
        self.space_token = byte_tokenizer.convert_ids_to_tokens(self.space_token_id)
        if self.eos_token_id is None:
            raise ValueError("Byte tokenizer must have an EOS token.")
        self.device = torch.device("cpu")  # for some reason cpu is faster than cuda

        self.build_initial_vocab()
        # Efficiently keep track of subwords without needing to decode at every step
        self.segment_counts = defaultdict(int)

        # Create ids and signal tensors for processing
        self.logger.info("Creating ids and signal tensors...")
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.int64).to(self.device)
        inf_value = torch.tensor([1e9], dtype=torch.float32).to(self.device)
        true_value = torch.tensor([True], dtype=torch.bool).to(self.device)
        self.ids = torch.cat(
            [torch.cat([torch.tensor(x, dtype=torch.int64).to(self.device), eos_token]) for x in dataset["input_ids"]]
        )
        self.signal = torch.cat(
            [torch.cat([torch.tensor(x, dtype=torch.float32).to(self.device), inf_value]) for x in dataset[measure]]
        )
        self.pre_token_boundaries = torch.cat(
            [torch.cat([torch.tensor(x, dtype=torch.bool).to(self.device), true_value]) for x in dataset["pre_token_boundaries"]]
        )
        self.total_tokens = len(self.ids)
        self.logger.info("Ids and signal tensors created.")

        # Set all EOS, UNK and PAD tokens to large value to prevent them from being included within tokens
        self.signal[self.ids == self.eos_token_id] = inf_value
        self.signal[self.ids == byte_tokenizer.unk_token_id] = inf_value
        self.signal[self.ids == byte_tokenizer.pad_token_id] = inf_value

        self.logger.info("Sorting signal tensor for threshold training...")

        # Track the left and right boundaries of the segments
        self.left_boundaries = torch.zeros_like(self.signal, dtype=torch.int64) - 1
        self.right_boundaries = torch.zeros_like(self.signal, dtype=torch.int64) - 1
        self.sorted_indices = torch.argsort(self.signal)
        self.current_minimum_idx = 0

        self.logger.info("Signal tensor sorted.")

        self.stats = pd.DataFrame(columns=["num_moves", "vocab_size", "unique_segments", "threshold"])

    def build_initial_vocab(self):
        """Convert byte vocab to be compatible with wordpiece tokenizer."""
        byte_vocab = self.byte_tokenizer.get_vocab()
        keys = list(byte_vocab.keys())
        # Any byte can start a word or be a continuation token
        self.base_vocab = {PAD_TOKEN: 0, EOS_TOKEN: 1}
        for key in keys:
            if key != PAD_TOKEN and key != EOS_TOKEN and key != UNK_TOKEN:
                # Check if the key is a letter, if so it probably appears
                # at the start of a word, so we add it to the vocab with the space token
                if key.isalpha() and key != self.space_token:
                    self.base_vocab[self.space_token + key] = len(self.base_vocab)
                # if key != self.space_token:
                #     self.base_vocab[self.space_token + key] = len(self.base_vocab)
                self.base_vocab[key] = len(self.base_vocab)
                self.base_vocab["##" + key] = len(self.base_vocab)
        self.base_vocab[UNK_TOKEN] = len(self.base_vocab)
        self.vocab = self.base_vocab.copy()

    def update_vocab(self):
        self.vocab = self.base_vocab.copy()
        self.merges = []
        vocab_size = len(self.vocab)
        discovered = self.segment_counts.items()
        for token_ref, count in discovered:
            if count < self.frequency_threshold:
                break  # Can break here because we've sorted the segments by frequency
            is_start_of_word, token_ids = token_ref
            token = ''.join([self.byte_tokenizer.convert_ids_to_tokens(t) for t in token_ids])
            if not is_start_of_word:
                token = "##" + token
            else:
                token = self.space_token + token
            if token not in self.vocab:
                self.vocab[token] = vocab_size
                vocab_size += 1

    def wordpiece_step(self, min_idx):
        # Don't merge left if we're at a pre-token boundary
        if self.pre_token_boundaries[min_idx]:
            left_boundary = min_idx
        elif self.include_left_byte:
            if min_idx > 1 and self.left_boundaries[min_idx - 2] != -1:
                left_boundary = self.left_boundaries[min_idx - 2]
            elif min_idx > 0:
                left_boundary = min_idx - 1
            else:
                left_boundary = min_idx
        else:
            if min_idx > 0 and self.left_boundaries[min_idx - 1] != -1:
                left_boundary = self.left_boundaries[min_idx - 1]
            else:
                left_boundary = min_idx
        
        # Don't merge right if right token is a pre-token boundary
        if min_idx < self.total_tokens - 1 and self.right_boundaries[min_idx + 1] != -1 and not self.pre_token_boundaries[min_idx + 1]:
            right_boundary = self.right_boundaries[min_idx + 1]
        else:
            right_boundary = min_idx

        # Update left and right boundaries
        self.left_boundaries[left_boundary : right_boundary + 1] = left_boundary
        self.right_boundaries[left_boundary : right_boundary + 1] = right_boundary

        # Create hashable subword
        seg = self.ids[left_boundary : right_boundary + 1]
        seg_tuple = tuple(seg.tolist())
        is_start = self.ids[left_boundary - 1] == self.space_token_id and not self.pre_token_boundaries[left_boundary]
        token_ref = (bool(is_start), seg_tuple)

        self.segment_counts[token_ref] += 1

        if not self.keep_intermediate_vocab:
            # Decrease counts for the tokens we merged with
            if left_boundary != min_idx:
                prev_token_ref = (bool(is_start), tuple(self.ids[left_boundary:min_idx].tolist()))
                self.segment_counts[prev_token_ref] = max(0, self.segment_counts[prev_token_ref] - 1)
            if right_boundary != min_idx:
                next_token_ref = (False, tuple(self.ids[min_idx + 1 : right_boundary + 1].tolist()))
                self.segment_counts[next_token_ref] = max(0, self.segment_counts[next_token_ref] - 1)

    def train(self, final_vocab_size):
        # Initialize tqdm progress bar
        pbar = tqdm(total=final_vocab_size, desc="Building vocabulary", unit="items")

        while self.current_minimum_idx < self.total_tokens:
            min_idx = self.sorted_indices[self.current_minimum_idx]
            self.wordpiece_step(min_idx)

            if self.current_minimum_idx % 1000 == 0:
                self.update_vocab()
                vocab_size = len(self.vocab)
                unique_segments = len(self.segment_counts)
                self.stats = pd.concat(
                    [
                        self.stats,
                        pd.DataFrame(
                            [
                                [
                                    self.current_minimum_idx,
                                    vocab_size,
                                    unique_segments,
                                    self.signal[self.sorted_indices[self.current_minimum_idx]].item(),
                                ]
                            ],
                            columns=self.stats.columns,
                        ),
                    ],
                    ignore_index=True,
                )
                # print(f"Step {i}: Vocab size: {vocab_size}, Unique segments: {unique_segments}, Threshold: {signal[sorted_indices[i]].item()}")
                if pbar.n < vocab_size:
                    pbar.update(vocab_size - pbar.n)
                if vocab_size > final_vocab_size:
                    self.logger.info(f"Final vocab size {vocab_size} exceeds target {final_vocab_size}.")
                    self.logger.info("Filtering discovered tokens by frequency.")
                    self.vocab = {k: v for k, v in self.vocab.items() if v < final_vocab_size}
                    vocab_size = len(self.vocab)
                if vocab_size >= final_vocab_size:
                    self.logger.info(f"Final vocab size reached: {vocab_size}")
                    return self.vocab

            self.current_minimum_idx += 1

        self.logger.info("Reached end of signal tensor without reaching final vocab size.")
        self.update_vocab()
        self.logger.info(f"Final vocab size reached: {len(self.vocab)}")
        return self.vocab

    def save_vocab_and_stats(self, path: Path):
        """Saves the vocabulary to a JSON file and the stats to a CSV file.

        Args:
            path (Path): The path to save the vocabulary.
        """
        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f)
        self.logger.info(f"Vocabulary saved to {path / 'vocab.json'}")

        self.stats.to_csv(path / "stats.csv", index=False)
        self.logger.info(f"Stats saved to {path / 'stats.csv'}")

    def create_tokenizer(self) -> PreTrainedTokenizerFast:
        """Create a WordPiece tokenizer using the trained vocabulary.
        Returns:
            PreTrainedTokenizerFast: The subword WordPiece-like tokenizer with custom vocabulary.
        """
        tokenizer = Tokenizer(models.WordPiece(vocab=self.vocab, unk_token=UNK_TOKEN))  # type: ignore
        tokenizer.pre_tokenizer = self.byte_tokenizer._tokenizer.pre_tokenizer
        tokenizer.post_processor = self.byte_tokenizer._tokenizer.post_processor
        tokenizer.decoder = self.byte_tokenizer._tokenizer.decoder

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
            bos_token=EOS_TOKEN,
            eos_token=EOS_TOKEN,
            add_prefix_space=ADD_PREFIX_SPACE,
        )

        return wrapped_tokenizer


class ByteCurveTokenizerTrainer:
    def __init__(
        self,
        dataset: Dataset,
        byte_tokenizer: PreTrainedTokenizerFast,
        measure: str,
        frequency_threshold: int | float | None = None,
        threshold_percentile: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Class for training a byte-curve tokenizer using information measures derived from a byte-level LM.

        Args:
            dataset (Dataset): The dataset to use for training containing information measures for each token.
            byte_tokenizer (ByteLevelBPETokenizer): The byte-level tokenizer to use for the initial vocabulary.
            measure (str): The information measure to use ('Entropy', 'Surprisal', etc.).
            frequency_threshold (int, optional): Frequency threshold for tokens to be included in the vocabulary. If int, use
                as a threshold for the number of tokens. If float, uses as a percentile for the number of tokens to keep.
            threshold_percentile (int, optional): Besides curves, also include tokens in a span if they fall under the threshold,
                calculated as a percentile of the signal values. This is useful for very predictable tokens that might cause the curve
                to slightly increase at low signal values.
            logger (logging.Logger, optional): Logger for debugging and information.
        """
        if measure not in dataset.column_names:
            raise ValueError(f"Measure '{measure}' not found in dataset columns.")
        self.measure = measure
        self.dataset = dataset
        self.byte_tokenizer = byte_tokenizer
        self.frequency_threshold = frequency_threshold
        self.threshold_percentile = threshold_percentile
        self.logger = logger if logger else get_logger("tokenizer")

        self.space_token_id = byte_tokenizer.encode(" ")[0]
        self.space_token = byte_tokenizer.convert_ids_to_tokens(self.space_token_id)

        self.vocab = {}
        self.subwords = {}
        self.inverse_vocab = {}
        self.subword_vocab = {}
        
        if threshold_percentile is not None:
            if threshold_percentile < 0 or threshold_percentile > 100:
                raise ValueError("Threshold percentile must be between 0 and 100.")
            self.logger.info(f"Using threshold percentile of {threshold_percentile} - bytes under this value will be grouped.")
            self.logger.info(f"Getting threshold value from the dataset...")
            self.threshold = self.get_threshold()
            self.logger.info(f"Threshold value: {self.threshold}")
    
    def get_threshold(self) -> float:
        """Get the threshold value for the tokenizer.

        Returns:
            float: The threshold value.
        """
        signal_values = []
        for item in tqdm(self.dataset):
            signal_values.extend(item[self.measure])
        signal_values = np.array(signal_values)
        signal_values = signal_values[signal_values != 0]
        signal_values = signal_values[signal_values != -np.inf]
        signal_values = signal_values[signal_values != np.inf]
        signal_values = signal_values[~np.isnan(signal_values)]
        signal_values = signal_values[signal_values != 0]
        signal_values = np.sort(signal_values)
        signal_values = np.percentile(signal_values, self.threshold_percentile)

    def find_subword_boundaries(self, examples):

        subword_start_list = []
        subword_end_list = []

        for i in range(len(examples[self.measure])):

            signal = np.array(examples[self.measure][i])
            ids = np.array(examples["input_ids"][i])

            # Identify subwords according to monotonically decreasing signal
            diff_in_signal = np.diff(signal)
            diff_in_signal = np.append(1, diff_in_signal)
            diff_in_signal[ids == 1] = 1
            below_threshold = np.where((signal < self.threshold), True, False)
            decreasing = np.where((diff_in_signal < 0), True, False)
            subword_mask = np.zeros(len(signal), dtype=bool)
            subword_mask = np.where((decreasing | below_threshold), subword_mask, True)

            # Subword boundaries are whenever subword_mask flips values
            subword_boundary = np.concatenate((subword_mask, [False]))
            subword_boundary = np.diff(subword_boundary).astype(bool)

            # Even flip locations are word boundary starts, odd flip locations are word boundary ends
            subword_starts = subword_boundary.copy()
            subword_ends = subword_boundary.copy()
            true_indices = np.where(subword_boundary)[0]
            subword_starts[true_indices[1::2]] = False
            subword_ends[true_indices[::2]] = False 

            subword_start_list.append(subword_starts)
            subword_end_list.append(subword_ends)

        examples["subword_starts"] = subword_start_list
        examples["subword_ends"] = subword_end_list
        return examples
    
    def create_vocab(self, examples):
        """Create a vocabulary from the dataset and save ids to tmp_vocab."""

        valid_word_start_list = []
        valid_word_length_list = []
        ids = examples["input_ids"]
        subword_starts = examples["subword_starts"]
        subword_ends = examples["subword_ends"]
        pre_token_boundaries = examples["pre_token_boundaries"]

        for i in range(len(examples[self.measure])):
            valid_word_starts = np.zeros(len(ids[i]), dtype=bool) - 1
            valid_word_lengths = np.zeros(len(ids[i]), dtype=np.int32)
            for start, end in zip(np.where(subword_starts[i])[0], np.where(subword_ends[i])[0]):
                if pre_token_boundaries[i][end]:
                    end -= 1
                while ids[i][start] == self.space_token_id or ids[i][start] in self.byte_tokenizer.all_special_ids:
                    start += 1
                if start >= end:
                    continue
                for mid in range(start+1, end+1):
                    if pre_token_boundaries[i][mid]:
                        token = self.create_token(ids[i], pre_token_boundaries[i], start, mid)
                        if token and not token in self.byte_tokenizer.get_vocab():
                            if token in self.subwords:
                                self.subwords[token] += 1
                            else:
                                self.subwords[token] = 1
                                self.subword_vocab[token] = len(self.subword_vocab)
                            valid_word_starts[start] = self.subword_vocab[token]
                            valid_word_lengths[start] = mid - start
                        start = mid
                        while ids[i][start] == self.space_token_id or ids[i][start] in self.byte_tokenizer.all_special_ids:
                            start += 1
                if start != end:
                    token = self.create_token(ids[i], pre_token_boundaries[i], start, end+1)
                    if token and not token in self.byte_tokenizer.get_vocab():
                        if token in self.subwords:
                            self.subwords[token] += 1
                        else:
                            self.subwords[token] = 1
                            self.subword_vocab[token] = len(self.subword_vocab)
                        valid_word_starts[start] = self.subword_vocab[token]
                        valid_word_lengths[start] = end - start + 1
            valid_word_start_list.append(valid_word_starts)
            valid_word_length_list.append(valid_word_lengths)
        examples["valid_word_start_vocab"] = valid_word_start_list
        examples["valid_word_start_lengths"] = valid_word_length_list
        return examples

    def create_token(self, ids, pre_token_boundaries, start, end):
        token = ''.join([self.byte_tokenizer.convert_ids_to_tokens(t) for t in ids[start:end]])
        if ids[start-1] == self.space_token_id:
            token = self.space_token + token
        elif not pre_token_boundaries[start]:
            token = '##' + token
        return token

    def train(self, final_vocab_size):
        """ Train the tokenizer """
        logger.info("Training tokenizer...")
        if self.frequency_threshold != 0:
            self.logger.info('Finding byte curves')
            self.byte_curves(final_vocab_size)

        logger.info("Creating vocabulary from discovered subwords...")
        self.vocab = self.byte_tokenizer.get_vocab()
        self.vocab[UNK_TOKEN] = len(self.vocab)
        keys = list(self.byte_tokenizer.get_vocab().keys())
        for key in keys:
            if key != PAD_TOKEN and key != EOS_TOKEN and key != UNK_TOKEN:
                if key.isalpha():
                    self.vocab[self.space_token_id + key] = len(self.vocab)
                self.vocab["##" + key] = len(self.vocab)
        for key in self.subwords.keys():
            if not key in self.vocab:
                self.vocab[key] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if len(self.subwords) > final_vocab_size:
            self.logger.info(f"Final vocab size {len(self.subwords)} exceeds target {final_vocab_size}.")
            self.logger.info("Filtering discovered tokens by frequency.")
            self.vocab = {k: v for k, v in self.vocab.items() if v < final_vocab_size}
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        else:
            self.logger.info(f"Running BPE to learn remaining subwords.")
            self.bpe()
        self.logger.info(f"Final vocab size reached: {len(self.vocab)}")
        return self.vocab
    
    def byte_curves(self, final_vocab_size):
        """Run byte curves to learn remaining subwords."""
        self.dataset = self.dataset.map(
            self.find_subword_boundaries,
            batched=True,
            desc="Finding subword boundaries"
        )
        dataset = dataset.map(
            self.create_vocab,
            batched=True,
            batch_size=100,
            desc="Creating vocabulary from identified subwords",
            num_proc=1,
        )
        # Sort subwords by frequency
        self.subwords = sorted(self.subwords.items(), key=lambda x: x[1], reverse=True)
        logger.info(f'Found {len(self.subwords)} subwords.')
        if self.frequency_threshold is None:
            self.logger.info("Frequency threshold not provided, keeping all discovered vocabulary items.")
        if isinstance(self.frequency_threshold, int):
            logger.info(f'Using frequency threshold of {self.frequency_threshold} to filter subwords.')
            self.subwords = {k: v for k, v in self.subwords.items() if v > self.frequency_threshold}
            logger.info(f'Filtered subwords to {len(self.subwords)} items.')
        elif isinstance(self.frequency_threshold, float):
            total_items = self.frequency_threshold * final_vocab_size
            logger.info(f'Using percentage threshold, keeping most frequent subwords to reach {self.frequency_threshold}% of target vocab size.')
            if self.frequency_threshold < 0 or self.frequency_threshold > 1:
                raise ValueError("Frequency threshold must be between 0 and 1.")
            self.subwords = self.subwords[:int(total_items)]
            logger.info(f'Filtered subwords to {len(self.subwords)} items.')

    def bpe(self, final_vocab_size):
        """Run BPE to learn remaining subwords."""
        self.logger.info("Creating vector ID to apply merges and calculate frequencies")
        tmp_id_to_vocab_id = {i : self.vocab[v] for v, i in self.subword_vocab.items() if v in self.vocab}

        ids = torch.cat([torch.tensor(x, dtype=torch.int64) for x in self.dataset["input_ids"]])
        pre_token_boundaries = torch.cat([torch.tensor(x, dtype=torch.bool) for x in self.dataset["pre_token_boundaries"]])

        # Check we actually have tokens to apply merges to
        if len(tmp_id_to_vocab_id) != 0:
            valid_word_start_vocab = torch.cat([torch.tensor(x, dtype=torch.int32) for x in self.dataset["valid_word_start_vocab"]])
            valid_word_start_lengths = torch.cat([torch.tensor(x, dtype=torch.int32) for x in self.dataset["valid_word_start_lengths"]])

            # Set valid_word_start_indices to -1 if the value is not in tmp_id_to_vocab_id
            valid_word_start_vocab = np.where(np.isin(valid_word_start_vocab, list(tmp_id_to_vocab_id.keys())), valid_word_start_vocab, -1)
            valid_word_start_indices = np.where(valid_word_start_vocab != -1)[0]
            delete_mask = np.zeros(len(ids), dtype=bool)

            for i in tqdm(valid_word_start_indices[::-1]):
                length = valid_word_start_lengths[i]
                new_id = tmp_id_to_vocab_id[valid_word_start_vocab[i]]
                if ids[i-1] == self.space_token_id:
                    i -= 1
                    length += 1
                ids[i] = new_id
                delete_mask[i+1:i+length] = True

            old_length = len(ids)
            ids = np.delete(ids, np.where(delete_mask)[0])
            new_pre_token_boundaries = np.delete(pre_token_boundaries, np.where(delete_mask)[0]).bool()
            self.logger.info('Shortened data from:', old_length, 'to new length:', len(ids))

        def add_continuation_token(t):
            if not self.inverse_vocab[t].startswith("##"):
                return self.vocab["##" + self.inverse_vocab[t]]
            else:
                return t
    
        ids[~new_pre_token_boundaries] = ids[~new_pre_token_boundaries].apply_(add_continuation_token)
        
        num_ids = len(self.vocab)
        pbar = tqdm(total=final_vocab_size, desc="Generating IDs")
        while num_ids <  final_vocab_size:
            pbar.update(num_ids - pbar.n)
            # Get pair counts - could do this more efficiently by only updating
            # where we merge later
            pairs =  new_ids * (num_ids + 1) + new_ids.roll(1)
            pairs[new_pre_token_boundaries] = -1
            unique_pairs, inverse_indices = torch.unique(pairs, return_inverse=True)
            pair_counts = torch.bincount(inverse_indices)
            pair_counts[unique_pairs == -1] = 0

            # Find best pair
            best_pair = unique_pairs[pair_counts.argmax()]
            merge_positions = pairs == best_pair

            # Decode unique pair ID to get left and right id
            left_token = best_pair % (num_ids + 1)
            left_token = self.inverse_vocab[left_token.item()]
            right_token = best_pair // (num_ids + 1)
            right_token = self.inverse_vocab[right_token.item()]

            # Create new token and add to vocabulary
            joined = left_token + right_token.replace("##", "")
            if joined in self.inverse_vocab.values():
                new_id = self.vocab[joined]
            else:
                new_id = num_ids
                self.inverse_vocab[new_id] = joined
                self.vocab[joined] = new_id
                num_ids += 1

            # Apply the merge everywhere
            new_ids[merge_positions.roll(-1)] = new_id
            new_ids = new_ids[~merge_positions]
            new_pre_token_boundaries = new_pre_token_boundaries[~merge_positions]

        pbar.close()

    def save_vocab_and_stats(self, path: Path):
        """Saves the vocabulary to a JSON file and the stats to a CSV file.

        Args:
            path (Path): The path to save the vocabulary.
        """
        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f)
        self.logger.info(f"Vocabulary saved to {path / 'vocab.json'}")

    def create_tokenizer(self) -> PreTrainedTokenizerFast:
        """Create a WordPiece tokenizer using the trained vocabulary.
        Returns:
            PreTrainedTokenizerFast: The subword WordPiece-like tokenizer with custom vocabulary.
        """
        tokenizer = Tokenizer(models.WordPiece(vocab=self.vocab, unk_token=UNK_TOKEN))  # type: ignore
        tokenizer.pre_tokenizer = self.byte_tokenizer._tokenizer.pre_tokenizer
        tokenizer.post_processor = self.byte_tokenizer._tokenizer.post_processor
        tokenizer.decoder = self.byte_tokenizer._tokenizer.decoder

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN,
            bos_token=EOS_TOKEN,
            eos_token=EOS_TOKEN,
            add_prefix_space=ADD_PREFIX_SPACE,
        )

        return wrapped_tokenizer


@app.command()
def create_frequencytokenizer(
    model_type: Annotated[
        str,
        typer.Argument(
            help=f"Type of model whose predictions are used to train subwords. Supported models: {SUPPORTED_MODELS}"
        ),
    ],
    merge_type: Annotated[
        str,
        typer.Argument(help=f"Type of merge to perform. Supported options: {FrequencyTokenizerTrainer.VALID_MERGE_TYPES}"),
    ],
    corpus: Annotated[
        str,
        typer.Argument(
            help=f"Corpus to use for training the subword tokenizer. Supported corpora: {SUPPORTED_CORPORA}"
        ),
    ] = FINEWEBEDU_REPO_ID,
    frequency_threshold: Annotated[int, typer.Option(help="Frequency threshold for merging tokens.")] = 20,
    num_training_rows: Annotated[int, typer.Option(help="Number of training rows to use.")] = 100000,
    vocab_sizes: Annotated[
        list[int], typer.Option(help="Vocabulary sizes for the tokenizer.")
    ] = DEFAULT_TOKENIZER_SIZES,
) -> None:
    tokenizer_name = merge_type
    if corpus == COMMONCORPUS_REPO_ID:
        tokenizer_name += "multi"
    folder_path = Path(TOK_REPO_ID) / tokenizer_name
    api = HfApi()

    logger.info(f"💡 Will save the tokenizers locally to to: {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Sort vocab_sizes if not already sorted
    vocab_sizes.sort()
    logger.info(f"Using vocab sizes: {vocab_sizes}")

    logger.info("⚙️ Loading bytelevel tokenizer and byte LLM data")
    byte_tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder=BYTELEVEL_TOK_FOLDER)
    dataset = load_dataset(f"{HF_USERNAME}/{corpus}", name=BYTE_LLM_PREDICTION_DATA, split=model_type)

    # Limit the dataset to the specified number of rows
    if num_training_rows > 0:
        if num_training_rows > len(dataset):
            logger.warning(f"Requested {num_training_rows} rows, but only {len(dataset)} available. Using all rows.")
            num_training_rows = len(dataset)
        else:
            dataset = dataset.select(range(num_training_rows))
    logger.info(f"Using {len(dataset)} rows for training")

    logger.info("⚙️ Creating the InfoTokenizer Trainer")
    trainer = FrequencyTokenizerTrainer(
        dataset=dataset,
        byte_tokenizer=byte_tokenizer,
        merge_type=merge_type,
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
def create_thresholdtokenizer(
    model_type: Annotated[
        str,
        typer.Argument(
            help=f"Type of model whose predictions are used to train subwords. Supported models: {SUPPORTED_MODELS}"
        ),
    ],
    measure: Annotated[str, typer.Argument(help="Measure to use for training the subword tokenizer (e.g. Entropy)")],
    corpus: Annotated[
        str,
        typer.Argument(
            help=f"Corpus to use for training the subword tokenizer. Supported corpora: {SUPPORTED_CORPORA}"
        ),
    ] = FINEWEBEDU_REPO_ID,
    keep_intermediate_vocab: Annotated[bool, typer.Option(help="If True, keep intermediate vocabularies.")] = True,
    include_left_byte: Annotated[bool, typer.Option(help="If True, include left byte in merges.")] = False,
    frequency_threshold: Annotated[int, typer.Option(help="Frequency threshold for merging tokens.")] = 20,
    num_training_rows: Annotated[int, typer.Option(help="Number of training rows to use.")] = 100000,
    vocab_sizes: Annotated[
        list[int], typer.Option(help="Vocabulary sizes for the tokenizer.")
    ] = DEFAULT_TOKENIZER_SIZES,
) -> None:
    if measure == "SpaceProbability":
        measure = "Space Probability"

    tokenizer_name = (
        f"{model_type}_{measure}_threshold"
        + ("B" if not keep_intermediate_vocab else "")
        + ("L" if include_left_byte else "")
    )
    folder_path = Path(TOK_REPO_ID) / tokenizer_name
    api = HfApi()

    logger.info(f"💡 Will save the tokenizers locally to to: {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Sort vocab_sizes if not already sorted
    vocab_sizes.sort()
    logger.info(f"Using vocab sizes: {vocab_sizes}")

    logger.info("⚙️ Loading bytelevel tokenizer and byte LLM data")
    byte_tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder=BYTELEVEL_TOK_FOLDER)
    dataset = load_dataset(f"{HF_USERNAME}/{corpus}", name=BYTE_LLM_PREDICTION_DATA, split=model_type)

    # Limit the dataset to the specified number of rows
    if num_training_rows > 0:
        if num_training_rows > len(dataset):
            logger.warning(f"Requested {num_training_rows} rows, but only {len(dataset)} available. Using all rows.")
            num_training_rows = len(dataset)
        else:
            dataset = dataset.select(range(num_training_rows))
    logger.info(f"Using {len(dataset)} rows for training")

    logger.info("⚙️ Creating the InfoTokenizer Trainer")
    trainer = ThresholdTokenizerTrainer(
        dataset=dataset,
        byte_tokenizer=byte_tokenizer,
        measure=measure,
        include_left_byte=include_left_byte,
        keep_intermediate_vocab=keep_intermediate_vocab,
        frequency_threshold=frequency_threshold,
        logger=logger,
    )

    logger.info("⚙️ Training the TresholdTokenizer")
    for vocab_size in vocab_sizes:
        folder_path_specific = folder_path / str(vocab_size)
        folder_path_specific.mkdir(parents=True, exist_ok=True)
        trainer.train(final_vocab_size=vocab_size)
        tokenizer = trainer.create_tokenizer()
        tokenizer.save_pretrained(str(folder_path_specific))
        trainer.save_vocab_and_stats(folder_path_specific)
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
def create_bytespantokenizer(
    model_type: Annotated[
        str,
        typer.Argument(
            help=f"Type of model whose predictions are used to train subwords. Supported models: {SUPPORTED_MODELS}"
        ),
    ],
    measure: Annotated[str, typer.Argument(help="Measure to use for training the subword tokenizer (e.g. Entropy)")],
    vocab_size : Annotated[
        int,
        typer.Argument(help="Vocabulary size for the tokenizer."),
    ],
    corpus: Annotated[
        str,
        typer.Argument(
            help=f"Corpus to use for training the subword tokenizer. Supported corpora: {SUPPORTED_CORPORA}"
        ),
    ] = FINEWEBEDU_REPO_ID,
    frequency_threshold: Annotated[int | float | None, typer.Option(help="Frequency threshold for keeping discovered subwords.")] = 20,
    threshold_percentile: Annotated[int | None, typer.Option(help="Percentile threshold for grouping subwords.")] = None,
    num_training_rows: Annotated[int, typer.Option(help="Number of training rows to use.")] = 100000,
) -> None:
    if measure == "SpaceProbability":
        measure = "Space Probability"

    tokenizer_name = (
        f"{model_type}_{measure}_bytespan"
    )
    folder_path = Path(TOK_REPO_ID) / tokenizer_name
    api = HfApi()

    logger.info(f"💡 Will save the tokenizers locally to to: {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Sort vocab_sizes if not already sorted
    vocab_sizes.sort()
    logger.info(f"Using vocab sizes: {vocab_sizes}")

    logger.info("⚙️ Loading bytelevel tokenizer and byte LLM data")
    byte_tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder=BYTELEVEL_TOK_FOLDER)
    dataset = load_dataset(f"{HF_USERNAME}/{corpus}", name=BYTE_LLM_PREDICTION_DATA, split=model_type)

    # Limit the dataset to the specified number of rows
    if num_training_rows > 0:
        if num_training_rows > len(dataset):
            logger.warning(f"Requested {num_training_rows} rows, but only {len(dataset)} available. Using all rows.")
            num_training_rows = len(dataset)
        else:
            dataset = dataset.select(range(num_training_rows))
    logger.info(f"Using {len(dataset)} rows for training")

    logger.info("⚙️ Creating the InfoTokenizer Trainer")
    trainer = ByteCurveTokenizerTrainer(
        dataset=dataset,
        byte_tokenizer=byte_tokenizer,
        measure=measure,
        frequency_threshold=frequency_threshold,
        threshold_percentile=threshold_percentile,
        logger=logger,
    )

    logger.info("⚙️ Training the ByteSpan Tokenizer")
    folder_path_specific = folder_path / str(vocab_size)
    folder_path_specific.mkdir(parents=True, exist_ok=True)
    trainer.train(final_vocab_size=vocab_size)
    tokenizer = trainer.create_tokenizer()
    tokenizer.save_pretrained(str(folder_path_specific))
    trainer.save_vocab_and_stats(folder_path_specific)
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
