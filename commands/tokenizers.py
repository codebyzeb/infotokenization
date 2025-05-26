import json
import logging
import os
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
    BYTE_DATA_NGRAM_EXTRACTION,
)
from commands.extract import SUPPORTED_CORPORA, SUPPORTED_MODELS
from commands.data import AddPreTokenizationBoundaries
from src.utilities import get_logger

app = typer.Typer()

ADD_PREFIX_SPACE = True  # Note that we will add a prefix_space to the pre_tokenizer
PAD_TOKEN = "<|padding|>"
EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"
DEFAULT_TOKENIZER_SIZES = [8_064, 16_000, 32_000, 64_000, 128_000]

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
            [torch.tensor(x, dtype=torch.int64).to(self.device) for x in dataset["input_ids"]]
        )
        self.pre_token_boundaries = torch.cat(
            [torch.tensor(x, dtype=torch.bool).to(self.device) for x in dataset["pre_token_boundaries"]]
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
        self.pre_token_boundaries = self.pre_token_boundaries[~merge_positions]

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
        discovered = sorted(self.segment_counts.items(), key=lambda x: x[1], reverse=True)
        for token_ref, count in discovered:
            if count < self.frequency_threshold:
                break  # Can break here because we've sorted the segments by frequency
            is_start_of_word, token_ids = token_ref
            token = ''.join(self.byte_tokenizer.convert_ids_to_tokens(token_ids))
            if not is_start_of_word:
                token = "##" + token
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

        # Shift left by one if we can include the space token byte
        if not self.pre_token_boundaries[left_boundary] and self.ids[left_boundary-1] == self.space_token_id:
            left_boundary -= 1
        
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
        is_start = self.pre_token_boundaries[left_boundary]
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
        proportion_bytespan: int | float | None = None,
        threshold_percentile: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Class for training a byte-curve tokenizer using information measures derived from a byte-level LM.

        Args:
            dataset (Dataset): The dataset to use for training containing information measures for each token.
            byte_tokenizer (ByteLevelBPETokenizer): The byte-level tokenizer to use for the initial vocabulary.
            measure (str): The information measure to use ('Entropy', 'Surprisal', etc.).
            proportion_bytespan (int, optional): If set, determines the percentage of the final vocabulary that will come from bytespans, the remainder will be from BPE.
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
        self.proportion_bytespan = proportion_bytespan
        self.threshold_percentile = threshold_percentile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger if logger else get_logger("tokenizer")

        self.space_token_id = byte_tokenizer.encode(" ")[0]
        self.space_token = byte_tokenizer.convert_ids_to_tokens(self.space_token_id)

        self.initial_vocab = self.byte_tokenizer.get_vocab()
        keys = list(self.initial_vocab.keys())
        self.initial_vocab[UNK_TOKEN] = len(self.initial_vocab)
        for key in keys:
            if key != PAD_TOKEN and key != EOS_TOKEN and key != UNK_TOKEN:
                if key != self.space_token:
                    self.initial_vocab[self.space_token + key] = len(self.initial_vocab)
                self.initial_vocab["##" + key] = len(self.initial_vocab)
        self.vocab = self.initial_vocab.copy()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.logger.info(f"Initial vocabulary size: {len(self.vocab)}")
        
        if threshold_percentile is not None:
            if threshold_percentile < 0 or threshold_percentile > 100:
                raise ValueError("Threshold percentile must be between 0 and 100.")
            self.logger.info(f"Using threshold percentile of {threshold_percentile} - bytes under this value will be grouped.")
            self.logger.info(f"Getting threshold value from the dataset...")
            self.threshold = self.get_threshold()
            self.logger.info(f"Threshold value: {self.threshold}")
        else:
            self.threshold = None

        self.subword_frequencies = None
        self.subword_spans_to_tokens = {}
    
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
        return np.percentile(signal_values, self.threshold_percentile)

    def find_subword_boundaries(self, examples):

        subword_start_list = []
        subword_length_list = []

        for i in range(len(examples[self.measure])):

            signal = np.array(examples[self.measure][i])
            ids = np.array(examples["input_ids"][i])
            pre_token_boundaries = np.array(examples["pre_token_boundaries"][i])

            # Identify subwords according to monotonically decreasing signal
            subword_mask = np.zeros(len(signal), dtype=bool)
            diff_in_signal = np.diff(signal)
            diff_in_signal = np.append(1, diff_in_signal)
            diff_in_signal[ids == 1] = 1
            decreasing = np.where((diff_in_signal < 0), True, False)
            if self.threshold:
                below_threshold = np.where((signal < self.threshold), True, False)
                subword_mask = np.where((decreasing | below_threshold), subword_mask, True)
            else:
                subword_mask = np.where(decreasing, subword_mask, True)

            # Subword boundaries are whenever subword_mask flips values
            subword_boundary = np.concatenate((subword_mask, [False]))
            subword_boundary = np.diff(subword_boundary).astype(bool)

            # Even flip locations are word boundary starts, odd flip locations are word boundary ends
            subword_starts = subword_boundary.copy()
            subword_ends = subword_boundary.copy()
            true_indices = np.where(subword_boundary)[0]
            subword_starts[true_indices[1::2]] = False
            subword_ends[true_indices[::2]] = False 

            adjusted_starts = np.zeros(len(ids), dtype=bool)
            subword_lengths = np.zeros(len(ids), dtype=np.int32)

            # Split subwords that cross a pre-tokenization boundary
            for start, end in zip(np.where(subword_starts)[0], np.where(subword_ends)[0]):
                # Include left byte if it's a word start id
                if not pre_token_boundaries[start] and ids[start-1] == self.space_token_id:
                    start -= 1
                # Don't include final byte if it's the start of a new pre-token
                if pre_token_boundaries[end]:
                    end -= 1
                while ids[start] in self.byte_tokenizer.all_special_ids:
                    start += 1
                if start >= end:
                    continue
                for mid in range(start+1, end+1):
                    if pre_token_boundaries[mid]:
                        adjusted_starts[start] = True
                        subword_lengths[start] = mid - start
                        start = mid
                        while ids[start] in self.byte_tokenizer.all_special_ids:
                            start += 1
                if start != end:
                    adjusted_starts[start] = True
                    subword_lengths[start] = end - start + 1

            subword_start_list.append(adjusted_starts)
            subword_length_list.append(subword_lengths)

        examples["subword_starts"] = subword_start_list
        examples["subword_lengths"] = subword_length_list
        return examples
    
    def create_vocab_from_spans(self):
        """ Get a vocabulary by converting subword spans to tokens and counting their frequency."""

        subword_frequencies = {}
        subword_spans_to_tokens = {}

        for example in tqdm(self.dataset, desc="Creating vocabulary from identified subwords"):

            ids = np.array(example["input_ids"])
            pre_token_boundaries = np.array(example["pre_token_boundaries"])
            word_starts = np.array(example["subword_starts"])
            word_lengths = np.array(example["subword_lengths"])

            word_start_indices = np.where(word_starts)[0]
            for start in word_start_indices:
                length = word_lengths[start]
                id_span = ids[start:start + length].tolist()
                token = ''.join(self.byte_tokenizer.convert_ids_to_tokens(id_span))
                is_pre_token_start = pre_token_boundaries[start].item()
                if not is_pre_token_start:
                    token = '##' + token
                if token and not token in self.vocab:
                    if token in subword_frequencies:
                        subword_frequencies[token] += 1
                    else:
                        subword_frequencies[token] = 1
                    subword_spans_to_tokens[(is_pre_token_start, tuple(id_span))] = token

        return subword_frequencies, subword_spans_to_tokens

    def train(self, final_vocab_size):
        """ Train the tokenizer """
        # Can call train multiple times, so always reset vocabulary first
        self.vocab = self.initial_vocab.copy()
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        if self.proportion_bytespan == 0.0:
            self.logger.info('Proportion bytespan set to 0. Skipping byte spans, will just use BPE.')
        else:
            if self.subword_frequencies is None:
                self.logger.info('Finding subwords using byte curves...')
                self.dataset = self.dataset.map(
                    self.find_subword_boundaries,
                    batched=True,
                    num_proc=min(12, os.cpu_count()),
                    desc="Finding subword boundaries"
                )
                self.subword_frequencies, self.subword_spans_to_tokens = self.create_vocab_from_spans()
                logger.info(f'Found {len(self.subword_frequencies)} unique subwords in dataset.')
            else:
                self.logger.info('Using existing subword frequencies and spans.')
            sorted_subword_frequencies = sorted(self.subword_frequencies.items(), key=lambda x: x[1], reverse=True)
            if self.proportion_bytespan is None:
                self.logger.info("Proportion bytespan not provided, keeping all discovered vocabulary items.")
            else:
                total_items = self.proportion_bytespan * final_vocab_size - len(self.vocab)
                logger.info(f'Using percentage threshold, keeping most frequent subwords to reach {100*self.proportion_bytespan}% of target vocab size.')
                if self.proportion_bytespan < 0 or self.proportion_bytespan > 1:
                    raise ValueError("Frequency threshold must be between 0 and 1.")
                sorted_subword_frequencies = dict(sorted_subword_frequencies[:int(total_items)])
                logger.info(f'Filtered subwords to {len(sorted_subword_frequencies)} most frequent items.')

            self.logger.info(f"Adding {len(sorted_subword_frequencies)} subwords to the initial vocabulary.")
            for key in sorted_subword_frequencies.keys():
                if not key in self.vocab:
                    self.vocab[key] = len(self.vocab)
                else:
                    raise RuntimeError(f"Error: subword '{key}' already exists in the vocabulary.")
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            assert len(self.vocab) == len(self.inverse_vocab), "Vocabulary and inverse vocabulary must have the same size."
            self.logger.info(f"Updated vocab size: {len(self.vocab)}")

        if len(self.vocab) >= final_vocab_size:
            self.logger.info(f"Final vocab size {len(self.vocab)} exceeds target {final_vocab_size}.")
            self.logger.info("Filtering discovered tokens by frequency.")
            self.vocab = {k: v for k, v in self.vocab.items() if v < final_vocab_size}
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        elif self.proportion_bytespan == 1.0:
            self.logger.error("Proportion bytespan set to 1.0, but not enough subwords found to reach final vocab size.")
            raise RuntimeError("Not enough subwords found to reach final vocab size.")
        else:
            self.logger.info(f"Running BPE to learn remaining subwords.")
            self.bpe(final_vocab_size)
        self.logger.info(f"Final vocab size reached: {len(self.vocab)}")
        return self.vocab

    def bpe(self, final_vocab_size):
        """Run BPE to learn remaining subwords."""

        subword_spans_to_ids = {span: self.vocab[token] for span, token in self.subword_spans_to_tokens.items() if token in self.vocab}
        if len(subword_spans_to_ids) == 0:
            self.logger.info("No valid bytespan subwords found, running BPE from byte-level tokens.")
            self.logger.info("Creating vector ID to apply merges and calculate frequencies...")
            ids = torch.cat([torch.tensor(x, dtype=torch.int64).to(self.device) for x in self.dataset["input_ids"]])
            pre_token_boundaries = torch.cat([torch.tensor(x, dtype=torch.bool).to(self.device) for x in self.dataset["pre_token_boundaries"]])
        else:
            old_length = sum(example["num_tokens"] for example in self.dataset)

            def merge_ids(examples):
                # Pre-allocate lists with known size
                batch_size = len(examples["input_ids"])
                ids_list = [None] * batch_size
                pre_token_boundaries_list = [None] * batch_size

                for i in range(batch_size):
                    # Convert to numpy arrays once at the start
                    ids = np.array(examples["input_ids"][i])
                    pre_token_boundaries = np.array(examples["pre_token_boundaries"][i])
                    subword_starts = np.array(examples["subword_starts"][i])
                    subword_lengths = np.array(examples["subword_lengths"][i])
                    
                    # Create delete mask
                    delete_mask = np.zeros(len(ids), dtype=bool)
                    
                    # Get valid starts directly from boolean array
                    starts = np.where(subword_starts)[0]
                    for start in starts:
                        length = subword_lengths[start]
                        # Use numpy slicing instead of converting to list
                        subword_span = tuple(ids[start:start+length])
                        is_pre_token_start = pre_token_boundaries[start]
                        subword_span = (is_pre_token_start, subword_span)
                        
                        if subword_span not in subword_spans_to_ids:
                            continue
                            
                        new_id = subword_spans_to_ids[subword_span]
                        ids[start] = new_id
                        delete_mask[start+1:start+length] = True

                    # Apply mask in one operation
                    ids_list[i] = ids[~delete_mask]
                    pre_token_boundaries_list[i] = pre_token_boundaries[~delete_mask]

                examples["input_ids"] = ids_list
                examples["pre_token_boundaries"] = pre_token_boundaries_list
                return examples
            
            merged_dataset = self.dataset.map(
                merge_ids,
                batched=True,
                num_proc= min(12, os.cpu_count()),
                desc="Merging subwords in dataset",
                remove_columns=[col for col in self.dataset.column_names if col not in ["input_ids", "pre_token_boundaries"]],
            )

            ids = torch.cat([torch.tensor(x, dtype=torch.int64).to(self.device) for x in merged_dataset["input_ids"]])
            pre_token_boundaries = torch.cat([torch.tensor(x, dtype=torch.bool).to(self.device) for x in merged_dataset["pre_token_boundaries"]])
            new_length = len(ids)

            self.logger.info(f'Shortened data from: {old_length} to new length: {new_length}')
            self.logger.info("Creating vector ID to apply merges and calculate frequencies...")

        # Move to CPU for apply_ operation
        ids_cpu = ids.cpu()
        ids_cpu[~pre_token_boundaries.cpu()] = ids_cpu[~pre_token_boundaries.cpu()].apply_(
            lambda t: (self.vocab["##" + self.inverse_vocab[t]] if not self.inverse_vocab[t].startswith("##") else t)
        )
        ids = ids_cpu.to(self.device)
        logger.info("Vectors prepared for BPE training.")
        
        num_ids = len(self.vocab)
        pbar = tqdm(total=final_vocab_size, desc="Generating IDs")
        while num_ids <  final_vocab_size:
            pbar.update(num_ids - pbar.n)
            # Get pair counts
            # TODO: We don't need to recalculate after every merge, could just update the locations where the merge occurs.
            pairs =  ids * (num_ids + 1) + ids.roll(1)
            pairs[pre_token_boundaries] = -1
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
            ids[merge_positions.roll(-1)] = new_id
            ids = ids[~merge_positions]
            pre_token_boundaries = pre_token_boundaries[~merge_positions]

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

    logger.info("⚙️ Loading bytelevel tokenizer and bytelevel data")
    byte_tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder=BYTELEVEL_TOK_FOLDER)
    dataset = load_dataset(f"{HF_USERNAME}/{corpus}", name=BYTE_DATA_NGRAM_EXTRACTION, split="train")

    # Limit the dataset to the specified number of rows
    if num_training_rows > 0:
        if num_training_rows > len(dataset):
            logger.warning(f"Requested {num_training_rows} rows, but only {len(dataset)} available. Using all rows.")
            num_training_rows = len(dataset)
        else:
            dataset = dataset.select(range(num_training_rows))
    logger.info(f"Using {len(dataset)} rows for training")

    if 'pre_token_boundaries' not in dataset.column_names:
        logger.info("Adding pre-tokenization boundaries to the dataset")
        dataset = dataset.map(
            AddPreTokenizationBoundaries(byte_tokenizer),
            batched=True,
            desc="Adding pre-tokenization boundaries",
            num_proc=min(os.cpu_count(), 8),
        )

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

    if 'pre_token_boundaries' not in dataset.column_names:
        logger.info("Adding pre-tokenization boundaries to the dataset")
        dataset = dataset.map(
            AddPreTokenizationBoundaries(byte_tokenizer),
            batched=True,
            desc="Adding pre-tokenization boundaries",
            num_proc=min(os.cpu_count(), 8),
        )

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
    corpus: Annotated[
        str,
        typer.Argument(
            help=f"Corpus to use for training the subword tokenizer. Supported corpora: {SUPPORTED_CORPORA}"
        ),
    ] = FINEWEBEDU_REPO_ID,
    proportion_bytespan: Annotated[float | None, typer.Option(help="If set, the bytespan subwords will only contribute this proportion of the final vocabulary (BPE will be used for the rest).")] = None,
    threshold_percentile: Annotated[int | None, typer.Option(help="Percentile threshold for grouping subwords.")] = None,
    num_training_rows: Annotated[int, typer.Option(help="Number of training rows to use.")] = 100000,
    vocab_sizes: Annotated[
        list[int], typer.Option(help="Vocabulary sizes for the tokenizer.")
    ] = DEFAULT_TOKENIZER_SIZES,
) -> None:
    if measure == "SpaceProbability":
        measure = "Space Probability"

    tokenizer_name = (
        f"{model_type}_{measure}_bytespan"
        + (f"P{proportion_bytespan}".replace('.','-') if proportion_bytespan is not None else "")
        + (f"T{threshold_percentile}".replace('.','-') if threshold_percentile is not None else "")
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

    if 'pre_token_boundaries' not in dataset.column_names:
        logger.info("Adding pre-tokenization boundaries to the dataset")
        dataset = dataset.map(
            AddPreTokenizationBoundaries(byte_tokenizer),
            batched=True,
            desc="Adding pre-tokenization boundaries",
            num_proc=min(os.cpu_count(), 8),
        )

    logger.info("⚙️ Creating the InfoTokenizer Trainer")
    trainer = ByteCurveTokenizerTrainer(
        dataset=dataset,
        byte_tokenizer=byte_tokenizer,
        measure=measure,
        proportion_bytespan=proportion_bytespan,
        threshold_percentile=threshold_percentile,
        logger=logger,
    )

    for vocab_size in vocab_sizes:
        logger.info(f"⚙️ Training the ByteSpan Tokenizer with vocab size: {vocab_size}")
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
