from copy import copy
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset as TorchDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from src.utilities import DictConfig, get_logger

logger = get_logger("data")


class PackedTokenDataset(TorchDataset):
    def __init__(self, data_path: str | Path, seq_len: int, eod_token_id: int = 0) -> None:
        """
        A map-style dataset that packs tokenized documents into fixed-length sequences.

        Args:
            data_path (str | Path): Path to dataset dict.
            seq_len (int): Length of each sequence.
            eod_token_id (int): End-of-document token (default: 0).
        """
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.eod_token_id = eod_token_id

        # Read datasets
        self.dataset: Dataset = load_from_disk(str(self.data_path))  # type: ignore

        # Precompute document offsets using NumPy for fast cumulative sums
        offsets_path = self.data_path / "offsets.npy"
        if offsets_path.exists():
            self.offsets = np.load(offsets_path)
        else:
            self.offsets = self._compute_offsets()
            np.save(offsets_path, self.offsets)

        self.total_tokens = self.offsets[-1]

        # Calculate total number of sequences
        # self.num_sequences = (self.total_tokens + seq_len - 1) // seq_len
        self.num_sequences = self.total_tokens // seq_len  # Drops remainder of tokens that do not fill seq_len

    def _compute_offsets(self) -> np.ndarray:
        """Precompute offsets using NumPy for fast cumulative sums."""
        doc_lengths = self.dataset.map(
            lambda x: {"length": [len(s) + 1 for s in x["input_ids"]]},  # +1 for EOD
            desc="Computing offsets",
            num_proc=min(8, cpu_count()),  # type: ignore
            load_from_cache_file=False,
            remove_columns=self.dataset.column_names,
            keep_in_memory=True,
            batched=True,
        ).with_format("numpy")["length"]
        offsets = np.cumsum(doc_lengths)
        return np.insert(offsets, 0, 0)

    def _find_document(self, pos: int) -> int:
        """Binary search to find the document containing the given token position."""
        return int(np.searchsorted(self.offsets, pos, side="right") - 1)

    def get_sequence(self, start_pos: int, end_pos: int) -> torch.Tensor:
        """Retrieve a sequence with minimal overhead and fast memory operations."""
        current_tokens = torch.empty(end_pos - start_pos, dtype=torch.long)
        pos = start_pos
        i = 0

        while pos < end_pos:
            # Find the relevant document for the current position
            doc_idx = self._find_document(pos)

            # Directly access the input_ids list to avoid repeated tensor conversion
            input_ids = self.dataset[doc_idx]["input_ids"]

            doc_start = pos - self.offsets[doc_idx]
            tokens_to_copy = min(len(input_ids) - doc_start, end_pos - pos)

            # Use PyTorch's in-place assignment without slicing
            current_tokens[i : i + tokens_to_copy] = torch.as_tensor(
                input_ids[doc_start : doc_start + tokens_to_copy], dtype=torch.long
            )

            # Update counters
            i += tokens_to_copy
            pos += tokens_to_copy

            # Add EOD token if the document ends, and more tokens are needed
            if doc_start + tokens_to_copy == len(input_ids) and pos < end_pos:
                current_tokens[i] = self.eod_token_id
                i += 1
                pos += 1

        return current_tokens

    def __len__(self) -> int:
        """Return the total number of sequences."""
        return self.num_sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrieve the sequence at the given index."""
        start_pos = index * self.seq_len
        end_pos = start_pos + self.seq_len
        return self.get_sequence(start_pos, end_pos)


@dataclass
class DataloaderConfig(DictConfig):
    batch_size: int | None = None
    eval_batch_size: int | None = None
    num_workers: int | None = cpu_count()
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False
    multiprocessing_context: str | None = None
    shuffle: bool = False
    prefetch_factor: int | None = None

    def get_train_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs.pop("eval_batch_size")
        return kwargs

    def get_val_kwargs(self) -> dict:
        kwargs = copy(self.to_dict())
        kwargs["batch_size"] = kwargs.pop("eval_batch_size")
        kwargs["shuffle"] = False
        return kwargs


class DataModule(LightningDataModule):
    train_ds: PackedTokenDataset
    val_ds: PackedTokenDataset

    def __init__(
        self,
        train_data_path: str | Path | None,
        val_data_path: str | Path | None,
        max_position_embeddings: int,
        eod_token_id: int,
        dataloader_config: DataloaderConfig,
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path) if train_data_path else train_data_path
        self.val_data_path = Path(val_data_path) if val_data_path else val_data_path
        self.max_position_embeddings = max_position_embeddings
        self.eod_token_id = eod_token_id
        self.dataloader_config = dataloader_config
        self.save_hyperparameters()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        if self.train_data_path:
            self.train_ds = PackedTokenDataset(
                data_path=str(self.train_data_path),
                seq_len=self.max_position_embeddings + 1,  # EOD token
                eod_token_id=self.eod_token_id,
            )
            logger.info(f"Train dataset loaded: {len(self.train_ds)=}")
            logger.info(f"{self.train_ds=}")

        if self.val_data_path:
            self.val_ds = PackedTokenDataset(
                data_path=str(self.val_data_path),
                seq_len=self.max_position_embeddings + 1,
                eod_token_id=self.eod_token_id,
            )
            logger.info(f"Validation dataset loaded: {len(self.val_ds)=}")
            logger.info(f"{self.val_ds=}")

    def train_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(self.train_ds, **self.dataloader_config.get_train_kwargs())

    def val_dataloader(self) -> StatefulDataLoader:
        return StatefulDataLoader(self.val_ds, **self.dataloader_config.get_val_kwargs())
