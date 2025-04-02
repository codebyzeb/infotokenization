# import json
import copy
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import colorlog
import polars as pl
import srsly
from hydra.utils import instantiate
from omegaconf import OmegaConf


@dataclass
class DictConfig:
    """Dataclass which is subscriptable like a dict"""

    def to_dict(self) -> dict[str, Any]:
        out = copy.deepcopy(self.__dict__)
        return out

    def __getitem__(self, k: str) -> Any:
        return self.__dict__[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)


def get_logger(name: str, level: Literal["error", "warning", "info", "debug"] = "info") -> logging.Logger:
    # Convert the level string to the corresponding logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure the logger and configure colorlog
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"  # noqa: E501
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def flatten(x: list[list]) -> list:
    return [i for j in x for i in j]


def remove_file(path: str | Path) -> None:
    path = Path(path)
    path.unlink(missing_ok=True)


def jsonl2parquet(filepath: str | Path, out_dir: str | Path) -> None:
    filepath = Path(filepath)
    assert filepath.name.endswith(".jsonl"), "Not a jsonl file"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl = srsly.read_jsonl(filepath)
    df = pl.DataFrame({k: flatten(v) for k, v in ld_to_dl(line).items()} for line in fl)  # type: ignore
    df = df.explode(df.columns)

    df.write_parquet(out_dir / f"{filepath.name.removesuffix('.jsonl')}.parquet")


def ld_to_dl(ld: list[dict]) -> dict[str, list]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def conf_to_dict(x: DictConfig | None) -> dict:
    if x is not None:
        return OmegaConf.to_container(x)  # type: ignore
    return {}


def instantiate_from_conf(list_cfg: list[DictConfig]) -> list:
    return [list(instantiate(cfg).values()) if cfg is not None else None for cfg in list_cfg]


# def load_tokenizer_with_vocab_size(path: str | Path, vocab_size: int) -> PreTrainedTokenizerFast:
#     path = Path(path)

#     # Edit conf to adapt to the new vocab_size
#     conf: dict = srsly.read_json(path / "tokenizer.json")  # type: ignore

#     # get the number of initial characters used by BPE, these are the things that get merged
#     len_alphabet = len(conf["model"]["vocab"]) - len(conf["model"]["merges"])

#     # the vocab size includes the initial alphabet
#     vocab_sorted_by_token_id = sorted(conf["model"]["vocab"].items(), key=lambda item: item[1])
#     conf["model"]["vocab"] = dict(vocab_sorted_by_token_id[:vocab_size])

#     # but the merges need to be less, such that vocab_size = num_merges + initial_alphabet
#     conf["model"]["merges"] = conf["model"]["merges"][: vocab_size - len_alphabet]

#     # Instantiate tokenizer using tokenizers library
#     backend_tok = Tokenizer.from_str(json.dumps(conf))
#     eos_token: str = srsly.read_yaml(path / "metadata.yaml")["eos_token"]  # type: ignore

#     # Instantiate PreTrainedTokenizerFast from object
#     # NOTE: we do not instantiate from file directly due to compatibility
#     # https://github.com/huggingface/tokenizers/issues/1562#issuecomment-2315349846
#     tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, clean_up_tokenization_spaces=True)

#     # Add common configs for decoder-only models
#     tok.padding_side = "left"  # type: ignore
#     tok.eos_token = eos_token  # type: ignore

#     return tok


# # src/utils/logging.py
# import logging
# import colorlog
# from typing import Literal, Optional
# from pytorch_lightning.utilities import rank_zero_only

# class RankZeroFilter(logging.Filter):
#     """Filter that only allows logging on rank 0 in distributed settings."""
#     def filter(self, record) -> bool:
#         try:
#             return rank_zero_only.rank == 0
#         except AttributeError:
#             return True  # If rank info isn't available, allow logging

# def get_logger(
#     name: str | None = None,
#     level: Literal["error", "warning", "info", "debug"] = "info",
#     log_file: str | None = None
# ) -> logging.Logger:
#     """
#     Configure logging for either a specific logger or the root logger.

#     Args:
#         name: Logger name. If None, configures the root logger.
#         level: Logging level (error, warning, info, debug).
#         log_file: Path to log file, if file logging is desired.

#     Returns:
#         Configured logger instance.
#     """
#     # Determine which logger to configure
#     logger = logging.getLogger(name) if name else logging.getLogger()

#     # Prevent duplicate handlers
#     if logger.handlers:
#         return logger

#     # Set logging level
#     log_level = getattr(logging, level.upper(), logging.INFO)
#     logger.setLevel(log_level)
#     logger.propagate = False

#     # Common formatters
#     color_formatter = colorlog.ColoredFormatter(
#         "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
#     )
#     plain_formatter = logging.Formatter(
#         "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
#     )

#     # Console handler (colored)
#     console_handler = colorlog.StreamHandler()
#     console_handler.setFormatter(color_formatter)
#     console_handler.addFilter(RankZeroFilter())
#     logger.addHandler(console_handler)

#     # File handler (if specified)
#     if log_file:
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setFormatter(plain_formatter)
#         file_handler.addFilter(RankZeroFilter())
#         logger.addHandler(file_handler)

#     return logger
