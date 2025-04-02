from datetime import timedelta
from pathlib import Path
from typing import Literal

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as _ModelCheckpoint

from src.utilities import get_logger

logger = get_logger("model_checkpoint")


class ModelCheckpoint(_ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = ""

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | None | Literal["link"] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
        save_initial_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            enable_version_counter,
        )
        self.save_initial_checkpoint = save_initial_checkpoint

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # If you are loading from checkpoint, then don't save the initial checkpoint
        if trainer.ckpt_path is not None:
            logger.info("Not saving initial checkpoint since we are loading from a checkpoint.")
            return

        self._save_none_monitor_checkpoint(trainer, {})
