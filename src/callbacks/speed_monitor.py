import time

from lightning import Trainer
from lightning.pytorch import Callback, LightningModule
from lightning.pytorch.utilities import rank_zero_only

from src.trainer import RunningStage


class SpeedMonitor(Callback):
    PREFIX = "time"

    def epoch_start(self, stage: str | RunningStage) -> None:
        setattr(self, f"{stage}_epoch_start_time", time.time())

    def epoch_end(self, trainer: Trainer, stage: str | RunningStage) -> None:
        setattr(self, f"{stage}_epoch_end_time", time.time())
        prev_time = getattr(self, f"{stage}_epoch_start_time") if hasattr(self, f"{stage}_epoch_start_time") else 0
        runtime = getattr(self, f"{stage}_epoch_end_time") - prev_time
        for pl_logger in trainer.loggers:
            pl_logger.log_metrics({f"{self.PREFIX}/{stage}_epoch (min)": runtime / 60}, step=trainer.global_step)

    def batch_start(self, stage: str | RunningStage) -> None:
        setattr(self, f"{stage}_batch_start_time", time.perf_counter())

    def batch_end(self, trainer: Trainer, stage: str | RunningStage) -> None:
        setattr(self, f"{stage}_batch_end_time", time.perf_counter())
        runtime = getattr(self, f"{stage}_batch_end_time") - getattr(self, f"{stage}_batch_start_time")
        for pl_logger in trainer.loggers:
            pl_logger.log_metrics({f"{self.PREFIX}/{stage}_batch (ms)": runtime * 1000}, step=trainer.global_step)

    def on_fit_start(self, *args, **kwargs) -> None:
        self.fit_start = time.time()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.fit_end = time.time()
        runtime = self.fit_end - self.fit_start
        for pl_logger in trainer.loggers:
            pl_logger.log_metrics({f"{self.PREFIX}/fit_time (min)": runtime / 60}, step=trainer.global_step)

    """
    Epoch start
    """

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.TRAIN)

    def on_validation_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.VALIDATION)

    """
    Epoch end
    """

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.epoch_end(trainer, RunningStage.TRAIN)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.epoch_end(trainer, RunningStage.VALIDATION)

    """
    Batch start
    """

    @rank_zero_only
    def on_train_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.TRAIN)

    @rank_zero_only
    def on_validation_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.VALIDATION)

    """
    Batch end
    """

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs) -> None:
        self.batch_end(trainer, RunningStage.TRAIN)

    @rank_zero_only
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs) -> None:
        self.batch_end(trainer, RunningStage.VALIDATION)
