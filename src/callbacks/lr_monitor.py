from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from torch.optim.optimizer import Optimizer


class SimpleLearningRateMonitor(Callback):
    PREFIX = "optim"

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        stats = {
            f"lr-{optimizer.__class__.__name__.lower()}-pg{idx}": pg["lr"]
            for idx, pg in enumerate(optimizer.param_groups)
        }

        for pl_logger in trainer.loggers:
            pl_logger.log_metrics({f"{self.PREFIX}/{k}": v for k, v in stats.items()}, step=trainer.global_step)
