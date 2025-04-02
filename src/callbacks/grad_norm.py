import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.grads import grad_norm
from torch import Tensor
from torch.optim.optimizer import Optimizer

from src.trainer import TensorBoardLogger


class GradNorm(Callback):
    PREFIX = "optim"

    def __init__(
        self,
        norm_type: float | int | str,
        group_separator: str = "/",
        histogram_freq: int | None = None,
        log_weight_distribution: bool = False,
        check_clipping: bool = False,
        only_total: bool = False,
    ) -> None:
        """Compute each parameter's gradient's norm and their overall norm before clipping is applied.

        The overall norm is computed over all gradients together, as if they
        were concatenated into a single vector.

        Args:
            norm_type: The type of the used p-norm, cast to float if necessary.
                Can be ``'inf'`` for infinity norm.
            group_separator: The separator string used by the logger to group
                the gradients norms in their own subfolder instead of the logs one.

        """
        self.group_separator = group_separator
        self.norm_type = float(norm_type)
        if self.norm_type <= 0:
            raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {self.norm_type}")

        self.histogram_freq = histogram_freq
        self.log_weight_distribution = log_weight_distribution
        self.check_clipping = check_clipping
        self.only_total = only_total

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # check if tensorboard is available
        self.tb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                self.tb_logger = logger

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        norms: dict[str, Tensor] = grad_norm(
            pl_module.model, norm_type=self.norm_type, group_separator=self.group_separator
        )  # type: ignore
        if self.only_total:
            norms = {f"grad_{self.norm_type}_norm_total": norms[f"grad_{self.norm_type}_norm_total"]}
        for pl_logger in trainer.loggers:
            pl_logger.log_metrics({f"{self.PREFIX}/{k}": v.item() for k, v in norms.items()}, step=trainer.global_step)

        if (
            self.tb_logger is not None
            and self.histogram_freq is not None
            and (trainer.global_step % self.histogram_freq == 0 or trainer.is_last_batch)
        ):
            # histogram of the norms
            norm_hist = torch.stack([v for k, v in norms.items() if not k.endswith("_norm_total")])
            self.tb_logger.experiment.add_histogram(
                tag=f"norms/total_{self.norm_type}_norm", values=norm_hist, global_step=trainer.global_step
            )

            # histogram of the grads
            for k, v in pl_module.named_parameters():
                self.tb_logger.experiment.add_histogram(tag=f"grad/{k}", values=v.grad, global_step=trainer.global_step)

                # histogram of the weights
                if self.log_weight_distribution:
                    self.tb_logger.experiment.add_histogram(
                        tag=f"weight/{k}", values=v, global_step=trainer.global_step
                    )

    def on_before_zero_grad(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        if self.check_clipping:
            norms = grad_norm(pl_module.model, norm_type=self.norm_type, group_separator=self.group_separator)
            for pl_logger in trainer.loggers:
                pl_logger.log_metrics(
                    {f"{self.PREFIX}/{k}_after_clipping": v for k, v in norms.items()}, step=trainer.global_step
                )
