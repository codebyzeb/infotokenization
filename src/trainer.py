from dataclasses import dataclass, field
from pathlib import Path

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
from lightning_utilities import StrEnum
from tbparse import SummaryReader
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim.adamw import AdamW
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, get_scheduler

from src.utilities import DictConfig, get_logger

logger = get_logger("module")

TYPE_TO_OPTIMIZER_CLASS = {"adamw": AdamW}


def load_hf_from_pl(checkpoint_path: str | Path) -> PreTrainedModel:
    logger.info(f"Reading checkpoint from {checkpoint_path=}")
    checkpoint = torch.load(
        str(checkpoint_path),
        weights_only=False,
        map_location="cpu",
        # mmap=True,
    )
    state_dict = {
        k.removeprefix("model.").removeprefix("_orig_mod."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    config = checkpoint["hyper_parameters"].get("config")
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    logger.info(f"Model {config=}")
    return model


class RunningStage(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class OptimCofig(DictConfig):
    # Optimizer config
    optim_name: str
    lr: float
    weight_decay: float = 0.0
    optim_kwargs: dict = field(default_factory=dict)
    keller_kwargs: dict = field(default_factory=dict)

    # Scheduler config
    scheduler_name: str | None = None
    num_warmup_steps: int | None = None
    scheduler_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.optim_name in TYPE_TO_OPTIMIZER_CLASS
        if self.scheduler_name is not None:
            assert self.scheduler_name in TYPE_TO_SCHEDULER_FUNCTION


class LanguageModel(LightningModule):
    def __init__(self, model: LlamaForCausalLM, config: PretrainedConfig, optim_config: OptimCofig) -> None:
        super().__init__()
        self.model = model
        self.config = config  # save it here so that we can find it in the checkpoints!
        self.optim_config = optim_config
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.model.forward(input_ids=input_ids).logits  # type: ignore

    def step(self, batch: Tensor, stage: RunningStage) -> Tensor | None:
        input_ids = batch[:, :-1]
        labels = batch[:, 1:].clone()
        logits = self.forward(input_ids)
        loss = cross_entropy(logits.permute(0, 2, 1), labels)

        self.log(
            f"{stage}/loss",
            loss.detach(),
            on_step=stage == RunningStage.TRAIN,
            on_epoch=stage == RunningStage.VALIDATION,
            prog_bar=True,
            logger=True,
            batch_size=labels.shape[0],
            sync_dist=True,
        )

        if stage == RunningStage.TRAIN:
            return loss

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self.step(batch, RunningStage.TRAIN)  # type: ignore

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self.step(batch, RunningStage.VALIDATION)  # type: ignore

    def configure_optimizers(self) -> dict:
        # Get params that require grad
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Create optim_groups taking care to check whether we want the keller optimiser
        decay_params, nodecay_params = [], []
        for _, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        logger.info(f"{len(decay_params)=}, with {sum(p.numel() for p in decay_params):,} params")
        logger.info(f"{len(nodecay_params)=}, with {sum(p.numel() for p in nodecay_params):,} params")

        optim_groups = [
            {"params": decay_params, "weight_decay": self.optim_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        out = {}

        # Create optimizer
        opt = TYPE_TO_OPTIMIZER_CLASS[self.optim_config.optim_name](
            optim_groups, lr=self.optim_config.lr, **self.optim_config.optim_kwargs
        )
        out["optimizer"] = opt

        # Maybe create scheduler
        if self.optim_config.scheduler_name is not None:
            lr_scheduler = get_scheduler(
                name=self.optim_config.scheduler_name,
                num_warmup_steps=self.optim_config.num_warmup_steps,
                optimizer=opt,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                scheduler_specific_kwargs=self.optim_config.scheduler_kwargs,
            )
            out["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return out


class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_NAME: str = "tensorboard"

    @property
    def logger_name(self) -> str:
        return self.LOGGER_NAME

    def save_to_parquet(self, path: str | Path) -> None:
        SummaryReader(str(self.log_dir)).scalars.to_parquet(path)
