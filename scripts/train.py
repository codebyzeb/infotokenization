import logging
import time
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from commands.configs import HF_USERNAME, TOK_REPO_ID
from src.data import DataloaderConfig, DataModule
from src.model import get_model
from src.trainer import LanguageModel, OptimCofig, TensorBoardLogger
from src.utilities import conf_to_dict, instantiate_from_conf

SEP_LINE = f"{'=' * 80}"

# Configure the logger and configure colorlog
logger = logging.getLogger("hydra")


@hydra.main(version_base=None, config_path="../conf", config_name="train_conf")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    # Load tokenizer
    tok_repo_id = f"{HF_USERNAME}/{TOK_REPO_ID}"
    logger.info(f"Loading tokenizer from {tok_repo_id}/{cfg.tok_name}")
    tok = AutoTokenizer.from_pretrained(tok_repo_id, subfolder=cfg.tok_name)

    # Load model
    model, config = get_model(cfg.model, tok)  # type: ignore
    logger.info(f"Model config:\n{model.config.to_json_string()}")
    logger.info(f"Attention implementation: {model.config._attn_implementation}")
    logger.info(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
    logger.info(f"Num parameters: {model.num_parameters() / 1e6:.1f}M")

    # Load datamodule
    dataloader_config = DataloaderConfig(**conf_to_dict(cfg.data))

    # FIXME: the datapath logic needs to be updated
    datamodule = DataModule(
        train_data_path=cfg.train_data_path,
        val_data_path=cfg.val_data_path,
        max_position_embeddings=model.config.max_position_embeddings,
        eod_token_id=tok.eos_token_id,  # type: ignore
        dataloader_config=dataloader_config,
    )

    # Maybe compile
    if cfg.torch_compile:
        model = torch.compile(model)

    # Load module
    optim_config = OptimCofig(**conf_to_dict(cfg.optim))  # type: ignore
    module = LanguageModel(model, config, optim_config)  # type: ignore

    # Load trainer
    loggers, callbacks = instantiate_from_conf([cfg.get(i) for i in ("loggers", "callbacks")])
    trainer = Trainer(**conf_to_dict(cfg.trainer), logger=loggers, callbacks=callbacks)

    # Train
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    ckpt_path = cfg.resume_from_checkpoint if Path(cfg.resume_from_checkpoint).exists() else None
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info(f"Training total time: {(time.perf_counter() - start_time) / 60:.1f} minutes")

    for log in trainer.loggers:
        if isinstance(log, TensorBoardLogger):
            log.save_to_parquet("tb_logs.parquet")


if __name__ == "__main__":
    main()
