import json
import logging
import shutil
import time
from pathlib import Path

import hydra
import lm_eval
import torch
from lightning import Trainer, seed_everything
from lm_eval.utils import handle_non_serializable, make_table
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
    ckpt_path = (
        cfg.resume_from_checkpoint if cfg.resume_from_checkpoint and Path(cfg.resume_from_checkpoint).exists() else None
    )
    trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info(f"Training total time: {(time.perf_counter() - start_time) / 60:.1f} minutes")

    for log in trainer.loggers:
        if isinstance(log, TensorBoardLogger):
            log.save_to_parquet("tb_logs.parquet")

    if cfg.evaluation.blimp and torch.cuda.is_available() and torch.cuda.current_device() == 0:
        start_time = time.perf_counter()
        logger.info("Evaluating BLiMP dataset...")
        # Temporarily save the model and tokenizer to a local directory
        logger.info("Temporarily saving model to .cache/eval_model")
        model.save_pretrained(".cache/eval_model")
        tok.save_pretrained(".cache/eval_model")
        task_manager = lm_eval.tasks.TaskManager()
        out_path = "blimp_results.json"
        logger.info("Running BLiMP tasks...")
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args="pretrained=.cache/eval_model",
            tasks=["blimp"],
            device="cuda:0",
            batch_size="auto",
            task_manager=task_manager,
            num_fewshot=0,
        )
        shutil.rmtree(".cache/eval_model")
        del results["samples"]
        logger.info(f"BLiMP results:\n{make_table(results)}")
        logger.info(f"Saving BLiMP results to {out_path}")
        with open(out_path, "w") as f:
            json.dump(results, indent=2, default=handle_non_serializable, ensure_ascii=False, fp=f)
        logger.info(f"BLiMP results saved to {out_path}")
        logger.info(f"Evaluation total time: {(time.perf_counter() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
