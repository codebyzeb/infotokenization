from pathlib import Path
from typing import Annotated

import srsly
import typer
import yaml
from huggingface_hub import HfApi, logging
from transformers import AutoTokenizer

from src.trainer import load_hf_from_pl
from src.utilities import get_logger

logger = get_logger("uploader")
logging.set_verbosity_info()  # or _debug for more info

app = typer.Typer()


@app.command()
def model(run_dir: Annotated[str, typer.Argument(help="Path to the directory of the training run.")]) -> None:
    run_path = Path(run_dir)

    logger.info("read hparams")
    hparams: dict = srsly.read_yaml(run_path / "hparams.yaml")  # type: ignore

    logger.info("write readme")
    lines = f"## Experiment Configuration\n```yaml\n{yaml.dump(hparams)}```".replace("/home/pl487", ".")
    with (run_path / "README.md").open("w") as fl:
        fl.writelines(lines)

    logger.info("create repo and upload common files")
    repo_id = f"pietrolesci/{hparams['model']}_{hparams['dataset']}_{hparams['tok_name']}"

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=run_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".hydra/*", ".checkpoints/*", "*.log", "*.err", "*.out"],  # Ignore all text logs
        revision="main",
    )

    logger.info("upload checkpoints to different branches")
    branches = [branch.name for branch in api.list_repo_refs(repo_id).branches]
    for p in (run_path / ".checkpoints").iterdir():
        if "last" in p.name or p.stem in branches:
            continue
        logger.info(f"Uploading {p.stem}")
        ckpt = load_hf_from_pl(p)
        ckpt.push_to_hub(repo_id, revision=p.stem)  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(f"./outputs/tokenizers/{hparams['tok_name']}")
    tokenizer.push_to_hub(repo_id, revision="main")


if __name__ == "__main__":
    app()
