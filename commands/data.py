import os
import shutil
from pathlib import Path

import typer
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.huggingface import ParquetWriter
from datatrove.utils.batching import batched
from huggingface_hub import HfApi
from rich import print
from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset

from commands.configs import BYTELEVEL_TOK_FOLDER, FINEWEBEDU_REPO_ID, HF_USERNAME, TOK_REPO_ID
from src import data

app = typer.Typer()


@app.command()
def finewebedu_tokenize(
    tok_path: str = f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder: str | None = BYTELEVEL_TOK_FOLDER, batch_size: int = 1000
) -> None:
    SOURCE_REPO_ID = "hf://datasets/pietrolesci/finewebedu-20B/data"
    TARGET_REPO_ID = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"
    tok_name = subfolder if subfolder is not None else Path(tok_path).name

    print(
        f"⚙️ Starting FineWebEdu tokenization pipeline\n\t{SOURCE_REPO_ID=}\n\t{TARGET_REPO_ID=}\n"
        f"Tokenizing with {tok_path}{'/' + subfolder if subfolder else ''} and {batch_size=}"
    )

    class DocumentTokenizer(PipelineStep):
        def __init__(self, pretrained_model_name_or_path: str, subfolder: str | None, batch_size: int) -> None:
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
            self.batch_size = batch_size

        def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:  # type: ignore
            for batch in batched(data, self.batch_size):
                with self.track_time(unit="batch"):
                    docs = [doc.text for doc in batch]
                    encoded_docs: list[list[int]] = self.tokenizer(docs)["input_ids"]  # type: ignore
                    for doc, encoded in zip(batch, encoded_docs, strict=True):
                        doc.metadata["input_ids"] = encoded
                        doc.metadata["num_tokens"] = len(encoded)  # for the future: this would have been convenient
                        yield doc

    out_path = Path("data") / f".tokenized_upload_{tok_name}"
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Tokenizing into {out_path=}")

    pipe_tokenize = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                SOURCE_REPO_ID,
                file_progress=True,
                doc_progress=True,
                shuffle_files=False,
                text_key="text",
                id_key="id",
                # limit=100,
            ),
            DocumentTokenizer(pretrained_model_name_or_path=tok_path, subfolder=subfolder, batch_size=batch_size),
            ParquetWriter(
                str(out_path),
                output_filename="${rank}.parquet",
                compression="zstd",
                adapter=lambda _, doc: {
                    "id": doc.id,
                    "input_ids": doc.metadata["input_ids"],
                    "num_tokens": doc.metadata["num_tokens"],
                },
                max_file_size=2 * (2**30),
            ),
        ],
        logging_dir=".datatrove/logs/finewebedu_tok",
        tasks=min(20, os.cpu_count() - 2),  # type: ignore
    )
    pipe_tokenize.run()
    print("✅ Successfully tokenized FineWebEdu dataset")

    print(f"🆙 Uploading to {TARGET_REPO_ID}")
    api = HfApi()
    api.create_repo(TARGET_REPO_ID, exist_ok=True, repo_type="dataset")
    print(f"🗂️ Repo created at {TARGET_REPO_ID}")

    api.upload_folder(repo_id=TARGET_REPO_ID, folder_path=str(out_path), path_in_repo=tok_name, repo_type="dataset")
    print(f"✅ Successfully uploaded to {TARGET_REPO_ID}")

    print("Cleaning up ./.datatrove cache")
    shutil.rmtree(".datatrove", ignore_errors=True)


@app.command()
def finewebedu_download(tok: str = "bytelevel", local_dir: str = "./data", cache_dir: str = ".cache") -> None:
    TARGET_REPO_ID = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"
    NUM_TRAIN = 20_000_000
    
    print(f"Downloading {TARGET_REPO_ID}/{tok} and saving to {local_dir} (cache in {cache_dir})")
    ds: DatasetDict = load_dataset(TARGET_REPO_ID, data_dir=tok, cache_dir=cache_dir, num_proc=min(12, os.cpu_count()))  # type: ignore

    total_size = len(ds["train"])
    print(f"Splitting {NUM_TRAIN} docs for training and {total_size - NUM_TRAIN} for validation")
    ds["validation"] = ds["train"].select(range(NUM_TRAIN, total_size))
    ds["train"] = ds["train"].select(range(NUM_TRAIN))

    out_path = f"{local_dir}/{TARGET_REPO_ID.split('/')[1]}/{tok}"
    print(f"Saving to {out_path}")
    ds.save_to_disk(out_path, max_shard_size="2GB", num_proc=min(12, os.cpu_count()))  # type: ignore

    print(f"Cleaning up {cache_dir} cache")
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
