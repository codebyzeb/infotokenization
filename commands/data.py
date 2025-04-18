import os
import shutil
from pathlib import Path

import typer
from datasets import Dataset, DatasetDict, load_dataset
from datatrove.data import DocumentsPipeline
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.huggingface import ParquetWriter
from datatrove.utils.batching import batched
from huggingface_hub import HfApi
from rich import print
from transformers import AutoTokenizer  # type: ignore

from commands.configs import (
    BYTE_DATA_FOLDER,
    BYTE_DATA_SUBSET_FOLDER,
    FINEWEBEDU_REPO_ID,
    HF_USERNAME,
    NUM_TRAIN_ROWS,
    TOK_REPO_ID,
)

app = typer.Typer()


@app.command()
def finewebedu_tokenize(
    tok_path: str = f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder: str | None = BYTE_DATA_FOLDER, batch_size: int = 1000
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
def finewebedu_subset(subset_size: int = NUM_TRAIN_ROWS, subfolder: str = BYTE_DATA_SUBSET_FOLDER) -> None:
    DATA_REPO_ID = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"

    print(f"⚙️ Creating a {subset_size}-row subset of FineWebEdu located at \n\t{DATA_REPO_ID=}")

    # Load the dataset
    dataset = load_dataset(DATA_REPO_ID, name=BYTE_DATA_FOLDER, split="train", streaming=True)
    dataset = list(dataset.take(subset_size))  # type: ignore
    dataset = Dataset.from_list(dataset)

    print(f"✅ Successfully created a {subset_size}-row subset of FineWebEdu dataset")
    print(f"🆙 Uploading the subset to {DATA_REPO_ID} on the HF Hub")

    dataset.push_to_hub(repo_id=DATA_REPO_ID, set_default=False, config_name=subfolder, max_shard_size="2GB")
    print("✅ Successfully created and uploaded the subset of FineWebEdu dataset")


@app.command()
def finewebedu_download(
    tok: str, local_dir: str = "./data", cache_dir: str = ".cache", num_train_rows: int = 20_000_000
) -> None:
    TARGET_REPO_ID = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"

    # Make cache dir absolute path
    cache_dir = os.path.abspath(cache_dir)
    local_dir = os.path.abspath(local_dir)
    print(f"Downloading {TARGET_REPO_ID}/{tok} and saving to {local_dir} (cache in {cache_dir})")
    ds: DatasetDict = load_dataset(TARGET_REPO_ID, name=tok, cache_dir=cache_dir, num_proc=min(12, os.cpu_count()))  # type: ignore

    total_size = len(ds["train"])
    print(f"Splitting {num_train_rows} docs for training and {total_size - num_train_rows} for validation")
    ds["validation"] = ds["train"].select(range(num_train_rows, total_size))
    ds["train"] = ds["train"].select(range(num_train_rows))

    out_path = f"{local_dir}/{TARGET_REPO_ID.split('/')[1]}/{tok}"
    print(f"Saving to {out_path}")
    ds.save_to_disk(out_path, max_shard_size="2GB", num_proc=min(12, os.cpu_count()))  # type: ignore

    print(f"Cleaning up {cache_dir} cache")
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
