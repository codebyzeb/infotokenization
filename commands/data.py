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
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from commands.configs import (
    BYTE_DATA_FOLDER,
    BYTE_DATA_NGRAM_EXTRACTION,
    BYTE_DATA_NGRAM_TRAINING,
    COMMONCORPUS_REPO_ID,
    FINEWEBEDU_REPO_ID,
    HF_USERNAME,
    LANGUAGES,
    NUM_TRAIN_ROWS,
    TOK_REPO_ID,
    TOKENS_PER_LANGUAGE,
)

app = typer.Typer()
class AddPreTokenizationBoundaries():

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        :param tokenizer: Tokenizer from `transformers`.
        """
        self.tokenizer = tokenizer

    def __call__(self, examples) -> dict:
        """Add pre-tokenization boundaries to the examples."""
        input_ids = examples["input_ids"]
        if len(examples) == 1:
            input_ids = [input_ids]
        pre_token_boundaries_list = []
        for ids in input_ids:
            text = self.tokenizer.decode(ids)
            pre_tokenized = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
            pre_token_boundaries = [False] * len(ids)
            prev_boundary = 0
            for t in pre_tokenized:
                pre_token_boundaries[prev_boundary] = True
                pre_token = t[0]
                left_boundary = t[1][0]
                right_boundary = t[1][1]
                token = text[left_boundary:right_boundary]
                tokenized_token = self.tokenizer.tokenize(token)
                tokenized_length = len(tokenized_token)
                # We tokenizing every pre_token individually, which may
                # incorrectly add an initial space token
                if pre_token[0] != 'Ġ':
                    tokenized_length -= 1 
                prev_boundary += tokenized_length
            assert(prev_boundary == len(ids))
            pre_token_boundaries_list.append(pre_token_boundaries)
        examples["pre_token_boundaries"] = pre_token_boundaries_list
        return examples 

@app.command()
def finewebedu_tokenize(
    tok_path: str = f"{HF_USERNAME}/{TOK_REPO_ID}",
    subfolder: str | None = BYTE_DATA_FOLDER,
    batch_size: int = 1000
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
def tokenize_and_subset_commoncorpus(
    tokens_per_language: int = TOKENS_PER_LANGUAGE * 8,  # Overshoot a bit here just in case, we cut this down later
    languages: list[str] = LANGUAGES,
    tok_path: str = f"{HF_USERNAME}/{TOK_REPO_ID}",
    subfolder: str | None = BYTE_DATA_FOLDER,
    batch_size: int = 1000,
) -> None:
    SOURCE_REPO_ID = "/home/zg258/projects/infotokenization/data/common_corpus"
    TARGET_REPO_ID = f"{HF_USERNAME}/{COMMONCORPUS_REPO_ID}"
    tok_name = subfolder if subfolder is not None else Path(tok_path).name

    print(
        f"⚙️ Starting Common Corpus tokenization pipeline\n\t{SOURCE_REPO_ID=}\n\t{TARGET_REPO_ID=}\n"
        f"Tokenizing with {tok_path}{'/' + subfolder if subfolder else ''} and {batch_size=}"
    )

    language_token_counts = {lang: 0 for lang in languages}

    class DocumentTokenizer(PipelineStep):
        def __init__(self, pretrained_model_name_or_path: str, subfolder: str | None, batch_size: int) -> None:
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
            self.batch_size = batch_size

        def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:  # type: ignore
            for batch in batched(data, self.batch_size):
                with self.track_time(unit="batch"):
                    filtered_batch = [doc for doc in batch if doc.metadata.get("language", None) in languages]
                    filtered_batch = [
                        doc
                        for doc in filtered_batch
                        if language_token_counts[doc.metadata["language"]] < tokens_per_language
                    ]
                    if not filtered_batch:
                        continue
                    docs = [doc.text for doc in filtered_batch]
                    encoded_docs: list[list[int]] = self.tokenizer(docs)["input_ids"]  # type: ignore
                    for doc, encoded in zip(filtered_batch, encoded_docs, strict=True):
                        # Within the batch, we may have now more than tokens_per_language
                        if language_token_counts[doc.metadata["language"]] > tokens_per_language:
                            continue
                        num_tokens = len(encoded)
                        doc.metadata["input_ids"] = encoded
                        doc.metadata["num_tokens"] = num_tokens
                        language_token_counts[doc.metadata["language"]] += num_tokens
                        yield doc

    out_path = Path("data") / f".tokenized_upload_{tok_name}"
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Tokenizing into {out_path=} with a subset of {tokens_per_language=} for each {languages=}")

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
                    "language": doc.metadata["language"],
                    "text": doc.text,
                },
                max_file_size=2 * (2**30),
            ),
        ],
        logging_dir=".datatrove/logs/{name}_tok",
    )
    pipe_tokenize.run()
    print("✅ Successfully tokenized Common Corpus dataset")
    print("Tokenization complete. Language token counts:")
    for language, count in language_token_counts.items():
        print(f"  - {language}: {count} tokens")

    print(f"🆙 Uploading to {TARGET_REPO_ID}")
    api = HfApi()
    api.create_repo(TARGET_REPO_ID, exist_ok=True, repo_type="dataset")
    print(f"🗂️ Repo created at {TARGET_REPO_ID}")

    api.upload_folder(repo_id=TARGET_REPO_ID, folder_path=str(out_path), path_in_repo=tok_name, repo_type="dataset")
    print(f"✅ Successfully uploaded to {TARGET_REPO_ID}")

    print("Cleaning up ./.datatrove cache")
    shutil.rmtree(".datatrove", ignore_errors=True)


@app.command()
def commoncorpus_subset(
    languages: list[str] = LANGUAGES,
    tokens_per_language: int = TOKENS_PER_LANGUAGE,
    subfolder: str = BYTE_DATA_NGRAM_TRAINING,
    shift_amount: int = 0,
    add_pre_tokenization_boundaries: bool = False,
) -> None:
    DATA_REPO_ID = f"{HF_USERNAME}/{COMMONCORPUS_REPO_ID}"

    print(f"⚙️ Creating a {tokens_per_language}-token subset of Common Corpus located at \n\t{DATA_REPO_ID=}")

    language_token_counts = {lang: 0 for lang in languages}

    def filter_fn(example):
        if example["language"] not in languages or language_token_counts[example["language"]] >= tokens_per_language * (
            shift_amount + 1
        ):
            return False
        language_token_counts[example["language"]] += len(example["input_ids"])
        # If we're shifting, we want to skip the first `shift_amount` * `tokens_per_language` tokens
        if language_token_counts[example["language"]] < tokens_per_language * (shift_amount):
            return False
        return True

    # Load the dataset
    dataset = load_dataset(DATA_REPO_ID, name=BYTE_DATA_FOLDER, split="train", streaming=True)
    dataset = dataset.filter(filter_fn)  # type: ignore
    dataset = list(dataset)
    dataset = Dataset.from_list(dataset)

    if add_pre_tokenization_boundaries:
        tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder=BYTE_DATA_FOLDER)
        dataset = dataset.map(AddPreTokenizationBoundaries(tokenizer), batched=True)


    print(
        f"✅ Successfully created a subset of Common Corpus dataset shifted by {shift_amount} * {tokens_per_language} tokens per language"
    )
    print("Language token counts:")
    for language, count in language_token_counts.items():
        print(f"  - {language}: {count} tokens")
    print(f"🆙 Uploading the subset to {DATA_REPO_ID} on the HF Hub")

    config_name = subfolder if shift_amount == 0 else f"{subfolder}_{shift_amount}"
    dataset.push_to_hub(repo_id=DATA_REPO_ID, set_default=False, config_name=config_name, max_shard_size="2GB")
    print("✅ Successfully created and uploaded the subset of Common Corpus dataset")


@app.command()
def finewebedu_subset(
    subset_size: int = NUM_TRAIN_ROWS,
    subfolder: str = BYTE_DATA_NGRAM_TRAINING,
    shift_amount: int = 0,
    add_pre_tokenization_boundaries: bool = False,
) -> None:
    DATA_REPO_ID = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"

    print(f"⚙️ Creating a {subset_size}-row subset of FineWebEdu located at \n\t{DATA_REPO_ID=}")

    # Load the dataset
    dataset = load_dataset(DATA_REPO_ID, name=BYTE_DATA_FOLDER, split="train", streaming=True)
    if shift_amount > 0:
        print(f"⚙️ Shifting the dataset by {shift_amount} * {subset_size} rows")
    for i in range(shift_amount):
        dataset = dataset.skip(subset_size)
    dataset = list(dataset.take(subset_size))  # type: ignore
    dataset = Dataset.from_list(dataset)

    if add_pre_tokenization_boundaries:
        tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{TOK_REPO_ID}", subfolder=BYTE_DATA_FOLDER)
        dataset = dataset.map(AddPreTokenizationBoundaries(tokenizer), batched=True)

    print(f"✅ Successfully created a {subset_size}-row subset of FineWebEdu dataset")
    print(f"🆙 Uploading the subset to {DATA_REPO_ID} on the HF Hub")

    config_name = subfolder if shift_amount == 0 else f"{subfolder}_{shift_amount}"
    dataset.push_to_hub(repo_id=DATA_REPO_ID, set_default=False, config_name=config_name, max_shard_size="2GB")
    print("✅ Successfully created and uploaded the subset of FineWebEdu dataset")


@app.command()
def finewebedu_download(
    tok: str,
    local_dir: str = "./data",
    cache_dir: str = ".cache",
    num_train_rows: int = 20_000_000
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


@app.command()
def download_bytelevel(local_dir: str = "./data", cache_dir: str = ".cache", repo_id: str = FINEWEBEDU_REPO_ID) -> None:
    TARGET_REPO_ID = f"{HF_USERNAME}/{repo_id}"

    # Make cache dir absolute path
    cache_dir = os.path.abspath(cache_dir)
    local_dir = os.path.abspath(local_dir)
    print(f"Downloading subset from {TARGET_REPO_ID} and saving to {local_dir} (cache in {cache_dir})")
    ds: DatasetDict = load_dataset(
        TARGET_REPO_ID, name=BYTE_DATA_NGRAM_TRAINING, cache_dir=cache_dir, num_proc=min(12, os.cpu_count())
    )  # type: ignore
    ds_val: Dataset = load_dataset(
        TARGET_REPO_ID, name=BYTE_DATA_NGRAM_EXTRACTION, cache_dir=cache_dir, num_proc=min(12, os.cpu_count())
    )  # type: ignore

    total_size = len(ds["train"])
    print(
        f"Using '{BYTE_DATA_NGRAM_TRAINING}' subset for training and 1/100th of '{BYTE_DATA_NGRAM_EXTRACTION}' subset for validation"
    )
    ds["validation"] = ds_val["train"].select(range(len(ds_val["train"]) // 100))
    if repo_id == COMMONCORPUS_REPO_ID:
        ds["train"] = ds["train"].remove_columns(["text"])
        ds["validation"] = ds["validation"].remove_columns(["text"])
        ds["train"] = ds["train"].remove_columns(["language"])
        ds["validation"] = ds["validation"].remove_columns(["language"])

    out_path = f"{local_dir}/{TARGET_REPO_ID.split('/')[1]}/{BYTE_DATA_NGRAM_TRAINING}"
    print(f"Saving to {out_path}")
    ds.save_to_disk(out_path, max_shard_size="2GB", num_proc=min(12, os.cpu_count()))  # type: ignore

    print(f"Cleaning up {cache_dir} cache")
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
