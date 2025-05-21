"""Commands for extracting information from byte-level LLMs and saving them as datasets."""

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from datasets import Dataset, load_dataset
from huggingface_hub import list_repo_files
from rich import print
from tokenizers import models
from transformers import AutoTokenizer  # type: ignore

from commands.configs import (
    BYTE_DATA_TOKENIZER_EVALUATION,
    BYTELEVEL_TOK_FOLDER,
    COMMONCORPUS_REPO_ID,
    FINEWEBEDU_REPO_ID,
    HF_USERNAME,
    LANGUAGES,
    TOK_REPO_ID,
)

app = typer.Typer()


SPACE_TOKEN = "Ġ"
NEWLINE_TOKEN = "Ċ"
CONTINUATION_TOKEN = "##"


class ExtractTokenizerStats:
    def __init__(self, byte_tokenizer, tokenizer):
        self.byte_tokenizer = byte_tokenizer
        self.tokenizer = tokenizer
        if type(self.tokenizer.backend_tokenizer.model) is models.BPE:
            self.tokenizer_type = "BPE"
        elif type(self.tokenizer.backend_tokenizer.model) is models.WordPiece:
            self.tokenizer_type = "WordPiece"
        else:
            raise ValueError(f"Unsupported tokenizer type: {type(self.tokenizer.backend_tokenizer.model)}")

    def is_new_word(self, token):
        if self.tokenizer_type == "BPE":
            return SPACE_TOKEN in token or NEWLINE_TOKEN in token
        elif self.tokenizer_type == "WordPiece":
            return not token.startswith(CONTINUATION_TOKEN)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def __call__(self, batch):
        text = [self.byte_tokenizer.decode(inp) for inp in batch["input_ids"]]
        tokenized = self.tokenizer(text)
        batch["token_ids"] = [tokens.ids for tokens in tokenized[:]]
        batch["num_tokens"] = [len(tokens.tokens) for tokens in tokenized[:]]

        total_words_list = []
        continuation_lengths_list = []
        num_full_words_list = []
        num_continuation_words_list = []
        num_tokens_list = []
        num_unk_list = []

        for tokens in tokenized[:]:
            tokens = tokens.tokens

            current_length = 0
            total_words = 0
            continuation_lengths = []
            num_full_words = 0
            num_continuation = 0
            num_unk = 0
            for token in tokens:
                if self.tokenizer_type == "WordPiece" and token == self.tokenizer.unk_token:
                    num_unk += 1
                if self.is_new_word(token):
                    if current_length == 0:
                        continue
                    total_words += 1
                    continuation_lengths.append(current_length)
                    if current_length == 1:
                        num_full_words += 1
                    else:
                        num_continuation += 1
                    current_length = 0
                if token == SPACE_TOKEN or token == NEWLINE_TOKEN:
                    continue
                current_length += 1
            if current_length > 0:
                total_words += 1
                continuation_lengths.append(current_length)
                if current_length == 1:
                    num_full_words += 1
                else:
                    num_continuation += 1
            total_words_list.append(total_words)
            continuation_lengths_list.append(continuation_lengths)
            num_full_words_list.append(num_full_words)
            num_continuation_words_list.append(num_continuation)
            num_tokens_list.append(len(tokens))
            num_unk_list.append(num_unk)
        batch["total_words"] = total_words_list
        batch["continuation_lengths"] = continuation_lengths_list
        batch["num_full_words"] = num_full_words_list
        batch["num_continuation_words"] = num_continuation_words_list
        batch["num_unk"] = num_unk_list
        return batch


@app.command()
def get_tokenizer_statistics_fineweb(
    output_path: Annotated[Path, typer.Option(help="Output path for the tokenizer statistics")] = Path(
        "tokenizer_stats_fineweb.csv"
    ),
    recalculate_if_exists: Annotated[bool, typer.Option(help="Recalculate if the file already exists")] = False,
) -> None:
    TOKENIZER_REPO = f"{HF_USERNAME}/{TOK_REPO_ID}"
    DATA_REPO = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"

    print(f"⚙️ Starting analysis of tokenizers in {TOKENIZER_REPO} directory")

    if os.path.exists(output_path) and not recalculate_if_exists:
        print(f"💡 File {output_path} already exists, not recalculating existing entries.")
        df = pd.read_csv(output_path)
        # Set the type of the split_lengths_distribution column to str
        df["split_lengths_distribution"] = df["split_lengths_distribution"].astype(str)
        print(f"✅ Successfully loaded tokenizer statistics from {output_path}")
    else:
        df = None

    byte_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=BYTELEVEL_TOK_FOLDER)
    byte_data: Dataset = load_dataset(DATA_REPO, BYTE_DATA_TOKENIZER_EVALUATION, split="train")  # type: ignore

    files = list_repo_files(TOKENIZER_REPO)
    folders = set()
    for file in files:
        folder = str(Path(file).parent)
        if "multi" not in folder:
            folders.add(folder)
        else:
            print(f"💡 Skipping {folder} tokenizer as it is a multilingual tokenizer")
    folders.remove(".")

    if os.path.exists(output_path) and not recalculate_if_exists:
        for tokenizer_name in df["tokenizer_name"].values:  # FIXME!
            if tokenizer_name in folders:
                folders.remove(tokenizer_name)
        if len(folders) == 0:
            print(f"💡 No new tokenizers found in {TOKENIZER_REPO} directory, terminating.")
            return
        print(f"💡 Found {len(folders)} new tokenizers in {TOKENIZER_REPO} directory")
    else:
        print(f"💡 Found {len(folders)} tokenizers in {TOKENIZER_REPO} directory")

    for folder in folders:
        print(f"⚙️ Processing tokenizer: {folder}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=folder)
        processed_dataset = byte_data.map(
            ExtractTokenizerStats(byte_tokenizer, tokenizer),
            batched=True,
            desc="Extracting statistics from dataset",
            num_proc=min(20, os.cpu_count() - 1),  # type: ignore
        )

        df_dataset: pd.DataFrame = processed_dataset.to_pandas()  # type: ignore
        total_words = 0
        total_tokens = 0
        total_continuation_words = 0
        split_length_distribution = defaultdict(int)
        num_unk = 0
        frequency = Counter()

        for _, row in df_dataset.iterrows():
            for length in row["continuation_lengths"]:
                split_length_distribution[int(length)] += 1
            total_words += row["total_words"]
            total_continuation_words += row["num_continuation_words"]
            total_tokens += row["num_tokens"]
            num_unk += row["num_unk"]
            frequency += Counter(row["token_ids"])

        fertility = sum([k * v for k, v in split_length_distribution.items()]) / total_words if total_words > 0 else 0
        proportion_continued = total_continuation_words / total_words if total_words > 0 else 0

        # Compute Renyi entropy
        token_freq = list(frequency.values())
        total_subwords = sum(token_freq)
        token_probs = [freq / total_subwords for freq in token_freq]

        power = 2.5
        scale = 1 / (1 - power)
        renyi = scale * np.log2(np.sum(np.array(token_probs) ** power)) / np.log2(len(token_probs))

        new_row = {
            "tokenizer_name": folder,
            "fertility": fertility,
            "proportion_continued": proportion_continued,
            "total_split_words": total_continuation_words,
            "total_words": total_words,
            "total_tokens": total_tokens,
            "split_lengths_distribution": str(split_length_distribution),
            "num_unk": num_unk,
            "renyi_efficiency": renyi,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) if df is not None else pd.DataFrame([new_row])
        df.to_csv(output_path, index=False)

    print(f"✅ Successfully extracted tokenizer statistics from {TOKENIZER_REPO} directory")
    print(f"✅ Successfully saved tokenizer statistics to {output_path}")


@app.command()
def get_tokenizer_statistics_common_corpus(
    output_path: Annotated[Path, typer.Option(help="Output path for the tokenizer statistics")] = Path(
        "tokenizer_stats_common.csv"
    ),
    recalculate_if_exists: Annotated[bool, typer.Option(help="Recalculate if the file already exists")] = False,
) -> None:
    TOKENIZER_REPO = f"{HF_USERNAME}/{TOK_REPO_ID}"
    DATA_REPO = f"{HF_USERNAME}/{COMMONCORPUS_REPO_ID}"

    print(f"⚙️ Starting analysis of tokenizers in {TOKENIZER_REPO} directory")

    if os.path.exists(output_path) and not recalculate_if_exists:
        print(f"💡 File {output_path} already exists, not recalculating existing entries.")
        df = pd.read_csv(output_path)
        print(f"✅ Successfully loaded tokenizer statistics from {output_path}")
    else:
        df = None

    byte_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=BYTELEVEL_TOK_FOLDER)
    byte_data = load_dataset(DATA_REPO, BYTE_DATA_TOKENIZER_EVALUATION, split="train")
    languages = LANGUAGES

    files = list_repo_files(TOKENIZER_REPO)
    folders = set()
    for file in files:
        if "multi" in str(Path(file).parent):
            folders.add(str(Path(file).parent))
        else:
            print(f"💡 Skipping {str(Path(file).parent)} tokenizer as it is a monolingual tokenizer")

    if os.path.exists(output_path) and not recalculate_if_exists:
        for tokenizer_name in df["tokenizer_name"].values:
            if tokenizer_name in folders:
                folders.remove(tokenizer_name)
        if len(folders) == 0:
            print(f"💡 No new tokenizers found in {TOKENIZER_REPO} directory, terminating.")
            return
        print(f"💡 Found {len(folders)} new tokenizers in {TOKENIZER_REPO} directory")
    else:
        print(f"💡 Found {len(folders)} tokenizers in {TOKENIZER_REPO} directory")

    for folder in folders:
        print(f"⚙️ Processing tokenizer: {folder}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=folder)
        processed_dataset = byte_data.map(
            ExtractTokenizerStats(byte_tokenizer, tokenizer),
            batched=True,
            desc="Extracting statistics from dataset",
            num_proc=min(20, os.cpu_count() - 1),
        )

        df_dataset = processed_dataset.to_pandas()
        total_words = {lang: 0 for lang in languages}
        total_tokens = {lang: 0 for lang in languages}
        unique_tokens = {lang : 0 for lang in languages}
        total_continuation_words = {lang: 0 for lang in languages}
        split_length_distribution = {lang: defaultdict(int) for lang in languages}
        num_unk = {lang: 0 for lang in languages}
        frequencies = {lang : Counter() for lang in languages}
        for _, row in df_dataset.iterrows():
            language = row["language"]
            for length in row["continuation_lengths"]:
                split_length_distribution[language][int(length)] += 1
            total_words[language] += row["total_words"]
            total_continuation_words[language] += row["num_continuation_words"]
            total_tokens[language] += row["num_tokens"]
            num_unk[language] += row["num_unk"]
            frequencies[language] += Counter(row["token_ids"])

        fertility = {
            lang: sum([k * v for k, v in split_length_distribution[lang].items()]) / total_words[lang] if total_words[lang] > 0 else 0
            for lang in languages
        }
        proportion_continued = {lang: (total_continuation_words[lang] / total_words[lang] if total_words[lang] > 0 else 0) for lang in languages}

        # Compute Renyi entropy
        token_freq = {lang: list(frequencies[lang].values()) for lang in languages}
        unique_tokens = {lang: len(token_freq[lang]) for lang in languages}
        total_subwords = {lang: sum(token_freq[lang]) for lang in languages}
        token_probs = {lang : [freq / total_subwords[lang] for freq in token_freq[lang]] for lang in languages}

        power = 2.5
        scale = 1 / (1 - power)
        renyi = {lang : scale * np.log2(np.sum(np.array(token_probs[lang]) ** power)) / np.log2(len(token_probs[lang])) for lang in languages}

        new_row = {
            "tokenizer_name": [folder] * len(languages),
            "language": languages,
            "fertility": [fertility[lang] for lang in languages],
            "proportion_continued": [proportion_continued[lang] for lang in languages],
            "total_split_words": [total_continuation_words[lang] for lang in languages],
            "total_words": [total_words[lang] for lang in languages],
            "total_tokens": [total_tokens[lang] for lang in languages],
            "unique_tokens": [unique_tokens[lang] for lang in languages],
            "num_unk": [num_unk[lang] for lang in languages],
            "renyi_efficiency": [renyi[lang] for lang in languages],
        }

        df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True) if df is not None else pd.DataFrame(new_row)

        df.to_csv(output_path, index=False)

    print(f"✅ Successfully extracted tokenizer statistics from {TOKENIZER_REPO} directory")
    print(f"✅ Successfully saved tokenizer statistics to {output_path}")


if __name__ == "__main__":
    app()
