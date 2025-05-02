"""Commands for extracting information from byte-level LLMs and saving them as datasets."""

from collections import defaultdict
from pathlib import Path
from typing import Annotated
import os

import dill as pickle
import typer
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from rich import print
from transformers import AutoTokenizer
import pandas as pd

from huggingface_hub import list_repo_files

from commands.configs import (
    BYTE_DATA_SUBSET_FOLDER,
    BYTE_LLM_PREDICTION_DATA,
    BYTELEVEL_TOK_FOLDER,
    FINEWEBEDU_REPO_ID,
    HF_USERNAME,
    TOK_REPO_ID,
)

app = typer.Typer()


SPACE_TOKEN = 'Ġ'
NEWLINE_TOKEN = 'Ċ'


class ExtractTokenizerStats:

    def __init__(self, byte_tokenizer, tokenizer):
        self.byte_tokenizer = byte_tokenizer
        self.tokenizer = tokenizer

    def __call__(self, batch):
        text = [self.byte_tokenizer.decode(inp) for inp in batch['input_ids']]
        tokenized = self.tokenizer(text)
        batch['total_words'] = [len(tokens.tokens) for tokens in tokenized[:]]
        
        total_words_list = []
        continuation_lengths_list = []
        num_full_words_list = []
        num_continuation_words_list = []
        num_tokens_list = []

        for tokens in tokenized[:]:
            tokens = tokens.tokens

            current_length = 0
            total_words = 0
            continuation_lengths = []
            num_full_words = 0
            num_continuation = 0
            for token in tokens:
                if SPACE_TOKEN in token or NEWLINE_TOKEN in token:
                    if current_length == 0:
                        continue
                    total_words += 1
                    continuation_lengths.append(current_length)
                    if current_length == 1:
                        num_full_words += 1
                    else:
                        num_continuation += 1
                    current_length = 0
                if token != SPACE_TOKEN and token != NEWLINE_TOKEN:
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
        batch['total_words'] = total_words_list
        batch['continuation_lengths'] = continuation_lengths_list
        batch['num_full_words'] = num_full_words_list
        batch['num_continuation_words'] = num_continuation_words_list
        batch['num_tokens'] = [len(tokens.tokens) for tokens in tokenized[:]]
        return batch

@app.command()
def get_tokenizer_statistics(
    output_path: Annotated[Path, typer.Option(help="Output path for the tokenizer statistics")] = Path('tokenizer_stats.csv'),
    recalculate_if_exists: Annotated[bool, typer.Option(help="Recalculate if the file already exists")] = False,
) -> None:
    TOKENIZER_REPO = f"{HF_USERNAME}/{TOK_REPO_ID}"
    DATA_REPO = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"

    print(f"⚙️ Starting analysis of tokenizers in {TOKENIZER_REPO} directory")

    if os.path.exists(output_path) and not recalculate_if_exists:
        print(f"💡 File {output_path} already exists, not recalculating existing entries.")
        df = pd.read_csv(output_path)
        # Set the type of the split_lengths_distribution column to str
        df['split_lengths_distribution'] = df['split_lengths_distribution'].astype(str)
        print(f"✅ Successfully loaded tokenizer statistics from {output_path}")
    else:
        df = None

    byte_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=BYTELEVEL_TOK_FOLDER)
    byte_data = load_dataset(DATA_REPO, BYTE_DATA_SUBSET_FOLDER, split="train")

    files = list_repo_files(TOKENIZER_REPO)
    folders = set()
    for file in files:
        folders.add(str(Path(file).parent))
    folders.remove('.')

    if os.path.exists(output_path) and not recalculate_if_exists:
        for tokenizer_name in df['tokenizer_name'].values:
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
            num_proc=min(20, os.cpu_count() - 1)
        )

        df_dataset = processed_dataset.to_pandas()
        total_words = 0
        total_tokens = 0
        total_continuation_words = 0
        split_length_distribution = defaultdict(int)
        for _, row in df_dataset.iterrows():
            for length in row['continuation_lengths']:
                split_length_distribution[int(length)] += 1
            total_words += row['total_words']
            total_continuation_words += row['num_continuation_words']
            total_tokens += row['num_tokens']

        fertility = sum([k * v for k, v in split_length_distribution.items()]) / total_words
        proportion_continued = total_continuation_words / total_words

        new_row = {
            "tokenizer_name": folder,
            "fertility": fertility,
            "proportion_continued": proportion_continued,
            "total_split_words": total_words,
            "total_words": total_tokens,
            "split_lengths_distribution": str(split_length_distribution)
        }

        if df is not None:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(output_path, index=False)

    print(f"✅ Successfully extracted tokenizer statistics from {TOKENIZER_REPO} directory")
    print(f"✅ Successfully saved tokenizer statistics to {output_path}")

if __name__ == "__main__":
    app()
