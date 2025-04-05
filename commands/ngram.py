import os
import shutil
import copy
import typer
import multiprocessing
from pathlib import Path
import dill as pickle 

from huggingface_hub import HfApi
from rich import print
from transformers import AutoTokenizer
from datasets import load_dataset

from nltk.lm import AbsoluteDiscountingInterpolated as NGRAM_MODEL
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.counter import NgramCounter

from commands.configs import BYTE_DATA_SUBSET_FOLDER, BYTELEVEL_TOK_FOLDER, FINEWEBEDU_REPO_ID, HF_USERNAME, TOK_REPO_ID, MAX_NGRAM_LENGTH, NUM_TRAIN_ROWS, NGRAM_MODEL_FOLDER, BYTE_MODELS_REPO_ID

app = typer.Typer()

def _update_counter_with_counter(counter1 : NgramCounter, counter2 : NgramCounter):
    """
    Add two Counter objects together, merging their counts.
    """
    # Update unigram counts
    for key, value in counter2[1].items():
        if key not in counter1[1]:
            counter1[1][key] = value
        else:
            counter1[1][key] += value

    # Update conditional frequency counts
    for order in range(2, len(counter1) + 1):
        for context, freqs in counter2[order].items():
            if context not in counter1[order]:
                counter1[order][context] = freqs.copy()
            else:
                for word, freq in freqs.items():
                    if word not in counter1[order][context]:
                        counter1[order][context][word] = freq
                    else:
                        counter1[order][context][word] += freq

def _combine_ngram_models(models):
    """
    Combine multiple Kneser-Ney models into a single model.

    :param models: A list of ngram models to combine.
    :return: A combined ngram model.
    """
    # Make a copy of the first model
    # to avoid modifying the original model
    combined_model = copy.deepcopy(models[0])

    # Update vocabulary
    for model in models[1:]:
        for word in model.vocab:
            if word not in combined_model.vocab:
                combined_model.vocab.counts[word] = model.vocab[word]
            else:
                combined_model.vocab.counts[word] += model.vocab[word]
    
    # Update the counts
    for model in models[1:]:
        _update_counter_with_counter(combined_model.counts, model.counts)

    return combined_model

def _train(data_chunk):
    train_data, padded_sents = padded_everygram_pipeline(order=MAX_NGRAM_LENGTH, text=data_chunk)
    model = NGRAM_MODEL(order=MAX_NGRAM_LENGTH)
    model.fit(train_data, padded_sents)
    
    return model

class TokenIterable:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __iter__(self):
        for sample in self.dataset:
            yield self.tokenizer.convert_ids_to_tokens(sample['input_ids'])

@app.command()
def train_ngram_model() -> None:
    data_path = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"
    tokenizer_path = f"{HF_USERNAME}/{TOK_REPO_ID}"
    folder_path = Path(f"{BYTE_MODELS_REPO_ID}/{NGRAM_MODEL_FOLDER}")
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"💡 Will save the ngram model locally to to: {folder_path}")

    # Load tokenizer data
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path, subfolder=BYTELEVEL_TOK_FOLDER)
    dataset = load_dataset(data_path, name=BYTE_DATA_SUBSET_FOLDER, split="train", streaming=True)
    dataset = dataset.select_columns("input_ids")

    # Split into chunks for parallel processing
    num_chunks = multiprocessing.cpu_count()
    print(f"💡 Splitting data into {num_chunks} shards for training ngram LMs")
    iterables = []
    for i in range(num_chunks):
        iterables.append(TokenIterable(dataset.shard(num_chunks, index=i).take(NUM_TRAIN_ROWS // num_chunks), tokenizer))

    print("⚙️ Training the KneserNeyInterpolated ngram models (this may take a while)...")
    with multiprocessing.Pool(num_chunks) as pool:
        models = pool.map(_train, iterables)
    model = _combine_ngram_models(models)
    print("✅ Successfully trained ngram models")

    with open(folder_path / f"{MAX_NGRAM_LENGTH}-gram-model.pkl", 'wb') as fout:
        pickle.dump(model, fout)

    repo_id = f"{HF_USERNAME}/{folder_path.parent}"
    print(f"🆙 Uploading the model to {repo_id} on the HF Hub")

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=folder_path, repo_id=repo_id, path_in_repo=folder_path.name, repo_type="model", revision="main"
    )

    print(f"✅ Successfully created and uploaded the ngram model to {repo_id}")

    shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == "__main__":
    app()
