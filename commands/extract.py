""" Commands for extracting information from byte-level LLMs and saving them as datasets."""

import os
import shutil
from pathlib import Path

import typer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from rich import print
import dill as pickle
from nltk.lm.api import LanguageModel
from transformers import AutoTokenizer
import math

from commands.configs import BYTE_LLM_PREDICTION_DATA, BYTE_MODELS_REPO_ID, BYTELEVEL_TOK_FOLDER, FINEWEBEDU_REPO_ID, HF_USERNAME, NGRAM_MODEL_FOLDER, TOK_REPO_ID, BYTE_DATA_SUBSET_FOLDER

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer()

class Predictor:
    """ Base class for predictors. """

    def __init__(self, model : LanguageModel, tokenizer : AutoTokenizer):
        """
        :param model: An ngram language model from `nltk.lm.model`.
        :param tokenizer: Tokenizer from `transformers`.
        """
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, examples):
        """ Process a batch of examples. """
        raise NotImplementedError("Subclasses should implement this method.")

class NGramPredictor(Predictor):
    """ Class for collecting information-theoretic measures from an ngram language model. """

    def __init__(self, model : LanguageModel, tokenizer : AutoTokenizer):
        """
        :param model: An ngram language model from `nltk.lm.model`.
        :param tokenizer: Tokenizer from `transformers`.
        """
        self.__init__(model, tokenizer)
        self.ctx_length = model.order - 1
        self.ctx_logscore_cache = {}

    def get_logscore(self, token, context):
        """ Get logscore for a token given a context. """
        if context not in self.ctx_logscore_cache:
            self.ctx_logscore_cache[context] = {token : self.model.logscore(token, context)}
        elif token not in self.ctx_logscore_cache[context]:
            self.ctx_logscore_cache[context][token] = self.model.logscore(token, context)
        # Return the cached logscore
        return self.ctx_logscore_cache[context][token]
    
    def __call__(self, examples):
        """ Process a batch of examples. """

        # Convert examples to token IDs
        texts = [["<s>"] * self.ctx_length + self.tokenizer.convert_ids_to_tokens(ex) for ex in examples['input_ids']]

        # Process each example
        entropies = []
        surprisals = []
        space_probs = []
        eos_probs = []
        for text in texts:
            entropies.append([])
            surprisals.append([])
            space_probs.append([])
            eos_probs.append([])
            
            for i in range(self.ctx_length, len(text)):
                # Calculate entropy
                context = tuple(text[i - self.ctx_length:i])
                entropy = 0
                for v in self.model.vocab:
                    logprob = self.get_logscore(v, context)
                    prob = math.pow(2, logprob)
                    if prob > 0:
                        entropy += prob * -math.log(prob, 2)
                entropies[-1].append(entropy)

                # Calculate surprisal
                surprisal = -self.get_logscore(text[i], context)
                surprisals[-1].append(surprisal)

                # Calculate probability of space token
                space_prob = -self.get_logscore('Ġ', context)
                space_probs[-1].append(space_prob)

                # Calculate the probability of the end of sentence token
                eos_prob = -self.get_logscore('</s>', context)
                eos_probs[-1].append(eos_prob)

        examples['Entropy'] = entropies
        examples['Surprisal'] = surprisals
        examples['Space Probability'] = space_probs
        examples['EOS Probability'] = eos_probs

        return examples
    

SUPPORTED_MODELS = ["ngram"]

@app.command()
def get_llm_predictions(
        model_type : str = "ngram",
    ) -> None:

    MODEL_REPO = f"{HF_USERNAME}/{BYTE_MODELS_REPO_ID}"
    TOKENIZER_REPO = f"{HF_USERNAME}/{TOK_REPO_ID}"
    DATA_REPO = f"{HF_USERNAME}/{FINEWEBEDU_REPO_ID}"
    TOKENIZER_NAME = BYTELEVEL_TOK_FOLDER

    if model_type == "ngram":
        MODEL_NAME = f"{NGRAM_MODEL_FOLDER}/5-gram-model.pkl"
        CACHE_FOLDER = CACHE_DIR / NGRAM_MODEL_FOLDER
        TARGET_FOLDER = Path(BYTE_LLM_PREDICTION_DATA) / NGRAM_MODEL_FOLDER
        PREDICTOR_CLASS = NGramPredictor
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: {SUPPORTED_MODELS}")

    print(f"⚙️ Starting extraction process using {model_type} model")
    print(f"⚙️ Downloading {model_type} model from {MODEL_REPO}/{MODEL_NAME}")

    MODEL_CACHE_PATH = CACHE_FOLDER / "model"
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_NAME,
        cache_dir=MODEL_CACHE_PATH,
        force_download=False,
    )
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Successfully downloaded {model_type} model to {MODEL_CACHE_PATH}")

    print(f"⚙️ Downloading {TOKENIZER_NAME} tokenizer from {TOKENIZER_REPO}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=TOKENIZER_NAME)

    print(f"⚙️ Downloading {BYTE_DATA_SUBSET_FOLDER} data from {DATA_REPO}")

    dataset = load_dataset(DATA_REPO, name=BYTE_DATA_SUBSET_FOLDER, split="train")

    print(f"⚙️ Processing dataset with {model_type} model (this can take a while)...")

    predictor = PREDICTOR_CLASS(model, tokenizer)
    processed_dataset = dataset.map(predictor, batched=True)

    print(f"✅ Successfully processed dataset with {model_type} model")

    PREDICTOR_CACHE_PATH = CACHE_FOLDER / "predictor"
    PREDICTOR_NAME = "predictor.pkl"
    print(f"⚙️ Saving {model_type} predictor to disk at {PREDICTOR_CACHE_PATH}/{PREDICTOR_NAME}")

    os.makedirs(PREDICTOR_CACHE_PATH, exist_ok=True)
    with open(PREDICTOR_CACHE_PATH / PREDICTOR_NAME, 'wb') as f:
        pickle.dump(predictor, f)

    PROCESSED_DATASET_CACHE_PATH = CACHE_FOLDER / "processed_dataset"
    print(f"⚙️ Saving processed dataset to disk at {PROCESSED_DATASET_CACHE_PATH}")

    processed_dataset.save_to_disk(PROCESSED_DATASET_CACHE_PATH, max_shard_size="2GB")
    
    print(f"🆙 Uploading the processed dataset to {DATA_REPO}/{TARGET_FOLDER}")
          
    processed_dataset.push_to_hub(
        repo_id=DATA_REPO,
        private=False,
        set_default=False,
        commit_message=f"Update prediction data with {model_type} processor",
        max_shard_size="2GB",
        config_name=BYTE_LLM_PREDICTION_DATA,
        data_dir = TARGET_FOLDER,
        split=model_type,
    )

if __name__ == "__main__":
    app()
