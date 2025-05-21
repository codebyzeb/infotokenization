"""Commands for extracting information from byte-level LLMs and saving them as datasets."""

import math
from pathlib import Path
from typing import Annotated, Any

import dill as pickle
import torch
import typer
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from nltk.lm.api import LanguageModel
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from commands.configs import (
    BYTE_DATA_NGRAM_EXTRACTION,
    BYTE_LLM_MODEL_FOLDER,
    BYTE_LLM_PREDICTION_DATA,
    BYTE_MODELS_REPO_ID,
    BYTELEVEL_TOK_FOLDER,
    COMMONCORPUS_REPO_ID,
    FINEWEBEDU_REPO_ID,
    HF_USERNAME,
    NGRAM_MODEL_FOLDER,
    TOK_REPO_ID,
)

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = typer.Typer()


class Predictor:
    """
    Base class for predictors.

    The __call__ method should add four fields to the examples and return them:
    - Entropy
    - Surprisal
    - Space Probability
    - EOS Probability

    The last three should be represented as negative log probabilities.
    """

    def __init__(self, model, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        :param model: A model from `nltk.lm.model` or `transformers`.
        :param tokenizer: Tokenizer from `transformers`.
        """
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, examples: Any) -> Any:
        """Process a batch of examples."""
        raise NotImplementedError("Subclasses should implement this method.")


class NGramPredictor(Predictor):
    """Class for collecting information-theoretic measures from an ngram language model."""

    def __init__(self, model: LanguageModel, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        :param model: An ngram language model from `nltk.lm.model`.
        :param tokenizer: Tokenizer from `transformers`.
        """
        super().__init__(model, tokenizer)
        self.ctx_length = model.order - 1
        self.ctx_logscore_cache = {}

    def get_logscore(self, token: str, context: tuple) -> float:
        """Get logscore for a token given a context."""
        if context not in self.ctx_logscore_cache:
            self.ctx_logscore_cache[context] = {token: self.model.logscore(token, context)}
        elif token not in self.ctx_logscore_cache[context]:
            self.ctx_logscore_cache[context][token] = self.model.logscore(token, context)
        # Return the cached logscore
        return self.ctx_logscore_cache[context][token]

    def __call__(self, examples: dict) -> dict:
        """Process a batch of examples."""

        # Convert examples to token IDs
        texts = [["<s>"] * self.ctx_length + self.tokenizer.convert_ids_to_tokens(ex) for ex in examples["input_ids"]]

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
                context = tuple(text[i - self.ctx_length : i])
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
                space_prob = -self.get_logscore("Ġ", context)
                space_probs[-1].append(space_prob)

                # Calculate the probability of the end of sentence token
                eos_prob = -self.get_logscore("</s>", context)
                eos_probs[-1].append(eos_prob)

        examples["Entropy"] = entropies
        examples["Surprisal"] = surprisals
        examples["Space Probability"] = space_probs
        examples["EOS Probability"] = eos_probs

        return examples


class LLMPredictor(Predictor):
    """Class for collecting information-theoretic measures from a byte-level LLM."""

    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, stride=None, batch_size=8
    ) -> None:
        """
        :param model: A byte-level LLM from `transformers`.
        :param tokenizer: Tokenizer from `transformers`.
        """
        super().__init__(model, tokenizer)
        self.ctx_length = model.config.max_position_embeddings
        self.stride = stride if stride else model.config.max_position_embeddings // 4
        self.batch_size = batch_size
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.space_token_id = tokenizer.encode(" ")[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        assert self.space_token_id == tokenizer.convert_tokens_to_ids("Ġ"), (
            "Space token ID does not match the expected value."
        )

    def __call__(self, examples: Any) -> Any:
        """Process a batch of examples."""

        entropies = torch.tensor([]).to(self.device)
        surprisals = torch.tensor([]).to(self.device)
        space_probs = torch.tensor([]).to(self.device)
        eos_probs = torch.tensor([]).to(self.device)

        # Create a long vector of input IDs, adding the EOS token at the end of each example (as is done in training)
        long_ids = [l for x in examples["input_ids"] for l in (x + [self.eos_token_id])]
        long_ids = torch.tensor(long_ids, dtype=torch.int64).to(self.device)

        # Simulate a context for the first `ctx_length` tokens by adding the last `ctx_length - stride` tokens of the final example
        long_ids = torch.cat([long_ids[-(self.ctx_length - self.stride) :], long_ids], dim=0)

        # Pad the sequence so it is a multiple of the stride
        if len(long_ids) % self.stride != 0:
            pad_length = self.stride - (len(long_ids) % self.stride)
            long_ids = torch.cat([long_ids, torch.full((pad_length,), self.pad_token_id).to(self.device)], dim=0)

        # Use the stride to split the long sequence into overlapping sequences.
        # This lets us process the entire sequence in chunks, while still maintaining the context.
        num_vectors = (len(long_ids) - self.ctx_length + self.stride) // self.stride
        strided_ids = long_ids.as_strided(size=(num_vectors, self.ctx_length), stride=(self.stride, 1))

        # Ensure we can process the entire sequence in batches of self.batch_size
        while len(strided_ids) % self.batch_size != 0:
            strided_ids = torch.cat(
                [strided_ids, torch.full((1, self.ctx_length), self.pad_token_id).to(self.device)], dim=0
            )
        strided_ids = strided_ids.view(-1, self.batch_size, self.ctx_length)

        # Feed the strided IDs to the model in batches
        # and collect the surprisals, entropies, and probabilities that we want
        num_batches = len(strided_ids)
        for i in range(num_batches):
            with torch.no_grad():
                ids = strided_ids[i]
                logits = self.model(ids).logits.detach()

                # Logits should predict the next token, so we need to shift them
                # It's not a problem that this removes a token since below we
                # are only using the last `stride` tokens
                # (which are the ones we want to predict)
                logits = logits[:, :-1, :]
                ids = ids[:, 1:]

                for batch_idx in range(self.batch_size):
                    stride_ids = ids[batch_idx][-self.stride :]
                    stride_logits = logits[batch_idx][-self.stride :]

                    surprisal = self.loss_fct(stride_logits, stride_ids)
                    entropy = torch.distributions.Categorical(logits=stride_logits).entropy()
                    space_prob = self.loss_fct(
                        stride_logits, torch.full(stride_ids.shape, self.space_token_id).to(self.device)
                    )
                    eos_prob = self.loss_fct(
                        stride_logits, torch.full(stride_ids.shape, self.eos_token_id).to(self.device)
                    )

                    surprisals = torch.cat([surprisals, surprisal], dim=0)
                    entropies = torch.cat([entropies, entropy], dim=0)
                    space_probs = torch.cat([space_probs, space_prob], dim=0)
                    eos_probs = torch.cat([eos_probs, eos_prob], dim=0)

        # We recover the original examples by splitting around the EOS token
        # (remember that we added context to the front of length ctx_length+stride that needs removing here)
        eos_indices = torch.where(long_ids[self.ctx_length - self.stride :] == self.eos_token_id)[0]
        eos_indices = torch.cat([torch.tensor([-1]).to(self.device), eos_indices])

        examples["Entropy"] = [
            entropies[eos_indices[i] + 1 : eos_indices[i + 1]].tolist() for i in range(len(eos_indices) - 1)
        ]
        examples["Surprisal"] = [
            surprisals[eos_indices[i] + 1 : eos_indices[i + 1]].tolist() for i in range(len(eos_indices) - 1)
        ]
        examples["Space Probability"] = [
            space_probs[eos_indices[i] + 1 : eos_indices[i + 1]].tolist() for i in range(len(eos_indices) - 1)
        ]
        examples["EOS Probability"] = [
            eos_probs[eos_indices[i] + 1 : eos_indices[i + 1]].tolist() for i in range(len(eos_indices) - 1)
        ]

        return examples


SUPPORTED_MODELS = ["5-gram", "fw57M", "fw57M-multi"]
SUPPORTED_CORPORA = [FINEWEBEDU_REPO_ID, COMMONCORPUS_REPO_ID]


@app.command()
def get_llm_predictions(
    model_type: Annotated[
        str, typer.Argument(help=f"Type of model to use for predictions. Supported types: {SUPPORTED_MODELS}")
    ],
    corpus: Annotated[
        str, typer.Argument(help=f"Corpus to use for predictions. Supported corpora: {SUPPORTED_CORPORA}")
    ] = FINEWEBEDU_REPO_ID,
) -> None:
    MODEL_REPO = f"{HF_USERNAME}/{BYTE_MODELS_REPO_ID}"
    TOKENIZER_REPO = f"{HF_USERNAME}/{TOK_REPO_ID}"
    DATA_REPO = f"{HF_USERNAME}/{corpus}"
    TOKENIZER_NAME = BYTELEVEL_TOK_FOLDER + ('2' if 'multi' in model_type else '')
    CACHE_FOLDER = CACHE_DIR / model_type
    MODEL_CACHE_PATH = CACHE_FOLDER / "model"
    TARGET_FOLDER = Path(BYTE_LLM_PREDICTION_DATA) / model_type

    print(f"⚙️ Starting extraction process using {model_type} model")

    if model_type in ["5-gram"]:  # ngram models
        MODEL_NAME = f"{NGRAM_MODEL_FOLDER}/{model_type}-model.pkl"
        PREDICTOR_CLASS = NGramPredictor
        print(f"⚙️ Downloading {model_type} model from {MODEL_REPO}/{MODEL_NAME}")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO, filename=MODEL_NAME, cache_dir=MODEL_CACHE_PATH, force_download=False
        )
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    elif model_type in ["fw57M", "fw57M-multi"]:  # byte-level LLMs
        MODEL_NAME = f"{BYTE_LLM_MODEL_FOLDER}/{model_type}-tied"
        PREDICTOR_CLASS = LLMPredictor
        print(f"⚙️ Downloading {model_type} model from {MODEL_REPO}/{MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO, subfolder=MODEL_NAME, cache_dir=MODEL_CACHE_PATH, force_download=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: {SUPPORTED_MODELS}")

    # Download the model
    print(f"✅ Successfully downloaded {model_type} model to {MODEL_CACHE_PATH}")

    # Download the tokenizer
    print(f"⚙️ Downloading {TOKENIZER_NAME} tokenizer from {TOKENIZER_REPO}")
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder=TOKENIZER_NAME)  # type: ignore

    # Download and process the dataset
    print(f"⚙️ Downloading {BYTE_DATA_NGRAM_EXTRACTION} data from {DATA_REPO}")
    dataset: Dataset = load_dataset(DATA_REPO, name=BYTE_DATA_NGRAM_EXTRACTION, split="train")  # type: ignore

    print(f"⚙️ Processing dataset with {model_type} model (this can take a while)...")
    predictor = PREDICTOR_CLASS(model, tokenizer)
    processed_dataset = dataset.map(predictor, batched=True)
    print(f"✅ Successfully processed dataset with {model_type} model")

    PREDICTOR_CACHE_PATH = CACHE_FOLDER / "predictor"
    PREDICTOR_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTOR_NAME = "predictor.pkl"
    print(f"⚙️ Saving {model_type} predictor to disk at {PREDICTOR_CACHE_PATH}/{PREDICTOR_NAME}")
    with (PREDICTOR_CACHE_PATH / PREDICTOR_NAME).open("wb") as f:
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
        data_dir=str(TARGET_FOLDER),
        split=model_type.replace("-", ""),
    )


if __name__ == "__main__":
    app()
