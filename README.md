## General Intro and Setup

Here we explain how to use the repo.


### Setup

First, clone the repo

```bash
git clone https://github.com/codebyzeb/infotokenization.git
```

Install the [`uv`](https://docs.astral.sh/uv/concepts/projects) environment manager

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# or simply update it
uv self update
```

Then, install the dependencies. Because of flash attention we do a two step installation process —
see [`uv` docs](https://docs.astral.sh/uv/concepts/projects/config/#build-isolation)

```bash
uv sync
uv sync --extra flash  # if you also want flash-attention
```

To lint and format your code you can use 

```bash
make lint
make format
```

### Run commands via the CLI

We define commands in the `/commands` folder. You can run any available command as follows

```bash
uv run cli.py <command> <subcommand> <args> <options>
```


For example, to create a simple byte-level tokenizer

```bash
# Note: no args or options needed here
uv run cli.py tokenizers create-bytelevel
```




## Reproducing Experiments

Here we describe how to run the experiments.


### Train the ByteLevel model

Download the preprocessed data from the Hugging Face Hub

```bash
uv run cli.py data download-bytelevel
```

This saves the data to `./data/finewebedu/bytelevel`. Similarly, to download common-corpus:

```bash
uv run cli.py data download-bytelevel --repo-id common-corpus
```

Once you have the data, you can train the bytelevel models as follows:

```bash
uv run scripts/train.py \
    dataset=finewebedu \
    tok_name=bytelevel \
    model=fw57M-tied \
    hydra.run.dir=outputs/finewebedu/bytelevel

uv run scripts/train.py \
    dataset=common-corpus \
    tok_name=bytelevel \
    model=fw57M-tied \
    hydra.run.dir=outputs/common-corpus/bytelevel
```

The models will be saved locally to `outputs/`. The training script can also take other arguments, see `launch_slurm_llm_trainer.wilkes3` for how our models were trained on a CSD3 cluster. The model can then be uploaded, e.g:

```bash
uv run cli.py upload model outputs/finewebedu/bytelevel
```

Note that you may need to adjust the configurations in `commands/configs.py` to match your Huggingface credentials.

### Extract predictions from the trained model

Predictions can be extracted from the model as follow:

```bash
uv run cli.py extract get-llm-predictions fw57M
```

This automatically downloads the bytelevel model, the first subset of finewebedu, extracts preditions, and uploads these as a new subset of finewebedu. To do the same for common-corpus:

```bash
uv run cli.py extract get-llm-predictions fw57M-multi common-corpus
```

### Train ByteSpan tokenizer

To create a ByteSpan tokenizer with the global constraint:

```bash
uv run cli.py tokenizers create-thresholdtokenizer Entropy
```

Entropy can be replaced with Surprisal or other cues (same with below). To create a ByteSpan tokenizer with the monotonic constraint:

```bash
uv run cli.py tokenizers create-bytespantokenizer Entropy
```

To create a ByteSpan tokenizer with the combined constraint:

```bash
uv run cli.py tokenizers create-bytespantokenizer Entropy --threshold-percentile=30
```

This sets the threshold to be the 30th percentile value in the data. For either the monotonic constraint or the combined constraint, the `--proportion-bytespan` flag can be set to only use ByteSpan to learn a fixed portion of the vocabulary, seeding BPE for the rest:

```bash
uv run cli.py tokenizers create-bytespantokenizer Entropy --threshold-percentile=30 --proportion-bytespan=50
```

All commands can be adjusted to train a multilingual tokenizer on common corpus, e.g:

```bash
uv run cli.py tokenizers create-bytespantokenizer Entropy common-corpus --threshold-percentile=30 --proportion-bytespan=50
```

### Train BPE tokenizer

We implement our own trainer for producing a BPE-style tokenizer. It can be trained without needing the LM predictions:

```bash
uv run cli.py tokenizers create-frequencytokenizer frequency
```

To create a BPE-style tokenizer that supports WordPiece inference, simply train a ByteSpan tokenizer with `proportion-bytespan=0`"

```bash
uv run cli.py tokenizers create-bytespantokenizer Entropy --proportion-bytespan=0
```

### Tokenizer analysis

To run our analysis pipeline, you must first install [morphscore](https://github.com/catherinearnett/morphscore). Use our forked version which fixes a couple of bugs with the pipeline:

```bash
git clone https://github.com/codebyzeb/morphscore.git
```

Then run the analysis command:

```bash
uv run cli.py analysis get-tokenizer-statistics-fineweb
uv run cli.py analysis get-tokenizer-statistics-common-corpus
```

This automatically downloads all tokenizers from our repository and analyze all tokenizers, saving results to `tokenizer_stats_fineweb.csv` for the English tokenizers and `tokenizer_stats_common.csv` for the multilingual tokenizers.

### Train a model using tokenizers

Our model training script requires the data to be pre-tokenized. For a particular tokenizers, run the following:

```bash
uv run cli.py data finewebedu-tokenize --subfolder=fw57M_Surprisal_bytespanP1-0T30_64000
```

This will tokenize all of finewebedu with the chosen tokenizer and upload the IDs as a new subset of our copy of finewebedu. This data must be re-downloaded to prepare it for training:

```bash
uv run cli.py data finewebedu-download fw57M_Surprisal_bytespanP1-0T30_64000 --num-train-rows=16000000
```

For 50k steps, 16000000 rows is more than enough (less than 1 epoch). The training script can then be run, similarly to above:

```bash
uv run scripts/train.py \
    dataset=finewebedu \
    tok_name=fw57M_Surprisal_bytespanP1-0T30_64000 \
    model=fw57M-tied \
    hydra.run.dir=outputs/finewebedu/fw57M_Surprisal_bytespanP1-0T30_64000
```