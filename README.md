## Setup

Clone the repo

```bash
git clone https://github.com/codebyzeb/infotokenization.git
```

First, install the [`uv`](https://docs.astral.sh/uv/concepts/projects) environment manager

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


## Run commands via the CLI

We define commands in the `/commands` folder. You can run any available command as follows

```bash
uv run cli.py <command> <subcommand> <args> <options>
```


For example, to create a simple byte-level tokenizer

```bash
# Note: no args or options needed here
uv run cli.py tokenizers create-bytelevel
```