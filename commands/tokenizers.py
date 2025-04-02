import shutil
from pathlib import Path

import typer
from huggingface_hub import HfApi
from rich import print
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from commands.configs import BYTELEVEL_TOK_FOLDER, HF_USERNAME, TOK_REPO_ID

app = typer.Typer()


@app.command()
def create_bytelevel() -> None:
    folder_path = Path(TOK_REPO_ID) / BYTELEVEL_TOK_FOLDER

    print(f"💡 Will save the tokenizer locally to to: {folder_path}")
    folder_path.mkdir(parents=True, exist_ok=True)

    print("⚙️ Creating the ByteLevel tokenizer")
    add_prefix_space = True  # Note that we will add a prefix_space to the pre_tokenizer
    PAD_TOKEN = "<|padding|>"
    EOS_TOKEN = "<|endoftext|>"

    # Define the tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space, use_regex=True)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    # "Train", i.e., add the properties that we need
    trainer = trainers.BpeTrainer(
        special_tokens=[PAD_TOKEN, EOS_TOKEN], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator([], trainer=trainer)

    # Load the tokenizer as a transformers-compatible tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        unk_token=None,
        bos_token=None,
        eos_token=EOS_TOKEN,
        add_prefix_space=add_prefix_space,
    )
    wrapped_tokenizer.save_pretrained(str(folder_path))

    repo_id = f"{HF_USERNAME}/{folder_path.parent}"
    print(f"🆙 Uploading the tokenizer to {repo_id} on the HF Hub")

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=folder_path, repo_id=repo_id, path_in_repo=folder_path.name, repo_type="model", revision="main"
    )

    print(f"✅ Successfully created and uploaded the tokenizer to {repo_id}")

    shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == "__main__":
    app()
