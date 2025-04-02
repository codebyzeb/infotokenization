import typer
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from commands.configs import BYTE_LEVEL_TOKENIZER_REPO_ID, HF_USERNAME

app = typer.Typer()


@app.command()
def create_bytelevel() -> None:
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
    wrapped_tokenizer.push_to_hub(f"{HF_USERNAME}/{BYTE_LEVEL_TOKENIZER_REPO_ID}")


if __name__ == "__main__":
    app()
