from importlib.util import find_spec

from lightning import seed_everything
from transformers import PreTrainedTokenizerFast
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def get_model(name: str, tok: PreTrainedTokenizerFast, seed: int) -> tuple[LlamaForCausalLM, PretrainedConfig]:
    # Check if flash attention is available, otherwise use sdpa

    attn_implementation = "flash_attention_2" if find_spec("flash_attn") is not None else "sdpa"

    kwargs = {
        "vocab_size": tok.vocab_size,
        "bos_token_id": tok.bos_token_id,  # type: ignore
        "eos_token_id": tok.eos_token_id,  # type: ignore
        "pad_token_id": tok.pad_token_id,  # type: ignore
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "max_position_embeddings": 2048,
        "_attn_implementation": attn_implementation,
    }
    seed_everything(seed)

    if name.startswith("fw57M"):
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=24,
            num_key_value_heads=24,
            num_hidden_layers=6,
            tie_word_embeddings="tied" in name,
            initializer_range=0.02,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000.0,
            **kwargs,
        )

    # elif name == "fw57M-tied":
    #     # https://huggingface.co/HuggingFaceTB/SmolLM-360M/blob/main/config.json
    #     config = LlamaConfig(
    #         model_type="llama",
    #         hidden_act="silu",
    #         hidden_size=1024,
    #         intermediate_size=2560,
    #         num_attention_heads=15,
    #         num_key_value_heads=5,
    #         num_hidden_layers=32,
    #         tie_word_embeddings=True,
    #         initializer_range=0.02,
    #         attention_bias=False,
    #         attention_dropout=0.0,
    #         mlp_bias=False,
    #         pretraining_tp=1,
    #         rms_norm_eps=1e-05,
    #         rope_scaling=None,
    #         rope_theta=10000.0,
    #         **kwargs,
    #     )

    # elif name == "fw850M":
    #     config = LlamaConfig(
    #         model_type="llama",
    #         hidden_act="silu",
    #         hidden_size=1536,
    #         intermediate_size=6144,
    #         num_attention_heads=32,
    #         num_key_value_heads=4,
    #         num_hidden_layers=24,
    #         tie_word_embeddings=False,
    #         initializer_range=0.02,
    #         attention_bias=False,
    #         attention_dropout=0.0,
    #         mlp_bias=False,
    #         pretraining_tp=1,
    #         rms_norm_eps=1e-05,
    #         rope_scaling=None,
    #         rope_theta=10000.0,
    #         **kwargs,
    #     )

    else:
        raise ValueError

    model = LlamaForCausalLM(config)

    return model, config
