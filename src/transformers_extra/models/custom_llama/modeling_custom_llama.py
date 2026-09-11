from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
)

from .configuration_custom_llama import CustomLlamaConfig


class CustomLlamaForCausalLM(LlamaForCausalLM):
    config_class = CustomLlamaConfig

    def __init__(self, config: CustomLlamaConfig):
        super().__init__(config)
        # Add any custom modifications here