from transformers import LlamaConfig


class CustomLlamaConfig(LlamaConfig):
    model_type = "customllama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)