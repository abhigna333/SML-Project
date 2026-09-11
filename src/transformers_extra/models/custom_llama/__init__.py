from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_custom_llama import CustomLlamaConfig
from .modeling_custom_llama import CustomLlamaForCausalLM


AutoConfig.register("customllama", CustomLlamaConfig)
AutoModelForCausalLM.register(CustomLlamaConfig, CustomLlamaForCausalLM)