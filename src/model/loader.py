from transformers import AutoModelForCausalLM, TrainingArguments, AutoConfig
from ..args import ModelArguments
from ..transformers_extra.models.custom_llama.modeling_custom_llama import CustomLlamaForCausalLM
from ..transformers_extra.models.DTRNet_smollm.modeling_llama_DTRNet import LlamaForCausalLM
from ..transformers_extra.models.DTRNet_smollm.modeling_llama_MoD import LlamaForCausalLM as LlamaForCausalLM_MoD
from ..transformers_extra.models.DTRNet_smollm.modeling_llama_DLLM import LlamaForCausalLM as LlamaForCausalLM_DLLM
from ..transformers_extra.models.DTRNet_smollm.configuration_llama import LlamaConfig

def load_model(
    training_args: TrainingArguments | None = None, model_args: ModelArguments = None, model_name_or_path: str = None
):
    """
    High-level model loading function that handles both standard and custom models.
    For standard models, uses AutoModelForCausalLM directly.
    For custom models, delegates to ModelCodeManager.
    """
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args and training_args.gradient_checkpointing else True,
    )
    if not model_args.custom_model:

        if model_args.is_dtrnet:
            config = LlamaConfig.from_pretrained(model_args.model_name_or_path, **model_kwargs)
            config.update({
                "aux_loss_coeff": model_args.aux_loss_coeff,
                "dtrnet_layers": model_args.dtrnet_layers
            })
            model = LlamaForCausalLM(config)
            return model
        elif model_args.is_mod:
            config = LlamaConfig.from_pretrained(model_args.model_name_or_path, **model_kwargs)
            config.update({
                "topk": model_args.topk,
            })
            model = LlamaForCausalLM_MoD(config)
            return model
        elif model_args.is_dllm:
            config = LlamaConfig.from_pretrained(model_args.model_name_or_path, **model_kwargs)
            config.update({
                "aux_loss_coeff": model_args.aux_loss_coeff,
                "dynamic_reserve_initials": model_args.dynamic_reserve_initials,
                "dynamic_active_target": model_args.dynamic_active_target,
            })
            model = LlamaForCausalLM_DLLM(config)
            return model
        else:
            return AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path if model_name_or_path is None else model_name_or_path, **model_kwargs
            )
        
    return CustomLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path if model_name_or_path is None else model_name_or_path, **model_kwargs
    )
