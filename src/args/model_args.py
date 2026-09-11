import logging
from dataclasses import dataclass, field

from trl import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments(ModelConfig):
    """
    Arguments pertaining to which model we are going to fine-tune.

    Args:
        TODO: add your arguments here
    """

    custom_model: bool | None = field(
        default=False, metadata={"help": "Wether it should load using Auto model or the Custom model"}
    )

    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Tookenizer checkpoint."},
    )

    is_dtrnet: bool = field(
        default=False,
        metadata={"help": "Whether to use DTRNet."},
    )

    is_mod: bool = field(
        default=False,
        metadata={"help": "Whether to use MoD."},
    )

    is_dllm: bool = field(
        default=False,
        metadata={"help": "Whether to use DLLM."},
    )
    
    dtrnet_layers:  list[int] | None = field(
        default= None,
        metadata={"help": "List of layers to use DTRNet"},
    )

    aux_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for auxiliary loss."},
    )

    topk: float = field(
        default=0.125,
        metadata={"help": "Top-k value for MoD"},
    )

    dynamic_reserve_initials: int = field(
        default=2,
        metadata={"help": "Initial dynamic reserve for D-LLM."},
    )

    dynamic_active_target: float = field(
        default=0.55,
        metadata={"help": "Target for dynamic active tokens in D-LLM."},
    )


    def __post_init__(self):
        super().__post_init__()
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
