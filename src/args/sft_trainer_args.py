import logging
from dataclasses import dataclass, field

from trl import SFTConfig


logger = logging.getLogger(__name__)


@dataclass
class SFTTrainerArguments(SFTConfig):
    """
    Arguments specific to Supervised Fine-Tuning (SFT) training.

    Args:
        TODO: add your arguments here
    """

    save_modeling_code: str | None = field(default=None, metadata={"help": "Path to the modeling code to save."})

    auto_map: dict[str, str] | None = field(default=None, metadata={"help": "Auto map for the model."})
    
