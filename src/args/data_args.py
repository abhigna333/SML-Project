import logging
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.

    """

    dataset_name: str | None = field(default=None, metadata={"help": "Dataset name."})

    dataset_config: str | None = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )

    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})

    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})

    dataset_mixer: dict[str, float] | None = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )

    dataset_configs: list[str] | None = field(
        default=None,
        metadata={"help": "list of dataset config names. If given must be the same length as 'dataset_mixer' keys."},
    )

    processed_dataset_dir: str | None = field(
        default=None,
        metadata={"help": "Path of processed dataset."},
    )
