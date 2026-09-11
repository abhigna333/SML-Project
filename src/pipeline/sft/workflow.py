import os

import rootutils
import transformers
import wandb
# from liger_kernel.transformers import apply_liger_kernel_to_llama
from loguru import logger
from transformers import AutoTokenizer
import random
import wandb
import torch
import random
import numpy as np

from trl import (
    TrlParser,
    get_peft_config,
)


rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from src.args import DataArguments, ModelArguments, SFTTrainerArguments  # noqa: E402
from src.data import get_datasets  # noqa: E402
from src.model import load_model  # noqa: E402
from src.pipeline.sft.trainer import CustomSFTTrainer as SFTTrainer  # noqa: E402

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(data_args: DataArguments, training_args: SFTTrainerArguments, model_args: ModelArguments):

    set_seed(42)

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # # Initialize wandb with resume capability using the run name from training args
    # if hasattr(training_args, 'run_name') and training_args.run_name:
    #     wandb.init(name=training_args.run_name, resume="allow")
    # else:
    #     # Fallback to using the experiment name from the config file
    #     experiment_name = os.path.basename(training_args.config).replace('.yaml', '')
    #     wandb.init(name=experiment_name, resume="allow")

    logger.info("Starting SFT training workflow")
    logger.info(f"Model: {model_args}")
    logger.info(f"Dataset: {data_args}")
    logger.info(f"Training arguments: {training_args}")

    ################
    # Model init kwargs & Tokenizer
    ################
    logger.info("Loading model...")
    model = load_model(training_args, model_args)
    model.train()  # Ensure model is in training mode
    # if training_args.use_liger_kernel:
    #     apply_liger_kernel_to_llama(
    #         rope=True, cross_entropy=True, fused_linear_cross_entropy=False, rms_norm=True, swiglu=True, model=model
    #     )
    logger.info("Model loaded successfully")

    # Create tokenizer
    logger.info("Initializing tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    # Set padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer initialized successfully")

    print("@@@@@ Printing Model @@@@@@@")
    print(model)

    ################
    # Dataset
    ################
    logger.info("Loading datasets...")
    if data_args.processed_dataset_dir is not None and os.path.exists(data_args.processed_dataset_dir):
        logger.info(f"Skipping dataset loading as {data_args.processed_dataset_dir} exists")
        dataset = None #load_from_disk(data_args.processed_dataset_dir)
    else:
        dataset = get_datasets(data_args)
    logger.info(
        f"Datasets loaded successfully. Train split: {data_args.dataset_train_split}, Test split: {data_args.dataset_test_split}"
    )

    logger.info(f"Datasets: {dataset}")

    ################
    # Training
    ################
    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[data_args.dataset_train_split] if dataset is not None else None,
        eval_dataset=dataset[data_args.dataset_test_split]
        if training_args.eval_strategy != "no" and dataset is not None
        else None,
        processing_class=tokenizer, #PF-Change: This is the old way of doing it.
        # tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        processed_dataset_dir=data_args.processed_dataset_dir,
    )
    logger.info("Trainer initialized successfully")

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed successfully")

    # Save and push to hub
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info(f"Pushing model to hub with dataset name: {data_args.dataset_name}")
        trainer.push_to_hub(dataset_name=data_args.dataset_name)
        logger.info("Model pushed to hub successfully")

    logger.info("SFT training workflow completed")


def make_parser():
    logger.debug("Creating argument parser")
    dataclass_types = (DataArguments, SFTTrainerArguments, ModelArguments)
    parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    logger.info("Starting SFT training script")
    parser = make_parser()
    args = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    data_args, training_args, model_args, config_remaining_strings = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    if config_remaining_strings:
        logger.error(f"Remaining strings from config: {config_remaining_strings}")
        raise ValueError(f"Remaining strings from config: {config_remaining_strings}")
    main(data_args, training_args, model_args)
