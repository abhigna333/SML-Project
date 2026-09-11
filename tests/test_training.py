import json
import os
import tempfile

from transformers import AutoTokenizer

from src.args import DataArguments, ModelArguments, SFTTrainerArguments
from src.data import get_datasets
from src.model import load_model
from src.pipeline.sft.trainer import CustomSFTTrainer


def test_training_workflow():
    """Test the complete training workflow with save/load/resume functionality."""

    # Setup temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Initial training setup
        auto_map = {
            "AutoModelForCausalLM": "modeling_custom_llama.CustomLlamaForCausalLM",
            "AutoConfig": "configuration_custom_llama.CustomLlamaConfig",
        }
        architectures = ["CustomLlamaForCausalLM"]

        model_args = ModelArguments(
            model_name_or_path="HuggingFaceTB/SmolLM-135M",
            custom_model=True,
            trust_remote_code=True,
            auto_map=auto_map,
        )

        data_args = DataArguments(
            dataset_name="tatsu-lab/alpaca", dataset_train_split="train", dataset_test_split="test"
        )

        training_args = SFTTrainerArguments(
            output_dir=temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            bf16=True,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            evaluation_strategy="steps",
            eval_steps=100,
        )

        # 2. Initialize model and tokenizer
        model = load_model(training_args, model_args)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )

        # 3. Load dataset
        dataset = get_datasets(data_args)

        # 4. Initial training with custom state and code config
        custom_state = {"initial_training": True}
        trainer = CustomSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset[data_args.dataset_train_split],
            eval_dataset=dataset[data_args.dataset_test_split],
            processing_class=tokenizer,
            custom_state=custom_state,
            auto_map=auto_map,
            architectures=architectures,
        )

        # Train for a few steps
        trainer.train()

        # Save checkpoint
        checkpoint_dir = os.path.join(temp_dir, "checkpoint-final")
        trainer.save_model(checkpoint_dir)

        # Verify saved files
        assert os.path.exists(os.path.join(checkpoint_dir, "custom_state.json")), "Custom state not saved"
        assert os.path.exists(os.path.join(checkpoint_dir, "code_config.json")), "Code config not saved"
        assert os.path.exists(os.path.join(checkpoint_dir, "custom_code")), "Custom code directory not created"

        # Verify code config content
        with open(os.path.join(checkpoint_dir, "code_config.json")) as f:
            code_config = json.load(f)
            assert code_config["auto_map"] == auto_map, "Auto map not preserved"
            assert code_config["architectures"] == architectures, "Architectures not preserved"

        # 5. Load checkpoint and resume training
        new_model = load_model(training_args, model_args)
        new_trainer = CustomSFTTrainer(
            model=new_model,
            args=training_args,
            train_dataset=dataset[data_args.dataset_train_split],
            eval_dataset=dataset[data_args.dataset_test_split],
            processing_class=tokenizer,
        )

        # Load checkpoint with custom state and code
        loaded_state = new_trainer.load_checkpoint(checkpoint_dir)
        assert loaded_state["initial_training"], "Custom state was not preserved"
        assert loaded_state["code_config"]["auto_map"] == auto_map, "Auto map was not preserved"
        assert loaded_state["code_config"]["architectures"] == architectures, "Architectures were not preserved"
        assert "custom_code" in loaded_state, "Custom code was not loaded"

        # Continue training
        new_trainer.train()


if __name__ == "__main__":
    test_training_workflow()
