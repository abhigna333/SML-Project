import importlib
import inspect
import logging
import os
from typing import Any, dict, type

import torch
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


logger = logging.getLogger(__name__)


class ModelCodeManager:
    """
    Manages saving, loading, and configuration of custom model code.
    This class handles:
    1. Persistence and restoration of custom model architectures
    2. Loading of models with proper configuration
    3. Management of custom model code and configurations
    """

    def __init__(self):
        self.custom_code_dir = "custom_code"
        self.config_filename = "config.py"
        self.modeling_filename = "modeling.py"

    def load_model(
        self,
        model_name_or_path: str,
        custom_model: bool = False,
        auto_map: dict[str, str] | None = None,
        trust_remote_code: bool = False,
        torch_dtype: str | torch.dtype | None = None,
        use_cache: bool = True,
        model_revision: str | None = None,
        attn_implementation: str | None = None,
        config_class: type[PretrainedConfig] | None = None,
        model_class: type[PreTrainedModel] | None = None,
    ) -> PreTrainedModel:
        """
        Load a model with proper configuration.
        Args:
            model_name_or_path: Path or name of the model to load
            custom_model: Whether this is a custom model implementation
            auto_map: Mapping of auto classes to implementation paths
            trust_remote_code: Whether to trust remote code
            torch_dtype: Torch data type for the model
            use_cache: Whether to use KV cache
            model_revision: Model revision to use
            attn_implementation: Attention implementation to use
            config_class: Optional custom config class
            model_class: Optional custom model class

        Returns:
            The loaded model
        """
        model_kwargs = {
            "revision": model_revision,
            "trust_remote_code": trust_remote_code,
            "attn_implementation": attn_implementation,
            "torch_dtype": torch_dtype,
            "use_cache": use_cache,
        }

        if custom_model:
            if auto_map:
                model_kwargs["auto_map"] = auto_map
                model_kwargs["trust_remote_code"] = True

            if config_class and model_class:
                config = config_class.from_pretrained(model_name_or_path, **model_kwargs)
                model = model_class.from_pretrained(model_name_or_path, config=config, **model_kwargs)
            else:
                logger.warning(
                    "Custom model specified but no custom classes provided, falling back to AutoModelForCausalLM"
                )
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        return model

    def save_custom_model_code(
        self,
        output_dir: str,
        model: Any,
        auto_map: dict[str, str] | None = None,
        architectures: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Saves custom model code and configuration.

        Args:
            output_dir: Directory to save the model code
            model: The model instance
            auto_map: Dictionary mapping AutoClass names to implementation paths
            architectures: List of model architecture names

        Returns:
            Dict containing the code configuration
        """
        if not hasattr(model, "config") or not hasattr(model.config, "_name_or_path"):
            logger.warning("Model does not have expected configuration attributes, skipping code saving")
            return {}

        # Create custom code directory
        custom_code_path = os.path.join(output_dir, self.custom_code_dir)
        os.makedirs(custom_code_path, exist_ok=True)

        try:
            # Get model class module
            model_class = model.__class__
            model_module = inspect.getmodule(model_class)

            if model_module is None:
                logger.warning(f"Could not find module for model class {model_class.__name__}")
                return {}

            # Get config class module
            config_class = model.config.__class__
            config_module = inspect.getmodule(config_class)

            if config_module is None:
                logger.warning(f"Could not find module for config class {config_class.__name__}")
                return {}

            # Save modeling code
            modeling_source = inspect.getsource(model_module)
            modeling_path = os.path.join(custom_code_path, self.modeling_filename)
            with open(modeling_path, "w") as f:
                f.write(modeling_source)
            logger.info(f"Saved model code to {modeling_path}")

            # Save config code
            config_source = inspect.getsource(config_module)
            config_path = os.path.join(custom_code_path, self.config_filename)
            with open(config_path, "w") as f:
                f.write(config_source)
            logger.info(f"Saved config code to {config_path}")

            # Create code config
            code_config = {
                "custom_code_path": self.custom_code_dir,
                "auto_map": auto_map or {},
                "architectures": architectures or [model_class.__name__],
                "config_class": config_class.__name__,
                "model_class": model_class.__name__,
                "model_module": model_module.__name__,
                "config_module": config_module.__name__,
            }

            return code_config

        except Exception as e:
            logger.error(f"Error saving custom model code: {str(e)}")
            return {}

    def load_custom_model_code(self, checkpoint_dir: str) -> dict[str, Any]:
        """
        Loads custom model code from checkpoint.

        Args:
            checkpoint_dir: Directory containing the checkpoint

        Returns:
            Dict containing the loaded code configuration and modules
        """
        custom_code_path = os.path.join(checkpoint_dir, self.custom_code_dir)

        if not os.path.exists(custom_code_path):
            logger.warning(f"No custom code directory found at {custom_code_path}")
            return {}

        try:
            # Import custom modules
            modeling_path = os.path.join(custom_code_path, self.modeling_filename)
            spec = importlib.util.spec_from_file_location("custom_modeling", modeling_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not create module spec for {modeling_path}")
                return {}

            custom_modeling = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_modeling)
            logger.info(f"Loaded custom modeling code from {modeling_path}")

            config_path = os.path.join(custom_code_path, self.config_filename)
            spec = importlib.util.spec_from_file_location("custom_config", config_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not create module spec for {config_path}")
                return {}

            custom_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config)
            logger.info(f"Loaded custom config code from {config_path}")

            return {
                "custom_modeling": custom_modeling,
                "custom_config": custom_config,
                "custom_code_path": custom_code_path,
            }

        except Exception as e:
            logger.error(f"Error loading custom model code: {str(e)}")
            return {}
