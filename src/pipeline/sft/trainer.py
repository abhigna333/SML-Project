# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import override
from collections import defaultdict
from functools import partial

from torch import nn
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)
from transformers.trainer import logger

from datasets import Dataset, IterableDataset, load_from_disk
from src.args import SFTTrainerArguments
from trl import SFTTrainer

def _report_patching(module, metric_states):
    def report_metrics(metrics, metric_states):
        for k, v in metrics.items():
            metric_states[k].append(v)

    if isinstance(module, nn.Module):
        module.report_metrics = partial(
            report_metrics, metric_states=metric_states
        )

class CustomSFTTrainer(SFTTrainer):
    r"""Extended SFTTrainer with custom model code management."""

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: SFTTrainerArguments | TrainingArguments | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processed_dataset_dir: str | None = None,
        **kwargs,
    ) -> None:
        
        self._model_metrics = defaultdict(list)
        if model is not None:
            report_patching = partial(_report_patching, metric_states=self._model_metrics)
            model.apply(report_patching)

        logger.info(f"processed_dataset_dir: {processed_dataset_dir}")
        if processed_dataset_dir is not None and os.path.isdir(processed_dataset_dir):
            logger.info(f"Loading train and eval datasets from {processed_dataset_dir}")
            if os.path.exists(os.path.join(processed_dataset_dir, "train_dataset")):
                train_dataset = load_from_disk(os.path.join(processed_dataset_dir, "train_dataset"))
            if os.path.exists(os.path.join(processed_dataset_dir, "eval")):
                eval_dataset = load_from_disk(os.path.join(processed_dataset_dir, "eval_dataset"))

        self.args = args
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)

        # Pass trainer to model for step tracking
        if hasattr(self.model, "trainer"):
            self.model.trainer = self

        self.model.config.auto_map = args.auto_map

        if (
            self.train_dataset is not None
            and processed_dataset_dir
            and not os.path.exists(os.path.join(processed_dataset_dir, "train_dataset"))
        ):
            logger.info(f"Saving train dataset to {processed_dataset_dir}")
            self.train_dataset.save_to_disk(os.path.join(processed_dataset_dir, "train_dataset"))
        if (
            self.eval_dataset is not None
            and processed_dataset_dir
            and not os.path.exists(os.path.join(processed_dataset_dir, "eval_dataset"))
        ):
            logger.info(f"Saving eval dataset to {processed_dataset_dir}")
            self.eval_dataset.save_to_disk(os.path.join(processed_dataset_dir, "eval_dataset"))

        # Log a few samples from the training dataset
        if self.train_dataset is not None:
            logger.info("Logging a few samples from the training dataset:")
            for i, sample in enumerate(self.train_dataset.select(range(min(3, len(self.train_dataset))))):
                logger.info(f"Sample {i}: {sample}")

    def _save(self, output_dir: str | None = None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.args.save_modeling_code and os.path.isdir(self.args.save_modeling_code):
            for file in os.listdir(self.args.save_modeling_code):
                if file.endswith(".py") and file != "__init__.py":
                    src_path = os.path.join(self.args.save_modeling_code, file)
                    dst_path = os.path.join(output_dir, file)
                    with open(src_path, encoding="utf-8") as src, open(dst_path, "w", encoding="utf-8") as dst:
                        dst.write(src.read())

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # return super().compute_loss(model, inputs, *args, **kwargs)

        # Start timing the forward pass
        outputs = super().compute_loss(model, inputs, *args, **kwargs)
        # Update metrics as before
        for k, v in self._model_metrics.items():
            if len(v) > 0:
                self._metrics['train'][k] = [self.accelerator.gather_for_metrics(sum(v) / len(v)).mean().item()]
            else:
                logger.info(f"No metrics for {k}")
        self._model_metrics.clear()
        return outputs
