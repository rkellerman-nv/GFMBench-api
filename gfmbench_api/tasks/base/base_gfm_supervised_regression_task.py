# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module does not embed third-party data download URLs.
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfmbench_api.metrics import RegressionPearsonR, RegressionR2
from gfmbench_api.tasks.base.base_gfm_supervised_task import BaseGFMSupervisedTask
from gfmbench_api.utils.binning import get_valid_token_counts, pool_tokens_to_bins


class BaseGFMSupervisedRegressionTask(BaseGFMSupervisedTask):
    """
    Base class for sequence-level and binned regression tasks (e.g. CAGE).

    Dataset format: (sequence, label, conditional_input) tuples
    Model inference: infer_sequence_to_sequence(sequences, conditional_input) and, for binned
    tasks, sequence_pos_to_prob_pos(sequences, pos). There is no regression-specific model
    method. A model that owns a trained regression head (e.g. a fine-tuning wrapper) returns
    the head's outputs in the slots of infer_sequence_to_sequence:
        - sequence_representative: [batch_size, num_labels], used by sequence-level tasks
        - sequence_embeddings: [batch_size, num_tokens, num_labels] per-token head outputs,
          used by binned tasks. The task pools them into bins (see
          gfmbench_api.utils.binning.pool_tokens_to_bins), using sequence_pos_to_prob_pos
          to find each sequence's number of valid tokens.
    A model signals that it owns a regression head with a truthy ``has_regression_head``
    attribute. A model without it (e.g. a raw backbone returning hidden embeddings) is not
    queried and its metrics are None. A model that declares a head but returns outputs
    whose width differs from num_labels raises ValueError.

    Not supported by this contract: heads applied after pooling (e.g. an MLP head on pooled
    bin embeddings), pooling by bp overlap, and full fine-tuning of the backbone.

    The bin size determines the label and prediction layout:
        - get_bin_size_bp() is None: sequence-level, [batch_size, num_labels] -
          one value per label per sequence (reported as num_bins = 1)
        - get_bin_size_bp() is an int: binned, [batch_size, num_bins, num_labels] -
          one value per label per bin, where
          num_bins = max_sequence_length // bin_size_bp (must divide evenly and
          be at least 2)

    Evaluation reports macro Pearson r and macro R^2, computed per label after pooling
    all leading dimensions (samples, and bins for binned tasks).

    Subclasses must implement:
        - get_bin_size_bp(): Return the bin size in bp for binned outputs, or None
          for sequence-level outputs
        - _get_num_labels(): Return number of regression outputs per sequence or bin
        - _create_datasets(): Return train, validation, test datasets
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema for conditional inputs or None
    """

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for regression tasks."""
        return {
            "has_finetuning_data": True,
            "has_validation_data": self.validation_dataset is not None,
            "num_labels": self._validate_num_labels(),
            "task_type": "regression",
            "num_bins": self._get_num_bins(),
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame(),
        }

    def _get_num_bins(self) -> int:
        """Return the number of output bins per sequence (1 for sequence-level tasks).

        Derived as ``max_sequence_length // get_bin_size_bp()``. Raises if the bin
        size is invalid, the sequence length is not an exact multiple of it, or the
        result is fewer than 2 bins (a single bin is not a binned output).
        """
        bin_size_bp = self.get_bin_size_bp()
        if bin_size_bp is None:
            return 1
        if (
            isinstance(bin_size_bp, bool)
            or not isinstance(bin_size_bp, int)
            or bin_size_bp <= 0
        ):
            raise ValueError(
                f"get_bin_size_bp() must return a positive int or None, got {bin_size_bp!r}"
            )
        if self.max_sequence_length % bin_size_bp != 0:
            raise ValueError(
                f"max_sequence_length ({self.max_sequence_length}) must be a multiple of "
                f"bin_size_bp ({bin_size_bp})"
            )
        num_bins = self.max_sequence_length // bin_size_bp
        if num_bins < 2:
            raise ValueError(
                f"Binned regression requires at least 2 bins, but max_sequence_length "
                f"({self.max_sequence_length}) / bin_size_bp ({bin_size_bp}) gives {num_bins}"
            )
        return num_bins

    def _predict_batch(
        self,
        model: Any,
        sequences: Any,
        conditional_input: Any,
        num_bins: int,
        num_labels: int,
    ) -> Optional[np.ndarray]:
        """Return regression predictions for a batch, or None if the model has no regression head.

        A model has a regression head only if it sets a truthy ``has_regression_head``
        attribute; other models are skipped before any inference, token mapping or pooling.

        Shape: [batch_size, num_labels] for sequence-level tasks,
        [batch_size, num_bins, num_labels] for binned tasks.
        """
        if not getattr(model, "has_regression_head", False):
            return None
        _, token_outputs, sequence_output = self._safe_model_call(
            model, "infer_sequence_to_sequence", sequences, conditional_input, num_outputs=3
        )
        if num_bins == 1:
            preds = sequence_output
        elif token_outputs is None:
            preds = None
        else:
            token_outputs = torch.as_tensor(np.asarray(token_outputs), dtype=torch.float32)
            if token_outputs.ndim != 3:
                raise ValueError(
                    "Expected per-token regression outputs with 3 dimensions "
                    f"[batch, tokens, labels], but got shape {tuple(token_outputs.shape)}"
                )
            if token_outputs.shape[-1] != num_labels:
                raise ValueError(
                    f"Expected {num_labels} labels, but got {token_outputs.shape[-1]}"
                )
            valid_counts = get_valid_token_counts(model, list(sequences), token_outputs.shape[1])
            preds = pool_tokens_to_bins(token_outputs, valid_counts, num_bins).numpy()
        if preds is None:
            return None
        preds = np.asarray(preds)
        if preds.shape[-1] != num_labels:
            raise ValueError(f"Expected {num_labels} labels, but got {preds.shape[-1]}")
        return preds

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """
        Evaluate the model on the given dataset.

        Args:
            model: Model instance to evaluate (must implement infer_sequence_to_sequence
                and sequence_pos_to_prob_pos; see the class docstring for the output contract)
            dataset: The dataset to evaluate on.

        Returns:
            dict: Scores with metric names as keys:
                - 'regression_pearsonr_macro': Pearson r, averaged over labels
                - 'regression_r2_macro': R^2 score, averaged over labels
        """
        # Create dataloader from dataset
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        # Initialize metric classes (both handle sequence-level and binned predictions)
        metrics = [RegressionPearsonR(), RegressionR2()]
        num_bins = self._get_num_bins()
        num_labels = self._validate_num_labels()
        is_binned = num_bins > 1
        expected_ndim = 3 if is_binned else 2
        layout = "binned" if is_binned else "sequence"

        for sequences, labels, conditional_input in tqdm(data_loader, desc="Evaluating"):
            # Shape: [batch_size, num_labels] for sequence-level tasks,
            # [batch_size, num_bins, num_labels] for binned tasks
            preds = self._predict_batch(model, sequences, conditional_input, num_bins, num_labels)
            labels_np = (
                labels.detach().cpu().numpy()
                if hasattr(labels, "detach")
                else np.asarray(labels)
            )

            # Verify model output is valid
            if preds is not None:
                # Verify that the predictions match the expected layout, the target
                # shape, and the number of labels
                if preds.ndim != expected_ndim:
                    raise ValueError(
                        f"Expected {layout} regression predictions with "
                        f"{expected_ndim} dimensions, but got shape {preds.shape}"
                    )
                if preds.shape != labels_np.shape:
                    raise ValueError(
                        f"Prediction shape {preds.shape} does not match target shape "
                        f"{labels_np.shape}"
                    )
                if is_binned and preds.shape[1] != num_bins:
                    raise ValueError(
                        f"Expected {num_bins} bins, but got {preds.shape[1]}"
                    )

            # Calculate intermediate values for each metric
            for metric in metrics:
                metric.calc(preds, labels_np)

        # Get final results dynamically using metric.name as the result key
        return {metric.name: metric.get_final_results() for metric in metrics}

    @abstractmethod
    def get_bin_size_bp(self) -> Optional[int]:
        """Return a positive bin width in bp, or None for sequence-level outputs.

        Binned tasks must use a fixed effective ``max_sequence_length`` that is
        an exact multiple of this width and yields at least two bins. Their
        datasets and predictions must have shape ``[batch, num_bins, num_labels]``.
        Sequence-level outputs have shape ``[batch, num_labels]`` and report
        ``num_bins = 1`` without a bin dimension.
        """
        pass
