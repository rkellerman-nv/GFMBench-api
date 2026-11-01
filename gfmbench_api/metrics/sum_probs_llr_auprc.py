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
import numpy as np
from sklearn.metrics import average_precision_score
from .base_metric import BaseMetric
from ._scores import sum_probs_neg_llr


class SumProbsLLRAUPRC(BaseMetric):
    """
    Log-Likelihood Ratio (LLR) based AUPRC metric for variant effect prediction.
    
    Receives probabilities for variant and reference sequences and ground truth labels.
    Calculates LLR as log(P(variant)) - log(P(reference)) and computes AUPRC.
    
    Lower LLR indicates pathogenic variants, so we use -LLR for the metric calculation.
    """
    
    def __init__(self):
        """Initialize storage for variant/reference probabilities and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._llr_scores_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "sum_probs_llr_auprc"
    
    def _calc_impl(self, variant_probs, reference_probs, gt, epsilon=1e-10):
        """Calculate and store negative LLR scores."""
        neg_llr_scores = sum_probs_neg_llr(variant_probs, reference_probs, epsilon)

        # Store only the LLR scores and labels
        self._llr_scores_list.append(neg_llr_scores)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUPRC from stored LLR scores."""
        if not self._llr_scores_list:
            return None
        
        # Concatenate all batches
        all_llr_scores = np.concatenate(self._llr_scores_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate AUPRC
        return average_precision_score(all_gt, all_llr_scores)

