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
from ._scores import snv_neg_cosine_sim_scores


class SNVVariantEffectCosineSimAUPRC(BaseMetric):
    """
    SNV (Single Nucleotide Variant) position variant effect AUPRC metric.
    
    Receives embeddings of all variant and reference samples, per-sequence indices of the SNV position,
    and ground truth labels. Performs AUPRC on the 1-cosine similarity (or negative cosine similarity).
    
    This metric is used for variant effect prediction tasks where we want to measure
    precision-recall trade-off for distinguishing between pathogenic and benign variants.
    
    For tokenizers like BPE where the output position varies per sequence, this metric
    handles per-sequence position arrays.
    """
    
    def __init__(self):
        """Initialize storage for variant/reference embeddings, SNV indices, and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._cosine_sim_scores_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "snv_variant_effect_cosinesim_auprc"
    
    def _calc_impl(self, variant_embeddings, reference_embeddings, variant_snv_indices, reference_snv_indices, gt):
        """Calculate and store negative cosine similarity at SNV position for each sequence.
        
        Args:
            variant_embeddings: [batch_size, seq_len, hidden_dim] embeddings for variant sequences
            reference_embeddings: [batch_size, seq_len, hidden_dim] embeddings for reference sequences
            variant_snv_indices: [batch_size] array of SNV positions in variant sequences
            reference_snv_indices: [batch_size] array of SNV positions in reference sequences
            gt: [batch_size] ground truth labels
        """
        neg_cosine_sim, valid_mask = snv_neg_cosine_sim_scores(
            variant_embeddings, reference_embeddings, variant_snv_indices, reference_snv_indices
        )
        if neg_cosine_sim is None:
            return
        gt = gt[valid_mask]

        # Store only the cosine similarity scores and labels
        self._cosine_sim_scores_list.append(neg_cosine_sim)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUPRC from stored cosine similarity scores at SNV position."""
        if not self._cosine_sim_scores_list:
            return None
        
        # Concatenate all batches
        all_cosine_sim_scores = np.concatenate(self._cosine_sim_scores_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate AUPRC
        return average_precision_score(all_gt, all_cosine_sim_scores)

