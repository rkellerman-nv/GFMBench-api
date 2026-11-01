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
from sklearn.metrics import roc_auc_score
from .base_metric import BaseMetric
from ._scores import neg_cosine_sim_scores


class SequenceEmbeddingsCosineSimAUROC(BaseMetric):
    """
    Sequence embeddings cosine similarity based AUROC metric.
    
    Receives representative embeddings for variant and reference sequences (e.g., CLS token 
    for BERT, last token for GPT) and ground truth labels. Computes AUROC based on 
    cosine similarity between sequence representatives.
    
    Lower similarity indicates pathogenic variants, so we use negative cosine similarity.
    """
    
    def __init__(self):
        """Initialize storage for variant/reference representations and ground truth labels."""
        super().__init__()
    
    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._cosine_sim_scores_list = []
        self._gt_list = []
    
    @property
    def name(self):
        """Return the key name for results dictionary."""
        return "sequence_embeddings_cosinesim_auroc"
    
    def _calc_impl(self, variant_repr, reference_repr, gt):
        """Calculate and store negative cosine similarity scores."""
        neg_cosine_sim = neg_cosine_sim_scores(variant_repr, reference_repr)

        # Store only the cosine similarity scores and labels
        self._cosine_sim_scores_list.append(neg_cosine_sim)
        self._gt_list.append(gt)
    
    def get_final_results(self):
        """Calculate AUROC from stored cosine similarity scores."""
        if not self._cosine_sim_scores_list:
            return None
        
        # Concatenate all batches
        all_cosine_sim_scores = np.concatenate(self._cosine_sim_scores_list)
        all_gt = np.concatenate(self._gt_list)
        
        # Calculate AUROC
        return roc_auc_score(all_gt, all_cosine_sim_scores)

