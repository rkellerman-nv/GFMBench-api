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
from sklearn.metrics import average_precision_score, roc_auc_score
from .base_metric import BaseMetric


class _AffectedRegionVariantEffectMetric(BaseMetric):
    """
    Shared logic for variant effect metrics over the tokens a variant affects.

    Receives per-token embeddings of variant and reference sequences and, per sample,
    the token span affected by the variant on each side (see
    BaseGFMModel.get_affected_token_spans). Each span is mean-pooled to one vector and
    the variant/reference vectors are compared. Because the spans are per-sequence, the
    variant and reference may differ in token count (BPE and similar tokenizers).

    Subclasses define the distance (`_score`), the metric (`_final`) and `name`.
    Higher score means more pathogenic.
    """

    def __init__(self, context_tokens: int = 0):
        """
        Args:
            context_tokens: context width used to compute the spans; only used to name the metric.
        """
        self.context_tokens = context_tokens
        super().__init__()

    def reset(self):
        """Reset internal storage."""
        super().reset()
        self._scores_list = []
        self._gt_list = []

    def _score(self, variant_pooled, reference_pooled):
        raise NotImplementedError

    def _final(self, gt, scores):
        raise NotImplementedError

    @staticmethod
    def _pool(embeddings, starts, ends):
        """Mean-pool embeddings[i, starts[i]:ends[i]] per sample; returns (pooled, valid_mask)."""
        batch_size, seq_len, hidden_dim = embeddings.shape
        pooled = np.zeros((batch_size, hidden_dim))
        valid = np.zeros(batch_size, dtype=bool)
        for i in range(batch_size):
            start, end = int(starts[i]), min(int(ends[i]), seq_len)
            if start < 0 or end <= start:
                continue
            pooled[i] = embeddings[i, start:end, :].mean(axis=0)
            valid[i] = True
        return pooled, valid

    def _calc_impl(self, variant_embeddings, reference_embeddings, spans, gt):
        """Pool the affected region of each pair and store the distance score.

        Args:
            variant_embeddings: [batch_size, seq_len, hidden_dim]
            reference_embeddings: [batch_size, seq_len, hidden_dim]
            spans: [batch_size, 4] int array of (var_start, var_end, ref_start, ref_end),
                   half-open token ranges; negative start marks an undetermined span
            gt: [batch_size] ground truth labels
        """
        spans = np.asarray(spans)
        variant_pooled, variant_valid = self._pool(variant_embeddings, spans[:, 0], spans[:, 1])
        reference_pooled, reference_valid = self._pool(reference_embeddings, spans[:, 2], spans[:, 3])
        valid = variant_valid & reference_valid
        if not np.any(valid):
            return

        self._scores_list.append(self._score(variant_pooled[valid], reference_pooled[valid]))
        self._gt_list.append(np.asarray(gt)[valid])

    def get_final_results(self):
        """Calculate the metric from stored scores."""
        if not self._scores_list:
            return None
        return self._final(np.concatenate(self._gt_list), np.concatenate(self._scores_list))


class _CosineMixin:
    @staticmethod
    def _score(variant_pooled, reference_pooled):
        # Negative cosine similarity: less similar -> higher pathogenicity score
        v = variant_pooled / (np.linalg.norm(variant_pooled, axis=1, keepdims=True) + 1e-10)
        r = reference_pooled / (np.linalg.norm(reference_pooled, axis=1, keepdims=True) + 1e-10)
        return -np.sum(v * r, axis=1)


class _L2Mixin:
    @staticmethod
    def _score(variant_pooled, reference_pooled):
        return np.linalg.norm(variant_pooled - reference_pooled, axis=1)


class _AUROCMixin:
    @staticmethod
    def _final(gt, scores):
        return roc_auc_score(gt, scores)


class _AUPRCMixin:
    @staticmethod
    def _final(gt, scores):
        return average_precision_score(gt, scores)


class AffectedRegionCosineSimAUROC(_CosineMixin, _AUROCMixin, _AffectedRegionVariantEffectMetric):
    """AUROC on negative cosine similarity of mean-pooled affected-token embeddings."""

    @property
    def name(self):
        return f"affected_region_k{self.context_tokens}_cosinesim_auroc"


class AffectedRegionCosineSimAUPRC(_CosineMixin, _AUPRCMixin, _AffectedRegionVariantEffectMetric):
    """AUPRC on negative cosine similarity of mean-pooled affected-token embeddings."""

    @property
    def name(self):
        return f"affected_region_k{self.context_tokens}_cosinesim_auprc"


class AffectedRegionL2AUROC(_L2Mixin, _AUROCMixin, _AffectedRegionVariantEffectMetric):
    """AUROC on L2 distance of mean-pooled affected-token embeddings."""

    @property
    def name(self):
        return f"affected_region_k{self.context_tokens}_l2_auroc"


class AffectedRegionL2AUPRC(_L2Mixin, _AUPRCMixin, _AffectedRegionVariantEffectMetric):
    """AUPRC on L2 distance of mean-pooled affected-token embeddings."""

    @property
    def name(self):
        return f"affected_region_k{self.context_tokens}_l2_auprc"


class _AffectedRegionLengthMetric(_AffectedRegionVariantEffectMetric):
    """
    Diagnostics on the size of the affected region, not on embeddings.

    Region size is a tokenizer property (1 token for single-nucleotide, more for BPE or
    Hamiltonian re-segmentation), so it can confound the embedding metrics. Use spans
    computed with no context (k=0) so the length is the raw divergence.
    """

    def __init__(self):
        super().__init__(context_tokens=0)

    @staticmethod
    def _lengths(spans):
        """Per-pair (variant tokens + reference tokens) / 2 for valid spans; returns (lengths, valid)."""
        spans = np.asarray(spans)
        valid = (spans[:, 0] >= 0) & (spans[:, 2] >= 0)
        lengths = ((spans[:, 1] - spans[:, 0]) + (spans[:, 3] - spans[:, 2])) / 2.0
        return lengths[valid].astype(float), valid

    def _calc_impl(self, variant_embeddings, reference_embeddings, spans, gt):
        lengths, valid = self._lengths(spans)
        if lengths.size == 0:
            return
        self._scores_list.append(lengths)
        self._gt_list.append(np.asarray(gt)[valid])


class AffectedRegionLengthAUROC(_AffectedRegionLengthMetric):
    """
    AUROC of affected-region token count alone, as a leakage baseline.

    Should be near 0.5 for SNV tasks. A high value means region size correlates with the
    label (e.g. indels), and the embedding-based affected-region scores can look good for
    the wrong reason.
    """

    @property
    def name(self):
        return "affected_region_length_auroc"

    def _final(self, gt, scores):
        return roc_auc_score(gt, scores)


class AffectedRegionMeanLength(_AffectedRegionLengthMetric):
    """Mean affected-region size in tokens (average of variant and reference sides)."""

    @property
    def name(self):
        return "affected_region_mean_tokens"

    def _final(self, gt, scores):
        return float(np.mean(scores))
