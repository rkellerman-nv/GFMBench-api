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
"""
Variants of the affected-region metrics that use more than the region itself.

- `AffectedRegionWindowMetric`: the window around the affected region is set in BASES, not
  tokens, so tokenizers with different token lengths are compared over the same stretch of DNA.
  The window can be summarised by pooling its embeddings before comparing them ("pooled"), or
  by comparing aligned token embeddings and then summarising the per-token distances: the
  maximum over the unchanged neighbouring tokens ("flankmax", how far the change propagates) or
  the base-weighted mean over the whole window ("mean").
- `AffectedRegionSizeStratifiedAUROC`: AUROC computed within strata of affected-region size, so
  region size (a tokenizer property that can correlate with the label) cannot drive the score.

Both consume the same per-token embeddings as the metrics in affected_region_variant_effect.py
and need BaseGFMModel.get_token_base_offsets (the base-window metrics) and
BaseGFMModel.get_affected_token_spans.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .affected_region_variant_effect import (
    _AffectedRegionVariantEffectMetric,
    _CosineMixin,
    _L2Mixin,
)
from .base_metric import BaseMetric

# Both take [n, hidden] arrays and return [n] scores where higher means more pathogenic.
_DISTANCES = {
    "cosinesim": _CosineMixin._score,
    "l2": _L2Mixin._score,
}
_SCORES = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
}
_SUMMARIES = ("pooled", "flankmax", "mean")


def _num_tokens(offsets):
    """Number of tokens in an offsets row ([0, b0, b0 + b1, ...] padded with -1)."""
    return int(np.count_nonzero(offsets >= 0)) - 1


def _base_window(offsets, start, end, width_bases):
    """
    Tokens overlapping the base window around the token region [start, end).

    The window is the region's base interval widened by `width_bases` on each side and clipped
    to the sequence. Returns (first_token, last_token_exclusive, weights) where weights[i] is
    the number of window bases covered by token first_token + i (a token cut by the window edge
    counts only its overlap), or None if the region does not index tokens of this row.
    """
    n = _num_tokens(offsets)
    if start < 0 or end <= start or end > n:
        return None
    bounds = offsets[: n + 1].astype(np.int64)
    low = max(0, bounds[start] - width_bases)
    high = min(bounds[n], bounds[end] + width_bases)
    first = int(np.searchsorted(bounds, low, side="right")) - 1
    last = int(np.searchsorted(bounds, high, side="left"))
    weights = np.minimum(bounds[first + 1:last + 1], high) - np.maximum(bounds[first:last], low)
    if weights.sum() <= 0:
        return None
    return first, last, weights.astype(float)


def _weighted_mean(embeddings, weights):
    return (weights[:, None] * embeddings).sum(axis=0) / weights.sum()


def _pooled_window(embeddings, offsets, start, end, width_bases):
    """Base-weighted mean embedding over the window, or None if it is not defined."""
    window = _base_window(offsets, start, end, width_bases)
    if window is None or window[1] > embeddings.shape[0]:
        return None
    first, last, weights = window
    return _weighted_mean(embeddings[first:last], weights)


def _paired_token_distances(var_emb, ref_emb, var_offsets, ref_offsets, var_span, ref_span,
                            suffix_starts, width_bases, distance):
    """
    Per-element distances between a variant and its reference over the window, with weights.

    Elements, in order: the unchanged tokens before the affected region, the affected region as
    ONE element (its base-weighted mean embedding on each side, weight = its mean size in bases),
    then the unchanged tokens after it. Flank weights are the variant side's window overlap in
    bases.

    Tokens are paired by identity, not by distance from the region: the common prefix sits at the
    same token indices on both sides, and the common suffix is shifted by
    ref_suffix_start - var_suffix_start. (The spans' ends cannot give that shift: a span is at
    least one token wide, so on the shorter side of an insertion or deletion it also contains the
    first unchanged token.)

    Returns (distances, weights, core_index), or None if the window is not defined; core_index is
    the position of the affected region in the returned arrays.
    """
    var_start, var_end = var_span
    ref_start, ref_end = ref_span
    var_suffix_start, ref_suffix_start = suffix_starts
    var_window = _base_window(var_offsets, var_start, var_end, width_bases)
    ref_window = _base_window(ref_offsets, ref_start, ref_end, width_bases)
    if var_window is None or ref_window is None or var_suffix_start < 0 or ref_suffix_start < 0:
        return None
    var_first, var_last, var_weights = var_window
    ref_first, ref_last, _ = ref_window
    if var_last > var_emb.shape[0] or ref_last > ref_emb.shape[0]:
        return None

    var_core_bases = np.diff(var_offsets[var_start:var_end + 1]).astype(float)
    ref_core_bases = np.diff(ref_offsets[ref_start:ref_end + 1]).astype(float)
    if var_core_bases.sum() <= 0 or ref_core_bases.sum() <= 0:
        return None
    var_core = _weighted_mean(var_emb[var_start:var_end], var_core_bases)
    ref_core = _weighted_mean(ref_emb[ref_start:ref_end], ref_core_bases)

    # Before the region: token v on one side is token v on the other
    left_first = max(var_first, ref_first)
    left_last = max(left_first, min(var_start, ref_start))
    # After the region: variant token v is reference token v + shift
    shift = ref_suffix_start - var_suffix_start
    right_first = max(var_end, ref_end - shift)
    right_last = max(right_first, min(var_last, ref_last - shift))

    var_side = np.concatenate([var_emb[left_first:left_last], var_core[None],
                               var_emb[right_first:right_last]])
    ref_side = np.concatenate([ref_emb[left_first:left_last], ref_core[None],
                               ref_emb[right_first + shift:right_last + shift]])
    weights = np.concatenate([
        var_weights[left_first - var_first:left_last - var_first],
        [(var_core_bases.sum() + ref_core_bases.sum()) / 2.0],
        var_weights[right_first - var_first:right_last - var_first],
    ])
    return _DISTANCES[distance](var_side, ref_side), weights, left_last - left_first


class AffectedRegionWindowMetric(BaseMetric):
    """
    Variant effect score over a window of `window_bases` bases around the affected region.

    Args:
        window_bases: bases of unchanged sequence to include on each side of the affected
            region (0 = the region itself, weighted by base length).
        distance: "cosinesim" (negative cosine similarity) or "l2"; higher = more pathogenic.
        summary: "pooled" - base-weighted mean embedding of the window on each side, then one
            distance; "mean" - distances between aligned token embeddings (the affected region
            counts as one element), then their base-weighted mean; "flankmax" - the largest of
            those distances over the unchanged neighbouring tokens only (the affected region is
            excluded, since it is almost always the most changed element and the maximum over it
            would just repeat the pooled score with window 0). Pairs without any unchanged
            neighbour in the window are skipped for "flankmax".
        score: "auroc" or "auprc".

    Receives (variant_embeddings, reference_embeddings, spans, suffix_starts, variant_token_offsets,
    reference_token_offsets, gt): the k=0 affected token spans and the token indices where the
    common suffix starts on each side (BaseGFMModel.get_affected_suffix_starts; only the flankmax
    and mean summaries use them). Pairs whose region or window is undefined are skipped.
    """

    def __init__(self, window_bases, distance="cosinesim", summary="pooled", score="auroc"):
        if distance not in _DISTANCES or summary not in _SUMMARIES or score not in _SCORES:
            raise ValueError(f"unsupported distance/summary/score: {distance}/{summary}/{score}")
        if window_bases < 0:
            raise ValueError("window_bases must be >= 0")
        self.window_bases = int(window_bases)
        self.distance = distance
        self.summary = summary
        self.score = score
        super().__init__()

    @property
    def name(self):
        summary = "" if self.summary == "pooled" else f"{self.summary}_"
        return f"affected_region_b{self.window_bases}_{summary}{self.distance}_{self.score}"

    def reset(self):
        super().reset()
        self._scores_list = []
        self._gt_list = []

    def _calc_impl(self, variant_embeddings, reference_embeddings, spans, suffix_starts,
                   variant_offsets, reference_offsets, gt):
        spans = np.asarray(spans)
        suffix_starts = np.asarray(suffix_starts)
        gt = np.asarray(gt)
        scores, kept = [], []
        for i in range(len(spans)):
            var_span, ref_span = (int(spans[i, 0]), int(spans[i, 1])), (int(spans[i, 2]), int(spans[i, 3]))
            if var_span[0] < 0 or ref_span[0] < 0:
                continue
            if self.summary == "pooled":
                var_vec = _pooled_window(variant_embeddings[i], variant_offsets[i], *var_span, self.window_bases)
                ref_vec = _pooled_window(reference_embeddings[i], reference_offsets[i], *ref_span, self.window_bases)
                if var_vec is None or ref_vec is None:
                    continue
                score = _DISTANCES[self.distance](var_vec[None], ref_vec[None])[0]
            else:
                paired = _paired_token_distances(
                    variant_embeddings[i], reference_embeddings[i], variant_offsets[i], reference_offsets[i],
                    var_span, ref_span, (int(suffix_starts[i, 0]), int(suffix_starts[i, 1])),
                    self.window_bases, self.distance,
                )
                if paired is None:
                    continue
                distances, weights, core_index = paired
                if self.summary == "flankmax":
                    flank = np.delete(distances, core_index)
                    if flank.size == 0:
                        continue
                    score = flank.max()
                else:
                    score = (weights * distances).sum() / weights.sum()
            scores.append(score)
            kept.append(i)
        if kept:
            self._scores_list.append(np.asarray(scores))
            self._gt_list.append(gt[kept])

    def get_final_results(self):
        if not self._scores_list:
            return None
        return _SCORES[self.score](np.concatenate(self._gt_list), np.concatenate(self._scores_list))


class AffectedRegionSizeStratifiedAUROC(BaseMetric):
    """
    AUROC of the affected-region distance within strata of affected-region size.

    The region size (mean of the variant and reference side, in tokens, at k=0) is a tokenizer
    property that can correlate with the label; when it does, an ordinary AUROC can reflect size
    rather than the embeddings. Here pathogenic and benign variants are only compared within the
    same size stratum, and the per-stratum AUROCs are averaged with weight (positives x negatives),
    i.e. the probability that a pathogenic variant outscores a benign one of the same size
    stratum. Strata with fewer than MIN_PER_CLASS variants of either class are dropped. If all
    variants share one stratum (single-nucleotide SNVs) this equals the plain AUROC.

    Args:
        context_tokens: width of the spans whose embeddings are pooled (as in AffectedRegion*).
        distance: "cosinesim" or "l2".

    Receives (variant_embeddings, reference_embeddings, spans, size_spans, gt): `spans` for the
    pooled region (any k) and `size_spans` the k=0 spans that define the strata.
    """

    # Upper edges of the strata on size = (variant tokens + reference tokens) / 2: 1 | 1.5-2 | 2.5-3 | 3.5-6 | 6.5-12 | >12
    STRATUM_EDGES = (1.25, 2.25, 3.25, 6.25, 12.25)
    MIN_PER_CLASS = 5

    def __init__(self, context_tokens=0, distance="cosinesim"):
        if distance not in _DISTANCES:
            raise ValueError(f"unsupported distance: {distance}")
        self.context_tokens = context_tokens
        self.distance = distance
        super().__init__()

    @property
    def name(self):
        return f"affected_region_k{self.context_tokens}_{self.distance}_sizestrat_auroc"

    def reset(self):
        super().reset()
        self._scores_list = []
        self._gt_list = []
        self._size_list = []

    def _calc_impl(self, variant_embeddings, reference_embeddings, spans, size_spans, gt):
        spans = np.asarray(spans)
        size_spans = np.asarray(size_spans)
        var_pooled, var_valid = _AffectedRegionVariantEffectMetric._pool(variant_embeddings, spans[:, 0], spans[:, 1])
        ref_pooled, ref_valid = _AffectedRegionVariantEffectMetric._pool(reference_embeddings, spans[:, 2], spans[:, 3])
        valid = var_valid & ref_valid & (size_spans[:, 0] >= 0) & (size_spans[:, 2] >= 0)
        if not np.any(valid):
            return
        sizes = ((size_spans[:, 1] - size_spans[:, 0]) + (size_spans[:, 3] - size_spans[:, 2])) / 2.0
        self._scores_list.append(_DISTANCES[self.distance](var_pooled[valid], ref_pooled[valid]))
        self._gt_list.append(np.asarray(gt)[valid])
        self._size_list.append(sizes[valid])

    def get_final_results(self):
        if not self._scores_list:
            return None
        scores = np.concatenate(self._scores_list)
        gt = np.concatenate(self._gt_list)
        strata = np.digitize(np.concatenate(self._size_list), self.STRATUM_EDGES)
        concordant, pairs = 0.0, 0.0
        for stratum in np.unique(strata):
            in_stratum = strata == stratum
            n_pos = int(np.sum(gt[in_stratum] == 1))
            n_neg = int(np.sum(gt[in_stratum] == 0))
            if min(n_pos, n_neg) < self.MIN_PER_CLASS:
                continue
            concordant += roc_auc_score(gt[in_stratum], scores[in_stratum]) * n_pos * n_neg
            pairs += n_pos * n_neg
        return concordant / pairs if pairs else None
