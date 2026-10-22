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

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from gfmbench_api.metrics import (
    AffectedRegionCosineSimAUROC,
    AffectedRegionL2AUROC,
    AffectedRegionSizeStratifiedAUROC,
    AffectedRegionWindowMetric,
)
from gfmbench_api.metrics.affected_region_windows import (
    _base_window,
    _paired_token_distances,
    _pooled_window,
)
from gfmbench_api.utils.token_diff import padded_token_base_offsets

# Token lengths [2, 3, 1, 4] bases -> token starts 0, 2, 5, 6 and 10 bases covered
OFFSETS = np.array([0, 2, 5, 6, 10], dtype=np.int32)


def test_padded_offsets():
    offsets = padded_token_base_offsets([[1, 2, 1], [3]])
    assert offsets.tolist() == [[0, 1, 3, 4], [0, 3, -1, -1]]
    assert offsets.dtype == np.int32
    assert padded_token_base_offsets([[]]).tolist() == [[0]]


def test_window_weights_count_overlap_in_bases():
    first, last, weights = _base_window(OFFSETS, 1, 2, 0)
    assert (first, last, weights.tolist()) == (1, 2, [3.0])
    # one base each side: bases [1, 6) -> 1 base of token 0, all of token 1, 1 base of token 2
    first, last, weights = _base_window(OFFSETS, 1, 2, 1)
    assert (first, last, weights.tolist()) == (0, 3, [1.0, 3.0, 1.0])


def test_window_is_clipped_to_the_sequence():
    first, last, weights = _base_window(OFFSETS, 0, 1, 5)
    assert (first, last, weights.tolist()) == (0, 4, [2.0, 3.0, 1.0, 1.0])
    first, last, weights = _base_window(OFFSETS, 3, 4, 100)
    assert (first, last) == (0, 4)
    assert weights.sum() == 10.0  # the whole sequence, no more


def test_window_rejects_regions_outside_the_row():
    assert _base_window(OFFSETS, -1, 2, 1) is None
    assert _base_window(OFFSETS, 2, 2, 1) is None
    assert _base_window(OFFSETS, 3, 5, 1) is None  # token 4 does not exist


def test_pooled_window_is_base_weighted_mean():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    pooled = _pooled_window(embeddings, OFFSETS, 1, 2, 1)
    np.testing.assert_allclose(pooled, (1 * embeddings[0] + 3 * embeddings[1] + 1 * embeddings[2]) / 5)


def test_pooled_window_ignores_padding_beyond_the_row():
    offsets = np.array([0, 1, 2, -1, -1], dtype=np.int32)  # 2 tokens, padded
    embeddings = np.arange(8, dtype=float).reshape(4, 2)
    np.testing.assert_allclose(_pooled_window(embeddings, offsets, 0, 1, 10), embeddings[:2].mean(axis=0))


def test_single_nucleotide_base_window_equals_token_context():
    """With one base per token, a window of w bases is exactly w context tokens (incl. clipping)."""
    rng = np.random.default_rng(0)
    length, hidden = 12, 5
    offsets = np.tile(np.arange(length + 1, dtype=np.int32), (6, 1))
    positions = np.array([0, 1, 5, 6, 10, 11])
    labels = np.array([0, 1, 0, 1, 0, 1])
    variant = rng.normal(size=(6, length, hidden))
    reference = rng.normal(size=(6, length, hidden))
    spans0 = np.stack([positions, positions + 1, positions, positions + 1], axis=1)
    for width in (0, 1, 2, 4, 8):
        low, high = np.maximum(0, positions - width), np.minimum(length, positions + 1 + width)
        spans_k = np.stack([low, high, low, high], axis=1)
        for distance, token_cls in (("l2", AffectedRegionL2AUROC), ("cosinesim", AffectedRegionCosineSimAUROC)):
            by_tokens = token_cls(context_tokens=width)
            by_tokens.calc(variant, reference, spans_k, labels)
            by_bases = AffectedRegionWindowMetric(width, distance, "pooled", "auroc")
            by_bases.calc(variant, reference, spans0, np.stack([positions + 1, positions + 1], axis=1), offsets, offsets, labels)
            np.testing.assert_allclose(by_bases._scores_list[0], by_tokens._scores_list[0])
            assert by_bases.get_final_results() == pytest.approx(by_tokens.get_final_results())


def test_paired_distances_with_resegmented_region():
    # variant: tokens of 1,1,2,1,1 bases, region = the 2-base token; reference: six 1-base tokens, region = tokens 2 and 3
    var_offsets = np.array([0, 1, 2, 4, 5, 6])
    ref_offsets = np.array([0, 1, 2, 3, 4, 5, 6])
    var_emb = np.zeros((5, 2))
    ref_emb = np.zeros((6, 2))
    var_emb[2] = [3, 0]
    ref_emb[2] = ref_emb[3] = [1, 0]   # reference region mean is [1, 0] -> core distance 2
    var_emb[3] = [0, 4]
    ref_emb[4] = [0, 1]                # right flank distance 3
    distances, weights, core_index = _paired_token_distances(
        var_emb, ref_emb, var_offsets, ref_offsets, (2, 3), (2, 4), (3, 4), 1, "l2"
    )
    assert core_index == 1
    np.testing.assert_allclose(distances, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(weights, [1.0, 2.0, 1.0])  # core weight = mean region size in bases (2 and 2)

    for summary, expected in (("flankmax", 3.0), ("mean", 7 / 4)):
        metric = AffectedRegionWindowMetric(1, "l2", summary, "auroc")
        metric.reset()
        metric._scores_list = []
        # score of one pair, through the metric's own code path
        batch_var = var_emb[None]
        batch_ref = ref_emb[None]
        metric._calc_impl(batch_var, batch_ref, np.array([[2, 3, 2, 4]]), np.array([[3, 4]]),
                          var_offsets[None], ref_offsets[None], np.array([1]))
        assert metric._scores_list[0][0] == pytest.approx(expected)


def test_paired_distances_at_the_sequence_edge_have_no_flank_on_that_side():
    offsets = np.arange(5)
    emb_var = np.zeros((4, 2))
    emb_ref = np.zeros((4, 2))
    emb_var[0] = [5, 0]
    distances, weights, core_index = _paired_token_distances(emb_var, emb_ref, offsets, offsets, (0, 1), (0, 1), (1, 1), 2, "l2")
    assert core_index == 0
    # no tokens before the region; two after it
    np.testing.assert_allclose(distances, [5.0, 0.0, 0.0])
    np.testing.assert_allclose(weights, [1.0, 1.0, 1.0])


def _identity_embeddings(ids):
    """One row per token whose value is the token id, so equal tokens have distance 0."""
    return np.array([[float(i), 0.0] for i in ids])


def test_deletion_pairs_flank_tokens_by_identity_not_by_span_end():
    # reference a b c d e f; the variant deletes c. Both spans are one token wide, but the
    # variant's span is the unchanged token d (the minimum-width rule), so the flanks must
    # be paired through the suffix shift, not through the span ends.
    ref_ids = [10, 11, 12, 13, 14, 15]
    var_ids = [10, 11, 13, 14, 15]
    var_emb, ref_emb = _identity_embeddings(var_ids), _identity_embeddings(ref_ids)
    var_offsets, ref_offsets = np.arange(len(var_ids) + 1), np.arange(len(ref_ids) + 1)
    distances, weights, core_index = _paired_token_distances(
        var_emb, ref_emb, var_offsets, ref_offsets, (2, 3), (2, 3), (2, 3), 10, "l2"
    )
    assert core_index == 2
    # left flank 10, 11 | core: variant d (13) against reference c (12) | right flank 14, 15 against 14, 15
    np.testing.assert_allclose(distances, [0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(weights, np.ones(5))


def test_insertion_pairs_flank_tokens_by_identity_not_by_span_end():
    ref_ids = [10, 11, 13, 14, 15]
    var_ids = [10, 11, 12, 13, 14, 15]   # inserts 12; reference span is the unchanged token 13
    var_emb, ref_emb = _identity_embeddings(var_ids), _identity_embeddings(ref_ids)
    var_offsets, ref_offsets = np.arange(len(var_ids) + 1), np.arange(len(ref_ids) + 1)
    distances, _, _ = _paired_token_distances(
        var_emb, ref_emb, var_offsets, ref_offsets, (2, 3), (2, 3), (3, 2), 10, "l2"
    )
    np.testing.assert_allclose(distances, [0.0, 0.0, 1.0, 0.0, 0.0])


def test_unequal_token_counts_with_resegmentation_keep_flanks_paired():
    # variant re-segments two reference tokens (12, 13) into three (20, 21, 22), then re-converges
    ref_ids = [10, 11, 12, 13, 14, 15]
    var_ids = [10, 11, 20, 21, 22, 14, 15]
    var_emb, ref_emb = _identity_embeddings(var_ids), _identity_embeddings(ref_ids)
    var_offsets, ref_offsets = np.arange(len(var_ids) + 1), np.arange(len(ref_ids) + 1)
    distances, weights, _ = _paired_token_distances(
        var_emb, ref_emb, var_offsets, ref_offsets, (2, 5), (2, 4), (5, 4), 10, "l2"
    )
    # core: mean(20, 21, 22) = 21 vs mean(12, 13) = 12.5
    np.testing.assert_allclose(distances, [0.0, 0.0, 8.5, 0.0, 0.0])
    np.testing.assert_allclose(weights, [1.0, 1.0, 2.5, 1.0, 1.0])


def _score_one_pair(summary, width, var_ids, ref_ids, var_span, ref_span, suffix_starts):
    var_emb, ref_emb = _identity_embeddings(var_ids), _identity_embeddings(ref_ids)
    metric = AffectedRegionWindowMetric(width, "l2", summary, "auroc")
    metric._calc_impl(var_emb[None], ref_emb[None], np.array([[*var_span, *ref_span]]), np.array([suffix_starts]),
                      np.arange(len(var_ids) + 1)[None], np.arange(len(ref_ids) + 1)[None], np.array([1]))
    return metric._scores_list[0][0] if metric._scores_list else None


def test_flankmax_looks_only_at_unchanged_neighbours():
    ref_ids = [10, 11, 12, 13, 14, 15]
    var_ids = [10, 11, 13, 14, 15]
    args = (var_ids, ref_ids, (2, 3), (2, 3), (2, 3))
    # the region differs by 1, every unchanged neighbour is identical: the mean sees the region, flankmax does not
    assert _score_one_pair("flankmax", 10, *args) == 0.0
    assert _score_one_pair("mean", 10, *args) == pytest.approx(1 / 5)
    # a window of 0 bases has no neighbours: flankmax is undefined and the pair is skipped
    assert _score_one_pair("flankmax", 0, *args) is None
    assert _score_one_pair("mean", 0, *args) == pytest.approx(1.0)
    # a perturbed neighbour is what flankmax reports
    var_emb = _identity_embeddings(var_ids)
    var_emb[4, 0] += 7.0   # right flank token 15
    metric = AffectedRegionWindowMetric(10, "l2", "flankmax", "auroc")
    metric._calc_impl(var_emb[None], _identity_embeddings(ref_ids)[None], np.array([[2, 3, 2, 3]]), np.array([[2, 3]]),
                      np.arange(6)[None], np.arange(7)[None], np.array([1]))
    assert metric._scores_list[0][0] == pytest.approx(7.0)


def test_window_metric_skips_invalid_pairs_and_names():
    metric = AffectedRegionWindowMetric(10, "cosinesim", "flankmax", "auprc")
    assert metric.name == "affected_region_b10_flankmax_cosinesim_auprc"
    assert AffectedRegionWindowMetric(0).name == "affected_region_b0_cosinesim_auroc"
    assert AffectedRegionWindowMetric(5, "l2", "pooled", "auprc").name == "affected_region_b5_l2_auprc"
    with pytest.raises(ValueError):
        AffectedRegionWindowMetric(5, "manhattan")
    with pytest.raises(ValueError):
        AffectedRegionWindowMetric(-1)

    offsets = np.tile(np.arange(7), (3, 1))
    emb = np.zeros((3, 6, 2))
    var = emb.copy()
    var[1, 2] = 4.0
    var[2, 3] = 1.0
    spans = np.array([[-1, -1, -1, -1], [2, 3, 2, 3], [3, 4, 3, 4]])
    metric = AffectedRegionWindowMetric(0, "l2", "pooled", "auroc")
    metric.calc(var, emb, spans, np.zeros((3, 2), dtype=int), offsets, offsets, np.array([1, 1, 0]))
    assert len(metric._gt_list[0]) == 2  # the invalid first pair was dropped
    assert metric.get_final_results() == 1.0


def test_window_metric_returns_none_without_offsets_or_data():
    metric = AffectedRegionWindowMetric(5)
    emb = np.zeros((1, 3, 2))
    metric.calc(emb, emb, np.array([[0, 1, 0, 1]]), np.array([[1, 1]]), None, None, np.array([1]))
    assert metric.get_final_results() is None


def _stratified_batch(sizes, distances, labels, tokens=4):
    n = len(sizes)
    variant = np.zeros((n, tokens, 2))
    reference = np.zeros((n, tokens, 2))
    for i, (size, distance) in enumerate(zip(sizes, distances)):
        variant[i, :size, 0] = distance
    spans = np.array([[0, s, 0, s] for s in sizes])
    return variant, reference, spans, np.asarray(labels)


def test_size_stratified_auroc_removes_size_confound():
    # larger regions are both more often pathogenic and farther from the reference;
    # inside a size stratum the distance carries no label information
    sizes = [1] * 36 + [3] * 36
    labels = [0] * 30 + [1] * 6 + [0] * 6 + [1] * 30
    distances = [1.0] * 36 + [5.0] * 36
    variant, reference, spans, gt = _stratified_batch(sizes, distances, labels)

    plain = AffectedRegionL2AUROC(context_tokens=0)
    plain.calc(variant, reference, spans, gt)
    assert plain.get_final_results() > 0.7

    stratified = AffectedRegionSizeStratifiedAUROC(context_tokens=0, distance="l2")
    stratified.calc(variant, reference, spans, spans, gt)
    assert stratified.name == "affected_region_k0_l2_sizestrat_auroc"
    assert stratified.get_final_results() == pytest.approx(0.5)


def test_size_stratified_auroc_keeps_within_stratum_signal():
    sizes = [1] * 36 + [3] * 36
    labels = [0] * 30 + [1] * 6 + [0] * 6 + [1] * 30
    base = [1.0] * 36 + [5.0] * 36
    distances = [d + 0.5 * y for d, y in zip(base, labels)]
    variant, reference, spans, gt = _stratified_batch(sizes, distances, labels)
    stratified = AffectedRegionSizeStratifiedAUROC(context_tokens=0, distance="l2")
    stratified.calc(variant, reference, spans, spans, gt)
    assert stratified.get_final_results() == pytest.approx(1.0)


def test_size_stratified_equals_plain_auroc_for_one_stratum():
    rng = np.random.default_rng(1)
    n = 60
    labels = rng.integers(0, 2, n)
    distances = rng.normal(size=n) + labels
    variant, reference, spans, gt = _stratified_batch([1] * n, np.abs(distances), labels)
    plain = AffectedRegionL2AUROC(context_tokens=0)
    plain.calc(variant, reference, spans, gt)
    stratified = AffectedRegionSizeStratifiedAUROC(context_tokens=0, distance="l2")
    stratified.calc(variant, reference, spans, spans, gt)
    assert stratified.get_final_results() == pytest.approx(plain.get_final_results())


def test_size_stratified_drops_small_strata_and_returns_none_if_none_left():
    # a stratum with fewer than MIN_PER_CLASS of one class is dropped
    sizes = [1] * 20 + [3] * 7
    labels = [0] * 10 + [1] * 10 + [0] * 6 + [1] * 1
    distances = [1.0] * 10 + [2.0] * 10 + [9.0] * 7
    variant, reference, spans, gt = _stratified_batch(sizes, distances, labels)
    metric = AffectedRegionSizeStratifiedAUROC(context_tokens=0, distance="l2")
    metric.calc(variant, reference, spans, spans, gt)
    assert metric.get_final_results() == pytest.approx(1.0)  # only the size-1 stratum counts

    variant, reference, spans, gt = _stratified_batch([1] * 4, [1.0] * 4, [0, 1, 0, 1])
    only_small = AffectedRegionSizeStratifiedAUROC(context_tokens=0, distance="l2")
    only_small.calc(variant, reference, spans, spans, gt)
    assert only_small.get_final_results() is None


def test_size_stratified_skips_invalid_spans():
    variant, reference, spans, gt = _stratified_batch([1] * 12, [1.0] * 6 + [2.0] * 6, [0] * 6 + [1] * 6)
    spans = spans.copy()
    spans[0] = -1
    metric = AffectedRegionSizeStratifiedAUROC(context_tokens=0, distance="cosinesim")
    metric.calc(variant, reference, spans, spans, gt)
    assert len(metric._gt_list[0]) == 11
