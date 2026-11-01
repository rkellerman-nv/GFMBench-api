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

from gfmbench_api.metrics import (
    AffectedContextExtentMetric,
    AffectedContextProfileMetric,
    ContextLikelihoodMeanTokens,
    ContextLikelihoodMetric,
    ForcedSegmentationFraction,
    ForcedSegmentationRegionMetric,
    SegmentationPhaseMetric,
    TokenizerStatisticAUROC,
)
from gfmbench_api.metrics.affected_region_extensions import context_positions, context_profile
from gfmbench_api.tasks.base.base_gfm_zero_shot_task import REGION_GROUPS_ENV, BaseGFMZeroShotTask
from gfmbench_api.utils.token_diff import (
    changed_base_interval,
    padded_token_base_offsets,
    padded_token_ids,
    paired_flank_tokens,
    tokens_covering_bases,
)


# ----------------------------------------------------------------------------- token_diff helpers


def test_padded_token_ids():
    assert padded_token_ids([[5, 6, 7], [9]]).tolist() == [[5, 6, 7], [9, -1, -1]]


def test_changed_base_interval():
    assert changed_base_interval("AAAAAA", "AAGAAA") == (2, 3)
    assert changed_base_interval("AAAAAA", "ACAACA") == (1, 5)
    assert changed_base_interval("aaaa", "AAAA") is None  # case only
    assert changed_base_interval("AAAA", "AAA") is None  # indel
    assert changed_base_interval("AAAA", "AAAA") is None


def test_tokens_covering_bases():
    offsets = np.array([0, 2, 5, 6, 10, -1, -1])  # token lengths 2, 3, 1, 4
    assert tokens_covering_bases(offsets, 3, 4) == (1, 2)
    assert tokens_covering_bases(offsets, 1, 3) == (0, 2)  # straddles the boundary at 2
    assert tokens_covering_bases(offsets, 5, 6) == (2, 3)
    assert tokens_covering_bases(offsets, 9, 12) == (3, 4)  # clipped to the covered bases
    assert tokens_covering_bases(offsets, 10, 12) == (-1, -1)


def _offsets(token_bases):
    return padded_token_base_offsets([token_bases])[0]


def test_paired_flank_tokens_snv_same_tokenization():
    offsets = _offsets([1, 1, 1, 1, 1, 1, 1, 1])  # 8 single-base tokens, SNV at token 3
    left, left_gaps, right, right_gaps, shift = paired_flank_tokens(
        (3, 4), (3, 4), (4, 4), offsets, offsets, max_gap_bases=100
    )
    assert shift == 0
    assert left.tolist() == [2, 1, 0] and right.tolist() == [4, 5, 6, 7]
    assert left_gaps.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert right_gaps.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]


def test_paired_flank_tokens_deletion_shifts_the_right_side():
    # variant lost one single-base token: ref tokens 0..7, variant tokens 0..6, deleted ref token 3
    ref = _offsets([1] * 8)
    var = _offsets([1] * 7)
    # spans have min width 1; the common suffix starts at var 3 / ref 4
    left, _, right, _, shift = paired_flank_tokens((3, 4), (3, 4), (3, 4), var, ref, max_gap_bases=100)
    assert shift == 1
    # variant token 3 is the swallowed unchanged token (ref 4), which the spans cover; the rest pair by shift
    assert left.tolist() == [2, 1, 0]
    assert right.tolist() == [4, 5, 6]


def test_paired_flank_tokens_max_gap_and_undefined():
    offsets = _offsets([3, 3, 3, 3, 3, 3])
    left, _, right, _, _ = paired_flank_tokens((2, 3), (2, 3), (3, 3), offsets, offsets, max_gap_bases=4)
    assert left.tolist() == [1, 0][:len(left)] and len(left) == 2  # gaps 0-3 and 3-6 start below 4
    assert right.tolist() == [3, 4]
    assert paired_flank_tokens((-1, -1), (-1, -1), (-1, -1), offsets, offsets, 10) is None


# ----------------------------------------------------------------------------- profile and metrics


def _embeddings(ids, dim=4):
    """One-hot-ish embedding per token id, so equal ids have zero distance."""
    return np.array([np.eye(dim)[i % dim] * (1 + i // dim) for i in ids], dtype=float)


def test_context_profile_changed_flank_token_is_nonzero_only_where_it_changed():
    offsets = _offsets([1] * 8)
    ref_ids = [0, 1, 2, 3, 0, 1, 2, 3]
    var_ids = list(ref_ids)
    var_ids[3] = 1  # the substituted token
    var_emb, ref_emb = _embeddings(var_ids), _embeddings(ref_ids)
    var_emb[5] = ref_emb[5] + 0.5  # the context change: one downstream token moves
    profile = context_profile(var_emb, ref_emb, offsets, offsets, (3, 4), (3, 4), (4, 4))
    assert profile["core"]["l2"] > 0
    gaps, dist = profile["right"]
    assert gaps.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert dist["l2"][0] == 0 and dist["l2"][2] == 0 and dist["l2"][3] == 0
    assert dist["l2"][1] == pytest.approx(np.linalg.norm([0.5] * 4))
    _, left_dist = profile["left"]
    assert np.allclose(left_dist["l2"], 0)


def test_context_profile_pairs_by_identity_across_an_indel():
    # reference [a b c d e f], variant lost c: [a b d e f]; token d pairs with d, not with e
    ref_emb = _embeddings([0, 1, 2, 3, 0, 1])
    var_emb = np.stack([ref_emb[0], ref_emb[1], ref_emb[3], ref_emb[4], ref_emb[5]])
    var_offsets, ref_offsets = _offsets([1] * 5), _offsets([1] * 6)
    # common suffix starts at var 2 (token d) / ref 3; spans have min width 1
    profile = context_profile(var_emb, ref_emb, var_offsets, ref_offsets, (2, 3), (2, 3), (2, 3))
    _, dist = profile["right"]
    assert np.allclose(dist["l2"], 0)  # all unchanged tokens are identical across the deletion


def _profile(core, right, gaps):
    return {"core": {"cosinesim": core, "l2": core}, "left": (np.zeros((0, 2)), {"cosinesim": np.zeros(0), "l2": np.zeros(0)}),
            "right": (np.asarray(gaps, dtype=float), {"cosinesim": np.asarray(right, float), "l2": np.asarray(right, float)})}


def test_profile_metric_bins_and_auroc():
    gaps = [[0, 4], [4, 12], [12, 30]]
    # pathogenic variants change the first bin more; benign ones do not
    profiles = [_profile(1.0, [0.9, 0.1, 0.0], gaps), _profile(1.0, [0.8, 0.2, 0.0], gaps),
                _profile(1.0, [0.1, 0.1, 0.0], gaps), _profile(1.0, [0.2, 0.1, 0.0], gaps)]
    gt = np.array([1, 1, 0, 0])
    near = AffectedContextProfileMetric("down", 0, 10, "l2", "auroc")
    near.calc(profiles, gt)
    assert near.get_final_results() == 1.0
    far = AffectedContextProfileMetric("down", 12, 30, "l2", "auroc")
    far.calc(profiles, gt)
    assert far.get_final_results() == 0.5  # all zero -> all tied
    mean = AffectedContextProfileMetric("down", 0, 4, "l2", "mean")
    mean.calc(profiles, gt)
    assert mean.get_final_results() == pytest.approx(np.mean([0.9, 0.8, 0.1, 0.2]))
    # bin [0, 10) mixes tokens by overlap: 4 bases of 0.9 and 6 bases of the second token (0.1)
    mixed = AffectedContextProfileMetric("down", 0, 10, "l2", "mean")
    mixed.calc(profiles[:1], gt[:1])
    assert mixed.get_final_results() == pytest.approx((4 * 0.9 + 6 * 0.1) / 10)
    # no token on the up side -> every variant is skipped
    up = AffectedContextProfileMetric("up", 0, 10, "l2", "auroc")
    up.calc(profiles, gt)
    assert up.get_final_results() is None


def test_extent_metric_reach_and_decay():
    gaps = [[0, 4], [4, 12], [12, 30]]
    far_reaching = _profile(1.0, [0.9, 0.8, 0.7], gaps)  # still >= half of the core 30 bases out
    local = _profile(1.0, [0.9, 0.1, 0.0], gaps)
    reach = AffectedContextExtentMetric("down", "reach", "l2")
    reach.calc([far_reaching, far_reaching, local, local], np.array([1, 1, 0, 0]))
    assert reach.get_final_results() == 1.0
    decay = AffectedContextExtentMetric("down", "decay", "l2")
    decay.calc([far_reaching, local], np.array([1, 0]))
    assert decay.get_final_results() == 1.0
    # a region that did not change has no defined reach
    flat = AffectedContextExtentMetric("down", "reach", "l2")
    flat.calc([_profile(0.0, [0.0, 0.0, 0.0], gaps)], np.array([1]))
    assert flat.get_final_results() is None


def test_context_positions_layout():
    offsets = _offsets([2] * 8)  # 8 two-base tokens
    var_pos, ref_pos, bases = context_positions((3, 4), (3, 4), (4, 4), offsets, offsets, max_gap_bases=5, per_side=3)
    # down: tokens 4, 5, 6 start at gaps 0, 2, 4 (<5); up: tokens 2, 1, 0 start (gap_start) 0, 2, 4
    assert var_pos.tolist() == [4, 5, 6, 2, 1, 0]
    assert ref_pos.tolist() == var_pos.tolist()
    assert bases.tolist() == [2] * 6
    var_pos, _, _ = context_positions((3, 4), (3, 4), (4, 4), offsets, offsets, max_gap_bases=3, per_side=3)
    assert var_pos.tolist() == [4, 5, -1, 2, 1, -1]


def test_context_positions_after_a_deletion_use_shifted_reference_tokens():
    var, ref = _offsets([1] * 7), _offsets([1] * 8)
    var_pos, ref_pos, _ = context_positions((3, 4), (3, 4), (3, 4), var, ref, max_gap_bases=10, per_side=2)
    assert var_pos[:2].tolist() == [4, 5] and ref_pos[:2].tolist() == [5, 6]


def test_context_likelihood_metric_sides_and_normalization():
    nan = np.nan
    # slots: down, down, up, up
    delta = np.array([[2.0, 1.0, nan, nan], [0.1, 0.0, 0.0, nan], [nan, nan, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    bases = np.array([[2, 2, -1, -1], [1, 1, 1, -1], [-1, -1, 4, 4], [1, 1, 1, 1]])
    gt = np.array([1, 0, 1, 0])
    both = ContextLikelihoodMetric("both", "sum", "auroc")
    both.calc(delta, bases, gt)
    assert both.get_final_results() == 1.0
    down = ContextLikelihoodMetric("down", "sum", "auroc")
    down.calc(delta, bases, gt)  # variant 2 has no down tokens and is skipped
    assert down.get_final_results() == 1.0
    perbase = ContextLikelihoodMetric("both", "perbase", "auprc")
    perbase.calc(delta, bases, gt)
    assert perbase.get_final_results() == 1.0
    count = ContextLikelihoodMeanTokens()
    count.calc(delta, bases, gt)
    assert count.get_final_results() == pytest.approx((2 + 3 + 2 + 4) / 4)


def test_forced_fraction_and_region_metric():
    forced = np.array([[1, 2, 1, 2], [0, 2, 0, 1], [-1, -1, -1, -1]])
    natural = np.array([[0, 1, 0, 2], [0, 2, 0, 1], [0, 1, 0, 1]])
    valid = ForcedSegmentationFraction("valid")
    valid.calc(forced, natural)
    assert valid.get_final_results() == pytest.approx(2 / 3)
    single = ForcedSegmentationFraction("single_token")
    single.calc(forced, natural)
    assert single.get_final_results() == pytest.approx(0.5)
    resegmented = ForcedSegmentationFraction("resegmented")
    resegmented.calc(forced, natural)
    assert resegmented.get_final_results() == pytest.approx(0.5)  # row 0 differs, row 1 equal

    metric = ForcedSegmentationRegionMetric(0, "l2", "auroc")
    assert metric.name == "forced_region_k0_l2_auroc"
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(6, 5, 4))
    var = ref.copy()
    var[:3, 1] += 5.0  # the first three change at token 1
    spans = np.tile([1, 2, 1, 2], (6, 1))
    metric.calc(var, ref, spans, np.array([1, 1, 1, 0, 0, 0]))
    assert metric.get_final_results() == 1.0
    assert ForcedSegmentationRegionMetric(0, "l2", "auroc").get_final_results() is None


def test_tokenizer_statistic_uses_prior_and_orients_scores(tmp_path):
    counts = np.ones(4096)
    counts[7] = 10_000  # token 7 is common, token 9 rare
    path = tmp_path / "freq.npy"
    np.save(path, counts)
    ids = np.array([[1, 7, 2], [1, 9, 2]])
    ref_ids = np.array([[1, 7, 2], [1, 7, 2]])
    spans = np.array([[1, 2, 1, 2], [1, 2, 1, 2]])
    offsets = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    metric = TokenizerStatisticAUROC("surprisal_delta", str(path))
    metric.calc(ids, ref_ids, spans, offsets, offsets, np.array([0, 1]))
    assert metric.get_final_results() == 1.0  # the variant that introduces the rare token is "pathogenic"
    rare = TokenizerStatisticAUROC("max_alt_surprisal", str(path))
    rare.calc(ids, ref_ids, spans, offsets, offsets, np.array([0, 1]))
    assert rare.get_final_results() == 1.0
    bases = TokenizerStatisticAUROC("region_bases", str(path))
    bases.calc(ids, ref_ids, spans, offsets, offsets, np.array([0, 1]))
    assert bases.get_final_results() == 0.5


def test_tokenizer_statistic_without_prior_counts_reference_tokens():
    ids = np.array([[3, 3, 3], [3, 5, 3]])
    ref_ids = np.array([[3, 3, 3], [3, 3, 3]])
    spans = np.array([[1, 2, 1, 2], [1, 2, 1, 2]])
    offsets = np.array([[0, 1, 2, 3]] * 2)
    metric = TokenizerStatisticAUROC("surprisal_delta")
    metric.calc(ids, ref_ids, spans, offsets, offsets, np.array([0, 1]))
    assert metric.get_final_results() == 1.0  # token 5 never occurs in the references


def test_segmentation_phase_metric_modes():
    rng = np.random.default_rng(1)
    shifts = len(SegmentationPhaseMetric.SHIFTS)
    batch = 8
    gt = np.array([0, 1] * 4)
    ref = [rng.normal(size=(batch, 4, 3)) for _ in range(shifts)]
    var = [r.copy() for r in ref]
    for s in range(shifts):
        var[s][gt == 1, 1] += 3.0 + s  # pathogenic variants move token 1 at every shift
    spans = [np.tile([1, 2, 1, 2], (batch, 1)) for _ in range(shifts)]
    spans[2][0] = [1, 3, 1, 2]  # variant 0 has an unstable region size
    mean = SegmentationPhaseMetric("mean", "l2", "auroc")
    mean.calc(var, ref, spans, gt)
    assert mean.get_final_results() == 1.0
    shift = SegmentationPhaseMetric("shift3", "l2", "auroc")
    shift.calc(var, ref, spans, gt)
    assert shift.get_final_results() == 1.0
    unstable = SegmentationPhaseMetric("unstable_fraction")
    unstable.calc(var, ref, spans, gt)
    assert unstable.get_final_results() == pytest.approx(1 / batch)
    # a variant that is invalid at any shift is dropped from every mode
    spans[1][1] = [-1, -1, -1, -1]
    dropped = SegmentationPhaseMetric("unstable_fraction")
    dropped.calc(var, ref, spans, gt)
    assert dropped.get_final_results() == pytest.approx(1 / (batch - 1))
    with pytest.raises(ValueError):
        SegmentationPhaseMetric("mean").calc(var[:2], ref[:2], spans[:2], gt)


# ----------------------------------------------------------------------------- task wiring


def _task_stub():
    abstract = {name: (lambda self, *a, **k: None) for name in BaseGFMZeroShotTask.__abstractmethods__}
    stub = type("StubTask", (BaseGFMZeroShotTask,), abstract)
    return object.__new__(stub)


def test_region_groups_env(monkeypatch):
    monkeypatch.delenv(REGION_GROUPS_ENV, raising=False)
    assert BaseGFMZeroShotTask._enabled_region_groups() == ["core"]
    monkeypatch.setenv(REGION_GROUPS_ENV, "forced, phase")
    assert BaseGFMZeroShotTask._enabled_region_groups() == ["forced", "phase"]
    monkeypatch.setenv(REGION_GROUPS_ENV, "all")
    assert BaseGFMZeroShotTask._enabled_region_groups() == list(BaseGFMZeroShotTask.REGION_METRIC_GROUPS)
    monkeypatch.setenv(REGION_GROUPS_ENV, "none")
    assert BaseGFMZeroShotTask._enabled_region_groups() == []
    monkeypatch.setenv(REGION_GROUPS_ENV, "forced,typo")
    with pytest.raises(ValueError, match="typo"):
        BaseGFMZeroShotTask._enabled_region_groups()


def test_metric_names_are_unique_and_groups_are_disjoint(monkeypatch):
    task = _task_stub()
    names = {}
    for group in BaseGFMZeroShotTask.REGION_METRIC_GROUPS:
        monkeypatch.setenv(REGION_GROUPS_ENV, group)
        group_names = [metric.name for metric, _ in task._get_affected_region_metrics()]
        assert group_names and len(set(group_names)) == len(group_names)
        names[group] = set(group_names)
    for a in names:
        for b in names:
            if a < b:
                assert not names[a] & names[b], (a, b)
    monkeypatch.setenv(REGION_GROUPS_ENV, "all")
    assert len(task._get_affected_region_metrics()) == sum(len(n) for n in names.values())
    monkeypatch.setenv(REGION_GROUPS_ENV, "none")
    assert task._get_affected_region_metrics() == []
