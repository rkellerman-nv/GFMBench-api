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

from types import SimpleNamespace

import numpy as np
import pytest

from gfmbench_api.metrics import AffectedRegionL2AUROC, AffectedRegionLengthAUROC, AffectedRegionMeanLength
from gfmbench_api.utils.token_diff import clipped_common_suffix_starts, divergent_token_span


def test_snv_same_length_single_token():
    assert divergent_token_span([1, 2, 3, 4], [1, 2, 9, 4]) == (2, 3, 2, 3)


def test_bpe_style_reconvergence():
    # variant re-segments two tokens into three, then re-converges
    assert divergent_token_span([1, 2, 3, 7, 8], [1, 5, 6, 4, 7, 8]) == (1, 3, 1, 4)


def test_context_is_clamped():
    assert divergent_token_span([1, 2, 3], [9, 2, 3], context_tokens=5) == (0, 3, 0, 3)


def test_pure_insertion_has_min_width_one():
    ref_start, ref_end, var_start, var_end = divergent_token_span([1, 2, 3], [1, 2, 4, 3])
    assert (var_start, var_end) == (2, 3)
    assert ref_end - ref_start == 1


def test_identical_and_empty():
    assert divergent_token_span([1, 2], [1, 2]) == (-1, -1, -1, -1)
    assert divergent_token_span([1, 2], [1, 2], context_tokens=2) == (-1, -1, -1, -1)
    assert divergent_token_span([1, 2, 3], [1, 2, 3, 4]) == (2, 3, 3, 4)
    assert divergent_token_span([], [1]) == (-1, -1, -1, -1)


def affected_spans_for_tokens(ref_ids, var_ids, context_tokens=0):
    from usage_examples.sanity_models.dna_bert2_model import DNABERT2Model

    def tokenizer(seq, *, add_special_tokens, truncation, max_length=None):
        assert add_special_tokens is False
        ids = {"ref": ref_ids, "var": var_ids}[seq]
        return {"input_ids": ids[:max_length] if truncation else ids}

    model = SimpleNamespace(tokenizer=tokenizer, max_length=8)
    return DNABERT2Model.get_affected_token_spans(
        model, ["var"], ["ref"], context_tokens
    )[0].tolist()


@pytest.mark.parametrize("context_tokens, expected", [
    (0, [1, 4, 1, 3]),
    (2, [0, 6, 0, 5]),
])
def test_wrapper_preserves_bpe_reconvergence_before_truncation(context_tokens, expected):
    assert affected_spans_for_tokens(
        [1, 2, 3, 7, 8, 9, 10, 11],
        [1, 5, 6, 4, 7, 8, 9, 10, 11],
        context_tokens,
    ) == expected


@pytest.mark.parametrize("context_tokens", [0, 2])
def test_wrapper_skips_mutation_outside_retained_context(context_tokens):
    assert affected_spans_for_tokens(
        [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 99], context_tokens
    ) == [-1, -1, -1, -1]


def test_wrapper_clips_partially_retained_affected_region():
    assert affected_spans_for_tokens(
        [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 99, 98, 97, 8]
    ) == [4, 6, 4, 6]


def test_wrapper_skips_identical_tokenizations():
    assert affected_spans_for_tokens([1, 2, 3], [1, 2, 3]) == [-1, -1, -1, -1]


def test_metric_pools_per_side_spans_and_skips_invalid():
    hidden = 4
    ref = np.zeros((4, 6, hidden))
    var = np.zeros((4, 6, hidden))
    # label-1 samples differ only inside their (differently sized) spans
    var[1, 2:5] = 3.0
    var[3, 1:3] = 3.0
    spans = np.array([
        [1, 2, 1, 2],
        [2, 5, 2, 4],
        [-1, -1, -1, -1],
        [1, 3, 1, 3],
    ])
    gt = np.array([0, 1, 0, 1])
    metric = AffectedRegionL2AUROC(context_tokens=0)
    metric.calc(var, ref, spans, gt)
    assert metric.name == "affected_region_k0_l2_auroc"
    assert metric.get_final_results() == 1.0
    assert len(metric._gt_list[0]) == 3  # invalid sample dropped


def test_length_diagnostics():
    emb = np.zeros((4, 6, 2))
    # (var_start, var_end, ref_start, ref_end): label-1 samples have larger regions
    spans = np.array([
        [1, 2, 1, 2],   # length 1
        [1, 4, 1, 3],   # length 2.5
        [-1, -1, -1, -1],  # invalid, ignored
        [2, 5, 2, 5],   # length 3
    ])
    gt = np.array([0, 1, 0, 1])

    auroc = AffectedRegionLengthAUROC()
    auroc.calc(emb, emb, spans, gt)
    assert auroc.name == "affected_region_length_auroc"
    assert auroc.get_final_results() == 1.0

    mean = AffectedRegionMeanLength()
    mean.calc(emb, emb, spans, gt)
    assert mean.name == "affected_region_mean_tokens"
    assert mean.get_final_results() == pytest.approx((1 + 2.5 + 3) / 3)


def test_common_suffix_starts_are_exact_not_minimum_width():
    # reference a b c d e f, variant deletes c: the shorter side's span is d (min width) but its
    # suffix starts at d too, while the reference suffix starts one token later, at d.
    assert divergent_token_span([1, 2, 3, 4, 5, 6], [1, 2, 4, 5, 6]) == (2, 3, 2, 3)
    assert clipped_common_suffix_starts([1, 2, 3, 4, 5, 6], [1, 2, 4, 5, 6], 100) == (2, 3)
    # substitution that re-segments 2 tokens into 3
    assert clipped_common_suffix_starts([1, 2, 3, 4, 5, 6], [1, 2, 9, 8, 7, 5, 6], 100) == (5, 4)
    # clipped to the retained tokens
    assert clipped_common_suffix_starts([1, 2, 3, 4, 5, 6], [1, 2, 9, 8, 7, 5, 6], 4) == (4, 4)


def test_common_suffix_starts_identical_or_empty():
    assert clipped_common_suffix_starts([1, 2], [1, 2], 10) == (-1, -1)
    assert clipped_common_suffix_starts([], [1], 10) == (-1, -1)
