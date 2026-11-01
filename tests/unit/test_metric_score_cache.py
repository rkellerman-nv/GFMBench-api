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

from unittest import mock

import numpy as np
import pytest

from gfmbench_api.metrics import (
    SequenceEmbeddingsCosineSimAUPRC,
    SequenceEmbeddingsCosineSimAUROC,
    SequenceEmbeddingsL2AUPRC,
    SequenceEmbeddingsL2AUROC,
    SNVVariantEffectCosineSimAUPRC,
    SNVVariantEffectCosineSimAUROC,
    SNVVariantEffectPredictionMaskedLLRAUPRC,
    SNVVariantEffectPredictionMaskedLLRAUROC,
    SumProbsLLRAUPRC,
    SumProbsLLRAUROC,
)
from gfmbench_api.metrics import _scores
from gfmbench_api.metrics.base_metric import memoize_last_call

rng = np.random.default_rng(0)
LABELS = np.array([0, 1, 0, 1])
PROBS = (rng.random((4, 10)), rng.random((4, 10)))
REPR = (rng.normal(size=(4, 8)), rng.normal(size=(4, 8)))
EMB = (rng.normal(size=(4, 6, 8)), rng.normal(size=(4, 6, 8)), np.arange(4), np.arange(4))
TOKEN = (rng.random(4), rng.random(4))

PAIRS = [
    (SumProbsLLRAUROC, SumProbsLLRAUPRC, "sum_probs_neg_llr", PROBS),
    (SequenceEmbeddingsCosineSimAUROC, SequenceEmbeddingsCosineSimAUPRC, "neg_cosine_sim_scores", REPR),
    (SequenceEmbeddingsL2AUROC, SequenceEmbeddingsL2AUPRC, "l2_distance_scores", REPR),
    (SNVVariantEffectCosineSimAUROC, SNVVariantEffectCosineSimAUPRC, "snv_neg_cosine_sim_scores", EMB),
    (SNVVariantEffectPredictionMaskedLLRAUROC, SNVVariantEffectPredictionMaskedLLRAUPRC,
     "masked_token_neg_llr", TOKEN),
]


@pytest.mark.parametrize("roc_cls, prc_cls, score_fn, inputs", PAIRS)
def test_pair_scores_each_batch_once(roc_cls, prc_cls, score_fn, inputs):
    calls = []
    original = getattr(_scores, score_fn).__wrapped__

    def counting(*args):
        calls.append(1)
        return original(*args)

    # Re-wrap the counting function the same way the module does.
    wrapped = memoize_last_call(counting)
    with mock.patch(f"gfmbench_api.metrics.{roc_cls.__module__.split('.')[-1]}.{score_fn}", wrapped), \
         mock.patch(f"gfmbench_api.metrics.{prc_cls.__module__.split('.')[-1]}.{score_fn}", wrapped):
        roc, prc = roc_cls(), prc_cls()
        roc.calc(*inputs, LABELS)
        prc.calc(*inputs, LABELS)
    assert len(calls) == 1


def test_non_array_arguments_are_never_matched():
    calls = []

    @memoize_last_call
    def fn(a):
        calls.append(1)
        return a

    fn([1, 2])
    fn([1, 2])
    assert len(calls) == 2
