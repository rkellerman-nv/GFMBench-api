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

pytest.importorskip("ham_dna_tokenizer")
pytest.importorskip("torch")

from usage_examples.sanity_models.dnabert2_small_model import DNABERT2SmallModel


def _bpe_like_encode(sequence):
    """Greedy 'AC' merge: a stand-in for a tokenizer where one SNV re-segments its neighbourhood."""
    ids, bases, i = [], [], 0
    while i < len(sequence):
        if sequence[i : i + 2] == "AC":
            ids.append(100)
            bases.append(2)
            i += 2
        else:
            ids.append(ord(sequence[i]))
            bases.append(1)
            i += 1
    return ids, bases


@pytest.fixture(scope="module")
def nucleotide_model():
    return DNABERT2SmallModel(tokenizer_kind="nucleotide", max_length=12)


@pytest.fixture(scope="module")
def bpe_like_model():
    model = DNABERT2SmallModel(tokenizer_kind="bpe", max_length=12)
    model.encode = _bpe_like_encode
    return model


def test_nucleotide_snv_is_single_token(nucleotide_model):
    spans = nucleotide_model.get_affected_token_spans(["AAGAAA"], ["AAAAAA"], 0)
    assert spans.tolist() == [[2, 3, 2, 3]]
    spans = nucleotide_model.get_affected_token_spans(["AAGAAA"], ["AAAAAA"], 1)
    assert spans.tolist() == [[1, 4, 1, 4]]


def test_nucleotide_deletion_has_min_width_on_variant_side(nucleotide_model):
    spans = nucleotide_model.get_affected_token_spans(["AATAA"], ["AATTAA"], 0)
    var_start, var_end, ref_start, ref_end = spans[0]
    assert var_end - var_start == 1 and ref_end - ref_start == 1


def test_bpe_like_resegmentation_reconverges(bpe_like_model):
    # ref: A C G T T  -> [AC][G][T][T];  var: A A G T T -> [A][A][G][T][T]
    spans = bpe_like_model.get_affected_token_spans(["AAGTT"], ["ACGTT"], 0)
    assert spans.tolist() == [[0, 2, 0, 1]]


def test_truncated_away_mutation_is_invalid(nucleotide_model):
    # max_length=12 -> 10 content tokens; the SNV is at base 15
    ref = "A" * 20
    var = "A" * 15 + "G" + "A" * 4
    assert nucleotide_model.get_affected_token_spans([var], [ref], 0).tolist() == [[-1, -1, -1, -1]]


def test_identical_pair_is_invalid(nucleotide_model):
    assert nucleotide_model.get_affected_token_spans(["ACGT"], ["ACGT"], 2).tolist() == [[-1, -1, -1, -1]]


def test_spans_index_the_returned_embeddings(nucleotide_model):
    var, ref = ["AAGAAA", "ACGTAC"], ["AAAAAA", "ACGTAA"]
    spans = nucleotide_model.get_affected_token_spans(var, ref, 2)
    _, var_emb, _ = nucleotide_model.infer_sequence_to_sequence(var)
    _, ref_emb, _ = nucleotide_model.infer_sequence_to_sequence(ref)
    assert (spans[:, 1] <= var_emb.shape[1]).all() and (spans[:, 3] <= ref_emb.shape[1]).all()
    assert (spans >= 0).all()
