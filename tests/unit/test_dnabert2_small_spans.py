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


def test_nucleotide_token_base_offsets_pad_to_longest(nucleotide_model):
    offsets = nucleotide_model.get_token_base_offsets(["AAGAAA", "AA"])
    assert offsets.tolist() == [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, -1, -1, -1, -1]]


def test_bpe_like_token_base_offsets_follow_token_lengths(bpe_like_model):
    # tokens: A, AC, A, AC -> 1, 2, 1, 2 bases
    offsets = bpe_like_model.get_token_base_offsets(["AACAAC"])
    assert offsets.tolist() == [[0, 1, 3, 4, 6]]


def test_token_base_offsets_cover_only_the_retained_tokens(nucleotide_model):
    # max_length 12 keeps 10 content tokens: the offsets must stop there, like the embeddings
    offsets = nucleotide_model.get_token_base_offsets(["A" * 15])
    assert offsets.tolist() == [list(range(11))]


def test_nucleotide_suffix_starts(nucleotide_model):
    assert nucleotide_model.get_affected_suffix_starts(["AAGAAA"], ["AAAAAA"]).tolist() == [[3, 3]]
    # deletion: the variant's unchanged suffix starts one token before the reference's
    assert nucleotide_model.get_affected_suffix_starts(["AATAA"], ["AATTAA"]).tolist() == [[3, 4]]
    assert nucleotide_model.get_affected_suffix_starts(["AAA"], ["AAA"]).tolist() == [[-1, -1]]


def test_bpe_like_suffix_starts_follow_resegmentation(bpe_like_model):
    # AAC|AAA tokenizes A, AC, A, A, A; AAG|AAA tokenizes A, A, G, A, A, A
    starts = bpe_like_model.get_affected_suffix_starts(["AAGAAA"], ["AACAAA"])
    assert starts.tolist() == [[3, 2]]


# --------------------------------------------------------------- forced segmentation and context tools


def test_forced_segmentation_keeps_reference_boundaries(bpe_like_model):
    # ref A A T T -> [A][A][T][T]; var A C T T re-segments naturally to [AC][T][T], but with the
    # reference's boundaries only the substituted token is re-tokenized: [A][C][T][T]
    var, ref = "ACTT", "AATT"
    assert bpe_like_model.get_affected_token_spans([var], [ref], 0).tolist() == [[0, 1, 0, 2]]  # natural
    assert bpe_like_model.get_forced_segmentation_spans([var], [ref], 0).tolist() == [[1, 2, 1, 2]]
    assert bpe_like_model.get_forced_segmentation_spans([var], [ref], 1).tolist() == [[0, 3, 0, 3]]
    forced = bpe_like_model._forced_segmentation(var, ref)
    assert forced[0] == (ord("A"), ord("C"), ord("T"), ord("T"))


def test_forced_segmentation_retokenizes_a_multi_base_reference_token(bpe_like_model):
    # ref [AC][G][T]; var A A G T: the substituted base lies in the 2-base token [AC], which is
    # replaced by the tokenization of "AA" on its own
    forced = bpe_like_model._forced_segmentation("AAGT", "ACGT")
    assert forced == ((ord("A"), ord("A"), ord("G"), ord("T")), 0, 2, 1, 3)
    spans = bpe_like_model.get_forced_segmentation_spans(["AAGT"], ["ACGT"], 0)
    assert spans.tolist() == [[0, 2, 0, 1]]


def test_forced_segmentation_is_undefined_for_indels_identical_and_truncated(nucleotide_model):
    assert nucleotide_model.get_forced_segmentation_spans(["AAA"], ["AAAA"]).tolist() == [[-1, -1, -1, -1]]
    assert nucleotide_model.get_forced_segmentation_spans(["ACGT"], ["ACGT"]).tolist() == [[-1, -1, -1, -1]]
    ref = "A" * 20
    var = "A" * 15 + "G" + "A" * 4
    assert nucleotide_model.get_forced_segmentation_spans([var], [ref]).tolist() == [[-1, -1, -1, -1]]


def test_forced_segmentation_equals_natural_for_single_nucleotide(nucleotide_model):
    nucleotide_model.eval()
    ref = ["ACGTACGTAC", "TTGACCATGA"]
    var = ["ACGAACGTAC", "TTGACTATGA"]
    forced = nucleotide_model.infer_forced_segmentation_embeddings(var, ref)
    _, natural, _ = nucleotide_model.infer_sequence_to_sequence(var)
    np.testing.assert_allclose(forced, natural, atol=1e-6)
    np.testing.assert_array_equal(
        nucleotide_model.get_forced_segmentation_spans(var, ref, 2),
        nucleotide_model.get_affected_token_spans(var, ref, 2),
    )


def test_forced_embeddings_differ_from_natural_when_resegmentation_happens(bpe_like_model):
    bpe_like_model.eval()
    forced = bpe_like_model.infer_forced_segmentation_embeddings(["ACTT"], ["AATT"])
    _, natural, _ = bpe_like_model.infer_sequence_to_sequence(["ACTT"])
    # natural has 3 tokens ([AC][T][T]); forced has 4 ([A][C][T][T]), so token 1 differs
    assert not np.allclose(forced[0, :3], natural[0, :3])


def test_get_token_ids_matches_embedded_tokens(bpe_like_model):
    ids = bpe_like_model.get_token_ids(["ACGT", "AAAAAAAAAAAAAAAAAAAA"])
    assert ids[0, :3].tolist() == [100, ord("G"), ord("T")] and ids[0, 3:].tolist() == [-1] * (ids.shape[1] - 3)
    assert ids.shape[1] == 10  # truncated to the 10 content tokens of max_length=12


def test_masked_token_logprobs_match_the_snv_masked_probabilities(nucleotide_model):
    nucleotide_model.eval()
    seqs = ["ACGTACGTAC", "TTGACCATGA"]
    positions = np.array([[3, -1, 20], [0, 9, 4]])
    logprobs = nucleotide_model.infer_masked_token_logprobs(seqs, positions)
    assert logprobs.shape == positions.shape
    assert np.isnan(logprobs[0, 1]) and np.isnan(logprobs[0, 2])  # unused slot, beyond the sequence
    for row, slot in [(0, 0), (1, 0), (1, 1), (1, 2)]:
        pos = positions[row, slot]
        letter = seqs[row][pos]
        _, ref_probs = nucleotide_model.infer_masked_sequence_to_token_probs([seqs[row]], pos, [letter], [letter])
        assert logprobs[row, slot] == pytest.approx(np.log(ref_probs[0]), abs=1e-4)
    assert (logprobs[~np.isnan(logprobs)] < 0).all()


def test_masked_token_logprobs_chunking_is_consistent(nucleotide_model, monkeypatch):
    nucleotide_model.eval()
    seqs = ["ACGTACGTAC", "TTGACCATGA"]
    positions = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    whole = nucleotide_model.infer_masked_token_logprobs(seqs, positions)
    import usage_examples.sanity_models.dnabert2_small_model as module

    monkeypatch.setattr(module, "_MASKED_LOGPROB_TOKEN_BUDGET", 1)  # one position per forward pass
    np.testing.assert_allclose(nucleotide_model.infer_masked_token_logprobs(seqs, positions), whole, atol=1e-5)


def test_task_layer_produces_every_group_for_single_nucleotide(monkeypatch, nucleotide_model):
    from gfmbench_api.tasks.base.base_gfm_zero_shot_task import REGION_GROUPS_ENV
    from tests.unit.test_affected_region_extensions import _task_stub

    monkeypatch.setenv(REGION_GROUPS_ENV, "all")
    nucleotide_model.eval()
    rng = np.random.default_rng(0)
    refs, vars_, labels = [], [], []
    for i in range(24):
        ref = "".join(rng.choice(list("ACGT"), size=230))
        position = int(rng.integers(110, 120))
        alt = rng.choice([b for b in "ACGT" if b != ref[position]])
        refs.append(ref)
        vars_.append(ref[:position] + alt + ref[position + 1:])
        labels.append(i % 2)
    labels = np.array(labels)
    # long enough that every context-profile bin (out to 100 bases) has tokens on both sides
    from usage_examples.sanity_models.dnabert2_small_model import DNABERT2SmallModel

    model = DNABERT2SmallModel(tokenizer_kind="nucleotide", max_length=240).eval()
    task = _task_stub()
    task.disable_safe_model_call = True  # surface errors instead of silently returning None
    _, var_emb, var_repr = model.infer_sequence_to_sequence(vars_)
    _, ref_emb, ref_repr = model.infer_sequence_to_sequence(refs)
    outputs = {"variant_embeddings": var_emb, "reference_embeddings": ref_emb, "labels": labels}
    task._add_affected_region_outputs(model, vars_, refs, outputs)
    results = {}
    for metric, keys in task._get_affected_region_metrics():
        metric.calc(*[outputs[key] for key in keys])
        results[metric.name] = metric.get_final_results()
    # nothing is silently empty on a model that implements everything
    assert all(value is not None for value in results.values()), [k for k, v in results.items() if v is None]
    # single-nucleotide: forced segmentation is the natural one, regions are one token, nothing is unstable
    assert results["forced_region_valid_fraction"] == 1.0
    assert results["forced_region_single_token_fraction"] == 1.0
    assert results["forced_region_resegmented_fraction"] == 0.0
    assert results["affected_region_phase_unstable_fraction"] == 0.0
    assert results["forced_region_k0_cosinesim_auroc"] == pytest.approx(results["affected_region_k0_cosinesim_auroc"])
    assert results["forced_region_k2_l2_auroc"] == pytest.approx(results["affected_region_k2_l2_auroc"])
    assert results["affected_region_phase_shift0_k0_cosinesim_auroc"] == pytest.approx(
        results["affected_region_k0_cosinesim_auroc"], abs=0.2  # same variants unless a shift drops one
    )
    assert results["context_llr_mean_tokens"] > 0


def test_task_layer_with_resegmenting_tokenizer_and_indels(monkeypatch):
    """Multi-base tokens, SNVs and deletions together: nothing raises and forced differs from natural."""
    from gfmbench_api.tasks.base.base_gfm_zero_shot_task import REGION_GROUPS_ENV
    from tests.unit.test_affected_region_extensions import _task_stub

    monkeypatch.setenv(REGION_GROUPS_ENV, "all")
    model = DNABERT2SmallModel(tokenizer_kind="bpe", max_length=240)
    model.encode = _bpe_like_encode
    model.eval()
    rng = np.random.default_rng(3)
    refs, vars_, labels = [], [], []
    for i in range(40):
        ref = "".join(rng.choice(list("ACGT"), size=200))
        position = int(rng.integers(90, 110))
        if i % 4 == 3:  # deletion of 1-3 bases
            var = ref[:position] + ref[position + 1 + i % 3:]
        else:  # SNV, often creating or destroying an "AC" merge
            alt = rng.choice([b for b in "ACGT" if b != ref[position]])
            var = ref[:position] + alt + ref[position + 1:]
        refs.append(ref)
        vars_.append(var)
        labels.append(i % 2)
    labels = np.array(labels)
    task = _task_stub()
    task.disable_safe_model_call = True
    _, var_emb, _ = model.infer_sequence_to_sequence(vars_)
    _, ref_emb, _ = model.infer_sequence_to_sequence(refs)
    outputs = {"variant_embeddings": var_emb, "reference_embeddings": ref_emb, "labels": labels}
    task._add_affected_region_outputs(model, vars_, refs, outputs)
    results = {}
    for metric, keys in task._get_affected_region_metrics():
        metric.calc(*[outputs[key] for key in keys])
        results[metric.name] = metric.get_final_results()
    assert results["forced_region_valid_fraction"] == pytest.approx(30 / 40)  # deletions have no forced segmentation
    assert 0 < results["forced_region_resegmented_fraction"] <= 1
    assert 0 <= results["affected_region_phase_unstable_fraction"] <= 1  # this toy merge is phase-independent
    for name in ("forced_region_k0_cosinesim_auroc", "affected_region_phase_mean_k0_l2_auroc",
                 "affected_context_down_0_10_l2_auroc", "context_llr_both_sum_auroc",
                 "tokenizer_surprisal_delta_auroc"):
        assert results[name] is not None and 0 <= results[name] <= 1, name
