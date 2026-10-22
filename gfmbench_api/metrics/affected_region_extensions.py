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
Metrics that use the tokenization itself and how a variant changes the tokens after it.

Five groups, each switched on by name (see BaseGFMZeroShotTask.REGION_METRIC_GROUPS):

- ``forced``: embed the variant with the REFERENCE's token boundaries outside the mutated token(s),
  so only the substituted bases are re-tokenized. Comparing this with the natural variant
  tokenization separates "the substitution changed the embedding" from "the substitution
  re-segmented the sequence". (`ForcedSegmentationRegionMetric`, `ForcedSegmentationFraction`)
- ``tokstat``: model-free statistics of how the tokenization changed (token rarity, bases in the
  region). A control: if these score well, the region metrics partly measure the tokenizer.
  (`TokenizerStatisticAUROC`)
- ``phase``: the same region score after trimming 0-3 bases from the window start, which changes
  where BPE and Hamiltonian boundaries fall. (`SegmentationPhaseMetric`)
- ``profile``: how far the embedding change reaches beyond the region, from the unchanged tokens
  paired by identity (`context_profile`, `AffectedContextProfileMetric`, `AffectedContextExtentMetric`).
- ``context_llr``: how much less likely the variant makes the neighbouring unchanged tokens under the
  masked LM (`context_positions`, `ContextLikelihoodMetric`).

Higher scores mean more pathogenic unless a metric's description says otherwise.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from gfmbench_api.utils.token_diff import paired_flank_tokens

from .affected_region_variant_effect import _AffectedRegionVariantEffectMetric
from .affected_region_windows import _DISTANCES, _SCORES, _weighted_mean
from .base_metric import BaseMetric

# How far from the affected region the context profile looks, in bases
PROFILE_MAX_GAP_BASES = 100
# The tokenizer prior assumes this vocabulary size for add-one smoothing
_VOCABULARY_SIZE = 4096


def _profile_distance(distance, var, ref):
    """Non-negative distance per row: 1 - cosine similarity, or the L2 distance."""
    scores = _DISTANCES[distance](var, ref)
    return scores + 1.0 if distance == "cosinesim" else scores


# --------------------------------------------------------------------------- forced segmentation


class ForcedSegmentationRegionMetric(_AffectedRegionVariantEffectMetric):
    """
    Region score of the variant embedded with the reference's token boundaries.

    Same pooling and scoring as AffectedRegionCosineSimAUROC and friends, but it receives the
    embeddings of the forced-segmentation variant and the spans from
    BaseGFMModel.get_forced_segmentation_spans. Defined for equal-length substitutions only;
    other variants have an invalid span and are skipped.

    Args:
        context_tokens: unchanged tokens pooled on each side of the region
        distance: "cosinesim" or "l2"
        score: "auroc" or "auprc"
    """

    def __init__(self, context_tokens=0, distance="cosinesim", score="auroc"):
        if distance not in _DISTANCES or score not in _SCORES:
            raise ValueError(f"unsupported distance/score: {distance}/{score}")
        self.distance = distance
        self.score = score
        super().__init__(context_tokens)

    @property
    def name(self):
        return f"forced_region_k{self.context_tokens}_{self.distance}_{self.score}"

    def _score(self, variant_pooled, reference_pooled):
        return _DISTANCES[self.distance](variant_pooled, reference_pooled)

    def _final(self, gt, scores):
        return _SCORES[self.score](gt, scores)


class ForcedSegmentationFraction(BaseMetric):
    """
    Diagnostics of the forced segmentation (not a score), over all variants of the task.

    kind:
        "valid": share of variants with a forced segmentation (equal-length substitutions that the
            model can embed); 0 for indel tasks.
        "single_token": among those, share whose substituted bases form one vocabulary token, so
            the reference's boundaries could be kept exactly (always 1 for single-nucleotide).
        "resegmented": among variants valid for both, share whose natural variant tokenization
            differs from the forced one (the natural affected region is not just the mutated
            token(s)).

    Receives (forced_spans_k0, natural_spans_k0).
    """

    KINDS = ("valid", "single_token", "resegmented")

    def __init__(self, kind):
        if kind not in self.KINDS:
            raise ValueError(f"unsupported kind: {kind}")
        self.kind = kind
        super().__init__()

    @property
    def name(self):
        return f"forced_region_{self.kind}_fraction"

    def reset(self):
        super().reset()
        self._hits = 0
        self._total = 0

    def _calc_impl(self, forced_spans, natural_spans):
        forced = np.asarray(forced_spans)
        natural = np.asarray(natural_spans)
        forced_ok = forced[:, 0] >= 0
        if self.kind == "valid":
            self._hits += int(forced_ok.sum())
            self._total += len(forced)
        elif self.kind == "single_token":
            self._hits += int(((forced[:, 1] - forced[:, 0] == 1) & forced_ok).sum())
            self._total += int(forced_ok.sum())
        else:
            both = forced_ok & (natural[:, 0] >= 0)
            differs = (forced[:, :2] != natural[:, :2]).any(axis=1) | (forced[:, 2:] != natural[:, 2:]).any(axis=1)
            self._hits += int((differs & both).sum())
            self._total += int(both.sum())

    def get_final_results(self):
        return self._hits / self._total if self._total else None


# --------------------------------------------------------------------------- tokenizer statistics


class TokenizerStatisticAUROC(BaseMetric):
    """
    AUROC of a model-free statistic of how the variant changed the tokenization.

    Statistics (higher = predicted more pathogenic; AUROC < 0.5 means the opposite direction):
        "surprisal_delta": sum of -log p(token) over the variant's region tokens minus the same
            over the reference's, with p a unigram token frequency ("rarer tokens appear").
        "surprisal_absdelta": the absolute value of that difference.
        "max_alt_surprisal": the rarest token in the variant's region.
        "region_bases": bases covered by the region, mean of the two sides.
    p comes from ``prior_counts_path`` (an .npy array of token counts, e.g. over the pretraining
    corpus; see usage_examples/make_token_frequencies.py) or, without one, from the token counts of
    the reference sequences of this task, which is a weak prior for tasks with few distinct
    reference sequences. Counts are add-one smoothed.

    Receives (variant_token_ids, reference_token_ids, spans, variant_offsets, reference_offsets, gt),
    the k=0 spans index the same token lists.
    """

    STATISTICS = ("surprisal_delta", "surprisal_absdelta", "max_alt_surprisal", "region_bases")

    def __init__(self, statistic, prior_counts_path=None):
        if statistic not in self.STATISTICS:
            raise ValueError(f"unsupported statistic: {statistic}")
        self.statistic = statistic
        self._prior = None
        if prior_counts_path:
            counts = np.load(prior_counts_path).astype(np.float64)
            self._prior = (counts + 1.0) / (counts.sum() + _VOCABULARY_SIZE)
        super().__init__()

    @property
    def name(self):
        return f"tokenizer_{self.statistic}_auroc"

    def reset(self):
        super().reset()
        self._records = []
        self._counts = np.zeros(_VOCABULARY_SIZE, dtype=np.float64)

    def _calc_impl(self, variant_ids, reference_ids, spans, variant_offsets, reference_offsets, gt):
        spans = np.asarray(spans)
        gt = np.asarray(gt)
        for i in range(len(spans)):
            ref_row = np.asarray(reference_ids[i])
            if self._prior is None:
                ids = ref_row[ref_row >= 0]
                np.add.at(self._counts, np.clip(ids, 0, _VOCABULARY_SIZE - 1), 1.0)
            var_start, var_end, ref_start, ref_end = (int(x) for x in spans[i])
            if var_start < 0 or ref_start < 0:
                continue
            var_region = np.asarray(variant_ids[i])[var_start:var_end]
            ref_region = ref_row[ref_start:ref_end]
            if var_region.size == 0 or ref_region.size == 0 or (var_region < 0).any() or (ref_region < 0).any():
                continue
            var_bases = int(variant_offsets[i][var_end] - variant_offsets[i][var_start])
            ref_bases = int(reference_offsets[i][ref_end] - reference_offsets[i][ref_start])
            self._records.append((var_region, ref_region, (var_bases + ref_bases) / 2.0, int(gt[i])))

    def _surprisal(self, probabilities, ids):
        return -np.log(probabilities[np.clip(ids, 0, _VOCABULARY_SIZE - 1)])

    def get_final_results(self):
        if not self._records:
            return None
        if self._prior is not None:
            probabilities = self._prior
        else:
            probabilities = (self._counts + 1.0) / (self._counts.sum() + _VOCABULARY_SIZE)
        scores, labels = [], []
        for var_region, ref_region, bases, label in self._records:
            alt = self._surprisal(probabilities, var_region)
            ref = self._surprisal(probabilities, ref_region)
            if self.statistic == "surprisal_delta":
                score = alt.sum() - ref.sum()
            elif self.statistic == "surprisal_absdelta":
                score = abs(alt.sum() - ref.sum())
            elif self.statistic == "max_alt_surprisal":
                score = alt.max()
            else:
                score = bases
            scores.append(score)
            labels.append(label)
        if len(set(labels)) < 2:
            return None
        return roc_auc_score(labels, scores)


# --------------------------------------------------------------------------- segmentation phase


class SegmentationPhaseMetric(BaseMetric):
    """
    The region score after trimming 0..S-1 bases from the start of both sequences.

    BPE and Hamiltonian boundaries depend on where the sequence starts, so each shift re-segments
    the window; single-nucleotide tokenization is unaffected (a control). Only variants whose
    region is defined at every shift are used, so all modes score the same set of variants.

    Modes:
        "mean": the per-variant mean of the S region distances (an average over segmentation phases).
        "shift{s}": the region distance at shift s alone (shift0 is the plain k=0 score on this set).
        "spread": the standard deviation of the S distances (is instability itself informative?).
        "unstable_fraction": share of variants whose variant-side region size (tokens) differs
            between shifts (a diagnostic, not a score).

    Receives (variant_embeddings, reference_embeddings, spans, gt) where the first three are lists
    with one entry per shift, each as for AffectedRegionCosineSimAUROC (k=0 spans).
    """

    SHIFTS = (0, 1, 2, 3)

    def __init__(self, mode="mean", distance="cosinesim", score="auroc"):
        valid = {"mean", "spread", "unstable_fraction"} | {f"shift{s}" for s in self.SHIFTS}
        if mode not in valid or distance not in _DISTANCES or score not in _SCORES:
            raise ValueError(f"unsupported mode/distance/score: {mode}/{distance}/{score}")
        self.mode = mode
        self.distance = distance
        self.score = score
        super().__init__()

    @property
    def name(self):
        if self.mode == "unstable_fraction":
            return "affected_region_phase_unstable_fraction"
        return f"affected_region_phase_{self.mode}_k0_{self.distance}_{self.score}"

    def reset(self):
        super().reset()
        self._distances = []
        self._sizes = []
        self._gt = []

    def _calc_impl(self, variant_embeddings, reference_embeddings, spans, gt):
        n_shifts = len(self.SHIFTS)
        if not (len(variant_embeddings) == len(reference_embeddings) == len(spans) == n_shifts):
            raise ValueError(f"expected {n_shifts} shifts")
        pool = _AffectedRegionVariantEffectMetric._pool
        batch = np.asarray(gt).shape[0]
        distances = np.zeros((batch, n_shifts))
        sizes = np.zeros((batch, n_shifts))
        valid = np.ones(batch, dtype=bool)
        for s in range(n_shifts):
            span = np.asarray(spans[s])
            var_pooled, var_valid = pool(variant_embeddings[s], span[:, 0], span[:, 1])
            ref_pooled, ref_valid = pool(reference_embeddings[s], span[:, 2], span[:, 3])
            valid &= var_valid & ref_valid
            distances[:, s] = _DISTANCES[self.distance](var_pooled, ref_pooled)
            sizes[:, s] = span[:, 1] - span[:, 0]
        if valid.any():
            self._distances.append(distances[valid])
            self._sizes.append(sizes[valid])
            self._gt.append(np.asarray(gt)[valid])

    def get_final_results(self):
        if not self._distances:
            return None
        distances = np.concatenate(self._distances)
        sizes = np.concatenate(self._sizes)
        gt = np.concatenate(self._gt)
        if self.mode == "unstable_fraction":
            return float((sizes.max(axis=1) != sizes.min(axis=1)).mean())
        if self.mode == "mean":
            values = distances.mean(axis=1)
        elif self.mode == "spread":
            values = distances.std(axis=1)
        else:
            values = distances[:, int(self.mode[len("shift"):])]
        if len(set(gt.tolist())) < 2:
            return None
        return _SCORES[self.score](gt, values)


# --------------------------------------------------------------------------- context profile


def context_profile(var_emb, ref_emb, var_offsets, ref_offsets, var_span, ref_span, suffix_starts,
                    max_gap_bases=PROFILE_MAX_GAP_BASES):
    """
    Embedding change of the affected region and of the unchanged tokens around it.

    Unchanged tokens are paired by identity (`paired_flank_tokens`), so a flank token's distance
    is purely a context effect: the token is the same on both sides, only its surroundings changed.
    The region is one element, the base-weighted mean embedding of its tokens on each side.

    Returns None if undefined, else a dict {"core": {distance: float}, "left": (gaps, {distance:
    array}), "right": (gaps, {distance: array})}; distances are 1 - cosine and L2, and each side's
    arrays run nearest-to-farthest from the region, with gaps [n, 2] in bases from its edge.
    """
    flank = paired_flank_tokens(var_span, ref_span, suffix_starts, var_offsets, ref_offsets, max_gap_bases)
    if flank is None:
        return None
    left, left_gaps, right, right_gaps, shift = flank
    var_start, var_end = var_span
    ref_start, ref_end = ref_span
    var_bases = np.diff(np.asarray(var_offsets[var_start:var_end + 1])).astype(float)
    ref_bases = np.diff(np.asarray(ref_offsets[ref_start:ref_end + 1])).astype(float)
    if var_end > var_emb.shape[0] or ref_end > ref_emb.shape[0] or var_bases.sum() <= 0 or ref_bases.sum() <= 0:
        return None
    if (right.size and (right.max() >= var_emb.shape[0] or right.max() + shift >= ref_emb.shape[0])) or (
        left.size and left.max() >= min(var_emb.shape[0], ref_emb.shape[0])
    ):
        return None
    var_core = _weighted_mean(var_emb[var_start:var_end], var_bases)[None]
    ref_core = _weighted_mean(ref_emb[ref_start:ref_end], ref_bases)[None]
    profile = {"core": {}, "left": (left_gaps, {}), "right": (right_gaps, {})}
    for distance in _DISTANCES:
        profile["core"][distance] = float(_profile_distance(distance, var_core, ref_core)[0])
        for side, idx, ref_idx in (("left", left, left), ("right", right, right + shift)):
            profile[side][1][distance] = (
                _profile_distance(distance, var_emb[idx], ref_emb[ref_idx]) if idx.size else np.zeros(0)
            )
    return profile


class AffectedContextProfileMetric(BaseMetric):
    """
    Mean embedding change of the unchanged tokens within a base interval beside the region.

    Args:
        side: "down" (after the region, higher base index) or "up" (before it)
        lo, hi: the interval, in bases from the region's edge; a token counts with the length of
            its overlap with [lo, hi)
        distance: "cosinesim" (1 - cosine) or "l2"
        output: "auroc" (the mean as a pathogenicity score) or "mean" (the mean over variants, a
            diagnostic of the profile itself)

    Variants without any token in the interval are skipped. Receives (profiles, gt) with `profiles`
    one `context_profile` result (or None) per variant.
    """

    def __init__(self, side, lo, hi, distance="cosinesim", output="auroc"):
        if side not in ("up", "down") or distance not in _DISTANCES or output not in ("auroc", "mean") or not 0 <= lo < hi:
            raise ValueError(f"unsupported arguments: {side}/{lo}/{hi}/{distance}/{output}")
        self.side, self.lo, self.hi = side, int(lo), int(hi)
        self.distance = distance
        self.output = output
        super().__init__()

    @property
    def name(self):
        return f"affected_context_{self.side}_{self.lo}_{self.hi}_{self.distance}_{self.output}"

    def reset(self):
        super().reset()
        self._values = []
        self._gt = []

    def _calc_impl(self, profiles, gt):
        gt = np.asarray(gt)
        key = "right" if self.side == "down" else "left"
        for i, profile in enumerate(profiles):
            if profile is None:
                continue
            gaps, distances = profile[key]
            if not len(gaps):
                continue
            overlap = np.clip(np.minimum(gaps[:, 1], self.hi) - np.maximum(gaps[:, 0], self.lo), 0, None)
            if overlap.sum() <= 0:
                continue
            self._values.append(float((overlap * distances[self.distance]).sum() / overlap.sum()))
            self._gt.append(int(gt[i]))

    def get_final_results(self):
        if not self._values:
            return None
        if self.output == "mean":
            return float(np.mean(self._values))
        if len(set(self._gt)) < 2:
            return None
        return roc_auc_score(self._gt, self._values)


class AffectedContextExtentMetric(BaseMetric):
    """
    AUROC of how far the embedding change reaches beyond the affected region.

    kind:
        "reach": the largest distance, in bases from the region's edge, of an unchanged token whose
            embedding change is at least half the region's own change (0 if none is).
        "decay": the base-weighted mean change of the unchanged tokens within 50 bases divided by
            the region's change (1 = the context changes as much as the region).
    Variants whose region does not change (distance 0) or that have no unchanged token on the side
    are skipped. Receives (profiles, gt).
    """

    REACH_FRACTION = 0.5
    DECAY_WINDOW_BASES = 50

    def __init__(self, side, kind, distance="cosinesim"):
        if side not in ("up", "down") or kind not in ("reach", "decay") or distance not in _DISTANCES:
            raise ValueError(f"unsupported arguments: {side}/{kind}/{distance}")
        self.side, self.kind, self.distance = side, kind, distance
        super().__init__()

    @property
    def name(self):
        return f"affected_context_{self.side}_{self.kind}_{self.distance}_auroc"

    def reset(self):
        super().reset()
        self._values = []
        self._gt = []

    def _calc_impl(self, profiles, gt):
        gt = np.asarray(gt)
        key = "right" if self.side == "down" else "left"
        for i, profile in enumerate(profiles):
            if profile is None:
                continue
            gaps, distances = profile[key]
            core = profile["core"][self.distance]
            if not len(gaps) or core <= 1e-12:
                continue
            values = distances[self.distance]
            if self.kind == "reach":
                strong = values >= self.REACH_FRACTION * core
                value = float(gaps[strong, 1].max()) if strong.any() else 0.0
            else:
                overlap = np.clip(np.minimum(gaps[:, 1], self.DECAY_WINDOW_BASES) - gaps[:, 0], 0, None)
                if overlap.sum() <= 0:
                    continue
                value = float((overlap * values).sum() / overlap.sum() / core)
            self._values.append(value)
            self._gt.append(int(gt[i]))

    def get_final_results(self):
        if not self._values or len(set(self._gt)) < 2:
            return None
        return roc_auc_score(self._gt, self._values)


# --------------------------------------------------------------------------- context likelihood


def context_positions(var_span, ref_span, suffix_starts, var_offsets, ref_offsets, max_gap_bases, per_side):
    """
    Unchanged tokens next to the region whose masked likelihood is compared, paired by identity.

    Up to `per_side` nearest tokens on each side that start within `max_gap_bases` of the region.
    Returns (var_positions, ref_positions, bases), each of length 2 * per_side: the first `per_side`
    entries are the tokens after the region (nearest first), the rest the tokens before it, with -1
    for unused slots. `bases` is the token length in bases.
    """
    var_pos = np.full(2 * per_side, -1, dtype=np.int64)
    ref_pos = np.full(2 * per_side, -1, dtype=np.int64)
    bases = np.full(2 * per_side, -1, dtype=np.int64)
    flank = paired_flank_tokens(var_span, ref_span, suffix_starts, var_offsets, ref_offsets, max_gap_bases)
    if flank is None:
        return var_pos, ref_pos, bases
    left, _, right, _, shift = flank
    for slot, (idx, offset) in enumerate(((right, shift), (left, 0))):
        idx = idx[:per_side]
        sl = slice(slot * per_side, slot * per_side + len(idx))
        var_pos[sl] = idx
        ref_pos[sl] = idx + offset
        bases[sl] = np.asarray(var_offsets)[idx + 1] - np.asarray(var_offsets)[idx]
    return var_pos, ref_pos, bases


class ContextLikelihoodMetric(BaseMetric):
    """
    How much the variant lowers the masked-LM likelihood of its unchanged neighbouring tokens.

    For each paired neighbour token the model masks it and gives log p(token | rest) once in the
    reference and once in the variant; the per-token difference is log p_ref - log p_var. A variant
    that makes its surroundings less plausible scores high. Neighbours are the nearest tokens on
    each side, see `context_positions`.

    Args:
        side: "down" (tokens after the region), "up" (before it) or "both"
        normalize: "sum" of the differences, or "perbase": that sum per base of the tokens used
            (comparable across tokenizers whose tokens cover different numbers of bases)
        score: "auroc" or "auprc"

    Receives (context_delta, context_bases, gt): [batch, 2 * per_side] arrays (see
    `context_positions` for the slot layout), delta NaN where a slot is unused.
    """

    def __init__(self, side="both", normalize="sum", score="auroc"):
        if side not in ("down", "up", "both") or normalize not in ("sum", "perbase") or score not in _SCORES:
            raise ValueError(f"unsupported arguments: {side}/{normalize}/{score}")
        self.side, self.normalize, self.score = side, normalize, score
        super().__init__()

    @property
    def name(self):
        return f"context_llr_{self.side}_{self.normalize}_{self.score}"

    def reset(self):
        super().reset()
        self._values = []
        self._gt = []

    def _calc_impl(self, context_delta, context_bases, gt):
        delta = np.asarray(context_delta, dtype=np.float64)
        bases = np.asarray(context_bases, dtype=np.float64)
        gt = np.asarray(gt)
        half = delta.shape[1] // 2
        columns = {"down": slice(0, half), "up": slice(half, None), "both": slice(0, None)}[self.side]
        delta, bases = delta[:, columns], bases[:, columns]
        valid = np.isfinite(delta)
        keep = valid.any(axis=1)
        if not keep.any():
            return
        total = np.where(valid, delta, 0.0).sum(axis=1)
        if self.normalize == "perbase":
            total = total / np.maximum(np.where(valid, bases, 0.0).sum(axis=1), 1.0)
        self._values.append(total[keep])
        self._gt.append(gt[keep])

    def get_final_results(self):
        if not self._values:
            return None
        gt = np.concatenate(self._gt)
        if len(set(gt.tolist())) < 2:
            return None
        return _SCORES[self.score](gt, np.concatenate(self._values))


class ContextLikelihoodMeanTokens(BaseMetric):
    """Diagnostic: mean number of neighbouring tokens with a likelihood comparison per variant."""

    @property
    def name(self):
        return "context_llr_mean_tokens"

    def reset(self):
        super().reset()
        self._total = 0
        self._count = 0

    def _calc_impl(self, context_delta, context_bases, gt):
        delta = np.asarray(context_delta, dtype=np.float64)
        self._total += int(np.isfinite(delta).sum())
        self._count += delta.shape[0]

    def get_final_results(self):
        return self._total / self._count if self._count else None
