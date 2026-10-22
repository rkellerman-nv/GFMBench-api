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
from typing import Optional, Sequence, Tuple

import numpy as np


def _strip_common_tokens(reference_ids: Sequence[int], variant_ids: Sequence[int]):
    """Lengths of the longest common token prefix and (non-overlapping) suffix, or None if the
    token lists are identical."""
    n_ref, n_var = len(reference_ids), len(variant_ids)
    prefix = 0
    max_prefix = min(n_ref, n_var)
    while prefix < max_prefix and reference_ids[prefix] == variant_ids[prefix]:
        prefix += 1

    if prefix == n_ref == n_var:
        return None

    suffix = 0
    max_suffix = max_prefix - prefix
    while suffix < max_suffix and reference_ids[n_ref - 1 - suffix] == variant_ids[n_var - 1 - suffix]:
        suffix += 1
    return prefix, suffix


def divergent_token_span(
    reference_ids: Sequence[int],
    variant_ids: Sequence[int],
    context_tokens: int = 0,
) -> Tuple[int, int, int, int]:
    """
    Locate the tokens that differ between a reference and a variant tokenization.

    The longest common prefix and the longest common suffix of the two token-ID lists
    are stripped; what remains on each side is the region the variant affected. This
    is tokenizer-agnostic: for single-nucleotide tokenization it is the mutated token,
    for BPE-like tokenizers it extends until the tokenizations re-converge.

    Each region is then widened by `context_tokens` on both sides (clamped to the
    sequence) and is always at least one token wide, so pure token-level insertions or
    deletions still yield a region on the side that lost tokens.

    Args:
        reference_ids: token IDs of the reference sequence (no special tokens)
        variant_ids: token IDs of the variant sequence (no special tokens)
        context_tokens: number of unchanged tokens to include on each side of the region

    Returns:
        (ref_start, ref_end, var_start, var_end): half-open token index ranges.
        (-1, -1, -1, -1) if either sequence is empty or the token lists are identical.
    """
    n_ref, n_var = len(reference_ids), len(variant_ids)
    if n_ref == 0 or n_var == 0:
        return -1, -1, -1, -1

    stripped = _strip_common_tokens(reference_ids, variant_ids)
    if stripped is None:
        return -1, -1, -1, -1
    prefix, suffix = stripped

    def widen(start: int, end: int, n: int) -> Tuple[int, int]:
        start = min(start, n - 1)
        end = max(end, start + 1)
        return max(0, start - context_tokens), min(n, end + context_tokens)

    ref_start, ref_end = widen(prefix, n_ref - suffix, n_ref)
    var_start, var_end = widen(prefix, n_var - suffix, n_var)
    return ref_start, ref_end, var_start, var_end


def clipped_divergent_spans(
    reference_ids: Sequence[int],
    variant_ids: Sequence[int],
    context_tokens: int,
    max_tokens: int,
) -> Tuple[int, int, int, int]:
    """
    Affected-token spans for a pair whose model input is truncated to `max_tokens` tokens.

    The diff is taken on the FULL tokenizations, because truncating each side independently
    can cut them at different genomic positions and defeat common-suffix matching. The
    spans are then widened by `context_tokens` and clipped to the retained tokens. A pair
    whose affected tokens are all truncated away is invalid.

    Args:
        reference_ids: full (untruncated) reference token IDs, no special tokens
        variant_ids: full (untruncated) variant token IDs, no special tokens
        context_tokens: unchanged tokens to include on each side of the affected region
        max_tokens: number of content tokens the model actually embeds

    Returns:
        (var_start, var_end, ref_start, ref_end), the order of
        BaseGFMModel.get_affected_token_spans; (-1, -1, -1, -1) if invalid.
    """
    ref_start, ref_end, var_start, var_end = divergent_token_span(reference_ids, variant_ids)
    if ref_start < 0 or ref_start >= max_tokens or var_start >= max_tokens:
        return -1, -1, -1, -1
    return (
        max(0, var_start - context_tokens),
        min(len(variant_ids), max_tokens, var_end + context_tokens),
        max(0, ref_start - context_tokens),
        min(len(reference_ids), max_tokens, ref_end + context_tokens),
    )


def padded_token_base_offsets(token_bases_per_sequence: Sequence[Sequence[int]]) -> np.ndarray:
    """
    Cumulative base offsets of each sequence's tokens, padded with -1 into one array.

    Row i holds [0, b0, b0 + b1, ...]: entry j is the base index where token j starts and the
    entry after the last token is the number of bases the tokens cover. Pass the tokens the
    model actually embeds (after truncation), so the offsets index the same tokens as the
    embeddings.

    Args:
        token_bases_per_sequence: per sequence, the number of bases each retained token spans

    Returns:
        np.ndarray of shape [batch_size, max_tokens + 1], int32, padded with -1.
    """
    width = max((len(bases) for bases in token_bases_per_sequence), default=0) + 1
    offsets = np.full((len(token_bases_per_sequence), width), -1, dtype=np.int32)
    for i, bases in enumerate(token_bases_per_sequence):
        offsets[i, 0] = 0
        offsets[i, 1:len(bases) + 1] = np.cumsum(np.asarray(bases, dtype=np.int64))
    return offsets


def clipped_common_suffix_starts(
    reference_ids: Sequence[int],
    variant_ids: Sequence[int],
    max_tokens: int,
) -> Tuple[int, int]:
    """
    Token index where the unchanged common suffix begins, on each side.

    Unlike the spans from `divergent_token_span` (which are at least one token wide), these are
    the exact boundaries: the tokens from here on are the same on both sides, so the variant
    token at index v pairs with the reference token at index v + (ref_start - var_start). The
    spans' minimum width can swallow the first unchanged token of the shorter side (a pure
    insertion or deletion), so the spans' ends do not give this alignment.

    Args:
        reference_ids: full (untruncated) reference token IDs, no special tokens
        variant_ids: full (untruncated) variant token IDs, no special tokens
        max_tokens: number of content tokens the model actually embeds

    Returns:
        (var_suffix_start, ref_suffix_start), each clipped to `max_tokens` (a suffix that
        starts at or beyond the retained tokens is empty there); (-1, -1) if either sequence
        is empty or the token lists are identical.
    """
    if len(reference_ids) == 0 or len(variant_ids) == 0:
        return -1, -1
    stripped = _strip_common_tokens(reference_ids, variant_ids)
    if stripped is None:
        return -1, -1
    _, suffix = stripped
    return (min(len(variant_ids) - suffix, max_tokens),
            min(len(reference_ids) - suffix, max_tokens))


def padded_token_ids(token_ids_per_sequence: Sequence[Sequence[int]]) -> np.ndarray:
    """Token IDs of each sequence padded with -1 into one int32 array [batch, max_tokens]."""
    width = max((len(ids) for ids in token_ids_per_sequence), default=0)
    padded = np.full((len(token_ids_per_sequence), width), -1, dtype=np.int32)
    for i, ids in enumerate(token_ids_per_sequence):
        padded[i, :len(ids)] = np.asarray(ids, dtype=np.int32)
    return padded


def changed_base_interval(reference_sequence: str, variant_sequence: str) -> Optional[Tuple[int, int]]:
    """
    Half-open base interval [first, last) spanned by the substitutions between two equal-length
    sequences (case-insensitive), or None if the lengths differ or the sequences are identical.
    """
    if len(reference_sequence) != len(variant_sequence):
        return None
    ref, var = reference_sequence.upper(), variant_sequence.upper()
    if ref == var:
        return None
    first = next(i for i, (a, b) in enumerate(zip(ref, var)) if a != b)
    last = len(ref) - next(i for i, (a, b) in enumerate(zip(reversed(ref), reversed(var))) if a != b)
    return first, last


def tokens_covering_bases(offsets: Sequence[int], first_base: int, last_base: int) -> Tuple[int, int]:
    """
    Half-open token range [a, b) of the tokens that overlap the bases [first_base, last_base).

    `offsets` is one row of `padded_token_base_offsets` (token j starts at offsets[j]; -1 padded).
    Returns (-1, -1) if no token overlaps the interval (it lies past the covered bases).
    """
    bounds = np.asarray(offsets)
    bounds = bounds[bounds >= 0].astype(np.int64)
    if bounds.size < 2 or last_base <= first_base or first_base >= bounds[-1]:
        return -1, -1
    first = int(np.searchsorted(bounds, first_base, side="right")) - 1
    last = int(np.searchsorted(bounds, min(last_base, bounds[-1]), side="left"))
    return first, last


def paired_flank_tokens(var_span, ref_span, suffix_starts, var_offsets, ref_offsets, max_gap_bases):
    """
    Unchanged tokens next to the affected region, paired between variant and reference by identity.

    Tokens before the region sit at the same index on both sides. Tokens after it are shifted by
    `ref_suffix_start - var_suffix_start` (the spans' ends cannot give that shift, see
    `clipped_common_suffix_starts`). Both lists are ordered nearest-to-farthest and hold VARIANT
    token indices; a left token u pairs with reference token u, a right token v with reference
    token v + shift. A token is kept if it starts less than `max_gap_bases` from the region's edge
    on the variant side. Gaps are in bases from that edge: (gap_start, gap_end) of the token.

    Args:
        var_span, ref_span: (start, end) k=0 affected token spans of each side
        suffix_starts: (var_suffix_start, ref_suffix_start)
        var_offsets, ref_offsets: rows of `padded_token_base_offsets` of each side
        max_gap_bases: how far from the region to look

    Returns:
        None if undefined, else (left_idx, left_gaps, right_idx, right_gaps, shift) with gaps of
        shape [n, 2].
    """
    var_start, var_end = var_span
    ref_start, ref_end = ref_span
    var_suffix_start, ref_suffix_start = suffix_starts
    if var_start < 0 or ref_start < 0 or var_suffix_start < 0 or ref_suffix_start < 0:
        return None
    var_bounds = np.asarray(var_offsets)
    var_bounds = var_bounds[var_bounds >= 0].astype(np.int64)
    ref_bounds = np.asarray(ref_offsets)
    ref_bounds = ref_bounds[ref_bounds >= 0].astype(np.int64)
    n_var, n_ref = var_bounds.size - 1, ref_bounds.size - 1
    if var_end <= var_start or ref_end <= ref_start or var_end > n_var or ref_end > n_ref:
        return None

    left = np.arange(min(var_start, ref_start) - 1, -1, -1)
    left_gaps = np.stack([var_bounds[var_start] - var_bounds[left + 1],
                          var_bounds[var_start] - var_bounds[left]], axis=1)
    keep = left_gaps[:, 0] < max_gap_bases
    left, left_gaps = left[keep], left_gaps[keep]

    shift = ref_suffix_start - var_suffix_start
    first_right = max(var_end, ref_end - shift)
    right = np.arange(first_right, min(n_var, n_ref - shift))
    right_gaps = np.stack([var_bounds[right] - var_bounds[var_end],
                           var_bounds[right + 1] - var_bounds[var_end]], axis=1)
    keep = right_gaps[:, 0] < max_gap_bases
    right, right_gaps = right[keep], right_gaps[keep]
    return left, left_gaps, right, right_gaps, shift
