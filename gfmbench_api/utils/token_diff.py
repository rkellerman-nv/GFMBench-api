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
from typing import Sequence, Tuple


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

    prefix = 0
    max_prefix = min(n_ref, n_var)
    while prefix < max_prefix and reference_ids[prefix] == variant_ids[prefix]:
        prefix += 1

    if prefix == n_ref == n_var:
        return -1, -1, -1, -1

    suffix = 0
    max_suffix = max_prefix - prefix
    while suffix < max_suffix and reference_ids[n_ref - 1 - suffix] == variant_ids[n_var - 1 - suffix]:
        suffix += 1

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
