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
"""Per-batch score functions shared by the AUROC and AUPRC variants of each metric."""
import numpy as np
from .base_metric import memoize_last_call


@memoize_last_call
def sum_probs_neg_llr(variant_probs, reference_probs, epsilon):
    """Negative LLR of the summed log probabilities: [batch_size, seq_len] -> [batch_size]."""
    # Verify probabilities are valid (between 0 and 1)
    assert np.all((variant_probs >= 0) & (variant_probs <= 1)), \
        "Variant probabilities must be between 0 and 1"
    assert np.all((reference_probs >= 0) & (reference_probs <= 1)), \
        "Reference probabilities must be between 0 and 1"

    # Calculate log-likelihood for all sequences in batch
    # Sum log probabilities across sequence dimension: [batch_size, seq_len] -> [batch_size]
    log_likelihood_variant = np.sum(np.log(variant_probs + epsilon), axis=1)
    log_likelihood_reference = np.sum(np.log(reference_probs + epsilon), axis=1)

    # Calculate LLR: [batch_size]
    llr_scores = log_likelihood_variant - log_likelihood_reference

    # We use -LLR because lower LLR indicates pathogenic variants (label=1)
    # and sklearn metrics expect higher scores to indicate the positive class
    return -llr_scores



@memoize_last_call
def masked_token_neg_llr(variant_probs, reference_probs, epsilon):
    """Negative LLR of the masked-position token probabilities: [batch_size] -> [batch_size]."""
    # Verify probabilities are valid (between 0 and 1)
    assert np.all((variant_probs >= 0) & (variant_probs <= 1)), \
        "Variant probabilities must be between 0 and 1"
    assert np.all((reference_probs >= 0) & (reference_probs <= 1)), \
        "Reference probabilities must be between 0 and 1"

    # Calculate log-likelihood for variant and reference nucleotides
    log_likelihood_variant = np.log(variant_probs + epsilon)
    log_likelihood_reference = np.log(reference_probs + epsilon)

    # We use -LLR because lower LLR indicates pathogenic variants (label=1)
    # and sklearn metrics expect higher scores to indicate the positive class
    return -(log_likelihood_variant - log_likelihood_reference)



@memoize_last_call
def l2_distance_scores(variant_repr, reference_repr):
    """L2 (Euclidean) distance between representatives: [batch_size, hidden_dim] -> [batch_size]."""
    return np.linalg.norm(variant_repr - reference_repr, axis=1)



@memoize_last_call
def neg_cosine_sim_scores(variant_repr, reference_repr):
    """Negative cosine similarity between representatives: [batch_size, hidden_dim] -> [batch_size]."""
    # Normalize embeddings to unit vectors for cosine similarity
    variant_repr_norm = variant_repr / (np.linalg.norm(variant_repr, axis=1, keepdims=True) + 1e-10)
    reference_repr_norm = reference_repr / (np.linalg.norm(reference_repr, axis=1, keepdims=True) + 1e-10)

    # Compute cosine similarity: dot product of normalized vectors
    cosine_sim = np.sum(variant_repr_norm * reference_repr_norm, axis=1)

    # Lower cosine similarity (less similar) indicates pathogenic variants (label=1)
    # So we use negative cosine similarity as the prediction score
    return -cosine_sim



@memoize_last_call
def snv_neg_cosine_sim_scores(variant_embeddings, reference_embeddings, variant_snv_indices, reference_snv_indices):
    """Negative cosine similarity at the SNV position of each sequence.

    Returns (scores, valid_mask): scores cover only the sequences whose SNV indices are
    valid (scores is None if none are).

    Args:
        variant_embeddings: [batch_size, seq_len, hidden_dim] embeddings for variant sequences
        reference_embeddings: [batch_size, seq_len, hidden_dim] embeddings for reference sequences
        variant_snv_indices: [batch_size] array of SNV positions in variant sequences
        reference_snv_indices: [batch_size] array of SNV positions in reference sequences
    """
    batch_size = variant_embeddings.shape[0]
    hidden_dim = variant_embeddings.shape[2]
    
    # Initialize arrays for extracted embeddings
    variant_snv_emb = np.zeros((batch_size, hidden_dim))
    reference_snv_emb = np.zeros((batch_size, hidden_dim))
    valid_mask = np.ones(batch_size, dtype=bool)
    
    # Extract embeddings at the specific SNV position for each sequence
    for i in range(batch_size):
        var_idx = variant_snv_indices[i]
        ref_idx = reference_snv_indices[i]
        
        # Check if indices are valid and within bounds
        if (var_idx is None or var_idx < 0 or var_idx >= variant_embeddings.shape[1] or
            ref_idx is None or ref_idx < 0 or ref_idx >= reference_embeddings.shape[1]):
            valid_mask[i] = False
            continue
        
        # Extract embeddings at SNV position
        variant_snv_emb[i] = variant_embeddings[i, var_idx, :]
        reference_snv_emb[i] = reference_embeddings[i, ref_idx, :]
    
    # If no valid samples, there is nothing to score
    if not np.any(valid_mask):
        return None, valid_mask
    
    # Filter to only valid samples
    variant_snv_emb = variant_snv_emb[valid_mask]
    reference_snv_emb = reference_snv_emb[valid_mask]
    
    # Normalize embeddings to unit vectors for cosine similarity
    variant_snv_norm = variant_snv_emb / (np.linalg.norm(variant_snv_emb, axis=1, keepdims=True) + 1e-10)
    reference_snv_norm = reference_snv_emb / (np.linalg.norm(reference_snv_emb, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarity at SNV position
    # Cosine similarity = dot product of normalized vectors
    cosine_sim = np.sum(variant_snv_norm * reference_snv_norm, axis=1)  # [batch_size]
    
    # Lower cosine similarity (less similar) indicates pathogenic variants (label=1)
    # So we use negative cosine similarity as the prediction score
    # This way, higher scores indicate higher probability of being pathogenic
    neg_cosine_sim = -cosine_sim
    
    return neg_cosine_sim, valid_mask

