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
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

class BaseGFMModel(ABC):
    """
    Base class for Genomic Foundation Models.
    Concrete classes must implement abstract methods.
    """
    
    @abstractmethod
    def __init__(self, device: str = 'cpu') -> None:
        """
        Initialize the model.
        
        Args:
            device: torch device ('cpu' or 'cuda')
        """
        pass
    
    @abstractmethod
    def infer_sequence_to_labels_probs(
        self, 
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Get label probabilities for sequences (for classification tasks).
        
        Args:
            sequences: list of str - batch of DNA sequences (e.g., ["ATCG", "GCTA"])
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            Probabilities with shape [batch_size, num_classes] for single-label
            tasks, [batch_size, num_labels] for binary multi-label tasks, or
            [batch_size, num_labels, num_classes] for multi-class multi-label
            tasks. Returns None if not implemented.
        """
        pass

    def infer_sequence_to_regression(
        self,
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return continuous predictions for a supervised regression task.

        The task's ``output_spatiality`` determines the expected shape:
        ``[batch_size, num_outputs]`` for sequence-level regression or
        ``[batch_size, num_bins, num_outputs]`` for binned regression.

        Models without a fitted regression head may return ``None``.
        """
        return None

    @abstractmethod
    def infer_sequence_to_sequence(
        self, 
        sequences: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get sequence-level outputs for zero-shot benchmarks including probabilities,
        embeddings, and representative embeddings.
        
        Args:
            sequences: list of str - batch of DNA sequences (e.g., ["ATCG", "GCTA"])
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            tuple: (sequence_probs, sequence_embeddings, sequence_representative)
                - sequence_probs: np.ndarray of shape [batch_size, seq_len] OR None
                                 Probability for each nucleotide in each sequence.
                                 For input ["ATCT", "ATCG"], returns:
                                 [[P(A), P(T), P(C), P(T)], [P(A), P(T), P(C), P(G)]]
                                 Returns None if model doesn't have vocabulary projection head (e.g., BERT without MLM head)
                                 
                - sequence_embeddings: np.ndarray of shape [batch_size, seq_len, hidden_dim] OR None
                                      Embeddings (logits before vocabulary projection) for each position.
                                      Returns None if model doesn't support this output.
                                      
                - sequence_representative: np.ndarray of shape [batch_size, hidden_dim] OR None
                                          Single embedding representing the entire sequence.
                                          For BERT: CLS token embedding
                                          For GPT: Last token embedding
                                          Returns None if model doesn't support this output.
        
        Note:
            - This method is used for two prediction approaches:
              1. LLR calculation using sequence_probs (Method 1)
              2. Cosine similarity calculation using sequence_representative (Method 2, BEND-style)
            - Models can return None for outputs they don't support
            - The benchmark will skip calculating metrics for unsupported outputs
            - At least one output should be non-None for the method to be useful
        """
        pass
    
    @abstractmethod
    def sequence_pos_to_prob_pos(self, sequences: List[str], pos: int) -> np.ndarray:
        """
        Map input DNA sequence position to output sequence position (accounting for tokenization).
        
        This is crucial for models with different tokenization strategies to correctly
        identify which output position corresponds to a given input nucleotide position.
        For tokenizers like BPE where the output position depends on the sequence content,
        this method computes the mapping for each sequence in the batch.
        
        Args:
            sequences: List[str] - batch of DNA sequences to analyze
                                  For BPE tokenizers, the tokenization depends on sequence content
            pos: int - position in the input DNA sequence (0-based index)
                     For sequence "ATCGTA", pos=0 is 'A', pos=3 is 'G', etc.
        
        Returns:
            np.ndarray: array of shape [batch_size] containing the corresponding output position
                       for each sequence. This is the index in sequence_embeddings/sequence_probs
                       that corresponds to the prediction for the nucleotide at input position `pos`.
                       Returns -1 for sequences where the position cannot be determined.
        
        Examples:
            - Single nucleotide tokenizer (e.g., SimpleDNATransformer):
              Input: ["ATCG", "GCTA"] → Tokens: [[CLS, A, T, C, G], [CLS, G, C, T, A]]
              pos=0 ('A'/'G') → output_pos=[0, 0] (after removing CLS, same for all sequences)
              
            - Autoregressive model (e.g., GPT):
              Input: ["ATCG", "GCTA"] → Output predicts next token at each position
              pos=1 ('T'/'C') → output_pos=[0, 0] (prediction at position 0 for token at 1)
              
            - K-mer tokenizer (e.g., 6-mer DNABERT):
              Input: ["ATCGATCG", "GCTAGCTA"] → 6-mers vary by sequence
              pos=2 → output_pos=[0, 0] (both in first k-mer)
              
            - BPE tokenizer (sequence-dependent):
              Input: ["ATCGATCG", "AAAAAACG"] → tokens depend on learned vocabulary
              pos=2 → output_pos=[1, 0] (can vary based on how each sequence tokenizes)
        
        Note:
            - This method accounts for special tokens (CLS, SEP) being removed from outputs
            - For k-mer models, returns the token index where the nucleotide is at position k//2
            - For autoregressive models, returns pos-1 (since output at i predicts token at i+1)
            - For BPE and other content-dependent tokenizers, analyzes each sequence separately
            - Returns -1 for any sequence where the position cannot be determined
        """
        pass
    
    def get_affected_token_spans(
        self,
        variant_sequences: List[str],
        reference_sequences: List[str],
        context_tokens: int = 0,
    ) -> np.ndarray:
        """
        Optional. Token spans of each variant/reference pair that the variant affected.

        Used by the affected-region variant effect metrics. Tokenize each pair (no special
        tokens) and pass the token IDs to `gfmbench_api.utils.token_diff.divergent_token_span`,
        which strips the common token prefix/suffix and widens by `context_tokens`.

        Args:
            variant_sequences: List[str] - batch of variant DNA sequences
            reference_sequences: List[str] - batch of matching reference DNA sequences
            context_tokens: int - unchanged tokens to include on each side of the affected region

        Returns:
            np.ndarray of shape [batch_size, 4] with rows (var_start, var_end, ref_start, ref_end).
            Half-open ranges indexing the sequence dimension of the embeddings returned by
            infer_sequence_to_sequence (i.e. after special tokens are removed). Use -1 for
            a pair whose span cannot be determined.

        Not abstract: models that do not implement it get None for these metrics.
        """
        raise NotImplementedError

    def get_affected_suffix_starts(
        self,
        variant_sequences: List[str],
        reference_sequences: List[str],
    ) -> np.ndarray:
        """
        Optional. Token index where the unchanged common suffix of each pair begins, per side.

        Used by the affected-region metrics that compare aligned tokens around the region. The
        spans from `get_affected_token_spans` are at least one token wide, which can swallow the
        first unchanged token of the shorter side of an insertion or deletion, so their ends do
        not tell which variant token matches which reference token. Pass the token IDs to
        `gfmbench_api.utils.token_diff.clipped_common_suffix_starts`.

        Args:
            variant_sequences: List[str] - batch of variant DNA sequences
            reference_sequences: List[str] - batch of matching reference DNA sequences

        Returns:
            np.ndarray of shape [batch_size, 2] with rows (var_suffix_start, ref_suffix_start),
            indices into the embeddings returned by infer_sequence_to_sequence, clipped to the
            number of embedded tokens. The variant token at index v matches the reference token
            at v + (ref_suffix_start - var_suffix_start) from there on. Use -1 for a pair that is
            undetermined or tokenizes identically.

        Not abstract: models that do not implement it get None for these metrics.
        """
        raise NotImplementedError

    def get_token_base_offsets(self, sequences: List[str]) -> np.ndarray:
        """
        Optional. Where each token starts, in bases, for the tokens the model embeds.

        Used by the affected-region metrics that define their window in bases rather than
        tokens (so tokenizers with different token lengths can be compared over the same
        stretch of DNA). Build the array with
        `gfmbench_api.utils.token_diff.padded_token_base_offsets` from the number of bases
        in each content token, after truncation to the model's length limit.

        Args:
            sequences: List[str] - batch of DNA sequences

        Returns:
            np.ndarray of shape [batch_size, max_tokens + 1], int, padded with -1. Row i is
            [0, b0, b0 + b1, ...]; entry j is the base index where token j of the embeddings
            returned by infer_sequence_to_sequence starts (CLS/SEP removed), and the entry after
            the last token is the number of bases those tokens cover.

        Not abstract: models that do not implement it get None for these metrics.
        """
        raise NotImplementedError

    def get_token_ids(self, sequences: List[str]) -> np.ndarray:
        """
        Optional. The token IDs the model embeds for each sequence (no CLS/SEP, after truncation).

        Used by the tokenizer-statistics metrics. Build the array with
        `gfmbench_api.utils.token_diff.padded_token_ids`.

        Returns:
            np.ndarray of shape [batch_size, max_tokens], int, padded with -1; column j is the
            token behind column j of the embeddings returned by infer_sequence_to_sequence.

        Not abstract: models that do not implement it get None for these metrics.
        """
        raise NotImplementedError

    def infer_forced_segmentation_embeddings(
        self,
        variant_sequences: List[str],
        reference_sequences: List[str],
    ) -> np.ndarray:
        """
        Optional. Embeddings of each variant tokenized with its REFERENCE's token boundaries.

        Only the tokens that overlap the substituted bases are re-tokenized (the substituted
        bases, cut out of the reference's boundaries, are tokenized on their own); every other
        token is the reference's own token. This removes the re-segmentation a substitution
        causes elsewhere, so it separates a change of the embedding from a change of the
        segmentation. Defined for equal-length variants; for others return any embeddings
        (the natural tokenization is fine) and mark them invalid in `get_forced_segmentation_spans`.

        Returns:
            np.ndarray [batch_size, num_tokens, hidden_dim] as infer_sequence_to_sequence's
            embeddings (CLS/SEP removed).

        Not abstract: models that do not implement it get None for these metrics.
        """
        raise NotImplementedError

    def get_forced_segmentation_spans(
        self,
        variant_sequences: List[str],
        reference_sequences: List[str],
        context_tokens: int = 0,
    ) -> np.ndarray:
        """
        Optional. Token spans of the substituted tokens for `infer_forced_segmentation_embeddings`.

        Returns:
            np.ndarray [batch_size, 4] of (var_start, var_end, ref_start, ref_end): half-open
            ranges of the re-tokenized substituted bases in the forced variant and of the
            reference tokens they replace, each widened by `context_tokens` unchanged tokens and
            clipped to the embedded tokens. -1 for variants that are not equal-length
            substitutions, are identical, or whose substituted tokens are truncated away.

        Not abstract: models that do not implement it get None for these metrics.
        """
        raise NotImplementedError

    def infer_masked_token_logprobs(self, sequences: List[str], token_positions: np.ndarray) -> np.ndarray:
        """
        Optional. Log-probability of the true token at given positions, each masked on its own.

        Used by the context-likelihood metrics. For each requested position the model sees the
        sequence with only that token masked and returns log p(true token | rest).

        Args:
            sequences: List[str] - batch of DNA sequences
            token_positions: np.ndarray [batch_size, P] of token indices into the embeddings
                returned by infer_sequence_to_sequence; -1 marks an unused slot

        Returns:
            np.ndarray [batch_size, P] float; NaN for unused slots.

        Not abstract: models that do not implement it get None for these metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_masked_sequence_to_token_probs(
        self, 
        sequences: List[str], 
        variant_pos: int,
        variant_letters: List[str], 
        reference_letters: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get token probabilities at a specific position using masked prediction.
        
        This method is used for SNV variant effect prediction where we mask the variant
        position and predict probabilities for both the variant and reference nucleotides.
        
        Args:
            sequences: List[str] - batch of DNA sequences
            variant_pos: int - position of the variant in the input sequence (0-based)
            variant_letters: List[str] - variant nucleotide for each sequence (e.g., ['A', 'T', 'G'])
            reference_letters: List[str] - reference nucleotide for each sequence (e.g., ['G', 'C', 'A'])
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                - variant_token_probs: Array of shape [batch_size] containing the probability
                                      of the variant nucleotide at the masked position for each sequence
                - reference_token_probs: Array of shape [batch_size] containing the probability
                                        of the reference nucleotide at the masked position for each sequence
                - Returns (None, None) if model doesn't support masked prediction
        
        Implementation notes:
            - Mask the sequences at variant_pos (e.g., replace with [MASK] token or similar)
            - Run model inference to predict the masked token
            - Extract probability for variant_letters[i] and reference_letters[i] for each sequence i
            - This method can leverage existing methods like tokenization and inference
            - Return (None, None) for models that don't support masked prediction
        
        Example:
            sequences = ["ATCG", "GCTA"]
            variant_pos = 1
            variant_letters = ["G", "A"]  # New nucleotides
            reference_letters = ["T", "C"]  # Original nucleotides
            
            # Masks position 1: ["A[MASK]CG", "G[MASK]TA"]
            # Returns probabilities: (P(G), P(A)), (P(T), P(C))
        """
        pass
    
    @abstractmethod
    def infer_variant_ref_sequences_to_labels_probs(
        self, 
        variant_sequences: List[str],
        ref_sequences: List[str],
        conditional_input: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Get label probabilities for variant/reference sequence pairs.
        
        This method is used for variant effect prediction tasks where the model
        receives both variant and reference sequences as input and predicts
        classification labels.
        
        Args:
            variant_sequences: List[str] - batch of variant DNA sequences
            ref_sequences: List[str] - batch of reference DNA sequences
            conditional_input: Optional[np.ndarray] - optional metadata inputs of shape [batch_size, num_metadata_inputs].
        
        Returns:
            Probabilities with shape [batch_size, num_classes] for single-label
            tasks, [batch_size, num_labels] for binary multi-label tasks, or
            [batch_size, num_labels, num_classes] for multi-class multi-label
            tasks. Returns None if not implemented.
        """
        pass
