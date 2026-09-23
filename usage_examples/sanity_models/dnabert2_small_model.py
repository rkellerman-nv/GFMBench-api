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

"""DNABERT2-small (~8M param) adapter for GFMBench, covering both the Hamiltonian
and BPE tokenizer flavors trained in the ham-dna-tokenizer repo.

Requires the ``ham-dna-tokenizer`` package (and its ``model``/``dnabert2-style``
extras) to be importable; it supplies the model architecture (``DNABERT2SmallForMaskedLM``)
and both tokenizer loaders behind a shared no-special-token encoding contract.
"""

from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ham_dna_tokenizer.dnabert2_model import DNABERT2SmallConfig, DNABERT2SmallForMaskedLM
from ham_dna_tokenizer.mlm_training import load_sequence_encoder

_CLS_ID = 1
_SEP_ID = 2
_PAD_ID = 3
_MASK_ID = 4

TokenizerKind = Literal["hamiltonian", "bpe"]


class DNABERT2SmallModel(nn.Module):
    """ham-dna-tokenizer's ~8M-parameter DNABERT2-style MLM encoder for GFMBench.

    Instantiate once per tokenizer flavor by passing ``tokenizer_kind`` ("hamiltonian"
    or "bpe") and the matching ``tokenizer_path`` (a Hamiltonian vocabulary JSON or a
    BPE ``tokenizer.json``, per the ham-dna-tokenizer README's training pipeline).
    """

    def __init__(
        self,
        device="cpu",
        tokenizer_kind: TokenizerKind = "hamiltonian",
        tokenizer_path: str = "",
        max_length: int = 128,
        pretrained: bool = True,
    ):
        """
        Args:
            device: torch device ('cpu' or 'cuda')
            tokenizer_kind: 'hamiltonian' or 'bpe', selecting the tokenizer flavor
            tokenizer_path: path to the trained tokenizer artifact for that flavor
            max_length: maximum sequence length including CLS/SEP (<= config.max_position_embeddings)
            pretrained: unused placeholder for registry symmetry with other wrappers;
                weights are always randomly initialized here and loaded via load_checkpoint
        """
        super().__init__()
        if not tokenizer_path:
            raise ValueError("tokenizer_path is required")
        self.device = device
        self.tokenizer_kind = tokenizer_kind
        self.tokenizer_path = Path(tokenizer_path)
        self.config = DNABERT2SmallConfig()
        if max_length > self.config.max_position_embeddings:
            raise ValueError(
                f"max_length ({max_length}) exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings})"
            )
        self.max_length = max_length

        print(f"Loading DNABERT2-small ({tokenizer_kind}) tokenizer from: {self.tokenizer_path}")
        self.encode = load_sequence_encoder(tokenizer_kind, self.tokenizer_path)

        self.model = DNABERT2SmallForMaskedLM(self.config)
        self.add_module("model", self.model)
        self.model.to(device)

        self.cls_index = 0
        self.hidden_dim = self.config.hidden_size
        self.mlm_head_loaded = True

        print(
            f"DNABERT2-small ({tokenizer_kind}) loaded. Hidden dim: {self.hidden_dim}, "
            f"max_length: {self.max_length}, params: {self.model.num_parameters():,}"
        )

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def get_hidden_dim(self):
        return self.hidden_dim

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, param in self.model.named_parameters(prefix="", recurse=recurse):
            yield name, param

    def state_dict(self, prefix: str = "", keep_vars: bool = False):
        return self.model.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def _encode_content(self, sequence: str) -> Tuple[List[int], List[int]]:
        """Tokenize DNA content only, truncated to fit CLS/SEP within max_length."""
        content_length = self.max_length - 2
        token_ids, token_bases = self.encode(sequence)
        return token_ids[:content_length], token_bases[:content_length]

    def tokenize(self, sequences: List[str]):
        batch_ids = []
        for sequence in sequences:
            content_ids, _ = self._encode_content(sequence)
            ids = [_CLS_ID, *content_ids, _SEP_ID]
            batch_ids.append(ids)
        width = max(len(ids) for ids in batch_ids)
        input_ids = torch.full((len(batch_ids), width), _PAD_ID, dtype=torch.long)
        for row, ids in enumerate(batch_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(_PAD_ID)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def infer_sequence_to_sequence(self, sequences, conditional_input=None):
        encoded = self.tokenize(sequences)

        with torch.no_grad():
            outputs = self.model(**encoded)
        hidden_states = outputs.last_hidden_state

        sequence_representative = hidden_states[:, self.cls_index, :]
        hidden_states_seq = hidden_states[:, 1:-1, :]

        return (
            None,
            hidden_states_seq.detach().cpu().numpy(),
            sequence_representative.detach().cpu().numpy(),
        )

    def sequence_pos_to_prob_pos(self, sequences, pos):
        batch_size = len(sequences)
        output_positions = np.zeros(batch_size, dtype=np.int32)

        for i, seq in enumerate(sequences):
            _, token_bases = self._encode_content(seq)
            cursor = 0
            token_idx = -1
            for idx, length in enumerate(token_bases):
                if cursor <= pos < cursor + length:
                    token_idx = idx
                    break
                cursor += length
            output_positions[i] = token_idx

        return output_positions

    def infer_sequence_to_labels_probs(self, sequences, conditional_input=None):
        return None

    def infer_variant_ref_sequences_to_labels_probs(
        self, variant_sequences, ref_sequences, conditional_input=None
    ):
        return None

    def infer_masked_sequence_to_token_probs(
        self,
        sequences: List[str],
        variant_pos: int,
        variant_letters: List[str],
        reference_letters: List[str],
        conditional_input=None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.mlm_head_loaded:
            return None, None

        batch_size = len(sequences)
        encoded = self.tokenize(sequences)
        input_ids = encoded["input_ids"]

        mask_token_positions = np.full(batch_size, -1, dtype=np.int64)
        for i, seq in enumerate(sequences):
            _, token_bases = self._encode_content(seq)
            cursor = 0
            token_idx = -1
            for idx, length in enumerate(token_bases):
                if cursor <= variant_pos < cursor + length:
                    token_idx = idx
                    break
                cursor += length
            if token_idx >= 0:
                # +1 shifts past the CLS token prepended in tokenize().
                mask_token_positions[i] = token_idx + 1
                input_ids[i, token_idx + 1] = _MASK_ID

        if (mask_token_positions < 0).any():
            return None, None

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=encoded["attention_mask"])
            probs = torch.softmax(outputs.logits, dim=-1)

        nucleotide_token_ids = {}
        for nuc in ["A", "T", "C", "G"]:
            tokens, _ = self.encode(nuc)
            if len(tokens) == 1:
                nucleotide_token_ids[nuc] = tokens[0]

        if len(nucleotide_token_ids) < 4:
            return None, None

        variant_probs_list = []
        reference_probs_list = []
        for i in range(batch_size):
            mask_pos = mask_token_positions[i]
            var_token_id = nucleotide_token_ids.get(variant_letters[i].upper())
            ref_token_id = nucleotide_token_ids.get(reference_letters[i].upper())
            if var_token_id is None or ref_token_id is None:
                variant_probs_list.append(0.0)
                reference_probs_list.append(0.0)
                continue
            variant_probs_list.append(probs[i, mask_pos, var_token_id].item())
            reference_probs_list.append(probs[i, mask_pos, ref_token_id].item())

        return (
            np.array(variant_probs_list, dtype=np.float32),
            np.array(reference_probs_list, dtype=np.float32),
        )

    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model" in state:
            self.model.load_state_dict(state["model"], strict=False)
            print("Loaded model weights (including MLM head)")
        elif "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"], strict=False)
            print("Loaded model weights (including MLM head)")
        else:
            self.model.load_state_dict(state, strict=False)
            print("Loaded model weights (direct state_dict)")
        self.mlm_head_loaded = True

    def save_checkpoint(self, checkpoint_path: str, extra_state: dict = None):
        import os

        os.makedirs(
            os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".",
            exist_ok=True,
        )

        checkpoint = {"model": self.model.state_dict()}
        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    def get_tokenizer(self):
        return self.encode
