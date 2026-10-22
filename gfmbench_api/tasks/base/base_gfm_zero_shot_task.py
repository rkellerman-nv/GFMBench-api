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
import os
from abc import abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gfmbench_api.metrics import (
    AffectedContextExtentMetric,
    AffectedContextProfileMetric,
    AffectedRegionCosineSimAUPRC,
    AffectedRegionCosineSimAUROC,
    AffectedRegionL2AUPRC,
    AffectedRegionL2AUROC,
    AffectedRegionLengthAUROC,
    AffectedRegionMeanLength,
    AffectedRegionSizeStratifiedAUROC,
    AffectedRegionWindowMetric,
    ContextLikelihoodMeanTokens,
    ContextLikelihoodMetric,
    ForcedSegmentationFraction,
    ForcedSegmentationRegionMetric,
    SegmentationPhaseMetric,
    SequenceEmbeddingsCosineSimAUPRC,
    SequenceEmbeddingsCosineSimAUROC,
    SequenceEmbeddingsL2AUPRC,
    SequenceEmbeddingsL2AUROC,
    SumProbsLLRAUPRC,
    SumProbsLLRAUROC,
    TokenizerStatisticAUROC,
)
from gfmbench_api.metrics.affected_region_extensions import context_positions, context_profile
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask
from gfmbench_api.utils.caching_utils import SequenceInferenceCache

# Environment variables that switch on the optional affected-region metric groups, and the token
# frequency table (an .npy of counts per token ID) used as the tokenizer prior by the "tokstat" group
REGION_GROUPS_ENV = "GFMBENCH_REGION_GROUPS"
TOKEN_FREQUENCY_ENV = "GFMBENCH_TOKEN_FREQ"


class BaseGFMZeroShotTask(BaseGFMTask):
    """
    Base class for zero-shot variant effect prediction tasks.
    No fine-tuning is performed - evaluates model's pre-trained capabilities.
    
    Dataset format: (variant_sequence, reference_sequence, label, conditional_input) tuples
    
    Subclasses must implement:
        - _get_additional_metrics(): Return list of additional metric objects
        - _update_additional_metrics(): Update additional metrics for a batch
        - _is_snv_only(): Return whether this is an SNV-only task
        - _create_test_dataset(): Return test dataset
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
        - get_conditional_input_meta_data_frame(): Return metadata schema or None
        - use_reference_cache(): Return whether to cache reference infer_sequence_to_sequence calls
    """
    
    # Context widths (unchanged tokens on each side of the affected region) evaluated by the
    # affected-region metrics
    AFFECTED_REGION_CONTEXT_TOKENS = (0, 1, 2, 4, 8)
    # Window widths in bases (unchanged DNA on each side of the affected region). Comparable
    # across tokenizers, unlike a width in tokens. Pooled: embeddings are pooled, then compared;
    # distance summaries: aligned token distances, then the max over unchanged neighbours / the mean.
    AFFECTED_REGION_POOLED_WINDOW_BASES = (0, 10, 25, 50, 100)
    AFFECTED_REGION_DISTANCE_SUMMARY_WINDOW_BASES = (10, 50)
    # Context widths whose scores are also reported within strata of region size
    AFFECTED_REGION_SIZE_STRATIFIED_CONTEXT_TOKENS = (0, 2)

    # Optional groups of affected-region metrics beyond the "core" ones above (see
    # gfmbench_api/metrics/affected_region_extensions.py). Enabled with the environment variable
    # GFMBENCH_REGION_GROUPS, a comma-separated list of names or "all"; unset means "core" only,
    # so a run costs what it did before these groups existed. "none" runs no region metrics.
    #   forced      variant embedded with the reference's token boundaries (one extra forward pass)
    #   tokstat     model-free tokenizer statistics (no model call)
    #   phase       region score after trimming 1-3 bases from the window start (3 extra rounds
    #               of variant and reference forward passes)
    #   profile     embedding change of the unchanged tokens around the region (no model call)
    #   context_llr masked-LM likelihood of the neighbouring unchanged tokens (many small forward passes)
    REGION_METRIC_GROUPS = ("core", "forced", "tokstat", "phase", "profile", "context_llr")
    FORCED_SEGMENTATION_CONTEXT_TOKENS = (0, 2)
    # Bins (bases from the region edge) of the context profile
    CONTEXT_PROFILE_BINS = ((0, 10), (10, 25), (25, 50), (50, 100))
    # Context-likelihood neighbours: the nearest tokens starting within this many bases, at most this many per side
    CONTEXT_LLR_MAX_GAP_BASES = 25
    CONTEXT_LLR_TOKENS_PER_SIDE = 6

    @classmethod
    def _enabled_region_groups(cls) -> List[str]:
        """Region metric groups requested through GFMBENCH_REGION_GROUPS (default: core only)."""
        raw = os.environ.get(REGION_GROUPS_ENV, "core").strip()
        if raw == "none":
            return []
        names = list(cls.REGION_METRIC_GROUPS) if raw == "all" else [n.strip() for n in raw.split(",") if n.strip()]
        unknown = [n for n in names if n not in cls.REGION_METRIC_GROUPS]
        if unknown:
            raise ValueError(
                f"unknown region metric group(s) {unknown} in {REGION_GROUPS_ENV}; "
                f"choose from {list(cls.REGION_METRIC_GROUPS)}, 'all' or 'none'"
            )
        return names

    def _get_affected_region_metrics(self) -> List[tuple]:
        """Affected-region metrics of the enabled groups, as (metric, arg_keys)."""
        metrics = []
        groups = self._enabled_region_groups()
        if "core" in groups:
            metrics += self._get_core_region_metrics()
        if "forced" in groups:
            metrics += self._get_forced_segmentation_metrics()
        if "tokstat" in groups:
            metrics += self._get_tokenizer_statistics_metrics()
        if "phase" in groups:
            metrics += self._get_segmentation_phase_metrics()
        if "profile" in groups:
            metrics += self._get_context_profile_metrics()
        if "context_llr" in groups:
            metrics += self._get_context_likelihood_metrics()
        return metrics

    def _get_forced_segmentation_metrics(self) -> List[tuple]:
        metrics = []
        for k in self.FORCED_SEGMENTATION_CONTEXT_TOKENS:
            keys = ('forced_variant_embeddings', 'reference_embeddings', f'forced_spans_k{k}', 'labels')
            for distance in ('cosinesim', 'l2'):
                for score in ('auroc', 'auprc'):
                    metrics.append((ForcedSegmentationRegionMetric(k, distance, score), keys))
        for kind in ForcedSegmentationFraction.KINDS:
            metrics.append((ForcedSegmentationFraction(kind), ('forced_spans_k0', 'affected_spans_k0')))
        return metrics

    def _get_tokenizer_statistics_metrics(self) -> List[tuple]:
        prior_path = os.environ.get(TOKEN_FREQUENCY_ENV) or None
        keys = ('variant_token_ids', 'reference_token_ids', 'affected_spans_k0',
                'variant_token_offsets', 'reference_token_offsets', 'labels')
        return [(TokenizerStatisticAUROC(statistic, prior_path), keys) for statistic in TokenizerStatisticAUROC.STATISTICS]

    def _get_segmentation_phase_metrics(self) -> List[tuple]:
        keys = ('phase_variant_embeddings', 'phase_reference_embeddings', 'phase_spans', 'labels')
        metrics = []
        for distance in ('cosinesim', 'l2'):
            for score in ('auroc', 'auprc'):
                metrics.append((SegmentationPhaseMetric('mean', distance, score), keys))
            metrics.append((SegmentationPhaseMetric('spread', distance, 'auroc'), keys))
            for shift in SegmentationPhaseMetric.SHIFTS:
                metrics.append((SegmentationPhaseMetric(f'shift{shift}', distance, 'auroc'), keys))
        metrics.append((SegmentationPhaseMetric('unstable_fraction'), keys))
        return metrics

    def _get_context_profile_metrics(self) -> List[tuple]:
        keys = ('context_profiles', 'labels')
        metrics = []
        for side in ('down', 'up'):
            for lo, hi in self.CONTEXT_PROFILE_BINS:
                for distance in ('cosinesim', 'l2'):
                    metrics.append((AffectedContextProfileMetric(side, lo, hi, distance, 'auroc'), keys))
                metrics.append((AffectedContextProfileMetric(side, lo, hi, 'cosinesim', 'mean'), keys))
            for kind in ('reach', 'decay'):
                for distance in ('cosinesim', 'l2'):
                    metrics.append((AffectedContextExtentMetric(side, kind, distance), keys))
        return metrics

    def _get_context_likelihood_metrics(self) -> List[tuple]:
        keys = ('context_delta', 'context_bases', 'labels')
        metrics = [(ContextLikelihoodMetric(side, normalize, 'auroc'), keys)
                   for normalize in ('sum', 'perbase') for side in ('down', 'up', 'both')]
        metrics += [(ContextLikelihoodMetric('both', normalize, 'auprc'), keys) for normalize in ('sum', 'perbase')]
        metrics.append((ContextLikelihoodMeanTokens(), keys))
        return metrics

    def _get_core_region_metrics(self) -> List[tuple]:
        """The core affected-region metrics (mutation tokens + changed downstream tokens), as (metric, arg_keys)."""
        metrics = []
        for k in self.AFFECTED_REGION_CONTEXT_TOKENS:
            keys = ('variant_embeddings', 'reference_embeddings', f'affected_spans_k{k}', 'labels')
            for metric_cls in (AffectedRegionCosineSimAUROC, AffectedRegionCosineSimAUPRC,
                               AffectedRegionL2AUROC, AffectedRegionL2AUPRC):
                metrics.append((metric_cls(context_tokens=k), keys))
        # Region-size diagnostics (always on k=0 spans): label-leakage baseline and mean size
        length_keys = ('variant_embeddings', 'reference_embeddings', 'affected_spans_k0', 'labels')
        metrics.append((AffectedRegionLengthAUROC(), length_keys))
        metrics.append((AffectedRegionMeanLength(), length_keys))
        # AUROC within strata of region size (strata always come from the k=0 spans)
        for k in self.AFFECTED_REGION_SIZE_STRATIFIED_CONTEXT_TOKENS:
            keys = ('variant_embeddings', 'reference_embeddings', f'affected_spans_k{k}', 'affected_spans_k0', 'labels')
            for distance in ('cosinesim', 'l2'):
                metrics.append((AffectedRegionSizeStratifiedAUROC(context_tokens=k, distance=distance), keys))
        # Windows in bases: need the k=0 spans, where the unchanged suffix starts, and each token's base offsets
        window_keys = ('variant_embeddings', 'reference_embeddings', 'affected_spans_k0', 'affected_suffix_starts',
                       'variant_token_offsets', 'reference_token_offsets', 'labels')
        configs = [(w, 'pooled') for w in self.AFFECTED_REGION_POOLED_WINDOW_BASES]
        configs += [(w, summary) for w in self.AFFECTED_REGION_DISTANCE_SUMMARY_WINDOW_BASES
                    for summary in ('flankmax', 'mean')]
        for window_bases, summary in configs:
            for distance in ('cosinesim', 'l2'):
                for score in ('auroc', 'auprc'):
                    metrics.append((AffectedRegionWindowMetric(window_bases, distance, summary, score), window_keys))
        return metrics

    def _add_affected_region_outputs(
        self,
        model: Any,
        variant_sequences: List[str],
        reference_sequences: List[str],
        outputs: Dict[str, Optional[np.ndarray]],
    ) -> None:
        """Ask the model for the affected token spans of each context width; None if unsupported."""
        # k=0 is always requested: the region-size diagnostics are defined on it
        for k in sorted({0, *self.AFFECTED_REGION_CONTEXT_TOKENS}):
            outputs[f'affected_spans_k{k}'], = self._safe_model_call(
                model, 'get_affected_token_spans', variant_sequences, reference_sequences, k, num_outputs=1
            )
        outputs['affected_suffix_starts'], = self._safe_model_call(
            model, 'get_affected_suffix_starts', variant_sequences, reference_sequences, num_outputs=1
        )
        outputs['variant_token_offsets'], = self._safe_model_call(
            model, 'get_token_base_offsets', variant_sequences, num_outputs=1
        )
        outputs['reference_token_offsets'], = self._safe_model_call(
            model, 'get_token_base_offsets', reference_sequences, num_outputs=1
        )
        self._add_region_group_outputs(model, variant_sequences, reference_sequences, outputs)

    def _add_region_group_outputs(
        self,
        model: Any,
        variant_sequences: List[str],
        reference_sequences: List[str],
        outputs: Dict[str, Optional[np.ndarray]],
    ) -> None:
        """Model outputs the enabled optional region groups need; None where the model lacks a method."""
        groups = self._enabled_region_groups()
        if "forced" in groups:
            outputs['forced_variant_embeddings'], = self._safe_model_call(
                model, 'infer_forced_segmentation_embeddings', variant_sequences, reference_sequences, num_outputs=1
            )
            for k in self.FORCED_SEGMENTATION_CONTEXT_TOKENS:
                outputs[f'forced_spans_k{k}'], = self._safe_model_call(
                    model, 'get_forced_segmentation_spans', variant_sequences, reference_sequences, k, num_outputs=1
                )
            if 0 not in self.FORCED_SEGMENTATION_CONTEXT_TOKENS:
                outputs['forced_spans_k0'], = self._safe_model_call(
                    model, 'get_forced_segmentation_spans', variant_sequences, reference_sequences, 0, num_outputs=1
                )
        if "tokstat" in groups:
            outputs['variant_token_ids'], = self._safe_model_call(model, 'get_token_ids', variant_sequences, num_outputs=1)
            outputs['reference_token_ids'], = self._safe_model_call(model, 'get_token_ids', reference_sequences, num_outputs=1)
        if "phase" in groups:
            self._add_phase_outputs(model, variant_sequences, reference_sequences, outputs)
        if "profile" in groups:
            outputs['context_profiles'] = self._context_profiles(outputs)
        if "context_llr" in groups:
            self._add_context_likelihood_outputs(model, variant_sequences, reference_sequences, outputs)

    def _add_phase_outputs(self, model, variant_sequences, reference_sequences, outputs) -> None:
        """Embeddings and k=0 spans after trimming each shift's bases from the start of both sequences."""
        variant_embeddings, reference_embeddings, spans = [], [], []
        for shift in SegmentationPhaseMetric.SHIFTS:
            if shift == 0:
                variant_embeddings.append(outputs['variant_embeddings'])
                reference_embeddings.append(outputs['reference_embeddings'])
                spans.append(outputs['affected_spans_k0'])
                continue
            shifted_variants = [seq[shift:] for seq in variant_sequences]
            shifted_references = [seq[shift:] for seq in reference_sequences]
            _, var_emb, _ = self._safe_model_call(model, 'infer_sequence_to_sequence', shifted_variants, None, num_outputs=3)
            _, ref_emb, _ = self._safe_model_call(model, 'infer_sequence_to_sequence', shifted_references, None, num_outputs=3)
            span, = self._safe_model_call(
                model, 'get_affected_token_spans', shifted_variants, shifted_references, 0, num_outputs=1
            )
            variant_embeddings.append(var_emb)
            reference_embeddings.append(ref_emb)
            spans.append(span)
        complete = all(x is not None for group in (variant_embeddings, reference_embeddings, spans) for x in group)
        outputs['phase_variant_embeddings'] = variant_embeddings if complete else None
        outputs['phase_reference_embeddings'] = reference_embeddings if complete else None
        outputs['phase_spans'] = spans if complete else None

    @staticmethod
    def _context_profiles(outputs) -> Optional[list]:
        """One context_profile per variant, or None if the model lacks what it needs."""
        needed = ('variant_embeddings', 'reference_embeddings', 'affected_spans_k0', 'affected_suffix_starts',
                  'variant_token_offsets', 'reference_token_offsets')
        if any(outputs.get(key) is None for key in needed):
            return None
        spans, suffix = outputs['affected_spans_k0'], outputs['affected_suffix_starts']
        profiles = []
        for i in range(len(spans)):
            if spans[i, 0] < 0:
                profiles.append(None)
                continue
            profiles.append(context_profile(
                outputs['variant_embeddings'][i], outputs['reference_embeddings'][i],
                outputs['variant_token_offsets'][i], outputs['reference_token_offsets'][i],
                (int(spans[i, 0]), int(spans[i, 1])), (int(spans[i, 2]), int(spans[i, 3])),
                (int(suffix[i, 0]), int(suffix[i, 1])),
            ))
        return profiles

    def _add_context_likelihood_outputs(self, model, variant_sequences, reference_sequences, outputs) -> None:
        """log p_ref - log p_variant of the paired neighbouring tokens, each masked on its own."""
        outputs['context_delta'] = outputs['context_bases'] = None
        needed = ('affected_spans_k0', 'affected_suffix_starts', 'variant_token_offsets', 'reference_token_offsets')
        if any(outputs.get(key) is None for key in needed):
            return
        spans, suffix = outputs['affected_spans_k0'], outputs['affected_suffix_starts']
        width = 2 * self.CONTEXT_LLR_TOKENS_PER_SIDE
        var_pos = np.full((len(spans), width), -1, dtype=np.int64)
        ref_pos = np.full((len(spans), width), -1, dtype=np.int64)
        bases = np.full((len(spans), width), -1, dtype=np.int64)
        for i in range(len(spans)):
            if spans[i, 0] < 0:
                continue
            var_pos[i], ref_pos[i], bases[i] = context_positions(
                (int(spans[i, 0]), int(spans[i, 1])), (int(spans[i, 2]), int(spans[i, 3])),
                (int(suffix[i, 0]), int(suffix[i, 1])),
                outputs['variant_token_offsets'][i], outputs['reference_token_offsets'][i],
                self.CONTEXT_LLR_MAX_GAP_BASES, self.CONTEXT_LLR_TOKENS_PER_SIDE,
            )
        variant_logprobs, = self._safe_model_call(model, 'infer_masked_token_logprobs', variant_sequences, var_pos, num_outputs=1)
        reference_logprobs, = self._safe_model_call(model, 'infer_masked_token_logprobs', reference_sequences, ref_pos, num_outputs=1)
        if variant_logprobs is None or reference_logprobs is None:
            return
        outputs['context_delta'] = (reference_logprobs - variant_logprobs).astype(np.float64)
        outputs['context_bases'] = bases

    def __init__(self, root_data_dir_path: str,
                 task_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize zero-shot variant task.
        
        Args:
            root_data_dir_path: path to root data directory
            task_config: optional configuration dictionary with user's settings
        """
        super().__init__(root_data_dir_path, task_config)

    def get_finetune_dataset(self) -> Optional[Dataset]:
        """Return None (zero-shot tasks don't require fine-tuning)."""
        return None

    def get_task_attributes(self) -> Dict[str, Any]:
        """Return task attributes for zero-shot variant tasks."""
        return {
            "has_finetuning_data": False,
            "has_validation_data": False,
            "is_variant_effect_prediction": True,
            "is_snv_only_variants": self._is_snv_only(),
            "conditional_input_metadata": self.get_conditional_input_meta_data_frame()
        }

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        """Create datasets for zero-shot task (test only, no training or validation)."""
        test_dataset = self._create_test_dataset()
        return None, None, test_dataset

    def _eval_dataset(self, model: Any, dataset: Any) -> Dict[str, Optional[float]]:
        """
        Evaluate the model using zero-shot methods for variant pathogenicity prediction.
        Binary classification task (benign=0, pathogenic=1).
        
        Common metrics (defined in base class):
        - LLR-based (using sequence probabilities): auroc_all_probs_llr, auprc_all_probs_llr
        - Cosine similarity on sequence embeddings: sequence_embeddings_cosinesim_auroc/auprc
        
        Additional metrics are provided by subclasses via _get_additional_metrics().
        
        Args:
            model: Any object with inference methods (should be in eval mode).
            dataset: The dataset to evaluate on.
        
        Returns:
            dict: Scores with metric names as keys.
        """
        # Create dataloader from dataset
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        # Define common metrics with their argument keys
        # Each entry: (metric, arg_keys) where arg_keys specifies which outputs to pass
        common_metrics = [
            (SumProbsLLRAUROC(), ('variant_probs', 'reference_probs', 'labels')),
            (SumProbsLLRAUPRC(), ('variant_probs', 'reference_probs', 'labels')),
            (SequenceEmbeddingsCosineSimAUROC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsCosineSimAUPRC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsL2AUROC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsL2AUPRC(), ('variant_repr', 'reference_repr', 'labels')),
        ]
        
        # Get additional metrics from subclass
        additional_metrics = self._get_additional_metrics()

        ref_cache = SequenceInferenceCache(max_size_gb=self.cache_size)
        for batch_data in tqdm(data_loader, desc="Evaluating (Zero-Shot)"):
            variant_sequences, reference_sequences, labels, conditional_input = batch_data

            # Get sequence-to-sequence outputs for both variant and reference
            # Returns: (sequence_probs, sequence_embeddings, sequence_representative)
            variant_probs_np, variant_embeddings_np, variant_repr_np = self._safe_model_call(
                model, 'infer_sequence_to_sequence', variant_sequences, conditional_input, num_outputs=3
            )
            infer_ref = partial(
                self._safe_model_call, model, 'infer_sequence_to_sequence', num_outputs=3
            )
            reference_probs_np, reference_embeddings_np, reference_repr_np = ref_cache.cached_call(
                infer_ref,
                reference_sequences,
                conditional_input,
                disable=(self.cache_size == 0) or not self.use_reference_cache(),
            )

            # Build outputs dict for metric argument lookup
            outputs = {
                'variant_probs': variant_probs_np,
                'reference_probs': reference_probs_np,
                'variant_embeddings': variant_embeddings_np,
                'reference_embeddings': reference_embeddings_np,
                'variant_repr': variant_repr_np,
                'reference_repr': reference_repr_np,
                'labels': labels,
            }

            # Update common metrics using their specified argument keys
            for metric, arg_keys in common_metrics:
                args = [outputs[key] for key in arg_keys]
                metric.calc(*args)

            # Update additional metrics (subclass handles this)
            self._update_additional_metrics(
                additional_metrics, model,
                variant_sequences, reference_sequences, labels,
                outputs
            )

        ref_cache.clear()

        # Aggregate results from all metrics
        results = {}
        for metric, _ in common_metrics:
            results[metric.name] = metric.get_final_results()
        for metric, _ in additional_metrics:
            results[metric.name] = metric.get_final_results()
        
        return results

    @abstractmethod
    def use_reference_cache(self) -> bool:
        """Return True to cache reference infer_sequence_to_sequence calls during eval.

        Disable when reference sequences rarely repeat (low cache hit rate); caching
        then adds lookup overhead without saving inference work.
        """
        pass

    @abstractmethod
    def _get_additional_metrics(self) -> List[tuple]:
        """
        Subclasses must implement this: return list of additional metrics with arg_keys.
        
        Each entry is a tuple: (metric, arg_keys)
        arg_keys specifies which outputs to pass to metric.calc()
        
        Returns:
            List of (metric, arg_keys) tuples (can be empty)
        """
        pass

    @abstractmethod
    def _update_additional_metrics(
        self,
        additional_metrics: List[Any],
        model: Any,
        variant_sequences: List[str],
        reference_sequences: List[str],
        labels_np: np.ndarray,
        common_outputs: Dict[str, Optional[np.ndarray]]
    ) -> None:
        """
        Subclasses must implement this: update additional metrics for a batch.
        
        This method is called once per batch. Subclasses can make additional
        model calls if needed (e.g., for SNV-specific metrics).
        
        Args:
            additional_metrics: List of metric objects from _get_additional_metrics()
            model: Model instance for additional inference calls
            variant_sequences: Batch of variant sequences
            reference_sequences: Batch of reference sequences
            labels_np: Labels as numpy array
            common_outputs: Dict with common model outputs (probs, embeddings, repr)
        """
        pass

    @abstractmethod
    def _is_snv_only(self) -> bool:
        """Subclasses must implement this: return whether this is an SNV-only task."""
        pass

    @abstractmethod
    def _create_test_dataset(self) -> Dataset:
        """
        Subclasses must implement this: create and return test dataset.
        The dataset must yield (variant_sequences, reference_sequences, labels, conditional_input) tuples.
        conditional_input should be None for tasks without metadata.
        
        Returns:
            Dataset: test dataset with (variant_sequences, reference_sequences, labels, conditional_input) tuples
        """
        pass

