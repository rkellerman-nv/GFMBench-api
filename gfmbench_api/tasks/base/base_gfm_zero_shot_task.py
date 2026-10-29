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
from abc import abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from gfmbench_api.metrics import (
    SequenceEmbeddingsCosineSimAUPRC,
    SequenceEmbeddingsCosineSimAUROC,
    SequenceEmbeddingsL2AUPRC,
    SequenceEmbeddingsL2AUROC,
    SumProbsLLRAUPRC,
    SumProbsLLRAUROC
)
from gfmbench_api.tasks.base.base_gfm_task import BaseGFMTask
from gfmbench_api.utils.caching_utils import SequenceInferenceCache
from gfmbench_api.utils.sharding import (
    ShardSpec,
    dataset_fingerprint,
    load_all_shards,
    num_batches,
    shard_batch_ids,
    shard_sample_indices,
    write_shard,
)


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

    def eval_test_set(self, model: Any) -> Optional[Dict[str, Optional[float]]]:
        """Evaluate on the test set; in sharded mode only this shard's batches are run.

        Returns the scores, or ``None`` when sharded and other shards are still running.
        Normally only the last shard to finish merges and returns the scores; if several
        finish together they may all return the same (identical) scores.
        """
        return self._eval_dataset(model, self.test_dataset, shard=self.shard)

    def _get_prefetch_variant_pos(self) -> Optional[int]:
        """Variant position to hand to the model's prefetch hook (None for non-SNV tasks)."""
        return None

    def _prefetch_batch(self, model: Any, batch_data: Any) -> None:
        """Offer an upcoming batch to the model's optional ``prefetch_batch`` hook."""
        prefetch = getattr(model, "prefetch_batch", None)
        if prefetch is None:
            return
        variant_sequences, reference_sequences = batch_data[0], batch_data[1]
        try:
            prefetch(variant_sequences, reference_sequences, self._get_prefetch_variant_pos())
        except Exception:
            pass  # prefetching is an optimization; the inline path still computes everything

    def _eval_dataset(
        self, model: Any, dataset: Any, shard: Optional[ShardSpec] = None
    ) -> Optional[Dict[str, Optional[float]]]:
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
            shard: If given, evaluate only this shard's batches and merge through files
                (see utils/sharding.py).
        
        Returns:
            dict: Scores with metric names as keys; ``None`` for a shard that is not the
            last one to finish.
        """
        # Create dataloader from dataset (in sharded mode: only this shard's whole batches)
        shard_batches: List[int] = []
        loader_dataset = dataset
        if shard is not None:
            shard_batches = shard_batch_ids(len(dataset), self.batch_size, shard)
            loader_dataset = Subset(
                dataset, shard_sample_indices(len(dataset), self.batch_size, shard)
            )
        data_loader = DataLoader(
            loader_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        common_metrics = self._common_metrics()
        
        # Get additional metrics from subclass
        additional_metrics = self._get_additional_metrics()

        all_metrics = [metric for metric, _ in common_metrics + additional_metrics]
        shard_records: Dict[int, Dict[str, Any]] = {}

        ref_cache = SequenceInferenceCache(max_size_gb=self.cache_size)
        batches = iter(data_loader)
        batch_data = next(batches, None)
        if batch_data is not None and self.prefetch:
            self._prefetch_batch(model, batch_data)
        progress = tqdm(total=len(data_loader), desc="Evaluating (Zero-Shot)")
        batch_pos = 0
        try:
            while batch_data is not None:
                # Hand the next batch to the model first so its host-side preprocessing
                # overlaps with the GPU work for the current batch.
                upcoming = next(batches, None)
                if upcoming is not None and self.prefetch:
                    self._prefetch_batch(model, upcoming)
                marks = [metric.state_lengths() for metric in all_metrics] if shard is not None else None

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

                if shard is not None:
                    shard_records[shard_batches[batch_pos]] = {
                        metric.name: metric.state_since(mark)
                        for metric, mark in zip(all_metrics, marks)
                    }
                batch_pos += 1
                progress.update(1)
                batch_data = upcoming
        finally:
            progress.close()
            ref_cache.clear()

        if shard is not None:
            return self._finish_shard(shard, dataset, shard_records, common_metrics, additional_metrics)

        # Aggregate results from all metrics
        results = {}
        for metric, _ in common_metrics:
            results[metric.name] = metric.get_final_results()
        for metric, _ in additional_metrics:
            results[metric.name] = metric.get_final_results()
        
        return results

    @staticmethod
    def _common_metrics() -> List[tuple]:
        """Fresh metric objects shared by all zero-shot tasks, with their argument keys.

        Each entry is (metric, arg_keys) where arg_keys specifies which outputs to pass.
        """
        return [
            (SumProbsLLRAUROC(), ('variant_probs', 'reference_probs', 'labels')),
            (SumProbsLLRAUPRC(), ('variant_probs', 'reference_probs', 'labels')),
            (SequenceEmbeddingsCosineSimAUROC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsCosineSimAUPRC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsL2AUROC(), ('variant_repr', 'reference_repr', 'labels')),
            (SequenceEmbeddingsL2AUPRC(), ('variant_repr', 'reference_repr', 'labels')),
        ]

    def _finish_shard(
        self,
        shard: ShardSpec,
        dataset: Any,
        shard_records: Dict[int, Dict[str, Any]],
        common_metrics: List[tuple],
        additional_metrics: List[tuple],
    ) -> Optional[Dict[str, Optional[float]]]:
        """Publish this shard's metric state; if every shard is done, merge and score."""
        write_shard(shard, self.get_task_name(), {
            "fingerprint": dataset_fingerprint(dataset),
            "num_samples": len(dataset),
            "batch_size": self.batch_size,
            "batches": shard_records,
        })
        shards = load_all_shards(shard, self.get_task_name())
        if shards is None:
            return None

        expected = {key: shards[0][key] for key in ("fingerprint", "num_samples", "batch_size")}
        for payload in shards:
            if any(payload[key] != value for key, value in expected.items()):
                raise RuntimeError(
                    f"Shards of task '{self.get_task_name()}' were produced from different "
                    "datasets or batch sizes; clear the shard directory and rerun."
                )
        merged: Dict[int, Dict[str, Any]] = {}
        for payload in shards:
            if merged.keys() & payload["batches"].keys():
                raise RuntimeError(f"Duplicate batches across shards of '{self.get_task_name()}'.")
            merged.update(payload["batches"])
        if sorted(merged) != list(range(num_batches(len(dataset), self.batch_size))):
            raise RuntimeError(f"Shards of '{self.get_task_name()}' do not cover every batch.")

        # Fresh metrics fed with the shard states in original batch order.
        results_metrics = [metric for metric, _ in self._common_metrics() + self._get_additional_metrics()]
        for batch_id in sorted(merged):
            for metric in results_metrics:
                metric.extend_state(merged[batch_id][metric.name])
        return {metric.name: metric.get_final_results() for metric in results_metrics}

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

