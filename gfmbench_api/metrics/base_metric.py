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
import logging
from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    """
    Base class for all metrics.
    
    Usage pattern:
    1. Initialize the metric: metric = SomeMetric()
    2. Process batches: call metric.calc(batch_data) for each batch
    3. Get final result: score = metric.get_final_results()
    4. Reset for new run: metric.reset()
    
    The calc() method performs on-the-fly calculations per batch and stores only
    the minimal required intermediate results, making it memory-efficient.
    The get_final_results() method aggregates these intermediate results to compute
    the final metric score.
    
    Note: calc() automatically handles None inputs by printing a one-time warning
    and skipping that batch. Metrics with no valid data return None from get_final_results().
    """
    
    def __init__(self):
        """Initialize the metric. Subclasses can override to set up storage."""
        self.reset()
    
    def reset(self):
        """
        Reset the metric's internal state for a new evaluation run.
        
        Subclasses must override this to initialize their storage lists and must
        call super().reset() to properly initialize base class state (_none_warned flag).
        """
        self._none_warned = False
    
    def _state_names(self):
        """Names of the per-batch accumulator lists (every ``list`` attribute)."""
        return sorted(name for name, value in vars(self).items() if isinstance(value, list))

    def state_lengths(self):
        """Current length of every accumulator list, used as a marker for ``state_since``."""
        extra = sorted(n for n, v in vars(self).items() if not isinstance(v, list) and n != "_none_warned")
        if extra:
            raise TypeError(
                f"{self.name}: per-batch state export only supports list attributes, found {extra}"
            )
        return {name: len(getattr(self, name)) for name in self._state_names()}

    def state_since(self, lengths):
        """Accumulator entries appended after the ``state_lengths()`` marker was taken."""
        return {name: list(getattr(self, name)[start:]) for name, start in lengths.items()}

    def extend_state(self, delta):
        """Append entries exported by ``state_since`` (possibly from another process)."""
        names = set(self._state_names())
        unknown = set(delta) - names
        if unknown:
            raise KeyError(f"{self.name}: unknown metric state {sorted(unknown)}")
        for name, items in delta.items():
            getattr(self, name).extend(items)

    def calc(self, *args, **kwargs):
        """
        Template method: checks for None inputs and delegates to _calc_impl().
        
        This method handles None input checking and one-time warning, then
        delegates to the concrete implementation in _calc_impl().
        
        Args:
            *args: Variable positional arguments specific to each metric
            **kwargs: Variable keyword arguments specific to each metric
        """
        # Check if any required positional arguments are None
        if any(arg is None for arg in args):
            if not self._none_warned:
                logging.warning(f"Model did not provide required inputs for '{self.name}' metric. This metric will return None.")
                self._none_warned = True
            return
        
        # Delegate to concrete implementation
        self._calc_impl(*args, **kwargs)

    @staticmethod
    def _reshape_classification_inputs(probs, gt):
        """Return probabilities [samples, labels, classes] and labels [samples, labels]."""
        probs = np.asarray(probs)
        gt = np.asarray(gt)

        if gt.ndim == 1:
            if probs.ndim != 2:
                raise ValueError(
                    "Single-label probabilities must have shape [samples, classes]"
                )
            gt = gt[:, np.newaxis]
            probs = probs[:, np.newaxis, :]
        elif gt.ndim == 2:
            if probs.ndim == 2:
                # Binary multi-label probabilities omit the class dimension.
                probs = np.stack((1.0 - probs, probs), axis=-1)
            elif probs.ndim != 3:
                raise ValueError(
                    "Multi-label probabilities must have shape "
                    "[samples, labels, classes]"
                )

        return probs, gt
    
    @abstractmethod
    def _calc_impl(self, *args, **kwargs):
        """
        Concrete implementation of calc logic (implemented by subclasses).
        
        This method performs on-the-fly calculations per batch, converting inputs
        to numpy and computing intermediate values (e.g., predictions, scores).
        Only the minimal required information is stored for final aggregation.
        
        All inputs are expected to be numpy arrays (or convertible to numpy).
        All calculations are performed on CPU.
        Guaranteed to receive no None values (handled by base class).
        
        Args:
            *args: Variable positional arguments specific to each metric
            **kwargs: Variable keyword arguments specific to each metric
        """
        pass
    
    @abstractmethod
    def get_final_results(self):
        """
        Compute the final metric score from stored intermediate values.
        
        Must be called after calc() has been called at least once.
        Aggregates the intermediate values stored during calc() calls
        and computes the final metric.
        
        Returns:
            float or dict: The calculated metric score(s)
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """
        Return the metric name used as the key in results dictionaries.
        
        This allows tasks to dynamically construct result dictionaries
        without hardcoding key names.
        
        Returns:
            str: The key name for this metric in the results dictionary
                 (e.g., 'accuracy', 'mcc', 'auroc_all_probs_llr')
        """
        pass

