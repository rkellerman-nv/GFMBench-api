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
from typing import Any, Dict, List, Optional

import numpy as np

from gfmbench_api.tasks.base.base_gfm_zero_shot_task import BaseGFMZeroShotTask


class BaseGFMZeroShotGeneralIndelTask(BaseGFMZeroShotTask):
    """
    Base class for zero-shot tasks for general indel variants.
    No fine-tuning is performed - evaluates model's pre-trained capabilities.
    
    This class handles datasets with any variant types (insertions, deletions, etc.)
    and uses the common zero-shot metrics plus the affected-region metrics (no SNV-position
    metrics, since an indel has no single variant position). The affected region is found by
    diffing the variant and reference tokenizations, so it needs no position.
    
    Subclasses must implement:
        - _create_test_dataset(): Return test dataset
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
    """

    def _get_additional_metrics(self) -> List[tuple]:
        """Return the affected-region metrics (the only additional metrics for general indel tasks)."""
        return self._get_affected_region_metrics()

    def _update_additional_metrics(
        self,
        additional_metrics: List[tuple],
        model: Any,
        variant_sequences: List[str],
        reference_sequences: List[str],
        labels_np: np.ndarray,
        common_outputs: Dict[str, Optional[np.ndarray]]
    ) -> None:
        """Update affected-region metrics for a batch."""
        self._add_affected_region_outputs(model, variant_sequences, reference_sequences, common_outputs)
        for metric, arg_keys in additional_metrics:
            metric.calc(*[common_outputs[key] for key in arg_keys])

    def _is_snv_only(self) -> bool:
        """Return False since this handles general indel variants, not just SNV."""
        return False

