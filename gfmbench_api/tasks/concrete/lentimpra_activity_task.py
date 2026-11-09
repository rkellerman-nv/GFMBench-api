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

# Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
# - https://zenodo.org/records/8219231 — CC-BY-4.0 (Agarwal et al., Nature 2025,
#   doi:10.1038/s41586-024-08430-9; Table S3 library design, DOI 10.5281/zenodo.8219231)
# - https://www.encodeproject.org/help/citing-encode/ — ENCODE data use policy
#   (unrestricted use, cite the ENCODE Consortium; not a formal licence). Activity
#   files: ENCSR022GQD (HepG2), ENCSR382BVV (K562), ENCSR244FWB (WTC11).
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_regression_task import (
    BaseGFMSupervisedRegressionTask,
)
from gfmbench_api.utils.human_dna_local import load_manifest
from gfmbench_api.utils.preprocutils import truncate_sequence_from_ends

REQUIRED_COLUMNS = ("sequence", "activity")


class _LentiMPRAActivityTask(BaseGFMSupervisedRegressionTask):
    """Shared lentiMPRA regulatory-activity regression (one subclass per cell type).

    Predicts the activity of a 200 bp synthetic insert cloned upstream of a minimal
    promoter in a lentiviral MPRA (Agarwal et al., Nature 2025). The label is the
    mean over replicates of log2(RNA/DNA) (not mean-centred). Inserts are oligos,
    so sequences ship in the manifest (no reference genome). Both measured
    orientations of an element are separate examples (reversing the insert changes
    the construct, so no equal-prediction constraint applies); they share a
    ``group_id`` and the split is by held-out chromosome, so they never straddle
    splits. Controls are excluded from the manifests.
    """

    CELL_TYPE: str = ""
    INSERT_LENGTH = 200

    def _get_default_max_seq_len(self) -> int:
        return self.INSERT_LENGTH

    def _get_num_labels(self) -> int:
        return 1

    def is_spatial_binned(self) -> bool:
        return False

    def _manifest_config(self) -> str:
        return f"mpra_lenti/{self.CELL_TYPE}"

    def _get_task_data_dir_name(self) -> str:
        return "lentimpra"  # the three cell-type tasks share one data directory

    def _build_split(self, split: str) -> list:
        df = load_manifest(
            self.task_config,
            os.path.join(self.root_data_dir_path, self._get_task_data_dir_name()),
            hf_config=self._manifest_config(),
            split=split,
        )
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{self._manifest_config()}/{split}.parquet missing columns {missing}")
        if self.max_num_samples is not None and len(df) > self.max_num_samples:
            df = df.sample(n=self.max_num_samples, random_state=0).reset_index(drop=True)
        rows = []
        for seq, act in zip(df["sequence"], df["activity"]):
            seq = truncate_sequence_from_ends(str(seq), self.max_sequence_length)
            label = torch.tensor([float(act)], dtype=torch.float32)  # shape [1]
            rows.append((seq, label, np.array([])))
        return rows

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        logging.info("Loading %s manifests...", self.get_task_name())
        return (
            self._build_split("train"),
            self._build_split("validation"),
            self._build_split("test"),
        )

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None


class LentiMPRAHepG2Task(_LentiMPRAActivityTask):
    """lentiMPRA activity in HepG2 (200 bp insert -> mean log2 RNA/DNA)."""

    CELL_TYPE = "hepg2"

    def get_task_name(self) -> str:
        return "lentimpra_hepg2"


class LentiMPRAK562Task(_LentiMPRAActivityTask):
    """lentiMPRA activity in K562 (200 bp insert -> mean log2 RNA/DNA)."""

    CELL_TYPE = "k562"

    def get_task_name(self) -> str:
        return "lentimpra_k562"


class LentiMPRAWTC11Task(_LentiMPRAActivityTask):
    """lentiMPRA activity in WTC11 (200 bp insert -> mean log2 RNA/DNA)."""

    CELL_TYPE = "wtc11"

    def get_task_name(self) -> str:
        return "lentimpra_wtc11"
