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
# - https://dbarchive.biosciencedbc.jp/data/fantom5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/hg38_fair+new_CAGE_peaks_phase1and2.bed.gz — CC-BY-4.0 (FANTOM5 © RIKEN licensed under CC Attribution 4.0 International)
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import BaseGFMSupervisedSingleSeqTask
from gfmbench_api.utils.human_dna_local import extract_window, load_manifest, open_reference_genome
from gfmbench_api.utils.preprocutils import truncate_sequence_from_ends

TSS_STRAND_HF_CONFIG = "tss_strand"
TSS_STRAND_WINDOW = 511


class Fantom5TSSStrandTask(BaseGFMSupervisedSingleSeqTask):
    """FANTOM5 TSS strand prediction (binary: plus=0 vs minus=1).

    Inputs are reference-orientation windows centred on the summit of confidently
    unidirectional CAGE peaks (hg38). The input is never reverse-complemented by the
    target strand. Splits are by held-out chromosome.
    """

    def _get_default_max_seq_len(self) -> int:
        return TSS_STRAND_WINDOW

    def _get_num_labels(self) -> int:
        return 1

    def _get_num_classes(self) -> int:
        return 2

    def get_task_name(self) -> str:
        return "fantom5_tss_strand"

    def _build_split(self, genome, manifest: pd.DataFrame) -> list:
        if self.max_num_samples is not None:
            # Seeded random subset: manifests are not guaranteed to be strand-mixed in file order.
            if len(manifest) > self.max_num_samples:
                manifest = manifest.sample(n=self.max_num_samples, random_state=0)
        rows, dropped = [], 0
        for chrom, center, label in zip(manifest["chrom"], manifest["center"], manifest["label"]):
            seq = extract_window(genome, chrom, int(center), TSS_STRAND_WINDOW)
            if seq is None:
                dropped += 1
                continue
            seq = truncate_sequence_from_ends(seq, self.max_sequence_length)
            rows.append((seq, torch.tensor(int(label), dtype=torch.long), np.array([])))
        if dropped:
            logging.warning("%s: dropped %d rows whose window could not be extracted", self.get_task_name(), dropped)
        return rows

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        data_dir = os.path.join(self.root_data_dir_path, self._get_task_data_dir_name())
        genome = open_reference_genome(self.root_data_dir_path, "hg38")
        splits = []
        for split in ("train", "validation", "test"):
            manifest = load_manifest(self.task_config, data_dir, TSS_STRAND_HF_CONFIG, split)
            splits.append(self._build_split(genome, manifest))
        return splits[0], splits[1], splits[2]

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        """Return None as this task has no conditional metadata inputs."""
        return None
