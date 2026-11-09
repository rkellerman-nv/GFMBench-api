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
# - https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63874/suppl/ — NCBI GEO GSE63874 (G4-seq, Chambers et al., Nat Biotechnol 2015, doi:10.1038/nbt.3295); licence unverified: "NCBI GEO: files may be freely downloaded and reproduced unless otherwise stated"
# - https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz — LicenseRef-UCSC-Genome-Browser
"""Strand-resolved G-quadruplex (G4-seq) multi-label classification.

Each example is a 511 bp hg19 window; the two labels are
[G4 structure on the plus strand, G4 structure on the minus strand]. Labels come
from G4-seq polymerase-stalling hits (K+ condition by default) that overlap the
central region of the window. NOTE: in the GEO files "*_plus_*" means the read
aligned to the forward strand, i.e. the G4 sits on the REVERSE strand; the builder
(scripts/build_human_dna_tasks/build_g4seq_strand.py) inverts this before writing
the manifest, so manifest labels are already "G4 on the plus/minus strand".
Sequences are not shipped: they are cut from hg19 at load time.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gfmbench_api.tasks.base.base_gfm_supervised_single_seq_task import (
    BaseGFMSupervisedSingleSeqTask,
)
from gfmbench_api.utils.human_dna_local import (
    extract_window,
    load_manifest,
    open_reference_genome,
)
from gfmbench_api.utils.preprocutils import truncate_sequence_from_ends

G4SEQ_HF_CONFIG = "g4seq_strand"
G4SEQ_ASSEMBLY = "hg19"
G4SEQ_WINDOW_BP = 511


class G4SeqStrandTask(BaseGFMSupervisedSingleSeqTask):
    """Strand-resolved G4 (representative negatives; test split = 'test')."""

    TEST_SPLIT_FILE = "test"
    TASK_NAME = "g4seq_strand"

    def _get_default_max_seq_len(self) -> int:
        return G4SEQ_WINDOW_BP

    def _get_num_labels(self) -> int:
        return 2  # [G4 on plus strand, G4 on minus strand]

    def _get_num_classes(self) -> int:
        return 2

    def get_task_name(self) -> str:
        return self.TASK_NAME

    def _get_task_data_dir_name(self) -> str:
        return G4SEQ_HF_CONFIG  # matched variant shares this directory

    def _build_split(self, manifest: pd.DataFrame, genome) -> list:
        dataset = []
        for row in manifest.itertuples(index=False):
            seq = extract_window(genome, row.chrom, int(row.center), G4SEQ_WINDOW_BP)
            if seq is None:
                continue
            seq = truncate_sequence_from_ends(seq, self.max_sequence_length)
            label = torch.tensor([int(row.label_plus), int(row.label_minus)], dtype=torch.long)
            dataset.append((seq, label, np.array([])))
            if self.max_num_samples is not None and len(dataset) >= self.max_num_samples:
                break
        return dataset

    def _create_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset], Dataset]:
        data_dir = os.path.join(self.root_data_dir_path, self._get_task_data_dir_name())
        genome = open_reference_genome(self.root_data_dir_path, G4SEQ_ASSEMBLY)
        splits = {}
        for name, file_split in (
            ("train", "train"),
            ("validation", "validation"),
            ("test", self.TEST_SPLIT_FILE),
        ):
            manifest = load_manifest(self.task_config, data_dir, G4SEQ_HF_CONFIG, file_split)
            splits[name] = self._build_split(manifest, genome)
            logging.info("%s %s: %d examples", self.get_task_name(), name, len(splits[name]))
        return splits["train"], splits["validation"], splits["test"]

    def get_conditional_input_meta_data_frame(self) -> Optional[pd.DataFrame]:
        return None


class G4SeqStrandMatchedTask(G4SeqStrandTask):
    """Strand-resolved G4 with GC/dinucleotide-matched negatives in the test split."""

    TEST_SPLIT_FILE = "test_matched"
    TASK_NAME = "g4seq_strand_matched"
