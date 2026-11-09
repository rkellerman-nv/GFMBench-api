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

"""Shared loading helpers for the human-DNA tasks (TSS strand, G4 strand, lentiMPRA).

Coordinate-based tasks ship parquet *manifests* (no sequences). Sequences are cut
from the UCSC reference genome at load time, in reference orientation.

Third-party URL notices for this file (Python packages: THIRD_PARTY_NOTICES.md):
- https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz — LicenseRef-UCSC-Genome-Browser
- https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz — LicenseRef-UCSC-Genome-Browser
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta

from gfmbench_api.utils.lrb_local import (
    _get_chromosome,
    builder_pad_sequence,
    ensure_assembly_genome,
)

# Hugging Face dataset repo holding the manifests. The namespace is a placeholder
# until the dataset is published; override per run with task_config["hf_repo_id"].
HUMAN_DNA_HF_REPO_ID = "TBD/gfmbench-human-dna-tasks"
HUMAN_DNA_HF_REVISION = "main"  # pin to a release tag when publishing

SPLIT_FILES = {"train": "train", "validation": "validation", "test": "test"}


def load_manifest(
    task_config: Optional[Dict[str, Any]],
    data_dir: str,
    hf_config: str,
    split: str,
) -> pd.DataFrame:
    """Return one manifest split as a DataFrame.

    Resolution order: ``task_config["manifest_dir"]/<hf_config>/<split>.parquet``
    (local override, used by tests and pre-release builds), then the cached copy in
    ``data_dir``, then a download from the Hugging Face dataset repo.
    """
    cfg = task_config or {}
    rel = f"{hf_config}/{split}.parquet"

    manifest_dir = cfg.get("manifest_dir")
    if manifest_dir:
        return pd.read_parquet(os.path.join(manifest_dir, rel))

    cached = os.path.join(data_dir, rel)
    if not os.path.exists(cached):
        logging.info("Downloading %s from Hugging Face...", rel)
        os.makedirs(data_dir, exist_ok=True)
        hf_hub_download(
            repo_id=cfg.get("hf_repo_id", HUMAN_DNA_HF_REPO_ID),
            filename=rel,
            repo_type="dataset",
            revision=cfg.get("hf_revision", HUMAN_DNA_HF_REVISION),
            local_dir=data_dir,
        )
    return pd.read_parquet(cached)


def open_reference_genome(root_data_dir_path: str, assembly: str) -> Fasta:
    """Open (downloading if needed) the UCSC reference genome for ``assembly``."""
    genome_dir = os.path.join(root_data_dir_path, "reference_genome")
    return Fasta(ensure_assembly_genome(genome_dir, assembly), one_based_attributes=False)


def extract_window(genome: Fasta, chrom: str, center: int, length: int) -> Optional[str]:
    """Reference-orientation window of ``length`` bp centred on 0-based ``center``.

    Returns ``None`` when the window leaves the chromosome (such rows are dropped
    by the builders, so this should not happen for released manifests). Never
    reverse-complements: the input must not be oriented by the target strand.
    """
    chromosome = _get_chromosome(genome, chrom)
    if chromosome is None:
        return None
    seq = builder_pad_sequence(chromosome, int(center), int(length))
    return None if seq is None else seq.upper()
