# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E smoke tests for the human-DNA tasks (TSS strand, G4 strand, lentiMPRA).

Offline: tiny parquet manifests (the same layout as the Hugging Face dataset) and a toy
reference genome are generated in tmp_path; the task -> mock model -> metrics path is run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from gfmbench_api.tasks.concrete.fantom5_tss_strand_task import Fantom5TSSStrandTask
from gfmbench_api.tasks.concrete.g4seq_strand_task import G4SeqStrandMatchedTask, G4SeqStrandTask
from gfmbench_api.tasks.concrete.lentimpra_activity_task import (
    LentiMPRAHepG2Task,
    LentiMPRAK562Task,
    LentiMPRAWTC11Task,
)
from tests.e2e.conftest import MockGFMModel

CHROM_LEN = 4000
CHROMS = ["chr1", "chr2", "chr3"]
CLASSIFICATION_METRICS = {
    "classification_accuracy",
    "classification_mcc",
    "classification_auroc",
    "classification_auprc",
}
REGRESSION_METRICS = {"regression_pearsonr_macro", "regression_r2_macro"}


class MockRegressionModel(MockGFMModel):
    """Mock model that also implements sequence-level regression."""

    def infer_sequence_to_regression(self, sequences, conditional_input=None):
        return np.array([[len(s) % 7 + s.count("G") / max(len(s), 1)] for s in sequences], dtype=np.float32)


def _write_genome(root, assembly: str) -> None:
    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(root, "reference_genome"), exist_ok=True)
    with open(os.path.join(root, "reference_genome", f"{assembly}.fa"), "w") as f:
        for c in CHROMS:
            seq = "".join(rng.choice(list("ACGT"), CHROM_LEN))
            f.write(f">{c}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")


def _coords(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "id": [f"s{seed}_{i}" for i in range(n)],
        "chrom": rng.choice(CHROMS, n),
        "center": rng.randint(400, CHROM_LEN - 400, n),
    })


def _write(mdir, config: str, split: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.join(mdir, config), exist_ok=True)
    df.to_parquet(os.path.join(mdir, config, f"{split}.parquet"), index=False)


@pytest.fixture
def root(tmp_path):
    _write_genome(str(tmp_path), "hg38")
    _write_genome(str(tmp_path), "hg19")
    return str(tmp_path)


def _cfg(mdir, **kw):
    return {"manifest_dir": str(mdir), "max_num_samples": 12, "batch_size": 4, **kw}


def test_smoke_tss_strand(root, tmp_path):
    mdir = tmp_path / "manifests"
    for i, split in enumerate(["train", "validation", "test"]):
        df = _coords(20, i)
        df["label"] = np.arange(20) % 2
        df["strand"] = np.where(df["label"] == 0, "+", "-")
        _write(mdir, "tss_strand", split, df)
    task = Fantom5TSSStrandTask(root, _cfg(mdir, max_sequence_length=101))
    res = task.eval_test_set(MockGFMModel(num_labels=2))
    assert set(res) == CLASSIFICATION_METRICS and all(v is not None for v in res.values())


@pytest.mark.parametrize("cls,test_file", [(G4SeqStrandTask, "test"), (G4SeqStrandMatchedTask, "test_matched")])
def test_smoke_g4_strand(root, tmp_path, cls, test_file):
    mdir = tmp_path / "manifests"
    for i, split in enumerate(["train", "validation", "test", "test_matched"]):
        df = _coords(20, i)
        df["label_plus"] = np.arange(20) % 2
        df["label_minus"] = (np.arange(20) // 2) % 2
        _write(mdir, "g4seq_strand", split, df)
    task = cls(root, _cfg(mdir, max_sequence_length=101))
    assert task.get_task_attributes()["num_labels"] == 2
    res = task.eval_test_set(MockGFMModel(num_labels=2))
    assert set(res) == CLASSIFICATION_METRICS and all(v is not None for v in res.values())


@pytest.mark.parametrize("cls,cell", [
    (LentiMPRAHepG2Task, "hepg2"), (LentiMPRAK562Task, "k562"), (LentiMPRAWTC11Task, "wtc11"),
])
def test_smoke_lentimpra(tmp_path, cls, cell):
    mdir = tmp_path / "manifests"
    rng = np.random.RandomState(1)
    for i, split in enumerate(["train", "validation", "test"]):
        n = 20
        df = pd.DataFrame({
            "id": [f"{split}_{j}" for j in range(n)],
            "sequence": ["".join(rng.choice(list("ACGT"), 200)) for _ in range(n)],
            "activity": rng.randn(n).astype(np.float32),
            "orientation": ["F", "R"] * (n // 2),
            "group_id": [f"g{split}{j // 2}" for j in range(n)],
        })
        _write(mdir, f"mpra_lenti/{cell}", split, df)
    task = cls(str(tmp_path), _cfg(mdir, max_sequence_length=150))
    res = task.eval_test_set(MockRegressionModel(num_labels=1))
    assert set(res) == REGRESSION_METRICS and all(v is not None for v in res.values())
