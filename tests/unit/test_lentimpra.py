# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the lentiMPRA builder helpers and tasks (synthetic data, no network)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_human_dna_tasks"),
)
import build_lenti_mpra as b  # noqa: E402
from splits import assert_no_leakage  # noqa: E402

from gfmbench_api.tasks.concrete.lentimpra_activity_task import (  # noqa: E402
    LentiMPRAHepG2Task,
    LentiMPRAK562Task,
    LentiMPRAWTC11Task,
)

RNG = np.random.default_rng(0)


def _insert():
    return "".join(RNG.choice(list("ACGT"), b.INSERT_LEN))


def _oligo(insert):
    return "A" * b.ADAPTOR_5P + insert + "C" * 15


def _design(cell="hepg2", n_per_chrom=6, chroms=("chr1", "chr8", "chr10", "chr3")):
    rows = []
    for ci, chrom in enumerate(chroms):
        for i in range(n_per_chrom):
            start = 1000 * (i + 1) if i % 3 != 1 else 1000 * i + 100  # i=1 overlaps i=0 neighbour
            ins = _insert()
            nm = f"peak{ci}_{i}"
            fwd, rev = (nm, nm + "_Reversed:") if cell != "wtc11" else (nm + "_F", nm + "_R")
            for name, strand, seq in ((fwd, "+", ins), (rev, "-", b.revcomp(ins))):
                rows.append(dict(name=name, category="potential enhancer", chrom=chrom.replace("chr", "") if cell == "wtc11" else chrom,
                                 start=start, end=start + 200, strand=strand, design_seq=_oligo(seq)))
    return pd.DataFrame(rows)


def _encode(design, reps=(1, 2, 3), bc=30):
    rows = []
    for name in design["name"]:
        base = RNG.normal()
        for r in reps:
            rows.append(dict(condition="X", replicate=r, name=name, log2=base + RNG.normal(0, 0.1), n_obs_bc=bc))
    return pd.DataFrame(rows)


def test_parse_orientation_per_cell_type():
    assert b.parse_orientation("peak5_Reversed:", "k562") == ("peak5", "R")
    assert b.parse_orientation("peak5", "hepg2") == ("peak5", "F")
    assert b.parse_orientation("seq5396_R", "wtc11") == ("seq5396", "R")
    assert b.parse_orientation("seq5396_F", "wtc11") == ("seq5396", "F")
    assert b.parse_orientation("ENSG00000000003_Reversed:", "hepg2") == ("ENSG00000000003", "R")


def test_clean_design_filters_and_normalises():
    d = _design("wtc11", n_per_chrom=2, chroms=("chr1",))
    bad_len = d.iloc[[0]].copy()
    bad_len["name"] = "short_F"
    bad_len["design_seq"] = "ACGT" * 5
    bad_n = d.iloc[[1]].copy()
    bad_n["name"] = "withN_F"
    bad_n["design_seq"] = "A" * 15 + "N" + "G" * 199 + "C" * 15
    out, counts = b.clean_design(pd.concat([d, bad_len, bad_n]), "wtc11")
    assert counts["dropped_non_230nt"] == 1 and counts["dropped_non_acgt_insert"] == 1
    assert set(out["chrom"]) == {"chr1"}  # WTC11 prefix normalised
    assert (out["insert"].str.len() == 200).all()
    assert set(out["orientation"]) == {"F", "R"}


def test_aggregate_activity_filters_and_replicate_correlation():
    enc = pd.DataFrame(
        [
            ("good", 1, 1.0, 30), ("good", 2, 1.2, 30), ("good", 3, 0.8, 30),
            ("two_ok", 1, 2.0, 30), ("two_ok", 2, 4.0, 30), ("two_ok", 3, 9.0, 3),  # rep 3 dropped
            ("one_ok", 1, 2.0, 30), ("one_ok", 2, 2.0, 3),  # only 1 rep passes -> dropped
        ],
        columns=["name", "replicate", "log2", "n_obs_bc"],
    )
    enc["condition"] = "X"
    act, rep = b.aggregate_activity(enc, min_barcodes=10, min_replicates=2)
    a = act.set_index("name")
    assert set(a.index) == {"good", "two_ok"}
    assert a.loc["two_ok", "activity"] == pytest.approx(3.0) and a.loc["two_ok", "n_replicates"] == 2
    assert a.loc["good", "activity"] == pytest.approx(1.0)
    assert rep["dropped_low_barcode_observations"] == 2
    assert "rep1_vs_rep2" in rep["replicate_pearson"]


def test_groups_keep_orientations_together_and_merge_overlaps():
    d, _ = b.clean_design(_design("hepg2", n_per_chrom=3, chroms=("chr1",)), "hepg2")
    g = b.assign_groups(d)
    d["g"] = g.to_numpy()
    for base, sub in d.groupby("base_name"):
        assert sub["g"].nunique() == 1  # F and R share the group
    # elements 0 and 1 overlap -> merged
    assert d[d.base_name == "peak0_0"]["g"].iloc[0] == d[d.base_name == "peak0_1"]["g"].iloc[0]
    assert d[d.base_name == "peak0_2"]["g"].iloc[0] != d[d.base_name == "peak0_0"]["g"].iloc[0]


def test_groups_merge_promoters_by_ensg():
    d, _ = b.clean_design(
        pd.DataFrame(
            [
                dict(name="ENSG00000000001", category="promoter", chrom="chr1", start=100, end=300, strand="+", design_seq=_oligo(_insert())),
                dict(name="ENSG00000000001_Reversed:", category="promoter", chrom="chr1", start=100, end=300, strand="-", design_seq=_oligo(_insert())),
                dict(name="ENSG00000000001_b", category="promoter", chrom="chr1", start=50000, end=50200, strand="+", design_seq=_oligo(_insert())),
            ]
        ),
        "hepg2",
    )
    d["ensg"] = ["ENSG00000000001"] * 3
    assert b.assign_groups(d).nunique() == 1


@pytest.mark.parametrize("cell", ["hepg2", "k562", "wtc11"])
def test_build_cell_split_no_leakage_and_controls(cell):
    design = _design(cell)
    ctrl = design.iloc[:2].copy()
    ctrl["name"] = [f"c{i}_shuffled" if cell != "wtc11" else f"c{i}_F" for i in range(2)]
    ctrl["category"] = "negative control, shuffled"
    ctrl["chrom"] = None
    design = pd.concat([design, ctrl], ignore_index=True)
    enc = _encode(design)
    enc = enc[enc["name"] != design["name"].iloc[4]]  # one design row without activity
    tables, prov = b.build_cell(design, enc, cell)
    assert prov["design_counts"]["design_rows_without_encode_activity"] == 1
    assert len(tables["controls"]) == 2
    main = pd.concat([tables[s] for s in ("train", "validation", "test")])
    assert (main["category"] != "negative control, shuffled").all()
    assert set(tables["test"]["chrom"]) == {"chr8"} and set(tables["validation"]["chrom"]) == {"chr10"}
    assert set(tables["train"]["chrom"]) == {"chr1", "chr3"}
    assert_no_leakage(main)
    assert set(main["orientation"]) == {"F", "R"}
    assert main["activity"].dtype == np.float64 and (main["n_replicates"] == 3).all()


def _write_manifest(root, cell, n=(12, 5, 7)):
    d = os.path.join(root, "mpra_lenti", cell)
    os.makedirs(d)
    for split, k in zip(("train", "validation", "test"), n):
        pd.DataFrame(
            {"id": [f"{split}{i}" for i in range(k)], "sequence": [_insert() for _ in range(k)],
             "activity": RNG.normal(size=k), "orientation": ["F", "R"] * (k // 2) + ["F"] * (k % 2)}
        ).to_parquet(os.path.join(d, f"{split}.parquet"))


@pytest.mark.parametrize(
    "cls,name,cell",
    [(LentiMPRAHepG2Task, "lentimpra_hepg2", "hepg2"), (LentiMPRAK562Task, "lentimpra_k562", "k562"),
     (LentiMPRAWTC11Task, "lentimpra_wtc11", "wtc11")],
)
def test_task_loads_shapes_truncation_and_sampling(tmp_path, cls, name, cell):
    _write_manifest(str(tmp_path), cell)
    t = cls(str(tmp_path / "data"), {"manifest_dir": str(tmp_path)})
    assert t.get_task_name() == name and t.is_spatial_binned() is False
    attrs = t.get_task_attributes()
    assert attrs["task_type"] == "regression" and attrs["num_labels"] == 1
    assert attrs["output_spatiality"] == "sequence" and attrs["has_validation_data"] is True
    assert attrs["conditional_input_metadata"] is None
    assert (len(t.train_dataset), len(t.validation_dataset), len(t.test_dataset)) == (12, 5, 7)
    seq, label, cond = t.train_dataset[0]
    assert len(seq) == 200 and label.shape == (1,) and label.dtype == torch.float32 and cond.size == 0

    short = cls(str(tmp_path / "data"), {"manifest_dir": str(tmp_path), "max_sequence_length": 100})
    assert len(short.test_dataset[0][0]) == 100

    small = cls(str(tmp_path / "data"), {"manifest_dir": str(tmp_path), "max_num_samples": 4})
    assert (len(small.train_dataset), len(small.validation_dataset), len(small.test_dataset)) == (4, 4, 4)
