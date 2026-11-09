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

"""Unit tests for the FANTOM5 TSS-strand builder and task (synthetic data, no network)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from pyfaidx import Fasta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_human_dna_tasks"))
import build_tss_strand as bts  # noqa: E402

from gfmbench_api.tasks.concrete.fantom5_tss_strand_task import Fantom5TSSStrandTask  # noqa: E402
from gfmbench_api.utils.human_dna_local import extract_window  # noqa: E402

W = 31  # small window for toy data
CHROM_LEN = 4000


def _toy_seq(seed: int, n: int = CHROM_LEN) -> str:
    return "".join(np.random.RandomState(seed).choice(list("ACGT"), n))


def _write_fasta(path, chroms):
    with open(path, "w") as f:
        for name, seq in chroms.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")
    return str(path)


def _peaks(rows):
    df = pd.DataFrame(rows, columns=["chrom", "summit", "strand", "peak_id"])
    df["score"], df["thick_end"], df["item_rgb"] = 0, df["summit"] + 1, "0,0,0"
    df["start"], df["end"], df["thick_start"] = df["summit"] - 5, df["summit"] + 5, df["summit"]
    return df


def test_unidirectional_filter():
    df = _peaks([
        ("chr1", 1000, "+", "drop_a"),   # opposite summit 100 bp away
        ("chr1", 1100, "-", "drop_b"),
        ("chr1", 2000, "+", "keep_same"),  # same-strand neighbour only
        ("chr1", 2050, "+", "keep_same2"),
        ("chr1", 3000, "+", "keep_far"),  # opposite beyond margin
        ("chr1", 3501, "-", "keep_far2"),
    ])
    out = bts.filter_unidirectional(df, margin=500)
    assert set(out["peak_id"]) == {"keep_same", "keep_same2", "keep_far", "keep_far2"}
    # a margin of 501 also removes the 3000/3501 pair
    assert set(bts.filter_unidirectional(df, margin=501)["peak_id"]) == {"keep_same", "keep_same2"}


def test_keep_canonical():
    df = _peaks([("chrM", 5, "+", "m"), ("chrY", 5, "+", "y"), ("chr2", 5, "-", "ok"), ("chrX", 5, ".", "dot")])
    assert list(bts.keep_canonical(df)["peak_id"]) == ["ok"]


def test_window_extraction_reference_orientation(tmp_path):
    seq = _toy_seq(0)
    genome = Fasta(_write_fasta(tmp_path / "g.fa", {"chr1": seq}), one_based_attributes=False)
    center = 1000
    win = extract_window(genome, "chr1", center, W)
    assert len(win) == W
    assert win[W // 2] == seq[center]  # centred on the summit
    assert win == seq[center - W // 2: center - W // 2 + W]
    # reference orientation regardless of strand: identical window for a minus-strand peak
    assert extract_window(genome, "chr1", center, W) == win
    assert bts.window_coords(center, W) == (center - W // 2, center - W // 2 + W)


def _synthetic_inputs(tmp_path):
    chroms = {c: _toy_seq(i) for i, c in enumerate(["chr1", "chr8", "chr10", "chr3"])}
    chroms["chr3"] = chroms["chr3"][:2000] + "N" * 100 + chroms["chr3"][2100:]
    fa = _write_fasta(tmp_path / "toy.fa", chroms)
    rows = []
    for c in ["chr1", "chr8", "chr10", "chr3"]:
        for k, pos in enumerate([100, 600, 1000, 1500, 2050, 3000, 3900]):
            rows.append((c, pos, "+" if k % 2 == 0 else "-", f"{c}_{k}"))
    rows += [("chrM", 500, "+", "m"), ("chr1", 1005, "-", "opp_of_chr1_2")]
    bed = tmp_path / "peaks.bed"
    df = _peaks(rows)
    df[["chrom", "start", "end", "peak_id", "score", "strand", "thick_start", "thick_end", "item_rgb"]].to_csv(
        bed, sep="\t", header=False, index=False
    )
    return str(bed), fa


def test_end_to_end_builder_no_leakage(tmp_path):
    bed, fa = _synthetic_inputs(tmp_path)
    out = tmp_path / "out"
    prov = bts.build(bed, fa, str(out), margin=50, length=W, cluster_margin=0,
                     val_chroms=["chr10"], test_chroms=["chr8"])
    dfs = {s: pd.read_parquet(out / "tss_strand" / f"{s}.parquet") for s in ("train", "validation", "test")}
    allm = pd.concat(dfs.values())
    assert list(allm.columns) == bts.MANIFEST_COLUMNS
    assert set(dfs["test"]["chrom"]) == {"chr8"} and set(dfs["validation"]["chrom"]) == {"chr10"}
    assert set(dfs["train"]["chrom"]) <= {"chr1", "chr3"}
    # no group or chromosome crosses splits
    assert allm.groupby("group_id")["split"].nunique().max() == 1
    assert allm.groupby("chrom")["split"].nunique().max() == 1
    assert "chrM" not in set(allm["chrom"])
    assert "opp_of_chr1_2" not in set(allm["source_id"])
    assert "chr1_2" not in set(allm["source_id"])  # lost its unidirectionality
    # N-containing window (chr3 summit 2050, window 2035..2066 overlaps N 2000..2099) dropped
    assert "chr3_4" not in set(allm["source_id"])
    # window leaving the chromosome (summit 100 is fine for W=31; 3900 fine) -> check bounds
    assert (allm["start"] >= 0).all() and (allm["end"] <= CHROM_LEN).all()
    assert (allm["label"] == (allm["strand"] == "-").astype(int)).all()
    assert (allm["end"] - allm["start"] == W).all()
    assert os.path.exists(out / "tss_strand" / "provenance.json")
    assert prov["peak_counts"]["final"] == len(allm)


def test_window_filter_drops_chrom_edge(tmp_path):
    fa = _write_fasta(tmp_path / "g.fa", {"chr1": _toy_seq(1, 200)})
    genome = Fasta(fa, one_based_attributes=False)
    df = _peaks([("chr1", 5, "+", "edge"), ("chr1", 100, "+", "ok"), ("chr1", 196, "+", "edge2")])
    assert list(bts.filter_windows(df, genome, W)["peak_id"]) == ["ok"]


def test_group_ids_merge_overlapping_windows():
    df = _peaks([("chr1", 100, "+", "a"), ("chr1", 110, "+", "b"), ("chr1", 1000, "+", "c")])
    m = bts.make_manifest(df, W, val_chroms=["chr10"], test_chroms=["chr8"])
    assert m["group_id"].iloc[0] == m["group_id"].iloc[1] != m["group_id"].iloc[2]


@pytest.fixture
def task_root(tmp_path):
    bed, fa = _synthetic_inputs(tmp_path)
    mdir = tmp_path / "manifests"
    bts.build(bed, fa, str(mdir), margin=50, length=511, val_chroms=["chr10"], test_chroms=["chr8"])
    root = tmp_path / "root"
    os.makedirs(root / "reference_genome")
    os.replace(fa, root / "reference_genome" / "hg38.fa")
    return str(root), str(mdir), fa


def _make_task(root, mdir, **cfg):
    return Fantom5TSSStrandTask(root, {"manifest_dir": mdir, **cfg})


def test_task_loads_and_is_reference_oriented(task_root):
    root, mdir, _ = task_root
    t = _make_task(root, mdir)
    assert t.get_task_name() == "fantom5_tss_strand"
    assert t._get_num_classes() == 2 and t._get_num_labels() == 1
    assert t.get_conditional_input_meta_data_frame() is None
    assert t._get_default_max_seq_len() == 511
    train, val, test = t._create_datasets()
    assert val is not None and len(val) > 0 and len(train) > 0 and len(test) > 0
    man = pd.read_parquet(os.path.join(mdir, "tss_strand", "train.parquet"))
    assert len(train) == len(man)
    genome = Fasta(os.path.join(root, "reference_genome", "hg38.fa"), one_based_attributes=False)
    ref_len = 511
    for (seq, label, cond), (_, r) in zip(train, man.iterrows()):
        assert label.dtype.is_floating_point is False and int(label) == r["label"]
        assert cond.shape == (0,)
        # reference orientation, even for minus-strand samples
        assert seq == extract_window(genome, r["chrom"], int(r["center"]), ref_len)


def test_task_truncation_and_max_samples(task_root):
    root, mdir, _ = task_root
    genome = Fasta(os.path.join(root, "reference_genome", "hg38.fa"), one_based_attributes=False)
    t = _make_task(root, mdir, max_sequence_length=101)
    assert t.max_sequence_length == 101
    man = pd.read_parquet(os.path.join(mdir, "tss_strand", "train.parquet"))
    train, _, _ = t._create_datasets()
    for (seq, label, _), (_, r) in zip(train, man.iterrows()):
        full = extract_window(genome, r["chrom"], int(r["center"]), 511)
        assert len(seq) == 101
        assert seq == full[255 - 50: 255 + 51]  # centred crop of the reference window
    t2 = _make_task(root, mdir, max_num_samples=3)
    tr2, va2, te2 = t2._create_datasets()
    assert len(tr2) <= 3 and len(va2) <= 3 and len(te2) <= 3 and len(tr2) > 0
