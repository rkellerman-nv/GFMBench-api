# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
import sys

import numpy as np
import pandas as pd
import pytest
from pyfaidx import Fasta

# scripts/build_human_dna_tasks may not be a package: import via sys.path.
_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_human_dna_tasks")
sys.path.insert(0, os.path.abspath(_BUILD_DIR))
import build_g4seq_strand as b  # noqa: E402
from splits import assert_no_leakage  # noqa: E402

from gfmbench_api.tasks.concrete.g4seq_strand_task import (  # noqa: E402
    G4SeqStrandMatchedTask,
    G4SeqStrandTask,
)


def iv(*pairs, chrom="chr1"):
    a = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return {chrom: b.merge_intervals(a[:, 0], a[:, 1])}


def make_genome(tmp_path, seed=0, lens=None):
    lens = lens or {"chr1": 3000, "chr2": 3000, "chrM": 200}
    rng = np.random.default_rng(seed)
    path = tmp_path / "toy.fa"
    seqs = {c: "".join(rng.choice(list("ACGT"), n)) for c, n in lens.items()}
    path.write_text("".join(f">{c}\n{s}\n" for c, s in seqs.items()))
    return str(path), seqs


# ---------------------------------------------------------------- strand
def test_strand_inversion_constant_and_loader(tmp_path):
    assert b.g4_strand_of_file_strand("plus") == "minus"
    assert b.g4_strand_of_file_strand("minus") == "plus"
    (tmp_path / "GSE63874_Na_K_plus_hits_intersect.bed").write_text("chr1\t100\t200\nchrM\t1\t50\n")
    with gzip.open(tmp_path / "GSE63874_Na_K_minus_hits_intersect.bed.gz", "wt") as f:
        f.write("chr1\t500\t600\n")
    g4, info = b.load_condition_hits(str(tmp_path), "K")
    assert list(g4["minus"]["chr1"][0]) == [100]  # from the *_plus_* file
    assert list(g4["plus"]["chr1"][0]) == [500]  # from the *_minus_* file
    assert "chrM" not in g4["minus"]  # chrM dropped
    assert all(len(v["sha256"]) == 64 for v in info.values())


# ---------------------------------------------------------------- labelling
def test_centre_labels_both_neither_and_ambiguity():
    W, C = 31, 7
    g4 = {"plus": iv((100, 110), (300, 320), (500, 510)), "minus": iv((100, 105), (302, 304), (595, 600))}
    cents = np.array([105, 310, 500, 200, 585])
    lp, lm, amb = b.label_windows(g4, "chr1", cents, W, C)
    # 105: both strands overlap centre [102,109) -> [1,1]
    assert (lp[0], lm[0], amb[0]) == (1, 1, False)
    # 310: plus centre hit; minus hit at 302-304 is inside window but outside centre [307,314) -> ambiguous
    assert (lp[1], lm[1], amb[1]) == (1, 0, True)
    # 500: plus only, minus none within window [485,516)
    assert (lp[2], lm[2], amb[2]) == (1, 0, False)
    # 200: nothing
    assert (lp[3], lm[3], amb[3]) == (0, 0, False)
    # 585: window [570,601) hits minus 595-600 outside centre [582,589) -> ambiguous
    assert amb[4] and (lp[4], lm[4]) == (0, 0)


def test_build_positives_drops_ambiguous_and_boundary():
    g4 = {"plus": iv((100, 110), (300, 320), (5, 9)), "minus": iv((302, 304))}
    df, counts = b.build_positives(g4, {"chr1": 1000}, 31, 7)
    assert counts["dropped_ambiguous"] == 1 and counts["dropped_boundary"] == 1
    # centre 310 (plus locus 300-320) is ambiguous because of the minus hit at 302-304
    # outside its centre; the minus locus (centre 303) overlaps both -> [1,1] kept.
    assert df["center"].tolist() == [105, 303]
    assert df.loc[0, ["label_plus", "label_minus"]].tolist() == [1, 0]
    assert df.loc[1, ["label_plus", "label_minus"]].tolist() == [1, 1]
    assert (df["end"] - df["start"] == 31).all()


# ---------------------------------------------------------------- mask
def write_bg(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")


def test_stream_bedgraph_mask_and_intersection(tmp_path):
    write_bg(tmp_path / "p.bedGraph.gz", [("chr1", 0, 15, 10.0), ("chr1", 15, 30, 11.0), ("chr1", 45, 60, 3.0),
                                          ("chr1", 60, 75, "nan"), ("chrM", 0, 15, 1.0)])
    write_bg(tmp_path / "m.bedGraph.gz", [("chr1", 10, 25, 5.0), ("chr1", 25, 50, 5.0)])
    mask = b.build_callable_mask([(str(tmp_path / "p.bedGraph.gz"), str(tmp_path / "m.bedGraph.gz"))])
    assert mask["chr1"][0].tolist() == [10, 45] and mask["chr1"][1].tolist() == [30, 50]
    assert "chrM" not in mask
    b.write_bed_intervals(str(tmp_path / "mask.bed"), mask)
    back, _ = b.read_bed_intervals(str(tmp_path / "mask.bed"))
    assert back["chr1"][0].tolist() == [10, 45]


def test_unsorted_bedgraph_raises(tmp_path):
    write_bg(tmp_path / "x.bedGraph.gz", [("chr1", 0, 5, 1), ("chr2", 0, 5, 1), ("chr1", 10, 15, 1)])
    with pytest.raises(ValueError):
        b.stream_bedgraph_coverage(str(tmp_path / "x.bedGraph.gz"))


# ---------------------------------------------------------------- negatives
def test_negatives_inside_mask_avoid_hits_and_buffer():
    mask = {"chr1": (np.array([0, 2000]), np.array([1000, 2600]))}
    excl = iv((500, 520))
    rng = np.random.default_rng(1)
    neg = b.sample_negatives(mask, excl, ["chr1"], 100, 31, 10, rng, chrom_sizes={"chr1": 3000})
    assert len(neg) == 100
    assert ((neg.start >= 0) & (neg.end <= 1000) | (neg.start >= 2000) & (neg.end <= 2600)).all()
    assert not b.any_overlap(excl, "chr1", neg.start.values - 10, neg.end.values + 10).any()
    assert ((neg.end - neg.start) == 31).all() and ((neg.center - neg.start) == 15).all()
    assert (neg.start >= 2000).any() and (neg.start < 1000).any()  # both mask segments used


def test_negatives_reject_N(tmp_path):
    seq = "A" * 100 + "N" * 40 + "C" * 160
    fa = tmp_path / "n.fa"
    fa.write_text(f">chr1\n{seq}\n")
    g = Fasta(str(fa), one_based_attributes=False)
    mask = {"chr1": (np.array([0]), np.array([300]))}
    neg = b.sample_negatives(mask, {}, ["chr1"], 30, 21, 0, np.random.default_rng(0), genome=g)
    for r in neg.itertuples():
        assert "N" not in b.fetch(g, "chr1", r.start, r.end)


# ---------------------------------------------------------------- matching
def test_matching_gc_tolerance_and_dinuc_choice():
    rng = np.random.default_rng(0)
    pos_f = np.zeros((1, 16), dtype=np.float32); pos_f[0, 0] = 1.0
    pool_f = np.zeros((3, 16), dtype=np.float32)
    pool_f[0, 0] = 1.0       # same dinuc, GC off by .1 -> excluded
    pool_f[1, 0], pool_f[1, 1] = 0.9, 0.1   # candidate, L1=0.2
    pool_f[2, 0], pool_f[2, 5] = 0.5, 0.5   # candidate, L1=1.0
    pick = b.match_negatives(pos_f, np.array([0.4]), pool_f, np.array([0.5, 0.41, 0.39]), 0.02, rng)
    assert pick.tolist() == [1]
    # without replacement + no candidate -> -1
    pick = b.match_negatives(np.vstack([pos_f, pos_f]), np.array([0.4, 0.4]), pool_f[1:2], np.array([0.41]), 0.02, rng)
    assert sorted(pick.tolist()) == [-1, 0]


def test_dinuc_and_gc():
    d, gc = b.dinuc_and_gc("ACGTNN")
    assert gc == pytest.approx(0.5) and d.sum() == pytest.approx(1.0) and d[0 * 4 + 1] > 0


# ---------------------------------------------------------------- QC
def test_quadparser_enrichment_strand_logic(tmp_path):
    g4 = "GGGAGGGAGGGAGGG"
    plus_seq = "A" * 10 + g4 + "A" * 10
    rc_seq = b.revcomp(plus_seq)
    fa = tmp_path / "q.fa"
    fa.write_text(f">chr1\n{plus_seq}{'A' * 30}{rc_seq}{'T' * 5}\n")
    g = Fasta(str(fa), one_based_attributes=False)
    c_plus = len(plus_seq) // 2
    c_minus = len(plus_seq) + 30 + len(rc_seq) // 2
    df = pd.DataFrame({"chrom": ["chr1"] * 3, "center": [c_plus, c_minus, 70],
                       "label_plus": [1, 0, 0], "label_minus": [0, 1, 0]})
    qc = b.quadparser_enrichment(g, df, 21)
    assert qc["plus_fwd"] == 1 and qc["plus_rc"] == 0
    assert qc["minus_rc"] == 1 and qc["minus_fwd"] == 0
    assert qc["strand_consistent"]


def test_quadparser_regex_and_revcomp():
    assert b.QUADPARSER.search("TTGGGTGGGTGGGTGGGTT")
    assert b.revcomp("AACG") == "CGTT"


def test_coordinate_check_runs(tmp_path):
    path, seqs = make_genome(tmp_path)
    g = Fasta(path, one_based_attributes=False)
    res = b.check_coordinate_offset(g, {"plus": iv((10, 40)), "minus": iv((100, 130))}, n=5)
    assert set(res) == {"quad_rate_offset_0", "quad_rate_offset_1"}
    with pytest.raises(ValueError):
        b.check_coordinate_offset(g, {"plus": iv((2990, 3005)), "minus": {}}, n=5)


# ---------------------------------------------------------------- end to end
def toy_build(tmp_path, **kw):
    path, seqs = make_genome(tmp_path)
    g = Fasta(path, one_based_attributes=False)
    sizes = {c: len(g[c]) for c in ("chr1", "chr2")}
    plus_hits = {"chr1": (np.array([300, 700, 1100, 1500]), np.array([320, 720, 1120, 1520])),
                 "chr2": (np.array([300, 700, 1100, 1500]), np.array([320, 720, 1120, 1520]))}
    minus_hits = {"chr2": (np.array([1500]), np.array([1520]))}
    g4 = {"plus": plus_hits, "minus": minus_hits}
    excl = {c: b.merge_intervals(*(np.concatenate(x) for x in zip(*[v for h in (plus_hits, minus_hits)
            for k, v in h.items() if k == c]))) for c in ("chr1", "chr2")}
    mask = {"chr1": (np.array([0]), np.array([3000])), "chr2": (np.array([0]), np.array([3000]))}
    out, stats = b.build_dataset(g4, excl, mask, g, sizes, window=31, centre_width=7, buffer=10, neg_ratio=1.0,
                                 pool_factor=20, gc_tol=0.1, seed=0, val_chroms=(), test_chroms=("chr2",), **kw)
    return out, stats, g


def test_build_dataset_splits_no_leakage_and_matched(tmp_path):
    out, stats, g = toy_build(tmp_path)
    assert set(out) == {"train", "validation", "test", "test_matched"}
    cols = ["id", "chrom", "start", "end", "center", "assembly", "label_plus", "label_minus", "condition",
            "group_id", "split", "source_id", "negative_set"]
    assert list(out["test"].columns) == cols
    assert (out["train"].chrom == "chr1").all() and (out["test"].chrom == "chr2").all()
    assert (out["test_matched"].chrom == "chr2").all()
    pos = lambda d: d[d.negative_set == "positive"]  # noqa: E731
    pd.testing.assert_frame_equal(pos(out["test"]).sort_values("id").reset_index(drop=True),
                                  pos(out["test_matched"]).sort_values("id").reset_index(drop=True))
    assert set(out["test"].negative_set) == {"positive", "representative"}
    assert set(out["test_matched"].negative_set) == {"positive", "matched"}
    assert len(out["test_matched"][out["test_matched"].negative_set == "matched"]) == stats["matching"]["n_matched"]
    # plus on file-minus: chr2 locus 1500 is both-strand -> [1,1] retained
    assert ((out["test"].label_plus == 1) & (out["test"].label_minus == 1)).sum() == 1
    assert out["train"].label_minus.sum() == 0
    full = pd.concat([out["train"], out["test"]])
    assert_no_leakage(full)
    bad = full.reset_index(drop=True)
    bad.loc[0, "split"] = "validation"  # same chrom/group now in two splits
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_main_cli_end_to_end(tmp_path):
    path, _ = make_genome(tmp_path)
    hits = tmp_path / "hits"
    hits.mkdir()
    for cond in ("K", "PDS"):
        for fs in ("plus", "minus"):
            (hits / f"GSE63874_Na_{cond}_{fs}_hits_intersect.bed").write_text(
                "chr1\t300\t320\nchr1\t800\t830\nchr2\t300\t320\nchr2\t900\t915\n" if fs == "plus" else "chr2\t1500\t1520\nchr1\t1500\t1520\n")
    mask = tmp_path / "mask.bed"
    mask.write_text("chr1\t0\t3000\nchr2\t0\t3000\n")
    prov = b.main(["--hits-dir", str(hits), "--genome", path, "--out", str(tmp_path / "out"), "--mask-bed", str(mask),
                   "--window", "31", "--centre-width", "7", "--buffer", "10", "--test-chroms", "chr2",
                   "--val-chroms", "--gc-tol", "0.2", "--check-coordinates"])
    d = tmp_path / "out" / "g4seq_strand"
    assert {p.name for p in d.iterdir()} >= {"train.parquet", "validation.parquet", "test.parquet",
                                              "test_matched.parquet", "provenance.json"}
    assert prov["condition"] == "K" and prov["assembly"] == "hg19"


# ---------------------------------------------------------------- task
@pytest.fixture
def task_root(tmp_path):
    path, seqs = make_genome(tmp_path, lens={"chr1": 3000})
    root = tmp_path / "root"
    (root / "reference_genome").mkdir(parents=True)
    os.replace(path, root / "reference_genome" / "hg19.fa")
    man = tmp_path / "manifests" / "g4seq_strand"
    man.mkdir(parents=True)

    def frame(centers, lp, lm):
        return pd.DataFrame({"id": [f"i{c}" for c in centers], "chrom": "chr1", "start": np.array(centers) - 255,
                             "end": np.array(centers) + 256, "center": centers, "assembly": "hg19",
                             "label_plus": lp, "label_minus": lm, "condition": "K", "group_id": "chr1:0",
                             "split": "train", "source_id": "s", "negative_set": "n"})
    frame([600, 900, 1200, 1500], [1, 0, 1, 0], [0, 1, 1, 0]).to_parquet(man / "train.parquet")
    frame([1800], [1], [0]).to_parquet(man / "validation.parquet")
    frame([2000, 2100], [1, 0], [0, 0]).to_parquet(man / "test.parquet")
    frame([2200, 2300, 2400], [0, 1, 0], [1, 0, 0]).to_parquet(man / "test_matched.parquet")
    return root, man.parent, seqs["chr1"]


def test_task_loads_shapes_and_sequences(task_root):
    root, man, chr1 = task_root
    t = G4SeqStrandTask(str(root), {"manifest_dir": str(man)})
    assert t.get_task_name() == "g4seq_strand" and t._get_task_data_dir_name() == "g4seq_strand"
    assert t._get_num_labels() == 2 and t._get_num_classes() == 2 and t.max_sequence_length == 511
    assert len(t.train_dataset) == 4 and len(t.validation_dataset) == 1 and len(t.test_dataset) == 2
    seq, label, cond = t.train_dataset[0]
    assert seq == chr1[600 - 255:600 + 256] and len(seq) == 511
    assert tuple(label.shape) == (2,) and label.dtype.is_floating_point is False and label.tolist() == [1, 0]
    assert cond.size == 0
    assert t.get_conditional_input_meta_data_frame() is None
    assert t.get_task_attributes()["classification_mode"] == "multi_label"


def test_task_truncation_and_max_num_samples(task_root):
    root, man, chr1 = task_root
    t = G4SeqStrandTask(str(root), {"manifest_dir": str(man), "max_sequence_length": 101, "max_num_samples": 2})
    assert len(t.train_dataset) == 2 and len(t.test_dataset) == 2
    assert t.train_dataset[0][0] == chr1[600 - 50:600 + 51]


def test_matched_task_uses_test_matched(task_root):
    root, man, chr1 = task_root
    t = G4SeqStrandMatchedTask(str(root), {"manifest_dir": str(man)})
    assert t.get_task_name() == "g4seq_strand_matched" and t._get_task_data_dir_name() == "g4seq_strand"
    assert len(t.test_dataset) == 3 and len(t.train_dataset) == 4
    assert t.test_dataset[1][1].tolist() == [1, 0] and t.test_dataset[0][0] == chr1[2200 - 255:2200 + 256]


def test_task_drops_windows_off_chromosome(task_root, tmp_path):
    root, man, _ = task_root
    df = pd.read_parquet(man / "g4seq_strand" / "test.parquet")
    df.loc[0, ["center"]] = 100  # window leaves chromosome -> extract_window None
    df.to_parquet(man / "g4seq_strand" / "test.parquet")
    t = G4SeqStrandTask(str(root), {"manifest_dir": str(man)})
    assert len(t.test_dataset) == 1
