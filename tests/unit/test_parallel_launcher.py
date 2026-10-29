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

"""Tests for the multi-GPU launcher's scheduling and report merging (no GPU needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from usage_examples import run_benchmark_parallel as launcher


def test_items_are_ordered_heaviest_first_and_big_zero_shot_tasks_are_sharded():
    tasks = ["brca1", "vepeval_clinvar", "var_bench_sqtl", "gue_promoter_all"]
    items = launcher.build_work_items(tasks, cap=None, num_gpus=8, shards=8, shard_min_samples=20000)
    costs = [i.cost for i in items]
    assert costs == sorted(costs, reverse=True)
    by_task = {t: [i for i in items if i.task == t] for t in tasks}
    assert len(by_task["vepeval_clinvar"]) == 8
    assert {i.rank for i in by_task["vepeval_clinvar"]} == set(range(8))
    assert all(i.world == 8 for i in by_task["vepeval_clinvar"])
    # fine-tuning tasks and small tasks are never sharded
    assert len(by_task["var_bench_sqtl"]) == 1 and len(by_task["brca1"]) == 1
    assert items[0].task == "var_bench_sqtl"


def test_sample_cap_prevents_sharding_and_shrinks_cost():
    items = launcher.build_work_items(["vepeval_clinvar"], cap=5000, num_gpus=8, shards=8, shard_min_samples=20000)
    assert len(items) == 1 and items[0].world == 1
    assert launcher.estimate_cost("vepeval_clinvar", 5000) < launcher.estimate_cost("vepeval_clinvar", None)
    forced = launcher.build_work_items(["vepeval_clinvar"], cap=5000, num_gpus=8, shards=4, shard_min_samples=0)
    assert len(forced) == 4


def test_single_gpu_or_single_shard_disables_sharding():
    assert len(launcher.build_work_items(["vepeval_clinvar"], None, num_gpus=1, shards=1, shard_min_samples=0)) == 1
    assert len(launcher.build_work_items(["vepeval_clinvar"], None, num_gpus=8, shards=1, shard_min_samples=0)) == 1


def test_merge_keeps_score_text_byte_for_byte(tmp_path):
    awkward = "0.30000000000000004"
    tiny = "1.0000000000000002"
    (tmp_path / "a.csv").write_text(f"task,metric,algo\nt1,m1,{awkward}\nt1,m2,\n")
    (tmp_path / "b.csv").write_text(f"task,metric,algo\nt2,m1,{tiny}\n")
    out = tmp_path / "final.csv"
    launcher.merge_reports(out, [tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "missing.csv"])
    text = out.read_text()
    assert awkward in text and tiny in text
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(df["task"]) == ["t1", "t1", "t2"]
    assert df.loc[1, "algo"] == ""  # an unavailable metric stays unavailable


def test_merge_updates_existing_report_and_fills_missing_models(tmp_path):
    out = tmp_path / "final.csv"
    out.write_text("task,metric,old\nt0,m,0.5\nt1,m,0.1\n")
    (tmp_path / "p.csv").write_text("task,metric,new\nt1,m,0.9\nt2,m,0.7\n")
    launcher.merge_reports(out, [tmp_path / "p.csv"])
    df = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("task")
    assert df.loc["t0", "old"] == "0.5" and df.loc["t0", "new"] == launcher.NO_RESULTS
    assert df.loc["t1", "old"] == "0.1" and df.loc["t1", "new"] == "0.9"
    assert df.loc["t2", "old"] == launcher.NO_RESULTS


def test_merge_accepts_identical_duplicate_parts_but_rejects_conflicts(tmp_path):
    (tmp_path / "a.csv").write_text("task,metric,algo\nt,m,0.25\n")
    (tmp_path / "b.csv").write_text("task,metric,algo\nt,m,0.25\n")
    (tmp_path / "c.csv").write_text("task,metric,algo\nt,m,0.26\n")
    out = tmp_path / "final.csv"
    launcher.merge_reports(out, [tmp_path / "a.csv", tmp_path / "b.csv"])
    assert len(pd.read_csv(out)) == 1
    with pytest.raises(RuntimeError):
        launcher.merge_reports(tmp_path / "x.csv", [tmp_path / "a.csv", tmp_path / "c.csv"])


def test_rerun_overwrites_scores_of_an_existing_report(tmp_path):
    out = tmp_path / "final.csv"
    out.write_text("task,metric,algo\nt,m1,0.5\nt,m2,\n")
    (tmp_path / "p.csv").write_text("task,metric,algo\nt,m1,0.6\nt,m2,0.7\n")
    launcher.merge_reports(out, [tmp_path / "p.csv"])
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(df["algo"]) == ["0.6", "0.7"]


def test_merge_failure_does_not_mask_the_original_error(monkeypatch, tmp_path):
    monkeypatch.setattr("usage_examples.run_benchmark.TASK_REGISTRY", {"t": object})
    monkeypatch.setattr(launcher, "resolve_gpus", lambda _: [0])

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    def bad_merge(*args, **kwargs):
        raise RuntimeError("Conflicting scores")

    monkeypatch.setattr(launcher, "run_queue", interrupted)
    monkeypatch.setattr(launcher, "merge_reports", bad_merge)
    with pytest.raises(KeyboardInterrupt):
        launcher.main(["--csv_path", str(tmp_path / "r.csv"), "--gpus", "0"])
