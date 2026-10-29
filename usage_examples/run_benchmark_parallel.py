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

"""Run usage_examples/run_benchmark.py on several GPUs at once.

Every task is independent (run_benchmark re-seeds and rebuilds the model per task), so
each work item runs as its own run_benchmark process pinned to one GPU, pulled from a
shared queue (heaviest first). Very large zero-shot tasks can additionally be split into
batch shards that run on different GPUs and are merged back in the original batch order
(see gfmbench_api/utils/sharding.py). Each process writes its own report CSV; they are
merged here as text, so scores are byte-identical to what a single sequential run writes.

Usage (all run_benchmark.py options are forwarded):

    python usage_examples/run_benchmark_parallel.py \\
        --csv_path report.csv --root_data_dir_path /path/to/data --model DNABERT2
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_BENCHMARK = Path(__file__).resolve().parent / "run_benchmark.py"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

NO_RESULTS = "NO_RESULTS"
EPOCHS = 3  # run_benchmark default; only used for scheduling estimates

# Approximate full-size workload per task, used only to order the queue and decide
# sharding: (test samples, forward passes per test sample, train samples,
# sequences per train sample, relative sequence length).
TASK_WORKLOAD: Dict[str, Tuple[int, int, int, int, float]] = {
    "vepeval_clinvar": (258944, 3, 0, 0, 1.0),
    "brca1": (3893, 3, 0, 0, 1.0),
    "clinvar_indel": (23432, 2, 0, 0, 1.0),
    "loleve_causal_eqtl": (1229, 2, 0, 0, 1.0),
    "lrb_variant_effect_causal_eqtl": (8862, 2, 89060, 2, 1.0),
    "lrb_variant_effect_pathogenic_omim": (200406, 2, 0, 0, 1.0),
    "lrb_regulatory_element_enhancer": (192201, 1, 99999, 1, 0.5),
    "lrb_chromatin_features_histone_marks": (100000, 1, 100000, 1, 0.5),
    "bend_variant_effects_disease": (295495, 3, 0, 0, 1.0),
    "bend_variant_effects_expression": (105263, 2, 0, 0, 1.0),
    "gue_promoter_all": (5920, 1, 47356, 1, 0.1),
    "gue_splice_site": (4562, 1, 36496, 1, 0.1),
    "gue_transcription_factor": (5000, 1, 128344, 1, 0.1),
    "songlab_clinvar": (40976, 3, 0, 0, 1.0),
    "traitgym_complex": (11400, 2, 0, 0, 1.0),
    "traitgym_mendelian": (3380, 2, 0, 0, 1.0),
    "var_bench_coding_pathogenicity": (8729, 2, 74143, 2, 1.0),
    "var_bench_non_coding_pathogenicity": (21622, 2, 273873, 2, 1.0),
    "var_bench_expression": (9147, 2, 147015, 2, 1.0),
    "var_bench_common_vs_rare": (6092, 2, 193908, 2, 1.0),
    "var_bench_meqtl": (2887, 2, 73601, 2, 1.0),
    "var_bench_sqtl": (54500, 2, 1000741, 2, 1.0),
}
_DEFAULT_WORKLOAD = (50000, 2, 50000, 2, 1.0)

# Zero-shot tasks (no fine-tuning) whose test set is big enough to split across GPUs.
SHARDABLE_TASKS = {
    "vepeval_clinvar",
    "lrb_variant_effect_pathogenic_omim",
    "bend_variant_effects_disease",
    "bend_variant_effects_expression",
    "songlab_clinvar",
    "clinvar_indel",
}


@dataclass
class WorkItem:
    task: str
    rank: int
    world: int
    cost: float
    gpu: Optional[int] = None
    returncode: Optional[int] = None
    seconds: float = 0.0

    @property
    def name(self) -> str:
        return self.task if self.world == 1 else f"{self.task}[{self.rank}/{self.world}]"

    @property
    def stem(self) -> str:
        return f"{self.task}.r{self.rank}of{self.world}"


def estimate_test_samples(task: str, cap: Optional[int]) -> int:
    n = TASK_WORKLOAD.get(task, _DEFAULT_WORKLOAD)[0]
    return n if cap is None else min(n, cap)


def estimate_cost(task: str, cap: Optional[int]) -> float:
    """Relative cost: test forwards plus ~3x-forward training steps over EPOCHS epochs."""
    test_n, test_fw, train_n, train_seqs, length = TASK_WORKLOAD.get(task, _DEFAULT_WORKLOAD)
    if cap is not None:
        test_n, train_n = min(test_n, cap), min(train_n, cap)
    return length * (test_n * test_fw + train_n * train_seqs * EPOCHS * 3)


def build_work_items(
    tasks: List[str], cap: Optional[int], num_gpus: int, shards: int, shard_min_samples: int
) -> List[WorkItem]:
    items: List[WorkItem] = []
    for task in tasks:
        cost = estimate_cost(task, cap)
        world = 1
        if (
            task in SHARDABLE_TASKS
            and shards > 1
            and num_gpus > 1
            and estimate_test_samples(task, cap) >= shard_min_samples
        ):
            world = shards
        items.extend(WorkItem(task, rank, world, cost / world) for rank in range(world))
    return sorted(items, key=lambda item: -item.cost)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run run_benchmark.py on multiple GPUs; unknown options are forwarded to it.",
    )
    parser.add_argument("--csv_path", required=True, help="Final merged report CSV.")
    parser.add_argument("--tasks", nargs="*", default=None, metavar="TASK",
                        help="Run only these tasks (default: all).")
    parser.add_argument("--exclude_tasks", nargs="*", default=[], metavar="TASK")
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--sanity_check_mode", action="store_true")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids (default: all visible GPUs).")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Scratch dir for per-process logs/CSVs/shards "
                             "(default: <csv_path>.parallel). Cleared at start.")
    parser.add_argument("--shards", type=int, default=None,
                        help="Split each large zero-shot task into this many batch shards "
                             "(default: number of GPUs; 1 disables sharding).")
    parser.add_argument("--shard_min_samples", type=int, default=20000,
                        help="Only shard tasks with at least this many test samples (default: 20000).")
    return parser.parse_known_args(argv)


def resolve_gpus(spec: Optional[str]) -> List[str]:
    if spec:
        return [gpu.strip() for gpu in spec.split(",") if gpu.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def run_queue(items: List[WorkItem], gpus: List[str], command_for, log_dir: Path) -> None:
    """One worker thread per GPU pulls the next item (heaviest first) until the queue is empty."""
    pending: "queue.Queue[WorkItem]" = queue.Queue()
    for item in items:
        pending.put(item)
    print_lock = threading.Lock()
    start = time.perf_counter()

    def worker(gpu: str) -> None:
        while True:
            try:
                item = pending.get_nowait()
            except queue.Empty:
                return
            item.gpu = int(gpu) if gpu.isdigit() else None
            began = time.perf_counter()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            try:
                with open(log_dir / f"{item.stem}.log", "w") as log:
                    item.returncode = subprocess.run(
                        command_for(item), env=env, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT
                    ).returncode
            except OSError as exc:
                item.returncode = -1
                print(f"could not run {item.name}: {exc}", flush=True)
            item.seconds = time.perf_counter() - began
            with print_lock:
                status = "ok" if item.returncode == 0 else f"FAILED (rc={item.returncode})"
                print(
                    f"[{time.perf_counter() - start:7.1f}s] gpu {gpu}: {item.name} "
                    f"{status} in {item.seconds:.1f}s",
                    flush=True,
                )

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _read_strings(path: Path) -> pd.DataFrame:
    # Everything stays text: scores are never parsed, so they cannot change by a bit.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def merge_reports(final_csv: Path, part_csvs: List[Path]) -> None:
    """Merge per-process report CSVs into ``final_csv`` (existing rows are kept/updated)."""
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    from_part: set = set()  # (key, column) already written by a part file in this merge
    model_columns: List[str] = []

    def absorb(df: pd.DataFrame, source: str) -> None:
        for column in df.columns:
            if column not in ("task", "metric") and column not in model_columns:
                model_columns.append(column)
        for record in df.to_dict("records"):
            key = (record.pop("task"), record.pop("metric"))
            slot = rows.setdefault(key, {})
            for column, value in record.items():
                if source == "part":
                    # A new score replaces an old report's value (as BenchmarkReport does), but
                    # two part files of this run must never disagree.
                    if (key, column) in from_part and slot[column] != value:
                        raise RuntimeError(f"Conflicting scores for {key} / {column}: {slot[column]} vs {value}")
                    from_part.add((key, column))
                slot[column] = value

    if final_csv.exists():
        absorb(_read_strings(final_csv), "existing")
    for part in part_csvs:
        if part.exists():
            absorb(_read_strings(part), "part")

    records = [
        {"task": task, "metric": metric, **{c: slot.get(c, NO_RESULTS) for c in model_columns}}
        for (task, metric), slot in rows.items()
    ]
    pd.DataFrame(records, columns=["task", "metric", *model_columns]).to_csv(final_csv, index=False)


def main(argv=None) -> int:
    args, forwarded = parse_args(argv)

    from usage_examples.run_benchmark import TASK_REGISTRY

    unknown = set(args.tasks or []) - set(TASK_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown task name(s): {sorted(unknown)}")
    tasks = [
        name for name in TASK_REGISTRY
        if (args.tasks is None or name in args.tasks) and name not in args.exclude_tasks
    ]
    cap = args.max_num_samples if args.max_num_samples is not None else (100 if args.sanity_check_mode else None)

    gpus = resolve_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs found; pass --gpus or run run_benchmark.py directly.")
    shards = args.shards if args.shards is not None else len(gpus)

    final_csv = Path(args.csv_path).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else Path(f"{final_csv}.parallel")
    log_dir, csv_dir, shard_dir = work_dir / "logs", work_dir / "csv", work_dir / "shards"
    for directory in (log_dir, csv_dir, shard_dir):
        # Only directories this launcher owns are cleared (never the user-supplied work_dir
        # itself); stale shards/CSVs from an earlier run must never be merged.
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    run_id = uuid.uuid4().hex
    items = build_work_items(tasks, cap, len(gpus), shards, args.shard_min_samples)

    def command_for(item: WorkItem) -> List[str]:
        command = [
            sys.executable, str(_RUN_BENCHMARK), *forwarded,
            "--csv_path", str(csv_dir / f"{item.stem}.csv"),
            "--tasks", item.task,
        ]
        if args.max_num_samples is not None:
            command += ["--max_num_samples", str(args.max_num_samples)]
        if args.sanity_check_mode:
            command.append("--sanity_check_mode")
        if item.world > 1:
            command += [
                "--shard_rank", str(item.rank), "--shard_world", str(item.world),
                "--shard_dir", str(shard_dir), "--shard_run_id", run_id,
            ]
        return command

    print(f"{len(items)} work items for {len(tasks)} tasks on GPUs {gpus}; logs in {log_dir}", flush=True)
    began = time.perf_counter()
    part_csvs = [csv_dir / f"{item.stem}.csv" for item in sorted(items, key=lambda i: tasks.index(i.task))]
    try:
        run_queue(items, gpus, command_for, log_dir)
    except BaseException:
        # Keep the scores of every finished task even on Ctrl-C, but never let a merge
        # failure mask the original error.
        try:
            merge_reports(final_csv, part_csvs)
        except Exception as merge_error:
            print(f"Could not merge partial results: {merge_error}", file=sys.stderr, flush=True)
        raise
    merge_reports(final_csv, part_csvs)
    elapsed = time.perf_counter() - began

    failed = [item for item in items if item.returncode != 0]
    print(f"Total wall time: {elapsed:.1f}s; merged report: {final_csv}", flush=True)
    if failed:
        print("FAILED work items (see logs): " + ", ".join(item.name for item in failed), flush=True)
        return 1
    missing = [t for t in tasks if not any(
        p.exists() for p in (csv_dir / f"{i.stem}.csv" for i in items if i.task == t))]
    if missing:
        print(f"No result produced for: {missing}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
