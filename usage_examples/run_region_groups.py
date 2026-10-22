#!/usr/bin/env python
"""Run the optional affected-region metric groups for the three DNABERT2-small arms in parallel.

Each job is one (arm, task) pair run by ``run_benchmark.py`` with GFMBENCH_REGION_GROUPS set, writing its own
CSV. A task's AUROC needs all of its variants, so a task is the smallest unit that can run on its own; jobs are
started largest first (longest-processing-time order) on a pool of GPU slots, so the biggest tasks do not end up
last. Finished jobs leave a ``.done`` marker, so rerunning the same command resumes where it stopped.

Usage (from the GFMBench-api directory):
    python usage_examples/run_region_groups.py --dry-run                  # show the plan only
    python usage_examples/run_region_groups.py                            # 8 GPUs x 3 jobs, all five groups
    python usage_examples/run_region_groups.py --gpus 0,1 --jobs_per_gpu 2 --arms ham --tasks brca1,traitgym_mendelian
    python usage_examples/run_region_groups.py --groups forced,context_llr --out region_runs4
    python usage_examples/run_region_groups.py --summarize_only           # re-merge and re-check an --out directory

Outputs under --out: ``jobs/<arm>__<task>.{csv,log,done}`` per job, ``<arm>.csv`` (all of an arm's jobs merged),
``check.txt`` (which metric groups came back empty). The exit status is non-zero if a job failed or a group
came back empty for a job whose model supports it.
"""
import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HAM_REPO = REPO.parent / "hamiltonian-tokenizer"
ROOT_DATA_DIR = "/mnt/lustre/data/projects/sense/common/data/"

# arm -> (run_benchmark model name, run directory, tokenizer artifact relative to the tokenizer repo, tokenizer kind)
ARMS = {
    "single": ("DNABERT2Small_Nucleotide", "gb-nucleotide-ext", None, "nucleotide"),
    "bpe": ("DNABERT2Small_BPE", "gb-bpe-ext", "artifacts/gb-bpe-4096/tokenizer.json", "bpe"),
    "ham": ("DNABERT2Small_Hamiltonian", "gb-ham-ext", "artifacts/gb-ham-4091.json", "hamiltonian"),
}
# Zero-shot tasks and their size in batches of 32 variants (from a full run); proportional to the job's duration
TASK_BATCHES = {
    "bend_variant_effects_disease": 9235,
    "vepeval_clinvar": 8092,
    "bend_variant_effects_expression": 3290,
    "songlab_clinvar": 1281,
    "clinvar_indel": 729,
    "traitgym_complex": 357,
    "brca1": 122,
    "traitgym_mendelian": 106,
    "loleve_causal_eqtl": 39,
}
# Relative cost per batch: the single-nucleotide arm embeds 4x longer sequences
ARM_COST = {"single": 3.0, "bpe": 1.0, "ham": 1.2}
ALL_GROUPS = ["forced", "tokstat", "phase", "profile", "context_llr"]
# Metric name prefix -> group, to check that every group produced values
PREFIX_TO_GROUP = [
    ("forced_region_", "forced"),
    ("tokenizer_", "tokstat"),
    ("affected_region_phase_", "phase"),
    ("affected_context_", "profile"),
    ("context_llr_", "context_llr"),
]


def settings_key(args) -> str:
    """What a job's results depend on; a job counts as done only if it ran with the same settings."""
    return f"groups={args.groups};seed={args.seed};checkpoint={args.checkpoint_step};sanity={args.sanity}"


def is_done(args, out_dir: Path, arm: str, task: str) -> bool:
    stem = out_dir / "jobs" / f"{arm}__{task}"
    done, result = stem.with_suffix(".done"), stem.with_suffix(".csv")
    return done.exists() and result.exists() and settings_key(args) in done.read_text()


def group_of(metric: str):
    for prefix, group in PREFIX_TO_GROUP:
        if metric.startswith(prefix):
            return group
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="region_runs3", help="output directory (default region_runs3)")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="comma-separated GPU ids")
    parser.add_argument("--jobs_per_gpu", type=int, default=3,
                        help="concurrent jobs per GPU; the models are small and the jobs are CPU-bound (default 3)")
    parser.add_argument("--arms", default="single,bpe,ham")
    parser.add_argument("--tasks", default=",".join(TASK_BATCHES))
    parser.add_argument("--groups", default=",".join(ALL_GROUPS),
                        help="region metric groups to run; 'core' is not needed (it is in region_runs2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_step", default="00300000")
    parser.add_argument("--threads", type=int, default=4, help="OMP/MKL threads per job")
    parser.add_argument("--sanity", action="store_true", help="100 variants per task (run_benchmark --sanity_check_mode)")
    parser.add_argument("--force", action="store_true", help="rerun jobs that are already done")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--summarize_only", action="store_true", help="only merge the job CSVs and check them")
    return parser.parse_args()


def split(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_token_frequencies(arm: str) -> Path:
    """The tokenizer prior of the tokstat group: token counts over a sample of the pretraining corpus."""
    path = REPO / "token_freq" / f"{arm}.npy"
    if path.exists():
        return path
    path.parent.mkdir(exist_ok=True)
    _, _, tokenizer, kind = ARMS[arm]
    command = [sys.executable, str(REPO / "usage_examples" / "make_token_frequencies.py"), "--kind", kind,
               "--out", str(path), "--num_lines", "20000"]
    if tokenizer:
        command += ["--tokenizer_path", str(HAM_REPO / tokenizer)]
    print(f"[prep] token frequencies for {arm}: {' '.join(command[1:])}", flush=True)
    subprocess.run(command, check=True, cwd=REPO)
    return path


def job_command(args, arm: str, task: str, csv_path: Path):
    model, run, tokenizer, _ = ARMS[arm]
    command = [sys.executable, "usage_examples/run_benchmark.py", "--model", model,
               "--checkpoint_path", str(HAM_REPO / "runs" / run / f"checkpoint-{args.checkpoint_step}.pt")]
    if tokenizer:
        command += ["--tokenizer_path", str(HAM_REPO / tokenizer)]
    command += ["--linear_prob", "--seed", str(args.seed), "--include_tasks", task,
                "--csv_path", str(csv_path), "--root_data_dir", ROOT_DATA_DIR]
    if args.sanity:
        command.append("--sanity_check_mode")
    return command


def run_job(args, job, gpu, out_dir, lock, results):
    arm, task = job
    stem = out_dir / "jobs" / f"{arm}__{task}"
    csv_path, log_path, done_path = (stem.with_suffix(s) for s in (".csv", ".log", ".done"))
    csv_path.unlink(missing_ok=True)  # a fresh file: run_benchmark skips tasks already in its CSV
    done_path.unlink(missing_ok=True)  # a failed rerun must not leave the old marker behind
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GFMBENCH_REGION_GROUPS=args.groups,
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               GFMBENCH_TOKEN_FREQ=str(REPO / "token_freq" / f"{arm}.npy"))
    start = time.time()
    with open(log_path, "w") as log:
        code = subprocess.call(job_command(args, arm, task, csv_path), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    ok = code == 0 and csv_path.exists() and csv_path.stat().st_size > 0
    if ok:
        done_path.write_text(f"{time.time() - start:.0f}s\n{settings_key(args)}\n")
    with lock:
        results[job] = ok
        print(f"[{time.strftime('%H:%M:%S')}] {'done  ' if ok else 'FAILED'} {arm}/{task} on GPU {gpu} "
              f"({(time.time() - start) / 60:.1f} min){'' if ok else f' -> see {log_path}'}", flush=True)


def worker(args, slot_gpu, jobs, out_dir, lock, results):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        run_job(args, job, slot_gpu, out_dir, lock, results)


def summarize(args, out_dir: Path, jobs) -> int:
    """Merge each arm's job CSVs and report metric groups that came back empty."""
    problems = []
    lines = []
    for arm in split(args.arms):
        rows = []
        for task in split(args.tasks):
            path = out_dir / "jobs" / f"{arm}__{task}.csv"
            if not is_done(args, out_dir, arm, task):
                problems.append(f"{arm}/{task}: job not finished, or finished with different settings")
                continue
            with open(path, newline="") as handle:
                job_rows = list(csv.DictReader(handle))
            rows += job_rows
            counts = {}
            for row in job_rows:
                group = group_of(row["metric"])
                if group:
                    seen, filled = counts.get(group, (0, 0))
                    counts[group] = (seen + 1, filled + (row["my_algo"] not in ("", None)))
            for group in (g for g in split(args.groups) if g != "core"):  # core metrics have no group prefix
                seen, filled = counts.get(group, (0, 0))
                lines.append(f"{arm:7s}{task:34s}{group:12s}{filled:4d}/{seen:<4d}")
                if seen == 0 or filled == 0:
                    problems.append(f"{arm}/{task}: group '{group}' has no values ({filled}/{seen})")
        if rows:
            with open(out_dir / f"{arm}.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["task", "metric", "my_algo"])
                writer.writeheader()
                writer.writerows({key: row.get(key, "") for key in ("task", "metric", "my_algo")} for row in rows)
    report = "metric values filled / total per (arm, task, group)\n" + "\n".join(lines)
    if problems:
        report += "\n\nPROBLEMS\n" + "\n".join(problems)
    (out_dir / "check.txt").write_text(report + "\n")
    print(report.split("\n\nPROBLEMS")[0] if not problems else report)
    return 1 if problems else 0


def main() -> int:
    args = parse_args()
    out_dir = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    arms, tasks, groups = split(args.arms), split(args.tasks), split(args.groups)
    unknown = [a for a in arms if a not in ARMS] + [t for t in tasks if t not in TASK_BATCHES] + \
              [g for g in groups if g not in ALL_GROUPS + ["core"]]
    if unknown:
        raise SystemExit(f"unknown arm/task/group: {unknown}")
    jobs_list = sorted(((a, t) for a in arms for t in tasks),
                       key=lambda job: -TASK_BATCHES[job[1]] * ARM_COST[job[0]])
    (out_dir / "jobs").mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        return summarize(args, out_dir, jobs_list)

    pending = [job for job in jobs_list if args.force or not is_done(args, out_dir, *job)]
    gpus = split(args.gpus)
    slots = [gpu for gpu in gpus for _ in range(args.jobs_per_gpu)]
    total_cost = sum(TASK_BATCHES[t] * ARM_COST[a] for a, t in pending)
    print(f"{len(pending)} of {len(jobs_list)} jobs to run on {len(gpus)} GPU(s) x {args.jobs_per_gpu} slots; "
          f"groups: {args.groups}; output: {out_dir}")
    for arm, task in pending:
        print(f"   {arm:7s}{task:34s}relative cost {TASK_BATCHES[task] * ARM_COST[arm] / total_cost:5.1%}")
    if args.dry_run:
        print("dry run: nothing started")
        return 0
    if not pending:
        return summarize(args, out_dir, jobs_list)
    if "tokstat" in groups:
        for arm in arms:
            ensure_token_frequencies(arm)
    for arm in arms:
        checkpoint = HAM_REPO / "runs" / ARMS[arm][1] / f"checkpoint-{args.checkpoint_step}.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint: {checkpoint}")

    jobs = queue.Queue()
    for job in pending:
        jobs.put(job)
    lock, results = threading.Lock(), {}
    threads = [threading.Thread(target=worker, args=(args, gpu, jobs, out_dir, lock, results)) for gpu in slots]
    start = time.time()
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("interrupted: running jobs are left to finish their current task; rerun to resume", flush=True)
        raise
    print(f"all jobs finished in {(time.time() - start) / 60:.1f} min; failed: "
          f"{[f'{a}/{t}' for (a, t), ok in results.items() if not ok] or 'none'}")
    status = summarize(args, out_dir, jobs_list)
    return 1 if status or not all(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
