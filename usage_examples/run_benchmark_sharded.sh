#!/usr/bin/env bash
# Run run_benchmark.py as N parallel shards, one process per GPU, then merge the reports.
#
# Usage: usage_examples/run_benchmark_sharded.sh NUM_SHARDS CSV_PATH [run_benchmark.py args...]
# e.g.   usage_examples/run_benchmark_sharded.sh 8 ham_full_lp.csv --model DNABERT2Small_Hamiltonian \
#          --checkpoint_path ckpt.pt --linear_prob --root_data_dir_path /path/to/data
# Shard i runs on GPU (i % visible GPUs) and writes CSV_PATH.shard<i> and CSV_PATH.shard<i>.log.
# Exits non-zero (and does not write CSV_PATH) if any shard fails or leaves no report.
set -euo pipefail

NUM_SHARDS=$1
CSV_PATH=$2
shift 2
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if ((NUM_GPUS < 1)); then
  echo "No GPUs found (set NUM_GPUS to override)" >&2
  exit 1
fi
HERE=$(cd "$(dirname "$0")" && pwd)

# Stale reports from earlier runs would otherwise be merged in if a shard dies early.
rm -f "$CSV_PATH".shard*

pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) python "$HERE/run_benchmark.py" "$@" \
    --csv_path "$CSV_PATH.shard$i" --shard "$i/$NUM_SHARDS" > "$CSV_PATH.shard$i.log" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Shard $i failed; see $CSV_PATH.shard$i.log" >&2
    status=1
  fi
done

python - "$CSV_PATH" "$NUM_SHARDS" "$status" <<'PY' || status=1
import os, sys
import pandas as pd

csv_path, n, failed = sys.argv[1], int(sys.argv[2]), sys.argv[3] != "0"
paths = [f"{csv_path}.shard{i}" for i in range(n)]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    print(f"Missing shard reports: {missing}", file=sys.stderr)
if missing or failed:
    print("Not writing merged report: incomplete run", file=sys.stderr)
    sys.exit(1)
# Shards own disjoint tasks, so rows never overlap; outer-join on columns for algo names.
merged = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
dups = merged.duplicated(["task", "metric"])
if dups.any():
    print(f"Duplicate task/metric rows across shards: {merged[dups][['task','metric']].values.tolist()}", file=sys.stderr)
    sys.exit(1)
merged.to_csv(csv_path, index=False)
print(f"Merged {n} shard reports into {csv_path}")
PY
exit $status
