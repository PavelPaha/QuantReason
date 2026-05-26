#!/usr/bin/env bash
# Per-GPU sequential queue of perf runs.
# Args: $1 = GPU id, $@... = list of HF model ids
set -uo pipefail
GPU="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/results/perf_h1h5/_queue_gpu${GPU}.log"
mkdir -p "$(dirname "$LOG")"
echo "=== [$(date)] queue start on GPU $GPU, models: $* ===" | tee -a "$LOG"
for m in "$@"; do
  bash "$ROOT/scripts/launch_perf_h1h5.sh" "$GPU" "$m" 2>&1 | tee -a "$LOG"
done
echo "=== [$(date)] queue DONE on GPU $GPU ===" | tee -a "$LOG"
