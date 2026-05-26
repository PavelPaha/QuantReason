#!/usr/bin/env bash
# Launch perf-sweep for hypotheses H1-H5.
#
# Args:
#   $1: GPU id (e.g. 2)
#   $2: model id (HuggingFace)
#
# Sweeps for the given (gpu, model):
#   - kv_cache_dtype ∈ {auto, fp8} (H5)
#   - max_new_tokens ∈ {2048, 8192, 32768} (H1)
#   - bs1 (memory-bound) and bs16 (compute-bound transition, H4 confirmation)
#
# Each cell writes a JSON+CSV row under results/perf_h1h5/.
set -uo pipefail

GPU="$1"
MODEL="$2"
N_PROMPTS="${N_PROMPTS:-10}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/results/perf_h1h5"
PYTHON="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p "$OUT_DIR"

slug="$(echo "$MODEL" | tr '/' '__')"
LOG="$OUT_DIR/_run_gpu${GPU}_${slug}.log"

echo "=== [$(date +%H:%M:%S)] GPU=$GPU MODEL=$MODEL ===" | tee -a "$LOG"

# bs=1 is the central regime for our paper. Sweep kv-dtype and mnt.
for KV in auto fp8; do
  for MNT in 2048 8192 32768; do
    MML=$((MNT + 4096))
    sub="$OUT_DIR/bs1_kv${KV}_mnt${MNT}"
    mkdir -p "$sub"
    echo ">>> [$(date +%H:%M:%S)] $MODEL bs=1 kv=$KV mnt=$MNT" | tee -a "$LOG"
    "$PYTHON" "$ROOT/scripts/bench_qwen_throughput.py" \
      --model "$MODEL" \
      --batch-mode bs1 \
      --n-prompts "$N_PROMPTS" \
      --max-new-tokens "$MNT" \
      --max-model-len "$MML" \
      --gpu "$GPU" \
      --gpu-memory-utilization 0.85 \
      --kv-cache-dtype "$KV" \
      --no-enforce-eager \
      --warmup 0 \
      --no-isolated \
      --output-dir "$sub" >> "$LOG" 2>&1
    echo "<<< [$(date +%H:%M:%S)] done bs=1 kv=$KV mnt=$MNT" | tee -a "$LOG"
  done
done

# bs=16 only at mnt=8192 — checks compute-bound transition (H4)
for KV in auto; do
  MNT=8192
  MML=$((MNT + 4096))
  sub="$OUT_DIR/bs16_kv${KV}_mnt${MNT}"
  mkdir -p "$sub"
  echo ">>> [$(date +%H:%M:%S)] $MODEL bs=16 kv=$KV mnt=$MNT" | tee -a "$LOG"
  "$PYTHON" "$ROOT/scripts/bench_qwen_throughput.py" \
    --model "$MODEL" \
    --batch-mode bs16 \
    --n-prompts "$N_PROMPTS" \
    --max-new-tokens "$MNT" \
    --max-model-len "$MML" \
    --gpu "$GPU" \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype "$KV" \
    --no-enforce-eager \
    --warmup 0 \
    --no-isolated \
    --output-dir "$sub" >> "$LOG" 2>&1
  echo "<<< [$(date +%H:%M:%S)] done bs=16 kv=$KV mnt=$MNT" | tee -a "$LOG"
done

echo "=== [$(date +%H:%M:%S)] FINISHED $MODEL on GPU $GPU ===" | tee -a "$LOG"
