#!/usr/bin/env bash
# Perf sweep: Qwen3-32B-NVFP4 @ 32k context, compare KV cache dtypes.
#
# Usage: launch_perf_nvfp4_kv.sh <gpu_id> [kv_dtype ...]
# Default kv dtypes: auto fp8 nvfp4
set -uo pipefail

GPU="$1"
shift
KV_TYPES=("${@:-auto fp8 nvfp4}")
MODEL="${MODEL:-RedHatAI/Qwen3-32B-NVFP4}"
N_PROMPTS="${N_PROMPTS:-10}"
MNT="${MNT:-32768}"
MML=$((MNT + 4096))

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/results/perf_nvfp4_kv"
PYTHON="$ROOT/.venv/bin/python"
# shellcheck disable=SC1091
source "$ROOT/scripts/vllm_env.sh"
mkdir -p "$OUT_DIR"

slug="$(echo "$MODEL" | tr '/' '__')"
LOG="$OUT_DIR/_run_gpu${GPU}_${slug}.log"

echo "=== [$(date +%H:%M:%S)] GPU=$GPU MODEL=$MODEL mnt=$MNT kv=${KV_TYPES[*]} ===" | tee -a "$LOG"

for KV in "${KV_TYPES[@]}"; do
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

echo "=== [$(date +%H:%M:%S)] FINISHED perf on GPU $GPU ===" | tee -a "$LOG"
