#!/usr/bin/env bash
# Sweep Qwen3 throughput: bs1, bs16, max batch modes.
#
# Usage:
#   bash scripts/run_qwen_throughput_sweep.sh                  # all groups, GPU 0
#   bash scripts/run_qwen_throughput_sweep.sh fp16 1           # fp16 only, GPU 1
#   bash scripts/run_qwen_throughput_sweep.sh w4a16 2 32        # w4a16, GPU 2, 32 prompts
#
# Each model runs in its own Python subprocess (--isolated, default in bench script).
# Requires: bash scripts/install_vllm_b200.sh --nightly  (on B200)
set -euo pipefail
cd "$(dirname "$0")/.."

GROUP="${1:-all}"
GPU="${2:-0}"
N_PROMPTS="${3:-64}"
OUT="results/throughput_qwen/gpu${GPU}_${GROUP}_n${N_PROMPTS}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="results/_logs_gpu${GPU}/qwen_throughput_${GROUP}_n${N_PROMPTS}_${TS}.log"
mkdir -p "$(dirname "$LOG")" "$OUT"

run_group() {
  local g="$1"
  echo "=== group=$g gpu=$GPU n_prompts=$N_PROMPTS ==="
  python scripts/bench_qwen_throughput.py \
    --group "$g" \
    --batch-mode bs1,bs16,max \
    --n-prompts "$N_PROMPTS" \
    --gpu "$GPU" \
    --output-dir "$OUT/$g" \
    "$@"
}

COMMON=(--max-new-tokens 512 --max-model-len 4096 --warmup 1 --enforce-eager)

if [[ "$GROUP" == "all" ]]; then
  {
    for g in fp16 w4a16 w2a16 nvfp4 qwen36 gptq_int4; do
      run_group "$g" "${COMMON[@]}"
    done
  } 2>&1 | tee "$LOG"
else
  run_group "$GROUP" "${COMMON[@]}" 2>&1 | tee "$LOG"
fi

echo "Done. Log: $LOG"
echo "Results: $OUT"
