#!/usr/bin/env bash
# MoE KV sweep: BS=8, mnt=32k, kv={auto,fp8,nvfp4}, all 8 GPUs
#
# Models (MoE only):
#   GPU 0 → Qwen3-30B-A3B fp16
#   GPU 1 → Qwen3.6-35B-A3B fp16
#   GPU 2 → Qwen3.6-35B-A3B NVFP4
#   GPU 3 → Qwen3.6-35B-A3B 2Bit-GSQ
#   GPU 4 → Qwen3.5-35B-A3B fp16
#   GPU 5 → Qwen3.5-35B-A3B GPTQ-Int4
#   GPU 6 → Qwen3-30B-A3B GPTQ-Int4 (redo consistent setup)
#   GPU 7 → (spare — runs 30B int4 as duplicate for variance check)
#
# VRAM budget at BS=8, mml=36864:
#   35B fp16  : 70GB weights + 8×7.10GB KV = 126.8GB < 152GB ✓ (max BS=11)
#   35B NVFP4 : 17.5GB + 56.8GB KV        =  74.3GB ✓
#   35B 2bit  :  8.8GB + 56.8GB KV        =  65.5GB ✓
#   30B fp16  : 61GB  + 8×4.53GB KV       =  97.2GB ✓

set -uo pipefail

BS=8
MNT=32768
MML=$((MNT + 4096))      # 36864
N_PROMPTS=$((2 * BS))    # 16
WARMUP=2
GMU=0.85

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
source "$ROOT/scripts/vllm_env.sh"

OUT_BASE="$ROOT/results/perf_moe_kv"
mkdir -p "$OUT_BASE"

run_bench() {
    local GPU="$1" MODEL="$2" KV="$3"
    local TAG; TAG="$(echo "$MODEL" | tr '/' '__')"
    local OUT_DIR="$OUT_BASE/bs${BS}_kv${KV}_mnt${MNT}"
    local LOG="$OUT_BASE/_gpu${GPU}_$(echo "$TAG" | cut -c1-35)_kv${KV}.log"
    mkdir -p "$OUT_DIR"
    echo ">>> [$(date +%H:%M:%S)] GPU=$GPU  model=$MODEL  kv=$KV" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/scripts/bench_qwen_throughput.py" \
        --model "$MODEL" \
        --batch-mode max \
        --max-num-seqs-max "$BS" \
        --n-prompts "$N_PROMPTS" \
        --max-new-tokens "$MNT" \
        --max-model-len "$MML" \
        --gpu "$GPU" \
        --gpu-memory-utilization "$GMU" \
        --kv-cache-dtype "$KV" \
        --no-enforce-eager \
        --warmup "$WARMUP" \
        --no-isolated \
        --output-dir "$OUT_DIR" >> "$LOG" 2>&1
    local RC=$?
    echo "<<< [$(date +%H:%M:%S)] exit=$RC  GPU=$GPU  model=$MODEL  kv=$KV" | tee -a "$LOG"
    return $RC
}

worker() {
    local GPU="$1"; shift
    while [[ $# -ge 2 ]]; do
        local MODEL="$1" KV="$2"; shift 2
        run_bench "$GPU" "$MODEL" "$KV" || true
    done
    echo "=== GPU $GPU DONE ===" >> "$OUT_BASE/_gpu${GPU}_done.log"
}

echo "=== [$(date)] MoE KV sweep  BS=$BS mnt=$MNT GPUs=0-7 ===" | tee "$OUT_BASE/_sweep.log"

# GPU 0 — 30B-A3B fp16
worker 0 \
    "Qwen/Qwen3-30B-A3B" auto \
    "Qwen/Qwen3-30B-A3B" fp8 \
    "Qwen/Qwen3-30B-A3B" nvfp4 &

# GPU 1 — Qwen3.6-35B-A3B fp16
worker 1 \
    "Qwen/Qwen3.6-35B-A3B" auto \
    "Qwen/Qwen3.6-35B-A3B" fp8 \
    "Qwen/Qwen3.6-35B-A3B" nvfp4 &

# GPU 2 — Qwen3.6-35B-A3B NVFP4
worker 2 \
    "RedHatAI/Qwen3.6-35B-A3B-NVFP4" auto \
    "RedHatAI/Qwen3.6-35B-A3B-NVFP4" fp8 \
    "RedHatAI/Qwen3.6-35B-A3B-NVFP4" nvfp4 &

# GPU 3 — Qwen3.6-35B-A3B 2Bit-GSQ
worker 3 \
    "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ" auto \
    "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ" fp8 \
    "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ" nvfp4 &

# GPU 4 — Qwen3.5-35B-A3B fp16
worker 4 \
    "Qwen/Qwen3.5-35B-A3B" auto \
    "Qwen/Qwen3.5-35B-A3B" fp8 \
    "Qwen/Qwen3.5-35B-A3B" nvfp4 &

# GPU 5 — Qwen3.5-35B-A3B GPTQ-Int4
worker 5 \
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" auto \
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" fp8 \
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" nvfp4 &

# GPU 6 — Qwen3-30B-A3B GPTQ-Int4
worker 6 \
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4" auto \
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4" fp8 \
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4" nvfp4 &

# GPU 7 — Qwen3.6-35B-A3B NVFP4 duplicate (variance check) + 2Bit-GSQ extra
worker 7 \
    "Qwen/Qwen3.6-35B-A3B" fp8 \
    "Qwen/Qwen3.6-35B-A3B" nvfp4 &

wait
echo "=== [$(date)] ALL WORKERS DONE ===" | tee -a "$OUT_BASE/_sweep.log"
