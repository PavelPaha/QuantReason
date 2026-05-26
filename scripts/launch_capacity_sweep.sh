#!/usr/bin/env bash
# Batch-capacity sweep: run each model at the MAX batch size allowed by each KV dtype.
#
# VRAM budget (B200 183GB × gmu=0.85 → 155 GB):
#   Qwen3-32B w2a16  (weights=8.2GB, KV/seq fp16=9.0GB @mml=36864):
#     fp16 KV → max_bs = floor((155-8.2)/9.00) = 16
#     fp8  KV → max_bs = floor((155-8.2)/4.50) = 32
#     nvfp4 KV→ max_bs = floor((155-8.2)/2.25) = 65 (cap at 64)
#
#   Qwen3-8B w4a16   (weights=4.1GB, KV/seq fp16=5.06GB):
#     fp16 KV → max_bs = floor((155-4.1)/5.06) = 29
#     fp8  KV → max_bs = floor((155-4.1)/2.53) = 59 (cap at 56)
#     nvfp4 KV→ max_bs = floor((155-4.1)/1.27) = 118 (cap at 64)
#
# n_prompts = max(16, BS) so at least 1 full batch is measured.
# Runs on GPUs 0-3 (assumed free from MoE runs).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
source "$ROOT/scripts/vllm_env.sh"

MNT=32768
MML=$((MNT + 4096))   # 36864
WARMUP=2
GMU=0.85
OUT_BASE="$ROOT/results/perf_capacity"
mkdir -p "$OUT_BASE"

run_bench() {
    local GPU=$1 MODEL=$2 KV=$3 BS=$4
    local N; N=$(( BS > 16 ? BS : 16 ))
    local OUT_DIR="$OUT_BASE/bs${BS}_kv${KV}_mnt${MNT}"
    mkdir -p "$OUT_DIR"
    echo ">>> [$(date +%H:%M:%S)] GPU=$GPU  model=$MODEL  kv=$KV  BS=$BS  n=$N"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/scripts/bench_qwen_throughput.py" \
        --model "$MODEL" \
        --batch-mode max \
        --max-num-seqs-max "$BS" \
        --max-model-len "$MML" \
        --max-new-tokens "$MNT" \
        --kv-cache-dtype "$KV" \
        --n-prompts "$N" \
        --warmup "$WARMUP" \
        --gpu "$GPU" \
        --gpu-memory-utilization "$GMU" \
        --no-enforce-eager \
        --no-isolated \
        --output-dir "$OUT_DIR" \
        >> "$OUT_BASE/_gpu${GPU}_$(echo "$MODEL" | tr '/' '_')_kv${KV}_bs${BS}.log" 2>&1
    echo "<<< [$(date +%H:%M:%S)] done GPU=$GPU  model=$MODEL  kv=$KV  BS=$BS"
}

worker() {
    local GPU="$1"; shift
    while [[ $# -ge 3 ]]; do
        run_bench "$GPU" "$1" "$2" "$3" || true
        shift 3
    done
    echo "=== GPU $GPU DONE ===" | tee "$OUT_BASE/_gpu${GPU}_done.log"
}

echo "=== [$(date)] Capacity sweep start  mnt=$MNT GPUs=0-3 ===" | tee "$OUT_BASE/_sweep.log"

# GPU 0: Qwen3-32B w2a16 at max BS for each KV dtype
# weights=8.2GB; KV/seq fp16=9.0GB → max=16; fp8=32; nvfp4=64
worker 0 \
    "kaitchup/Qwen3-32B-autoround-2bit-gptq"  auto   16 \
    "kaitchup/Qwen3-32B-autoround-2bit-gptq"  fp8    32 \
    "kaitchup/Qwen3-32B-autoround-2bit-gptq"  nvfp4  64 \
    &

# GPU 1: Qwen3-8B w4a16 at max BS for each KV dtype
# weights=4.1GB; KV/seq fp16=5.06GB → max=29; fp8=56; nvfp4=64 (capped)
worker 1 \
    "RedHatAI/Qwen3-8B-quantized.w4a16"  auto   29 \
    "RedHatAI/Qwen3-8B-quantized.w4a16"  fp8    56 \
    "RedHatAI/Qwen3-8B-quantized.w4a16"  nvfp4  64 \
    &

# GPU 2: Qwen3-8B fp16 at max BS for each KV dtype
# weights=16.4GB; KV/seq fp16=5.06GB → max=27; fp8=54; nvfp4=64 (capped)
worker 2 \
    "Qwen/Qwen3-8B"  auto   27 \
    "Qwen/Qwen3-8B"  fp8    54 \
    "Qwen/Qwen3-8B"  nvfp4  64 \
    &

# GPU 3: Qwen3-32B w4a16 at max BS for each KV dtype
# weights=16.4GB; KV/seq fp16=9.0GB → max=15; fp8=30; nvfp4=60
worker 3 \
    "RedHatAI/Qwen3-32B-quantized.w4a16"  auto   15 \
    "RedHatAI/Qwen3-32B-quantized.w4a16"  fp8    30 \
    "RedHatAI/Qwen3-32B-quantized.w4a16"  nvfp4  60 \
    &

wait
echo "=== [$(date)] ALL WORKERS DONE ===" | tee -a "$OUT_BASE/_sweep.log"
