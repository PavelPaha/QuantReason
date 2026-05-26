#!/usr/bin/env bash
# Launch BS=1, mnt=32k benchmarks for 8B paper models on GPUs 4 and 5
# Models: Qwen3-8B fp16, w4a16, w2a16 × kv={auto,fp8,nvfp4}
#         + 32B w4a16 kv={auto,fp8,nvfp4} (missing from existing data)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
source "$ROOT/scripts/vllm_env.sh"

MNT=32768
MML=$((MNT + 4096))   # 36864
N=16
WARMUP=2
GMU=0.85
OUT_BASE="$ROOT/results/perf_bs1_8b"
mkdir -p "$OUT_BASE"

run_bench() {
    local GPU=$1 MODEL=$2 KV=$3
    local OUT_DIR="$OUT_BASE/bs1_kv${KV}_mnt${MNT}"
    mkdir -p "$OUT_DIR"
    echo ">>> [$(date +%H:%M:%S)] GPU=$GPU  model=$MODEL  kv=$KV"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$ROOT/scripts/bench_qwen_throughput.py" \
        --model "$MODEL" \
        --batch-mode max \
        --max-num-seqs-max 1 \
        --max-model-len $MML \
        --max-new-tokens $MNT \
        --kv-cache-dtype "$KV" \
        --n-prompts $N \
        --warmup $WARMUP \
        --gpu "$GPU" \
        --gpu-memory-utilization $GMU \
        --no-enforce-eager \
        --no-isolated \
        --output-dir "$OUT_DIR" \
        >> "$OUT_DIR/_gpu${GPU}_$(echo "$MODEL" | tr '/' '_').log" 2>&1
    echo "<<< [$(date +%H:%M:%S)] done GPU=$GPU  model=$MODEL  kv=$KV"
}

worker() {
    local GPU="$1"; shift
    while [[ $# -ge 2 ]]; do
        run_bench "$GPU" "$1" "$2" || true
        shift 2
    done
    echo "=== GPU $GPU DONE ==="
}

echo "=== [$(date)] BS=1 8B sweep start ==="

# GPU 4: 8B fp16, 8B w4a16
worker 4 \
    "Qwen/Qwen3-8B"                        auto \
    "Qwen/Qwen3-8B"                        fp8  \
    "Qwen/Qwen3-8B"                        nvfp4 \
    "RedHatAI/Qwen3-8B-quantized.w4a16"    auto \
    "RedHatAI/Qwen3-8B-quantized.w4a16"    fp8  \
    "RedHatAI/Qwen3-8B-quantized.w4a16"    nvfp4 \
    &

# GPU 5: 8B w2a16, 32B w4a16
worker 5 \
    "kaitchup/Qwen3-8B-autoround-2bit-gptq"     auto \
    "kaitchup/Qwen3-8B-autoround-2bit-gptq"     fp8  \
    "kaitchup/Qwen3-8B-autoround-2bit-gptq"     nvfp4 \
    "RedHatAI/Qwen3-32B-quantized.w4a16"        auto \
    "RedHatAI/Qwen3-32B-quantized.w4a16"        fp8  \
    "RedHatAI/Qwen3-32B-quantized.w4a16"        nvfp4 \
    &

wait
echo "=== [$(date)] ALL DONE ==="
