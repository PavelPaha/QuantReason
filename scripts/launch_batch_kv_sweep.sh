#!/usr/bin/env bash
# Batch KV sweep: BS=8, mnt=32k, kv={auto,fp8,nvfp4}, GPUs 0-5
#
# Max batch analysis (B200 183GB, gmu=0.85 → 152GB available, mml=36864):
#   32B fp16:   weights=65.6GB, KV/seq=9.66GB → max BS=8  (binding constraint)
#   32B w4a16:  weights=16.4GB, KV/seq=9.66GB → max BS=14
#   32B w2a16:  weights= 8.2GB, KV/seq=9.66GB → max BS=14
#   32B NVFP4:  weights=16.4GB, KV/seq=9.66GB → max BS=14
#   8B  fp16:   weights=16.4GB, KV/seq=5.44GB → max BS=24
#   8B  w4a16:  weights= 4.1GB, KV/seq=5.44GB → max BS=27
#   8B  NVFP4:  weights= 4.1GB, KV/seq=5.44GB → max BS=27
#   MoE-30B:    weights=15.2GB, KV/seq=3.62GB → max BS=37
#
# Universal safe BS=8 (constrained by 32B fp16 with fp16 KV).
#
# GPU assignment (GPUs 6,7 excluded — busy):
#   GPU 0 → Qwen3-32B fp16          (kv=auto, fp8, nvfp4)
#   GPU 1 → Qwen3-32B w4a16         (kv=auto, fp8, nvfp4)
#   GPU 2 → Qwen3-32B w2a16         (kv=auto, fp8, nvfp4)
#   GPU 3 → Qwen3-32B NVFP4         (kv=auto, fp8, nvfp4)
#   GPU 4 → Qwen3-8B fp16 + w4a16   (kv=auto, fp8, nvfp4)
#   GPU 5 → Qwen3-8B NVFP4 + MoE    (kv=auto, fp8, nvfp4)

set -uo pipefail

BS=8
MNT=32768
MML=$((MNT + 4096))      # 36864
N_PROMPTS=$((2 * BS))    # 16  — 2 полных батча, достаточно для замера throughput
WARMUP=2
GMU=0.85

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
source "$ROOT/scripts/vllm_env.sh"

OUT_BASE="$ROOT/results/perf_batch_kv"
mkdir -p "$OUT_BASE"

run_bench() {
    local GPU="$1" MODEL="$2" KV="$3"
    local TAG
    TAG="$(echo "$MODEL" | tr '/' '__')"
    local OUT_DIR="$OUT_BASE/bs${BS}_kv${KV}_mnt${MNT}"
    local LOG="$OUT_BASE/_gpu${GPU}_$(echo "$TAG" | cut -c1-30)_kv${KV}.log"
    mkdir -p "$OUT_DIR"
    echo ">>> [$(date +%H:%M:%S)] GPU=$GPU  model=$MODEL  kv=$KV  BS=$BS  mnt=$MNT" | tee -a "$LOG"
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
    local GPU="$1"
    shift
    # remaining args: model kv model kv ...
    while [[ $# -ge 2 ]]; do
        local MODEL="$1" KV="$2"
        shift 2
        run_bench "$GPU" "$MODEL" "$KV" || true
    done
    echo "=== [$(date +%H:%M:%S)] GPU $GPU DONE ===" >> "$OUT_BASE/_gpu${GPU}_done.log"
}

echo "=== [$(date)] Batch KV sweep start  BS=$BS mnt=$MNT GPUs=0-5 ===" | tee "$OUT_BASE/_sweep.log"

# GPU 0 — 32B fp16
worker 0 \
    "Qwen/Qwen3-32B" auto \
    "Qwen/Qwen3-32B" fp8 \
    "Qwen/Qwen3-32B" nvfp4 \
    &

# GPU 1 — 32B w4a16
worker 1 \
    "RedHatAI/Qwen3-32B-quantized.w4a16" auto \
    "RedHatAI/Qwen3-32B-quantized.w4a16" fp8 \
    "RedHatAI/Qwen3-32B-quantized.w4a16" nvfp4 \
    &

# GPU 2 — 32B w2a16
worker 2 \
    "kaitchup/Qwen3-32B-autoround-2bit-gptq" auto \
    "kaitchup/Qwen3-32B-autoround-2bit-gptq" fp8 \
    "kaitchup/Qwen3-32B-autoround-2bit-gptq" nvfp4 \
    &

# GPU 3 — 32B NVFP4
worker 3 \
    "RedHatAI/Qwen3-32B-NVFP4" auto \
    "RedHatAI/Qwen3-32B-NVFP4" fp8 \
    "RedHatAI/Qwen3-32B-NVFP4" nvfp4 \
    &

# GPU 4 — 8B fp16 + 8B w4a16
worker 4 \
    "Qwen/Qwen3-8B" auto \
    "Qwen/Qwen3-8B" fp8 \
    "Qwen/Qwen3-8B" nvfp4 \
    "RedHatAI/Qwen3-8B-quantized.w4a16" auto \
    "RedHatAI/Qwen3-8B-quantized.w4a16" fp8 \
    "RedHatAI/Qwen3-8B-quantized.w4a16" nvfp4 \
    &

# GPU 5 — 8B NVFP4 + 30B MoE
worker 5 \
    "RedHatAI/Qwen3-8B-NVFP4" auto \
    "RedHatAI/Qwen3-8B-NVFP4" fp8 \
    "RedHatAI/Qwen3-8B-NVFP4" nvfp4 \
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4" auto \
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4" fp8 \
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4" nvfp4 \
    &

wait
echo "=== [$(date)] ALL WORKERS DONE ===" | tee -a "$OUT_BASE/_sweep.log"
