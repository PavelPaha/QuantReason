#!/usr/bin/env bash
# Launch throughput sweep in parallel across multiple GPUs.
# Each group runs on its own GPU.
#
# Usage:
#   bash scripts/launch_throughput_sweep.sh          # default: 4 free GPUs (0,2,3,7)
#   bash scripts/launch_throughput_sweep.sh 0,2,3,7  # explicit GPU list
#
# Groups: fp16, w4a16, w2a16, nvfp4, qwen36, gptq_int4
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-0,2,3,7}"
N_PROMPTS="${2:-64}"

# Split GPU string into array
IFS=',' read -ra GPU_ARR <<< "$GPUS"

GROUP_LIST=(fp16 nvfp4 w4a16 w2a16 qwen36 gptq_int4)
N_GPUS=${#GPU_ARR[@]}

VENV_PYTHON="$(pwd)/.venv/bin/python"
VENV_BIN="$(pwd)/.venv/bin"

mkdir -p results/throughput_qwen results/_logs_throughput

echo "=== Throughput Sweep Launcher ==="
echo "GPUs: ${GPU_ARR[*]}"
echo "n_prompts: $N_PROMPTS"
echo "Groups: ${GROUP_LIST[*]}"
echo ""

# Build per-GPU queues: GPU_QUEUE[gpu_idx]="group1 group2 ..."
declare -A GPU_QUEUE
for i in "${!GROUP_LIST[@]}"; do
    GPU_IDX=$((i % N_GPUS))
    GPU="${GPU_ARR[$GPU_IDX]}"
    GPU_QUEUE[$GPU]="${GPU_QUEUE[$GPU]:-} ${GROUP_LIST[$i]}"
done

echo "Per-GPU group assignment:"
for GPU in "${GPU_ARR[@]}"; do
    echo "  GPU $GPU: ${GPU_QUEUE[$GPU]}"
done
echo ""

# Launch one background worker per GPU that runs its groups sequentially
pids=()
for GPU in "${GPU_ARR[@]}"; do
    QUEUE="${GPU_QUEUE[$GPU]}"
    (
        for GROUP in $QUEUE; do
            LOG="results/_logs_throughput/throughput_${GROUP}_gpu${GPU}.log"
            echo "  [GPU $GPU] starting group=$GROUP → $LOG"
            env \
                PATH="$VENV_BIN:$PATH" \
                VLLM_USE_V1=0 \
                WANDB_DISABLED=true \
                CUDA_VISIBLE_DEVICES="$GPU" \
                "$VENV_PYTHON" scripts/bench_qwen_throughput.py \
                    --group "$GROUP" \
                    --batch-mode bs1,bs16,max \
                    --n-prompts "$N_PROMPTS" \
                    --gpu "$GPU" \
                    --max-new-tokens 512 \
                    --max-model-len 4096 \
                    --warmup 1 \
                    --no-enforce-eager \
                    --gpu-memory-utilization 0.80 \
                    --output-dir "results/throughput_qwen" \
                > "$LOG" 2>&1
            echo "  [GPU $GPU] done group=$GROUP"
        done
    ) &
    pids+=($!)
done

echo "Per-GPU workers launched. PIDs: ${pids[*]}"
echo "Monitor with:"
echo "  tail -f results/_logs_throughput/throughput_fp16_gpu0.log"
echo "  watch -n30 'cat results/throughput_qwen/throughput_summary.csv | column -t -s,'"
echo ""

# Wait for all GPU workers
wait "${pids[@]}"
echo ""
echo "=== All groups done. Results: results/throughput_qwen/ ==="
