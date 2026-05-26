# Running the experiments

What you need to reproduce the paper runs: final YAML configs, how to launch
quality sweeps and throughput benchmarks, and which library versions were used.

Plotting scripts are **not** in the repo — read JSON/CSV from `results/` and
draw whatever you want.

## Hardware & stack

Single 8×B200 box (183 GB per card, CUDA 12.9 driver). Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-vllm-cu130.txt   # vLLM 0.21 — has nvfp4 KV
pip install -e .
source scripts/vllm_env.sh                   # puts vLLM's libcudart on PATH
```

Pinned versions live in `requirements-vllm-cu130.txt` (vLLM **0.21.0**, torch
2.11+cu130, transformers 5.8). `requirements-vllm-cu124.txt` is an older stack
without `--kv-cache-dtype nvfp4`.

Before editing YAMLs, set `actors[].backend_kwargs.cuda_visible_devices` to
your GPU ids.

## Config layout

Two families of final configs — nothing else under `configs/` is needed for
reproduction:

| Directory | What it is |
|-----------|------------|
| `configs/final/` | Hybrid reasoning baseline: Qwen3-32B FP16 vs GPTQ 2-bit, single- and two-stage variants on 7–9 benchmarks. See [`configs/final/README.md`](configs/final/README.md) for the full matrix and per-dataset commands. |
| `configs/final_qwen32b_fp16/` | FP16 32B baseline cells for the NVFP4 paper table. |
| `configs/final_qwen32b_nvfp4_kv4/` | Qwen3-32B NVFP4 weights + NVFP4 KV at 32k context. |
| `configs/final_qwen8b_nvfp4/` | Qwen3-8B NVFP4 hybrid / single-stage (FP16 planner). |
| `configs/final_qwen8b_moe35b_nvfp4/` | Qwen3.6-35B MoE NVFP4 executor under FP16 planner. |

Regenerate the `configs/final/` YAMLs from templates:

```bash
python scripts/generate_final_configs.py
```

## Data

Benchmark JSON/parquet for ARC, PIQA, WinoGrande, AIME are committed under
`data/`. GSM8K and MATH-500 load from HuggingFace at runtime.

One-time prep if you need to refresh vendored files:

```bash
python scripts/sync_aime2026_data.py
python scripts/prepare_arc_easy_data.py
python scripts/prepare_piqa_data.py
python scripts/prepare_winogrande_data.py
python scripts/prepare_strategyqa_data.py
```

## Quality / accuracy runs

Single config:

```bash
python scripts/run_experiment.py configs/final/math500/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final_qwen8b_nvfp4/math500/hybrid_fp16_nvfp4.yaml -v
```

Artifacts land in `results/<category>/<run_id>/` (`traces.jsonl`,
`judgements.jsonl`, `summary.json`, …).

Staged hybrid runs unload the planner between waves; vLLM EngineCore children
are reaped automatically so the next model fits on the same card.

## Throughput / KV-cache benchmarks

Entry point: `scripts/bench_qwen_throughput.py`. Loads any model from the
built-in catalogue (`--list-models`), runs MATH-500 prompts at the batch size,
`max_new_tokens`, and `--kv-cache-dtype` you pass (`auto`, `fp8`, `nvfp4`, …).

```bash
python scripts/bench_qwen_throughput.py \
    --model "Qwen/Qwen3-32B" \
    --batch-mode max \
    --max-num-seqs-max 8 \
    --n-prompts 16 \
    --max-new-tokens 32768 \
    --max-model-len 36864 \
    --kv-cache-dtype auto \
    --gpu 0 \
    --gpu-memory-utilization 0.85 \
    --no-enforce-eager --warmup 2 --no-isolated \
    --output-dir results/perf_batch_kv/bs8_kvauto_mnt32768
```

The number you want is `throughput_tokens_per_sec` in the output JSON.

Launch scripts wrap the paper sweeps:

| Script | Sweep |
|--------|-------|
| `scripts/launch_perf_h1h5.sh` | BS=1, context × KV dtype (H1–H5) |
| `scripts/launch_8b_bs1_sweep.sh` | BS=1 for 8B model family |
| `scripts/launch_batch_kv_sweep.sh` | BS=8, mnt=32k, dense models × KV dtypes |
| `scripts/launch_moe_kv_sweep.sh` | Same for MoE models |
| `scripts/launch_capacity_sweep.sh` | Max batch that fits per (model, KV dtype) |
| `scripts/launch_perf_nvfp4_kv.sh` | 32B NVFP4 × {auto, fp8, nvfp4} at BS=1 |
| `scripts/launch_perf_queue.sh` | Sequential queue on free GPUs |

Example full perf section (~6 h on 8×B200):

```bash
source scripts/vllm_env.sh
bash scripts/launch_perf_h1h5.sh 0 "Qwen/Qwen3-32B"    # per (gpu, model)
bash scripts/launch_8b_bs1_sweep.sh
bash scripts/launch_batch_kv_sweep.sh
bash scripts/launch_capacity_sweep.sh
bash scripts/launch_moe_kv_sweep.sh
```

Perf sweeps use `--no-isolated` (reuse one Python process). Quality runs keep
the default isolated mode.

MoE models need bf16 weights under vLLM's FlashInfer backend — don't flip
their `precision` to fp16.
