# Experiment configs

All reproduction YAMLs live under `configs/<dataset>/`:

```
configs/math500/
├── qwen32b_fp16/                 # Qwen3-32B FP16 (single + hybrid FP16→FP16)
│   ├── single_fp16.yaml
│   └── hybrid_fp16_fp16.yaml
├── qwen32b_gptq2bit/             # GPTQ 2-bit (single + hybrid FP16 plan → GPTQ)
│   ├── single_gptq2bit.yaml
│   └── hybrid_fp16_gptq2bit.yaml
├── qwen32b_nvfp4/                # Qwen3-32B NVFP4 weights, default KV cache
│   ├── single_nvfp4.yaml
│   └── hybrid_fp16_nvfp4.yaml    # 32B FP16 plan → 32B NVFP4
├── qwen32b_nvfp4_kv4/            # Qwen3-32B NVFP4 + NVFP4 KV cache
│   ├── single_nvfp4_kv4.yaml
│   └── hybrid_fp16_nvfp4_kv4.yaml
├── qwen8b_fp16/                  # Qwen3-8B FP16 baselines (paper comparison)
│   ├── single_fp16.yaml
│   └── hybrid_fp16_fp16.yaml
├── qwen8b_nvfp4/                 # Qwen3-8B / 32B NVFP4 family
│   ├── single_8b_nvfp4.yaml
│   ├── hybrid_fp16_nvfp4.yaml    # 8B FP16 plan → 32B NVFP4
│   └── hybrid_fp16_8b_nvfp4.yaml # 8B FP16 plan → 8B NVFP4
└── qwen8b_moe35b_nvfp4/          # MoE 35B NVFP4 single + hybrid
    ├── single_moe35b_nvfp4.yaml
    └── hybrid_fp16_moe35b_nvfp4.yaml
```

| Path pattern | Description |
|--------------|-------------|
| `configs/<dataset>/qwen32b_fp16/` | Qwen3-32B FP16 single- and hybrid-FP16 variants |
| `configs/<dataset>/qwen32b_gptq2bit/` | GPTQ 2-bit single- and hybrid variants |
| `configs/<dataset>/qwen32b_nvfp4/` | Qwen3-32B NVFP4 single + 32B FP16→32B-NVFP4 hybrid |
| `configs/<dataset>/qwen32b_nvfp4_kv4/` | Qwen3-32B NVFP4 + NVFP4 KV single/hybrid |
| `configs/<dataset>/qwen8b_fp16/` | Qwen3-8B FP16 single/hybrid baselines for NVFP4 paper sweeps |
| `configs/<dataset>/qwen8b_nvfp4/` | 8B NVFP4 single; hybrid 8B→32B-NVFP4 and 8B→8B-NVFP4 |
| `configs/<dataset>/qwen8b_moe35b_nvfp4/` | Qwen3.6-35B MoE NVFP4 single + FP16 planner hybrid |

Nine datasets: `aime2026`, `arc_challenge`, `arc_easy`, `gpqa_diamond`, `gsm8k`, `math500`, `piqa`, `strategyqa`, `winogrande`.

The sections below document the **hybrid baseline** matrix. NVFP4 families use the same variant names inside their subfolder.

## Hybrid baseline — variant matrix (36 configs)

Four variants per dataset under `qwen32b_fp16/` or `qwen32b_gptq2bit/`:

| Variant | Pipeline |
|---------|----------|
| `single_fp16` | One stage, full FP16 answer |
| `single_gptq2bit` | One stage, GPTQ 2-bit answer |
| `hybrid_fp16_fp16` | FP16 plan → FP16 reasoning |
| `hybrid_fp16_gptq2bit` | FP16 plan → GPTQ 2-bit reasoning |

| Dataset | Task type | Notes |
|---------|-----------|--------|
| `gsm8k/` | numeric, `\boxed{}` | 500 examples |
| `math500/` | math, `\boxed{}` | 500 examples |
| `aime2026/` | integer 0–999 | 30 examples |
| `arc_easy/` | MCQ (A–D) | full test split (`max_examples: null`) |
| `arc_challenge/` | MCQ (A–D) | full test split |
| `gpqa_diamond/` | MCQ | ~198 examples |
| `strategyqa/` | yes/no | full test split |
| `winogrande/` | MCQ (1 / 2) | winogrande_xl validation |
| `piqa/` | MCQ (1 / 2) | validation split |

- **single** — one `answer` stage.
- **hybrid** — plan (FP16, up to 1024 tokens, `plan_scaffold` handoff) → reasoning; plan text is injected into **user** (`prompt_plan_in_user`).

Regenerate all YAMLs from the template (default GPU `"0"`):

```bash
python scripts/generate_final_configs.py
# then set cuda_visible_devices in the YAMLs you will run
```

## Data prep (one time)

Vendored files live under `data/`. Refresh if needed:

```bash
python scripts/sync_aime2026_data.py
python scripts/prepare_arc_easy_data.py
python scripts/prepare_piqa_data.py
python scripts/prepare_winogrande_data.py
python scripts/prepare_strategyqa_data.py
```

GSM8K and MATH-500 are fetched from HuggingFace `datasets` at run time.

## Before you run

1. Set GPU ids in each YAML: `actors[].backend_kwargs.cuda_visible_devices` (e.g. `"5"` or `"4,5"` for TP=2).
2. For **GPQA FP16 single** at 32k context, if you hit OOM use `tensor_parallel_size: 2` on two cards (see comment in `gpqa_diamond/single_fp16.yaml`).
3. Staged hybrid runs unload models between waves; vLLM EngineCore children are reaped automatically.

## KV cache dtype

vLLM reads `kv_cache_dtype` from `actors[].backend_kwargs` (forwarded to `vllm.LLM(...)`). If the field is **omitted**, vLLM uses **`auto`** (KV dtype follows model precision — typically bf16/fp16).

Supported values (vLLM 0.21+): `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `nvfp4`.

```yaml
actors:
- actor_id: qwen3_32b_nvfp4
  model_id: RedHatAI/Qwen3-32B-NVFP4
  backend: vllm
  backend_kwargs:
    cuda_visible_devices: '5'
    max_model_len: 32768
    kv_cache_dtype: nvfp4   # fp8 | nvfp4 | auto (omit = auto)
```

In **hybrid** configs set `kv_cache_dtype` on each actor separately if plan and reason stages need different KV settings.

Built-in comparison pair (32B NVFP4, same model, different KV):

| Config | KV cache |
|--------|----------|
| `qwen32b_nvfp4/single_nvfp4.yaml` | default (`auto`) |
| `qwen32b_nvfp4_kv4/single_nvfp4_kv4.yaml` | `nvfp4` |

To try another dtype on an existing config, add or change `kv_cache_dtype` under the relevant actor's `backend_kwargs` (or copy the YAML to a new variant name).

For **throughput** sweeps across KV dtypes, use `scripts/bench_qwen_throughput.py --kv-cache-dtype …` — see the root [README](../README.md#throughput--kv-cache-benchmarks).

## Running one config

From the repo root:

```bash
python scripts/run_experiment.py configs/<dataset>/qwen32b_fp16/<variant>.yaml -v
python scripts/run_experiment.py configs/<dataset>/qwen32b_gptq2bit/<variant>.yaml -v
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `-v` | Verbose log |
| `--max-examples N` | Override example count (quick smoke) |
| `--output-dir PATH` | Change results base directory |
| `--staged-batch-size N` | vLLM micro-batch per staged wave |

Artifacts: `results/<category>/<run_id>/` — `config.json`, `traces.jsonl`, `judgements.jsonl`, `summary.json`.

---

## Launch commands (all 36 configs)

Set GPUs in the YAMLs first. Paths are relative to the repo root.

### GSM8K

```bash
python scripts/run_experiment.py configs/gsm8k/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/gsm8k/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/gsm8k/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/gsm8k/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### MATH-500

```bash
python scripts/run_experiment.py configs/math500/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### AIME-2026

```bash
python scripts/run_experiment.py configs/aime2026/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/aime2026/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/aime2026/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/aime2026/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### ARC-Easy

```bash
python scripts/run_experiment.py configs/arc_easy/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/arc_easy/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/arc_easy/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/arc_easy/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### ARC-Challenge

```bash
python scripts/run_experiment.py configs/arc_challenge/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/arc_challenge/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/arc_challenge/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/arc_challenge/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### GPQA Diamond

```bash
python scripts/run_experiment.py configs/gpqa_diamond/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/gpqa_diamond/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/gpqa_diamond/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/gpqa_diamond/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### StrategyQA

```bash
python scripts/run_experiment.py configs/strategyqa/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/strategyqa/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/strategyqa/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/strategyqa/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### WinoGrande

```bash
python scripts/run_experiment.py configs/winogrande/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/winogrande/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/winogrande/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/winogrande/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

### PIQA

```bash
python scripts/run_experiment.py configs/piqa/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/piqa/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/piqa/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/piqa/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

---

## Prompts and evaluation

- **GSM8K / MATH-500 / AIME (single)**: problem + `\boxed{}` in user; assistant opens the thinking block.
- **AIME**: answer is an integer in 0–999.
- **ARC / GPQA**: multiple choice, letter `answerKey`.
- **StrategyQA**: yes/no.
- **WinoGrande / PIQA**: answer `1` or `2`.
- **Hybrid plan stage**: `plan_scaffold` handoff, empty closed thinking block.
- **Hybrid reason stage**: plan in user + open thinking block.

## Context (full runs)

- `max_model_len: 32768`, `max_total_tokens: 32768`, generation up to **30720** tokens (plan capped at 1024).
- `staged_batch_size`: **500** (gsm8k, math500, arc_*, gpqa, strategyqa, winogrande, piqa), **30** (aime2026).

## NVFP4 and MoE config families

Same benchmark folders, different model stacks:

```bash
# examples
python scripts/run_experiment.py configs/math500/qwen8b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_nvfp4/hybrid_fp16_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_nvfp4_kv4/hybrid_fp16_nvfp4_kv4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_nvfp4/single_8b_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_nvfp4/hybrid_fp16_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_nvfp4/hybrid_fp16_8b_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_nvfp4_kv4/single_nvfp4_kv4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_moe35b_nvfp4/hybrid_fp16_moe35b_nvfp4.yaml -v
```

See the root [README](../README.md) for environment setup and throughput benchmarks.
