# Multi-stage reasoning acceleration

## Why this project exists

Long reasoning tasks — math, hard QA benchmarks, multi-step chains — are costly not only in output tokens but also in generation time. The model does not simply append a short answer: it builds a plan, runs intermediate steps, self-checks, sometimes loops, or takes a long time before producing a final answer in the required format (e.g. `\boxed{...}`).

This repository explores whether that workload can be sped up via **hybrid execution** of the reasoning trace. You do not have to generate the entire chain with one full-precision model. Different parts of the trace may have different sensitivity to precision:

- **planning** is better left to a full-precision model;
- the **long main reasoning phase** can be delegated to a quantized / low-bit actor;
- **finalization, verification, and escape from loops** can again use a more accurate model.

The goal is not abstract “faster vs slower inference”, but **reasoning-pipeline speedup**: where wall-clock, throughput, and token cost improve, and what you pay in accuracy, parse rate, reasoning length, loop failures, commit gap, and other quality signals.

The central object is **`Trace`** (`quantlab/core/trace.py`). For each example, one reasoning history is built and extended sequentially by different **actors**. Each actor appends a **`TraceSegment`**: which backend and model produced the chunk, precision mode, token count, pipeline role, and — when available — how long generation took.

Between segments there is a **handoff**: the next model either receives the generated trace as a text prefix and does a full prefill, or continues via a lower-level state transfer if the backend supports it. The trace is one coherent solution record, not a set of independent generations.

After the pipeline finishes, the runner:

- extracts the final answer;
- runs the benchmark-specific judge;
- computes standard quality metrics;
- computes generation-behavior metrics;
- aggregates per-segment timing.

Results are stored as structured artifacts (`traces.jsonl`, summary tables, metric files) so you can compare execution schemes: full-precision baseline, quantized-only, full → low-bit → full, per-segment vLLM replay, and other variants.

The rest of this README describes which pipelines are stable to run in practice, what lands in `results/`, and how to add metrics, benchmark adapters, actors, and standalone latency replay.

## Running the experiments

What you need to reproduce the paper runs: final YAML configs, how to launch quality sweeps and throughput benchmarks, and which library versions were used.

Plotting scripts are **not** in the repo — read JSON/CSV from `results/` and draw whatever you want.

### Hardware & stack

Single 8×B200 box (183 GB per card, CUDA 12.9 driver). Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-vllm-cu130.txt   # vLLM 0.21 — has nvfp4 KV
pip install -e .
source scripts/vllm_env.sh                   # puts vLLM's libcudart on PATH
```

Pinned versions live in `requirements-vllm-cu130.txt` (vLLM **0.21.0**, torch 2.11+cu130, transformers 5.8). `requirements-vllm-cu124.txt` is an older stack without `--kv-cache-dtype nvfp4`.

Before editing YAMLs, set `actors[].backend_kwargs.cuda_visible_devices` to your GPU ids.

### Config layout

Everything lives under `configs/<dataset>/`:

| Path | What it is |
|------|------------|
| `configs/<dataset>/qwen32b_fp16/*.yaml` | Qwen3-32B FP16 single- and hybrid-FP16 variants |
| `configs/<dataset>/qwen32b_gptq2bit/*.yaml` | GPTQ 2-bit single- and hybrid variants |
| `configs/<dataset>/qwen32b_nvfp4/` | Qwen3-32B NVFP4, default KV cache |
| `configs/<dataset>/qwen32b_nvfp4_kv4/` | Qwen3-32B NVFP4 + NVFP4 KV at 32k |
| `configs/<dataset>/qwen8b_fp16/` | Qwen3-8B FP16 baselines (NVFP4 paper comparison) |
| `configs/<dataset>/qwen8b_nvfp4/` | Qwen3-8B NVFP4 single + FP16→32B-NVFP4 hybrid |
| `configs/<dataset>/qwen8b_moe35b_nvfp4/` | Qwen3.6-35B MoE NVFP4 single + hybrid |

Regenerate the `configs/` YAMLs from templates:

```bash
python scripts/generate_final_configs.py
```

### Data

Benchmark JSON/parquet for ARC, PIQA, WinoGrande, and AIME are committed under `data/`. GSM8K and MATH-500 load from HuggingFace at runtime.

One-time prep if you need to refresh vendored files:

```bash
python scripts/sync_aime2026_data.py
python scripts/prepare_arc_easy_data.py
python scripts/prepare_piqa_data.py
python scripts/prepare_winogrande_data.py
python scripts/prepare_strategyqa_data.py
```

### Quality / accuracy runs

Single config:

```bash
python scripts/run_experiment.py configs/math500/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_nvfp4/hybrid_fp16_nvfp4.yaml -v
```

Artifacts land in `results/<category>/<run_id>/` (`traces.jsonl`, `judgements.jsonl`, `summary.json`, …).

Staged hybrid runs unload the planner between waves; vLLM EngineCore children are reaped automatically so the next model fits on the same card.

#### KV cache dtype

For accuracy runs, set vLLM's KV cache type per actor in the YAML:

```yaml
actors:
- actor_id: ...
  backend: vllm
  backend_kwargs:
    kv_cache_dtype: nvfp4   # optional; omit = auto (default)
```

Values: `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `nvfp4` (requires vLLM 0.21+). In hybrid configs you can set it independently on the plan and reason actors.

Ready-made pair for 32B NVFP4: `qwen32b_nvfp4/single_nvfp4.yaml` (default KV) vs `qwen32b_nvfp4_kv4/single_nvfp4_kv4.yaml` (`kv_cache_dtype: nvfp4`). Details and examples: [configs/README.md](configs/README.md#kv-cache-dtype-accuracy-runs).

### Throughput / KV-cache benchmarks

Entry point: `scripts/bench_qwen_throughput.py`. Loads any model from the built-in catalogue (`--list-models`), runs MATH-500 prompts at the batch size, `max_new_tokens`, and `--kv-cache-dtype` you pass (`auto`, `fp8`, `nvfp4`, …).

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

Perf sweeps use `--no-isolated` (reuse one Python process). Quality runs keep the default isolated mode.

MoE models need bf16 weights under vLLM's FlashInfer backend — don't flip their `precision` to fp16.

## What has been exercised in practice

These modes were used in real runs, not just defined in code:

- **MATH-500 hybrid pipeline** (`configs/math500/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml`): plan on **full BF16 Qwen3-32B**, answer on **2-bit GPTQ** via vLLM; **staged** (plan all examples, then answers), trace/metric persistence, optional resume from wave checkpoints.
- **Same tasks, full FP16** (`configs/math500/qwen32b_fp16/hybrid_fp16_fp16.yaml`): both stages on one FP16 model for comparison with hybrid.
- **Timing from saved traces**: `scripts/replay_timing.py` re-runs segments through vLLM (pin one model or follow `actor_id` from config).
- On shared-GPU vLLM setups, use reasonable `gpu_memory_utilization` and a working `PATH` (including `ninja` for FlashInfer JIT under `nohup`).

### Hybrid vs FP16 plan–answer on one slice (n=50, `seed=42`)

On saved runs, **hybrid** (BF16 plan → **2-bit** answer) and **FP16 baseline** (same two stages, both on **Qwen3-32B FP16**) can show **accuracy** / **parse_rate** that look “against” the larger model. The gap is mostly about **judge output format**, not full precision being worse per se: the baseline’s second stage often hits **`max_new_tokens: 4096`**, text stops **before** final `\boxed{…}`, and the judge marks **`parse_success = false`** (long “re-check” loops before answering). Hybrid’s second actor follows a different trajectory and typical answer length, so **successful answer extraction is more frequent**.

**Temperature:** both configs use **`temperature: 0.0`** (greedy, no extra stochasticity). The observed gap is **not** “different temperatures between runs”. Intuition: higher temperature generally **increases length variance and verbosity**; with the same token cap it might change how often `\boxed{}` appears. Here the effect is dominated by **`max_new_tokens` + second-stage generation patterns** (full vs quantized), not sampling.

**KV cache transfer between different models** and true cross-wave KV handoff were **not validated** in these runs. The repo has hints and an example config (`kv_cache_handoff.yaml`), but reported comparisons rely on **`full_prefill`** (full text in the prompt on the next stage).

---

## Optional: Weights & Biases logging

Add a block to the config:

```yaml
wandb:
  enabled: true
  project: quantlab
  entity: your-team   # optional
  group: math500
  tags: ["hybrid", "math500"]
  mode: online        # or offline
  progress_log_interval: 10
  log_per_example_table: true
  per_example_table_key: per_example_metrics
  upload_run_artifact: false
```

Then the runner:

- logs run-level metrics to W&B;
- mirrors numeric fields from `summary.json` as `summary/*` scalars for cross-run charts;
- streams progress as examples are processed;
- attaches a per-example table at the end (`example_id`, `experiment_name`, `run_id`, judgement fields, per-example metrics);
- optionally uploads the whole run folder as an artifact.

If `wandb` is not installed or the `wandb:` block is missing, local runs behave as before: artifacts go to `results/<run_id>/` and external logging is skipped.

Upload a finished run from saved artifacts:

```bash
python scripts/log_saved_run_to_wandb.py results/<run_id> --project quantlab
# or reuse wandb settings from a YAML:
python scripts/log_saved_run_to_wandb.py results/<run_id> \
  --wandb-config configs/math500/qwen32b_fp16/hybrid_fp16_fp16.yaml
```

---

## Running an experiment from YAML

```bash
python scripts/run_experiment.py configs/math500/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `-v`, `--verbose` | Verbose console log |
| `--max-examples N` | Override `benchmark.max_examples` in YAML |
| `--output-dir DIR` | Base directory for `results/` (default from YAML) |
| `--staged` / `--no-staged` | Wave mode: all examples finish stage 0, then stage 1, … Overrides `staged_execution` in YAML |
| `--staged-batch-size K` | vLLM micro-batch **per wave** (K≥2): higher throughput; see timing caveat below |
| `--resume-run-id ID` | Continue in the same `results/<ID>/` folder |
| `--resume-after-wave W` | Staged: wave W is **fully** done for all examples (start at W+1). Without the flag — auto from latest `trace_checkpoints/wave_*.jsonl`, including resuming a partial wave |

Limited staged run example:

```bash
python scripts/run_experiment.py configs/math500/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml \
  --staged -v --max-examples 50
```

Logs will show `run_id=…` and a path like `results/<run_id>/`.

**Resume interrupted run (non-staged):** if some examples already exist in `traces.jsonl` / `judgements.jsonl`, use the same `output.base_dir` and previous `run_id`:

```bash
python scripts/run_experiment.py path/to/cfg.yaml -v \
  --resume-run-id 20260519_101131_52f75c
```

New rows append to the same jsonl files; `summary.json` is recomputed at the end from all judgements.

**Staged (interrupted wave / stage):** with `staged_wave_checkpoints: true`, `trace_checkpoints/wave_<n>.jsonl` updates after each example. Resume:

```bash
python scripts/run_experiment.py path/to/cfg.yaml --staged -v \
  --resume-run-id 20260512_012442_a81368
```

The runner picks up the latest `wave_*.jsonl`, finishes incomplete examples on the current stage, then continues waves. To pin the last **fully** completed wave: `--resume-after-wave W`.

### How models are loaded (default)

In a typical **staged** run (`--staged` or `staged_execution: true` in YAML), execution proceeds **by pipeline depth**, not example-by-example end-to-end:

1. **Wave 0** — first stage (e.g. plan) for **all** selected examples with **one actor/model**.
2. If **`staged_unload_between_waves: true`** (common on a single GPU), actors are **unloaded** between waves to free memory for the next model.
3. **Wave 1** — partial traces are loaded and the second stage runs (e.g. full answer with another model — hybrid BF16 planner → 2-bit solver).

So in validated setups, models run **sequentially in time**: “plan all tasks”, unload, load another weights set for “solve all tasks”. That avoids holding two heavy checkpoints on **one** GPU.

**Keeping two different models loaded in parallel** (separate processes, two vLLM engines, different cards via per-actor `cuda_visible_devices`, possibly TP) is possible in principle but **was not tuned for reporting**: GPU placement, OOM, and vLLM behavior need care. This README documents the **wave-by-wave** path only.

---

## What is saved in `results/<run_id>/` (artifacts)

Typical set (`quantlab/artifacts/store.py`):

| File / folder | Purpose |
|---------------|---------|
| `config.json` | Frozen run config (actors, pipeline, benchmark); for replay and reproducibility |
| `traces.jsonl` | One JSON per example: `prompt`, `segments[]` with text, `token_count`, `actor_id`, `role`, **`timing`** (if backend provided stats) |
| `judgements.jsonl` | Extracted answer, `is_correct`, `parse_success`, ground truth |
| `metrics.jsonl` | One metrics row per example (accuracy, reasoning_length, per-actor tokens, timing aggregates — see metrics below) |
| `timing.jsonl` | Per-example timing summary by `actor_id` (aggregated from segments) |
| `errors.jsonl` | Traceback when an example fails |
| `summary.json` | `n_examples`, `n_judged`, mean accuracy / parse_rate |
| `trace_checkpoints/` | If staged checkpoints enabled: `wave_0.jsonl`, `wave_1.jsonl`, … — partial traces after each wave |

Select **fully successful** runs can be kept under `results/<run_id>/` for snapshots; see `results/README.md` for what is versioned.

Optional in config: `timing_replay` runs replay from the runner into a subfolder (see `ExperimentConfig`).

Standalone replay writes per-segment results to `timing_replay/<example_id>.json` (see below).

---

## Pipelines and configs

Canonical reproduction configs live under **`configs/`** (see *Running the experiments* above).

Schema: **`quantlab/config/schema.py`** (`ExperimentConfig`, `StageConfig`, `ActorDef`, `OutputConfig`).

Per stage you set: `actor_id`, `stage_prompt`, `max_new_tokens`, `stop_sequences`, `exit_conditions`, `handoff_mode`. Current MATH-500 runs use **`full_prefill`**: the next actor sees full text; KV is **not** transferred between different models.

---

## Metrics and timing

### Enabling metrics

List metrics in YAML under **`metrics`** — each entry `- name: <name>`; optional **`kwargs`** (see `MetricConfig` in `quantlab/config/schema.py`). Registry: **`quantlab/metrics/registry.py`**; all names: `MetricRegistry.available()`.

Metrics run **per example after the trace is complete**, from `Trace` and **`JudgementResult`** (`quantlab/evaluation/judge.py` — answer extraction including `\boxed{}`, comparison to reference). Values go to **`metrics.jsonl`** as `{ "example_id": "...", "<metric_name>": ... }`; run-level accuracy in **`summary.json`**.

Hybrid MATH-500 configs typically mix **accuracy**, **parse_rate**, reasoning volume/behavior, per-actor token split, and segment timing (table below).

The table lists the **full registry**; a given YAML may enable a subset. Match **`judgements.jsonl`** (judge labels and extracted text) with **`metrics.jsonl`** by **`example_id`**.

### Interpreting runs (typical hybrid)

- **Quality:** **`accuracy`** matches aggregate **`is_correct`** in **`judgements.jsonl`**. **`parse_rate`** matches **`parse_success`**: a correct score requires successful final-answer extraction; low parse rate often means truncation, messy format, or missing `\boxed{}`, not necessarily wrong math.
- **Reasoning scale:** **`reasoning_length`** and **`actor_token_split`** — total tokens and planner (BF16) vs answerer (e.g. 2-bit) share; longer is not always better.
- **Structure / loops:** **`think_closed`**, **`commit_gap`**, **`tokens_to_first_correct`**, **`finish_commit`** — how early the model “commits” and whether `\boxed{}` structure appears. **`verification_spiral`**, **`loop_detected`**, **`loop_onset_tokens`** — loop / re-check heuristics; inspect on errors (**`accuracy`** = 0).
- **Speed:** **`total_generation_ms`**, **`segment_timing_ms`**, **`tokens_per_second`** — per segment/actor; for batches see caveat below. Raw timing also in **`timing.jsonl`**.

**`exact_match`** and **`stop_token_probe`** exist in the registry but are **off by default** in hybrid configs; enable in YAML if needed.

### Metric reference

| Name | Meaning (short) |
|------|-----------------|
| **`accuracy`** | `1` if judge marked correct (`is_correct`), else `0`. |
| **`parse_rate`** | `1` if final answer was extracted reliably (`parse_success`). |
| **`exact_match`** | Exact string match of `predicted` and reference (after `strip`; stricter than default judge). |
| **`reasoning_length`** | Total **generated** tokens in the trace (`total_generated_tokens`). |
| **`loop_detected`** | Loop heuristic on **sentence units**: same sentence ≥N times in a row **or** same phrase ≥M times globally (optional “hesitation” filter — see `generation.py` and YAML kwargs). |
| **`think_closed`** | For Qwen-style sessions: paired closing think tag after open tag in prompt/generation. |
| **`commit_gap`** | Approximate **tokens** after **first** answer-candidate pattern (`\boxed`-like). `-1` if none — proxy for late commit. |
| **`tokens_to_first_correct`** | **TTFA** proxy: tokens until first occurrence of extracted **correct** answer in text (char-fraction estimate, not exact tokenizer positions). `-1` if wrong or substring missing. |
| **`finish_commit`** | Binary proxy: answer text visible **before** first `\boxed` and `\boxed{}` parse succeeds (see docstring for limits). |
| **`verification_spiral`** | Count of re-check phrases (wait / let me check / correction, etc.) in suffix after first predicted answer (or full generation fallback). |
| **`loop_onset_tokens`** | Approximate token index where a problematic loop starts (min of streak / global-repeat triggers, same pipeline as `loop_detected`). `-1` if none or empty trace. |
| **`actor_token_split`** | **`dict`** `actor_id → token count` per actor. |
| **`total_generation_ms`** | Sum of **`total_ms`** over segments with `timing` — backend timings at example level. |
| **`segment_timing_ms`** | **`dict`** `actor_id → total ms` for that actor’s segments. |
| **`tokens_per_second`** | **`dict`** `actor_id → tok/s`: segment tokens / segment time (aggregated, not mean of per-segment rates). |
| **`stop_token_probe`** | Placeholder for future EOS logit integration; currently fixed string (`StopTokenProbeMetric`). |

**`timing.jsonl`** also stores coarse per-`actor_id` timing on save (`quantlab/runner.py` `_record_example`) — similar to **`segment_timing_ms`**, flat JSONL.

### Timing from a live run

- Each trace **segment** may carry **`timing`** (prefill/decode/total_ms, tokens/s) when vLLM returns `FinishedRequestStats` and matching succeeds.
- **`timing.jsonl`** aggregates by `actor_id` per example.

**Important for `staged_batch_size ≥ 2`:** multiple prompts share one batched vLLM call. Segment `timing` either uses **per-request vLLM stats** or, in the worst case, an **equal split of batch wall time** (see `VLLMBackend.generate_batch`). For **fair latency comparisons**, use standalone replay (one request at a time), not raw batched timing.

### Replay timing on saved traces

```bash
python scripts/replay_timing.py <run_id> --results-dir results -v
# pin one model for all segments:
python scripts/replay_timing.py <run_id> --model-id Qwen/Qwen3-32B --precision bf16
```

By default the backend follows `actor_id` and actor entries in `config.json`. Output: `results/<run_id>/timing_replay/*.json`.

---

## Tests

```bash
pytest tests/ -q
```

---

## Quick reference

| Task | Where to look |
|------|----------------|
| Reproduce hybrid MATH-500 | `configs/math500/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml` + `-v` |
| Compare with full FP16 | `configs/math500/qwen32b_fp16/hybrid_fp16_fp16.yaml` |
| Throughput / KV-cache | `scripts/bench_qwen_throughput.py`, *Throughput* section above |
| Draft batched staged throughput | `staged_batch_size` in YAML or `--staged-batch-size` |
| Quality in one number | **`summary.json`** — usually **`accuracy`** and **`parse_rate`** |
| Per-example breakdown | **`judgements.jsonl`** + same **`example_id`** in **`metrics.jsonl`** |
| Full reasoning text | `traces.jsonl` → `segments` |
| Isolated segment timing | `replay_timing.py` |
| Debug failures | `errors.jsonl`, tail of `nohup` log |
