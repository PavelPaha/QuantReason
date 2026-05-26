# Experiment configs

YAMLs: `configs/<dataset>/<model_family>/<variant>.yaml`

Datasets: `aime2026`, `arc_challenge`, `arc_easy`, `gpqa_diamond`, `gsm8k`, `math500`, `piqa`, `strategyqa`, `winogrande`.

```
configs/math500/
├── qwen32b_fp16/
│   ├── single_fp16.yaml
│   └── hybrid_fp16_fp16.yaml
├── qwen32b_gptq2bit/
│   ├── single_gptq2bit.yaml
│   └── hybrid_fp16_gptq2bit.yaml
├── qwen32b_nvfp4/
│   ├── single_nvfp4.yaml
│   └── hybrid_fp16_nvfp4.yaml
├── qwen32b_nvfp4_kv4/
│   ├── single_nvfp4_kv4.yaml
│   └── hybrid_fp16_nvfp4_kv4.yaml
├── qwen8b_fp16/
│   ├── single_fp16.yaml
│   └── hybrid_fp16_fp16.yaml
├── qwen8b_nvfp4/
│   ├── single_8b_nvfp4.yaml
│   ├── hybrid_fp16_nvfp4.yaml
│   └── hybrid_fp16_8b_nvfp4.yaml
└── qwen8b_moe35b_nvfp4/
    ├── single_moe35b_nvfp4.yaml
    └── hybrid_fp16_moe35b_nvfp4.yaml
```

**Folders**

| Folder | Model stack |
|--------|-------------|
| `qwen32b_fp16/` | Qwen3-32B FP16 |
| `qwen32b_gptq2bit/` | Qwen3-32B GPTQ 2-bit |
| `qwen32b_nvfp4/` | Qwen3-32B NVFP4, default KV |
| `qwen32b_nvfp4_kv4/` | Qwen3-32B NVFP4 + `kv_cache_dtype: nvfp4` |
| `qwen8b_fp16/` | Qwen3-8B FP16 baseline |
| `qwen8b_nvfp4/` | Qwen3-8B / 32B NVFP4 (see variants below) |
| `qwen8b_moe35b_nvfp4/` | Qwen3.6-35B MoE NVFP4 |

**Variants**

| File | Mode |
|------|------|
| `single_*` | One model, one stage |
| `hybrid_fp16_fp16` | FP16 plan → FP16 reasoning |
| `hybrid_fp16_gptq2bit` | FP16 plan → GPTQ 2-bit reasoning |
| `qwen32b_nvfp4/hybrid_fp16_nvfp4` | 32B FP16 plan → 32B NVFP4 |
| `qwen32b_nvfp4_kv4/hybrid_fp16_nvfp4_kv4` | 32B FP16 plan → 32B NVFP4 + NVFP4 KV |
| `qwen8b_nvfp4/hybrid_fp16_nvfp4` | 8B FP16 plan → 32B NVFP4 |
| `hybrid_fp16_8b_nvfp4` | 8B FP16 plan → 8B NVFP4 |
| `hybrid_fp16_moe35b_nvfp4` | 8B FP16 plan → MoE 35B NVFP4 |

Set `actors[].backend_kwargs.cuda_visible_devices` in the YAML before running.

```bash
python scripts/run_experiment.py configs/<dataset>/qwen32b_fp16/single_fp16.yaml -v
python scripts/run_experiment.py configs/<dataset>/qwen32b_fp16/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/<dataset>/qwen32b_gptq2bit/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/<dataset>/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml -v
```

NVFP4 / MoE (same `<dataset>/` folders):

```bash
python scripts/run_experiment.py configs/math500/qwen32b_nvfp4/single_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_nvfp4/hybrid_fp16_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_nvfp4_kv4/single_nvfp4_kv4.yaml -v
python scripts/run_experiment.py configs/math500/qwen32b_nvfp4_kv4/hybrid_fp16_nvfp4_kv4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_nvfp4/single_8b_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_nvfp4/hybrid_fp16_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_nvfp4/hybrid_fp16_8b_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_moe35b_nvfp4/single_moe35b_nvfp4.yaml -v
python scripts/run_experiment.py configs/math500/qwen8b_moe35b_nvfp4/hybrid_fp16_moe35b_nvfp4.yaml -v
```

Regenerate 32B FP16 / GPTQ baseline YAMLs: `python scripts/generate_final_configs.py`

More detail (KV cache, flags, env): [root README](../README.md).
