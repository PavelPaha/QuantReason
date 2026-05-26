# Final experiment configs

Канонические конфиги для сравнения **Qwen3-32B FP16** vs **GPTQ 2-bit** на семи бенчмарках.

## Матрица (28 конфигов)

| Dataset | `single_fp16` | `single_gptq2bit` | `hybrid_fp16_fp16` | `hybrid_fp16_gptq2bit` |
|---------|---------------|---------------------|--------------------|-------------------------|
| `gsm8k/` | single answer | single answer | plan FP16 → reason FP16 | plan FP16 → reason GPTQ |
| `math500/` | single answer | single answer | plan FP16 → reason FP16 | plan FP16 → reason GPTQ |
| `aime2026/` | single answer (integer 0–999) | single answer | plan FP16 → reason FP16 | plan FP16 → reason GPTQ |
| `arc_easy/` | MCQ | MCQ | plan FP16 → reason FP16 | plan FP16 → reason GPTQ |
| `gpqa_diamond/` | MCQ | MCQ | plan FP16 → reason FP16 | plan FP16 → reason GPTQ |
| `winogrande/` | MCQ (1/2) | MCQ | plan FP16 → reason FP16 | plan FP16 → reason GPTQ |
| `piqa/` | MCQ (1/2) | MCQ | plan FP16 → reason FP16 | plan FP16 → reason GPTQ |

- **single** — одна стадия `answer`.
- **hybrid** — plan (FP16, до 1024 tok, `plan_scaffold`) → reasoning; план в **user** (`prompt_plan_in_user`).

Перегенерация всех yaml из шаблона (GPU по умолчанию `"0"`):

```bash
python scripts/generate_final_configs.py
# затем вручную подставьте cuda_visible_devices в нужные yaml
```

## Подготовка данных (один раз)

```bash
# AIME-2026 (30 задач в train.json)
python scripts/sync_aime2026_data.py

# ARC-Easy, WinoGrande, PIQA
python scripts/prepare_arc_easy_data.py
python scripts/prepare_winogrande_data.py
python scripts/prepare_piqa_data.py
```

GSM8K и MATH-500 подгружаются через HuggingFace `datasets` при запуске.

## Перед запуском

1. Укажите GPU в yaml: `actors[].backend_kwargs.cuda_visible_devices` (строка `"5"` или `"4,5"` для TP=2).
2. Для **GPQA FP16 single** на 32k при OOM увеличьте `tensor_parallel_size: 2` и две карты (см. комментарий в `gpqa_diamond/single_fp16.yaml`).
3. Рекомендуется `export VLLM_USE_V1=0` (уже выставляется в `scripts/run_experiment.py`).

## Запуск одного конфига

Из корня репозитория:

```bash
python scripts/run_experiment.py configs/final/<dataset>/<variant>.yaml -v
```

Опции:

| Флаг | Назначение |
|------|------------|
| `-v` | Подробный лог в `results/.../run.log` |
| `--max-examples N` | Переопределить число примеров (smoke) |
| `--output-dir PATH` | Другая база для артефактов |
| `--staged-batch-size N` | Micro-batch в staged-режиме |

Артефакты: `results/<category>/<run_id>/` — `config.json`, `traces.jsonl` (промпт + `llm_prompt_full` + ответы), `judgements.jsonl`, `summary.json`.

---

## Все 28 конфигов — команды запуска

Подставьте свои GPU в yaml перед прогоном. Ниже пути относительно корня репозитория.

### GSM8K (500 примеров)

```bash
python scripts/run_experiment.py configs/final/gsm8k/single_fp16.yaml -v
python scripts/run_experiment.py configs/final/gsm8k/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/final/gsm8k/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final/gsm8k/hybrid_fp16_gptq2bit.yaml -v
```

### MATH-500 (500)

```bash
python scripts/run_experiment.py configs/final/math500/single_fp16.yaml -v
python scripts/run_experiment.py configs/final/math500/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/final/math500/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final/math500/hybrid_fp16_gptq2bit.yaml -v
```

### AIME-2026 (30)

```bash
python scripts/run_experiment.py configs/final/aime2026/single_fp16.yaml -v
python scripts/run_experiment.py configs/final/aime2026/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/final/aime2026/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final/aime2026/hybrid_fp16_gptq2bit.yaml -v
```

### ARC-Easy (test 2376; в конфиге `max_examples: null` = все)

```bash
python scripts/run_experiment.py configs/final/arc_easy/single_fp16.yaml -v
python scripts/run_experiment.py configs/final/arc_easy/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/final/arc_easy/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final/arc_easy/hybrid_fp16_gptq2bit.yaml -v
```

### GPQA Diamond (~198)

```bash
python scripts/run_experiment.py configs/final/gpqa_diamond/single_fp16.yaml -v
python scripts/run_experiment.py configs/final/gpqa_diamond/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/final/gpqa_diamond/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final/gpqa_diamond/hybrid_fp16_gptq2bit.yaml -v
```

### WinoGrande (validation, winogrande_xl)

```bash
python scripts/run_experiment.py configs/final/winogrande/single_fp16.yaml -v
python scripts/run_experiment.py configs/final/winogrande/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/final/winogrande/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final/winogrande/hybrid_fp16_gptq2bit.yaml -v
```

### PIQA (validation)

```bash
python scripts/run_experiment.py configs/final/piqa/single_fp16.yaml -v
python scripts/run_experiment.py configs/final/piqa/single_gptq2bit.yaml -v
python scripts/run_experiment.py configs/final/piqa/hybrid_fp16_fp16.yaml -v
python scripts/run_experiment.py configs/final/piqa/hybrid_fp16_gptq2bit.yaml -v
```

---

## Smoke на 10 примерах, контекст 2000 tok

Готовые патчи и батч на двух GPU (очередь **per-GPU**, без коллизий VRAM):

```bash
# Сгенерировать configs/final_validate_ctx2000/*.yaml
python scripts/run_final_validate_ctx2000.py --dry-run

# Прогон всех 28 (пропуск уже успешных)
python scripts/run_final_validate_ctx2000.py --resume --gpus 5,7

# Лог: results/final_validate_ctx2000_n10/batch_run_resume.log
# Сводка: results/final_validate_ctx2000_n10/validation_summary.json
```

Параметры validate: `max_examples=10`, `max_total_tokens=2000`, `max_model_len=8192`.

---

## Промпты и эвал

- **GSM8K / MATH-500 / AIME single**: задача + `\boxed{}` в user; assistant — открытый ``.
- **AIME**: ответ integer 0–999.
- **ARC / GPQA**: MCQ, буква `answerKey`.
- **WinoGrande / PIQA**: ответ `1` / `2`.
- **Hybrid plan**: `plan_scaffold`, закрытый пустой ``.
- **Hybrid reason**: план в user + открытый ``.

## Контекст (полные прогоны)

- `max_model_len: 32768`, `max_total_tokens: 32768`, генерация до **30720** tok (plan — 1024).
- `staged_batch_size`: 500 (gsm8k, math500, gpqa, winogrande, piqa), **30** (aime2026).

## Связь с `final_results/`

Имена run-папок в `final_results/<dataset>/qwen3-32b/`:

- `single_fp16`, `single_gptq2bit`
- `plan_fp16__reason_fp16`, `plan_fp16__reason_gptq2bit`
