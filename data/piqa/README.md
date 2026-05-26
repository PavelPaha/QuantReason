# PIQA (vendored)

Локальная копия [baber/piqa](https://huggingface.co/datasets/baber/piqa) — тот же датасет, что в [lm-evaluation-harness `piqa.yaml`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/piqa.yaml).

| Split | Rows | File |
|-------|------|------|
| `validation` | 1838 | `validation.parquet` |
| `train` | 16113 | `train.parquet` |

Стандартный eval: **validation**, метрика **accuracy** (label `0`/`1` → solution 1 / solution 2; в generative-прогонах ответ `1` / `2`).

Подготовка:

```bash
python scripts/prepare_piqa_data.py
```
