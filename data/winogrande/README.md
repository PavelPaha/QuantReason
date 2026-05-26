# WinoGrande (vendored)

Локальная копия [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande), конфиг **winogrande_xl**.

| Split | Rows | File |
|-------|------|------|
| `validation` | 1267 | `winogrande_xl/validation.parquet` |
| `test` | 1767 | `winogrande_xl/test.parquet` |
| `train` | 40398 | `winogrande_xl/train.parquet` |

Стандартный eval (как в [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/winogrande)): **validation**, subset **winogrande_xl**, метрика **accuracy** (exact match на ответ `1` / `2`).

Пересобрать из Hugging Face:

```bash
python scripts/prepare_winogrande_data.py
```
