# ARC-Easy (vendored)

Subset **ARC-Easy** from [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc).

| File | Split | Rows |
|------|-------|------|
| `train.json` | train | 2251 |
| `validation.json` | validation | 570 |
| `test.json` | test | 2376 |

Each row: `id`, `question`, `choices` (`label` + `text` lists), `answerKey` (letter A–E).

Regenerate:

```bash
python scripts/prepare_arc_easy_data.py
```

**Evaluation:** accuracy = exact match on `answerKey` (same as generative `acc` on letter labels; cf. lm-eval `arc_easy` task).
