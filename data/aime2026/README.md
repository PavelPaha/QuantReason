# AIME 2026 (MathArena)

Vendored copy of [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) (30 problems, AIME I + II).

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `problem_idx` | int | Problem index in MathArena |
| `answer` | int | Gold answer (integer 0–999) |
| `problem` | str | Problem statement (LaTeX) |

## License

CC BY-NC-SA 4.0 — see the Hugging Face dataset card.

## Refresh from Hugging Face

```bash
python scripts/sync_aime2026_data.py
```

## Citation

```bibtex
@article{dekoninck2026matharena,
  title={Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs},
  author={Jasper Dekoninck and Nikola Jovanović and Tim Gehrunger and Kári Rögnvaldsson and Ivo Petrov and Chenhao Sun and Martin Vechev},
  year={2026},
  eprint={2605.00674},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.00674},
}
```
