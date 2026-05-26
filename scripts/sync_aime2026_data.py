#!/usr/bin/env python3
"""Download MathArena/aime_2026 and write data/aime2026/train.json."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "aime2026" / "train.json"


def main() -> None:
    ds = load_dataset("MathArena/aime_2026", split="train")
    rows = [
        {
            "problem_idx": int(row["problem_idx"]),
            "answer": int(row["answer"]),
            "problem": row["problem"],
        }
        for row in ds
    ]
    rows.sort(key=lambda r: r["problem_idx"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
