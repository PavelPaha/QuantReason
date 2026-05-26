#!/usr/bin/env python3
"""Vendor ARC-Easy (allenai/ai2_arc) into data/arc_easy/*.json."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parents[1] / "data" / "arc_easy"


def export_split(split: str) -> int:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    rows = []
    for row in ds:
        rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "choices": {
                    "text": list(row["choices"]["text"]),
                    "label": list(row["choices"]["label"]),
                },
                "answerKey": row["answerKey"],
            }
        )
    path = OUT / f"{split}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        n = export_split(split)
        print(f"wrote {OUT / f'{split}.json'} ({n} rows)")


if __name__ == "__main__":
    main()
