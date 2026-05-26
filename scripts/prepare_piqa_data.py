#!/usr/bin/env python3
"""Download baber/piqa (lm-evaluation-harness) into data/piqa/."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1] / "data" / "piqa"
HF_ID = "baber/piqa"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "train"):
        ds = load_dataset(HF_ID, split=split)
        out = ROOT / f"{split}.parquet"
        ds.to_parquet(out)
        print(f"wrote {out} ({len(ds)} rows)")


if __name__ == "__main__":
    main()
