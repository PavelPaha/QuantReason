#!/usr/bin/env python3
"""Download allenai/winogrande (winogrande_xl) into data/winogrande/."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1] / "data" / "winogrande" / "winogrande_xl"
SUBSET = "winogrande_xl"
HF_ID = "allenai/winogrande"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test", "train"):
        ds = load_dataset(HF_ID, SUBSET, split=split)
        out = ROOT / f"{split}.parquet"
        ds.to_parquet(out)
        print(f"wrote {out} ({len(ds)} rows)")


if __name__ == "__main__":
    main()
