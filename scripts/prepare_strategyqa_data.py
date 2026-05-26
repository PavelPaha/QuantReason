#!/usr/bin/env python3
"""Download ChilleD/StrategyQA into data/strategyqa/."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1] / "data" / "strategyqa"
HF_ID = "ChilleD/StrategyQA"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        ds = load_dataset(HF_ID, split=split)
        out = ROOT / f"{split}.parquet"
        ds.to_parquet(out)
        print(f"wrote {out} ({len(ds)} rows)")


if __name__ == "__main__":
    main()
