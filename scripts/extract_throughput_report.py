#!/usr/bin/env python3
"""Build throughput JSON from a completed QuantLab run (batch=1, per-request timings)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _load_run_config(run_dir: Path) -> dict:
    for name in ("config.json", "config.yaml"):
        cfg_path = run_dir / name
        if cfg_path.exists():
            if name.endswith(".json"):
                return json.loads(cfg_path.read_text())
            return yaml.safe_load(cfg_path.read_text())
    return {}


def extract(run_dir: Path, out_path: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    cfg = _load_run_config(run_dir)
    actors = cfg.get("actors") or [{}]
    actor0 = actors[0] if actors else {}
    actor_id = (cfg.get("pipeline") or [{}])[0].get("actor_id") or actor0.get("actor_id", "actor")

    traces_path = run_dir / "traces.jsonl"
    if not traces_path.exists():
        raise FileNotFoundError(traces_path)

    per_example: list[dict] = []
    total_tokens = 0
    sum_ms = 0.0
    sum_prefill_ms = 0.0
    sum_decode_ms = 0.0
    sum_prompt_tokens = 0
    n_prefill = n_decode = n_prompt = 0

    for line in traces_path.read_text().splitlines():
        if not line.strip():
            continue
        tr = json.loads(line)
        eid = tr["example_id"]
        seg = tr["segments"][-1] if tr.get("segments") else {}
        timing = seg.get("timing") or {}
        tokens = int(seg.get("token_count") or timing.get("token_count") or 0)
        total_ms = float(timing.get("total_ms") or timing.get("generation_ms") or 0.0)
        prefill_ms = timing.get("prefill_ms")
        decode_ms = timing.get("decode_ms")
        tps = float(timing.get("tokens_per_second") or (tokens / max(total_ms / 1000, 1e-9)))

        prompt_tokens = None
        meta = seg.get("metadata") or {}
        if "num_prompt_tokens" in meta:
            prompt_tokens = int(meta["num_prompt_tokens"])
        elif tr.get("prompt"):
            # rough fallback: not used for weighted prefill if missing
            prompt_tokens = None

        row: dict = {
            "example_id": eid,
            "tokens": tokens,
            "total_ms": total_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "num_prompt_tokens": prompt_tokens,
            "tokens_per_second": tps,
        }
        if prefill_ms and prompt_tokens:
            row["prefill_phase_prompt_tokens_per_sec"] = prompt_tokens / max(prefill_ms / 1000, 1e-9)
        if decode_ms and tokens:
            row["decode_phase_tokens_per_sec"] = tokens / max(decode_ms / 1000, 1e-9)
        per_example.append(row)

        total_tokens += tokens
        sum_ms += total_ms
        if prefill_ms is not None:
            sum_prefill_ms += float(prefill_ms)
            n_prefill += 1
        if decode_ms is not None:
            sum_decode_ms += float(decode_ms)
            n_decode += 1
        if prompt_tokens is not None:
            sum_prompt_tokens += prompt_tokens
            n_prompt += 1

    n = len(per_example)
    report = {
        "model_id": actor0.get("model_id"),
        "benchmark": (cfg.get("benchmark") or {}).get("name"),
        "backend": actor0.get("backend"),
        "precision": actor0.get("precision"),
        "quantization": actor0.get("quantization"),
        "max_new_tokens": (cfg.get("pipeline") or [{}])[0].get("max_new_tokens"),
        "profile_chunk_tokens": None,
        "n_examples": n,
        "batch_size": cfg.get("staged_batch_size", 1),
        "load_ms": None,
        "total_new_tokens": total_tokens,
        "sum_generation_ms": sum_ms,
        "throughput_tokens_per_sec_end_to_end": total_tokens / max(sum_ms / 1000, 1e-9),
        "mean_per_request_tokens_per_sec": sum(r["tokens_per_second"] for r in per_example) / max(n, 1),
        "weighted_prefill_phase_prompt_tokens_per_sec": (
            sum_prompt_tokens / max(sum_prefill_ms / 1000, 1e-9) if n_prefill and n_prompt else None
        ),
        "weighted_decode_phase_tokens_per_sec": (
            total_tokens / max(sum_decode_ms / 1000, 1e-9) if n_decode else None
        ),
        "mean_prefill_ms": (sum_prefill_ms / n_prefill) if n_prefill else None,
        "mean_decode_ms": (sum_decode_ms / n_decode) if n_decode else None,
        "mean_prompt_tokens": (sum_prompt_tokens / n_prompt) if n_prompt else None,
        "n_rows_with_prefill_ms": n_prefill,
        "n_rows_with_decode_ms": n_decode,
        "n_rows_with_prompt_tokens": n_prompt,
        "per_example": per_example,
        "warmup": 0,
        "cuda_visible_devices": (actor0.get("backend_kwargs") or {}).get("cuda_visible_devices"),
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run_dir),
        "actor_id": actor_id,
    }

    if out_path is None:
        slug = (report["model_id"] or "model").replace("/", "-")
        out_path = run_dir.parent.parent / f"_throughput_{slug}_gpu1_bs1_n5.json"
    out_path = Path(out_path)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "per_example"}, indent=2))
    print(f"Wrote {out_path}")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()
    extract(args.run_dir, args.output)


if __name__ == "__main__":
    main()
