#!/usr/bin/env python3
"""Create results/<dest_run_id>/ with config.json + copied trace checkpoint for staged resume."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))

    p = argparse.ArgumentParser(
        description=(
            "Create a new run directory under results/ with experiment config.json "
            "and trace_checkpoints/wave_<N>.jsonl copied from another run "
            "(for --resume-run-id ... --resume-after-wave N)."
        )
    )
    p.add_argument(
        "--config-yaml",
        required=True,
        type=Path,
        help="Experiment YAML; used to write results/<dest>/config.json",
    )
    p.add_argument(
        "--source-run-id",
        required=True,
        help="Existing folder under results/ that has trace_checkpoints/wave_<wave>.jsonl",
    )
    p.add_argument(
        "--dest-run-id",
        required=True,
        help="New folder name under results/ (must not exist unless --overwrite)",
    )
    p.add_argument(
        "--wave",
        type=int,
        default=0,
        help="Checkpoint index N to copy from source (default: 0)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If dest exists, replace only checkpoint + config.json",
    )

    args = p.parse_args()
    yaml_path = args.config_yaml if args.config_yaml.is_absolute() else repo / args.config_yaml

    import yaml as pyyaml
    from quantlab.config.schema import ExperimentConfig

    raw = pyyaml.safe_load(yaml_path.read_text())
    config = ExperimentConfig.model_validate(raw)

    src_dir = repo / "results" / args.source_run_id
    ck = src_dir / "trace_checkpoints" / f"wave_{args.wave}.jsonl"
    if not ck.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ck}")

    dst_dir = repo / "results" / args.dest_run_id
    chk_dst_dir = dst_dir / "trace_checkpoints"
    if dst_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination exists: {dst_dir}. Remove it or pass --overwrite."
            )
    else:
        dst_dir.mkdir(parents=True)

    chk_dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ck, chk_dst_dir / f"wave_{args.wave}.jsonl")

    out_cfg = dst_dir / "config.json"
    out_cfg.write_text(
        json.dumps(config.model_dump(mode="python"), indent=2, ensure_ascii=False) + "\n"
    )

    print(f"Wrote {out_cfg}")
    print(f"Copied {ck} -> {chk_dst_dir / f'wave_{args.wave}.jsonl'}")
    print("")
    print("Then run:")
    print(
        f"  python scripts/run_experiment.py {yaml_path.relative_to(repo)} -v \\\n"
        f"    --staged --resume-run-id {args.dest_run_id} --resume-after-wave {args.wave}"
    )


if __name__ == "__main__":
    main()
