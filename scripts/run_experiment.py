#!/usr/bin/env python3
"""Run a QuantLab experiment from a YAML config file."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import click
import yaml


def _stdout_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    click.echo(f"[{timestamp}] {message}")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option("--max-examples", type=int, default=None, help="Override max_examples")
@click.option("--output-dir", type=str, default=None, help="Override output base_dir")
@click.option(
    "--staged/--no-staged",
    default=None,
    help="Run pipeline in waves (all examples stage 0, then stage 1, …). "
    "Overrides YAML staged_execution when set.",
)
@click.option("--resume-run-id", type=str, default=None)
@click.option(
    "--resume-after-wave",
    type=int,
    default=None,
    help="With --staged and --resume-run-id: last completed staged wave saved under "
    "trace_checkpoints/wave_<N>.jsonl; continue from wave N+1. "
    "Omit for non-staged resume (uses traces.jsonl / judgements.jsonl).",
)
@click.option(
    "--staged-batch-size",
    type=int,
    default=None,
    help="Override staged_batch_size (--staged / YAML staged_execution): vLLM micro-batch per wave "
    "(>=2 bundles prompts in one LLM.generate; traces unchanged, batch-degraded timings).",
)
def main(
    config_path: str,
    verbose: bool,
    max_examples: int | None,
    output_dir: str | None,
    staged: bool | None,
    resume_run_id: str | None,
    resume_after_wave: int | None,
    staged_batch_size: int | None,
) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from quantlab.config.schema import ExperimentConfig
    from quantlab.runner import run_experiment

    raw = yaml.safe_load(Path(config_path).read_text())

    if max_examples is not None:
        raw.setdefault("benchmark", {})["max_examples"] = max_examples
    if output_dir is not None:
        raw.setdefault("output", {})["base_dir"] = output_dir
    if staged_batch_size is not None:
        raw["staged_batch_size"] = staged_batch_size

    config = ExperimentConfig.model_validate(raw)
    if resume_after_wave is not None and resume_run_id is None:
        raise click.UsageError("--resume-after-wave requires --resume-run-id")
    use_staged = config.staged_execution if staged is None else staged
    if resume_run_id is not None and use_staged and resume_after_wave is None:
        raise click.UsageError(
            "staged resume requires --resume-after-wave (last completed wave index)"
        )
    run_id = run_experiment(
        config,
        verbose=verbose,
        staged_execution=staged,
        resume_run_id=resume_run_id,
        resume_after_wave=resume_after_wave,
    )
    _stdout_log(f"Run complete: {run_id}")


if __name__ == "__main__":
    main()
