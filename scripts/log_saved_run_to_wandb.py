#!/usr/bin/env python3
"""Log a completed local run to Weights & Biases from saved artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml


def _load_optional_wandb_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise click.ClickException("W&B config file must contain a mapping at the top level.")

    if "wandb" in raw:
        wandb_block = raw["wandb"] or {}
        if not isinstance(wandb_block, dict):
            raise click.ClickException("'wandb' section must be a mapping.")
        return dict(wandb_block)

    return dict(raw)


@click.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--wandb-config",
    "wandb_config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML with a top-level 'wandb:' block or plain WandbConfig fields.",
)
@click.option("--project", default=None, help="Override W&B project.")
@click.option("--entity", default=None, help="Override W&B entity.")
@click.option("--group", default=None, help="Override W&B group.")
@click.option("--name", default=None, help="Override W&B run name.")
@click.option("--job-type", default=None, help="Override W&B job_type.")
@click.option("--mode", default=None, help="Override W&B mode, e.g. online/offline.")
@click.option("--notes", default=None, help="Override W&B notes.")
@click.option("--tag", "tags", multiple=True, help="Add W&B tag (repeatable).")
@click.option(
    "--upload-run-artifact/--no-upload-run-artifact",
    default=None,
    help="Override whether the whole run directory is uploaded as an artifact.",
)
@click.option(
    "--log-per-example-table/--no-log-per-example-table",
    default=None,
    help="Override whether the reconstructed per-example table is uploaded.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    run_dir: Path,
    wandb_config_path: Path | None,
    project: str | None,
    entity: str | None,
    group: str | None,
    name: str | None,
    job_type: str | None,
    mode: str | None,
    notes: str | None,
    tags: tuple[str, ...],
    upload_run_artifact: bool | None,
    log_per_example_table: bool | None,
    verbose: bool,
) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from quantlab.config.schema import ExperimentConfig, WandbConfig
    from quantlab.wandb_logger import WandbRunLogger

    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    if not config_path.exists():
        raise click.ClickException(f"Missing config.json under {run_dir}")
    if not summary_path.exists():
        raise click.ClickException(f"Missing summary.json under {run_dir}")

    experiment_config = ExperimentConfig.model_validate(json.loads(config_path.read_text()))
    summary = json.loads(summary_path.read_text())
    run_id = str(summary.get("run_id") or run_dir.name)

    wandb_raw: dict[str, Any] = {}
    if experiment_config.wandb is not None:
        wandb_raw.update(experiment_config.wandb.model_dump(mode="python"))
    wandb_raw.update(_load_optional_wandb_config(wandb_config_path))

    overrides = {
        "project": project,
        "entity": entity,
        "group": group,
        "name": name,
        "job_type": job_type,
        "mode": mode,
        "notes": notes,
        "upload_run_artifact": upload_run_artifact,
        "log_per_example_table": log_per_example_table,
    }
    for key, value in overrides.items():
        if value is not None:
            wandb_raw[key] = value
    if tags:
        wandb_raw["tags"] = list(tags)

    wandb_raw["enabled"] = True
    wandb_config = WandbConfig.model_validate(wandb_raw)

    logger = WandbRunLogger(
        config=wandb_config,
        experiment_config=experiment_config,
        run_id=run_id,
        verbose=verbose,
    )
    if not logger.enabled:
        raise click.ClickException(
            "wandb package is not installed. Install it with `pip install -e \".[wandb]\"`."
        )

    logger.load_saved_run(run_dir=run_dir)
    logger.finish(summary=summary, run_dir=run_dir)
    click.echo(f"W&B logging complete: {run_id}")


if __name__ == "__main__":
    main()
