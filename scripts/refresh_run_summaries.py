#!/usr/bin/env python3
"""Rewrite results/<run_id>/summary.json from judgements.jsonl + metrics.jsonl + config.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command()
@click.option("--results-dir", default="results", show_default=True)
@click.option(
    "--run-id",
    multiple=True,
    help="Only these run folder names (basename). Repeatable. Default: all runnable dirs.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be written, do not save summary.json.",
)
def main(results_dir: str, run_id: tuple[str, ...], dry_run: bool) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from quantlab.artifacts.run_summary import build_summary, n_examples_from_config_or_rows
    from quantlab.artifacts.store import ArtifactStore

    store = ArtifactStore(base_dir=results_dir)
    target_ids = list(run_id) if run_id else store.list_runs()

    for rid in target_ids:
        rd = store.run_dir(rid)
        cfg_path = rd / "config.json"
        if not cfg_path.is_file():
            click.echo(f"skip (no config.json): {rid}", err=True)
            continue
        cfg = json.loads(cfg_path.read_text())
        judgements = store.load_judgements(rid)
        metrics_rows = store.load_metrics(rid)
        if not judgements and not metrics_rows:
            click.echo(f"skip (no judgements/metrics): {rid}", err=True)
            continue

        n_ex = n_examples_from_config_or_rows(
            cfg,
            n_judged=len(judgements),
            n_metrics_rows=len(metrics_rows),
        )
        summary = build_summary(
            run_id=rid,
            experiment_name=cfg.get("experiment_name", ""),
            n_examples=n_ex,
            judgements=judgements,
            metrics_rows=metrics_rows,
        )
        out_path = rd / "summary.json"
        if dry_run:
            click.echo(f"[dry-run] {out_path} ← n_judged={summary['n_judged']} agg_keys={len(summary['aggregated_metrics'])}")
            continue
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        click.echo(f"updated {out_path}")


if __name__ == "__main__":
    main()
