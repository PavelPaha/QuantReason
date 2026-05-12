#!/usr/bin/env python3
"""Aggregate and compare results across multiple runs."""
from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command()
@click.option("--results-dir", default="results", show_default=True)
@click.option("--run-ids", default=None, help="Comma-separated run IDs (defaults to all)")
@click.option("--out", default="results/comparison.csv", show_default=True)
@click.option("--format", "fmt", default="csv", type=click.Choice(["csv", "parquet"]), show_default=True)
@click.option("--metrics", default=None, help="Comma-separated metric names to include")
def main(
    results_dir: str,
    run_ids: str | None,
    out: str,
    fmt: str,
    metrics: str | None,
) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from quantlab.artifacts.aggregator import ResultAggregator
    from quantlab.artifacts.store import ArtifactStore

    store = ArtifactStore(base_dir=results_dir)
    aggregator = ResultAggregator(store)

    ids = [r.strip() for r in run_ids.split(",")] if run_ids else None
    metric_names = [m.strip() for m in metrics.split(",")] if metrics else None

    out_path = out if out.endswith(f".{fmt}") else f"{out}.{fmt}"
    aggregator.save_comparison(out_path, run_ids=ids, scalar_metrics=metric_names)

    click.echo(f"Comparison saved to {out_path}")

    # Also print a quick summary table
    df = aggregator.compare_runs(ids, metric_names)
    if not df.empty:
        click.echo("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
