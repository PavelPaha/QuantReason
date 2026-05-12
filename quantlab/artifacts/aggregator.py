from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from quantlab.artifacts.store import ArtifactStore


class ResultAggregator:
    """
    Reads metrics from multiple runs and produces comparison tables.

    Output can be a pandas DataFrame or written to CSV / Parquet.
    """

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def compare_runs(
        self,
        run_ids: Optional[list[str]] = None,
        scalar_metrics: Optional[list[str]] = None,
    ) -> "pd.DataFrame":
        import pandas as pd

        if run_ids is None:
            run_ids = self.store.list_runs()

        scalar_metrics = scalar_metrics or [
            "accuracy", "parse_rate", "reasoning_length",
            "loop_detected", "think_closed", "commit_gap",
            "total_generation_ms",
        ]

        rows = []
        for run_id in run_ids:
            config = self.store.load_config(run_id)
            metrics_list = self.store.load_metrics(run_id)
            if not metrics_list:
                continue

            row: dict[str, Any] = {
                "run_id": run_id,
                "experiment_name": config.get("experiment_name", ""),
            }
            for m in scalar_metrics:
                vals = [r[m] for r in metrics_list if m in r and isinstance(r[m], (int, float))]
                if vals:
                    row[f"{m}_mean"] = sum(vals) / len(vals)
                    row[f"{m}_n"] = len(vals)

            rows.append(row)

        return pd.DataFrame(rows)

    def save_comparison(
        self,
        out_path: str,
        run_ids: Optional[list[str]] = None,
        scalar_metrics: Optional[list[str]] = None,
    ) -> None:
        df = self.compare_runs(run_ids, scalar_metrics)
        p = Path(out_path)
        if p.suffix == ".parquet":
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)

    def per_example_table(self, run_id: str) -> "pd.DataFrame":
        import pandas as pd

        metrics = self.store.load_metrics(run_id)
        return pd.DataFrame(metrics)
