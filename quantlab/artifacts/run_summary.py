"""Build ``summary.json`` payloads from persisted judgements and metrics."""

from __future__ import annotations

from typing import Any, Optional


def coerce_numeric_for_agg(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def aggregate_metrics_mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Mean for each metric column and for each numeric sub-key in dict-valued metrics,
    excluding ``example_id``.
    """
    if not rows:
        return {}
    names = set()
    for r in rows:
        names.update(k for k in r if k != "example_id")
    out: dict[str, Any] = {}
    for name in sorted(names):
        col = [r.get(name) for r in rows]
        ref = next((v for v in col if v is not None), None)
        if ref is None:
            continue
        if isinstance(ref, dict):
            nested: dict[str, float] = {}
            subkeys = set()
            for v in col:
                if isinstance(v, dict):
                    subkeys.update(v.keys())
            for sk in sorted(subkeys):
                nums = [
                    n
                    for v in col
                    if isinstance(v, dict)
                    for n in [coerce_numeric_for_agg(v.get(sk))]
                    if n is not None
                ]
                if nums:
                    nested[sk] = sum(nums) / len(nums)
            if nested:
                out[name] = nested
        else:
            nums = [n for v in col for n in [coerce_numeric_for_agg(v)] if n is not None]
            if nums:
                out[name] = sum(nums) / len(nums)
    return out


def build_summary(
    *,
    run_id: str,
    experiment_name: str,
    n_examples: int,
    judgements: list[dict],
    metrics_rows: list[dict],
) -> dict[str, Any]:
    """Aggregate judgements + metrics into the runner's ``summary.json`` shape."""
    nj = len(judgements)
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "n_examples": n_examples,
        "n_judged": nj,
        "accuracy": (sum(1 for j in judgements if j.get("is_correct")) / nj) if nj else 0.0,
        "parse_rate": (sum(1 for j in judgements if j.get("parse_success")) / nj)
        if nj
        else 0.0,
        "n_metrics_rows": len(metrics_rows),
        "aggregated_metrics": aggregate_metrics_mean(metrics_rows),
    }


def n_examples_from_config_or_rows(
    config: dict[str, Any],
    *,
    n_judged: int,
    n_metrics_rows: int,
) -> int:
    """
    Planned example count when refreshing old runs: YAML ``benchmark.max_examples`` if set,
    otherwise max(judgements, metrics) row counts.
    """
    bm = config.get("benchmark") or {}
    max_ex = bm.get("max_examples")
    if isinstance(max_ex, int) and max_ex >= 0:
        return max_ex
    return max(n_judged, n_metrics_rows)
