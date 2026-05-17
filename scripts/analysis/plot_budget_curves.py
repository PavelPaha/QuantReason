#!/usr/bin/env python3
"""
Budget-curve analysis: accuracy and/or quantlab metrics vs truncated token budget.

Replaces plot_accuracy_vs_budget*.py, plot_loop_metrics_vs_budget.py,
plot_structure_metrics_vs_budget.py, and replot_budget_curves.py.

Examples
--------
# Accuracy for two sweep folders (mean ± std over seed runs):
python scripts/analysis/plot_budget_curves.py \\
  --run "BF16 32k:/path/to/bf16-32k" \\
  --run "GPTQ-4 32k:/path/to/gptq4-32k" \\
  --metrics accuracy --out-dir /tmp/budget_plots

# Four experiments + loop metrics (plan runs: --single-run):
python scripts/analysis/plot_budget_curves.py \\
  --run "BF16 32k:.../bf16-32k" --run "GPTQ-4 32k:.../gptq4-32k" \\
  --run "BF16+GPTQ plan 32k:.../bf16-gptq4-plan-32k" \\
  --run "GPTQ+GPTQ plan 32k:.../gptq4-gptq4-plan-32k" \\
  --single-run "BF16+GPTQ plan 32k,GPTQ+GPTQ plan 32k" \\
  --metrics loop_detected,loop_onset_tokens,verification_spiral \\
  --budgets-csv /path/to/accuracy_vs_budget_32k.csv

# Merge precomputed CSV with extra runs:
python scripts/analysis/plot_budget_curves.py \\
  --base-csv accuracy_vs_budget_32k.csv \\
  --run "BF16+GPTQ plan 32k:.../bf16-gptq4-plan-32k" \\
  --metrics accuracy

# Replot existing CSVs (no trace recompute):
python scripts/analysis/plot_budget_curves.py --replot-only --input-dir /path/to/csvs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quantlab.benchmarks.registry import BenchmarkRegistry
from quantlab.core.trace import Trace, TraceSegment
from quantlab.evaluation.judge import judge
from quantlab.metrics.registry import MetricRegistry

DEFAULT_MIN_BUDGET = 500
DEFAULT_MAX_BUDGET = 32768
DEFAULT_STEP = 500
DEFAULT_OUTPUT_TAG = "32k"

NANMEAN_METRICS = frozenset(
    {"loop_onset_tokens", "commit_gap", "tokens_to_first_correct"}
)

COLORS = {
    "BF16 32k": "#2563eb",
    "GPTQ-4 32k": "#dc2626",
    "BF16+GPTQ plan 32k": "#16a34a",
    "GPTQ+GPTQ plan 32k": "#9333ea",
}

LABEL_ORDER = (
    "BF16 32k",
    "GPTQ-4 32k",
    "BF16+GPTQ plan 32k",
    "GPTQ+GPTQ plan 32k",
)

METRIC_META: dict[str, dict[str, str | bool]] = {
    "accuracy": {
        "title": "Accuracy vs token budget",
        "ylabel": "Accuracy (MATH-500 judge)",
        "ymin_at_zero": True,
        "ylim_0_1": True,
    },
    "loop_detected": {
        "title": "Loop detected vs token budget",
        "ylabel": "Share of examples with loop detected",
        "ymin_at_zero": True,
        "ylim_0_1": True,
    },
    "loop_onset_tokens": {
        "title": "Loop onset vs token budget",
        "ylabel": "Mean loop onset (tokens)",
        "ymin_at_zero": True,
    },
    "verification_spiral": {
        "title": "Verification spiral vs token budget",
        "ylabel": "Mean verification-spiral count",
        "ymin_at_zero": True,
    },
    "commit_gap": {
        "title": "Commit gap vs token budget",
        "ylabel": "Mean commit gap (tokens after first \\boxed{})",
        "ymin_at_zero": True,
    },
    "finish_commit": {
        "title": "Finish commit vs token budget",
        "ylabel": "Share with finish-commit pattern",
        "ymin_at_zero": True,
        "ylim_0_1": True,
    },
    "tokens_to_first_correct": {
        "title": "Tokens to first correct vs token budget",
        "ylabel": "Mean tokens to first correct answer",
        "ymin_at_zero": True,
    },
    "think_closed": {
        "title": "Think block closed vs token budget",
        "ylabel": "Share with closed think block",
        "ymin_at_zero": True,
        "ylim_0_1": True,
    },
}

METRIC_PRESETS: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy",),
    "loop": ("loop_detected", "loop_onset_tokens", "verification_spiral"),
    "structure": (
        "commit_gap",
        "finish_commit",
        "tokens_to_first_correct",
        "think_closed",
    ),
}


@dataclass
class CurveSeries:
    label: str
    mean: np.ndarray
    std: np.ndarray | None


@dataclass
class ExperimentSpec:
    label: str
    root: Path
    force_single: bool = False


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def parse_run_arg(raw: str) -> ExperimentSpec:
    for sep in (":", "="):
        if sep in raw:
            label, path = raw.split(sep, 1)
            return ExperimentSpec(label.strip(), Path(path.strip()))
    raise argparse.ArgumentTypeError(
        f"expected LABEL:PATH or LABEL=PATH, got {raw!r}"
    )


def parse_budgets(args: argparse.Namespace) -> list[int]:
    if args.budgets is not None:
        raw = [p.strip() for p in args.budgets.replace(" ", ",").split(",") if p.strip()]
        budgets = sorted({int(p) for p in raw})
    elif args.budgets_file is not None:
        text = args.budgets_file.read_text()
        raw = [
            p.strip()
            for line in text.splitlines()
            for p in line.replace(",", " ").split()
            if p.strip()
        ]
        budgets = sorted({int(p) for p in raw})
    elif args.budgets_csv is not None:
        budgets = load_budgets_from_csv(args.budgets_csv)
    else:
        if args.step <= 0:
            raise ValueError("--step must be positive")
        if args.min_budget >= args.max_budget:
            raise ValueError("--min-budget must be less than --max-budget")
        budgets = list(range(args.min_budget, args.max_budget, args.step))
        if not budgets or budgets[-1] != args.max_budget:
            budgets.append(args.max_budget)
    if not budgets:
        raise ValueError("no budgets specified")
    if any(b <= 0 for b in budgets):
        raise ValueError("all budgets must be positive integers")
    return budgets


def load_budgets_from_csv(path: Path) -> list[int]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "budget" not in reader.fieldnames:
            raise ValueError(f"{path}: expected 'budget' column")
        return [int(row["budget"]) for row in reader]


def load_base_csv(path: Path) -> tuple[np.ndarray, dict[str, CurveSeries]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "budget" not in reader.fieldnames:
            raise ValueError(f"{path}: expected 'budget' column")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    labels: list[str] = []
    for col in fieldnames:
        if col.endswith("_mean"):
            labels.append(col[: -len("_mean")])

    budgets = np.array([int(r["budget"]) for r in rows], dtype=float)
    series: dict[str, CurveSeries] = {}
    for label in labels:
        mean = np.array([float(r[f"{label}_mean"]) for r in rows])
        std_col = f"{label}_std"
        std = (
            np.array([float(r[std_col]) for r in rows])
            if std_col in fieldnames
            else None
        )
        series[label] = CurveSeries(label=label, mean=mean, std=std)
    return budgets, series


def discover_run_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("traces.jsonl")}, key=lambda p: str(p))


def load_examples(example_ids: set[str], benchmark_seed: int = 42) -> dict:
    adapter = BenchmarkRegistry.build("math500")
    examples = adapter.load(split="test", max_examples=500, seed=benchmark_seed)
    by_id = {ex.example_id: ex for ex in examples}
    if example_ids - set(by_id):
        raise RuntimeError("benchmark example set mismatch")
    return by_id


def load_traces(run_dir: Path) -> list[Trace]:
    with (run_dir / "traces.jsonl").open() as f:
        return [Trace.from_dict(json.loads(line)) for line in f]


def resolve_metrics(raw: str | None, preset: str | None) -> list[str]:
    if preset is not None:
        if preset not in METRIC_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {list(METRIC_PRESETS)}")
        names = list(METRIC_PRESETS[preset])
    elif raw:
        names = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        raise ValueError("specify --metrics or --metric-preset")

    for name in names:
        if name == "accuracy":
            continue
        if name not in MetricRegistry.available():
            raise ValueError(
                f"unknown metric {name!r}; available: {sorted(MetricRegistry.available())}"
            )
    return names


# ---------------------------------------------------------------------------
# Trace truncation & metric collection
# ---------------------------------------------------------------------------


def truncate_trace(trace: Trace, budget: int) -> Trace:
    if budget <= 0:
        return Trace(
            example_id=trace.example_id,
            prompt=trace.prompt,
            segments=[],
            metadata=dict(trace.metadata),
            created_at=trace.created_at,
            finished_at=trace.finished_at,
        )
    remaining = budget
    new_segments: list[TraceSegment] = []
    for seg in trace.segments:
        if remaining <= 0:
            break
        if seg.token_count <= remaining:
            new_segments.append(deepcopy(seg))
            remaining -= seg.token_count
        else:
            if seg.token_count <= 0 or not seg.text:
                break
            char_off = max(
                1,
                min(int(len(seg.text) * remaining / seg.token_count), len(seg.text)),
            )
            left, _ = seg.split_at(char_off)
            left.token_count = remaining
            new_segments.append(left)
            remaining = 0
    return Trace(
        example_id=trace.example_id,
        prompt=trace.prompt,
        segments=new_segments,
        metadata=dict(trace.metadata),
        created_at=trace.created_at,
        finished_at=trace.finished_at,
    )


def _aggregate_values(metric_name: str, values: list[float | int]) -> float:
    arr = np.asarray(values, dtype=float)
    if metric_name in NANMEAN_METRICS:
        arr = arr[arr >= 0]
        return float(np.nanmean(arr)) if arr.size else float("nan")
    return float(np.mean(arr))


def collect_run_curve(
    run_dir: Path,
    examples: dict,
    budgets: list[int],
    metric_names: list[str],
    pbar: tqdm,
    exp_label: str,
) -> np.ndarray:
    traces = load_traces(run_dir)
    adapter = BenchmarkRegistry.build("math500")
    registry_metrics = {
        n: MetricRegistry.build(n) for n in metric_names if n != "accuracy"
    }
    out = np.zeros((len(budgets), len(metric_names)))
    for j, budget in enumerate(budgets):
        per_metric: dict[str, list[float | int]] = {n: [] for n in metric_names}
        for tr in traces:
            truncated = truncate_trace(tr, budget)
            judgement = judge(truncated, examples[tr.example_id], adapter)
            for name in metric_names:
                if name == "accuracy":
                    per_metric[name].append(1.0 if judgement.is_correct else 0.0)
                else:
                    per_metric[name].append(
                        registry_metrics[name].compute(truncated, judgement)
                    )
        for k, name in enumerate(metric_names):
            out[j, k] = _aggregate_values(name, per_metric[name])
        pbar.update(1)
        pbar.set_postfix(exp=exp_label, run=run_dir.name, budget=budget)
    return out


def collect_experiment(
    spec: ExperimentSpec,
    examples: dict,
    budgets: list[int],
    metric_names: list[str],
    pbar: tqdm,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    run_dirs = discover_run_dirs(spec.root)
    if not run_dirs:
        raise FileNotFoundError(f"no traces under {spec.root}")
    if spec.force_single and len(run_dirs) > 1:
        tqdm.write(
            f"warning: {spec.label}: {len(run_dirs)} runs found, using {run_dirs[0].name}"
        )
        run_dirs = run_dirs[:1]

    curves = [
        collect_run_curve(rd, examples, budgets, metric_names, pbar, spec.label)
        for rd in run_dirs
    ]
    stack = np.stack(curves, axis=0)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0, ddof=1) if len(run_dirs) > 1 else None
    is_single = spec.force_single or len(run_dirs) == 1
    return mean, std, is_single


# ---------------------------------------------------------------------------
# CSV & plotting
# ---------------------------------------------------------------------------


def output_stem(metric_name: str, prefix: str, tag: str) -> str:
    base = f"{metric_name}_vs_budget_{tag}"
    return f"{prefix}{base}" if prefix else base


def write_metric_csv(
    path: Path,
    budgets: list[int],
    results: dict[str, tuple[np.ndarray, np.ndarray | None, bool]],
    metric_idx: int,
) -> list[CurveSeries]:
    series_list: list[CurveSeries] = []
    with path.open("w", newline="") as f:
        cols = ["budget"]
        for label, (_, std_all, is_single) in results.items():
            cols.append(f"{label}_mean")
            if not is_single and std_all is not None:
                cols.append(f"{label}_std")
        w = csv.writer(f)
        w.writerow(cols)
        for i, b in enumerate(budgets):
            row: list[str | int] = [b]
            for label, (mean_all, std_all, is_single) in results.items():
                row.append(f"{mean_all[i, metric_idx]:.6f}")
                if not is_single and std_all is not None:
                    row.append(f"{std_all[i, metric_idx]:.6f}")
            w.writerow(row)

    for label, (mean_all, std_all, is_single) in results.items():
        m = mean_all[:, metric_idx]
        s = std_all[:, metric_idx] if std_all is not None else None
        series_list.append(CurveSeries(label=label, mean=m, std=s))
    return series_list


def series_from_raw(
    raw: dict[str, tuple[np.ndarray, np.ndarray | None, bool]],
    metric_idx: int,
) -> dict[str, CurveSeries]:
    out: dict[str, CurveSeries] = {}
    for label, (mean_all, std_all, is_single) in raw.items():
        m = mean_all[:, metric_idx]
        s = std_all[:, metric_idx] if (not is_single and std_all is not None) else None
        out[label] = CurveSeries(label=label, mean=m, std=s)
    return out


def write_series_csv(path: Path, budgets: list[int], series: dict[str, CurveSeries]) -> list[CurveSeries]:
    ordered = _sort_series(list(series.values()))
    with path.open("w", newline="") as f:
        cols = ["budget"]
        for s in ordered:
            cols.append(f"{s.label}_mean")
            if s.std is not None:
                cols.append(f"{s.label}_std")
        w = csv.writer(f)
        w.writerow(cols)
        for i, b in enumerate(budgets):
            row: list[str | int] = [b]
            for s in ordered:
                row.append(f"{s.mean[i]:.6f}")
                if s.std is not None:
                    row.append(f"{s.std[i]:.6f}")
            w.writerow(row)
    return ordered


def _sort_series(series_list: list[CurveSeries]) -> list[CurveSeries]:
    def sort_key(s: CurveSeries) -> tuple[int, str]:
        try:
            return (LABEL_ORDER.index(s.label), s.label)
        except ValueError:
            return (len(LABEL_ORDER), s.label)

    return sorted(series_list, key=sort_key)


def apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("ggplot")
        except OSError:
            pass
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )


def plot_curves(
    budgets: np.ndarray,
    series_list: list[CurveSeries],
    *,
    title: str,
    ylabel: str,
    ymin_at_zero: bool,
    ylim_0_1: bool,
    out_png: Path,
    dpi: int,
    pretty: bool,
) -> None:
    if pretty:
        apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series_list = _sort_series(series_list)

    for s in series_list:
        c = COLORS.get(s.label)
        if s.std is None:
            kwargs = dict(
                label=s.label,
                color=c,
                linewidth=2,
                linestyle="--",
            )
            if pretty:
                kwargs.update(
                    marker="o",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgewidth=1.4,
                )
            ax.plot(budgets, s.mean, **kwargs)
        else:
            kwargs = dict(label=s.label, color=c, linewidth=2)
            if pretty:
                kwargs.update(marker="o", markersize=5)
            ax.plot(budgets, s.mean, **kwargs)
            lower = np.maximum(s.mean - s.std, 0.0) if ymin_at_zero else s.mean - s.std
            ax.fill_between(budgets, lower, s.mean + s.std, color=c, alpha=0.22, linewidth=0)

    ax.set_xlabel("Generation token budget (truncated prefix)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    x_pad = (
        (budgets[-1] - budgets[0]) / max(len(budgets) - 1, 1)
        if len(budgets) > 1
        else budgets[-1] * 0.05
    )
    ax.set_xlim(0, budgets[-1] + x_pad)

    if ylim_0_1:
        ax.set_ylim(0, 1.0)
    elif ymin_at_zero:
        ax.autoscale(axis="y")
        _ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)
    else:
        ax.autoscale(axis="y")

    if not pretty:
        ax.grid(True, alpha=0.3)

    ax.legend(loc="best", framealpha=0.92 if pretty else None, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight" if pretty else None)
    plt.close(fig)


def _metric_key_from_stem(stem: str, tag: str) -> str:
    m = re.match(rf"^(?P<name>.+)_vs_budget_{re.escape(tag)}", stem)
    return m.group("name") if m else stem


def load_curve_csv(path: Path) -> tuple[np.ndarray, list[CurveSeries]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "budget" not in reader.fieldnames:
            raise ValueError(f"{path}: missing 'budget' column")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    labels_in_file: list[str] = []
    for col in fieldnames:
        if col.endswith("_mean"):
            labels_in_file.append(col[: -len("_mean")])

    budgets = np.array([int(r["budget"]) for r in rows], dtype=float)
    series_list: list[CurveSeries] = []
    for label in labels_in_file:
        mean = np.array([float(r[f"{label}_mean"]) for r in rows])
        std_col = f"{label}_std"
        std = (
            np.array([float(r[std_col]) for r in rows])
            if std_col in fieldnames
            else None
        )
        series_list.append(CurveSeries(label=label, mean=mean, std=std))
    return budgets, series_list


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_replot(args: argparse.Namespace) -> None:
    csv_paths = sorted(args.input_dir.glob(args.glob))
    if not csv_paths:
        raise SystemExit(f"No CSV matched {args.input_dir / args.glob}")

    filter_metrics: set[str] | None = None
    if args.metrics or args.metric_preset:
        filter_metrics = set(resolve_metrics(args.metrics, args.metric_preset))

    for csv_path in csv_paths:
        metric_key = _metric_key_from_stem(csv_path.stem, args.output_tag)
        if filter_metrics is not None and metric_key not in filter_metrics:
            continue

        meta = METRIC_META.get(metric_key, {})
        title = str(meta.get("title", f"{metric_key.replace('_', ' ')} vs token budget"))
        ylabel = str(meta.get("ylabel", metric_key))
        ymin_at_zero = bool(meta.get("ymin_at_zero", True))
        ylim_0_1 = bool(meta.get("ylim_0_1", False))

        budgets, series_list = load_curve_csv(csv_path)
        suffix = args.plot_suffix if args.pretty else ""
        out_png = csv_path.with_name(f"{csv_path.stem}{suffix}.png")
        plot_curves(
            budgets,
            series_list,
            title=title,
            ylabel=ylabel,
            ymin_at_zero=ymin_at_zero,
            ylim_0_1=ylim_0_1,
            out_png=out_png,
            dpi=args.dpi,
            pretty=args.pretty,
        )
        tqdm.write(f"Saved {out_png}")


def run_compute(args: argparse.Namespace) -> None:
    if not args.run:
        raise SystemExit("provide at least one --run LABEL:PATH (or use --replot-only)")

    metric_names = resolve_metrics(args.metrics, args.metric_preset)

    single_labels = {
        lb.strip()
        for lb in (args.single_run or "").split(",")
        if lb.strip()
    }
    experiments: list[ExperimentSpec] = []
    for raw in args.run:
        spec = parse_run_arg(raw)
        spec.force_single = spec.label in single_labels
        experiments.append(spec)

    sample_trace = next(experiments[0].root.rglob("traces.jsonl"))
    examples = load_examples(
        {json.loads(line)["example_id"] for line in open(sample_trace)}
    )

    base_series: dict[str, CurveSeries] | None = None
    base_budget_list: list[int] | None = None
    if args.base_csv is not None:
        if not args.base_csv.is_file():
            raise FileNotFoundError(f"base CSV not found: {args.base_csv}")
        base_budgets, base_series = load_base_csv(args.base_csv)
        base_budget_list = [int(b) for b in base_budgets]

    explicit_budgets = any(
        x is not None for x in (args.budgets, args.budgets_file, args.budgets_csv)
    )
    if base_budget_list is not None and not explicit_budgets:
        budgets = base_budget_list
    else:
        budgets = parse_budgets(args)
        if base_budget_list is not None and base_budget_list != budgets:
            tqdm.write(
                "warning: --base-csv budgets differ from active budget list; "
                "computed curves use the active list"
            )

    tqdm.write(f"Budgets ({len(budgets)}): {budgets[:8]}{'...' if len(budgets) > 8 else ''}")
    tqdm.write(f"Metrics: {metric_names}")

    n_runs = sum(
        1 if spec.force_single else len(discover_run_dirs(spec.root))
        for spec in experiments
    )
    n_jobs = n_runs * len(budgets)

    raw: dict[str, tuple[np.ndarray, np.ndarray | None, bool]] = {}
    with tqdm(total=n_jobs, desc="metrics@budget", unit="cell") as pbar:
        for spec in experiments:
            tqdm.write(f"\n{spec.label} ← {spec.root}")
            mean, std, is_single = collect_experiment(
                spec, examples, budgets, metric_names, pbar
            )
            raw[spec.label] = (mean, std, is_single)

    budget_arr = np.array(budgets, dtype=float)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mi, metric_name in enumerate(metric_names):
        stem = output_stem(metric_name, args.output_stem_prefix, args.output_tag)
        out_csv = args.out_dir / f"{stem}.csv"
        out_png = args.out_dir / (
            f"{stem}{args.plot_suffix}.png" if args.pretty else f"{stem}.png"
        )

        computed = series_from_raw(raw, mi)
        if base_series is not None and metric_name == "accuracy":
            merged = dict(base_series)
            merged.update(computed)
            series_list = write_series_csv(out_csv, budgets, merged)
        else:
            series_list = write_metric_csv(out_csv, budgets, raw, mi)

        meta = METRIC_META.get(metric_name, {})
        if not args.no_plot:
            plot_curves(
                budget_arr,
                series_list,
                title=str(meta.get("title", f"{metric_name} vs token budget")),
                ylabel=str(meta.get("ylabel", metric_name)),
                ymin_at_zero=bool(meta.get("ymin_at_zero", True)),
                ylim_0_1=bool(meta.get("ylim_0_1", False)),
                out_png=out_png,
                dpi=args.dpi,
                pretty=args.pretty,
            )
            tqdm.write(f"Saved {out_png}")
        tqdm.write(f"Saved {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="append",
        metavar="LABEL:PATH",
        help="Experiment label and directory with traces (repeatable)",
    )
    parser.add_argument(
        "--single-run",
        metavar="LABELS",
        help="Comma-separated labels: use only first run dir (dashed line, no std)",
    )
    parser.add_argument(
        "--base-csv",
        type=Path,
        help="Precomputed curves CSV (accuracy: all series; merged with --run output)",
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated metrics (include 'accuracy' for judge accuracy)",
    )
    parser.add_argument(
        "--metric-preset",
        choices=sorted(METRIC_PRESETS),
        help=f"Shortcut for metric sets: {', '.join(METRIC_PRESETS)}",
    )
    parser.add_argument("--out-dir", type=Path, default=Path.cwd() / "budget_curves")
    parser.add_argument("--output-stem-prefix", default="")
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument(
        "--plot-suffix",
        default="_pretty",
        help="Suffix before .png when --pretty (default: _pretty)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--pretty", action="store_true", help="Styled plots with markers")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV only")

    budget_group = parser.add_argument_group("budget thresholds")
    budget_group.add_argument("--budgets", metavar="LIST")
    budget_group.add_argument("--budgets-file", type=Path)
    budget_group.add_argument("--budgets-csv", type=Path)
    budget_group.add_argument("--min-budget", type=int, default=DEFAULT_MIN_BUDGET)
    budget_group.add_argument("--max-budget", type=int, default=DEFAULT_MAX_BUDGET)
    budget_group.add_argument("--step", type=int, default=DEFAULT_STEP)

    replot = parser.add_argument_group("replot from CSV (no traces)")
    replot.add_argument(
        "--replot-only",
        action="store_true",
        help="Only render PNGs from existing CSV files",
    )
    replot.add_argument("--input-dir", type=Path, default=Path.cwd())
    replot.add_argument(
        "--glob",
        default="*_vs_budget_*.csv",
        help="Glob under --input-dir for CSV files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.metrics and args.metric_preset:
        parser.error("use only one of --metrics and --metric-preset")

    budget_sources = sum(
        1
        for x in (args.budgets, args.budgets_file, args.budgets_csv)
        if x is not None
    )
    if budget_sources > 1 and not args.replot_only:
        parser.error("use only one of --budgets, --budgets-file, --budgets-csv")

    if args.replot_only:
        if not args.metrics and not args.metric_preset:
            args.metric_preset = None  # replot all matching CSVs
        run_replot(args)
        return

    if not args.metrics and not args.metric_preset:
        parser.error("specify --metrics or --metric-preset for compute mode")

    run_compute(args)


if __name__ == "__main__":
    main()
