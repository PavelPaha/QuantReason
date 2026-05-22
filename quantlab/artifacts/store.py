from __future__ import annotations

import json
import warnings
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from quantlab.core.trace import Trace
from quantlab.evaluation.judge import JudgementResult

T = TypeVar("T")


def _load_jsonl(
    path: Path,
    *,
    label: str,
    parse: Callable[[dict[str, Any]], T],
) -> list[T]:
    """
    Load JSONL records, skipping blank or corrupt lines.

    Interrupted runs may leave a truncated final line in append-only jsonl files.
    """
    if not path.exists():
        return []
    out: list[T] = []
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"Skipping corrupt {label} line {line_no} in {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not isinstance(row, dict):
            warnings.warn(
                f"Skipping non-object {label} line {line_no} in {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            out.append(parse(row))
        except Exception as exc:
            warnings.warn(
                f"Skipping invalid {label} line {line_no} in {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return out


def _new_run_id() -> str:
    """Timestamp-based run folder name under ``output.base_dir`` (no experiment_name prefix)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


@dataclass
class RunRecord:
    run_id: str
    experiment_name: str
    config: dict[str, Any]
    traces: list[dict]
    judgements: list[dict]
    metrics: list[dict]
    timing: list[dict]
    errors: list[dict]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ArtifactStore:
    """
    Saves and loads all artifacts for a single run.

    Layout::

        {base_dir}/
          {YYYYMMDD}_{HHMMSS}_{uuid6}/
            config.json
            traces.jsonl
            trace_checkpoints/   # staged runs: wave_0.jsonl, wave_1.jsonl, …
            judgements.jsonl
            metrics.jsonl
            timing.jsonl
            errors.jsonl
            summary.json
    """

    def __init__(self, base_dir: str = "results") -> None:
        self.base_dir = Path(base_dir)

    def new_run(self, experiment_name: str, config: dict) -> str:
        run_id = _new_run_id()
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        return run_id

    def save_trace(self, run_id: str, trace: Trace) -> None:
        path = self.base_dir / run_id / "traces.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

    def save_staged_wave_checkpoint(
        self,
        run_id: str,
        wave_index: int,
        traces_by_id: dict[str, Trace],
    ) -> None:
        """
        JSONL snapshot after staged wave ``wave_index`` (same dict shape as ``save_trace``).

        Lets you read partial traces (e.g. plan-only) if a later wave crashes.
        """
        if not traces_by_id:
            return
        chk = self.base_dir / run_id / "trace_checkpoints"
        chk.mkdir(parents=True, exist_ok=True)
        path = chk / f"wave_{wave_index}.jsonl"
        lines = [json.dumps(traces_by_id[eid].to_dict()) for eid in sorted(traces_by_id)]
        path.write_text("\n".join(lines) + "\n")

    def load_staged_wave_checkpoint(self, run_id: str, wave_index: int) -> list[Trace]:
        path = self.base_dir / run_id / "trace_checkpoints" / f"wave_{wave_index}.jsonl"
        return _load_jsonl(
            path,
            label=f"trace checkpoint wave_{wave_index}",
            parse=Trace.from_dict,
        )

    def list_staged_wave_checkpoint_indices(self, run_id: str) -> list[int]:
        """Sorted wave indices with ``trace_checkpoints/wave_<n>.jsonl`` present."""
        chk = self.base_dir / run_id / "trace_checkpoints"
        if not chk.is_dir():
            return []
        indices: list[int] = []
        for path in chk.glob("wave_*.jsonl"):
            suffix = path.stem.removeprefix("wave_")
            try:
                indices.append(int(suffix))
            except ValueError:
                continue
        return sorted(indices)

    def save_judgement(self, run_id: str, j: JudgementResult) -> None:
        path = self.base_dir / run_id / "judgements.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({
                "example_id": j.example_id,
                "predicted": j.predicted,
                "ground_truth": j.ground_truth,
                "is_correct": j.is_correct,
                "parse_success": j.parse_success,
            }) + "\n")

    def save_metrics(self, run_id: str, example_id: str, metrics: dict[str, Any]) -> None:
        path = self.base_dir / run_id / "metrics.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"example_id": example_id, **metrics}) + "\n")

    def save_timing(self, run_id: str, example_id: str, timing: dict[str, Any]) -> None:
        path = self.base_dir / run_id / "timing.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"example_id": example_id, **timing}) + "\n")

    def save_error(self, run_id: str, example_id: str, error: str) -> None:
        path = self.base_dir / run_id / "errors.jsonl"
        with path.open("a") as f:
            f.write(json.dumps({"example_id": example_id, "error": error}) + "\n")

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        path = self.base_dir / run_id / "summary.json"
        path.write_text(json.dumps(summary, indent=2))

    def list_judged_example_ids(self, run_id: str) -> set[str]:
        path = self.base_dir / run_id / "judgements.jsonl"
        return {
            str(row["example_id"])
            for row in _load_jsonl(path, label="judgement", parse=lambda d: d)
            if row.get("example_id") is not None
        }

    def list_completed_example_ids(self, run_id: str) -> set[str]:
        """
        Example IDs to skip when resuming a non-staged run.

        Uses judgements when present; otherwise a trace with ``finished_at`` set
        (pipeline finished, even if judgement was not persisted).
        """
        completed = self.list_judged_example_ids(run_id)
        for trace in self.load_traces(run_id):
            if trace.finished_at is not None:
                completed.add(trace.example_id)
        return completed

    def load_judgements(self, run_id: str) -> list[dict]:
        path = self.base_dir / run_id / "judgements.jsonl"
        return _load_jsonl(path, label="judgement", parse=lambda d: d)

    # ── loading ───────────────────────────────────────────────────────────────

    def load_traces(self, run_id: str) -> list[Trace]:
        path = self.base_dir / run_id / "traces.jsonl"
        return _load_jsonl(path, label="trace", parse=Trace.from_dict)

    def load_metrics(self, run_id: str) -> list[dict]:
        path = self.base_dir / run_id / "metrics.jsonl"
        return _load_jsonl(path, label="metric", parse=lambda d: d)

    def load_errors(self, run_id: str) -> list[dict]:
        path = self.base_dir / run_id / "errors.jsonl"
        return _load_jsonl(path, label="error", parse=lambda d: d)

    def list_runs(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )

    def load_config(self, run_id: str) -> dict:
        path = self.base_dir / run_id / "config.json"
        return json.loads(path.read_text())

    def run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id
