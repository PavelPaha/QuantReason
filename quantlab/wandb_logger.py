from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from quantlab.config.schema import ExperimentConfig, WandbConfig
from quantlab.evaluation.judge import JudgementResult


def _normalize_table_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _build_table_row(
    *,
    experiment_name: str,
    benchmark_name: str,
    run_id: str,
    example_id: str,
    judgement_data: dict[str, Any],
    metric_values: dict[str, Any],
    timing_data: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "example_id": example_id,
        "is_correct": int(bool(judgement_data.get("is_correct"))),
        "parse_success": int(bool(judgement_data.get("parse_success"))),
        "predicted": judgement_data.get("predicted"),
        "ground_truth": judgement_data.get("ground_truth"),
    }
    for key, value in metric_values.items():
        if key == "example_id":
            continue
        row[key] = _normalize_table_value(value)
    if timing_data:
        normalized_timing = {k: v for k, v in timing_data.items() if k != "example_id"}
        if normalized_timing:
            row["timing_by_actor"] = _normalize_table_value(normalized_timing)
    return row


def _build_summary_log_payload(summary: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            payload[f"summary/{key}"] = float(value)
        elif isinstance(value, (int, float)):
            payload[f"summary/{key}"] = float(value)
    return payload


def build_rows_from_run_directory(
    *,
    run_dir: Path,
    experiment_config: ExperimentConfig,
    run_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    judgements_path = run_dir / "judgements.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    timing_path = run_dir / "timing.jsonl"

    judgements_by_id: dict[str, dict[str, Any]] = {}
    metrics_by_id: dict[str, dict[str, Any]] = {}
    timing_by_id: dict[str, dict[str, Any]] = {}
    ordered_example_ids: list[str] = []

    if judgements_path.exists():
        for line in judgements_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            example_id = str(item["example_id"])
            judgements_by_id[example_id] = item
            ordered_example_ids.append(example_id)

    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            metrics_by_id[str(item["example_id"])] = item

    if timing_path.exists():
        for line in timing_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            timing_by_id[str(item["example_id"])] = item

    seen = set(ordered_example_ids)
    for source in (metrics_by_id, timing_by_id):
        for example_id in source:
            if example_id not in seen:
                ordered_example_ids.append(example_id)
                seen.add(example_id)

    effective_run_id = run_id or run_dir.name
    return [
        _build_table_row(
            experiment_name=experiment_config.experiment_name,
            benchmark_name=experiment_config.benchmark.name,
            run_id=effective_run_id,
            example_id=example_id,
            judgement_data=judgements_by_id.get(example_id, {"example_id": example_id}),
            metric_values=metrics_by_id.get(example_id, {}),
            timing_data=timing_by_id.get(example_id, {}),
        )
        for example_id in ordered_example_ids
    ]


class WandbRunLogger:
    """Optional W&B logger that mirrors local artifacts with summaries and per-example tables."""

    def __init__(
        self,
        *,
        config: Optional[WandbConfig],
        experiment_config: ExperimentConfig,
        run_id: str,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.experiment_config = experiment_config
        self.run_id = run_id
        self.verbose = verbose
        self.enabled = bool(config and config.enabled)
        self._wandb: Any = None
        self._run: Any = None
        self._rows: list[dict[str, Any]] = []
        self._judged_examples = 0
        self._error_examples = 0
        self._correct_examples = 0
        self._parsed_examples = 0
        self._progress_events = 0

        if not self.enabled:
            return

        try:
            import wandb
        except ImportError:
            self.enabled = False
            if verbose:
                print(
                    "[wandb] disabled: package is not installed. "
                    "Local artifacts will still be written."
                )
            return

        self._wandb = wandb
        run_name = config.name or f"{experiment_config.experiment_name}-{run_id}"
        wandb_init_kwargs: dict[str, Any] = {
            "project": config.project,
            "name": run_name,
            "job_type": config.job_type,
            "id": run_id,
            "resume": "allow",
            "config": {
                "run_id": run_id,
                "experiment": experiment_config.model_dump(mode="python"),
            },
        }
        if config.entity:
            wandb_init_kwargs["entity"] = config.entity
        if config.group:
            wandb_init_kwargs["group"] = config.group
        if config.tags:
            wandb_init_kwargs["tags"] = config.tags
        if config.notes:
            wandb_init_kwargs["notes"] = config.notes
        if config.mode:
            wandb_init_kwargs["mode"] = config.mode

        self._run = wandb.init(**wandb_init_kwargs)
        if verbose:
            print(
                f"[wandb] initialized project={config.project!r} "
                f"run_name={run_name!r} run_id={run_id}"
            )

    def record_example(
        self,
        *,
        example_id: str,
        judgement: JudgementResult,
        metric_values: dict[str, Any],
        timing_data: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return

        self._judged_examples += 1
        self._correct_examples += int(judgement.is_correct)
        self._parsed_examples += int(judgement.parse_success)

        self._rows.append(
            _build_table_row(
                experiment_name=self.experiment_config.experiment_name,
                benchmark_name=self.experiment_config.benchmark.name,
                run_id=self.run_id,
                example_id=example_id,
                judgement_data={
                    "is_correct": judgement.is_correct,
                    "parse_success": judgement.parse_success,
                    "predicted": judgement.predicted,
                    "ground_truth": judgement.ground_truth,
                },
                metric_values=metric_values,
                timing_data=timing_data,
            )
        )

        interval = max(int(self.config.progress_log_interval), 1)
        if self._judged_examples % interval == 0:
            self._log_progress()

    def record_error(self, *, example_id: str, error: str, wave_index: Optional[int] = None) -> None:
        if not self.enabled:
            return

        self._error_examples += 1
        payload: dict[str, Any] = {
            "progress/judged_examples": self._judged_examples,
            "progress/error_examples": self._error_examples,
            "progress/processed_examples": self._judged_examples + self._error_examples,
        }
        if wave_index is not None:
            payload["progress/wave_index"] = wave_index
        if self.verbose:
            print(f"[wandb] recorded error for {example_id}: {error.splitlines()[-1]}")
        self._log(payload)

    def log_wave_end(
        self,
        *,
        wave_index: int,
        total_waves: int,
        persisted_examples: int,
    ) -> None:
        if not self.enabled:
            return

        self._log(
            {
                "progress/wave_index": wave_index,
                "progress/total_waves": total_waves,
                "progress/persisted_examples": persisted_examples,
                "progress/judged_examples": self._judged_examples,
                "progress/error_examples": self._error_examples,
            }
        )

    def load_saved_run(self, *, run_dir: Path) -> None:
        if not self.enabled:
            return
        self._rows = build_rows_from_run_directory(
            run_dir=run_dir,
            experiment_config=self.experiment_config,
            run_id=self.run_id,
        )

    def finish(self, *, summary: dict[str, Any], run_dir: Optional[Path] = None) -> None:
        if not self.enabled:
            return

        assert self._run is not None

        summary_payload = _build_summary_log_payload(summary)
        if summary_payload:
            self._log(summary_payload)

        for key, value in summary.items():
            self._run.summary[key] = value

        if self.config.log_per_example_table and self._rows:
            import pandas as pd

            df = pd.DataFrame(self._rows)
            table = self._wandb.Table(dataframe=df)
            self._log({self.config.per_example_table_key: table})

        if self.config.upload_run_artifact and run_dir is not None:
            artifact_name = f"{self.config.artifact_name}-{self.run_id}"
            artifact = self._wandb.Artifact(name=artifact_name, type="quantlab-run")
            artifact.add_dir(str(run_dir))
            self._run.log_artifact(artifact)

        self._run.finish()

    def _log_progress(self) -> None:
        denom = max(self._judged_examples, 1)
        payload = {
            "progress/judged_examples": self._judged_examples,
            "progress/error_examples": self._error_examples,
            "progress/processed_examples": self._judged_examples + self._error_examples,
            "progress/running_accuracy": self._correct_examples / denom,
            "progress/running_parse_rate": self._parsed_examples / denom,
        }
        self._log(payload)

    def _log(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        self._progress_events += 1
        assert self._run is not None
        self._run.log(payload, step=self._progress_events)
