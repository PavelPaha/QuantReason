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
        except ImportError as exc:
            raise ImportError(
                "wandb logging is enabled in the experiment config, but the 'wandb' package "
                "is not installed. Install it with `pip install wandb` or add the optional "
                "dependency for this project."
            ) from exc

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

        row = {
            "experiment_name": self.experiment_config.experiment_name,
            "run_id": self.run_id,
            "benchmark_name": self.experiment_config.benchmark.name,
            "example_id": example_id,
            "is_correct": int(judgement.is_correct),
            "parse_success": int(judgement.parse_success),
            "predicted": judgement.predicted,
            "ground_truth": judgement.ground_truth,
        }
        for key, value in metric_values.items():
            row[key] = _normalize_table_value(value)
        if timing_data:
            row["timing_by_actor"] = _normalize_table_value(timing_data)
        self._rows.append(row)

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

    def finish(self, *, summary: dict[str, Any], run_dir: Optional[Path] = None) -> None:
        if not self.enabled:
            return

        assert self._run is not None

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
