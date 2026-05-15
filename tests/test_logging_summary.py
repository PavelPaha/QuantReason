from __future__ import annotations

import json
from pathlib import Path

from quantlab.config.schema import BenchmarkConfig, ExperimentConfig, StageConfig, WandbConfig
from quantlab.runner import _ProgressHeartbeat, _build_run_summary, _format_elapsed
from quantlab.wandb_logger import (
    WandbRunLogger,
    _build_summary_log_payload,
    _normalize_table_value,
    build_rows_from_run_directory,
)


def _minimal_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="demo",
        benchmark=BenchmarkConfig(name="math500"),
        actors=[],
        pipeline=[StageConfig(actor_id="noop")],
    )


def test_build_run_summary_adds_scalar_metric_means():
    summary = _build_run_summary(
        run_id="r1",
        config=_minimal_config(),
        n_examples=3,
        judgements=[
            {"is_correct": True, "parse_success": True},
            {"is_correct": False, "parse_success": True},
        ],
        metrics_rows=[
            {"example_id": "a", "reasoning_length": 10, "loop_detected": 0, "actor_token_split": {"fp": 10}},
            {"example_id": "b", "reasoning_length": 20, "loop_detected": 1, "actor_token_split": {"fp": 20}},
        ],
        error_count=1,
        output_dir="/tmp/r1",
    )

    assert summary["accuracy"] == 0.5
    assert summary["parse_rate"] == 1.0
    assert summary["n_errors"] == 1
    assert summary["reasoning_length_mean"] == 15
    assert summary["reasoning_length_n"] == 2
    assert summary["loop_detected_mean"] == 0.5
    assert "actor_token_split_mean" not in summary


def test_normalize_table_value_keeps_scalars_and_serializes_nested():
    assert _normalize_table_value("x") == "x"
    assert _normalize_table_value(3) == 3
    assert _normalize_table_value({"a": 1}) == '{"a": 1}'


def test_build_summary_log_payload_keeps_only_scalar_values():
    payload = _build_summary_log_payload(
        {
            "accuracy": 0.5,
            "n_examples": 3,
            "output_dir": "/tmp/r1",
            "metadata": {"a": 1},
            "is_smoke": True,
        }
    )

    assert payload == {
        "summary/accuracy": 0.5,
        "summary/n_examples": 3.0,
        "summary/is_smoke": 1.0,
    }


def test_build_rows_from_run_directory_reconstructs_joined_table(tmp_path: Path):
    (tmp_path / "judgements.jsonl").write_text(
        json.dumps(
            {
                "example_id": "ex-1",
                "predicted": "42",
                "ground_truth": "42",
                "is_correct": True,
                "parse_success": True,
            }
        )
        + "\n"
    )
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"example_id": "ex-1", "reasoning_length": 7, "nested_metric": {"a": 1}}) + "\n"
    )
    (tmp_path / "timing.jsonl").write_text(
        json.dumps({"example_id": "ex-1", "planner": {"total_ms": 12.5, "token_count": 3}}) + "\n"
    )

    rows = build_rows_from_run_directory(
        run_dir=tmp_path,
        experiment_config=_minimal_config(),
        run_id="restored-run",
    )

    assert rows == [
        {
            "experiment_name": "demo",
            "run_id": "restored-run",
            "benchmark_name": "math500",
            "example_id": "ex-1",
            "is_correct": 1,
            "parse_success": 1,
            "predicted": "42",
            "ground_truth": "42",
            "reasoning_length": 7,
            "nested_metric": '{"a": 1}',
            "timing_by_actor": '{"planner": {"token_count": 3, "total_ms": 12.5}}',
        }
    ]


def test_wandb_logger_disables_itself_when_dependency_is_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("wandb not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    logger = WandbRunLogger(
        config=WandbConfig(enabled=True, project="quantlab"),
        experiment_config=_minimal_config(),
        run_id="r1",
    )

    assert logger.enabled is False


def test_format_elapsed_renders_minutes_and_hours():
    assert _format_elapsed(59.9) == "00:59"
    assert _format_elapsed(3661) == "01:01:01"


def test_progress_heartbeat_logs_only_after_interval(monkeypatch):
    emitted: list[str] = []
    monotonic_values = iter([0.0, 100.0, 301.0])

    monkeypatch.setattr("quantlab.runner.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("quantlab.runner._stdout_log", emitted.append)

    hb = _ProgressHeartbeat(total_examples=10, interval_sec=300.0)
    hb.maybe_log(completed_examples=1, error_count=0, current_example=2)
    hb.maybe_log(completed_examples=3, error_count=1, current_example=4)

    assert emitted == [
        "[runner] progress done=3/10 (30.0%) errors=1 cursor=4/10 elapsed=05:01"
    ]
