from __future__ import annotations

from quantlab.config.schema import BenchmarkConfig, ExperimentConfig, StageConfig
from quantlab.runner import _build_run_summary
from quantlab.wandb_logger import _normalize_table_value


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
