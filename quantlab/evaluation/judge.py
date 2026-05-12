from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample
from quantlab.core.trace import Trace
from quantlab.evaluation.extractor import extract_from_trace


@dataclass
class JudgementResult:
    example_id: str
    predicted: Optional[str]
    ground_truth: str
    is_correct: bool
    parse_success: bool


def judge(
    trace: Trace,
    example: BenchmarkExample,
    adapter: BenchmarkAdapter,
) -> JudgementResult:
    predicted = extract_from_trace(trace, adapter)
    parse_success = predicted is not None
    correct = adapter.is_correct(predicted, example.ground_truth) if parse_success else False
    return JudgementResult(
        example_id=example.example_id,
        predicted=predicted,
        ground_truth=example.ground_truth,
        is_correct=correct,
        parse_success=parse_success,
    )


def judge_batch(
    traces: list[Trace],
    examples: list[BenchmarkExample],
    adapter: BenchmarkAdapter,
) -> list[JudgementResult]:
    return [judge(t, e, adapter) for t, e in zip(traces, examples)]
