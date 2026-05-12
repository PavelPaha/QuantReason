from __future__ import annotations

from quantlab.core.trace import Trace
from quantlab.evaluation.judge import JudgementResult
from quantlab.metrics.base import MetricBase


class AccuracyMetric(MetricBase):
    name = "accuracy"

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        return int(judgement.is_correct)


class ParseRateMetric(MetricBase):
    name = "parse_rate"

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        return int(judgement.parse_success)


class ExactMatchMetric(MetricBase):
    """Strict exact-match between predicted and ground truth strings."""

    name = "exact_match"

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        if judgement.predicted is None:
            return 0
        return int(judgement.predicted.strip() == judgement.ground_truth.strip())
