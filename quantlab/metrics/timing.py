from __future__ import annotations

from quantlab.core.trace import Trace
from quantlab.evaluation.judge import JudgementResult
from quantlab.metrics.base import MetricBase


class TotalGenerationTimeMetric(MetricBase):
    """Total wall-clock generation time in ms (sum of all segment timings)."""

    name = "total_generation_ms"

    def compute(self, trace: Trace, judgement: JudgementResult) -> float:
        total = 0.0
        for seg in trace.segments:
            if seg.timing and seg.timing.total_ms is not None:
                total += seg.timing.total_ms
        return total


class SegmentTimingMetric(MetricBase):
    """Per-actor total generation time in ms."""

    name = "segment_timing_ms"

    def compute(self, trace: Trace, judgement: JudgementResult) -> dict[str, float]:
        result: dict[str, float] = {}
        for seg in trace.segments:
            if seg.timing and seg.timing.total_ms is not None:
                result[seg.actor_id] = result.get(seg.actor_id, 0.0) + seg.timing.total_ms
        return result


class TokensPerSecondMetric(MetricBase):
    """Effective tokens/s for each actor (harmonic mean of segment tps)."""

    name = "tokens_per_second"

    def compute(self, trace: Trace, judgement: JudgementResult) -> dict[str, float]:
        actor_tokens: dict[str, int] = {}
        actor_ms: dict[str, float] = {}
        for seg in trace.segments:
            if seg.timing and seg.timing.total_ms is not None:
                actor_tokens[seg.actor_id] = actor_tokens.get(seg.actor_id, 0) + seg.token_count
                actor_ms[seg.actor_id] = actor_ms.get(seg.actor_id, 0.0) + seg.timing.total_ms
        return {
            aid: actor_tokens[aid] / max(actor_ms[aid] / 1000, 1e-9)
            for aid in actor_tokens
            if actor_ms.get(aid, 0) > 0
        }
