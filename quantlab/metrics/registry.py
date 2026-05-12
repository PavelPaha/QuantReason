from __future__ import annotations

from typing import Any, Type

from quantlab.metrics.base import MetricBase


class MetricRegistry:
    _registry: dict[str, Type[MetricBase]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(metric_cls: Type[MetricBase]) -> Type[MetricBase]:
            metric_cls.name = name
            cls._registry[name] = metric_cls
            return metric_cls
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> MetricBase:
        if name not in cls._registry:
            raise KeyError(f"Unknown metric {name!r}. Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)

    @classmethod
    def build_all(cls, names: list[str]) -> list[MetricBase]:
        return [cls.build(n) for n in names]


def _register_builtins() -> None:
    from quantlab.metrics import accuracy, generation, timing

    mapping = {
        "accuracy": accuracy.AccuracyMetric,
        "parse_rate": accuracy.ParseRateMetric,
        "exact_match": accuracy.ExactMatchMetric,
        "reasoning_length": generation.ReasoningLengthMetric,
        "loop_detected": generation.LoopDetectedMetric,
        "loop_onset_tokens": generation.LoopOnsetTokensMetric,
        "think_closed": generation.ThinkClosedMetric,
        "commit_gap": generation.CommitGapMetric,
        "tokens_to_first_correct": generation.TokensToFirstCorrectMetric,
        "finish_commit": generation.FinishCommitMetric,
        "verification_spiral": generation.VerificationSpiralMetric,
        "stop_token_probe": generation.StopTokenProbeMetric,
        "actor_token_split": generation.ActorTokenSplitMetric,
        "total_generation_ms": timing.TotalGenerationTimeMetric,
        "segment_timing_ms": timing.SegmentTimingMetric,
        "tokens_per_second": timing.TokensPerSecondMetric,
    }
    for name, cls in mapping.items():
        MetricRegistry._registry.setdefault(name, cls)


_register_builtins()
