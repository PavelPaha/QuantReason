from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quantlab.core.trace import Trace
from quantlab.evaluation.judge import JudgementResult


class MetricBase(ABC):
    """
    Computes a scalar or dict value from a (trace, judgement) pair.

    Metrics are stateless — they receive the full trace and the judgement and
    return a value.  For aggregate statistics (mean accuracy etc.) see the
    aggregator module.
    """

    name: str = "unnamed"

    @abstractmethod
    def compute(self, trace: Trace, judgement: JudgementResult) -> Any:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
