from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional


@dataclass
class SegmentTiming:
    label: str
    start: float
    end: Optional[float] = None
    token_count: int = 0

    @property
    def elapsed_ms(self) -> float:
        if self.end is None:
            return (time.perf_counter() - self.start) * 1000
        return (self.end - self.start) * 1000

    @property
    def tokens_per_second(self) -> float:
        elapsed_s = max(self.elapsed_ms / 1000, 1e-9)
        return self.token_count / elapsed_s


class TimingTracker:
    """
    Accumulates wall-clock timings for labelled segments of a pipeline run.

    Usage::

        tracker = TimingTracker()
        with tracker.measure("prefill"):
            ...
        report = tracker.report()
    """

    def __init__(self) -> None:
        self._segments: list[SegmentTiming] = []

    @contextmanager
    def measure(self, label: str, token_count: int = 0) -> Generator[SegmentTiming, None, None]:
        seg = SegmentTiming(label=label, start=time.perf_counter(), token_count=token_count)
        self._segments.append(seg)
        try:
            yield seg
        finally:
            seg.end = time.perf_counter()
            seg.token_count = max(seg.token_count, token_count)

    def record(self, label: str, elapsed_ms: float, token_count: int = 0) -> None:
        t = time.perf_counter()
        seg = SegmentTiming(
            label=label,
            start=t - elapsed_ms / 1000,
            end=t,
            token_count=token_count,
        )
        self._segments.append(seg)

    def report(self) -> dict[str, dict]:
        return {
            s.label: {
                "elapsed_ms": s.elapsed_ms,
                "token_count": s.token_count,
                "tokens_per_second": s.tokens_per_second if s.token_count > 0 else None,
            }
            for s in self._segments
        }

    def total_ms(self) -> float:
        return sum(s.elapsed_ms for s in self._segments)

    def reset(self) -> None:
        self._segments.clear()
