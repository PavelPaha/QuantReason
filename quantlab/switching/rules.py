from __future__ import annotations

import re

from quantlab.core.trace import Trace, TraceSegment
from quantlab.switching.base import SwitchCondition, SwitchDecision


class AfterNTokens(SwitchCondition):
    """Switch after the trace has accumulated at least N generated tokens."""

    name = "after_n_tokens"

    def __init__(self, n: int) -> None:
        self.n = n

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        total = trace.total_generated_tokens
        if total >= self.n:
            return SwitchDecision(should_switch=True, reason=f"total_tokens={total} >= {self.n}")
        return SwitchDecision(should_switch=False)


class AfterMarker(SwitchCondition):
    """Switch when the generated text contains a specific marker string."""

    name = "after_marker"

    def __init__(self, marker: str, keep_marker: bool = True) -> None:
        self.marker = marker
        self.keep_marker = keep_marker

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        text = new_segment.text
        idx = text.find(self.marker)
        if idx == -1:
            return SwitchDecision(should_switch=False)
        split_at = idx + len(self.marker) if self.keep_marker else idx
        return SwitchDecision(
            should_switch=True,
            split_char_offset=split_at,
            reason=f"marker={self.marker!r} found at offset {idx}",
        )


class AfterRegex(SwitchCondition):
    """Switch when the generated text matches a regex pattern."""

    name = "after_regex"

    def __init__(
        self,
        pattern: str,
        *,
        keep_match: bool = True,
        ignore_case: bool = False,
        multiline: bool = False,
    ) -> None:
        flags = 0
        if ignore_case:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        self.pattern = pattern
        self.keep_match = keep_match
        self._regex = re.compile(pattern, flags)

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        text = new_segment.text
        match = self._regex.search(text)
        if match is None:
            return SwitchDecision(should_switch=False)
        split_at = match.end() if self.keep_match else match.start()
        return SwitchDecision(
            should_switch=True,
            split_char_offset=split_at,
            reason=f"regex={self.pattern!r} matched at offset {match.start()}",
        )


class AfterFirstSegment(SwitchCondition):
    """Switch immediately after the first segment — useful for plan→reasoning handoff."""

    name = "after_first_segment"

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        # Switch if this is the first segment
        if len(trace.segments) == 1:
            return SwitchDecision(should_switch=True, reason="first_segment_done")
        return SwitchDecision(should_switch=False)


class AlwaysSwitch(SwitchCondition):
    """Switch unconditionally after every segment (single-shot stages)."""

    name = "always"

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        return SwitchDecision(should_switch=True, reason="always_switch")


class NeverSwitch(SwitchCondition):
    """Never switch — actor runs until a different condition fires."""

    name = "never"

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        return SwitchDecision(should_switch=False)
