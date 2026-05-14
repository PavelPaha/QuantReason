from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from quantlab.core.trace import Trace, TraceSegment


@dataclass
class SwitchDecision:
    """Result of evaluating a switch condition against the latest segment."""

    should_switch: bool
    # Character offset within the segment's text where the split should happen.
    # None means use the full segment (split after the segment ends).
    split_char_offset: Optional[int] = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # If set (usually from YAML ``target_stage_index`` on the firing condition),
    # the executor jumps here instead of ``stage_idx + 1`` / ``fallback_stage_index``.
    routing_stage_index: Optional[int] = None


class SwitchCondition(ABC):
    """
    Base class for all switching conditions.

    A condition examines the trace *after* a segment has been generated and
    returns a SwitchDecision.  Conditions can optionally specify a split point
    inside the segment — allowing the executor to attribute the tokens before
    the split to the current actor and pass only the remainder forward.
    """

    name: str = "unnamed"

    @abstractmethod
    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        """Evaluate whether a switch should occur after new_segment was generated."""
        ...

    def reset(self) -> None:
        """Reset internal state between examples. Override as needed."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
