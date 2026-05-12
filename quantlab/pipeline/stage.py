from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from quantlab.core.types import HandoffMode, SegmentRole
from quantlab.switching.base import SwitchCondition


@dataclass
class PipelineStage:
    """
    One step in a pipeline: an actor with its exit conditions.

    The executor runs the actor until *any* of the exit_conditions fires
    (or the actor's natural stop / max_new_tokens is reached), then moves
    to the next stage (or the fallback_stage_index if specified).

    Args:
        actor_id: ID of the actor to use (must be registered in the run's actor pool)
        exit_conditions: list of SwitchCondition objects; firing any one causes a switch
        handoff_mode: how this actor ingests the trace from the previous stage
        max_new_tokens: per-stage token cap (overrides actor config if set)
        stop_sequences: extra stop strings for this stage
        role: semantic role label for the generated segment
        fallback_stage_index: if set, a trigger-based switch routes here instead of stage+1
        loop_back_stage_index: if set, loop detectors / format failures route here
    """

    actor_id: str
    exit_conditions: list[SwitchCondition] = field(default_factory=list)
    handoff_mode: HandoffMode = HandoffMode.FULL_PREFILL
    max_new_tokens: Optional[int] = None
    stop_sequences: list[str] = field(default_factory=list)
    role: SegmentRole = SegmentRole.UNKNOWN
    fallback_stage_index: Optional[int] = None
    loop_back_stage_index: Optional[int] = None
    # Appended to trace.full_text before calling the actor; not stored in trace.
    # Use to inject stage-specific instructions (e.g., "Create a plan:", "Solve step by step:").
    stage_prompt: str = ""
