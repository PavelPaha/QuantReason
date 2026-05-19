from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from quantlab.core.types import HandoffMode, SegmentRole, StagePromptPlacement
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
        exit_condition_targets: parallel to exit_conditions; optional jump target per rule
        fallback_stage_index: if set, a trigger-based switch routes here instead of stage+1
        loop_back_stage_index: if set, loop detectors / format failures route here
        natural_next_stage_index: if set, natural completion routes here instead of stage+1
    """

    actor_id: str
    exit_conditions: list[SwitchCondition] = field(default_factory=list)
    exit_condition_targets: list[Optional[int]] = field(default_factory=list)
    exit_condition_end_pipeline: list[bool] = field(default_factory=list)
    handoff_mode: HandoffMode = HandoffMode.FULL_PREFILL
    max_new_tokens: Optional[int] = None
    stop_sequences: list[str] = field(default_factory=list)
    role: SegmentRole = SegmentRole.UNKNOWN
    fallback_stage_index: Optional[int] = None
    loop_back_stage_index: Optional[int] = None
    natural_next_stage_index: Optional[int] = None
    # Appended to handoff prefix before generate; see ``exclude_stage_prompt_from_trace``.
    stage_prompt: str = ""
    stage_system_prompt: str = ""
    stage_prompt_placement: StagePromptPlacement = StagePromptPlacement.ASSISTANT_SUFFIX
    exclude_stage_prompt_from_trace: bool = False
    handoff_plan_label: str = ""
