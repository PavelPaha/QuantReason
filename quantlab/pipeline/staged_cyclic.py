"""Helpers for staged execution with cyclic answer ↔ verification loops."""

from __future__ import annotations

from typing import Optional

from quantlab.core.trace import Trace
from quantlab.pipeline.executor import EXECUTOR_STATE_KEY


def trace_pending_stage_idx(trace: Trace) -> Optional[int]:
    """Next pipeline stage index, or ``None`` if this trace has finished."""
    if trace.finished_at is not None:
        return None
    st = trace.metadata.get(EXECUTOR_STATE_KEY)
    if not isinstance(st, dict):
        return None
    raw = st.get("stage_idx")
    if raw is None:
        return None
    return int(raw)


def trace_in_cyclic_loop(
    trace: Trace,
    *,
    loop_stage_indices: list[int],
    max_total_tokens: int,
) -> bool:
    """True while the trace should keep alternating loop stages (answer / periodic)."""
    sid = trace_pending_stage_idx(trace)
    if sid is None or sid not in loop_stage_indices:
        return False
    return trace.total_generated_tokens < max_total_tokens


def cyclic_stage_for_wave(
    wave_index: int,
    *,
    plan_stage_index: int,
    loop_stage_indices: list[int],
) -> int:
    """Map staged wave number to pipeline stage index (plan once, then loop alternation)."""
    if wave_index == 0:
        return plan_stage_index
    return loop_stage_indices[(wave_index - 1) % len(loop_stage_indices)]
