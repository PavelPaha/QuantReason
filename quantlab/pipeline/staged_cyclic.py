"""Helpers for staged execution with cyclic answer ↔ verification loops."""

from __future__ import annotations

from typing import Optional

from quantlab.core.trace import Trace
from quantlab.pipeline.executor import EXECUTOR_STATE_KEY


def example_runnable_at_stage(
    example_id: str,
    stage_idx: int,
    traces: dict[str, Trace],
    failed: set[str],
    *,
    plan_stage_index: int,
    use_staged_cyclic: bool,
) -> bool:
    """True if this example still needs pipeline stage ``stage_idx`` on the current wave."""
    if example_id in failed:
        return False
    if example_id not in traces:
        if use_staged_cyclic:
            return stage_idx == plan_stage_index
        return stage_idx == 0
    pending = trace_pending_stage_idx(traces[example_id])
    if pending is None:
        tr = traces[example_id]
        if tr.finished_at is None and not tr.segments:
            if use_staged_cyclic:
                return stage_idx == plan_stage_index
            return stage_idx == 0
        return False
    return pending == stage_idx


def staged_wave_has_pending_work(
    traces: dict[str, Trace],
    examples,
    *,
    failed: set[str],
    skip: set[str],
    current_stage: int,
    plan_stage_index: int,
    use_staged_cyclic: bool,
) -> bool:
    """True if any example still needs work at ``current_stage`` for this wave."""
    for example in examples:
        eid = example.example_id
        if eid in failed or eid in skip:
            continue
        if example_runnable_at_stage(
            eid,
            current_stage,
            traces,
            failed,
            plan_stage_index=plan_stage_index,
            use_staged_cyclic=use_staged_cyclic,
        ):
            return True
    return False


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


def trace_loop_generated_tokens(trace: Trace, loop_actor_ids: set[str]) -> int:
    """Sum of ``token_count`` for segments produced by loop-stage actors only."""
    if not loop_actor_ids:
        return 0
    return sum(s.token_count for s in trace.segments if s.actor_id in loop_actor_ids)


def trace_in_cyclic_loop(
    trace: Trace,
    *,
    loop_stage_indices: list[int],
    loop_actor_ids: set[str],
    max_total_tokens: int,
    max_loop_tokens: Optional[int] = None,
) -> bool:
    """True while the trace should keep alternating loop stages (answer / periodic)."""
    sid = trace_pending_stage_idx(trace)
    if sid is None or sid not in loop_stage_indices:
        return False
    if max_loop_tokens is not None:
        return trace_loop_generated_tokens(trace, loop_actor_ids) < max_loop_tokens
    return trace.total_generated_tokens < max_total_tokens


def cyclic_stage_for_wave(
    wave_index: int,
    *,
    plan_stage_index: int,
    loop_stage_indices: list[int],
    preface_stage_indices: Optional[list[int]] = None,
) -> int:
    """Map staged wave number to pipeline stage index.

    Wave 0 = plan, then optional one-shot preface stages (e.g. GPTQ bootstrap with a
    stage prompt), then cyclic alternation over ``loop_stage_indices``.
    """
    preface = preface_stage_indices or []
    if wave_index == 0:
        return plan_stage_index
    offset = wave_index - 1
    if offset < len(preface):
        return preface[offset]
    loop_offset = offset - len(preface)
    return loop_stage_indices[loop_offset % len(loop_stage_indices)]
