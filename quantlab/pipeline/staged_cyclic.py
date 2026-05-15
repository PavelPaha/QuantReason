"""Helpers for staged execution with cyclic answer ↔ verification loops."""

from __future__ import annotations

from typing import Iterator, Optional

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


def trace_needs_finalize(trace: Trace, finalize_stage_index: int) -> bool:
    return trace_pending_stage_idx(trace) == finalize_stage_index


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


def iter_staged_cyclic_waves(
    *,
    wave_start: int,
    loop_stage_indices: list[int],
    plan_stage_index: int,
    finalize_stage_index: Optional[int],
    traces: dict[str, Trace],
    failed: set[str],
    max_total_tokens: int,
) -> Iterator[tuple[int, int]]:
    """
    Yield ``(wave_index, stage_index)`` for staged cyclic runs.

    Stops when no trace remains in the loop stages under ``max_total_tokens``, then
    optionally emits one finalize wave.
    """
    if not loop_stage_indices:
        raise ValueError("staged_cyclic_loop_stage_indices must be non-empty")

    w = wave_start
    if w == 0:
        yield 0, plan_stage_index
        w = 1

    while True:
        active = [
            t
            for eid, t in traces.items()
            if eid not in failed
            and trace_in_cyclic_loop(
                t,
                loop_stage_indices=loop_stage_indices,
                max_total_tokens=max_total_tokens,
            )
        ]
        if not active:
            break
        yield w, cyclic_stage_for_wave(
            w,
            plan_stage_index=plan_stage_index,
            loop_stage_indices=loop_stage_indices,
        )
        w += 1

    if finalize_stage_index is not None:
        needs_finalize = any(
            eid not in failed and trace_needs_finalize(t, finalize_stage_index)
            for eid, t in traces.items()
        )
        if needs_finalize:
            yield w, finalize_stage_index
