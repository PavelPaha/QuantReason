from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Optional

from quantlab.actors.base import ActorBase
from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import HandoffMode
from quantlab.pipeline.stage import PipelineStage
from quantlab.switching.base import SwitchDecision

# Saved on Trace.metadata when execution pauses or completes, for staged/resumable runs.
EXECUTOR_STATE_KEY = "_executor"


def _executor_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class PipelineExecutor:
    """
    Runs a sequence of pipeline stages to build a Trace.

    The executor maintains a pool of actors keyed by actor_id.  For each
    stage it picks the corresponding actor, generates a segment, evaluates
    exit conditions, and advances to the next stage.

    KV-cache state is threaded through when the handoff mode is KV_CACHE —
    only consecutive stages using the *same* model can do this efficiently.

    Routing:
    - Per-condition ``target_stage_index`` (YAML) becomes ``SwitchDecision.routing_stage_index``
      and is applied before ``loop_back_stage_index`` / ``fallback_stage_index``.
    - ``natural_next_stage_index`` on a stage is used when the actor stops without any
      exit condition firing (instead of ``stage_idx + 1``). Use with cyclic pipelines;
      For cyclic pipelines with staged GPU waves, set ``staged_cyclic_loop_stage_indices``
      in the experiment config (see runner).
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        actors: dict[str, ActorBase],
        max_total_tokens: int = 8192,
        max_loop_tokens: Optional[int] = None,
        loop_actor_ids: Optional[frozenset[str]] = None,
        verbose: bool = False,
    ) -> None:
        self.stages = stages
        self.actors = actors
        self.max_total_tokens = max_total_tokens
        self.max_loop_tokens = max_loop_tokens
        self.loop_actor_ids = loop_actor_ids or frozenset()
        self.verbose = verbose

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        example_id: str,
        prompt: str,
        *,
        stop_before_stage: Optional[int] = None,
        initial_trace: Optional[Trace] = None,
        start_stage_idx: Optional[int] = None,
    ) -> Trace:
        """
        Run the pipeline for one example.

        ``stop_before_stage``:
            If set, stop *before* executing any stage index >= this value.
            Example: ``stop_before_stage=1`` runs only config stage 0, then saves
            ``metadata[EXECUTOR_STATE_KEY]["stage_idx"]`` for the next wave.

            Works with ``full_prefill`` handoffs across processes (KV state is dropped).

        ``initial_trace`` / ``start_stage_idx``:
            Resume a trace from a previous partial run. When ``initial_trace`` is set,
            ``example_id`` and ``prompt`` are taken from that trace unless you only
            pass ``initial_trace`` (see :meth:`continue_run`).
        """
        return self._run_inner(
            example_id=example_id,
            prompt=prompt,
            stop_before_stage=stop_before_stage,
            initial_trace=initial_trace,
            start_stage_idx=start_stage_idx,
        )

    def continue_run(
        self,
        trace: Trace,
        *,
        stop_before_stage: Optional[int] = None,
        start_stage_idx: Optional[int] = None,
    ) -> Trace:
        """
        Resume from a trace produced by an earlier partial :meth:`run` / :meth:`continue_run`.

        ``start_stage_idx`` defaults to ``trace.metadata[EXECUTOR_STATE_KEY]["stage_idx"]``.
        """
        prev = trace.metadata.get(EXECUTOR_STATE_KEY, {})
        sid = start_stage_idx if start_stage_idx is not None else int(prev.get("stage_idx", 0))
        return self._run_inner(
            example_id=trace.example_id,
            prompt=trace.prompt,
            stop_before_stage=stop_before_stage,
            initial_trace=trace,
            start_stage_idx=sid,
        )

    def _run_inner(
        self,
        example_id: str,
        prompt: str,
        *,
        stop_before_stage: Optional[int] = None,
        initial_trace: Optional[Trace] = None,
        start_stage_idx: Optional[int] = None,
    ) -> Trace:
        if initial_trace is not None:
            trace = initial_trace
            example_id = trace.example_id
            prompt = trace.prompt
            stage_idx = int(start_stage_idx if start_stage_idx is not None else 0)
        else:
            trace = Trace(example_id=example_id, prompt=prompt)
            stage_idx = int(start_stage_idx if start_stage_idx is not None else 0)

        if stage_idx < 0 or stage_idx > len(self.stages):
            raise ValueError(f"Invalid start_stage_idx={stage_idx}; pipeline has {len(self.stages)} stages")

        kv_state: Optional[Any] = None
        # Stateful conditions (loop detector, …): only reset for stages we (re)enter.
        self._reset_conditions(from_stage_idx=stage_idx if initial_trace is not None else 0)

        partial_stop = False

        while stage_idx < len(self.stages):
            if stop_before_stage is not None and stage_idx >= stop_before_stage:
                partial_stop = True
                break

            stage = self.stages[stage_idx]
            actor = self._get_actor(stage.actor_id)

            for cond in stage.exit_conditions:
                cond.reset()

            if self.verbose:
                _executor_log(
                    f"[executor] stage={stage_idx} actor={stage.actor_id} "
                    f"handoff={stage.handoff_mode.value} "
                    f"total_tokens={trace.total_generated_tokens}"
                )

            segment, kv_state = actor.generate(
                trace=trace,
                handoff_mode=stage.handoff_mode,
                kv_state=kv_state,
                max_new_tokens=stage.max_new_tokens,
                stop_sequences=stage.stop_sequences or None,
                role=stage.role,
                prompt_suffix=stage.stage_prompt,
            )

            # Evaluate exit conditions
            decision = self._evaluate_conditions(stage, trace, segment)

            # Apply optional split inside the segment
            if decision.should_switch and decision.split_char_offset is not None:
                left, right = segment.split_at(decision.split_char_offset)
                trace.append_segment(left)
                # The right part will be re-generated by the next actor (full-prefill)
                # or discarded; for now we discard it and let the next actor continue.
            else:
                trace.append_segment(segment)

            if self._pipeline_token_limit_reached(trace):
                if self.verbose:
                    _executor_log("[executor] token limit reached, stopping")
                trace.metadata.pop(EXECUTOR_STATE_KEY, None)
                trace.finished_at = time.time()
                return trace

            # Determine next stage
            if decision.should_switch:
                next_idx = self._next_stage(stage, stage_idx, decision)
                if next_idx is None:
                    stage_idx = len(self.stages)
                    break
                # If handoff mode changes or actor changes, invalidate KV state
                if next_idx >= len(self.stages):
                    stage_idx = next_idx
                    break
                if next_idx != stage_idx + 1 or self.stages[next_idx].actor_id != stage.actor_id:
                    kv_state = None
                stage_idx = next_idx
            else:
                # Natural stop (model finished), move forward
                if stage.natural_next_stage_index is not None:
                    ni = stage.natural_next_stage_index
                    if ni < 0 or ni >= len(self.stages):
                        raise ValueError(
                            f"natural_next_stage_index={ni} out of range for "
                            f"{len(self.stages)} pipeline stage(s)"
                        )
                    stage_idx = ni
                else:
                    stage_idx += 1
                kv_state = None

            if stop_before_stage is not None and stage_idx >= stop_before_stage:
                partial_stop = True
                break

        if partial_stop:
            trace.metadata[EXECUTOR_STATE_KEY] = {
                "stage_idx": stage_idx,
                "partial": True,
            }
            trace.finished_at = None
        else:
            trace.metadata.pop(EXECUTOR_STATE_KEY, None)
            trace.finished_at = time.time()
        return trace

    def run_batch(
        self,
        examples: list[tuple[str, str]],
    ) -> list[Trace]:
        """Run the pipeline on a list of (example_id, prompt) pairs."""
        return [self.run(eid, prompt) for eid, prompt in examples]

    def consume_segment_after_generate(
        self,
        trace: Trace,
        segment: TraceSegment,
        *,
        start_stage_idx: int,
        stop_before_stage: Optional[int] = None,
    ) -> Trace:
        """
        Run exit-condition bookkeeping for ``segment``, produced externally (e.g. vLLM batch).

        Assumes staged execution semantics: callers advance exactly one logical stage/wave before
        ``stop_before_stage`` cuts off. Mirrors :meth:`_run_inner`'s iteration after ``generate``.
        Call :meth:`_reset_conditions` with the wave index before invoking this method per-example
        if any stage attaches stateful SwitchConditions shared across batches.
        """
        stage_idx = int(start_stage_idx)

        if stage_idx < 0 or stage_idx >= len(self.stages):
            raise ValueError(
                f"Invalid start_stage_idx={stage_idx}; pipeline has {len(self.stages)} stages"
            )

        partial_stop = False

        while stage_idx < len(self.stages):
            if stop_before_stage is not None and stage_idx >= stop_before_stage:
                partial_stop = True
                break

            stage = self.stages[stage_idx]

            for cond in stage.exit_conditions:
                cond.reset()

            decision = self._evaluate_conditions(stage, trace, segment)

            if decision.should_switch and decision.split_char_offset is not None:
                left, _right = segment.split_at(decision.split_char_offset)
                trace.append_segment(left)
            else:
                trace.append_segment(segment)

            if self._pipeline_token_limit_reached(trace):
                if self.verbose:
                    _executor_log("[executor] token limit reached, stopping")
                trace.metadata.pop(EXECUTOR_STATE_KEY, None)
                trace.finished_at = time.time()
                return trace

            if decision.should_switch:
                next_idx = self._next_stage(stage, stage_idx, decision)
                if next_idx is None:
                    stage_idx = len(self.stages)
                    break
                stage_idx = next_idx
            else:
                if stage.natural_next_stage_index is not None:
                    ni = stage.natural_next_stage_index
                    if ni < 0 or ni >= len(self.stages):
                        raise ValueError(
                            f"natural_next_stage_index={ni} out of range for "
                            f"{len(self.stages)} pipeline stage(s)"
                        )
                    stage_idx = ni
                else:
                    stage_idx += 1

            if stop_before_stage is not None:
                # Staged wave: one external generate. Any in-pipeline cursor (including
                # cyclic target_stage_index back to an earlier stage) must stay partial.
                partial_stop = stage_idx < len(self.stages)

            break  # один внешний generate на волну staged

        if partial_stop:
            trace.metadata[EXECUTOR_STATE_KEY] = {
                "stage_idx": stage_idx,
                "partial": True,
            }
            trace.finished_at = None
        else:
            trace.metadata.pop(EXECUTOR_STATE_KEY, None)
            trace.finished_at = time.time()

        return trace

    # ── helpers ───────────────────────────────────────────────────────────────

    def reset_switch_conditions_from(self, from_stage_idx: int = 0) -> None:
        """Clear stateful switch conditions from ``from_stage_idx`` onward (resume / batched staged)."""

        self._reset_conditions(from_stage_idx)

    def _pipeline_token_limit_reached(self, trace: Trace) -> bool:
        if trace.total_generated_tokens >= self.max_total_tokens:
            return True
        if self.max_loop_tokens is None or not self.loop_actor_ids:
            return False
        from quantlab.pipeline.staged_cyclic import trace_loop_generated_tokens

        return trace_loop_generated_tokens(trace, set(self.loop_actor_ids)) >= self.max_loop_tokens

    def _reset_conditions(self, from_stage_idx: int = 0) -> None:
        for i, stage in enumerate(self.stages):
            if i < from_stage_idx:
                continue
            for cond in stage.exit_conditions:
                cond.reset()

    def _get_actor(self, actor_id: str) -> ActorBase:
        if actor_id not in self.actors:
            raise KeyError(f"Actor {actor_id!r} not in pool. Available: {list(self.actors)}")
        actor = self.actors[actor_id]
        if not actor.is_loaded():
            actor.load()
        return actor

    def _evaluate_conditions(
        self,
        stage: PipelineStage,
        trace: Trace,
        segment: TraceSegment,
    ) -> SwitchDecision:
        # Temporarily append the segment so conditions can read the full trace
        trace.segments.append(segment)
        decision = SwitchDecision(should_switch=False)
        targets = stage.exit_condition_targets or []
        end_flags = stage.exit_condition_end_pipeline or []
        for i, cond in enumerate(stage.exit_conditions):
            d = cond.evaluate(trace, segment)
            if d.should_switch:
                if i < len(end_flags) and end_flags[i]:
                    ri = len(self.stages)
                else:
                    ri = targets[i] if i < len(targets) else None
                decision = replace(d, routing_stage_index=ri)
                break
        trace.segments.pop()
        return decision

    def _is_loop_trigger(self, stage: PipelineStage, decision: SwitchDecision) -> bool:
        return "loop" in decision.reason or "format" in decision.reason

    def _next_stage(
        self,
        stage: PipelineStage,
        current_idx: int,
        decision: SwitchDecision,
    ) -> Optional[int]:
        if decision.routing_stage_index is not None:
            ri = decision.routing_stage_index
            if ri == len(self.stages):
                return ri
            if ri < 0 or ri >= len(self.stages):
                raise ValueError(
                    f"target_stage_index / routing_stage_index={ri} out of range for "
                    f"{len(self.stages)} pipeline stage(s)"
                )
            return ri
        if self._is_loop_trigger(stage, decision) and stage.loop_back_stage_index is not None:
            return stage.loop_back_stage_index
        if stage.fallback_stage_index is not None:
            return stage.fallback_stage_index
        next_idx = current_idx + 1
        return next_idx if next_idx < len(self.stages) else None
