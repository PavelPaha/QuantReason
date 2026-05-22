from __future__ import annotations

from quantlab.actors.base import ActorBase, ActorConfig
from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import HandoffMode, SegmentRole
from quantlab.pipeline.executor import EXECUTOR_STATE_KEY, PipelineExecutor
from quantlab.pipeline.staged_cyclic import (
    cyclic_stage_for_wave,
    example_runnable_at_stage,
    staged_wave_has_pending_work,
    trace_in_cyclic_loop,
    trace_loop_generated_tokens,
    trace_pending_stage_idx,
)
from quantlab.pipeline.stage import PipelineStage
from quantlab.switching.registry import ConditionRegistry


class StubActor(ActorBase):
    def __init__(self, actor_id: str, text: str) -> None:
        super().__init__(
            ActorConfig(actor_id=actor_id, model_id="stub", backend="transformers")
        )
        self._text = text

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def generate(self, trace, handoff_mode, kv_state=None, **kwargs):
        seg = TraceSegment(
            actor_id=self.actor_id,
            text=self._text,
            token_count=1,
            start_token_idx=trace.next_token_idx,
        )
        return seg, None


def _plan_answer_periodic_executor(max_total_tokens: int = 100) -> PipelineExecutor:
    always = ConditionRegistry.build("always")
    max_tok = ConditionRegistry.build("max_tokens_per_stage", max_tokens=2)
    stages = [
        PipelineStage(
            actor_id="plan",
            exit_conditions=[always],
            exit_condition_targets=[1],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
        PipelineStage(
            actor_id="answer",
            exit_conditions=[max_tok],
            exit_condition_targets=[2],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
        PipelineStage(
            actor_id="periodic",
            exit_conditions=[always],
            exit_condition_targets=[1],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
    ]
    actors = {
        "plan": StubActor("plan", "P"),
        "answer": StubActor("answer", "A"),
        "periodic": StubActor("periodic", "V"),
    }
    return PipelineExecutor(stages=stages, actors=actors, max_total_tokens=max_total_tokens)


def test_cyclic_stage_for_wave_alternates():
    loop = [1, 2]
    assert cyclic_stage_for_wave(0, plan_stage_index=0, loop_stage_indices=loop) == 0
    assert cyclic_stage_for_wave(1, plan_stage_index=0, loop_stage_indices=loop) == 1
    assert cyclic_stage_for_wave(2, plan_stage_index=0, loop_stage_indices=loop) == 2
    assert cyclic_stage_for_wave(3, plan_stage_index=0, loop_stage_indices=loop) == 1


def test_cyclic_stage_for_wave_plan_preface_then_loop():
    """Plan → GPTQ bootstrap (preface) → BF16 ↔ GPTQ loop."""
    assert (
        cyclic_stage_for_wave(
            0,
            plan_stage_index=0,
            preface_stage_indices=[1],
            loop_stage_indices=[2, 3],
        )
        == 0
    )
    assert (
        cyclic_stage_for_wave(
            1,
            plan_stage_index=0,
            preface_stage_indices=[1],
            loop_stage_indices=[2, 3],
        )
        == 1
    )
    assert (
        cyclic_stage_for_wave(
            2,
            plan_stage_index=0,
            preface_stage_indices=[1],
            loop_stage_indices=[2, 3],
        )
        == 2
    )
    assert (
        cyclic_stage_for_wave(
            3,
            plan_stage_index=0,
            preface_stage_indices=[1],
            loop_stage_indices=[2, 3],
        )
        == 3
    )
    assert (
        cyclic_stage_for_wave(
            4,
            plan_stage_index=0,
            preface_stage_indices=[1],
            loop_stage_indices=[2, 3],
        )
        == 2
    )


def test_staged_waves_match_nonstaged_for_plan_answer_periodic_loop():
    ex = _plan_answer_periodic_executor(max_total_tokens=8)
    full = ex.run("e1", "prompt:")

    staged = _plan_answer_periodic_executor(max_total_tokens=8)
    tr = staged.run("e1", "prompt:", stop_before_stage=1)
    assert trace_pending_stage_idx(tr) == 1

    loop_actors = {"answer", "periodic"}
    wave = 1
    while trace_in_cyclic_loop(
        tr,
        loop_stage_indices=[1, 2],
        loop_actor_ids=loop_actors,
        max_total_tokens=8,
    ):
        tr = staged.continue_run(
            tr,
            stop_before_stage=cyclic_stage_for_wave(wave, plan_stage_index=0, loop_stage_indices=[1, 2]) + 1,
        )
        wave += 1

    assert full.generated_text == tr.generated_text
    assert len(full.segments) == len(tr.segments)


def test_consume_segment_cyclic_routing_stays_partial():
    """Batched staged waves use consume_segment; periodic→answer must not set finished_at."""
    ex = _plan_answer_periodic_executor(max_total_tokens=100)
    tr = ex.run("e1", "prompt:", stop_before_stage=1)
    tr = ex.continue_run(tr, stop_before_stage=2)
    seg = TraceSegment(
        actor_id="periodic",
        text="V",
        token_count=1,
        start_token_idx=tr.next_token_idx,
    )
    ex.reset_switch_conditions_from(2)
    tr = ex.consume_segment_after_generate(
        tr, seg, start_stage_idx=2, stop_before_stage=3
    )
    assert tr.finished_at is None
    assert trace_pending_stage_idx(tr) == 1
    assert trace_in_cyclic_loop(
        tr,
        loop_stage_indices=[1, 2],
        loop_actor_ids={"answer", "periodic"},
        max_total_tokens=100,
    )


def test_pipeline_max_total_tokens_stops_executor():
    ex = _plan_answer_periodic_executor(max_total_tokens=3)
    tr = ex.run("e1", "prompt:")
    assert tr.total_generated_tokens <= 3
    assert EXECUTOR_STATE_KEY not in tr.metadata or tr.finished_at is not None


def test_pipeline_max_loop_tokens_excludes_plan():
    loop_actors = frozenset({"answer", "periodic"})
    ex = PipelineExecutor(
        stages=_plan_answer_periodic_executor(max_total_tokens=100).stages,
        actors={
            "plan": StubActor("plan", "P"),
            "answer": StubActor("answer", "A"),
            "periodic": StubActor("periodic", "V"),
        },
        max_total_tokens=100,
        max_loop_tokens=2,
        loop_actor_ids=loop_actors,
    )
    tr = ex.run("e1", "prompt:")
    assert trace_loop_generated_tokens(tr, set(loop_actors)) <= 2
    assert tr.total_generated_tokens > 2  # plan segment counts separately


def test_staged_wave_pending_work_linear_two_stage():
    ex = _plan_answer_periodic_executor()
    done = ex.run("done", "p:", stop_before_stage=1)
    pending = ex.run("pending", "p:", stop_before_stage=1)
    traces = {"done": done, "pending": pending}
    examples = [type("E", (), {"example_id": x})() for x in ("done", "pending")]

    assert trace_pending_stage_idx(done) == 1
    assert example_runnable_at_stage(
        "done", 1, traces, set(), plan_stage_index=0, use_staged_cyclic=False
    )
    assert example_runnable_at_stage(
        "pending", 1, traces, set(), plan_stage_index=0, use_staged_cyclic=False
    )

    finished = ex.continue_run(done, stop_before_stage=None)
    traces["done"] = finished
    assert not example_runnable_at_stage(
        "done", 1, traces, set(), plan_stage_index=0, use_staged_cyclic=False
    )
    assert staged_wave_has_pending_work(
        traces,
        examples,
        failed=set(),
        skip=set(),
        current_stage=1,
        plan_stage_index=0,
        use_staged_cyclic=False,
    )


def test_end_pipeline_on_condition():
    boxed = ConditionRegistry.build("candidate_answer_appeared")
    stages = [
        PipelineStage(
            actor_id="answer",
            exit_conditions=[boxed],
            exit_condition_end_pipeline=[True],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
    ]
    actors = {"answer": StubActor("answer", "\\boxed{42}")}
    ex = PipelineExecutor(stages=stages, actors=actors)
    tr = ex.run("e1", "prompt:")
    assert tr.finished_at is not None
    assert len(tr.segments) == 1
