from __future__ import annotations

from quantlab.actors.base import ActorBase, ActorConfig
from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import HandoffMode, SegmentRole
from quantlab.pipeline.executor import EXECUTOR_STATE_KEY, PipelineExecutor
from quantlab.pipeline.stage import PipelineStage


class StubActor(ActorBase):
    """Returns predetermined text once per call (in order)."""

    def __init__(self, actor_id: str, outputs: list[str]) -> None:
        super().__init__(
            ActorConfig(
                actor_id=actor_id,
                model_id="stub",
                backend="transformers",
            )
        )
        self.outputs = outputs
        self._idx = 0

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def generate(self, trace, handoff_mode, kv_state=None, **kwargs):
        text = self.outputs[self._idx]
        self._idx += 1
        seg = TraceSegment(
            actor_id=self.actor_id,
            text=text,
            token_count=max(1, len(text.split())),
            start_token_idx=trace.next_token_idx,
        )
        return seg, None


def _two_stage_executor() -> PipelineExecutor:
    stages = [
        PipelineStage(
            actor_id="a",
            exit_conditions=[],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
        PipelineStage(
            actor_id="b",
            exit_conditions=[],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
    ]
    actors = {
        "a": StubActor("a", ["<A>"]),
        "b": StubActor("b", ["<B>"]),
    }
    return PipelineExecutor(stages=stages, actors=actors)


def test_stop_before_stage_is_resumable():
    ex = _two_stage_executor()
    t1 = ex.run("e1", "P:", stop_before_stage=1)
    assert len(t1.segments) == 1
    assert t1.segments[0].text == "<A>"
    assert t1.metadata[EXECUTOR_STATE_KEY]["stage_idx"] == 1
    assert t1.metadata[EXECUTOR_STATE_KEY]["partial"] is True

    t2 = ex.continue_run(t1, stop_before_stage=None)
    assert len(t2.segments) == 2
    assert t2.generated_text == "<A><B>"
    assert EXECUTOR_STATE_KEY not in t2.metadata
    assert t2.finished_at is not None


def test_consume_segment_after_generate_wave0_equals_run_stop_before():
    ex = _two_stage_executor()
    baseline = ex.run("e1", "P:", stop_before_stage=1)

    seg = TraceSegment(
        actor_id="a",
        text="<A>",
        token_count=1,
        start_token_idx=0,
        role=SegmentRole.UNKNOWN,
    )
    fresh = Trace("e1", "P:")
    ex2 = _two_stage_executor()
    ex2.reset_switch_conditions_from(0)
    via = ex2.consume_segment_after_generate(
        fresh,
        seg,
        start_stage_idx=0,
        stop_before_stage=1,
    )
    assert len(via.segments) == len(baseline.segments)
    assert via.generated_text == baseline.generated_text
    assert EXECUTOR_STATE_KEY in via.metadata
    assert via.metadata[EXECUTOR_STATE_KEY] == baseline.metadata[EXECUTOR_STATE_KEY]


def test_unstaged_equals_staged_for_full_prefill_pipeline():
    ex = _two_stage_executor()
    full = ex.run("e1", "P:")
    waves = _two_stage_executor()
    w1 = waves.run("e1", "P:", stop_before_stage=1)
    w2 = waves.continue_run(w1, stop_before_stage=None)
    assert full.generated_text == w2.generated_text
    assert len(full.segments) == len(w2.segments)
