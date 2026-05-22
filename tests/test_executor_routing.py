from __future__ import annotations

from quantlab.actors.base import ActorBase, ActorConfig
from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import HandoffMode, SegmentRole
from quantlab.pipeline.executor import PipelineExecutor
from quantlab.pipeline.stage import PipelineStage
from quantlab.switching.registry import ConditionRegistry


class StubActor(ActorBase):
    """Returns a fixed string on every generate call."""

    def __init__(self, actor_id: str, text: str) -> None:
        super().__init__(
            ActorConfig(
                actor_id=actor_id,
                model_id="stub",
                backend="transformers",
            )
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


def test_per_condition_target_stage_index_cycles_until_max_tokens():
    always = ConditionRegistry.build("always")
    stages = [
        PipelineStage(
            actor_id="a",
            exit_conditions=[always],
            exit_condition_targets=[1],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
        PipelineStage(
            actor_id="b",
            exit_conditions=[always],
            exit_condition_targets=[0],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
    ]
    actors = {
        "a": StubActor("a", "A"),
        "b": StubActor("b", "B"),
    }
    ex = PipelineExecutor(stages=stages, actors=actors, max_total_tokens=6)
    tr = ex.run("e1", "P:")
    assert tr.generated_text == "ABABAB"
    assert len(tr.segments) == 6


def test_natural_next_stage_index_skips_linear_successor():
    never = ConditionRegistry.build("never")
    stages = [
        PipelineStage(
            actor_id="a",
            exit_conditions=[never],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
            natural_next_stage_index=2,
        ),
        PipelineStage(
            actor_id="b",
            exit_conditions=[never],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
        PipelineStage(
            actor_id="c",
            exit_conditions=[never],
            handoff_mode=HandoffMode.FULL_PREFILL,
            role=SegmentRole.UNKNOWN,
        ),
    ]
    actors = {
        "a": StubActor("a", "x"),
        "b": StubActor("b", "y"),
        "c": StubActor("c", "z"),
    }
    ex = PipelineExecutor(stages=stages, actors=actors)
    tr = ex.run("e2", "P:")
    assert tr.generated_text == "xz"
    assert len(tr.segments) == 2
