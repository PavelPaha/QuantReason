from __future__ import annotations

from quantlab.actors.base import ActorBase, ActorConfig
from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import HandoffMode, SegmentRole, StagePromptPlacement
from quantlab.pipeline.executor import EXECUTOR_STATE_KEY, PipelineExecutor
from quantlab.pipeline.stage import PipelineStage


class RecordingStubActor(ActorBase):
    """Records the handoff prefix length seen on each generate call."""

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
        self.prefix_lens: list[int] = []

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def generate(self, trace, handoff_mode, kv_state=None, prompt_suffix="", handoff_plan_label="", stage_prompt_placement=StagePromptPlacement.ASSISTANT_SUFFIX, **kwargs):
        self.prefix_lens.append(
            len(
                trace.build_llm_prompt(
                    handoff_mode,
                    stage_prompt=prompt_suffix,
                    stage_prompt_placement=stage_prompt_placement,
                    plan_label=handoff_plan_label,
                )
            )
        )
        text = self.outputs[self._idx]
        self._idx += 1
        seg = TraceSegment(
            actor_id=self.actor_id,
            text=text,
            token_count=max(1, len(text.split())),
            start_token_idx=trace.next_token_idx,
        )
        return seg, None


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


def test_prompt_plan_labeled_handoff_hides_prior_think():
    prompt = "<|im_start|>user\nTASK<|im_end|>\n<|im_start|>assistant\n<think>\n"
    stages = [
        PipelineStage(
            actor_id="plan",
            exit_conditions=[],
            handoff_mode=HandoffMode.PROMPT_WITHOUT_THINK,
            stage_prompt="PLAN:",
            stage_prompt_placement=StagePromptPlacement.USER_SUFFIX,
            exclude_stage_prompt_from_trace=True,
            role=SegmentRole.PLAN,
        ),
        PipelineStage(
            actor_id="reason",
            exit_conditions=[],
            handoff_mode=HandoffMode.PROMPT_PLAN_LABELED,
            handoff_plan_label="Plan:\n",
            stage_prompt="SOLVE:\n<think>\n",
            role=SegmentRole.REASONING,
        ),
    ]
    plan = RecordingStubActor("plan", ["step one"])
    reason = RecordingStubActor("reason", ["answer"])
    ex = PipelineExecutor(stages=stages, actors={"plan": plan, "reason": reason})
    trace = ex.run("e1", prompt)

    expected = (
        "<|im_start|>user\nTASK<|im_end|>\n<|im_start|>assistant\n"
        "Plan:\nstep one\nSOLVE:\n<think>\n"
    )
    assert reason.prefix_lens == [len(expected)]
    assert expected.count("<think>") == 1


def test_segments_only_handoff_hides_initial_prompt():
    prompt = "TASK:\n"
    stages = [
        PipelineStage(
            actor_id="plan",
            exit_conditions=[],
            handoff_mode=HandoffMode.FULL_PREFILL,
            stage_prompt="PLAN:",
            exclude_stage_prompt_from_trace=True,
            role=SegmentRole.PLAN,
        ),
        PipelineStage(
            actor_id="reason",
            exit_conditions=[],
            handoff_mode=HandoffMode.SEGMENTS_ONLY,
            stage_prompt="SOLVE:",
            role=SegmentRole.REASONING,
        ),
    ]
    plan = RecordingStubActor("plan", ["<plan>"])
    reason = RecordingStubActor("reason", ["<solve>"])
    ex = PipelineExecutor(stages=stages, actors={"plan": plan, "reason": reason})
    trace = ex.run("e1", prompt)

    assert trace.generated_text == "<plan><solve>"
    assert plan.prefix_lens == [len(prompt + "PLAN:")]
    assert reason.prefix_lens == [len("<plan>SOLVE:")]
    assert trace.segments[0].stage_prompt_sent == ""
    assert trace.segments[0].metadata["handoff_mode"] == "full_prefill"
    assert trace.segments[1].metadata["handoff_mode"] == "segments_only"


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
