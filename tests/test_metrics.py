from __future__ import annotations

import pytest

from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import SegmentRole, TimingInfo
from quantlab.evaluation.judge import JudgementResult
from quantlab.metrics.accuracy import AccuracyMetric, ExactMatchMetric, ParseRateMetric
from quantlab.metrics.generation import (
    ActorTokenSplitMetric,
    CommitGapMetric,
    FinishCommitMetric,
    LoopDetectedMetric,
    LoopOnsetTokensMetric,
    ReasoningLengthMetric,
    StopTokenProbeMetric,
    ThinkClosedMetric,
    TokensToFirstCorrectMetric,
    VerificationSpiralMetric,
)
from quantlab.metrics.timing import SegmentTimingMetric, TotalGenerationTimeMetric


def make_trace_with_segments(texts: list[tuple[str, str]]) -> Trace:
    trace = Trace(example_id="m1", prompt="Q?")
    idx = 0
    for actor_id, text in texts:
        tokens = len(text.split())
        trace.append_segment(TraceSegment(
            actor_id=actor_id,
            text=text,
            token_count=tokens,
            start_token_idx=idx,
            timing=TimingInfo(total_ms=50.0 * tokens, token_count=tokens),
        ))
        idx += tokens
    return trace


def make_judgement(correct: bool = True, parsed: bool = True, pred: str = "42") -> JudgementResult:
    return JudgementResult(
        example_id="m1",
        predicted=pred if parsed else None,
        ground_truth="42",
        is_correct=correct,
        parse_success=parsed,
    )


# ── accuracy ──────────────────────────────────────────────────────────────────

def test_accuracy_correct():
    trace = make_trace_with_segments([("fp", "hello")])
    assert AccuracyMetric().compute(trace, make_judgement(True)) == 1


def test_accuracy_wrong():
    trace = make_trace_with_segments([("fp", "hello")])
    assert AccuracyMetric().compute(trace, make_judgement(False)) == 0


def test_parse_rate():
    trace = make_trace_with_segments([("fp", "hello")])
    assert ParseRateMetric().compute(trace, make_judgement(parsed=True)) == 1
    assert ParseRateMetric().compute(trace, make_judgement(parsed=False)) == 0


def test_exact_match():
    trace = make_trace_with_segments([("fp", "hello")])
    j = make_judgement(True, True, "42")
    assert ExactMatchMetric().compute(trace, j) == 1
    j2 = make_judgement(True, True, " 42 ")
    assert ExactMatchMetric().compute(trace, j2) == 1


# ── generation ────────────────────────────────────────────────────────────────

def test_reasoning_length():
    trace = make_trace_with_segments([("fp", "one two three"), ("q", "four five")])
    assert ReasoningLengthMetric().compute(trace, make_judgement()) == 5


def test_loop_detected_no_loop():
    trace = make_trace_with_segments([("fp", "the quick brown fox jumps over the lazy dog.")])
    assert LoopDetectedMetric(min_consecutive=3).compute(trace, make_judgement()) == 0


def test_loop_detected_consecutive_sentence_block():
    s = (
        "I will restate this idea. "
        "I will restate this idea. "
        "I will restate this idea. "
        "Then stop."
    )
    trace = make_trace_with_segments([("fp", s)])
    assert LoopDetectedMetric(min_consecutive=3).compute(trace, make_judgement()) == 1


def test_loop_detected_global_repeat_without_streak():
    s = "Repeated argument meets minimum length criterion here."
    f = "Filler bridging material between repeated sentence units."
    parts: list[str] = []
    for i in range(10):
        parts.append(s)
        if i < 9:
            parts.append(f)
    body = " ".join(parts)
    trace = make_trace_with_segments([("fp", body)])
    m = LoopDetectedMetric(min_consecutive=999)
    assert m.compute(trace, make_judgement()) == 1


def test_loop_detected_global_repeat_skips_hesitation_sentence():
    w = (
        "Wait, let me reconsider this equation step carefully one more brief time "
        "before moving on entirely."
    )
    f = "Unrelated bridging filler between copies of hesitation style sentences."
    assert len(w) >= 15
    parts: list[str] = []
    for i in range(10):
        parts.append(w)
        if i < 9:
            parts.append(f)
    body = " ".join(parts)
    trace = make_trace_with_segments([("fp", body)])
    assert LoopDetectedMetric(min_consecutive=999).compute(trace, make_judgement()) == 0


def test_loop_detected_global_repeat_counts_hesitation_if_filter_off():
    w = (
        "Wait, let me reconsider this equation step carefully one more brief time "
        "before moving on entirely."
    )
    f = "Unrelated bridging filler between copies of hesitation style sentences."
    parts: list[str] = []
    for i in range(10):
        parts.append(w)
        if i < 9:
            parts.append(f)
    body = " ".join(parts)
    trace = make_trace_with_segments([("fp", body)])
    m = LoopDetectedMetric(
        min_consecutive=999,
        global_repeat_skip_sentence_patterns=(),
    )
    assert m.compute(trace, make_judgement()) == 1


def test_loop_detected_near_duplicate_not_adjacent_ok():
    a = (
        "The factors are listed here. Middle filler text. "
        "The factors are listed here. More stuff. "
        "The factors are listed here."
    )
    trace = make_trace_with_segments([("fp", a)])
    assert LoopDetectedMetric(min_consecutive=3).compute(trace, make_judgement()) == 0


def test_loop_detected_global_optional_off():
    s = "Same sentence repeats many times in this synthetic stress test."
    f = "We insert filler sentences so streak never completes here today."
    parts: list[str] = []
    for i in range(10):
        parts.append(s)
        if i < 9:
            parts.append(f)
    body = " ".join(parts)
    trace = make_trace_with_segments([("fp", body)])
    m = LoopDetectedMetric(min_consecutive=999, global_repeat_threshold=None)
    assert m.compute(trace, make_judgement()) == 0


def test_loop_onset_tokens_with_streak():
    sent = (
        "repeated derivation step without progressing anywhere useful at all yet. "
        "repeated derivation step without progressing anywhere useful at all yet. "
        "repeated derivation step without progressing anywhere useful at all yet. "
        "Then uniqueness."
    )
    trace = make_trace_with_segments([("fp", sent)])
    m_on = LoopOnsetTokensMetric(min_consecutive=3)
    det = LoopDetectedMetric(min_consecutive=3).compute(trace, make_judgement())
    onset = m_on.compute(trace, make_judgement())
    assert det == 1
    assert onset >= 0


def test_finish_commit_positive():
    text = 'We get 42. Thus \\boxed{42}'
    trace = make_trace_with_segments([("fp", text)])
    j = JudgementResult(
        example_id="m1",
        predicted="42",
        ground_truth="42",
        is_correct=True,
        parse_success=True,
    )
    assert FinishCommitMetric().compute(trace, j) == 1


def test_finish_commit_negative_boxed_before_pred():
    text = '\\boxed{42} mentions 42 again'
    trace = make_trace_with_segments([("fp", text)])
    j = JudgementResult(
        example_id="m1",
        predicted="42",
        ground_truth="42",
        is_correct=True,
        parse_success=True,
    )
    assert FinishCommitMetric().compute(trace, j) == 0


def test_finish_commit_no_parse():
    trace = make_trace_with_segments([("fp", "no box")])
    assert FinishCommitMetric().compute(trace, make_judgement(parsed=False, pred=None)) == 0


def test_verification_spiral_counts_after_prediction():
    text = (
        'The answer is forty two.\nForty-two is even. Wait, let me check again: forty two.\nHmm.'
    )
    trace = make_trace_with_segments([("fp", text)])
    j = JudgementResult(
        example_id="m1",
        predicted="forty two",
        ground_truth="42",
        is_correct=False,
        parse_success=True,
    )
    n = VerificationSpiralMetric().compute(trace, j)
    assert n >= 2  # Wait + Hmm (+ optional Let me)


def test_stop_token_probe():
    trace = make_trace_with_segments([("fp", "x")])
    assert StopTokenProbeMetric().compute(trace, make_judgement()) == "no_logits_in_trace"


def test_think_closed_present():
    trace = make_trace_with_segments([("fp", "<think> stuff </think> answer")])
    assert ThinkClosedMetric().compute(trace, make_judgement()) == 1


def test_think_closed_absent():
    trace = make_trace_with_segments([("fp", "<think> unclosed stuff here")])
    assert ThinkClosedMetric().compute(trace, make_judgement()) == 0


def test_think_open_only_in_prompt_missing_close():
    trace = Trace(
        example_id="m1",
        prompt="assistant\n<think>\n",
        segments=[
            TraceSegment(
                actor_id="fp",
                text="truncated reasoning with no closure tag ",
                token_count=6,
                start_token_idx=0,
                timing=TimingInfo(total_ms=100.0, token_count=6),
            ),
        ],
    )
    assert ThinkClosedMetric().compute(trace, make_judgement()) == 0


def test_think_open_in_prompt_closed_in_generation():
    trace = Trace(
        example_id="m1",
        prompt="system\nassistant\n<think>\n",
        segments=[
            TraceSegment(
                actor_id="gptq",
                text="finish\n</think>\n\\boxed{1}",
                token_count=10,
                start_token_idx=0,
                timing=TimingInfo(total_ms=50.0, token_count=10),
            ),
        ],
    )
    assert ThinkClosedMetric().compute(trace, make_judgement()) == 1


def test_actor_token_split():
    trace = make_trace_with_segments([("fp", "a b c"), ("quant", "d e")])
    result = ActorTokenSplitMetric().compute(trace, make_judgement())
    assert result["fp"] == 3
    assert result["quant"] == 2


# ── timing ────────────────────────────────────────────────────────────────────

def test_total_generation_ms():
    trace = make_trace_with_segments([("fp", "a b"), ("q", "c d e")])
    # fp: 2 tokens × 50ms = 100ms, q: 3 tokens × 50ms = 150ms
    total = TotalGenerationTimeMetric().compute(trace, make_judgement())
    assert abs(total - 250.0) < 1.0


def test_segment_timing_ms():
    trace = make_trace_with_segments([("fp", "a b"), ("q", "c d e")])
    result = SegmentTimingMetric().compute(trace, make_judgement())
    assert abs(result["fp"] - 100.0) < 1.0
    assert abs(result["q"] - 150.0) < 1.0


# ── registry ──────────────────────────────────────────────────────────────────

def test_metric_registry():
    from quantlab.metrics.registry import MetricRegistry

    m = MetricRegistry.build("accuracy")
    assert isinstance(m, AccuracyMetric)


def test_metric_registry_unknown():
    from quantlab.metrics.registry import MetricRegistry

    with pytest.raises(KeyError):
        MetricRegistry.build("nonexistent_metric")
