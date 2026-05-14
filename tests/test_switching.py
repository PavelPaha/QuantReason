from __future__ import annotations

import pytest

from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import SegmentRole
from quantlab.switching.rules import (
    AfterMarker,
    AfterNTokens,
    AfterRegex,
    AlwaysSwitch,
    NeverSwitch,
)
from quantlab.switching.triggers import (
    CandidateAnswerAppeared,
    LoopDetector,
    MaxTokensPerStage,
    ThinkBlockNotClosed,
)


def make_trace(generated: str, prompt: str = "") -> tuple[Trace, TraceSegment]:
    trace = Trace(example_id="x", prompt=prompt)
    words = generated.split()
    seg = TraceSegment(
        actor_id="a",
        text=generated,
        token_count=len(words),
        start_token_idx=0,
        role=SegmentRole.REASONING,
    )
    trace.append_segment(seg)
    return trace, seg


# ── rules ─────────────────────────────────────────────────────────────────────

def test_after_n_tokens_no_switch():
    cond = AfterNTokens(n=100)
    trace, seg = make_trace("short text")
    d = cond.evaluate(trace, seg)
    assert not d.should_switch


def test_after_n_tokens_switch():
    cond = AfterNTokens(n=3)
    trace, seg = make_trace("one two three four five")
    d = cond.evaluate(trace, seg)
    assert d.should_switch


def test_after_marker_not_present():
    cond = AfterMarker("</think>")
    trace, seg = make_trace("I am thinking ...")
    d = cond.evaluate(trace, seg)
    assert not d.should_switch


def test_after_marker_present_with_split():
    cond = AfterMarker("</think>", keep_marker=True)
    trace, seg = make_trace("I am thinking </think> Done.")
    d = cond.evaluate(trace, seg)
    assert d.should_switch
    assert d.split_char_offset == len("I am thinking </think>")


def test_after_regex_not_present():
    cond = AfterRegex(r"\bWait\b", keep_match=False)
    trace, seg = make_trace("I am thinking carefully.")
    d = cond.evaluate(trace, seg)
    assert not d.should_switch


def test_after_regex_present_without_match():
    cond = AfterRegex(r"\bWait\b[:,]?", keep_match=False)
    trace, seg = make_trace("First path. Wait, maybe there is another way.")
    d = cond.evaluate(trace, seg)
    assert d.should_switch
    assert d.split_char_offset == len("First path. ")


def test_after_regex_present_with_match_and_ignore_case():
    cond = AfterRegex(r"\balternatively\b[:,]?", keep_match=True, ignore_case=True)
    trace, seg = make_trace("We can solve it. alternatively: compute directly.")
    d = cond.evaluate(trace, seg)
    assert d.should_switch
    assert d.split_char_offset == len("We can solve it. alternatively:")


def test_always_switch():
    cond = AlwaysSwitch()
    trace, seg = make_trace("anything")
    assert cond.evaluate(trace, seg).should_switch


def test_never_switch():
    cond = NeverSwitch()
    trace, seg = make_trace("anything")
    assert not cond.evaluate(trace, seg).should_switch


# ── triggers ──────────────────────────────────────────────────────────────────

def test_loop_detector_no_loop():
    cond = LoopDetector(ngram_size=3, max_repeats=3)
    trace, seg = make_trace("the quick brown fox jumps over the lazy dog")
    assert not cond.evaluate(trace, seg).should_switch


def test_loop_detector_detects_loop():
    cond = LoopDetector(ngram_size=3, max_repeats=2)
    repeating = "foo bar baz " * 3
    trace, seg = make_trace(repeating)
    d = cond.evaluate(trace, seg)
    assert d.should_switch
    assert "loop" in d.reason


def test_think_not_closed_ok():
    cond = ThinkBlockNotClosed(min_tokens=2)
    trace, seg = make_trace("<think> some thoughts </think> done")
    assert not cond.evaluate(trace, seg).should_switch


def test_think_not_closed_triggers():
    cond = ThinkBlockNotClosed(min_tokens=2)
    trace, seg = make_trace("<think> unclosed thoughts here more words")
    d = cond.evaluate(trace, seg)
    assert d.should_switch


def test_candidate_answer_appeared():
    cond = CandidateAnswerAppeared()
    trace, seg = make_trace("so the answer is 42")
    d = cond.evaluate(trace, seg)
    assert d.should_switch


def test_max_tokens_per_stage():
    cond = MaxTokensPerStage(max_tokens=5)
    trace, seg = make_trace("one two three four five six seven")
    d = cond.evaluate(trace, seg)
    assert d.should_switch


# ── registry ──────────────────────────────────────────────────────────────────

def test_registry_build():
    from quantlab.switching.registry import ConditionRegistry

    cond = ConditionRegistry.build("after_n_tokens", n=10)
    assert isinstance(cond, AfterNTokens)
    assert cond.n == 10

    cond = ConditionRegistry.build("after_regex", pattern=r"\bWait\b")
    assert isinstance(cond, AfterRegex)
    assert cond.pattern == r"\bWait\b"


def test_registry_unknown():
    from quantlab.switching.registry import ConditionRegistry

    with pytest.raises(KeyError):
        ConditionRegistry.build("nonexistent_condition")
