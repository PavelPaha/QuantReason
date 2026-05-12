from __future__ import annotations

import pytest

from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import SegmentRole, TimingInfo


def make_segment(actor_id: str, text: str, start: int = 0) -> TraceSegment:
    return TraceSegment(
        actor_id=actor_id,
        text=text,
        token_count=len(text.split()),
        start_token_idx=start,
        role=SegmentRole.REASONING,
    )


def test_trace_text_assembly():
    trace = Trace(example_id="t1", prompt="Q: What is 2+2?\n")
    trace.append_segment(make_segment("fp", "Let me think. ", 0))
    trace.append_segment(make_segment("quant", "The answer is 4. #### 4", 3))

    assert "Let me think." in trace.generated_text
    assert "#### 4" in trace.generated_text
    assert trace.full_text.startswith("Q: What is 2+2?")


def test_trace_total_tokens():
    trace = Trace(example_id="t2", prompt="")
    trace.append_segment(make_segment("a", "one two three", 0))
    trace.append_segment(make_segment("b", "four five", 3))
    assert trace.total_generated_tokens == 5


def test_segment_split():
    seg = TraceSegment(
        actor_id="fp",
        text="Hello world, this is a test.",
        token_count=6,
        start_token_idx=0,
    )
    left, right = seg.split_at(12)
    assert left.text == "Hello world,"
    assert right.text == " this is a test."
    assert left.token_count + right.token_count == 6


def test_trace_serialization_roundtrip():
    trace = Trace(example_id="rt1", prompt="p")
    trace.append_segment(TraceSegment(
        actor_id="x",
        text="hello",
        token_count=1,
        start_token_idx=0,
        timing=TimingInfo(total_ms=50.0, token_count=1),
    ))
    d = trace.to_dict()
    restored = Trace.from_dict(d)
    assert restored.example_id == trace.example_id
    assert restored.generated_text == trace.generated_text
    assert restored.segments[0].timing.total_ms == 50.0


def test_get_segments_by_actor():
    trace = Trace(example_id="t3", prompt="")
    trace.append_segment(make_segment("fp", "a b"))
    trace.append_segment(make_segment("quant", "c d"))
    trace.append_segment(make_segment("fp", "e f"))

    fp_segs = trace.get_segments_by_actor("fp")
    assert len(fp_segs) == 2
    quant_segs = trace.get_segments_by_actor("quant")
    assert len(quant_segs) == 1
