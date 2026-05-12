from quantlab.core.trace import TraceSegment
from quantlab.core.types import SegmentRole, TimingInfo
from quantlab.timing.replay import (
    ReplayBackendSpec,
    replay_backend_spec_from_actor,
    replay_generation_token_budget,
    vllm_quantization_arg,
)


def sample_seg(text_tokens: int, timed_tokens: int | None):
    timing = (
        TimingInfo(
            generation_ms=1.0,
            total_ms=1.0,
            tokens_per_second=10.0,
            token_count=timed_tokens,
        )
        if timed_tokens is not None
        else None
    )
    return TraceSegment(
        actor_id="a",
        text="x" * max(1, text_tokens),
        token_count=text_tokens,
        start_token_idx=0,
        role=SegmentRole.UNKNOWN,
        timing=timing,
    )


def test_replay_token_budget_prefers_timing():
    assert replay_generation_token_budget(sample_seg(96, 600)) == 600


def test_replay_token_budget_falls_back_to_segment_tokens():
    s = TraceSegment(actor_id="a", text="hi", token_count=12, start_token_idx=0, role=SegmentRole.UNKNOWN)
    assert replay_generation_token_budget(s) == 12


def test_vllm_quantization_none():
    assert vllm_quantization_arg(None) is None
    assert vllm_quantization_arg("none") is None
    assert vllm_quantization_arg("  NONE  ") is None


def test_replay_spec_from_actor_gptq():
    actor = {
        "actor_id": "qa",
        "model_id": "org/model-gptq",
        "precision": "fp16",
        "quantization": "gptq",
        "backend_kwargs": {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.5, "cuda_visible_devices": "0"},
    }
    spec = replay_backend_spec_from_actor(actor)
    assert spec == ReplayBackendSpec(
        model_id="org/model-gptq",
        precision_mode="fp16",
        quantization="gptq",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.5,
        cuda_visible_devices="0",
    )


def test_replay_spec_quantization_normalized():
    actor = {
        "actor_id": "fp",
        "model_id": "org/model-fp",
        "precision": "bf16",
        "quantization": "none",
        "backend_kwargs": {},
    }
    spec = replay_backend_spec_from_actor(actor)
    assert spec.quantization is None
