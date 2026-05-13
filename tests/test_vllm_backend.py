import sys
import types

from quantlab.actors.base import ActorConfig
from quantlab.actors.impls import VLLMActor
from quantlab.actors.backends.vllm_backend import VLLMBackend
from quantlab.core.types import PrecisionMode, QuantizationMethod


def _dummy_llm_engine():
    return types.SimpleNamespace(
        logger_manager=types.SimpleNamespace(record=lambda *args, **kwargs: None)
    )


def test_vllm_backend_uses_override_args_from_backend_kwargs(monkeypatch):
    call_kwargs = {}

    class DummyLLM:
        def __init__(self, **kwargs):
            call_kwargs.update(kwargs)
            self.llm_engine = _dummy_llm_engine()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=DummyLLM))

    backend = VLLMBackend(
        model_id="org/model",
        precision=PrecisionMode.BF16,
        quantization=QuantizationMethod.NONE,
        dtype_override="float16",
        quantization_override="fp8",
        gpu_memory_utilization=0.5,
    )

    backend.load()

    assert call_kwargs["model"] == "org/model"
    assert call_kwargs["dtype"] == "float16"
    assert call_kwargs["quantization"] == "fp8"
    assert call_kwargs["gpu_memory_utilization"] == 0.5


def test_vllm_backend_falls_back_to_enum_mapping(monkeypatch):
    call_kwargs = {}

    class DummyLLM:
        def __init__(self, **kwargs):
            call_kwargs.update(kwargs)
            self.llm_engine = _dummy_llm_engine()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=DummyLLM))

    backend = VLLMBackend(
        model_id="org/model-gptq",
        precision=PrecisionMode.BF16,
        quantization=QuantizationMethod.GPTQ,
    )

    backend.load()

    assert call_kwargs["dtype"] == "bfloat16"
    assert call_kwargs["quantization"] == "gptq"


def test_vllm_actor_routes_reserved_backend_kwargs(monkeypatch):
    init_kwargs = {}

    class DummyBackend:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)

        def is_loaded(self):
            return False

    monkeypatch.setattr("quantlab.actors.impls.VLLMBackend", DummyBackend)

    actor = VLLMActor(
        ActorConfig(
            actor_id="a",
            model_id="org/model",
            backend="vllm",
            precision=PrecisionMode.BF16,
            quantization=QuantizationMethod.NONE,
            backend_kwargs={
                "quantization": "fp8",
                "dtype": "float16",
                "gpu_memory_utilization": 0.5,
            },
        )
    )

    actor._ensure_backend()

    assert init_kwargs["quantization_override"] == "fp8"
    assert init_kwargs["dtype_override"] == "float16"
    assert init_kwargs["gpu_memory_utilization"] == 0.5


def test_vllm_finished_stats_hook_accepts_extra_kwargs():
    backend = VLLMBackend(model_id="org/model")
    backend._llm = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            logger_manager=types.SimpleNamespace(record=lambda *args, **kwargs: None)
        )
    )

    backend._install_vllm_finished_stats_hook()

    iteration_stats = types.SimpleNamespace(finished_requests=["req-1"])
    backend._llm.llm_engine.logger_manager.record(
        None,
        iteration_stats,
        None,
        mm_cache_stats={"hits": 1},
    )

    assert backend._vllm_finished_stats == ["req-1"]
