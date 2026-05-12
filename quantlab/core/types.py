from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HandoffMode(str, Enum):
    FULL_PREFILL = "full_prefill"
    KV_CACHE = "kv_cache"


class PrecisionMode(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    NVFP4 = "nvfp4"
    INT8 = "int8"
    INT4 = "int4"


class QuantizationMethod(str, Enum):
    NONE = "none"
    GPTQ = "gptq"
    AQLM = "aqlm"
    QUIP_SHARP = "quip_sharp"
    QTIP = "qtip"
    YAQA = "yaqa"
    AWQ = "awq"
    BITSANDBYTES = "bitsandbytes"


class BackendType(str, Enum):
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


class SegmentRole(str, Enum):
    """Semantic role of a trace segment — for analysis only, not enforced by executor."""
    PREFILL = "prefill"
    PLAN = "plan"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    ANSWER = "answer"
    UNKNOWN = "unknown"


@dataclass
class GenerationParams:
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stop_sequences": self.stop_sequences,
            "seed": self.seed,
        }


@dataclass
class TimingInfo:
    #: Engine prefill phase (prompt → first output token), when available.
    prefill_ms: Optional[float] = None
    #: Decode phase (first → last new token), when available.
    decode_ms: Optional[float] = None
    #: Alias for total wall time kept for older readers; same as ``total_ms`` in practice.
    generation_ms: Optional[float] = None
    total_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
            "tokens_per_second": self.tokens_per_second,
            "token_count": self.token_count,
        }
