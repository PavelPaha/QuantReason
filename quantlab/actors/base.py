from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import (
    GenerationParams,
    HandoffMode,
    PrecisionMode,
    QuantizationMethod,
    SegmentRole,
)


@dataclass
class ActorConfig:
    actor_id: str
    model_id: str
    backend: str = "transformers"
    precision: PrecisionMode = PrecisionMode.BF16
    quantization: QuantizationMethod = QuantizationMethod.NONE
    quantization_config: dict[str, Any] = field(default_factory=dict)
    generation_params: GenerationParams = field(default_factory=GenerationParams)
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    device_map: str = "auto"

    def to_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "model_id": self.model_id,
            "backend": self.backend,
            "precision": self.precision.value,
            "quantization": self.quantization.value,
            "quantization_config": self.quantization_config,
            "generation_params": self.generation_params.to_dict(),
            "backend_kwargs": self.backend_kwargs,
            "device_map": self.device_map,
        }


class ActorBase(ABC):
    """
    An actor wraps a (model + backend + precision) combination.

    The executor calls `generate()` to extend a Trace.  The actor receives the
    full Trace and is responsible for constructing the prompt it passes to its
    backend (either plain text for full-prefill, or via KV-cache handoff).
    """

    def __init__(self, config: ActorConfig) -> None:
        self.config = config
        self._loaded = False

    @property
    def actor_id(self) -> str:
        return self.config.actor_id

    @abstractmethod
    def generate(
        self,
        trace: Trace,
        handoff_mode: HandoffMode = HandoffMode.FULL_PREFILL,
        kv_state: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        role: SegmentRole = SegmentRole.UNKNOWN,
        prompt_suffix: str = "",
    ) -> tuple[TraceSegment, Optional[Any]]:
        """
        Continue the trace.

        Args:
            trace: the current reasoning trajectory (used to build the prompt prefix)
            handoff_mode: how to ingest the existing trace context
            kv_state: KV-cache state from a previous actor (KV_CACHE mode only)
            max_new_tokens: override for this call
            stop_sequences: additional stop strings for this call
            role: semantic label for the produced segment
            prompt_suffix: appended to trace.full_text before calling the backend;
                           not stored in the trace (stage instruction, e.g. "Plan:")

        Returns:
            (segment, new_kv_state)
            new_kv_state is None for FULL_PREFILL mode.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    def unload(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.actor_id!r}, "
            f"model={self.config.model_id!r}, "
            f"backend={self.config.backend!r})"
        )
