from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from quantlab.core.types import GenerationParams, TimingInfo


class BackendBase(ABC):
    """
    Low-level generation backend.  One backend instance corresponds to one
    loaded model.  Backends do not know about Trace or pipelines — they take
    a plain string prompt and return generated text.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        params: GenerationParams,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, int, TimingInfo]:
        """
        Run greedy / sampling generation.

        Returns:
            (generated_text, token_count, timing)
        """
        ...

    def generate_with_kv(
        self,
        prompt: str,
        params: GenerationParams,
        past_key_values: Any,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, int, TimingInfo, Any]:
        """
        Continue generation from an existing KV cache.

        Returns:
            (generated_text, token_count, timing, new_past_key_values)

        Default implementation raises NotImplementedError.  Only transformers
        backend (and compatible custom backends) support this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support KV-cache handoff"
        )

    def get_kv_cache(
        self,
        prompt: str,
    ) -> tuple[Any, int]:
        """
        Run a prefill and return the resulting KV cache without decoding.

        Returns:
            (past_key_values, token_count)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support KV-cache export"
        )

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...
