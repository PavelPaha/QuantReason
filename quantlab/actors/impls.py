from __future__ import annotations

import time
from typing import Any, Optional

from quantlab.actors.base import ActorBase, ActorConfig
from quantlab.actors.backends.transformers_backend import TransformersBackend
from quantlab.actors.backends.vllm_backend import VLLMBackend
from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import HandoffMode, SegmentRole, StagePromptPlacement, TimingInfo


class VLLMActor(ActorBase):
    """Actor backed by vLLM.  Supports FULL_PREFILL handoff only."""

    def __init__(self, config: ActorConfig) -> None:
        super().__init__(config)
        self._backend: Optional[VLLMBackend] = None

    def _ensure_backend(self) -> VLLMBackend:
        if self._backend is None:
            bk = dict(self.config.backend_kwargs)
            cuda_vis = bk.pop("cuda_visible_devices", None)
            vllm_quantization = bk.pop("quantization", None)
            vllm_dtype = bk.pop("dtype", None)
            self._backend = VLLMBackend(
                model_id=self.config.model_id,
                precision=self.config.precision,
                quantization=self.config.quantization,
                quantization_config=self.config.quantization_config,
                cuda_visible_devices=cuda_vis,
                dtype_override=vllm_dtype,
                quantization_override=vllm_quantization,
                **bk,
            )
        return self._backend

    def load(self) -> None:
        self._ensure_backend().load()
        self._loaded = True

    def unload(self) -> None:
        if self._backend and self._backend.is_loaded():
            self._backend.unload()
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._backend is not None and self._backend.is_loaded()

    def generate(
        self,
        trace: Trace,
        handoff_mode: HandoffMode = HandoffMode.FULL_PREFILL,
        kv_state: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        role: SegmentRole = SegmentRole.UNKNOWN,
        prompt_suffix: str = "",
        handoff_plan_label: str = "",
        stage_prompt_placement: StagePromptPlacement = StagePromptPlacement.ASSISTANT_SUFFIX,
        stage_system_prompt: str = "",
    ) -> tuple[TraceSegment, Optional[Any]]:
        if handoff_mode == HandoffMode.KV_CACHE:
            raise ValueError("VLLMActor does not support KV_CACHE handoff. Use TransformersActor.")

        backend = self._ensure_backend()
        params = self.config.generation_params
        if max_new_tokens is not None:
            import dataclasses
            params = dataclasses.replace(params, max_new_tokens=max_new_tokens)

        prompt = trace.build_llm_prompt(
            handoff_mode,
            stage_prompt=prompt_suffix,
            stage_prompt_placement=stage_prompt_placement,
            stage_system_prompt=stage_system_prompt,
            plan_label=handoff_plan_label,
        )
        text, token_count, timing = backend.generate(prompt, params, stop_sequences)

        segment = TraceSegment(
            actor_id=self.actor_id,
            text=text,
            token_count=token_count,
            start_token_idx=trace.next_token_idx,
            role=role,
            timing=timing,
        )
        return segment, None

    def generate_batch_segments(
        self,
        traces: list[Trace],
        *,
        handoff_mode: HandoffMode = HandoffMode.FULL_PREFILL,
        prompt_suffix: str,
        handoff_plan_label: str = "",
        stage_prompt_placement: StagePromptPlacement = StagePromptPlacement.ASSISTANT_SUFFIX,
        stage_system_prompt: str = "",
        max_new_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        role: SegmentRole = SegmentRole.UNKNOWN,
    ) -> list[TraceSegment]:
        """One vLLM batch call for FULL_PREFILL (same constraints as :meth:`generate`)."""
        if not traces:
            return []
        backend = self._ensure_backend()
        params = self.config.generation_params
        if max_new_tokens is not None:
            import dataclasses

            params = dataclasses.replace(params, max_new_tokens=max_new_tokens)

        prompts = [
            tr.build_llm_prompt(
                handoff_mode,
                stage_prompt=prompt_suffix,
                stage_prompt_placement=stage_prompt_placement,
                stage_system_prompt=stage_system_prompt,
                plan_label=handoff_plan_label,
            )
            for tr in traces
        ]
        rows = backend.generate_batch(prompts, params, stop_sequences)

        segments: list[TraceSegment] = []
        for trace, (text, token_count, timing) in zip(traces, rows, strict=True):
            segments.append(
                TraceSegment(
                    actor_id=self.actor_id,
                    text=text,
                    token_count=token_count,
                    start_token_idx=trace.next_token_idx,
                    role=role,
                    timing=timing,
                )
            )
        return segments


class TransformersActor(ActorBase):
    """Actor backed by HuggingFace transformers.  Supports both handoff modes."""

    def __init__(self, config: ActorConfig) -> None:
        super().__init__(config)
        self._backend: Optional[TransformersBackend] = None

    def _ensure_backend(self) -> TransformersBackend:
        if self._backend is None:
            self._backend = TransformersBackend(
                model_id=self.config.model_id,
                precision=self.config.precision,
                quantization=self.config.quantization,
                quantization_config=self.config.quantization_config,
                device_map=self.config.device_map,
                **self.config.backend_kwargs,
            )
        return self._backend

    def load(self) -> None:
        self._ensure_backend().load()
        self._loaded = True

    def unload(self) -> None:
        if self._backend and self._backend.is_loaded():
            self._backend.unload()
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._backend is not None and self._backend.is_loaded()

    def generate(
        self,
        trace: Trace,
        handoff_mode: HandoffMode = HandoffMode.FULL_PREFILL,
        kv_state: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        role: SegmentRole = SegmentRole.UNKNOWN,
        prompt_suffix: str = "",
        handoff_plan_label: str = "",
        stage_prompt_placement: StagePromptPlacement = StagePromptPlacement.ASSISTANT_SUFFIX,
        stage_system_prompt: str = "",
    ) -> tuple[TraceSegment, Optional[Any]]:
        import dataclasses

        backend = self._ensure_backend()
        params = self.config.generation_params
        if max_new_tokens is not None:
            params = dataclasses.replace(params, max_new_tokens=max_new_tokens)

        new_kv: Optional[Any] = None

        if handoff_mode == HandoffMode.KV_CACHE and kv_state is not None:
            last_seg = trace.last_segment()
            incremental_text = (last_seg.text if last_seg else "") + prompt_suffix
            text, token_count, timing, new_kv = backend.generate_with_kv(
                incremental_text, params, kv_state, stop_sequences
            )
        else:
            prompt = trace.build_llm_prompt(
                handoff_mode,
                stage_prompt=prompt_suffix,
                stage_prompt_placement=stage_prompt_placement,
                stage_system_prompt=stage_system_prompt,
                plan_label=handoff_plan_label,
            )
            text, token_count, timing = backend.generate(prompt, params, stop_sequences)

        segment = TraceSegment(
            actor_id=self.actor_id,
            text=text,
            token_count=token_count,
            start_token_idx=trace.next_token_idx,
            role=role,
            timing=timing,
        )
        return segment, new_kv
