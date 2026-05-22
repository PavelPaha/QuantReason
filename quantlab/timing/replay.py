from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from quantlab.core.trace import Trace, TraceSegment
from quantlab.core.types import HandoffMode, TimingInfo


def _vllm_dtype_arg(precision_mode: str) -> str:
    """Map QuantLab-style names to vLLM ``LLM(dtype=...)`` literals."""
    key = precision_mode.strip().lower()
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        "float16": "float16",
        "float32": "float32",
        "bfloat16": "bfloat16",
        "auto": "auto",
        "half": "half",
        "float": "float",
    }.get(key, key)


def vllm_quantization_arg(quantization: Optional[str]) -> Optional[str]:
    """Map YAML/config quantization labels to optional vLLM ``quantization``."""
    if quantization is None:
        return None
    q = quantization.strip().lower()
    if q in ("", "none", "null"):
        return None
    return quantization.strip()


@dataclass(frozen=True)
class ReplayBackendSpec:
    """Enough state to instantiate vLLM for one actor."""

    model_id: str
    precision_mode: str
    quantization: Optional[str]
    tensor_parallel_size: int
    gpu_memory_utilization: float
    cuda_visible_devices: Optional[str] = None


def replay_generation_token_budget(seg: TraceSegment) -> int:
    """
    How many tokens the replay should decode to mimic the original run.

    Prefer ``timing.token_count`` (actual tokens emitted in that actor call).
    Fallback: ``segment.token_count`` — length of committed text kept in trace (may
    be much smaller after handoff truncation, yielding misleadingly fast replays).
    """
    if seg.timing is not None and getattr(seg.timing, "token_count", 0):
        return int(seg.timing.token_count)
    return int(seg.token_count)


def replay_backend_spec_from_actor(actor: Mapping[str, Any]) -> ReplayBackendSpec:
    """
    Build a replay engine spec from a serialized actor block (YAML / ``config.json``).
    Uses ``backend_kwargs`` for CUDA / parallelism / KV budget.
    """
    bk = actor.get("backend_kwargs") or {}
    raw_q = actor.get("quantization", "none")
    q_raw = raw_q.strip() if isinstance(raw_q, str) else ("none" if raw_q is None else str(raw_q))

    cuda = bk.get("cuda_visible_devices")
    if cuda is not None:
        cuda = str(cuda).strip() or None

    return ReplayBackendSpec(
        model_id=actor["model_id"],
        precision_mode=str(actor.get("precision", "bf16")),
        quantization=vllm_quantization_arg(q_raw),
        tensor_parallel_size=int(bk.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(bk.get("gpu_memory_utilization", 0.9)),
        cuda_visible_devices=cuda,
    )


@dataclass
class ReplayResult:
    segment_label: str
    actor_id: str
    token_count: int
    timing: TimingInfo
    prompt_len_chars: int


class VLLMTimingReplay:
    """
    Re-runs trace segments through vLLM to obtain comparable latency measurements.

    Even when the main experiment uses transformers for KV-cache handoff, you
    can use this class to replay each segment independently in vLLM for fair
    latency comparisons.

    Each segment is replayed as a standalone full-prefill request — the segment's
    prefix (prompt + preceding segments) is used as the input, and generation is
    capped at ``replay_generation_token_budget(segment)``: prefer the original
    run's timed token count when present (see traces), else the segment length.

    When ``actor_backend_specs`` is set, ``segment.actor_id`` selects weights,
    quantization, precision, and per-actor ``backend_kwargs`` from the experiment
    config. Otherwise a single backend (``model_id``, … constructor args) is used
    for every segment (ablations / comparisons).
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        precision_mode: str = "bf16",
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        cuda_visible_devices: Optional[str] = None,
        *,
        actor_backend_specs: Optional[dict[str, ReplayBackendSpec]] = None,
    ) -> None:
        self.actor_backend_specs = actor_backend_specs
        if actor_backend_specs is None:
            if not model_id:
                raise ValueError("model_id is required unless actor_backend_specs is provided")
            self._fixed_spec = ReplayBackendSpec(
                model_id=model_id,
                precision_mode=precision_mode,
                quantization=vllm_quantization_arg(quantization),
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                cuda_visible_devices=cuda_visible_devices,
            )
        else:
            self._fixed_spec = None
            if model_id is not None:
                raise ValueError("pass either model_id (fixed replay) or actor_backend_specs, not both")
        self._llm: Any = None
        self._loaded_spec: ReplayBackendSpec | None = None

    def _ensure_loaded(self, spec: ReplayBackendSpec) -> None:
        if self._llm is not None and self._loaded_spec == spec:
            return
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("vllm is not installed.") from e

        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded_spec = None
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

        if spec.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.cuda_visible_devices).strip()

        from quantlab.actors.backends.vllm_backend import _prepend_executable_dir_to_path

        _prepend_executable_dir_to_path()

        self._llm = LLM(
            model=spec.model_id,
            dtype=_vllm_dtype_arg(spec.precision_mode),
            quantization=spec.quantization,
            tensor_parallel_size=spec.tensor_parallel_size,
            gpu_memory_utilization=spec.gpu_memory_utilization,
        )
        self._loaded_spec = spec

    def replay_trace(
        self,
        trace: Trace,
        segments_to_replay: Optional[list[int]] = None,
    ) -> list[ReplayResult]:
        """
        Replay each segment of the trace independently.

        Args:
            trace: a completed trace
            segments_to_replay: indices of segments to replay; None = all

        Returns:
            list of ReplayResult, one per replayed segment
        """
        from vllm import SamplingParams

        indices = segments_to_replay or list(range(len(trace.segments)))
        results: list[ReplayResult] = []

        for seg_idx in indices:
            seg = trace.segments[seg_idx]
            if self.actor_backend_specs is not None:
                spec = self.actor_backend_specs.get(seg.actor_id)
                if spec is None:
                    known = ", ".join(sorted(self.actor_backend_specs))
                    raise KeyError(
                        f"Trace segment refers to actor_id={seg.actor_id!r}, "
                        f"which is absent from replay actor specs. Known: [{known}]"
                    )
                self._ensure_loaded(spec)
            else:
                assert self._fixed_spec is not None
                self._ensure_loaded(self._fixed_spec)

            handoff_raw = seg.metadata.get("handoff_mode", HandoffMode.FULL_PREFILL.value)
            handoff_mode = HandoffMode(handoff_raw)
            plan_label = seg.metadata.get("handoff_plan_label", "")
            prefix = trace.handoff_prefix(
                handoff_mode,
                plan_label=plan_label,
                segments=trace.segments[:seg_idx],
            ) + seg.stage_prompt_sent
            max_new = replay_generation_token_budget(seg)
            params = SamplingParams(
                temperature=0.0,
                max_tokens=max_new,
            )

            import time

            t0 = time.perf_counter()
            outputs = self._llm.generate([prefix], sampling_params=params)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            out = outputs[0].outputs[0]
            token_count = len(out.token_ids)

            timing = TimingInfo(
                generation_ms=elapsed_ms,
                total_ms=elapsed_ms,
                tokens_per_second=token_count / max(elapsed_ms / 1000, 1e-9),
                token_count=token_count,
            )
            results.append(
                ReplayResult(
                    segment_label=f"seg_{seg_idx}",
                    actor_id=seg.actor_id,
                    token_count=token_count,
                    timing=timing,
                    prompt_len_chars=len(prefix),
                )
            )

        return results

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded_spec = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
