from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Any, Optional

from quantlab.actors.backends.base import BackendBase
from quantlab.core.types import GenerationParams, PrecisionMode, QuantizationMethod, TimingInfo

if TYPE_CHECKING:
    pass


def _prepend_executable_dir_to_path() -> None:
    """Ensure conda/virtualenv ``bin`` (ninja, etc.) is visible to vLLM worker subprocesses."""
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    if not bindir:
        return
    sep = os.pathsep
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(sep) if p]
    if bindir not in parts:
        os.environ["PATH"] = bindir + (sep + current if current else "")


_QUANT_MAP: dict[QuantizationMethod, str] = {
    QuantizationMethod.GPTQ: "gptq",
    QuantizationMethod.AWQ: "awq",
    QuantizationMethod.AQLM: "aqlm",
    QuantizationMethod.BITSANDBYTES: "bitsandbytes",
    QuantizationMethod.QUIP_SHARP: "quip#",
    QuantizationMethod.QTIP: "qtip",
    QuantizationMethod.YAQA: "gguf",  # placeholder — update when vLLM adds native YAQA
}

_DTYPE_MAP: dict[PrecisionMode, str] = {
    PrecisionMode.FP16: "float16",
    PrecisionMode.BF16: "bfloat16",
    PrecisionMode.FP32: "float32",
    PrecisionMode.FP8: "float8",
}


class VLLMBackend(BackendBase):
    """
    Inference via vLLM's offline LLM API.

    KV-cache handoff between *different* models is not supported by vLLM's
    public API; use TransformersBackend for that.  Within a single model,
    vLLM's prefix-caching handles repeated prefixes automatically.

    To pin separate models to separate GPUs inside one QuantLab process, set in
    ``backend_kwargs``: ``cuda_visible_devices: \"0\"`` (physical index, comma
    separated if tensor_parallel_size > 1).  This maps to ``CUDA_VISIBLE_DEVICES``
    immediately before ``LLM(...)`` construction—best-effort; if you see freezes
    or wrong placement with your vLLM version, use one GPU per Python process instead.

    With vLLM V1, ``disable_log_stats`` defaults to ``False`` so the engine
    records per-request prefill/decode intervals; we surface them as
    ``TimingInfo.prefill_ms`` and ``TimingInfo.decode_ms`` (see vLLM's
    ``FinishedRequestStats``: prefill = scheduled → first token, decode = first →
    last token). Set ``disable_log_stats=True`` in ``backend_kwargs`` to match
    upstream ``LLM`` defaults and avoid extra stat logging (phase fields stay
    ``None``).

    Most ``backend_kwargs`` are forwarded directly to ``vllm.LLM(...)``. For the
    reserved constructor fields ``dtype`` and ``quantization``, use
    ``backend_kwargs.dtype`` / ``backend_kwargs.quantization`` in the YAML; they
    override the enum-derived defaults below.
    """

    def __init__(
        self,
        model_id: str,
        precision: PrecisionMode = PrecisionMode.BF16,
        quantization: QuantizationMethod = QuantizationMethod.NONE,
        quantization_config: Optional[dict] = None,
        cuda_visible_devices: Optional[str] = None,
        dtype_override: Any | None = None,
        quantization_override: Any | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.precision = precision
        self.quantization = quantization
        self.quantization_config = quantization_config or {}
        self.cuda_visible_devices = cuda_visible_devices
        self.dtype_override = dtype_override
        self.quantization_override = quantization_override
        self.llm_kwargs = llm_kwargs
        self._llm = None
        self._vllm_finished_stats: list[Any] = []

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("vllm is not installed. Run: pip install vllm") from e

        _prepend_executable_dir_to_path()

        quant_arg = self.quantization_override
        if isinstance(quant_arg, str):
            quant_arg = quant_arg.strip() or None
        if quant_arg is None and self.quantization != QuantizationMethod.NONE:
            quant_arg = _QUANT_MAP.get(self.quantization)

        dtype = self.dtype_override
        if isinstance(dtype, str):
            dtype = dtype.strip() or None
        if dtype is None:
            dtype = _DTYPE_MAP.get(self.precision, "auto")

        if self.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices).strip()

        kwargs = dict(self.llm_kwargs)
        kwargs.setdefault("tensor_parallel_size", 1)
        kwargs.setdefault("gpu_memory_utilization", 0.90)
        # vLLM's LLM() forces True when absent; explicit False enables IterationStats
        # used for prefill/decode breakdown (V1).
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = False

        self._llm = LLM(
            model=self.model_id,
            dtype=dtype,
            quantization=quant_arg,
            **kwargs,
        )
        self._install_vllm_finished_stats_hook()

    def unload(self) -> None:
        del self._llm
        self._llm = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def is_loaded(self) -> bool:
        return self._llm is not None

    def _install_vllm_finished_stats_hook(self) -> None:
        self._vllm_finished_stats = []
        eng = getattr(self._llm, "llm_engine", None)
        lm = getattr(eng, "logger_manager", None) if eng is not None else None
        if lm is None:
            return
        if getattr(lm, "_quantlab_wrapped_record", False):
            return

        _orig = lm.record

        def _record(*args: Any, **kwargs: Any) -> None:
            iteration_stats = None
            if len(args) >= 2:
                iteration_stats = args[1]
            elif "iteration_stats" in kwargs:
                iteration_stats = kwargs["iteration_stats"]
            if iteration_stats is not None:
                frs = getattr(iteration_stats, "finished_requests", None) or []
                if frs:
                    self._vllm_finished_stats.extend(frs)
            return _orig(*args, **kwargs)

        lm.record = _record  # type: ignore[method-assign]
        lm._quantlab_wrapped_record = True  # type: ignore[attr-defined]

    @staticmethod
    def _timing_from_finished(
        fr: Any | None,
        token_count: int,
        wall_ms_fallback: float,
    ) -> TimingInfo:
        if fr is None:
            return TimingInfo(
                generation_ms=wall_ms_fallback,
                total_ms=wall_ms_fallback,
                tokens_per_second=token_count / max(wall_ms_fallback / 1000, 1e-9),
                token_count=token_count,
            )
        pre_s = float(fr.prefill_time)
        dec_s = float(fr.decode_time)
        tot_s = float(fr.e2e_latency)
        tot_ms = tot_s * 1000.0
        return TimingInfo(
            prefill_ms=pre_s * 1000.0,
            decode_ms=dec_s * 1000.0,
            generation_ms=tot_ms,
            total_ms=tot_ms,
            tokens_per_second=token_count / max(tot_ms / 1000.0, 1e-9),
            token_count=token_count,
        )

    def _assign_finished_stats(self, outputs: list[Any]) -> dict[str, Any]:
        pool = list(self._vllm_finished_stats)
        self._vllm_finished_stats.clear()
        by_id: dict[str, Any] = {}
        for ro in outputs:
            rid = ro.request_id
            n = len(ro.outputs[0].token_ids)
            idx = next(
                (i for i, fr in enumerate(pool) if fr.num_generation_tokens == n),
                None,
            )
            if idx is not None:
                by_id[rid] = pool.pop(idx)
            elif pool:
                by_id[rid] = pool.pop(0)
        return by_id

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        params: GenerationParams,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, int, TimingInfo]:
        if not self.is_loaded():
            self.load()

        from vllm import SamplingParams

        stops = list(params.stop_sequences) + (stop_sequences or [])
        sampling = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k > 0 else -1,
            repetition_penalty=params.repetition_penalty,
            max_tokens=params.max_new_tokens,
            stop=stops or None,
            seed=params.seed,
        )

        self._vllm_finished_stats.clear()
        t0 = time.perf_counter()
        outputs = self._llm.generate([prompt], sampling_params=sampling)
        wall_ms = (time.perf_counter() - t0) * 1000

        ro = outputs[0]
        output = ro.outputs[0]
        text = output.text
        token_count = len(output.token_ids)

        fr = self._assign_finished_stats(outputs).get(ro.request_id)
        timing = self._timing_from_finished(fr, token_count, wall_ms)
        return text, token_count, timing

    def generate_batch(
        self,
        prompts: list[str],
        params: GenerationParams,
        stop_sequences: Optional[list[str]] = None,
    ) -> list[tuple[str, int, TimingInfo]]:
        """Batch generation — more efficient for benchmarking."""
        if not self.is_loaded():
            self.load()

        from vllm import SamplingParams

        stops = list(params.stop_sequences) + (stop_sequences or [])
        sampling = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if params.top_k > 0 else -1,
            repetition_penalty=params.repetition_penalty,
            max_tokens=params.max_new_tokens,
            stop=stops or None,
            seed=params.seed,
        )

        self._vllm_finished_stats.clear()
        t0 = time.perf_counter()
        all_outputs = self._llm.generate(prompts, sampling_params=sampling)
        wall_ms = (time.perf_counter() - t0) * 1000
        per_ms = wall_ms / max(len(prompts), 1)

        stats_by = self._assign_finished_stats(all_outputs)
        results = []
        for out in all_outputs:
            o = out.outputs[0]
            tc = len(o.token_ids)
            fr = stats_by.get(out.request_id)
            results.append((
                o.text,
                tc,
                self._timing_from_finished(fr, tc, per_ms),
            ))
        return results
