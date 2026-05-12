from __future__ import annotations

import time
from typing import Any, Optional

import torch

from quantlab.actors.backends.base import BackendBase
from quantlab.core.types import GenerationParams, PrecisionMode, QuantizationMethod, TimingInfo

_TORCH_DTYPE_MAP: dict[PrecisionMode, Any] = {
    PrecisionMode.FP32: torch.float32,
    PrecisionMode.FP16: torch.float16,
    PrecisionMode.BF16: torch.bfloat16,
}


def _build_bnb_config(quantization_config: dict) -> Any:
    from transformers import BitsAndBytesConfig

    bits = quantization_config.get("bits", 4)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization_config.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quantization_config.get("double_quant", True),
            bnb_4bit_compute_dtype=_TORCH_DTYPE_MAP.get(
                PrecisionMode(quantization_config.get("compute_dtype", "bf16")),
                torch.bfloat16,
            ),
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _build_gptq_config(quantization_config: dict) -> Any:
    from transformers import GPTQConfig

    return GPTQConfig(
        bits=quantization_config.get("bits", 4),
        disable_exllama=quantization_config.get("disable_exllama", False),
    )


class TransformersBackend(BackendBase):
    """
    Inference via HuggingFace transformers.

    Supports KV-cache handoff between actors that share the same architecture
    and vocabulary, making it the recommended backend for KV_CACHE handoff mode.
    """

    def __init__(
        self,
        model_id: str,
        precision: PrecisionMode = PrecisionMode.BF16,
        quantization: QuantizationMethod = QuantizationMethod.NONE,
        quantization_config: Optional[dict] = None,
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        extra_kwargs: Optional[dict] = None,
    ) -> None:
        self.model_id = model_id
        self.precision = precision
        self.quantization = quantization
        self.quantization_config = quantization_config or {}
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.extra_kwargs = extra_kwargs or {}
        self._model = None
        self._tokenizer = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_cfg = None
        if self.quantization == QuantizationMethod.BITSANDBYTES:
            quant_cfg = _build_bnb_config(self.quantization_config)
        elif self.quantization == QuantizationMethod.GPTQ:
            quant_cfg = _build_gptq_config(self.quantization_config)
        elif self.quantization == QuantizationMethod.AWQ:
            pass  # AWQ models load automatically when the correct model files are present

        dtype = _TORCH_DTYPE_MAP.get(self.precision, torch.bfloat16)
        load_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "torch_dtype": dtype,
            **self.extra_kwargs,
        }
        if quant_cfg is not None:
            load_kwargs["quantization_config"] = quant_cfg
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, **load_kwargs
        )
        self._model.eval()

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _device(self) -> torch.device:
        return next(self._model.parameters()).device

    def _tokenize(self, text: str) -> torch.Tensor:
        return self._tokenizer(text, return_tensors="pt").input_ids.to(self._device())

    def _decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _build_stop_ids(self, stop_sequences: list[str]) -> list[int]:
        stop_ids = []
        for s in stop_sequences:
            ids = self._tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[-1])
        return stop_ids

    # ── generation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        params: GenerationParams,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, int, TimingInfo]:
        if not self.is_loaded():
            self.load()

        input_ids = self._tokenize(prompt)
        stops = list(params.stop_sequences) + (stop_sequences or [])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.temperature > 0,
            "repetition_penalty": params.repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if params.temperature > 0:
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p
            if params.top_k > 0:
                gen_kwargs["top_k"] = params.top_k
        if params.seed is not None:
            torch.manual_seed(params.seed)

        t0 = time.perf_counter()
        output_ids = self._model.generate(input_ids, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        new_ids = output_ids[0, input_ids.shape[1]:]
        # Post-hoc stop-string truncation (simple single-token approach)
        if stops:
            text_full = self._decode(new_ids)
            for s in stops:
                idx = text_full.find(s)
                if idx != -1:
                    text_full = text_full[:idx]
            text = text_full
            token_count = len(new_ids)
        else:
            text = self._decode(new_ids)
            token_count = len(new_ids)

        timing = TimingInfo(
            generation_ms=elapsed_ms,
            total_ms=elapsed_ms,
            tokens_per_second=token_count / max(elapsed_ms / 1000, 1e-9),
            token_count=token_count,
        )
        return text, token_count, timing

    @torch.inference_mode()
    def generate_with_kv(
        self,
        prompt: str,
        params: GenerationParams,
        past_key_values: Any,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, int, TimingInfo, Any]:
        """Continue generation from an existing KV cache."""
        if not self.is_loaded():
            self.load()

        input_ids = self._tokenize(prompt)
        stops = list(params.stop_sequences) + (stop_sequences or [])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.temperature > 0,
            "repetition_penalty": params.repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "past_key_values": past_key_values,
            "return_dict_in_generate": True,
        }
        if params.temperature > 0:
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p
        if params.seed is not None:
            torch.manual_seed(params.seed)

        t0 = time.perf_counter()
        outputs = self._model.generate(input_ids, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        new_ids = outputs.sequences[0, input_ids.shape[1]:]
        text = self._decode(new_ids)
        token_count = len(new_ids)

        if stops:
            for s in stops:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]

        timing = TimingInfo(
            generation_ms=elapsed_ms,
            total_ms=elapsed_ms,
            tokens_per_second=token_count / max(elapsed_ms / 1000, 1e-9),
            token_count=token_count,
        )
        return text, token_count, timing, outputs.past_key_values

    @torch.inference_mode()
    def get_kv_cache(self, prompt: str) -> tuple[Any, int]:
        if not self.is_loaded():
            self.load()

        input_ids = self._tokenize(prompt)
        t0 = time.perf_counter()
        outputs = self._model(input_ids, use_cache=True, return_dict=True)
        _ = (time.perf_counter() - t0) * 1000
        return outputs.past_key_values, input_ids.shape[1]
