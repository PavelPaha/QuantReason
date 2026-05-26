#!/usr/bin/env python3
"""Benchmark vLLM throughput for Qwen3 model variants.

Three batch modes (``--batch-mode``):

  bs1  — sequential single-request inference (``max_num_seqs=1``).
  bs16 — fixed micro-batches of 16 prompts per ``LLM.generate`` call.
  max  — all prompts submitted at once; vLLM scheduler decides concurrency
         (``max_num_seqs=256`` by default).

By default each model runs in a **separate Python subprocess** so the GPU never
holds two checkpoints at once. Use ``--no-isolated`` only for local debugging.

Examples::

  # One model, all three batch modes, 64 MATH-500 prompts:
  python scripts/bench_qwen_throughput.py \\
      --model Qwen/Qwen3-8B --batch-mode bs1,bs16,max --gpu 0

  # Full fp16 sweep from the model table:
  python scripts/bench_qwen_throughput.py --group fp16 --batch-mode bs1,bs16,max

  # Quick check (5 prompts, bs1 only):
  python scripts/bench_qwen_throughput.py --group fp16 --n-prompts 5 --batch-mode bs1

GSQ 2-bit MoE (``ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ``) uses vLLM ``humming`` quantization;
install ``pip install humming-kernels`` and see the model card for vLLM compatibility notes:
https://huggingface.co/ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ

B200 (Blackwell) setup — do **not** use ``requirements-vllm-cu124.txt`` (CUDA 12.4 is too old)::

  bash scripts/install_vllm_b200.sh --nightly   # Qwen3.5 needs recent vLLM
  python scripts/bench_qwen_throughput.py --list-models
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "VLLM_USE_V1" not in os.environ:
    os.environ["VLLM_USE_V1"] = "0"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")

BatchMode = Literal["bs1", "bs16", "max"]

_QUANT_TO_VLLM: dict[str, str | None] = {
    "none": None,
    "gptq": "gptq",
    "moe_wna16": "moe_wna16",
    "compressed-tensors": "compressed-tensors",
    "awq": "awq",
    "humming": "humming",
}

_DTYPE_MAP = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    group: str
    label: str
    quantization: str = "none"
    precision: str = "fp16"
    tensor_parallel_size: int = 1
    skip: bool = False
    skip_reason: str = ""
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)


# Model list from the Qwen throughput evaluation table.
MODEL_CATALOG: list[ModelSpec] = [
    # fp16
    ModelSpec("Qwen/Qwen3-8B", "fp16", "Qwen3-8B fp16"),
    ModelSpec("Qwen/Qwen3-14B", "fp16", "Qwen3-14B fp16"),
    ModelSpec("Qwen/Qwen3-32B", "fp16", "Qwen3-32B fp16"),
    ModelSpec("Qwen/Qwen3-30B-A3B", "bf16", "Qwen3-30B-A3B fp16"),
    ModelSpec("Qwen/Qwen3.6-35B-A3B", "bf16", "Qwen3.6-35B-A3B fp16"),
    ModelSpec(
        "Qwen/Qwen3.5-35B-A3B",
        "bf16",
        "Qwen3.5-35B-A3B fp16",
        extra_llm_kwargs={"language_model_only": True},
    ),
    # w4a16 (RedHat compressed-tensors)
    ModelSpec(
        "RedHatAI/Qwen3-8B-quantized.w4a16",
        "w4a16",
        "Qwen3-8B w4a16",
        quantization="compressed-tensors",
    ),
    ModelSpec(
        "RedHatAI/Qwen3-14B-quantized.w4a16",
        "w4a16",
        "Qwen3-14B w4a16",
        quantization="compressed-tensors",
    ),
    ModelSpec(
        "RedHatAI/Qwen3-32B-quantized.w4a16",
        "w4a16",
        "Qwen3-32B w4a16",
        quantization="compressed-tensors",
    ),
    # w2a16 (GPTQ 2-bit)
    ModelSpec(
        "kaitchup/Qwen3-8B-autoround-2bit-gptq",
        "w2a16",
        "Qwen3-8B w2a16",
        quantization="gptq",
    ),
    ModelSpec(
        "kaitchup/Qwen3-14B-autoround-2bit-gptq",
        "w2a16",
        "Qwen3-14B w2a16",
        quantization="gptq",
    ),
    ModelSpec(
        "kaitchup/Qwen3-32B-autoround-2bit-gptq",
        "w2a16",
        "Qwen3-32B w2a16",
        quantization="gptq",
    ),
    # NVFP4
    ModelSpec(
        "RedHatAI/Qwen3-8B-NVFP4",
        "nvfp4",
        "Qwen3-8B NVFP4",
        quantization="compressed-tensors",
    ),
    ModelSpec(
        "RedHatAI/Qwen3-14B-NVFP4",
        "nvfp4",
        "Qwen3-14B NVFP4",
        quantization="compressed-tensors",
    ),
    ModelSpec(
        "RedHatAI/Qwen3-32B-NVFP4",
        "nvfp4",
        "Qwen3-32B NVFP4",
        quantization="compressed-tensors",
    ),
    # Qwen3.6 35B variants
    ModelSpec(
        "ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ",
        "qwen36",
        "Qwen3.6-35B 2Bit GSQ",
        quantization="humming",
    ),
    ModelSpec(
        "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        "qwen36",
        "Qwen3.6-35B NVFP4",
        quantization="compressed-tensors",
        precision="bf16",
    ),
    # Official Qwen GPTQ Int4 MoE
    ModelSpec(
        "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        "gptq_int4",
        "Qwen3-30B-A3B GPTQ-Int4",
        quantization="gptq",
    ),
    ModelSpec(
        "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        "gptq_int4",
        "Qwen3.5-35B-A3B GPTQ-Int4",
        quantization="moe_wna16",
        extra_llm_kwargs={"language_model_only": True},
    ),
]

GROUPS = sorted({m.group for m in MODEL_CATALOG})


@dataclass
class RequestResult:
    prompt_idx: int
    output_tokens: int
    wall_ms: float
    tokens_per_second: float
    prompt_tokens: int = 0
    ttft_ms: float | None = None
    decode_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None


@dataclass
class BenchResult:
    model_id: str
    label: str
    group: str
    batch_mode: BatchMode
    batch_size: int
    max_num_seqs: int
    n_prompts: int
    max_new_tokens: int
    total_output_tokens: int
    wall_ms: float
    throughput_tokens_per_sec: float
    mean_per_request_tokens_per_sec: float
    load_ms: float
    cuda_visible_devices: str
    tensor_parallel_size: int
    precision: str
    quantization: str
    enforce_eager: bool
    warmup_requests: int
    timestamp_utc: str
    kv_cache_dtype: str = "auto"
    ttft_ms_mean: float | None = None
    ttft_ms_p50: float | None = None
    ttft_ms_p90: float | None = None
    decode_ms_mean: float | None = None
    tpot_ms_mean: float | None = None
    tpot_ms_p50: float | None = None
    tpot_ms_p90: float | None = None
    e2e_ms_mean: float | None = None
    decode_tokens_per_sec_mean: float | None = None
    prompt_tokens_mean: float | None = None
    output_tokens_mean: float | None = None
    peak_vram_mib: float | None = None
    baseline_vram_mib: float | None = None
    delta_vram_mib: float | None = None
    torch_peak_vram_mib: float | None = None
    n_layers: int | None = None
    n_kv_heads: int | None = None
    n_attention_heads: int | None = None
    head_dim: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    bytes_per_kv_token_fp16: float | None = None
    weight_dtype: str | None = None
    num_total_params: int | None = None
    num_active_params: int | None = None
    architecture: str | None = None
    per_request: list[RequestResult] = field(default_factory=list)
    error: str | None = None


def _reserve_loopback_master_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def _configure_tp_env(tp: int) -> None:
    if tp <= 1:
        return
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = _reserve_loopback_master_port()
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")


def _build_qwen_prompt(problem: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful math assistant.\n"
        "<|im_start|>user\n"
        "Solve this problem and give your final answer in \\boxed{}.\n\n"
        f"{problem}\n"
        "<|im_start|>assistant\n"
    )


def load_prompts(n_prompts: int, seed: int = 42) -> list[str]:
    from quantlab.benchmarks.registry import BenchmarkRegistry

    adapter = BenchmarkRegistry.build("math500")
    examples = adapter.load(split="test", max_examples=n_prompts, seed=seed)
    return [_build_qwen_prompt(ex.raw["problem"]) for ex in examples]


def _chunked(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _max_num_seqs_for_mode(mode: BatchMode, max_num_seqs_max: int) -> int:
    if mode == "bs1":
        return 1
    if mode == "bs16":
        return 16
    return max_num_seqs_max


def _chunk_size_for_mode(mode: BatchMode) -> int | None:
    """Fixed chunk size for batched modes; None means submit all prompts at once."""
    if mode == "bs1":
        return 1
    if mode == "bs16":
        return 16
    return None


def _effective_batch_size(mode: BatchMode) -> int:
    chunk = _chunk_size_for_mode(mode)
    return chunk if chunk is not None else 0  # 0 = unconstrained / scheduler decides


def _resolve_models(
    *,
    group: str | None,
    model_ids: list[str] | None,
) -> list[ModelSpec]:
    if model_ids:
        by_id = {m.model_id: m for m in MODEL_CATALOG}
        resolved: list[ModelSpec] = []
        for mid in model_ids:
            if mid in by_id:
                resolved.append(by_id[mid])
            else:
                resolved.append(ModelSpec(mid, "custom", mid))
        return resolved
    if group:
        return [m for m in MODEL_CATALOG if m.group == group]
    return list(MODEL_CATALOG)


def _shutdown_llm(llm: Any) -> None:
    """Release vLLM engine/workers before loading the next model."""
    if llm is None:
        return
    try:
        eng = getattr(llm, "llm_engine", None)
        if eng is not None:
            shutdown = getattr(eng, "shutdown", None)
            if callable(shutdown):
                shutdown()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _cuda_memory_used_mib(gpu: str) -> float | None:
    """Total used memory (MiB) on the listed physical GPU indices."""
    gpu = str(gpu).strip()
    if not gpu:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={gpu}",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return sum(float(line.strip()) for line in out.splitlines() if line.strip())
    except Exception:
        pass
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total = 0.0
        n = torch.cuda.device_count()
        for dev in gpu.split(","):
            idx = int(dev.strip())
            if idx < n:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                total += (total_b - free_b) / (1024 * 1024)
        return total or None
    except Exception:
        return None


def _wait_for_gpu_release(
    gpu: str,
    *,
    baseline_mib: float | None,
    timeout_sec: float = 180.0,
    poll_sec: float = 2.0,
    slack_mib: float = 2048.0,
) -> None:
    """Block until GPU memory drops near the pre-load baseline."""
    if baseline_mib is None:
        time.sleep(poll_sec)
        return
    deadline = time.monotonic() + timeout_sec
    target = baseline_mib + slack_mib
    while time.monotonic() < deadline:
        used = _cuda_memory_used_mib(gpu)
        if used is None or used <= target:
            return
        time.sleep(poll_sec)
    used = _cuda_memory_used_mib(gpu)
    print(
        f"  WARN: GPU {gpu} still holds {used:.0f} MiB after unload "
        f"(target <= {target:.0f} MiB); continuing anyway.",
        flush=True,
    )


def _make_llm(
    spec: ModelSpec,
    *,
    gpu: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    enforce_eager: bool,
    tensor_parallel_size: int | None,
    kv_cache_dtype: str = "auto",
    enable_prefix_caching: bool = True,
):
    from vllm import LLM

    tp = tensor_parallel_size if tensor_parallel_size is not None else spec.tensor_parallel_size
    _configure_tp_env(tp)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    quant = _QUANT_TO_VLLM.get(spec.quantization, spec.quantization)
    if quant == "none":
        quant = None

    kwargs: dict[str, Any] = {
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "enable_prefix_caching": enable_prefix_caching,
        "enforce_eager": enforce_eager,
        "max_num_seqs": max_num_seqs,
        "disable_log_stats": False,
    }
    if kv_cache_dtype and kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    kwargs.update(spec.extra_llm_kwargs)

    dtype = _DTYPE_MAP.get(spec.precision, "auto")
    t0 = time.perf_counter()
    llm = LLM(
        model=spec.model_id,
        dtype=dtype,
        quantization=quant,
        **kwargs,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0
    return llm, load_ms, tp


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _extract_arch_info(llm: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        mc = llm.llm_engine.model_config
        hf = getattr(mc, "hf_config", None) or getattr(mc, "hf_text_config", None)
        if hf is None:
            return info
        info["architecture"] = getattr(hf, "model_type", None) or (
            hf.architectures[0] if getattr(hf, "architectures", None) else None
        )
        info["n_layers"] = getattr(hf, "num_hidden_layers", None)
        info["n_attention_heads"] = getattr(hf, "num_attention_heads", None)
        info["n_kv_heads"] = getattr(hf, "num_key_value_heads", None) or info.get(
            "n_attention_heads"
        )
        info["hidden_size"] = getattr(hf, "hidden_size", None)
        info["vocab_size"] = getattr(hf, "vocab_size", None)
        if info.get("hidden_size") and info.get("n_attention_heads"):
            info["head_dim"] = getattr(
                hf, "head_dim", info["hidden_size"] // info["n_attention_heads"]
            )
        info["weight_dtype"] = str(getattr(mc, "dtype", "auto"))
        if (
            info.get("n_layers")
            and info.get("n_kv_heads")
            and info.get("head_dim")
        ):
            info["bytes_per_kv_token_fp16"] = float(
                2 * info["n_layers"] * info["n_kv_heads"] * info["head_dim"] * 2
            )
    except Exception:
        pass
    try:
        params = 0
        for p in llm.llm_engine.model_executor.driver_worker.model_runner.model.parameters():
            params += p.numel()
        info["num_total_params"] = params
    except Exception:
        pass
    return info


def _run_generate(
    llm,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[list[RequestResult], float]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
        ignore_eos=True,
    )

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    per_ms = wall_ms / max(len(prompts), 1)

    results: list[RequestResult] = []
    for idx, out in enumerate(outputs):
        token_ids = out.outputs[0].token_ids
        n_tok = len(token_ids)
        prompt_tokens = len(getattr(out, "prompt_token_ids", []) or [])
        m = getattr(out, "metrics", None)
        ttft = decode_ms_v = tpot = e2e = None
        if m is not None:
            arr = getattr(m, "arrival_time", None)
            ftt = getattr(m, "first_token_time", None)
            fin = getattr(m, "finished_time", None)
            if ftt is not None and arr is not None:
                ttft = (ftt - arr) * 1000.0
            if fin is not None and ftt is not None:
                decode_ms_v = (fin - ftt) * 1000.0
                if n_tok > 1:
                    tpot = decode_ms_v / (n_tok - 1)
            if fin is not None and arr is not None:
                e2e = (fin - arr) * 1000.0
        results.append(
            RequestResult(
                prompt_idx=idx,
                output_tokens=n_tok,
                wall_ms=per_ms,
                tokens_per_second=n_tok / max(per_ms / 1000.0, 1e-9),
                prompt_tokens=prompt_tokens,
                ttft_ms=ttft,
                decode_ms=decode_ms_v,
                tpot_ms=tpot,
                e2e_ms=e2e,
            )
        )
    return results, wall_ms


def benchmark_model_mode(
    spec: ModelSpec,
    *,
    prompts: list[str],
    batch_mode: BatchMode,
    gpu: str,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs_max: int,
    enforce_eager: bool,
    tensor_parallel_size: int | None,
    warmup_requests: int,
    temperature: float,
    seed: int,
    kv_cache_dtype: str = "auto",
    enable_prefix_caching: bool = True,
) -> BenchResult:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    max_num_seqs = _max_num_seqs_for_mode(batch_mode, max_num_seqs_max)
    batch_size = _effective_batch_size(batch_mode)

    base = BenchResult(
        model_id=spec.model_id,
        label=spec.label,
        group=spec.group,
        batch_mode=batch_mode,
        batch_size=batch_size,
        max_num_seqs=max_num_seqs,
        n_prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        total_output_tokens=0,
        wall_ms=0.0,
        throughput_tokens_per_sec=0.0,
        mean_per_request_tokens_per_sec=0.0,
        load_ms=0.0,
        cuda_visible_devices=gpu,
        tensor_parallel_size=tensor_parallel_size or spec.tensor_parallel_size,
        precision=spec.precision,
        quantization=spec.quantization,
        enforce_eager=enforce_eager,
        warmup_requests=warmup_requests,
        timestamp_utc=ts,
        kv_cache_dtype=kv_cache_dtype,
    )

    if spec.skip:
        base.error = spec.skip_reason
        return base

    baseline_mib = _cuda_memory_used_mib(gpu)
    base.baseline_vram_mib = baseline_mib
    llm = None
    try:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        llm, load_ms, tp = _make_llm(
            spec,
            gpu=gpu,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            kv_cache_dtype=kv_cache_dtype,
            enable_prefix_caching=enable_prefix_caching,
        )
        base.load_ms = load_ms
        base.tensor_parallel_size = tp
        for k, v in _extract_arch_info(llm).items():
            if hasattr(base, k):
                setattr(base, k, v)

        if warmup_requests > 0:
            warm = prompts[: min(warmup_requests, len(prompts))]
            if batch_mode == "bs1":
                for p in warm:
                    _run_generate(llm, [p], max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
            else:
                _run_generate(llm, warm, max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)

        ttft_ms_probe: float | None = None
        if batch_mode == "bs1" and prompts:
            from vllm import SamplingParams
            try:
                sp = SamplingParams(
                    temperature=temperature,
                    max_tokens=1,
                    seed=seed,
                    ignore_eos=True,
                )
                t_probe = time.perf_counter()
                _ = llm.generate([prompts[0]], sampling_params=sp)
                ttft_ms_probe = (time.perf_counter() - t_probe) * 1000.0
            except Exception:
                ttft_ms_probe = None

        all_results: list[RequestResult] = []
        total_wall_ms = 0.0

        chunk_size = _chunk_size_for_mode(batch_mode)
        if chunk_size == 1:
            for i, prompt in enumerate(prompts):
                chunk_results, chunk_ms = _run_generate(
                    llm,
                    [prompt],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed + i,
                )
                for r in chunk_results:
                    r.prompt_idx = i
                    r.wall_ms = chunk_ms
                    r.tokens_per_second = r.output_tokens / max(chunk_ms / 1000.0, 1e-9)
                all_results.extend(chunk_results)
                total_wall_ms += chunk_ms
        elif chunk_size is not None:
            offset = 0
            for chunk in _chunked(prompts, chunk_size):
                chunk_results, chunk_ms = _run_generate(
                    llm,
                    chunk,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed + offset,
                )
                for j, r in enumerate(chunk_results):
                    r.prompt_idx = offset + j
                all_results.extend(chunk_results)
                total_wall_ms += chunk_ms
                offset += len(chunk)
        else:  # max
            all_results, total_wall_ms = _run_generate(
                llm,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
            )

        total_tokens = sum(r.output_tokens for r in all_results)
        base.total_output_tokens = total_tokens
        base.wall_ms = total_wall_ms
        base.throughput_tokens_per_sec = total_tokens / max(total_wall_ms / 1000.0, 1e-9)
        base.mean_per_request_tokens_per_sec = (
            sum(r.tokens_per_second for r in all_results) / max(len(all_results), 1)
        )
        base.per_request = all_results
        ttfts = [r.ttft_ms for r in all_results if r.ttft_ms is not None]
        tpots = [r.tpot_ms for r in all_results if r.tpot_ms is not None]
        decs = [r.decode_ms for r in all_results if r.decode_ms is not None]
        e2es = [r.e2e_ms for r in all_results if r.e2e_ms is not None]
        if ttfts:
            base.ttft_ms_mean = sum(ttfts) / len(ttfts)
            base.ttft_ms_p50 = _percentile(ttfts, 0.5)
            base.ttft_ms_p90 = _percentile(ttfts, 0.9)
        if tpots:
            base.tpot_ms_mean = sum(tpots) / len(tpots)
            base.tpot_ms_p50 = _percentile(tpots, 0.5)
            base.tpot_ms_p90 = _percentile(tpots, 0.9)
            base.decode_tokens_per_sec_mean = 1000.0 / base.tpot_ms_mean
        if decs:
            base.decode_ms_mean = sum(decs) / len(decs)
        if e2es:
            base.e2e_ms_mean = sum(e2es) / len(e2es)
        if all_results:
            base.prompt_tokens_mean = sum(r.prompt_tokens for r in all_results) / len(all_results)
            base.output_tokens_mean = sum(r.output_tokens for r in all_results) / len(all_results)
        if ttft_ms_probe is not None:
            base.ttft_ms_mean = ttft_ms_probe
            if batch_mode == "bs1" and base.output_tokens_mean and base.output_tokens_mean > 1:
                avg_per_request_wall = total_wall_ms / max(len(all_results), 1)
                avg_decode = avg_per_request_wall - ttft_ms_probe
                base.decode_ms_mean = avg_decode
                base.tpot_ms_mean = avg_decode / max(base.output_tokens_mean - 1, 1)
                base.decode_tokens_per_sec_mean = 1000.0 / base.tpot_ms_mean
        try:
            import torch
            if torch.cuda.is_available():
                base.torch_peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        used = _cuda_memory_used_mib(gpu)
        if used is not None:
            base.peak_vram_mib = used
            if baseline_mib is not None:
                base.delta_vram_mib = used - baseline_mib
        return base
    except Exception as exc:
        base.error = f"{type(exc).__name__}: {exc}"
        return base
    finally:
        _shutdown_llm(llm)
        _wait_for_gpu_release(gpu, baseline_mib=baseline_mib)


def _result_to_json_dict(res: BenchResult) -> dict[str, Any]:
    d = asdict(res)
    d["per_request"] = [asdict(r) for r in res.per_request]
    return d


def _r(x: float | None, n: int = 4) -> float | str:
    return "" if x is None else round(x, n)


def _append_csv_row(csv_path: Path, res: BenchResult) -> None:
    row = {
        "run_timestamp_utc": res.timestamp_utc,
        "model_id": res.model_id,
        "label": res.label,
        "group": res.group,
        "batch_mode": res.batch_mode,
        "batch_size": res.batch_size,
        "max_num_seqs": res.max_num_seqs,
        "backend": "vllm",
        "precision": res.precision,
        "quantization": res.quantization,
        "kv_cache_dtype": res.kv_cache_dtype,
        "enforce_eager": res.enforce_eager,
        "cuda_visible_devices": res.cuda_visible_devices,
        "tensor_parallel_size": res.tensor_parallel_size,
        "n_prompts": res.n_prompts,
        "max_new_tokens": res.max_new_tokens,
        "warmup_requests": res.warmup_requests,
        "load_ms": round(res.load_ms, 3),
        "total_output_tokens": res.total_output_tokens,
        "wall_ms": round(res.wall_ms, 3),
        "throughput_tokens_per_sec": round(res.throughput_tokens_per_sec, 4),
        "mean_per_request_tokens_per_sec": round(res.mean_per_request_tokens_per_sec, 4),
        "ttft_ms_mean": _r(res.ttft_ms_mean, 3),
        "ttft_ms_p50": _r(res.ttft_ms_p50, 3),
        "ttft_ms_p90": _r(res.ttft_ms_p90, 3),
        "decode_ms_mean": _r(res.decode_ms_mean, 3),
        "tpot_ms_mean": _r(res.tpot_ms_mean, 4),
        "tpot_ms_p50": _r(res.tpot_ms_p50, 4),
        "tpot_ms_p90": _r(res.tpot_ms_p90, 4),
        "e2e_ms_mean": _r(res.e2e_ms_mean, 3),
        "decode_tps_mean": _r(res.decode_tokens_per_sec_mean, 3),
        "prompt_tokens_mean": _r(res.prompt_tokens_mean, 1),
        "output_tokens_mean": _r(res.output_tokens_mean, 1),
        "peak_vram_mib": _r(res.peak_vram_mib, 1),
        "baseline_vram_mib": _r(res.baseline_vram_mib, 1),
        "delta_vram_mib": _r(res.delta_vram_mib, 1),
        "torch_peak_vram_mib": _r(res.torch_peak_vram_mib, 1),
        "n_layers": res.n_layers if res.n_layers is not None else "",
        "n_kv_heads": res.n_kv_heads if res.n_kv_heads is not None else "",
        "n_attention_heads": res.n_attention_heads if res.n_attention_heads is not None else "",
        "head_dim": res.head_dim if res.head_dim is not None else "",
        "hidden_size": res.hidden_size if res.hidden_size is not None else "",
        "vocab_size": res.vocab_size if res.vocab_size is not None else "",
        "bytes_per_kv_token_fp16": _r(res.bytes_per_kv_token_fp16, 2),
        "weight_dtype": res.weight_dtype or "",
        "num_total_params": res.num_total_params if res.num_total_params is not None else "",
        "architecture": res.architecture or "",
        "error": res.error or "",
    }
    write_header = not csv_path.exists()
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                existing = next(reader)
            except StopIteration:
                existing = []
        if existing != list(row.keys()):
            backup = csv_path.with_suffix(csv_path.suffix + ".legacy")
            if not backup.exists():
                csv_path.rename(backup)
                write_header = True
            else:
                write_header = False
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_summary(res: BenchResult) -> None:
    if res.error:
        print(f"  ERROR [{res.batch_mode}]: {res.error}")
        return
    extra = ""
    if res.tpot_ms_mean is not None:
        extra = (
            f", TTFT={res.ttft_ms_mean:.1f}ms, "
            f"TPOT={res.tpot_ms_mean:.2f}ms (decode {res.decode_tokens_per_sec_mean:.1f} tok/s)"
        )
    if res.peak_vram_mib is not None:
        extra += f", peak_vram={res.peak_vram_mib:.0f} MiB"
    print(
        f"  [{res.batch_mode}] kv={res.kv_cache_dtype} "
        f"throughput={res.throughput_tokens_per_sec:.2f} tok/s "
        f"(mean/request={res.mean_per_request_tokens_per_sec:.2f}, "
        f"tokens={res.total_output_tokens}, wall={res.wall_ms/1000:.1f}s, "
        f"load={res.load_ms/1000:.1f}s{extra})"
    )


def _parse_batch_modes(batch_mode_arg: str) -> list[BatchMode]:
    batch_modes: list[BatchMode] = []
    for part in batch_mode_arg.split(","):
        part = part.strip()
        if part not in ("bs1", "bs16", "max"):
            raise SystemExit(f"Unknown batch mode: {part!r}")
        batch_modes.append(part)  # type: ignore[arg-type]
    return batch_modes


def _save_result(res: BenchResult, output_dir: Path, csv_path: Path) -> None:
    slug = res.model_id.replace("/", "__")
    json_path = output_dir / f"{slug}_{res.batch_mode}.json"
    json_path.write_text(json.dumps(_result_to_json_dict(res), indent=2) + "\n")
    _append_csv_row(csv_path, res)
    print(f"  → {json_path}")


def run_single_model(
    spec: ModelSpec,
    *,
    prompts: list[str],
    batch_modes: list[BatchMode],
    args: argparse.Namespace,
    csv_path: Path,
) -> list[BenchResult]:
    print(f"\n=== {spec.label} ({spec.model_id}) ===")
    if spec.skip:
        print(f"  SKIP: {spec.skip_reason}")
        return []

    results: list[BenchResult] = []
    for mode in batch_modes:
        if args.dry_run:
            mns = _max_num_seqs_for_mode(mode, args.max_num_seqs_max)
            print(f"  [dry-run] batch_mode={mode}, max_num_seqs={mns}, n_prompts={len(prompts)}")
            continue

        print(f"  Running batch_mode={mode} …")
        res = benchmark_model_mode(
            spec,
            prompts=prompts,
            batch_mode=mode,
            gpu=args.gpu,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs_max=args.max_num_seqs_max,
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=args.tensor_parallel_size,
            warmup_requests=args.warmup,
            temperature=args.temperature,
            seed=args.seed,
            kv_cache_dtype=args.kv_cache_dtype,
            enable_prefix_caching=not args.disable_prefix_caching,
        )
        _print_summary(res)
        results.append(res)
        _save_result(res, args.output_dir, csv_path)
    return results


def _worker_command(args: argparse.Namespace, model_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-for-model",
        model_id,
        "--batch-mode",
        args.batch_mode,
        "--n-prompts",
        str(args.n_prompts),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu",
        args.gpu,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs-max",
        str(args.max_num_seqs_max),
        "--warmup",
        str(args.warmup),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
    ]
    if args.disable_prefix_caching:
        cmd.append("--disable-prefix-caching")
    if args.tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    else:
        cmd.append("--no-enforce-eager")
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Qwen3 model throughput with vLLM (bs1 / bs16 / max batch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--group",
        choices=GROUPS,
        help="Run all models from a catalog group (fp16, w4a16, w2a16, nvfp4, qwen36, gptq_int4).",
    )
    p.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Single HuggingFace model id (repeatable). Overrides --group.",
    )
    p.add_argument(
        "--batch-mode",
        default="bs1,bs16,max",
        help="Comma-separated: bs1, bs16, max.",
    )
    p.add_argument("--n-prompts", type=int, default=64, help="Number of MATH-500 prompts.")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Generation length cap.")
    p.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len.")
    p.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="vLLM kv_cache_dtype: auto, fp8, fp8_e4m3, fp8_e5m2, nvfp4 (vLLM 0.21+).",
    )
    p.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Disable vLLM prefix caching (cleaner per-request perf measurements).",
    )
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--tensor-parallel-size", type=int, default=None, help="Override TP for all models.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument(
        "--max-num-seqs-max",
        type=int,
        default=256,
        help="max_num_seqs for batch-mode=max.",
    )
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use enforce_eager=True for more stable timings (default: on).",
    )
    p.add_argument("--warmup", type=int, default=1, help="Warmup requests before timing.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "throughput_qwen",
        help="Directory for JSON reports and summary CSV.",
    )
    p.add_argument("--list-models", action="store_true", help="Print catalog and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    p.add_argument(
        "--isolated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each model in a separate Python process (default: on; one model on GPU at a time).",
    )
    p.add_argument(
        "--worker-for-model",
        default=None,
        help=argparse.SUPPRESS,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available model groups:", ", ".join(GROUPS))
        for m in MODEL_CATALOG:
            flag = "SKIP" if m.skip else "OK"
            print(f"  [{flag:4}] {m.group:8}  {m.model_id}  ({m.quantization})")
        return

    batch_modes = _parse_batch_modes(args.batch_mode)

    if args.worker_for_model:
        models = _resolve_models(group=None, model_ids=[args.worker_for_model])
        if not models:
            raise SystemExit(f"Unknown model: {args.worker_for_model!r}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "throughput_summary.csv"
        print(f"Loading {args.n_prompts} MATH-500 prompts …")
        prompts = load_prompts(args.n_prompts, seed=args.seed)
        print(
            f"[worker] model={args.worker_for_model}, batch modes={batch_modes}, "
            f"GPU={args.gpu}, enforce_eager={args.enforce_eager}"
        )
        run_single_model(models[0], prompts=prompts, batch_modes=batch_modes, args=args, csv_path=csv_path)
        return

    models = _resolve_models(group=args.group, model_ids=args.models)
    if not models:
        raise SystemExit("No models selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "throughput_summary.csv"

    print(f"Loading {args.n_prompts} MATH-500 prompts …")
    prompts = load_prompts(args.n_prompts, seed=args.seed)
    print(
        f"Models: {len(models)}, batch modes: {batch_modes}, GPU={args.gpu}, "
        f"enforce_eager={args.enforce_eager}, isolated={args.isolated}"
    )

    all_results: list[dict[str, Any]] = []

    for spec in models:
        if spec.skip:
            print(f"\n=== {spec.label} ({spec.model_id}) ===")
            print(f"  SKIP: {spec.skip_reason}")
            continue

        if args.isolated and not args.dry_run:
            cmd = _worker_command(args, spec.model_id)
            print(f"\n=== spawn worker: {spec.model_id} ===")
            subprocess.run(cmd, check=True)
            slug = spec.model_id.replace("/", "__")
            for mode in batch_modes:
                json_path = args.output_dir / f"{slug}_{mode}.json"
                if json_path.exists():
                    all_results.append(json.loads(json_path.read_text()))
            continue

        for res in run_single_model(
            spec, prompts=prompts, batch_modes=batch_modes, args=args, csv_path=csv_path
        ):
            all_results.append(_result_to_json_dict(res))

    if not args.dry_run and all_results:
        combined = args.output_dir / "throughput_all.json"
        combined.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\nSummary CSV: {csv_path}")
        print(f"Combined JSON: {combined}")


if __name__ == "__main__":
    main()
