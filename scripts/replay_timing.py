#!/usr/bin/env python3
"""Replay trace segments through vLLM for latency benchmarking."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command()
@click.argument("run_id")
@click.option("--results-dir", default="results", show_default=True)
@click.option(
    "--model-id",
    default=None,
    help="Force this HF model id for every segment (single backend). Omit to derive "
    "model / precision / quant / GPU knobs from segment.actor_id + run config.json actors.",
)
@click.option("--precision", default="bf16", show_default=True)
@click.option(
    "--quantization",
    default=None,
    help="Used only with --model-id: vLLM quantization (e.g. gptq); default vLLM auto/none.",
)
@click.option("--tensor-parallel-size", default=None, type=int, help="Fixed mode: default actor[0] or 1")
@click.option(
    "--gpu-memory-utilization",
    default=None,
    type=float,
    help="Fixed mode: default from actor[0] backend_kwargs or 0.9",
)
@click.option(
    "--cuda-visible-devices",
    default=None,
    type=str,
    help="Fixed mode: default CUDA_VISIBLE_DEVICES from actor[0] if set",
)
@click.option("--segments", default=None, help="Comma-separated segment indices to replay")
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    run_id: str,
    results_dir: str,
    model_id: str | None,
    precision: str,
    quantization: str | None,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    cuda_visible_devices: str | None,
    segments: str | None,
    verbose: bool,
) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from quantlab.artifacts.store import ArtifactStore
    from quantlab.timing.replay import VLLMTimingReplay, replay_backend_spec_from_actor

    store = ArtifactStore(base_dir=results_dir)
    config = store.load_config(run_id)

    actors = config["actors"]
    actor_specs = {a["actor_id"]: replay_backend_spec_from_actor(a) for a in actors}
    actor0 = actors[0]
    bk0 = actor0.get("backend_kwargs") or {}

    seg_indices = [int(x) for x in segments.split(",")] if segments else None

    traces = store.load_traces(run_id)

    if model_id is not None:
        tp = tensor_parallel_size if tensor_parallel_size is not None else int(bk0.get("tensor_parallel_size", 1))
        gpu_mu = (
            gpu_memory_utilization if gpu_memory_utilization is not None else float(bk0.get("gpu_memory_utilization", 0.9))
        )
        cuda_vis = cuda_visible_devices if cuda_visible_devices is not None else bk0.get("cuda_visible_devices")
        if verbose:
            click.echo(
                f"Replaying {len(traces)} traces fixed backend model={model_id!r} "
                f"precision={precision!r} cuda_visible_devices={cuda_vis!r} gpu_memory_utilization={gpu_mu}"
            )
        replay = VLLMTimingReplay(
            model_id=model_id,
            precision_mode=precision,
            quantization=quantization,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mu,
            cuda_visible_devices=cuda_vis if cuda_vis is None else str(cuda_vis).strip(),
        )
    else:
        if verbose:
            desc = ", ".join(f"{aid}→{spec.model_id}" for aid, spec in sorted(actor_specs.items()))
            click.echo(
                f"Replaying {len(traces)} traces per segment.actor_id backends: {desc}"
            )
        replay = VLLMTimingReplay(actor_backend_specs=actor_specs)

    replay_dir = store.run_dir(run_id) / "timing_replay"
    replay_dir.mkdir(exist_ok=True)

    for trace in traces:
        results = replay.replay_trace(trace, seg_indices)
        out = [
            {
                "segment_label": r.segment_label,
                "actor_id": r.actor_id,
                "token_count": r.token_count,
                "timing": r.timing.to_dict(),
                "prompt_len_chars": r.prompt_len_chars,
            }
            for r in results
        ]
        path = replay_dir / f"{trace.example_id}.json"
        path.write_text(json.dumps(out, indent=2))
        if verbose:
            for r in results:
                click.echo(f"  {r.segment_label}: {r.token_count} tokens, {r.timing.total_ms:.1f}ms")

    replay.unload()
    click.echo(f"Replay results saved to {replay_dir}")


if __name__ == "__main__":
    main()
