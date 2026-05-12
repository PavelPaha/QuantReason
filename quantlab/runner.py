from __future__ import annotations

import traceback
from typing import Any, Optional

from quantlab.actors.base import ActorConfig
from quantlab.actors.registry import ActorRegistry
from quantlab.artifacts.store import ArtifactStore
from quantlab.benchmarks.registry import BenchmarkRegistry
from quantlab.config.schema import ExperimentConfig
from quantlab.core.types import (
    GenerationParams,
    HandoffMode,
    PrecisionMode,
    QuantizationMethod,
    SegmentRole,
)
from quantlab.evaluation.judge import judge
from quantlab.metrics.base import MetricBase
from quantlab.metrics.registry import MetricRegistry
from quantlab.pipeline.executor import EXECUTOR_STATE_KEY, PipelineExecutor
from quantlab.pipeline.stage import PipelineStage
from quantlab.core.trace import Trace
from quantlab.switching.registry import ConditionRegistry


def _unload_all_actors(actors: dict[str, Any]) -> None:
    for actor in actors.values():
        actor.unload()


def _record_example(
    *,
    store: ArtifactStore,
    run_id: str,
    config: ExperimentConfig,
    metrics: list[MetricBase],
    adapter,
    example,
    trace: Trace,
    verbose: bool,
) -> tuple[bool, bool]:
    """Persist judgement, optional trace, metrics, timing; return (is_correct, parse_success)."""
    judgement = judge(trace, example, adapter)

    if config.output.save_traces:
        store.save_trace(run_id, trace)
    store.save_judgement(run_id, judgement)

    metric_values: dict[str, Any] = {}
    for metric in metrics:
        try:
            metric_values[metric.name] = metric.compute(trace, judgement)
        except Exception as e:
            metric_values[metric.name] = None
            if verbose:
                print(f"[runner] metric {metric.name} failed: {e}")

    store.save_metrics(run_id, example.example_id, metric_values)

    if config.output.save_timing:
        timing_data: dict[str, Any] = {}
        for seg in trace.segments:
            if seg.timing:
                timing_data.setdefault(seg.actor_id, {})
                t = timing_data[seg.actor_id]
                t["total_ms"] = t.get("total_ms", 0) + (seg.timing.total_ms or 0)
                t["token_count"] = t.get("token_count", 0) + seg.token_count
        store.save_timing(run_id, example.example_id, timing_data)

    return judgement.is_correct, judgement.parse_success


def _build_actor_config(actor_def) -> ActorConfig:
    return ActorConfig(
        actor_id=actor_def.actor_id,
        model_id=actor_def.model_id,
        backend=actor_def.backend,
        precision=PrecisionMode(actor_def.precision),
        quantization=QuantizationMethod(actor_def.quantization),
        quantization_config=actor_def.quantization_config,
        generation_params=GenerationParams(**actor_def.generation_params.model_dump()),
        device_map=actor_def.device_map,
        backend_kwargs=actor_def.backend_kwargs,
    )


def _build_stage(stage_cfg, condition_registry: type = ConditionRegistry) -> PipelineStage:
    conditions = [
        condition_registry.build(c.name, **c.kwargs)
        for c in stage_cfg.exit_conditions
    ]
    return PipelineStage(
        actor_id=stage_cfg.actor_id,
        exit_conditions=conditions,
        handoff_mode=HandoffMode(stage_cfg.handoff_mode),
        max_new_tokens=stage_cfg.max_new_tokens,
        stop_sequences=stage_cfg.stop_sequences,
        role=SegmentRole(stage_cfg.role),
        fallback_stage_index=stage_cfg.fallback_stage_index,
        loop_back_stage_index=stage_cfg.loop_back_stage_index,
        stage_prompt=stage_cfg.stage_prompt,
    )


def _incoming_wave_stage_idx(trace: Trace, wave_idx: int) -> int:
    """Interpreter stage cursor from partial staged metadata or ``wave_idx`` for fresh traces."""
    st = trace.metadata.get(EXECUTOR_STATE_KEY)
    if isinstance(st, dict) and st.get("stage_idx") is not None:
        return int(st["stage_idx"])
    return wave_idx


def run_experiment(
    config: ExperimentConfig,
    verbose: bool = False,
    *,
    staged_execution: Optional[bool] = None,
    resume_run_id: Optional[str] = None,
    resume_after_wave: Optional[int] = None,
) -> str:
    """
    Execute a full experiment run from an ExperimentConfig.

    Returns the run_id.

    staged_execution:
        If True (or YAML ``staged_execution: true``), runs one pipeline depth level
        across all examples before the next (same semantics as ``full_prefill``).
        Optional unload between waves frees GPU memory for single-model cards.

    resume_run_id / resume_after_wave:
        Only with staged execution. Loads ``trace_checkpoints/wave_<N>.jsonl`` where
        N = ``resume_after_wave``, reconnects surviving traces and continues pipelines
        from wave N+1. Skips ``_record_example`` for examples already present in
        ``judgements.jsonl``.
        Completed examples are flushed to ``traces.jsonl`` immediately after their
        final wave (along with judgement/metrics/timing), rather than waiting until
        the whole batch completes.

        ``staged_batch_size >= 2`` (YAML or CLI): each staged wave invokes vLLM with
        micro-batched prompts; ``traces.jsonl`` stays schema-compatible but per-segment
        ``timing`` reflects batch allocation (run ``replay_timing`` for faithful latencies).
    """
    use_staged = config.staged_execution if staged_execution is None else staged_execution

    if (resume_run_id is None) != (resume_after_wave is None):
        raise ValueError(
            "resume_run_id and resume_after_wave must be set together or both omitted"
        )
    if resume_run_id is not None and not use_staged:
        raise ValueError("resume_* requires staged execution (--staged or YAML staged_execution)")

    store = ArtifactStore(base_dir=config.output.base_dir)
    if resume_run_id is not None:
        rd = store.run_dir(resume_run_id)
        if not rd.is_dir() or not (rd / "config.json").exists():
            raise FileNotFoundError(f"resume run not found or missing config.json: {rd}")
        run_id = resume_run_id
        if verbose:
            print(f"[runner] resuming run_id={run_id} after wave checkpoint {resume_after_wave}")
    else:
        run_id = store.new_run(config.experiment_name, config.model_dump())

    if verbose:
        print(f"[runner] run_id={run_id}  experiment={config.experiment_name}")

    # Build actors
    actors = {}
    for actor_def in config.actors:
        ac = _build_actor_config(actor_def)
        actors[ac.actor_id] = ActorRegistry.build(ac)

    # Build pipeline stages
    stages = [_build_stage(s) for s in config.pipeline]

    # Build metrics
    metrics: list[MetricBase] = [
        MetricRegistry.build(m.name, **m.kwargs) for m in config.metrics
    ]

    # Load benchmark
    adapter = BenchmarkRegistry.build(config.benchmark.name)
    examples = adapter.load(
        split=config.benchmark.split,
        subset=config.benchmark.subset,
        max_examples=config.benchmark.max_examples,
        seed=config.benchmark.seed,
    )

    if verbose:
        print(f"[runner] loaded {len(examples)} examples from {config.benchmark.name}")

    executor = PipelineExecutor(
        stages=stages,
        actors=actors,
        verbose=verbose,
    )

    already_judged = (
        store.list_judged_example_ids(run_id) if resume_run_id is not None else set()
    )

    if use_staged:
        n_stages = len(stages)
        if resume_run_id is not None:
            assert resume_after_wave is not None
            ck = store.load_staged_wave_checkpoint(run_id, resume_after_wave)
            traces = {t.example_id: t for t in ck}
            if not traces:
                raise FileNotFoundError(
                    f"empty or missing checkpoint: trace_checkpoints/"
                    f"wave_{resume_after_wave}.jsonl under {run_id}"
                )
            if resume_after_wave < 0 or resume_after_wave >= n_stages:
                raise ValueError(
                    f"resume_after_wave={resume_after_wave} out of range for "
                    f"{n_stages} pipeline stage(s)"
                )
            wave_start = resume_after_wave + 1
            failed: set[str] = set()
            if verbose:
                print(f"[runner] resume: loaded {len(traces)} traces, waves {wave_start}..")
        else:
            traces = {}
            failed = set()
            wave_start = 0

        # Examples persisted to traces.jsonl / judgements immediately after the final wave
        persisted_examples: set[str] = set(already_judged)

        staged_bs_raw = getattr(config, "staged_batch_size", None)
        staged_bs = staged_bs_raw if staged_bs_raw is not None else 1

        for w in range(wave_start, n_stages):
            stop_before = (w + 1) if w < n_stages - 1 else None
            stage_row = stages[w]
            wave_actor = actors[stage_row.actor_id]

            staged_use_vllm_batch = staged_bs >= 2
            if staged_use_vllm_batch:
                from quantlab.actors.impls import VLLMActor

                if not isinstance(wave_actor, VLLMActor):
                    staged_use_vllm_batch = False
                    if verbose:
                        print(
                            f"[runner] staged_batch_size={staged_bs} ignored at wave {w}: "
                            f"{stage_row.actor_id!r} is not vLLM"
                        )

            def persist_final_wave(example, eida: str) -> None:
                if (
                    w == n_stages - 1
                    and eida not in failed
                    and eida not in already_judged
                    and eida not in persisted_examples
                ):
                    if verbose:
                        print(f"[runner] staged persist {eida}")
                    try:
                        _record_example(
                            store=store,
                            run_id=run_id,
                            config=config,
                            metrics=metrics,
                            adapter=adapter,
                            example=example,
                            trace=traces[eida],
                            verbose=verbose,
                        )
                        persisted_examples.add(eida)
                    except Exception as exc:
                        err = traceback.format_exc()
                        if verbose:
                            print(f"[runner] ERROR persisting {eida}: {exc}")
                        store.save_error(run_id, eida, err)

            def run_one_wave(i: int, example) -> None:
                eida = example.example_id
                if verbose and i % 10 == 0:
                    print(f"[runner] staged wave {w}/{n_stages}  example {i}/{len(examples)} ...")
                if eida not in traces:
                    traces[eida] = executor.run(
                        eida, example.prompt, stop_before_stage=stop_before
                    )
                else:
                    traces[eida] = executor.continue_run(
                        traces[eida], stop_before_stage=stop_before
                    )
                persist_final_wave(example, eida)

            def run_one_wave_batched(seg_i: int, example, segment) -> None:
                eida = example.example_id
                if verbose and seg_i % 10 == 0:
                    print(
                        f"[runner] staged wave {w}/{n_stages}  example {seg_i}/{len(examples)} "
                        f"(batch) ..."
                    )
                tr = traces[eida]
                start_sid = _incoming_wave_stage_idx(tr, w)
                executor.reset_switch_conditions_from(start_sid)
                traces[eida] = executor.consume_segment_after_generate(
                    tr,
                    segment,
                    start_stage_idx=start_sid,
                    stop_before_stage=stop_before,
                )
                persist_final_wave(example, eida)

            if staged_use_vllm_batch and verbose:
                print(
                    f"[runner] staged wave {w}: vLLM micro-batch size={staged_bs} "
                    f"(traces unchanged; segment timings degraded under batching)"
                )

            if staged_use_vllm_batch:
                queue_ix: list[int] = []
                queue_ex = []
                for i, example in enumerate(examples):
                    eida = example.example_id
                    if eida in failed:
                        continue
                    queue_ix.append(i)
                    queue_ex.append(example)
                qb = 0
                while qb < len(queue_ex):
                    chunk_ix = queue_ix[qb : qb + staged_bs]
                    chunk_ex = queue_ex[qb : qb + staged_bs]
                    qb += len(chunk_ex)
                    try:
                        tr_list = []
                        for ex in chunk_ex:
                            ee = ex.example_id
                            if ee not in traces:
                                traces[ee] = Trace(ee, ex.prompt)
                            tr_list.append(traces[ee])
                        segments = wave_actor.generate_batch_segments(
                            tr_list,
                            prompt_suffix=stage_row.stage_prompt,
                            max_new_tokens=stage_row.max_new_tokens,
                            stop_sequences=stage_row.stop_sequences or None,
                            role=stage_row.role,
                        )
                    except Exception as be:
                        if verbose:
                            print(
                                f"[runner] vLLM batch failed ({be!r}); "
                                "fallback sequential for chunk"
                            )
                        for j, example_fb in enumerate(chunk_ex):
                            eida = example_fb.example_id
                            if eida in failed:
                                continue
                            try:
                                run_one_wave(chunk_ix[j], example_fb)
                            except Exception as exc2:
                                err = traceback.format_exc()
                                if verbose:
                                    print(
                                        f"[runner] ERROR on {eida} "
                                        f"(staged wave {w}, seq FB): {exc2}"
                                    )
                                store.save_error(run_id, eida, err)
                                failed.add(eida)
                                traces.pop(eida, None)
                        continue

                    if len(segments) != len(chunk_ex):
                        raise RuntimeError(
                            "vLLM batch length mismatch "
                            f"({len(segments)} segments vs {len(chunk_ex)} examples)"
                        )

                    for j, example_bm in enumerate(chunk_ex):
                        eida = example_bm.example_id
                        if eida in failed:
                            continue
                        try:
                            run_one_wave_batched(chunk_ix[j], example_bm, segments[j])
                        except Exception as exc:
                            err = traceback.format_exc()
                            if verbose:
                                print(f"[runner] ERROR on {eida} (staged wave {w}): {exc}")
                            store.save_error(run_id, eida, err)
                            failed.add(eida)
                            traces.pop(eida, None)
            else:
                for i, example in enumerate(examples):
                    eid = example.example_id
                    if eid in failed:
                        continue
                    try:
                        run_one_wave(i, example)
                    except Exception as exc:
                        error_msg = traceback.format_exc()
                        if verbose:
                            print(f"[runner] ERROR on {eid} (staged wave {w}): {exc}")
                        store.save_error(run_id, eid, error_msg)
                        failed.add(eid)
                        traces.pop(eid, None)

            if (
                config.output.save_traces
                and config.output.staged_wave_checkpoints
                and traces
            ):
                store.save_staged_wave_checkpoint(run_id, w, traces)
                if verbose:
                    print(
                        f"[runner] checkpoint wave {w}: "
                        f"{len(traces)} traces → trace_checkpoints/wave_{w}.jsonl"
                    )

            if config.staged_unload_between_waves and w < n_stages - 1:
                _unload_all_actors(actors)

        for i, example in enumerate(examples):
            eid = example.example_id
            if eid not in traces or eid in already_judged or eid in persisted_examples:
                continue
            if verbose and i % 10 == 0:
                print(f"[runner] evaluate {i}/{len(examples)} ...")
            try:
                _record_example(
                    store=store,
                    run_id=run_id,
                    config=config,
                    metrics=metrics,
                    adapter=adapter,
                    example=example,
                    trace=traces[eid],
                    verbose=verbose,
                )
                persisted_examples.add(eid)
            except Exception as e:
                error_msg = traceback.format_exc()
                if verbose:
                    print(f"[runner] ERROR on {eid}: {e}")
                store.save_error(run_id, eid, error_msg)

    else:
        for i, example in enumerate(examples):
            if verbose and i % 10 == 0:
                print(f"[runner] {i}/{len(examples)} ...")
            try:
                trace = executor.run(example.example_id, example.prompt)
                _record_example(
                    store=store,
                    run_id=run_id,
                    config=config,
                    metrics=metrics,
                    adapter=adapter,
                    example=example,
                    trace=trace,
                    verbose=verbose,
                )

            except Exception as e:
                error_msg = traceback.format_exc()
                if verbose:
                    print(f"[runner] ERROR on {example.example_id}: {e}")
                store.save_error(run_id, example.example_id, error_msg)

    n = len(examples)
    all_j = store.load_judgements(run_id)
    nj = len(all_j)
    summary = {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "n_examples": n,
        "n_judged": nj,
        "accuracy": (sum(1 for j in all_j if j.get("is_correct")) / nj) if nj else 0.0,
        "parse_rate": (sum(1 for j in all_j if j.get("parse_success")) / nj) if nj else 0.0,
    }
    store.save_summary(run_id, summary)

    if verbose:
        print(f"[runner] done. accuracy={summary['accuracy']:.3f}  run_id={run_id}")

    # Optional vLLM timing replay
    if config.timing_replay and config.timing_replay.enabled:
        _run_timing_replay(run_id, config, store, verbose)

    return run_id


def _run_timing_replay(run_id, config, store, verbose):
    from quantlab.timing.replay import VLLMTimingReplay, replay_backend_spec_from_actor

    if verbose:
        print("[runner] running vLLM timing replay ...")

    rc = config.timing_replay
    bk = (config.actors[0].backend_kwargs or {})
    specs_by_actor = {
        a.actor_id: replay_backend_spec_from_actor(a.model_dump(mode="python")) for a in config.actors
    }
    if rc.model_id:
        replay = VLLMTimingReplay(
            model_id=rc.model_id,
            precision_mode=rc.precision,
            quantization=rc.quantization,
            tensor_parallel_size=rc.tensor_parallel_size,
            gpu_memory_utilization=float(bk.get("gpu_memory_utilization", 0.9)),
            cuda_visible_devices=bk.get("cuda_visible_devices"),
        )
        if verbose:
            print(f"[runner] timing replay fixed model_id={rc.model_id!r}")
    else:
        replay = VLLMTimingReplay(actor_backend_specs=specs_by_actor)
        if verbose:
            desc = ", ".join(f"{k}→{v.model_id}" for k, v in sorted(specs_by_actor.items()))
            print(f"[runner] timing replay per segment.actor_id: {desc}")

    traces = store.load_traces(run_id)
    replay_dir = store.run_dir(run_id) / "timing_replay"
    replay_dir.mkdir(exist_ok=True)

    import json
    for trace in traces:
        results = replay.replay_trace(trace, rc.segments_to_replay)
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

    replay.unload()
