from __future__ import annotations

import time
import traceback
from datetime import datetime
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
from quantlab.pipeline.staged_cyclic import (
    cyclic_stage_for_wave,
    trace_in_cyclic_loop,
    trace_pending_stage_idx,
)
from quantlab.pipeline.stage import PipelineStage
from quantlab.core.trace import Trace
from quantlab.switching.registry import ConditionRegistry
from quantlab.wandb_logger import WandbRunLogger


_STDOUT_PROGRESS_INTERVAL_SEC = 300.0


def _stdout_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _ProgressHeartbeat:
    def __init__(self, *, total_examples: int, interval_sec: float = _STDOUT_PROGRESS_INTERVAL_SEC) -> None:
        self.total_examples = max(int(total_examples), 0)
        self.interval_sec = float(interval_sec)
        self._started_at = time.monotonic()
        self._last_report_at = self._started_at

    def maybe_log(
        self,
        *,
        completed_examples: int,
        error_count: int,
        current_example: Optional[int] = None,
        current_wave: Optional[int] = None,
        total_waves: Optional[int] = None,
    ) -> None:
        now = time.monotonic()
        if now - self._last_report_at < self.interval_sec:
            return
        self._last_report_at = now

        parts = ["[runner] progress"]
        if self.total_examples > 0:
            percent = 100.0 * completed_examples / self.total_examples
            parts.append(f"done={completed_examples}/{self.total_examples} ({percent:.1f}%)")
        else:
            parts.append(f"done={completed_examples}")
        parts.append(f"errors={error_count}")
        if current_example is not None and self.total_examples > 0:
            parts.append(f"cursor={current_example}/{self.total_examples}")
        if current_wave is not None and total_waves is not None:
            parts.append(f"wave={current_wave + 1}/{total_waves}")
        parts.append(f"elapsed={_format_elapsed(now - self._started_at)}")
        _stdout_log(" ".join(parts))


def _unload_all_actors(actors: dict[str, Any]) -> None:
    for actor in actors.values():
        actor.unload()


def _build_timing_data(trace: Trace) -> dict[str, Any]:
    timing_data: dict[str, Any] = {}
    for seg in trace.segments:
        if seg.timing:
            timing_data.setdefault(seg.actor_id, {})
            t = timing_data[seg.actor_id]
            t["total_ms"] = t.get("total_ms", 0) + (seg.timing.total_ms or 0)
            t["token_count"] = t.get("token_count", 0) + seg.token_count
    return timing_data


def _build_run_summary(
    *,
    run_id: str,
    config: ExperimentConfig,
    n_examples: int,
    judgements: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    error_count: int,
    output_dir: str,
) -> dict[str, Any]:
    nj = len(judgements)
    summary = {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "benchmark_name": config.benchmark.name,
        "n_examples": n_examples,
        "n_judged": nj,
        "n_errors": error_count,
        "output_dir": output_dir,
        "accuracy": (sum(1 for j in judgements if j.get("is_correct")) / nj) if nj else 0.0,
        "parse_rate": (sum(1 for j in judgements if j.get("parse_success")) / nj) if nj else 0.0,
    }

    scalar_metrics: dict[str, list[float]] = {}
    for row in metrics_rows:
        for key, value in row.items():
            if key == "example_id" or key in {"accuracy", "parse_rate"}:
                continue
            if isinstance(value, bool):
                scalar_metrics.setdefault(key, []).append(int(value))
            elif isinstance(value, (int, float)):
                scalar_metrics.setdefault(key, []).append(value)

    for metric_name, values in scalar_metrics.items():
        if not values:
            continue
        summary[f"{metric_name}_mean"] = sum(values) / len(values)
        summary[f"{metric_name}_n"] = len(values)

    return summary


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
    wandb_logger: Optional[WandbRunLogger] = None,
) -> None:
    """Persist judgement, trace artifacts, timing, and optional external logging."""
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
                _stdout_log(f"[runner] metric {metric.name} failed: {e}")

    store.save_metrics(run_id, example.example_id, metric_values)

    timing_data = {}
    if config.output.save_timing:
        timing_data = _build_timing_data(trace)
        store.save_timing(run_id, example.example_id, timing_data)

    if wandb_logger is not None:
        wandb_logger.record_example(
            example_id=example.example_id,
            judgement=judgement,
            metric_values=metric_values,
            timing_data=timing_data,
        )


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
    targets = [c.target_stage_index for c in stage_cfg.exit_conditions]
    end_pipeline_flags = [c.end_pipeline for c in stage_cfg.exit_conditions]
    return PipelineStage(
        actor_id=stage_cfg.actor_id,
        exit_conditions=conditions,
        exit_condition_targets=targets,
        exit_condition_end_pipeline=end_pipeline_flags,
        handoff_mode=HandoffMode(stage_cfg.handoff_mode),
        max_new_tokens=stage_cfg.max_new_tokens,
        stop_sequences=stage_cfg.stop_sequences,
        role=SegmentRole(stage_cfg.role),
        fallback_stage_index=stage_cfg.fallback_stage_index,
        loop_back_stage_index=stage_cfg.loop_back_stage_index,
        natural_next_stage_index=stage_cfg.natural_next_stage_index,
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
    log_verbose = _stdout_log if verbose else None

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
        _stdout_log(f"[runner] resuming run_id={run_id} after wave checkpoint {resume_after_wave}")
    else:
        run_id = store.new_run(config.experiment_name, config.model_dump())

    _stdout_log(f"[runner] run_id={run_id} experiment={config.experiment_name}")

    wandb_logger = WandbRunLogger(
        config=config.wandb,
        experiment_config=config,
        run_id=run_id,
        verbose=verbose,
    )

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

    _stdout_log(f"[runner] loaded {len(examples)} examples from {config.benchmark.name}")
    heartbeat = _ProgressHeartbeat(total_examples=len(examples))

    executor = PipelineExecutor(
        stages=stages,
        actors=actors,
        max_total_tokens=config.pipeline_max_total_tokens,
        verbose=verbose,
    )

    already_judged = (
        store.list_judged_example_ids(run_id) if resume_run_id is not None else set()
    )
    error_count_so_far = len(store.load_errors(run_id)) if resume_run_id is not None else 0

    if use_staged:
        n_stages = len(stages)
        loop_stage_indices = config.staged_cyclic_loop_stage_indices
        use_staged_cyclic = bool(loop_stage_indices)
        plan_stage_index = config.staged_cyclic_plan_stage_index
        if use_staged_cyclic:
            if not loop_stage_indices:
                raise ValueError("staged_cyclic_loop_stage_indices must be non-empty when set")
            for idx in loop_stage_indices:
                if idx < 0 or idx >= n_stages:
                    raise ValueError(
                        f"staged_cyclic_loop_stage_indices contains {idx}, "
                        f"pipeline has {n_stages} stage(s)"
                    )
            if plan_stage_index < 0 or plan_stage_index >= n_stages:
                raise ValueError(f"staged_cyclic_plan_stage_index={plan_stage_index} out of range")

        if resume_run_id is not None:
            assert resume_after_wave is not None
            ck = store.load_staged_wave_checkpoint(run_id, resume_after_wave)
            traces = {t.example_id: t for t in ck}
            if not traces:
                raise FileNotFoundError(
                    f"empty or missing checkpoint: trace_checkpoints/"
                    f"wave_{resume_after_wave}.jsonl under {run_id}"
                )
            if not use_staged_cyclic and (
                resume_after_wave < 0 or resume_after_wave >= n_stages
            ):
                raise ValueError(
                    f"resume_after_wave={resume_after_wave} out of range for "
                    f"{n_stages} pipeline stage(s)"
                )
            if use_staged_cyclic and resume_after_wave < 0:
                raise ValueError(f"resume_after_wave={resume_after_wave} must be >= 0")
            wave_start = resume_after_wave + 1
            failed: set[str] = set()
            if log_verbose is not None:
                log_verbose(f"[runner] resume: loaded {len(traces)} traces, waves {wave_start}..")
        else:
            traces = {}
            failed = set()
            wave_start = 0

        # Examples persisted to traces.jsonl / judgements immediately after the final wave
        persisted_examples: set[str] = set(already_judged)

        staged_bs_raw = getattr(config, "staged_batch_size", None)
        staged_bs = staged_bs_raw if staged_bs_raw is not None else 1

        def _any_in_cyclic_loop() -> bool:
            assert loop_stage_indices is not None
            return any(
                trace_in_cyclic_loop(
                    traces[eid],
                    loop_stage_indices=loop_stage_indices,
                    max_total_tokens=executor.max_total_tokens,
                )
                for eid in traces
                if eid not in failed
            )

        def _example_runnable_at_stage(eida: str, stage_idx: int) -> bool:
            if eida in failed:
                return False
            if eida not in traces:
                if use_staged_cyclic:
                    return stage_idx == plan_stage_index
                return stage_idx == 0
            pending = trace_pending_stage_idx(traces[eida])
            if pending is None:
                return False
            return pending == stage_idx

        w = wave_start
        while True:
            if use_staged_cyclic:
                assert loop_stage_indices is not None
                if w == 0 and not traces:
                    current_stage = plan_stage_index
                    is_last_wave = False
                elif _any_in_cyclic_loop():
                    current_stage = cyclic_stage_for_wave(
                        w,
                        plan_stage_index=plan_stage_index,
                        loop_stage_indices=loop_stage_indices,
                    )
                    is_last_wave = False
                else:
                    break
                total_waves = w + 1
            else:
                if w >= n_stages:
                    break
                current_stage = w
                is_last_wave = w == n_stages - 1
                total_waves = n_stages

            stop_before = None if is_last_wave else current_stage + 1
            stage_row = stages[current_stage]
            wave_actor = actors[stage_row.actor_id]

            staged_use_vllm_batch = staged_bs >= 2
            if staged_use_vllm_batch:
                from quantlab.actors.impls import VLLMActor

                if not isinstance(wave_actor, VLLMActor):
                    staged_use_vllm_batch = False
                    if log_verbose is not None:
                        log_verbose(
                            f"[runner] staged_batch_size={staged_bs} ignored at wave {w}: "
                            f"{stage_row.actor_id!r} is not vLLM"
                        )

            def persist_final_wave(example, eida: str) -> None:
                nonlocal error_count_so_far
                tr = traces.get(eida)
                if use_staged_cyclic:
                    example_done = (
                        tr is not None
                        and trace_pending_stage_idx(tr) is None
                        and tr.finished_at is not None
                    )
                else:
                    example_done = is_last_wave
                if (
                    example_done
                    and eida not in failed
                    and eida not in already_judged
                    and eida not in persisted_examples
                ):
                    if log_verbose is not None:
                        log_verbose(f"[runner] staged persist {eida}")
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
                            wandb_logger=wandb_logger,
                        )
                        persisted_examples.add(eida)
                        heartbeat.maybe_log(
                            completed_examples=len(persisted_examples) + error_count_so_far,
                            error_count=error_count_so_far,
                            current_wave=w,
                            total_waves=total_waves,
                        )
                    except Exception as exc:
                        err = traceback.format_exc()
                        if log_verbose is not None:
                            log_verbose(f"[runner] ERROR persisting {eida}: {exc}")
                        store.save_error(run_id, eida, err)
                        error_count_so_far += 1
                        wandb_logger.record_error(example_id=eida, error=err, wave_index=w)

            def run_one_wave(i: int, example) -> None:
                eida = example.example_id
                if not _example_runnable_at_stage(eida, current_stage):
                    return
                heartbeat.maybe_log(
                    completed_examples=len(persisted_examples) + error_count_so_far,
                    error_count=error_count_so_far,
                    current_example=i + 1,
                    current_wave=w,
                    total_waves=total_waves,
                )
                if log_verbose is not None and i % 10 == 0:
                    log_verbose(
                        f"[runner] staged wave {w} stage={current_stage}  "
                        f"example {i}/{len(examples)} ..."
                    )
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
                if not _example_runnable_at_stage(eida, current_stage):
                    return
                heartbeat.maybe_log(
                    completed_examples=len(persisted_examples) + error_count_so_far,
                    error_count=error_count_so_far,
                    current_example=seg_i + 1,
                    current_wave=w,
                    total_waves=total_waves,
                )
                if log_verbose is not None and seg_i % 10 == 0:
                    log_verbose(
                        f"[runner] staged wave {w} stage={current_stage}  "
                        f"example {seg_i}/{len(examples)} (batch) ..."
                    )
                tr = traces[eida]
                start_sid = _incoming_wave_stage_idx(tr, current_stage)
                executor.reset_switch_conditions_from(start_sid)
                traces[eida] = executor.consume_segment_after_generate(
                    tr,
                    segment,
                    start_stage_idx=start_sid,
                    stop_before_stage=stop_before,
                )
                persist_final_wave(example, eida)

            if staged_use_vllm_batch and log_verbose is not None:
                log_verbose(
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
                        if log_verbose is not None:
                            log_verbose(
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
                                if log_verbose is not None:
                                    log_verbose(
                                        f"[runner] ERROR on {eida} "
                                        f"(staged wave {w}, seq FB): {exc2}"
                                    )
                                store.save_error(run_id, eida, err)
                                error_count_so_far += 1
                                failed.add(eida)
                                traces.pop(eida, None)
                                wandb_logger.record_error(example_id=eida, error=err, wave_index=w)
                                heartbeat.maybe_log(
                                    completed_examples=len(persisted_examples) + error_count_so_far,
                                    error_count=error_count_so_far,
                                    current_example=chunk_ix[j] + 1,
                                    current_wave=w,
                                    total_waves=total_waves,
                                )
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
                            if log_verbose is not None:
                                log_verbose(f"[runner] ERROR on {eida} (staged wave {w}): {exc}")
                            store.save_error(run_id, eida, err)
                            error_count_so_far += 1
                            failed.add(eida)
                            traces.pop(eida, None)
                            wandb_logger.record_error(example_id=eida, error=err, wave_index=w)
                            heartbeat.maybe_log(
                                completed_examples=len(persisted_examples) + error_count_so_far,
                                error_count=error_count_so_far,
                                current_example=chunk_ix[j] + 1,
                                current_wave=w,
                                total_waves=total_waves,
                            )
            else:
                for i, example in enumerate(examples):
                    eid = example.example_id
                    if eid in failed:
                        continue
                    try:
                        run_one_wave(i, example)
                    except Exception as exc:
                        error_msg = traceback.format_exc()
                        if log_verbose is not None:
                            log_verbose(f"[runner] ERROR on {eid} (staged wave {w}): {exc}")
                        store.save_error(run_id, eid, error_msg)
                        error_count_so_far += 1
                        failed.add(eid)
                        traces.pop(eid, None)
                        wandb_logger.record_error(example_id=eid, error=error_msg, wave_index=w)
                        heartbeat.maybe_log(
                            completed_examples=len(persisted_examples) + error_count_so_far,
                            error_count=error_count_so_far,
                            current_example=i + 1,
                            current_wave=w,
                            total_waves=total_waves,
                        )

            if (
                config.output.save_traces
                and config.output.staged_wave_checkpoints
                and traces
            ):
                store.save_staged_wave_checkpoint(run_id, w, traces)
                if log_verbose is not None:
                    log_verbose(
                        f"[runner] checkpoint wave {w}: "
                        f"{len(traces)} traces → trace_checkpoints/wave_{w}.jsonl"
                    )
            heartbeat.maybe_log(
                completed_examples=len(persisted_examples) + error_count_so_far,
                error_count=error_count_so_far,
                current_wave=w,
                total_waves=total_waves,
            )

            wandb_logger.log_wave_end(
                wave_index=w,
                total_waves=total_waves,
                persisted_examples=len(persisted_examples),
            )

            if config.staged_unload_between_waves and not is_last_wave:
                _unload_all_actors(actors)

            w += 1

        for i, example in enumerate(examples):
            eid = example.example_id
            if eid not in traces or eid in already_judged or eid in persisted_examples:
                continue
            heartbeat.maybe_log(
                completed_examples=len(persisted_examples) + error_count_so_far,
                error_count=error_count_so_far,
                current_example=i + 1,
                current_wave=n_stages - 1,
                total_waves=n_stages,
            )
            if log_verbose is not None and i % 10 == 0:
                log_verbose(f"[runner] evaluate {i}/{len(examples)} ...")
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
                    wandb_logger=wandb_logger,
                )
                persisted_examples.add(eid)
                heartbeat.maybe_log(
                    completed_examples=len(persisted_examples) + error_count_so_far,
                    error_count=error_count_so_far,
                    current_example=i + 1,
                    current_wave=n_stages - 1,
                    total_waves=n_stages,
                )
            except Exception as e:
                error_msg = traceback.format_exc()
                if log_verbose is not None:
                    log_verbose(f"[runner] ERROR on {eid}: {e}")
                store.save_error(run_id, eid, error_msg)
                error_count_so_far += 1
                wandb_logger.record_error(example_id=eid, error=error_msg)
                failed.add(eid)
                heartbeat.maybe_log(
                    completed_examples=len(persisted_examples) + error_count_so_far,
                    error_count=error_count_so_far,
                    current_example=i + 1,
                    current_wave=n_stages - 1,
                    total_waves=n_stages,
                )

    else:
        nonstaged_errors = error_count_so_far
        for i, example in enumerate(examples):
            heartbeat.maybe_log(
                completed_examples=i,
                error_count=nonstaged_errors,
                current_example=i + 1,
            )
            if log_verbose is not None and i % 10 == 0:
                log_verbose(f"[runner] {i}/{len(examples)} ...")
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
                    wandb_logger=wandb_logger,
                )
                heartbeat.maybe_log(
                    completed_examples=i + 1,
                    error_count=nonstaged_errors,
                    current_example=i + 1,
                )

            except Exception as e:
                error_msg = traceback.format_exc()
                if log_verbose is not None:
                    log_verbose(f"[runner] ERROR on {example.example_id}: {e}")
                store.save_error(run_id, example.example_id, error_msg)
                nonstaged_errors += 1
                wandb_logger.record_error(example_id=example.example_id, error=error_msg)
                heartbeat.maybe_log(
                    completed_examples=i + 1,
                    error_count=nonstaged_errors,
                    current_example=i + 1,
                )

    n = len(examples)
    all_j = store.load_judgements(run_id)
    all_metrics = store.load_metrics(run_id)
    error_count = len(store.load_errors(run_id))
    summary = _build_run_summary(
        run_id=run_id,
        config=config,
        n_examples=n,
        judgements=all_j,
        metrics_rows=all_metrics,
        error_count=error_count,
        output_dir=str(store.run_dir(run_id)),
    )
    store.save_summary(run_id, summary)

    _stdout_log(f"[runner] done. accuracy={summary['accuracy']:.3f} run_id={run_id}")

    # Optional vLLM timing replay
    if config.timing_replay and config.timing_replay.enabled:
        _run_timing_replay(run_id, config, store, verbose)

    wandb_logger.finish(summary=summary, run_dir=store.run_dir(run_id))

    return run_id


def _run_timing_replay(run_id, config, store, verbose):
    from quantlab.timing.replay import VLLMTimingReplay, replay_backend_spec_from_actor

    if verbose:
        _stdout_log("[runner] running vLLM timing replay ...")

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
            _stdout_log(f"[runner] timing replay fixed model_id={rc.model_id!r}")
    else:
        replay = VLLMTimingReplay(actor_backend_specs=specs_by_actor)
        if verbose:
            desc = ", ".join(f"{k}→{v.model_id}" for k, v in sorted(specs_by_actor.items()))
            _stdout_log(f"[runner] timing replay per segment.actor_id: {desc}")

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
