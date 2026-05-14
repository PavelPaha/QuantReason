from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── actor config ──────────────────────────────────────────────────────────────

class GenerationParamsConfig(BaseModel):
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = Field(default_factory=list)
    seed: Optional[int] = None


class ActorDef(BaseModel):
    """Complete definition of one actor (model + backend + precision + quantization)."""

    actor_id: str
    model_id: str
    backend: str = "transformers"
    precision: str = "bf16"
    quantization: str = "none"
    quantization_config: dict[str, Any] = Field(default_factory=dict)
    generation_params: GenerationParamsConfig = Field(default_factory=GenerationParamsConfig)
    device_map: str = "auto"
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


# ── switching condition config ─────────────────────────────────────────────────

class ConditionConfig(BaseModel):
    """Name + kwargs for a registered SwitchCondition."""

    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    # When this condition fires, go to this pipeline stage index (0-based).
    # If unset, routing falls back to ``StageConfig.fallback_stage_index`` or ``stage_idx + 1``.
    target_stage_index: Optional[int] = None


# ── pipeline stage config ─────────────────────────────────────────────────────

class StageConfig(BaseModel):
    actor_id: str
    exit_conditions: list[ConditionConfig] = Field(default_factory=list)
    handoff_mode: str = "full_prefill"
    max_new_tokens: Optional[int] = None
    stop_sequences: list[str] = Field(default_factory=list)
    role: str = "unknown"
    fallback_stage_index: Optional[int] = None
    loop_back_stage_index: Optional[int] = None
    # When the actor stops without any exit_condition firing (natural completion),
    # jump to this stage index instead of ``stage_idx + 1``. Use for cyclic pipelines
    # where the same stage index is re-entered (must use ``staged_execution: false``).
    natural_next_stage_index: Optional[int] = None
    # Stage-specific instruction injected into the prompt; not stored in the trace.
    stage_prompt: str = ""


# ── benchmark config ──────────────────────────────────────────────────────────

class BenchmarkConfig(BaseModel):
    name: str
    split: str = "test"
    subset: Optional[str] = None
    max_examples: Optional[int] = None
    seed: int = 42


# ── metrics config ────────────────────────────────────────────────────────────

class MetricConfig(BaseModel):
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


# ── output / run config ───────────────────────────────────────────────────────

class OutputConfig(BaseModel):
    base_dir: str = "results"
    save_traces: bool = True
    save_raw_generations: bool = True
    save_timing: bool = True
    save_errors: bool = True
    parquet_summary: bool = True
    # Staged runs: write trace_checkpoints/wave_<n>.jsonl after each wave.
    staged_wave_checkpoints: bool = True


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "quantlab"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "experiment"
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    mode: str = "online"
    progress_log_interval: int = 10
    log_per_example_table: bool = True
    per_example_table_key: str = "per_example_metrics"
    upload_run_artifact: bool = False
    artifact_name: str = "quantlab-run"


# ── full experiment config ────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """
    Complete description of one experiment run.

    Load from YAML::

        config = ExperimentConfig.model_validate(yaml.safe_load(open("configs/my_run.yaml")))
    """

    experiment_name: str
    description: str = ""

    benchmark: BenchmarkConfig
    actors: list[ActorDef]
    pipeline: list[StageConfig]

    metrics: list[MetricConfig] = Field(
        default_factory=lambda: [
            MetricConfig(name="accuracy"),
            MetricConfig(name="parse_rate"),
            MetricConfig(name="reasoning_length"),
            MetricConfig(name="loop_detected"),
            MetricConfig(name="think_closed"),
            MetricConfig(name="commit_gap"),
            MetricConfig(name="actor_token_split"),
            MetricConfig(name="total_generation_ms"),
            MetricConfig(name="segment_timing_ms"),
        ]
    )

    output: OutputConfig = Field(default_factory=OutputConfig)
    wandb: Optional[WandbConfig] = None

    # One pipeline wave at a time (stage 0 for all examples, then stage 1, …).
    # Uses full_prefill equivalence; see PipelineExecutor.run(stop_before_stage=…).
    staged_execution: bool = False
    staged_unload_between_waves: bool = True
    # If >= 2 with staged_execution: каждая волна вызывает vLLM.generate на чанках по N промптов
    # (ThroughputMode). Сегменты и traces.jsonl совпадают с одиночным режимом; тайминги в traces
    # отражают batched-сгенерённые оценки (для точных latencies используй replay_timing).
    staged_batch_size: Optional[int] = None

    # Replay timing — if set, run vLLM replay on completed traces
    timing_replay: Optional[TimingReplayConfig] = None


class TimingReplayConfig(BaseModel):
    enabled: bool = False
    model_id: str = Field(
        default="",
        description=(
            "If set, replay every segment through this HF model id (single backend). "
            "If empty, each trace segment selects model/precision/quant/GPU knobs from "
            "the actor with matching actor_id."
        ),
    )
    precision: str = "bf16"
    quantization: Optional[str] = None
    tensor_parallel_size: int = 1
    segments_to_replay: Optional[list[int]] = None
