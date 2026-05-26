#!/usr/bin/env python3
"""Dry-run staged pipeline: show llm_prompt_full per stage on one MATH-500 example."""
from __future__ import annotations

import yaml

from quantlab.actors.base import ActorBase, ActorConfig
from quantlab.benchmarks.registry import BenchmarkRegistry
from quantlab.config.schema import ExperimentConfig
from quantlab.core.trace import TraceSegment
from quantlab.core.types import SegmentRole, StagePromptPlacement
from quantlab.pipeline.executor import PipelineExecutor
from quantlab.runner import _build_stage


class EchoActor(ActorBase):
    """Returns canned text; records the exact prompt passed to generate."""

    CANNED = {
        SegmentRole.PLAN: (
            "1. Identify the geometric setup and label key points.\n"
            "2. Apply the relevant theorem or formula.\n"
            "3. Simplify and verify units.\n"
            "[PLAN_FINISH]"
        ),
        SegmentRole.REASONING: (
            "Let me work through the plan step by step.\n"
            "Using step 2, we get the intermediate result.\n"
            "Therefore the answer is \\boxed{42}."
        ),
    }

    def __init__(self, actor_id: str, role: SegmentRole) -> None:
        super().__init__(ActorConfig(actor_id=actor_id, model_id="echo", backend="transformers"))
        self.role = role
        self.last_llm_prompt: str = ""

    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def generate(self, trace, handoff_mode, kv_state=None, **kwargs):
        self.last_llm_prompt = trace.build_llm_prompt(
            handoff_mode,
            stage_prompt=kwargs.get("prompt_suffix", ""),
            stage_prompt_placement=kwargs.get(
                "stage_prompt_placement", StagePromptPlacement.ASSISTANT_SUFFIX
            ),
            stage_system_prompt=kwargs.get("stage_system_prompt", ""),
            plan_label=kwargs.get("handoff_plan_label", ""),
        )
        text = self.CANNED.get(kwargs.get("role", self.role), "(echo)")
        seg = TraceSegment(
            actor_id=self.actor_id,
            text=text,
            token_count=len(text.split()),
            start_token_idx=trace.next_token_idx,
            role=kwargs.get("role", self.role),
        )
        return seg, None


def main() -> None:
    cfg_path = "configs/math500/qwen32b_gptq2bit/hybrid_fp16_gptq2bit.yaml"
    config = ExperimentConfig.model_validate(yaml.safe_load(open(cfg_path)))
    adapter = BenchmarkRegistry.build(config.benchmark.name, benchmark=config.benchmark)
    example = adapter.load(split="test", max_examples=1, seed=42)[0]

    stages = [_build_stage(s) for s in config.pipeline]
    actors = {
        "qwen3_32b_fp16_plan": EchoActor("qwen3_32b_fp16_plan", SegmentRole.PLAN),
        "qwen3_32b_gptq2bit": EchoActor("qwen3_32b_gptq2bit", SegmentRole.REASONING),
    }
    ex = PipelineExecutor(
        stages=stages,
        actors=actors,
        max_total_tokens=8192,
        trace_include_llm_prompt=True,
    )

    print("=" * 72)
    print(f"EXAMPLE: {example.example_id}")
    print("=" * 72)
    print("\n── benchmark prompt (trace.prompt) ──\n")
    print(example.prompt)

    t1 = ex.run(example.example_id, example.prompt, stop_before_stage=1)
    print("\n" + "=" * 72)
    print("STAGE 0 — PLAN (qwen3_32b_fp16)")
    print("=" * 72)
    print(f"\nstage_prompt_sent in trace: {t1.segments[0].stage_prompt_sent!r}")
    print("\n── llm_prompt_full ──\n")
    print(t1.segments[0].llm_prompt_full)
    print("\n── generated plan ──\n")
    print(t1.segments[0].text)

    t2 = ex.continue_run(t1)
    print("\n" + "=" * 72)
    print("STAGE 1 — REASONING (qwen3_32b_gptq2bit)")
    print("=" * 72)
    print(f"\nhandoff_mode: {t2.segments[1].metadata.get('handoff_mode')}")
    print(f"handoff_plan_label: {t2.segments[1].metadata.get('handoff_plan_label')!r}")
    print(f"\nstage_prompt_sent in trace: {t2.segments[1].stage_prompt_sent!r}")
    print("\n── llm_prompt_full ──\n")
    print(t2.segments[1].llm_prompt_full)
    print("\n── generated reasoning ──\n")
    print(t2.segments[1].text)

    rp = t2.segments[1].llm_prompt_full or ""
    checks = [
        ("задача (user block) видна", example.raw["problem"][:40] in rp),
        ("Plan: label виден", "Plan:\n" in rp),
        ("plan stage_system НЕ виден", "Produce only a concise solution plan" not in rp),
        ("старый think из prompt убран", rp.count("<think>") == 1),
        ("инструкция про boxed видна", "\\boxed{}" in rp),
    ]
    print("\n" + "=" * 72)
    print("CHECKS")
    print("=" * 72)
    for name, ok in checks:
        print(f"  {'OK' if ok else 'FAIL'} {name}")


if __name__ == "__main__":
    main()
