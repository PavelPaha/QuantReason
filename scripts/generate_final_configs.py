#!/usr/bin/env python3
"""Generate configs/*.yaml — 4 configs per dataset."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "configs"

GPU = "0"
TP = 1
CTX_LEN = 32768
GEN_MAX = 30720  # max_new_tokens for answer/reason (room for prompt in 32k window)
PLAN_GEN = 1024
BATCH_SIZE = 500

METRICS_MATH_FULL = """metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: loop_detected
  - name: loop_onset_tokens
  - name: think_closed
  - name: commit_gap
  - name: tokens_to_first_correct
  - name: finish_commit
  - name: verification_spiral
  - name: stop_token_probe
  - name: actor_token_split
  - name: total_generation_ms
  - name: segment_timing_ms
  - name: tokens_per_second"""

METRICS_GPQA_SINGLE = """metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second"""

OUTPUT = """output:
  base_dir: results
  save_traces: true
  save_raw_generations: true
  save_timing: true
  save_errors: true
  parquet_summary: true
  trace_include_llm_prompt: true
  staged_wave_checkpoints: true"""

OUTPUT_NO_CHECKPOINTS = OUTPUT.replace("  staged_wave_checkpoints: true\n", "")


def actor_fp16(actor_id: str, max_new_tokens: int, max_model_len: int, util: float = 0.92) -> str:
    return f"""  - actor_id: {actor_id}
    model_id: Qwen/Qwen3-32B
    backend: vllm
    precision: fp16
    quantization: none
    backend_kwargs:
      cuda_visible_devices: "{GPU}"
      tensor_parallel_size: {TP}
      gpu_memory_utilization: {util}
      max_model_len: {max_model_len}
      enable_prefix_caching: true
      enforce_eager: false
    generation_params:
      max_new_tokens: {max_new_tokens}
      temperature: 0.6
      seed: 42
      stop_sequences: ["<|im_end|>", "<|endoftext|>"]"""


def actor_gptq(actor_id: str, max_new_tokens: int, max_model_len: int, util: float = 0.92) -> str:
    return f"""  - actor_id: {actor_id}
    model_id: kaitchup/Qwen3-32B-autoround-2bit-gptq
    backend: vllm
    precision: fp16
    quantization: gptq
    quantization_config:
      bits: 2
    backend_kwargs:
      cuda_visible_devices: "{GPU}"
      tensor_parallel_size: {TP}
      gpu_memory_utilization: {util}
      max_model_len: {max_model_len}
      enable_prefix_caching: true
      enforce_eager: false
    generation_params:
      max_new_tokens: {max_new_tokens}
      temperature: 0.6
      seed: 42
      stop_sequences: ["<|im_end|>", "<|endoftext|>"]"""


PLAN_STAGE_MATH = """  - actor_id: qwen3_32b_fp16_plan
    role: plan
    handoff_mode: prompt_without_think
    stage_prompt_placement: plan_scaffold
    max_new_tokens: 1024
    exclude_stage_prompt_from_trace: true
    stop_sequences:
      - "[PLAN_FINISH]"
      - "PLAN_FINISH"
    stage_system_prompt: |-
      You are a careful reasoning assistant. Produce only a concise solution plan.
      Do not solve the problem fully. Write only a concise plan with 3-6 short bullet points or numbered steps.
      Do not give the final answer, do not use \\boxed{}, and stop immediately after the plan.
    stage_prompt: |-
      Return only the plan. No final answer.
    exit_conditions:
      - name: after_marker
        kwargs: { marker: "[PLAN_FINISH]", keep_marker: true }
      - name: after_n_tokens
        kwargs: { n: 1024 }"""

# Single-stage: open think in assistant (same as hybrid reason).
SINGLE_STAGE = """  - actor_id: {aid}
    role: answer
    handoff_mode: full_prefill
    max_new_tokens: {mnt}
    stage_prompt: ""
    stage_assistant_suffix: "<think>\\n"
    stop_sequences:
      - "<|im_end|>"
      - "<|endoftext|>"
    exit_conditions: []"""

PLAN_STAGE_GPQA = PLAN_STAGE_MATH.replace(
    "Do not give the final answer, do not use \\boxed{}, and stop immediately after the plan.",
    "Do not give the final answer or letter choice (A/B/C/D), and stop immediately after the plan.",
)

REASON_BOXED = """  - actor_id: {actor_id}
    role: reasoning
    handoff_mode: prompt_plan_in_user
    handoff_plan_label: "Plan:\\n"
    stage_prompt_placement: user_suffix
    max_new_tokens: {gen_max}
    stage_prompt: |-
      Solve the problem using the plan above and give the final answer in \\boxed{{}}.
    stage_assistant_suffix: "<think>\\n"
    stop_sequences:
      - "<|im_end|>"
      - "<|endoftext|>"
    exit_conditions: []"""

REASON_GPQA = REASON_BOXED.replace(
    "give the final answer in \\boxed{}.",
    "end with a single line containing only the letter A, B, C, or D.",
)

PLAN_STAGE_WINOGRANDE = PLAN_STAGE_MATH.replace(
    "Do not give the final answer, do not use \\boxed{}, and stop immediately after the plan.",
    "Do not give the final answer or option number (1/2), and stop immediately after the plan.",
)

REASON_WINOGRANDE = REASON_BOXED.replace(
    "give the final answer in \\boxed{{}}.",
    "end with a single line containing only 1 or 2.",
)

PLAN_STAGE_STRATEGYQA = PLAN_STAGE_MATH.replace(
    "Do not give the final answer, do not use \\boxed{}, and stop immediately after the plan.",
    "Do not give the final yes/no answer, and stop immediately after the plan.",
)

REASON_STRATEGYQA = REASON_BOXED.replace(
    "give the final answer in \\boxed{{}}.",
    "end with a single line containing only yes or no.",
)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT.parents[1])}")


def _variant_subdir(variant: str) -> str:
    return "qwen32b_gptq2bit" if "gptq2bit" in variant else "qwen32b_fp16"


def gsm8k_configs() -> None:
    bench = """benchmark:
  name: gsm8k
  split: test
  max_examples: null
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "gsm8k/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_gsm8k_single_fp16
description: >
  GSM8K test (1319): FP16 single-stage. Problem + solve/\\\\boxed in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "gsm8k/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_gsm8k_single_gptq2bit
description: >
  GSM8K test (1319): GPTQ 2-bit single-stage. Same layout as FP16 single (open <|redacted_thinking>).

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_gsm8k_{name}",
                "description: >",
                "  GSM8K test (1319): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_MATH,
                REASON_BOXED.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"gsm8k/{_variant_subdir(name)}/{name}.yaml", body)


def math500_configs() -> None:
    bench = """benchmark:
  name: math500
  split: test
  max_examples: 500
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "math500/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_math500_single_fp16
description: >
  MATH-500 (500): FP16 single-stage. Problem + solve/\\\\boxed in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "math500/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_math500_single_gptq2bit
description: >
  MATH-500 (500): GPTQ 2-bit single-stage. Same layout as FP16 single (open <|redacted_thinking>).

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_math500_{name}",
                "description: >",
                "  MATH-500 (500): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_MATH,
                REASON_BOXED.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"math500/{_variant_subdir(name)}/{name}.yaml", body)


def aime2026_configs() -> None:
    batch = 30
    bench = """benchmark:
  name: aime2026
  split: train
  max_examples: 30
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {batch}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {batch}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "aime2026/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_aime2026_single_fp16
description: >
  AIME 2026 (30): FP16 single-stage. Integer 0-999 + \\\\boxed in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "aime2026/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_aime2026_single_gptq2bit
description: >
  AIME 2026 (30): GPTQ 2-bit single-stage. Same layout as FP16 single.

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_aime2026_{name}",
                "description: >",
                "  AIME 2026 (30): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_MATH,
                REASON_BOXED.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"aime2026/{_variant_subdir(name)}/{name}.yaml", body)


def arc_easy_configs() -> None:
    bench = """benchmark:
  name: arc_easy
  split: test
  max_examples: null
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "arc_easy/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_arc_easy_single_fp16
description: >
  ARC-Easy test (2376): FP16 single-stage MCQ. Question + choices in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

{METRICS_GPQA_SINGLE}

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "arc_easy/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_arc_easy_single_gptq2bit
description: >
  ARC-Easy test (2376): GPTQ 2-bit single-stage MCQ. Same layout as FP16 single (open <|redacted_thinking>).

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

{METRICS_GPQA_SINGLE}

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_arc_easy_{name}",
                "description: >",
                "  ARC-Easy test (2376): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_GPQA,
                REASON_GPQA.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"arc_easy/{_variant_subdir(name)}/{name}.yaml", body)


def arc_challenge_configs() -> None:
    bench = """benchmark:
  name: arc_challenge
  split: test
  max_examples: null
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "arc_challenge/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_arc_challenge_single_fp16
description: >
  ARC-Challenge test (1172): FP16 single-stage MCQ. Question + choices in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

{METRICS_GPQA_SINGLE}

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "arc_challenge/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_arc_challenge_single_gptq2bit
description: >
  ARC-Challenge test (1172): GPTQ 2-bit single-stage MCQ. Same layout as FP16 single (open <|redacted_thinking>).

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

{METRICS_GPQA_SINGLE}

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_arc_challenge_{name}",
                "description: >",
                "  ARC-Challenge test (1172): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_GPQA,
                REASON_GPQA.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"arc_challenge/{_variant_subdir(name)}/{name}.yaml", body)


def gpqa_configs() -> None:
    bench = """benchmark:
  name: gpqa
  subset: gpqa_diamond
  split: train
  max_examples: null
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "gpqa_diamond/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_gpqa_diamond_single_fp16
description: >
  GPQA Diamond (198): FP16 single-stage MCQ. Default TP=1 on GPU {GPU};
  use TP=2 if 32k OOM (e.g. cuda_visible_devices \"0,1\", tensor_parallel_size: 2).

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

{METRICS_GPQA_SINGLE}

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "gpqa_diamond/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_gpqa_diamond_single_gptq2bit
description: >
  GPQA Diamond (198): GPTQ 2-bit single-stage MCQ.

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

{METRICS_GPQA_SINGLE}

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_gpqa_diamond_{name}",
                "description: >",
                "  GPQA Diamond (198): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_GPQA,
                REASON_GPQA.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"gpqa_diamond/{_variant_subdir(name)}/{name}.yaml", body)


def winogrande_configs() -> None:
    bench = """benchmark:
  name: winogrande
  subset: winogrande_xl
  split: validation
  max_examples: null
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "winogrande/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_winogrande_single_fp16
description: >
  WinoGrande XL validation (1267): FP16 single-stage. lm-eval choice prompt in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "winogrande/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_winogrande_single_gptq2bit
description: >
  WinoGrande XL validation (1267): GPTQ 2-bit single-stage. Same prompt layout as FP16 single.

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_winogrande_{name}",
                "description: >",
                "  WinoGrande XL validation (1267): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_WINOGRANDE,
                REASON_WINOGRANDE.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"winogrande/{_variant_subdir(name)}/{name}.yaml", body)


def strategyqa_configs() -> None:
    bench = """benchmark:
  name: strategyqa
  split: test
  max_examples: null
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "strategyqa/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_strategyqa_single_fp16
description: >
  StrategyQA test (687): FP16 single-stage yes/no. Question in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "strategyqa/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_strategyqa_single_gptq2bit
description: >
  StrategyQA test (687): GPTQ 2-bit single-stage yes/no. Same prompt layout as FP16 single.

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_strategyqa_{name}",
                "description: >",
                "  StrategyQA test (687): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_STRATEGYQA,
                REASON_STRATEGYQA.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"strategyqa/{_variant_subdir(name)}/{name}.yaml", body)


def piqa_configs() -> None:
    bench = """benchmark:
  name: piqa
  split: validation
  max_examples: null
  seed: 42"""

    staged_single = f"""staged_execution: true
staged_unload_between_waves: false
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    staged_hybrid = f"""staged_execution: true
staged_unload_between_waves: true
staged_batch_size: {BATCH_SIZE}
max_total_tokens: {CTX_LEN}"""

    write(
        ROOT / "piqa/qwen32b_fp16/single_fp16.yaml",
        f"""experiment_name: final_piqa_single_fp16
description: >
  PIQA validation (1838): FP16 single-stage. lm-eval Question + two solutions in user; assistant opens <|redacted_thinking>.

{bench}

{staged_single}

actors:
{actor_fp16("qwen3_32b_fp16", GEN_MAX, CTX_LEN, 0.88)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_fp16", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    write(
        ROOT / "piqa/qwen32b_gptq2bit/single_gptq2bit.yaml",
        f"""experiment_name: final_piqa_single_gptq2bit
description: >
  PIQA validation (1838): GPTQ 2-bit single-stage. Same layout as FP16 single (open <|redacted_thinking>).

{bench}

{staged_single}

actors:
{actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)}

pipeline:
{SINGLE_STAGE.format(aid="qwen3_32b_gptq2bit", mnt=GEN_MAX)}

metrics:
  - name: accuracy
  - name: parse_rate
  - name: exact_match
  - name: reasoning_length
  - name: think_closed
  - name: total_generation_ms
  - name: tokens_per_second

{OUTPUT_NO_CHECKPOINTS}""",
    )

    for name, reason_actor, reason_block in [
        ("hybrid_fp16_fp16", "qwen3_32b_fp16_reason", actor_fp16("qwen3_32b_fp16_reason", GEN_MAX, CTX_LEN, 0.92)),
        ("hybrid_fp16_gptq2bit", "qwen3_32b_gptq2bit", actor_gptq("qwen3_32b_gptq2bit", GEN_MAX, CTX_LEN)),
    ]:
        body = "\n".join(
            [
                f"experiment_name: final_piqa_{name}",
                "description: >",
                "  PIQA validation (1838): FP16 plan → reasoning. Plan in user (prompt_plan_in_user).",
                "",
                bench,
                "",
                staged_hybrid,
                "",
                "actors:",
                actor_fp16("qwen3_32b_fp16_plan", PLAN_GEN, CTX_LEN, 0.90),
                reason_block,
                "",
                "pipeline:",
                PLAN_STAGE_WINOGRANDE,
                REASON_WINOGRANDE.format(actor_id=reason_actor, gen_max=GEN_MAX),
                "",
                METRICS_MATH_FULL,
                "",
                OUTPUT,
            ]
        )
        write(ROOT / f"piqa/{_variant_subdir(name)}/{name}.yaml", body)


def main() -> None:
    gsm8k_configs()
    math500_configs()
    aime2026_configs()
    arc_easy_configs()
    arc_challenge_configs()
    gpqa_configs()
    winogrande_configs()
    strategyqa_configs()
    piqa_configs()
    print(f"\nDone: 36 configs under {ROOT}")


if __name__ == "__main__":
    main()
