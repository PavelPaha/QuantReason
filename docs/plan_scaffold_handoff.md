# Plan scaffold → reasoning handoff

Двухстадийный режим: **plan генерируется отдельно** (свой system + scaffold), затем **подкладывается в reasoning** через assistant prefix `Plan:\n…`.

Отличие от `math500_qwen_hybrid.yaml`: там обе стадии на `full_prefill` — plan пишется внутри того же MATH chat template с открытым `<think>`.

## Стадия 0 — plan (`plan_scaffold`)

```
<|im_start|>system
{stage_system_prompt}
<|im_start|>user
Problem:
{задача из benchmark}
{stage_prompt}
<|im_start|>assistant
<think>

</think>


→ модель генерирует только план
```

YAML:

```yaml
stage_prompt_placement: plan_scaffold
stage_system_prompt: | ...
stage_prompt: | ...
exclude_stage_prompt_from_trace: true
```

## Стадия 1 — reasoning (`prompt_plan_labeled`)

```
<|im_start|>system
{MATH system из benchmark}
<|im_start|>user
{задача как в MATH-500}
<|im_start|>assistant
Plan:
{текст плана из wave 0}

{solve instruction + <think>}

→ GPTQ генерирует reasoning + \boxed{}
```

YAML:

```yaml
handoff_mode: prompt_plan_labeled
handoff_plan_label: "Plan:\n"
stage_prompt: | ...
```

## Быстрая проверка без GPU

```bash
python scripts/dry_run_staged_prompts.py
```

## Запуск smoke (1 пример)

```bash
python scripts/run_experiment.py configs/experiments/_smoke_hidden_plan_prompt_n1_gpu1.yaml --staged -v
```

## Прогон n=50 (как hybrid baseline)

```bash
python scripts/run_experiment.py configs/experiments/math500_fp16_plan_scaffold_gptq2bit_brief_baseline.yaml --staged -v
```

Промпты смотреть в `traces.jsonl` → `segments[].llm_prompt_full` (нужен `trace_include_llm_prompt: true`).

## Handoff modes (справочник)

| mode | prefix перед `stage_prompt` |
|------|-----------------------------|
| `full_prefill` | `trace.prompt + segments` (старый hybrid) |
| `prompt_plan_labeled` | MATH prompt без think + `Plan:\n` + plan text |
| `prompt_without_think` | prompt без `<think>` |
| `segments_only` | только segments |
| `plan_scaffold` | отдельный scaffold (только plan stage) |
