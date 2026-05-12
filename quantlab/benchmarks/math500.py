from __future__ import annotations

import re
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample

_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think carefully and solve the problem step by step. "
    "Put your final answer inside \\boxed{}."
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _normalize_math(s: str) -> str:
    s = s.strip().replace(" ", "")
    # strip leading zeros from integers
    s = re.sub(r"\b0+(\d)", r"\1", s)
    return s


class MATH500Adapter(BenchmarkAdapter):
    """Официальный MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (500 строк, ``test``)."""

    name = "math500"

    def load(
        self,
        split: str = "test",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
        if subset:
            ds = ds.filter(lambda x: x["subject"] == subset)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for row in ds:
            uid = str(row["unique_id"]).replace("/", "_").removesuffix(".json")
            examples.append(
                BenchmarkExample(
                    example_id=f"math500_{uid}",
                    prompt=self.build_prompt(row),
                    ground_truth=row["solution"],
                    raw=dict(row),
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        # Qwen3 chat template; ends with open <think>.
        # Typical hybrid pipeline: FP plans inside think, then GPTQ continues (think + \\boxed{} after close).
        return (
            "<|im_start|>system\n"
            f"{_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{raw['problem']}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
        )

    def extract_answer(self, generated_text: str) -> Optional[str]:
        matches = _BOXED_RE.findall(generated_text)
        if matches:
            return matches[-1].strip()
        return None

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        gt_match = _BOXED_RE.search(ground_truth)
        gt = gt_match.group(1) if gt_match else ground_truth
        return _normalize_math(predicted) == _normalize_math(gt)
