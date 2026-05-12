from __future__ import annotations

import re
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample

_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following grade-school math problem "
    "step by step. At the end, write your final numeric answer after '#### '."
)

_FINAL_RE = re.compile(r"####\s*([\-\d,\.]+)")


def _clean_number(s: str) -> str:
    return s.replace(",", "").strip()


class GSM8KAdapter(BenchmarkAdapter):
    """Adapter for GSM8K (grade-school math, 8500 problems)."""

    name = "gsm8k"

    def load(
        self,
        split: str = "test",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=split, trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for i, row in enumerate(ds):
            gt_match = _FINAL_RE.search(row["answer"])
            gt = _clean_number(gt_match.group(1)) if gt_match else row["answer"]
            examples.append(
                BenchmarkExample(
                    example_id=f"gsm8k_{i}",
                    prompt=self.build_prompt(row),
                    ground_truth=gt,
                    raw=dict(row),
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        return f"{_SYSTEM_PROMPT}\n\nQuestion: {raw['question']}\n\nAnswer:"

    def extract_answer(self, generated_text: str) -> Optional[str]:
        matches = _FINAL_RE.findall(generated_text)
        if matches:
            return _clean_number(matches[-1])
        # Fallback: last number in the text
        nums = re.findall(r"[\-\d,\.]+", generated_text)
        if nums:
            return _clean_number(nums[-1])
        return None

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        try:
            return float(_clean_number(predicted)) == float(_clean_number(ground_truth))
        except ValueError:
            return predicted.strip() == ground_truth.strip()
