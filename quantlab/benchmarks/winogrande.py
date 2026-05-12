from __future__ import annotations

import re
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample

_SYSTEM_PROMPT = (
    "Fill in the blank in the sentence below. Choose either option 1 or option 2. "
    "Reply with only the number 1 or 2."
)


class WinoGrandeAdapter(BenchmarkAdapter):
    """Adapter for WinoGrande commonsense reasoning benchmark."""

    name = "winogrande"

    def load(
        self,
        split: str = "validation",
        subset: Optional[str] = "winogrande_xl",
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("winogrande", subset or "winogrande_xl", split=split, trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for i, row in enumerate(ds):
            examples.append(
                BenchmarkExample(
                    example_id=f"winogrande_{i}",
                    prompt=self.build_prompt(row),
                    ground_truth=str(row["answer"]),
                    raw=dict(row),
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        sentence = raw["sentence"]
        opt1 = raw["option1"]
        opt2 = raw["option2"]
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Sentence: {sentence}\n"
            f"Option 1: {opt1}\n"
            f"Option 2: {opt2}\n\n"
            f"Answer:"
        )

    def extract_answer(self, generated_text: str) -> Optional[str]:
        text = generated_text.strip()
        m = re.search(r"\b([12])\b", text)
        if m:
            return m.group(1)
        return None

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        return predicted.strip() == ground_truth.strip()
