from __future__ import annotations

import re
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample

_SYSTEM_PROMPT = (
    "You are an expert scientist. Answer the following multiple-choice question. "
    "Think step by step, then write your final answer as a single letter (A, B, C, or D) "
    "on its own line at the end."
)

_LETTER_RE = re.compile(r"\b([ABCD])\b(?!.*\b[ABCD]\b)", re.DOTALL)


class GPQAAdapter(BenchmarkAdapter):
    """Adapter for GPQA (Graduate-Level Google-Proof Q&A)."""

    name = "gpqa"

    def load(
        self,
        split: str = "train",
        subset: Optional[str] = "gpqa_diamond",
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", subset or "gpqa_diamond", split=split, trust_remote_code=True)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for i, row in enumerate(ds):
            examples.append(
                BenchmarkExample(
                    example_id=f"gpqa_{i}",
                    prompt=self.build_prompt(row),
                    ground_truth=row.get("Correct Answer", row.get("correct_answer", "")),
                    raw=dict(row),
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        choices_text = "\n".join(
            f"{letter}. {raw.get(f'Incorrect Answer {i}' if letter != 'A' else 'Correct Answer', '')}"
            for i, letter in enumerate("ABCD", 1)
        )
        # Use shuffled choices when available
        question = raw.get("Question", raw.get("question", ""))
        ca = raw.get("Correct Answer", raw.get("correct_answer", ""))
        ia1 = raw.get("Incorrect Answer 1", "")
        ia2 = raw.get("Incorrect Answer 2", "")
        ia3 = raw.get("Incorrect Answer 3", "")
        choices_text = f"A. {ca}\nB. {ia1}\nC. {ia2}\nD. {ia3}"
        return f"{_SYSTEM_PROMPT}\n\nQuestion: {question}\n\n{choices_text}\n\nAnswer:"

    def extract_answer(self, generated_text: str) -> Optional[str]:
        last_line = generated_text.strip().split("\n")[-1].strip()
        m = re.match(r"^([ABCD])[).\s]?$", last_line)
        if m:
            return m.group(1)
        m = _LETTER_RE.search(generated_text)
        if m:
            return m.group(1)
        return None

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        return predicted.strip().upper() == ground_truth.strip().upper()
