from __future__ import annotations

import re
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample
from quantlab.benchmarks.boxed import extract_last_boxed

_SYSTEM_PROMPT = (
    "You are an expert scientist. Answer the following multiple-choice question. "
    "Think carefully step by step. "
    "Give your final answer as a single letter (A, B, C, or D) on the last line."
)

_LAST_LINE_RE = re.compile(r"^([ABCD])[).\s]?$", re.IGNORECASE)
_LAST_LETTER_RE = re.compile(r"\b([ABCD])\b(?!.*\b[ABCD]\b)", re.DOTALL | re.IGNORECASE)


def _normalize_choice(raw: str) -> Optional[str]:
    letter = raw.strip().upper()
    return letter if letter in {"A", "B", "C", "D"} else None


def extract_mcq_letter(generated_text: str) -> Optional[str]:
    """Extract A/B/C/D from model output (boxed, last line, or last letter in text)."""
    boxed = extract_last_boxed(generated_text)
    if boxed is not None:
        choice = _normalize_choice(boxed)
        if choice is not None:
            return choice

    last_line = generated_text.strip().split("\n")[-1].strip()
    m = _LAST_LINE_RE.match(last_line)
    if m:
        return m.group(1).upper()

    m = _LAST_LETTER_RE.search(generated_text)
    if m:
        return m.group(1).upper()
    return None


class GPQADiamondAdapter(BenchmarkAdapter):
    """GPQA Diamond: https://huggingface.co/datasets/fingertap/GPQA-Diamond (198 rows, ``test``)."""

    name = "gpqa_diamond"

    def load(
        self,
        split: str = "test",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import load_dataset

        if subset:
            raise ValueError("gpqa_diamond has no subsets; omit benchmark.subset")

        ds = load_dataset("fingertap/GPQA-Diamond", split=split)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for i, row in enumerate(ds):
            examples.append(
                BenchmarkExample(
                    example_id=f"gpqa_diamond_{i:03d}",
                    prompt=self.build_prompt(row),
                    ground_truth=str(row["answer"]).strip().upper(),
                    raw=dict(row),
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        return (
            "<|im_start|>system\n"
            f"{_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{raw['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
        )

    def extract_answer(self, generated_text: str) -> Optional[str]:
        return extract_mcq_letter(generated_text)

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        pred = _normalize_choice(predicted)
        ref = _normalize_choice(ground_truth)
        return pred is not None and ref is not None and pred == ref
