from __future__ import annotations

import re
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample
from quantlab.benchmarks.boxed import extract_last_boxed

_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think carefully and solve the problem step by step. "
    "Put your final answer inside \\boxed{}."
)

_INT_RE = re.compile(r"^-?\d+")


def _normalize_aime_int(raw: str) -> Optional[str]:
    s = raw.strip().replace(",", "").replace(" ", "")
    if not s:
        return None
    try:
        return str(int(s))
    except ValueError:
        m = _INT_RE.match(s)
        if m is None:
            return None
        return str(int(m.group(0)))


class AIME26Adapter(BenchmarkAdapter):
    """MathArena AIME 2026: https://huggingface.co/datasets/MathArena/aime_2026 (30 rows, ``train``)."""

    name = "aime26"

    def load(
        self,
        split: str = "train",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        from datasets import load_dataset

        if subset:
            raise ValueError("aime26 has no subsets; omit benchmark.subset")

        ds = load_dataset("MathArena/aime_2026", split=split)
        ds = ds.shuffle(seed=seed)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        examples = []
        for row in ds:
            idx = int(row["problem_idx"])
            answer = int(row["answer"])
            examples.append(
                BenchmarkExample(
                    example_id=f"aime26_{idx:02d}",
                    prompt=self.build_prompt(row),
                    ground_truth=str(answer),
                    raw=dict(row),
                    metadata={"problem_idx": idx},
                )
            )
        return examples

    def build_prompt(self, raw: dict[str, Any]) -> str:
        return (
            "<|im_start|>system\n"
            f"{_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{raw['problem']}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
        )

    def extract_answer(self, generated_text: str) -> Optional[str]:
        boxed = extract_last_boxed(generated_text)
        if boxed is None:
            return None
        return _normalize_aime_int(boxed)

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        pred = _normalize_aime_int(predicted)
        ref = _normalize_aime_int(ground_truth)
        return pred is not None and ref is not None and pred == ref
