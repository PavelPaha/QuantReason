from __future__ import annotations

import re
from math import isclose
from typing import Any, Optional

from quantlab.benchmarks.base import BenchmarkAdapter, BenchmarkExample
from quantlab.benchmarks.boxed import extract_last_boxed


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                post = substr[2:] if len(substr) > 2 else ""
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _strip_string(string: str) -> str:
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("bmatrix", "pmatrix")
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("\\{", "{").replace("\\}", "}")
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = re.sub(r"\\mbox\{.*?\}", "", string)
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("and", "").replace("\\mathbf", "")
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    return string


def _last_number(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d*\.?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def _parse_digits(num: str) -> Optional[float]:
    num = num.replace(",", "")
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            try:
                return float(num[:-1]) / 100
            except Exception:
                pass
    return None


def _math_equal(pred: str, ref: str) -> bool:
    if pred.strip().lower() == ref.strip().lower():
        return True
    p_num = _parse_digits(pred)
    r_num = _parse_digits(ref)
    if p_num is not None and r_num is not None:
        if isclose(p_num, r_num, rel_tol=1e-5, abs_tol=1e-8):
            return True
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify
        if simplify(parse_latex(pred) - parse_latex(ref)) == 0:
            return True
    except Exception:
        pass
    try:
        from latex2sympy2 import latex2sympy
        from sympy import simplify
        if simplify(latex2sympy(pred) - latex2sympy(ref)) == 0:
            return True
    except Exception:
        pass
    return False


_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think carefully and solve the problem step by step. "
    "Put your final answer inside \\boxed{}."
)


class MATH500Adapter(BenchmarkAdapter):
    """Официальный MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (500 строк, ``test``)."""

    name = "math500"

    def load(
        self,
        split: str = "test",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 67,
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
        return (
            "<|im_start|>system\n"
            f"{_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{raw['problem']}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n"
        )

    def extract_answer(self, generated_text: str) -> Optional[str]:
        pred = extract_last_boxed(generated_text)
        if pred is None:
            pred = _last_number(generated_text)
        if pred is None:
            return None
        return _strip_string(pred)

    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        if predicted is None:
            return False
        gt_boxed = extract_last_boxed(ground_truth)
        gt = _strip_string(gt_boxed if gt_boxed is not None else ground_truth)
        return _math_equal(predicted, gt)
