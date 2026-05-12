from __future__ import annotations

import re
from typing import Optional

from quantlab.benchmarks.base import BenchmarkAdapter
from quantlab.core.trace import Trace


def extract_from_trace(trace: Trace, adapter: BenchmarkAdapter) -> Optional[str]:
    """Extract the answer from the full generated text of a trace."""
    return adapter.extract_answer(trace.generated_text)


def find_first_candidate_token_idx(
    trace: Trace,
    adapter: BenchmarkAdapter,
) -> Optional[int]:
    """
    Return the approximate token index at which the correct answer first appeared.

    Uses a character-ratio estimate (not exact tokenization).
    """
    text = trace.generated_text
    extracted = adapter.extract_answer(text)
    if extracted is None:
        return None

    # Search for the extracted answer string in the generated text
    idx = text.find(extracted)
    if idx == -1:
        return None

    char_ratio = idx / max(len(text), 1)
    return round(trace.total_generated_tokens * char_ratio)


def find_think_close_position(text: str, close_tag: str = "</think>") -> Optional[int]:
    """Return the character offset of the close tag, or None."""
    idx = text.find(close_tag)
    return idx if idx != -1 else None
