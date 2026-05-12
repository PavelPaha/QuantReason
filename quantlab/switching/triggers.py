from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from quantlab.core.trace import Trace, TraceSegment
from quantlab.switching.base import SwitchCondition, SwitchDecision


class LoopDetector(SwitchCondition):
    """
    Switch when the same n-gram appears too many times within the generated text.

    Detects degenerate repetition loops where the model keeps emitting the
    same phrase or sentence.
    """

    name = "loop_detector"

    def __init__(self, ngram_size: int = 5, max_repeats: int = 4) -> None:
        self.ngram_size = ngram_size
        self.max_repeats = max_repeats

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        text = trace.generated_text
        words = text.split()
        if len(words) < self.ngram_size:
            return SwitchDecision(should_switch=False)

        counts: Counter = Counter()
        for i in range(len(words) - self.ngram_size + 1):
            ng = tuple(words[i: i + self.ngram_size])
            counts[ng] += 1

        worst_ng, worst_count = counts.most_common(1)[0]
        if worst_count >= self.max_repeats:
            return SwitchDecision(
                should_switch=True,
                reason=f"loop detected: {worst_ng!r} repeated {worst_count}x",
                metadata={"ngram": " ".join(worst_ng), "count": worst_count},
            )
        return SwitchDecision(should_switch=False)


class ThinkBlockNotClosed(SwitchCondition):
    """
    Switch when the reasoning block was never properly closed.

    Useful as a fallback trigger: if the main reasoning actor never emits
    ``</think>`` (or another close tag), hand off to the full-precision actor.
    """

    name = "think_not_closed"

    def __init__(
        self,
        open_tag: str = "<think>",
        close_tag: str = "</think>",
        min_tokens: int = 100,
    ) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self.min_tokens = min_tokens

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        if trace.total_generated_tokens < self.min_tokens:
            return SwitchDecision(should_switch=False)

        text = trace.generated_text
        if self.open_tag in text and self.close_tag not in text:
            return SwitchDecision(
                should_switch=True,
                reason=f"open tag {self.open_tag!r} present but close tag {self.close_tag!r} absent",
            )
        return SwitchDecision(should_switch=False)


class CandidateAnswerAppeared(SwitchCondition):
    """
    Switch when a candidate answer pattern is found in the generated text.

    Example usage: detect 'The answer is X' and switch to the verification actor.
    """

    name = "candidate_answer_appeared"

    def __init__(self, patterns: Optional[list[str]] = None) -> None:
        default_patterns = [
            r"the answer is\b",
            r"\\boxed\{",
            r"####\s*\d",
            r"therefore,?\s+the answer",
        ]
        self._patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or default_patterns)]

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        text = trace.generated_text
        for pat in self._patterns:
            m = pat.search(text)
            if m:
                return SwitchDecision(
                    should_switch=True,
                    reason=f"candidate answer pattern matched: {pat.pattern!r}",
                    metadata={"match_start": m.start()},
                )
        return SwitchDecision(should_switch=False)


class LongCommitGap(SwitchCondition):
    """
    Switch when the model produced a candidate answer but has not committed to
    it after generating many more tokens (commit gap too large).
    """

    name = "long_commit_gap"

    def __init__(
        self,
        candidate_pattern: str = r"\\boxed\{",
        max_gap_tokens: int = 512,
    ) -> None:
        self._pat = re.compile(candidate_pattern)
        self.max_gap_tokens = max_gap_tokens
        self._candidate_token_idx: Optional[int] = None

    def reset(self) -> None:
        self._candidate_token_idx = None

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        text = trace.generated_text
        if self._candidate_token_idx is None:
            m = self._pat.search(text)
            if m:
                # Rough estimate: character offset → token index
                char_ratio = m.start() / max(len(text), 1)
                self._candidate_token_idx = round(trace.total_generated_tokens * char_ratio)
        else:
            gap = trace.total_generated_tokens - self._candidate_token_idx
            if gap >= self.max_gap_tokens:
                return SwitchDecision(
                    should_switch=True,
                    reason=f"commit gap {gap} >= {self.max_gap_tokens}",
                    metadata={"candidate_token_idx": self._candidate_token_idx, "gap": gap},
                )
        return SwitchDecision(should_switch=False)


class MaxTokensPerStage(SwitchCondition):
    """Hard token budget for a single pipeline stage."""

    name = "max_tokens_per_stage"

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self._stage_start_tokens: Optional[int] = None

    def reset(self) -> None:
        self._stage_start_tokens = None

    def evaluate(self, trace: Trace, new_segment: TraceSegment) -> SwitchDecision:
        if self._stage_start_tokens is None:
            self._stage_start_tokens = trace.total_generated_tokens - new_segment.token_count

        used = trace.total_generated_tokens - self._stage_start_tokens
        if used >= self.max_tokens:
            return SwitchDecision(
                should_switch=True,
                reason=f"stage token budget exhausted: {used} >= {self.max_tokens}",
            )
        return SwitchDecision(should_switch=False)
