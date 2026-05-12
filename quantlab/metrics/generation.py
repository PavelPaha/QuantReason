from __future__ import annotations

import re
from collections import Counter
from quantlab.core.trace import Trace
from quantlab.evaluation.judge import JudgementResult
from quantlab.metrics.base import MetricBase

_LOOP_UNIT_SEP_SENTENCE = re.compile(r"(?:\n\s*\n+|(?<=[.!?])\s+)")

DEFAULT_GLOBAL_REPEAT_SKIP_SENTENCE_PATTERNS: tuple[str, ...] = (
    # Hesitation stubs — excluded only from global-repeat counting, not from streak-of-3.
    r"^wait,?\s+let me\b",
    r"^wait,?\s+i('?m|\s+need\s+to|\s+should)\b",
    r"^hmm+\b",
    r"^uh+\b",
    r"^um+\b",
    r"^oh wait\b",
    r"^let me (check|re-?consider|see if|verify)\b",
    r"^one moment\b",
    r"^hold on\b",
    r"^actually,?\s+wait\b",
)

_COMPILED_DEFAULT_GLOBAL_REPEAT_SKIP: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I) for p in DEFAULT_GLOBAL_REPEAT_SKIP_SENTENCE_PATTERNS
)


def _compile_global_repeat_skip_pattern_tuple(
    patterns: tuple[str, ...] | None,
) -> tuple[re.Pattern[str], ...]:
    """``None`` → встроенный список; ``()`` → фильтра выключен."""
    if patterns is None:
        return _COMPILED_DEFAULT_GLOBAL_REPEAT_SKIP
    return tuple(re.compile(p, re.I) for p in patterns)


def _norm_skipped_for_global_repeat(norm: str, compiled: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(norm) for p in compiled)


def _normalize_loop_unit(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


def _strip_span(segment: str, abs_start: int) -> tuple[int, int]:
    lstripped = segment.lstrip()
    lead = len(segment) - len(lstripped)
    stripped = lstripped.rstrip()
    tail = len(lstripped) - len(stripped)
    s = abs_start + lead
    e = abs_start + len(segment) - tail
    return s, max(s, e)


def _sentence_units(text: str, min_unit_chars: int) -> list[tuple[str, int, int]]:
    """(normalized, char_start inclusive, char_end exclusive) spans in ``text``."""
    out: list[tuple[str, int, int]] = []
    start = 0
    for m in _LOOP_UNIT_SEP_SENTENCE.finditer(text):
        seg = text[start : m.start()]
        if seg.strip():
            cs, ce = _strip_span(seg, start)
            norm = _normalize_loop_unit(text[cs:ce])
            if len(norm) >= min_unit_chars:
                out.append((norm, cs, ce))
        start = m.end()
    tail = text[start:]
    if tail.strip():
        cs, ce = _strip_span(tail, start)
        norm = _normalize_loop_unit(text[cs:ce])
        if len(norm) >= min_unit_chars:
            out.append((norm, cs, ce))
    return out


def _first_streak_end_char(units: list[tuple[str, int, int]], min_consecutive: int) -> int | None:
    """
    If the same normalized unit occurs ``min_consecutive`` times **in a row**,
    return the exclusive end offset in the original text of the last duplicate.
    Otherwise ``None``.
    """
    if min_consecutive <= 1 or len(units) < min_consecutive:
        return None
    streak = 1
    prev = units[0][0]
    for i in range(1, len(units)):
        if units[i][0] == prev:
            streak += 1
            if streak >= min_consecutive:
                return units[i][2]
        else:
            streak = 1
            prev = units[i][0]
    return None


def _sentence_global_burst_end(
    units: list[tuple[str, int, int]],
    threshold: int,
    *,
    global_skip_compiled: tuple[re.Pattern[str], ...],
) -> int | None:
    """Exclusive text offset after the occurrence that completes ``threshold`` copies."""
    if threshold <= 0 or not units:
        return None
    counts: Counter[str] = Counter()
    for norm, _cs, ce in units:
        if _norm_skipped_for_global_repeat(norm, global_skip_compiled):
            continue
        counts[norm] += 1
        if counts[norm] >= threshold:
            return ce
    return None


class ReasoningLengthMetric(MetricBase):
    """Total number of generated tokens."""

    name = "reasoning_length"

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        return trace.total_generated_tokens


class LoopDetectedMetric(MetricBase):
    """
    ``1`` when **either**, using **sentence** units only — split like
    ``_LOOP_UNIT_SEP_SENTENCE`` (blank-line blocks ``\\n\\n`` or whitespace after ``.!?``).

    **Streak:** the same normalized sentence (≥ ``min_unit_chars``) appears
    ``min_consecutive`` times **in a row**.

    **Global repeat:** one normalized sentence appears at least
    ``global_repeat_threshold`` times, **excluding** matches of
    ``global_repeat_skip_sentence_patterns`` so routine hesitation (e.g. “wait…”)
    does not falsely stack.

    Disable the global branch with ``global_repeat_threshold=None``.
    Use ``global_repeat_skip_sentence_patterns=()`` to disable the filter.
    """

    name = "loop_detected"

    def __init__(
        self,
        min_consecutive: int = 3,
        min_unit_chars: int = 15,
        *,
        global_repeat_threshold: int | None = 10,
        global_repeat_skip_sentence_patterns: tuple[str, ...] | None = None,
    ) -> None:
        self.min_consecutive = min_consecutive
        self.min_unit_chars = min_unit_chars
        self.global_repeat_threshold = global_repeat_threshold
        self._global_skip = _compile_global_repeat_skip_pattern_tuple(
            global_repeat_skip_sentence_patterns
        )

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        text = trace.generated_text
        units = _sentence_units(text, self.min_unit_chars)
        streak = _first_streak_end_char(units, self.min_consecutive) is not None

        glob = False
        thr = self.global_repeat_threshold
        if thr is not None and units:
            counts = Counter(
                u[0] for u in units if not _norm_skipped_for_global_repeat(u[0], self._global_skip)
            )
            glob = bool(counts) and counts.most_common(1)[0][1] >= thr

        return int(streak or glob)


class ThinkClosedMetric(MetricBase):
    """
    ``1`` if the reasoning wrapper was closed appropriately.

    Qwen chat runs often **open** ``<think>`` in the assembled
    *prompt* while only *streaming* ``</…>`` into segment text. Earlier logic
    only inspected ``generated_text``, so sessions that never echoed the opener
    in segments were falsely marked NA (always ``1``). We therefore treat an
    open tag anywhere in ``trace.prompt OR trace.generated_text`` as a live
    think block and require ``close_tag`` inside ``generated_text``.
    """

    name = "think_closed"

    def __init__(self, open_tag: str = "<think>", close_tag: str = "</think>") -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        gen = trace.generated_text
        if self.open_tag in gen:
            return int(self.close_tag in gen)
        if self.open_tag in trace.prompt:
            # Think started via template / prefill → closure must appear in completions.
            return int(self.close_tag in gen)
        return 1  # no think scaffolding → N/A / OK


class CommitGapMetric(MetricBase):
    """
    Number of tokens between the first candidate-answer appearance and the
    final token.  Returns -1 if no candidate answer was found.
    """

    name = "commit_gap"

    def __init__(self, candidate_pattern: str = r"\\boxed\{") -> None:
        self._pat = re.compile(candidate_pattern)

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        text = trace.generated_text
        m = self._pat.search(text)
        if m is None:
            return -1
        char_ratio = m.start() / max(len(text), 1)
        first_token_idx = round(trace.total_generated_tokens * char_ratio)
        return trace.total_generated_tokens - first_token_idx


class TokensToFirstCorrectMetric(MetricBase):
    """
    Approximate token index of the first occurrence of the correct answer
    (often reported as **TTFA** — time-to-first-correct-answer in token space).

    Uses character offset → proportional mapping to ``total_generated_tokens``;
    not exact tokenizer positions. Returns ``-1`` if the answer never appears
    or the trace is not correct.
    """

    name = "tokens_to_first_correct"

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        if not judgement.is_correct or judgement.predicted is None:
            return -1
        text = trace.generated_text
        idx = text.find(judgement.predicted)
        if idx == -1:
            return -1
        char_ratio = idx / max(len(text), 1)
        return round(trace.total_generated_tokens * char_ratio)


class FinishCommitMetric(MetricBase):
    """
    **Finish-commit proxy** (scenario 1, single-trace analogue of „finishability“).

    Without MC rollouts or logits this is binary: ``1`` iff the extracted
    ``\\boxed{}`` parses, we locate the predicted answer string strictly *before*
    the first ``\\boxed`` span, suggesting the model exposed the commitment
    before the formal wrapper. Otherwise ``0``.

    Limitations: ``str.find(predicted)`` can false-hit on incidental substrings
    / LaTeX; true *P(finish)* needs repeated sampling + engine hooks.
    """

    name = "finish_commit"

    def __init__(self, boxed_begin: str = r"\\boxed\s*\{") -> None:
        self._boxed = re.compile(boxed_begin)

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        if not judgement.parse_success or judgement.predicted is None:
            return 0
        text = trace.generated_text
        m = self._boxed.search(text)
        if m is None:
            return 0
        pred = str(judgement.predicted).strip()
        if not pred:
            return 0
        idx = text.find(pred)
        if idx < 0:
            return 0
        return int(idx < m.start())


class VerificationSpiralMetric(MetricBase):
    """
    **Verification spiral** (counts hesitation / self-correction cues).

    Patterns are counted only *after* the first occurrence of
    ``judgement.predicted`` in ``generated_text`` (if predictable); if the
    prediction substring is absent, suffix is the full text (fallback).
    """

    name = "verification_spiral"

    def __init__(self, patterns: tuple[str, ...] | None = None) -> None:
        self._patterns = patterns or (
            r"\bwait\b",
            r"\blet\s+me\s+(check|re-?consider|think)\b",
            r"\bhmm+\b",
            r"\bone\s+moment\b",
            r"\bons?\s+econd\s+thought\b",
            r"\bcorrection\b",
            r"\bi\s+(was\s+)?wrong\b",
            r"\bstupid\s+mistake\b",
            r"\bags?\sin\b",
        )
        self._compiled = [re.compile(p, re.I) for p in self._patterns]

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        text = trace.generated_text
        slice_start = 0
        if judgement.predicted is not None and str(judgement.predicted):
            needle = str(judgement.predicted).strip()
            if needle:
                pos = text.find(needle)
                if pos >= 0:
                    slice_start = pos + len(needle)
        tail = text[slice_start:]
        return sum(len(p.findall(tail)) for p in self._compiled)


def _loop_sentence_end_offsets(
    text: str,
    *,
    min_unit_chars: int,
    min_consecutive: int,
    global_repeat_threshold: int | None,
    global_skip_compiled: tuple[re.Pattern[str], ...],
) -> list[int]:
    units = _sentence_units(text, min_unit_chars)
    ends: list[int] = []
    e_streak = _first_streak_end_char(units, min_consecutive)
    if e_streak is not None:
        ends.append(e_streak)
    if global_repeat_threshold is not None:
        e_glob = _sentence_global_burst_end(
            units,
            global_repeat_threshold,
            global_skip_compiled=global_skip_compiled,
        )
        if e_glob is not None:
            ends.append(e_glob)
    return ends


class LoopOnsetTokensMetric(MetricBase):
    """Earliest onset among sentence streak-completion and global-repeat-completion."""

    name = "loop_onset_tokens"

    def __init__(
        self,
        min_consecutive: int = 3,
        min_unit_chars: int = 15,
        *,
        global_repeat_threshold: int | None = 10,
        global_repeat_skip_sentence_patterns: tuple[str, ...] | None = None,
    ) -> None:
        self.min_consecutive = min_consecutive
        self.min_unit_chars = min_unit_chars
        self.global_repeat_threshold = global_repeat_threshold
        self._global_skip = _compile_global_repeat_skip_pattern_tuple(
            global_repeat_skip_sentence_patterns
        )

    def compute(self, trace: Trace, judgement: JudgementResult) -> int:
        text = trace.generated_text
        ttl = trace.total_generated_tokens
        ends = _loop_sentence_end_offsets(
            text,
            min_unit_chars=self.min_unit_chars,
            min_consecutive=self.min_consecutive,
            global_repeat_threshold=self.global_repeat_threshold,
            global_skip_compiled=self._global_skip,
        )
        if not ends or ttl <= 0 or not text:
            return -1
        end_char = min(ends)
        return round(ttl * min(end_char, len(text)) / max(len(text), 1))


class StopTokenProbeMetric(MetricBase):
    """
    Reserved for **stop-token / EOS logit instrumentation** inside vLLM or
    other backends.

    QuantLab traces do not currently record per-step logits; this metric keeps
    a stable column for dashboards and experimental comparisons. Actual values
    require engine integration (planned).
    """

    name = "stop_token_probe"

    def compute(self, trace: Trace, judgement: JudgementResult) -> str:
        return "no_logits_in_trace"


class ActorTokenSplitMetric(MetricBase):
    """Tokens generated per actor — useful for understanding cost split."""

    name = "actor_token_split"

    def compute(self, trace: Trace, judgement: JudgementResult) -> dict[str, int]:
        result: dict[str, int] = {}
        for seg in trace.segments:
            result[seg.actor_id] = result.get(seg.actor_id, 0) + seg.token_count
        return result
