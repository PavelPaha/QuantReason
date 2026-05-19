from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from quantlab.core.types import HandoffMode, SegmentRole, StagePromptPlacement, TimingInfo


_THINK_OPEN_SUFFIXES = ("<think>\n", "<think>")
_USER_TURN_END = "<|im_end|>\n<|im_start|>assistant\n"
_CHAT_USER_START = "<|im_start|>user\n"
_CHAT_TURN_END = "<|im_end|>"
_PLAN_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def extract_chat_user_content(prompt: str) -> str:
    """Problem text from a Qwen-style chat prompt (between user turn markers)."""
    idx = prompt.find(_CHAT_USER_START)
    if idx == -1:
        raise ValueError(
            f"Cannot extract user content: expected {_CHAT_USER_START!r} in prompt"
        )
    start = idx + len(_CHAT_USER_START)
    end = prompt.find(_CHAT_TURN_END, start)
    if end == -1:
        raise ValueError(
            f"Cannot extract user content: expected {_CHAT_TURN_END!r} after user turn"
        )
    return prompt[start:end].rstrip("\n")


def _strip_think_tags(text: str) -> str:
    return (
        text.replace("</think>", "")
        .replace("<think>", "")
        .strip()
    )


@dataclass
class TraceSegment:
    """Contiguous block of tokens produced by a single actor in a single call."""

    actor_id: str
    text: str
    token_count: int
    start_token_idx: int
    role: SegmentRole = SegmentRole.UNKNOWN
    timing: Optional[TimingInfo] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: ``stage_prompt`` этого шага (суффикс к ``trace.full_text`` перед генерацией).
    stage_prompt_sent: str = ""
    #: Полный промпт одного вызова LLM (= ``full_text_before_segment + stage_prompt_sent``), опционально.
    llm_prompt_full: Optional[str] = None

    @property
    def end_token_idx(self) -> int:
        return self.start_token_idx + self.token_count

    def split_at(self, char_offset: int) -> tuple[TraceSegment, TraceSegment]:
        """Split this segment at a character offset.  token_count is estimated proportionally."""
        left_text = self.text[:char_offset]
        right_text = self.text[char_offset:]
        ratio = char_offset / max(len(self.text), 1)
        left_tokens = max(1, round(self.token_count * ratio))
        right_tokens = max(0, self.token_count - left_tokens)
        left = TraceSegment(
            actor_id=self.actor_id,
            text=left_text,
            token_count=left_tokens,
            start_token_idx=self.start_token_idx,
            role=self.role,
            timing=self.timing,
            metadata=dict(self.metadata),
            stage_prompt_sent=self.stage_prompt_sent,
            llm_prompt_full=self.llm_prompt_full,
        )
        right = TraceSegment(
            actor_id=self.actor_id,
            text=right_text,
            token_count=right_tokens,
            start_token_idx=self.start_token_idx + left_tokens,
            role=self.role,
            metadata=dict(self.metadata),
            stage_prompt_sent=self.stage_prompt_sent,
            llm_prompt_full=self.llm_prompt_full,
        )
        return left, right

    def to_dict(self) -> dict:
        out = {
            "actor_id": self.actor_id,
            "text": self.text,
            "token_count": self.token_count,
            "start_token_idx": self.start_token_idx,
            "role": self.role.value,
            "timing": self.timing.to_dict() if self.timing else None,
            "metadata": self.metadata,
            "stage_prompt_sent": self.stage_prompt_sent,
        }
        if self.llm_prompt_full is not None:
            out["llm_prompt_full"] = self.llm_prompt_full
        return out


@dataclass
class Trace:
    """
    Complete reasoning trajectory assembled from segments produced by different actors.

    The trace is the central object of the system.  Actors append segments; the
    pipeline executor stitches those segments together across handoffs.
    """

    example_id: str
    prompt: str
    segments: list[TraceSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    # ── text helpers ──────────────────────────────────────────────────────────

    @property
    def generated_text(self) -> str:
        """Concatenated text of all segments (no prompt)."""
        return "".join(s.text for s in self.segments)

    @property
    def full_text(self) -> str:
        """prompt + generated_text — what the model would see as a complete string."""
        return self.prompt + self.generated_text

    def handoff_prefix(
        self,
        mode: HandoffMode = HandoffMode.FULL_PREFILL,
        *,
        plan_label: str = "",
        segments: Optional[list[TraceSegment]] = None,
    ) -> str:
        """Context prefix passed to the actor before its ``stage_prompt`` suffix."""
        segs = self.segments if segments is None else segments
        generated = "".join(s.text for s in segs)
        if mode == HandoffMode.SEGMENTS_ONLY:
            return generated
        if mode == HandoffMode.PROMPT_WITHOUT_THINK:
            return self._prompt_without_think_scaffold()
        if mode == HandoffMode.PROMPT_PLAN_LABELED:
            return (
                self._prompt_without_think_scaffold()
                + plan_label
                + _strip_think_tags(generated)
                + "\n"
            )
        return self.prompt + generated

    def _prompt_without_think_scaffold(self) -> str:
        prompt = self.prompt
        for suffix in _THINK_OPEN_SUFFIXES:
            if prompt.endswith(suffix):
                return prompt[: -len(suffix)]
        return prompt

    def inject_user_stage_prompt(self, prompt: str, user_suffix: str) -> str:
        """Append ``user_suffix`` to the last user turn, before ``assistant`` opens."""
        if not user_suffix:
            return prompt
        idx = prompt.rfind(_USER_TURN_END)
        if idx == -1:
            raise ValueError(
                "Cannot inject user stage_prompt: expected chat template ending with "
                f"{_USER_TURN_END!r}"
            )
        return prompt[:idx] + "\n\n" + user_suffix + prompt[idx:]

    def build_plan_scaffold_prompt(
        self,
        *,
        stage_system_prompt: str,
        stage_prompt: str = "",
        problem: Optional[str] = None,
    ) -> str:
        """Plan stage: custom system + ``Problem:\\n…`` user turn + closed empty think block."""
        prob = extract_chat_user_content(self.prompt) if problem is None else problem
        user_body = f"Problem:\n{prob}"
        tail = stage_prompt.strip()
        if tail:
            user_body = f"{user_body}\n{tail}"
        sys_text = stage_system_prompt.strip()
        return (
            "<|im_start|>system\n"
            f"{sys_text}{_CHAT_TURN_END}\n"
            "<|im_start|>user\n"
            f"{user_body}{_CHAT_TURN_END}\n"
            f"{_PLAN_ASSISTANT_PREFIX}"
        )

    def build_llm_prompt(
        self,
        handoff_mode: HandoffMode = HandoffMode.FULL_PREFILL,
        *,
        stage_prompt: str = "",
        stage_prompt_placement: StagePromptPlacement = StagePromptPlacement.ASSISTANT_SUFFIX,
        stage_system_prompt: str = "",
        plan_label: str = "",
        segments: Optional[list[TraceSegment]] = None,
    ) -> str:
        """Full string passed to the backend for one generate call."""
        if stage_prompt_placement == StagePromptPlacement.PLAN_SCAFFOLD:
            if not stage_system_prompt.strip():
                raise ValueError(
                    "stage_system_prompt is required when stage_prompt_placement=plan_scaffold"
                )
            return self.build_plan_scaffold_prompt(
                stage_system_prompt=stage_system_prompt,
                stage_prompt=stage_prompt,
            )
        prefix = self.handoff_prefix(
            handoff_mode,
            plan_label=plan_label,
            segments=segments,
        )
        if stage_prompt_placement == StagePromptPlacement.USER_SUFFIX:
            return self.inject_user_stage_prompt(prefix, stage_prompt)
        return prefix + stage_prompt

    # ── token helpers ─────────────────────────────────────────────────────────

    @property
    def total_generated_tokens(self) -> int:
        return sum(s.token_count for s in self.segments)

    @property
    def next_token_idx(self) -> int:
        return self.total_generated_tokens

    # ── segment helpers ───────────────────────────────────────────────────────

    def append_segment(self, segment: TraceSegment) -> None:
        self.segments.append(segment)

    def get_segments_by_actor(self, actor_id: str) -> list[TraceSegment]:
        return [s for s in self.segments if s.actor_id == actor_id]

    def get_segments_by_role(self, role: SegmentRole) -> list[TraceSegment]:
        return [s for s in self.segments if s.role == role]

    def last_segment(self) -> Optional[TraceSegment]:
        return self.segments[-1] if self.segments else None

    # ── serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "total_generated_tokens": self.total_generated_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trace:
        segments = [
            TraceSegment(
                actor_id=s["actor_id"],
                text=s["text"],
                token_count=s["token_count"],
                start_token_idx=s["start_token_idx"],
                role=SegmentRole(s.get("role", "unknown")),
                timing=TimingInfo(**s["timing"]) if s.get("timing") else None,
                metadata=s.get("metadata", {}),
                stage_prompt_sent=s.get("stage_prompt_sent", ""),
                llm_prompt_full=s.get("llm_prompt_full"),
            )
            for s in d.get("segments", [])
        ]
        return cls(
            example_id=d["example_id"],
            prompt=d["prompt"],
            segments=segments,
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
            finished_at=d.get("finished_at"),
        )
