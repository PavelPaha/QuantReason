from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from quantlab.core.types import SegmentRole, TimingInfo


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
        )
        right = TraceSegment(
            actor_id=self.actor_id,
            text=right_text,
            token_count=right_tokens,
            start_token_idx=self.start_token_idx + left_tokens,
            role=self.role,
            metadata=dict(self.metadata),
        )
        return left, right

    def to_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "text": self.text,
            "token_count": self.token_count,
            "start_token_idx": self.start_token_idx,
            "role": self.role.value,
            "timing": self.timing.to_dict() if self.timing else None,
            "metadata": self.metadata,
        }


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
