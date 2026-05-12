from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class BenchmarkExample:
    example_id: str
    prompt: str
    ground_truth: str
    raw: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """
    Adapter for a benchmark dataset.

    Subclasses implement data loading, prompt construction, answer extraction,
    and answer comparison.  All of these are benchmark-specific and should not
    bleed into the generic pipeline/metrics code.
    """

    name: str = "unnamed"

    @abstractmethod
    def load(
        self,
        split: str = "test",
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
        seed: int = 42,
    ) -> list[BenchmarkExample]:
        """Load and return examples for the given split."""
        ...

    @abstractmethod
    def build_prompt(self, raw: dict[str, Any]) -> str:
        """Convert a raw dataset row into the model prompt."""
        ...

    @abstractmethod
    def extract_answer(self, generated_text: str) -> Optional[str]:
        """Extract a structured answer from generated text."""
        ...

    @abstractmethod
    def is_correct(self, predicted: Optional[str], ground_truth: str) -> bool:
        """Return True if the prediction matches the ground truth."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
