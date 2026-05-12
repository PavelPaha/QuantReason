from __future__ import annotations

from typing import Any, Type

from quantlab.switching.base import SwitchCondition


class ConditionRegistry:
    """
    Plugin registry for switch conditions.

    Usage::

        @ConditionRegistry.register("my_trigger")
        class MyTrigger(SwitchCondition):
            ...

        cond = ConditionRegistry.build("my_trigger", threshold=0.5)
    """

    _registry: dict[str, Type[SwitchCondition]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(cond_cls: Type[SwitchCondition]) -> Type[SwitchCondition]:
            cond_cls.name = name
            cls._registry[name] = cond_cls
            return cond_cls
        return decorator

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> SwitchCondition:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown condition {name!r}. Available: {list(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


def _register_builtins() -> None:
    from quantlab.switching import rules, triggers  # noqa: F401 — side-effect imports

    mapping = {
        "after_n_tokens": rules.AfterNTokens,
        "after_marker": rules.AfterMarker,
        "after_first_segment": rules.AfterFirstSegment,
        "always": rules.AlwaysSwitch,
        "never": rules.NeverSwitch,
        "loop_detector": triggers.LoopDetector,
        "think_not_closed": triggers.ThinkBlockNotClosed,
        "candidate_answer_appeared": triggers.CandidateAnswerAppeared,
        "long_commit_gap": triggers.LongCommitGap,
        "max_tokens_per_stage": triggers.MaxTokensPerStage,
    }
    for name, cls in mapping.items():
        ConditionRegistry._registry.setdefault(name, cls)


_register_builtins()
