from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from quantlab.actors.base import ActorBase, ActorConfig


class ActorRegistry:
    """Maps backend names to actor implementation classes."""

    _registry: dict[str, Type[ActorBase]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(actor_cls: Type[ActorBase]) -> Type[ActorBase]:
            cls._registry[name] = actor_cls
            return actor_cls
        return decorator

    @classmethod
    def build(cls, config: ActorConfig) -> ActorBase:
        backend_name = config.backend
        if backend_name not in cls._registry:
            raise KeyError(
                f"Unknown backend {backend_name!r}. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[backend_name](config)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


def _register_defaults() -> None:
    from quantlab.actors.impls import TransformersActor, VLLMActor

    ActorRegistry._registry.setdefault("transformers", TransformersActor)
    ActorRegistry._registry.setdefault("vllm", VLLMActor)


# Lazy registration on first import of registry module
_register_defaults()
