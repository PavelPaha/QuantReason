from quantlab.actors.base import ActorBase, ActorConfig

__all__ = ["ActorBase", "ActorConfig", "ActorRegistry"]


def __getattr__(name: str):
    if name == "ActorRegistry":
        from quantlab.actors.registry import ActorRegistry

        return ActorRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
