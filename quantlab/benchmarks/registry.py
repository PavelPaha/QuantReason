from __future__ import annotations

from typing import Type

from quantlab.benchmarks.base import BenchmarkAdapter


class BenchmarkRegistry:
    _registry: dict[str, Type[BenchmarkAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(adapter_cls: Type[BenchmarkAdapter]) -> Type[BenchmarkAdapter]:
            adapter_cls.name = name
            cls._registry[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def build(cls, name: str) -> BenchmarkAdapter:
        if name not in cls._registry:
            raise KeyError(f"Unknown benchmark {name!r}. Available: {list(cls._registry)}")
        return cls._registry[name]()

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


def _register_builtins() -> None:
    from quantlab.benchmarks.math500 import MATH500Adapter
    from quantlab.benchmarks.gsm8k import GSM8KAdapter
    from quantlab.benchmarks.gpqa import GPQAAdapter
    from quantlab.benchmarks.winogrande import WinoGrandeAdapter

    mapping = {
        "math500": MATH500Adapter,
        "gsm8k": GSM8KAdapter,
        "gpqa": GPQAAdapter,
        "winogrande": WinoGrandeAdapter,
    }
    for name, cls in mapping.items():
        BenchmarkRegistry._registry.setdefault(name, cls)


_register_builtins()
