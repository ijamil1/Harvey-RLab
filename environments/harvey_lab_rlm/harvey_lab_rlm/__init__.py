from __future__ import annotations

from typing import Any


def load_environment(**kwargs: Any):
    from .environment import load_environment as _load_environment

    return _load_environment(**kwargs)


def __getattr__(name: str):
    if name == "HarveyLabRLMEnv":
        from .environment import HarveyLabRLMEnv

        return HarveyLabRLMEnv
    raise AttributeError(name)


__all__ = ["HarveyLabRLMEnv", "load_environment"]
