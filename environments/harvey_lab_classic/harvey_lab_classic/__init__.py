from __future__ import annotations


def load_environment(*args, **kwargs):
    from .environment import load_environment as _load_environment

    return _load_environment(*args, **kwargs)

__all__ = ["load_environment"]
