"""Model evaluation; optional lm-eval dependencies are loaded on demand."""

from importlib import import_module

_EXPORTS = {
    "perplexity": ".ppl",
    "evaluate": ".lmeval",
    "TriCastLM": ".lmeval",
    "run_config": ".runner",
    "capture_env": ".envinfo",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_EXPORTS)
