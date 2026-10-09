"""PPL verdict for two evaluation records measured under the same conditions (SPEC AC13, AC14)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any


def _finite_number(value: Any) -> bool:
    """The runner's rule (``runner._finite_number``): a real, not a bool, finite as a float."""
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


# Two perplexities are comparable only for the same model revision, dataset, and window layout.
# Each value must also be valid by the runner's rules (runner._valid_env, runner._valid_metrics).
SAME_CONDITIONS: tuple[tuple[str, Callable[[Any], bool], str], ...] = (
    ("env.model_sha", _non_empty_string, "a non-empty string"),
    ("metrics.ppl.dataset_fingerprint", _non_empty_string, "a non-empty string"),
    ("metrics.ppl.n_tokens", _positive_int, "an int greater than 0"),
)


def _show(value: Any) -> str:
    """Short repr for an error message; a huge int's repr can exceed Python's digit limit and raise."""
    try:
        text = repr(value)
    except ValueError:
        return f"<{type(value).__name__} too large to print>"
    return text if len(text) <= 60 else text[:57] + "..."


def _lookup(record: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(record, Mapping):
            return None
        record = record.get(key)
    return record


def compare_ppl(base: Mapping[str, Any], emu: Mapping[str, Any], limit: float = 1e-3) -> dict[str, Any]:
    """Return ``relative = |emu - base| / base`` and ``pass = relative <= limit`` (limit 0.1% by default).

    Refuses with ``ValueError("<field path>: ...")`` when a record has no PPL metric, a comparison
    condition (model revision, dataset fingerprint, evaluated token count) is invalid in either record
    or differs between them, the baseline PPL is not a finite positive number, the limit is not a finite
    non-negative number, or the emulated PPL is not a number. A NaN or inf emulated PPL is a failing
    verdict (``relative = inf``), not an error. The input records are only read."""
    for name, record in (("base", base), ("emu", emu)):
        if not isinstance(_lookup(record, "metrics.ppl"), Mapping):
            raise ValueError(f"{name}.metrics.ppl: the record has no perplexity metric")
    for path, valid, kind in SAME_CONDITIONS:
        expected, actual = _lookup(base, path), _lookup(emu, path)
        for name, value in (("base", expected), ("emu", actual)):
            if not valid(value):
                raise ValueError(f"{path}: {name} has {_show(value)}, expected {kind}")
        if expected != actual:
            raise ValueError(f"{path}: base {_show(expected)} differs from emu {_show(actual)}")
    value = _lookup(base, "metrics.ppl.ppl")
    if not (_finite_number(value) and value > 0):
        raise ValueError(f"base.metrics.ppl.ppl: {_show(value)} is not a finite positive number")
    reference = float(value)
    if not (_finite_number(limit) and limit >= 0):
        raise ValueError(f"limit: {_show(limit)} is not a finite non-negative number")
    bound = float(limit)
    value = _lookup(emu, "metrics.ppl.ppl")
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"emu.metrics.ppl.ppl: {_show(value)} is not a number")
    try:
        emulated = float(value)
    except OverflowError:
        raise ValueError(f"emu.metrics.ppl.ppl: {_show(value)} is too large for a float") from None
    relative = abs(emulated - reference) / reference if math.isfinite(emulated) else math.inf
    return {"relative": relative, "pass": relative <= bound, "limit": bound}
