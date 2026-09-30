"""Rounding modes.

Names follow IEEE 754 where one exists. microxcaling's mode names collide with
IEEE meanings ("nearest" there is ties-away, "floor" is truncation of the
magnitude), so they are mapped explicitly by :func:`from_microxcaling` instead
of being accepted as aliases.
"""

from __future__ import annotations

from enum import Enum


class Rounding(str, Enum):
    RNE = "rne"  # to nearest, ties to even (IEEE default)
    RNA = "rna"  # to nearest, ties away from zero
    RTZ = "rtz"  # toward zero (truncate the magnitude)
    RUP = "rup"  # toward +inf
    RDN = "rdn"  # toward -inf
    SR = "sr"    # stochastic: away from zero with probability = discarded fraction

    @classmethod
    def parse(cls, value: str | Rounding) -> Rounding:
        if isinstance(value, Rounding):
            return value
        key = value.strip().lower()
        key = {"even": "rne", "trunc": "rtz", "truncate": "rtz", "stochastic": "sr"}.get(key, key)
        try:
            return cls(key)
        except ValueError:
            raise ValueError(f"unknown rounding {value!r}; use one of {[r.value for r in cls]}") from None

    @property
    def code(self) -> int:
        """Stable integer code used as a Triton constexpr."""
        return _CODES[self]


_CODES = {Rounding.RNE: 0, Rounding.RNA: 1, Rounding.RTZ: 2, Rounding.RUP: 3, Rounding.RDN: 4, Rounding.SR: 5}

_MICROXCALING = {"nearest": Rounding.RNA, "even": Rounding.RNE, "floor": Rounding.RTZ, "dither": Rounding.SR}


def from_microxcaling(name: str) -> Rounding:
    """Map a microxcaling ``round=`` string onto the equivalent :class:`Rounding`."""
    try:
        return _MICROXCALING[name]
    except KeyError:
        raise ValueError(f"unknown microxcaling rounding {name!r}") from None
