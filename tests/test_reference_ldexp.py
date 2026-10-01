"""The reference's power-of-two scaling is exact over the whole fp64 range.

``torch.ldexp`` is not: on CUDA it is off by an ulp for some exponents, and on the CPU it loses the
result once ``2**e`` is subnormal. ``math.ldexp`` (C ``ldexp``) is the oracle.
"""

from __future__ import annotations

import math

import torch

from tricast.reference.cast import _ldexp


def _oracle(m: list[int], e: list[int]) -> torch.Tensor:
    def one(a: int, b: int) -> float:
        try:
            return math.ldexp(float(a), b)
        except OverflowError:
            return math.copysign(math.inf, a)

    return torch.tensor([one(a, b) for a, b in zip(m, e, strict=True)], dtype=torch.float64)


def test_ldexp_matches_c_ldexp_from_subnormal_to_overflow():
    gen = torch.Generator().manual_seed(42)
    m = torch.randint(-(2**53) + 1, 2**53, (20000,), generator=gen, dtype=torch.int64)
    e = torch.randint(-2150, 1100, (20000,), generator=gen)  # underflow, subnormal, normal, overflow
    got = _ldexp(m.double(), e)
    assert torch.equal(got.view(torch.int64), _oracle(m.tolist(), e.tolist()).view(torch.int64))


def test_ldexp_keeps_zero_signs_and_specials():
    x = torch.tensor([0.0, -0.0, math.inf, -math.inf, math.nan, 1.5], dtype=torch.float64)
    got = _ldexp(x, torch.tensor([5, -5, 3, -3, 7, -1076]))
    assert torch.equal(torch.signbit(got[:4]), torch.tensor([False, True, False, True]))
    assert got[2] == math.inf and got[3] == -math.inf and math.isnan(got[4])
    assert got[5] == 0.0  # 1.5 * 2**-1076 is 3/8 of the smallest subnormal: rounds to zero
