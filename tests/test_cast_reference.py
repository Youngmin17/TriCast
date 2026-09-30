"""Exact casting checks against torch and independently enumerated format grids."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import replace
from fractions import Fraction
from functools import cache

import pytest
import torch

from tests.conftest import bit_equal, wide_fp32
from tricast.formats import (
    E8M0,
    FP4_E2M1,
    FP6_E2M3,
    FP6_E3M2,
    FP8_E4M3,
    FP8_E4M3FNUZ,
    FP8_E5M2,
    FP8_E5M2FNUZ,
    INT4,
    MXINT8,
    UE4M3,
    UINT4,
    FloatFormat,
    Format,
    IntFormat,
    Pow2Format,
    format_of_dtype,
    get_format,
)
from tricast.reference.cast import decode, round_to_format
from tricast.rounding import Rounding

SMALL_FORMATS = (FP4_E2M1, FP6_E3M2, FP6_E2M3, FP8_E4M3, INT4, MXINT8, UINT4,
                 get_format("e3m4:none"))
TORCH_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz,
                torch.float8_e5m2fnuz, torch.bfloat16, torch.float16)


def _pow2(exponent: int) -> Fraction:
    return Fraction(2) ** exponent


@cache
def _positive_grid(fmt: Format) -> tuple[Fraction, ...]:
    if isinstance(fmt, IntFormat):
        return tuple(Fraction(k) * _pow2(-fmt.frac_bits) for k in range(fmt.qmax + 1))
    if isinstance(fmt, Pow2Format):
        return tuple(_pow2(e) for e in range(fmt.emin, fmt.emax + 1))
    grid = {Fraction(0)}
    for field in range(2**fmt.ebits):
        if fmt.special == "ieee" and field == 2**fmt.ebits - 1:
            continue
        for fraction in range(2**fmt.mbits):
            if fmt.special == "fn" and field == 2**fmt.ebits - 1 and fraction == 2**fmt.mbits - 1:
                continue
            if field == 0:
                if fmt.subnormals:
                    grid.add(Fraction(fraction) * _pow2(fmt.emin - fmt.mbits))
            else:
                grid.add(Fraction(2**fmt.mbits + fraction) * _pow2(field - fmt.bias - fmt.mbits))
    return tuple(sorted(grid))


@cache
def _grid_cases(fmt: Format) -> tuple[tuple[Fraction, Fraction, Fraction, tuple[Fraction, ...]], ...]:
    """Enumerate the whole grid; search it exhaustively for nearest candidates."""
    grid = _positive_grid(fmt)
    samples = set(grid)
    for lower, upper in zip(grid, grid[1:], strict=False):
        for fraction in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
            samples.add(lower + (upper - lower) * fraction)
        midpoint = torch.tensor(float((lower + upper) / 2), dtype=torch.float32)
        for direction in (-math.inf, math.inf):
            neighbor = torch.nextafter(midpoint, torch.tensor(direction)).item()
            samples.add(Fraction(neighbor))
    cases = []
    for value in sorted(samples):
        index = bisect_left(grid, value)
        if index < len(grid) and grid[index] == value:
            lower = upper = value
        else:
            lower, upper = grid[index - 1], grid[index]
        distance = min(abs(candidate - value) for candidate in grid)
        nearest = tuple(candidate for candidate in grid if abs(candidate - value) == distance)
        cases.append((value, lower, upper, nearest))
    return tuple(cases)


def _oracle_case(
    case: tuple[Fraction, Fraction, Fraction, tuple[Fraction, ...]],
    negative: bool,
    mode: Rounding,
    noise: int,
    sr_bits: int,
    fmt: Format,
) -> float:
    value, lower, upper, nearest = case
    if lower == upper:
        result = lower
    elif mode is Rounding.RNE:
        # The parity of the magnitude-grid index equals the stored significand parity.
        result = next(candidate for candidate in nearest
                      if len(nearest) == 1 or _positive_grid(fmt).index(candidate) % 2 == 0)
    elif mode is Rounding.RNA:
        result = max(nearest)
    elif mode is Rounding.RTZ:
        result = lower
    elif mode is Rounding.RUP:
        result = lower if negative else upper
    elif mode is Rounding.RDN:
        result = upper if negative else lower
    else:
        uniform = Fraction(noise >> (32 - sr_bits), 2**sr_bits)
        result = upper if uniform < (value - lower) / (upper - lower) else lower
    return float(-result if negative else result)


@pytest.mark.parametrize("dtype", TORCH_DTYPES, ids=str)
def test_torch_cast_parity(dtype: torch.dtype, gen: torch.Generator) -> None:
    fmt = format_of_dtype(dtype)
    quantum = 2.0 ** (fmt.emax - fmt.mbits)
    landmarks = [0.0, fmt.min_subnormal / 2, fmt.min_subnormal, fmt.min_normal,
                 fmt.max_normal, fmt.max_normal + quantum / 2]
    base = torch.tensor(landmarks, dtype=torch.float32)
    boundaries = torch.cat((base, torch.nextafter(base, torch.full_like(base, math.inf)),
                            torch.nextafter(base, torch.full_like(base, -math.inf))))
    special = torch.tensor([0.0, -0.0, math.inf, -math.inf, math.nan])
    x = torch.cat((wide_fp32(100_000, gen, lo_exp=0, hi_exp=255), boundaries, -boundaries, special))
    actual = round_to_format(x, fmt, Rounding.RNE, saturate=False)
    expected = x.to(dtype).float()
    assert bit_equal(actual, expected)
    zero = expected == 0
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


@pytest.mark.parametrize("fmt", SMALL_FORMATS, ids=str)
@pytest.mark.parametrize("mode", list(Rounding), ids=str)
def test_exhaustive_grid_fraction_oracle(fmt: Format, mode: Rounding) -> None:
    values, expected, noises = [], [], []
    for index, case in enumerate(_grid_cases(fmt)):
        for negative in ((False, True) if fmt.signed else (False,)):
            noise = (0, 2**30, 2**31, 3 * 2**30, 2**32 - 1)[index % 5]
            values.append(float(-case[0] if negative else case[0]))
            noises.append(noise)
            expected.append(_oracle_case(case, negative, mode, noise, 32, fmt))
    actual = round_to_format(torch.tensor(values), fmt, mode, saturate=True,
                             noise=torch.tensor(noises, dtype=torch.int64))
    assert actual.dtype == torch.float32
    assert bit_equal(actual, torch.tensor(expected))


@pytest.mark.parametrize("fmt", (FP8_E5M2, FP8_E4M3, FP8_E4M3FNUZ, FP8_E5M2FNUZ, FP4_E2M1), ids=str)
@pytest.mark.parametrize("mode", list(Rounding), ids=str)
@pytest.mark.parametrize("saturate", (False, True))
def test_float_overflow_policy(fmt: FloatFormat, mode: Rounding, saturate: bool) -> None:
    x = torch.tensor([fmt.max_normal * 2, -fmt.max_normal * 2, math.inf, -math.inf, math.nan])
    expected = []
    for value in x.tolist():
        if math.isnan(value) or (not saturate and fmt.special in ("fn", "fnuz")):
            result = math.nan
        elif saturate or fmt.special == "none":
            result = math.copysign(fmt.max_normal, value)
        else:
            away = (mode in (Rounding.RNE, Rounding.RNA, Rounding.SR)
                    or (mode is Rounding.RUP and value > 0) or (mode is Rounding.RDN and value < 0))
            result = math.copysign(math.inf if away else fmt.max_normal, value)
        expected.append(result)
    actual = round_to_format(x, fmt, mode, saturate=saturate, noise=torch.zeros(5, dtype=torch.int64))
    assert bit_equal(actual, torch.tensor(expected))


@pytest.mark.parametrize("mode", list(Rounding), ids=str)
def test_integer_clamp_and_nan(mode: Rounding) -> None:
    x = torch.tensor([-math.inf, -20.0, -0.0, 0.0, 20.0, math.inf, math.nan])
    for fmt in (INT4, UINT4, get_format("int4:full"), MXINT8):
        lo, hi = fmt.qmin * 2.0**-fmt.frac_bits, fmt.max_normal
        expected = torch.tensor([lo, lo, 0.0, 0.0, hi, hi, math.nan])
        actual = round_to_format(x, fmt, mode, noise=torch.zeros(7, dtype=torch.int64))
        assert bit_equal(actual, expected)


@pytest.mark.parametrize("fmt", (FP4_E2M1, FP8_E4M3FNUZ, INT4, UINT4, UE4M3), ids=str)
def test_zero_sign_policy(fmt: Format) -> None:
    actual = round_to_format(torch.tensor([-0.0, 0.0]), fmt)
    keep_negative = fmt.signed and not (isinstance(fmt, FloatFormat) and fmt.special == "fnuz")
    assert torch.equal(torch.signbit(actual), torch.tensor([keep_negative, False]))


@pytest.mark.parametrize("mode", list(Rounding), ids=str)
def test_subnormals_flush_before_rounding(mode: Rounding) -> None:
    fmt = replace(FP6_E3M2, name="fp6_nosub", subnormals=False)
    below = torch.nextafter(torch.tensor(fmt.min_normal), torch.tensor(0.0)).item()
    x = torch.tensor([-below, -fmt.min_subnormal / 2, -0.0, 0.0, fmt.min_subnormal / 2, below,
                      -fmt.min_normal, fmt.min_normal])
    expected = torch.tensor([-0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -fmt.min_normal, fmt.min_normal])
    actual = round_to_format(x, fmt, mode, noise=torch.zeros(x.shape, dtype=torch.int64))
    assert bit_equal(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("mode", list(Rounding)[:-1], ids=str)
def test_pow2_value_rounding(mode: Rounding) -> None:
    x = torch.tensor([0.0, 2.0**-149, 2.0**-127, 1.0, 1.25, 1.5, 1.75, 2.0, -1.0, math.nan])
    middle = {Rounding.RNE: [1, 2, 2], Rounding.RNA: [1, 2, 2], Rounding.RTZ: [1, 1, 1],
              Rounding.RUP: [2, 2, 2], Rounding.RDN: [1, 1, 1]}[mode]
    expected = torch.tensor([2.0**-127] * 3 + [1.0, *middle, 2.0, math.nan, math.nan])
    assert bit_equal(round_to_format(x, E8M0, mode, saturate=False), expected)


@pytest.mark.parametrize("saturate", (False, True))
def test_pow2_overflow(saturate: bool) -> None:
    x = torch.tensor([math.inf, torch.finfo(torch.float32).max])
    expected = torch.full_like(x, E8M0.max_normal if saturate else math.nan)
    assert bit_equal(round_to_format(x, E8M0, saturate=saturate), expected)


def test_pow2_stochastic_is_undefined() -> None:
    with pytest.raises(ValueError, match="stochastic"):
        round_to_format(torch.ones(1), E8M0, Rounding.SR)


@pytest.mark.parametrize("value", (1.25, -1.25, 0.375, -0.375))
def test_stochastic_mean_within_four_standard_errors(value: float) -> None:
    count = 20_000
    x = torch.full((count,), value)
    actual = round_to_format(x, FP4_E2M1, Rounding.SR, generator=torch.Generator().manual_seed(42))
    grid = _positive_grid(FP4_E2M1)
    index = bisect_left(grid, Fraction(abs(value)))
    lower, upper = float(grid[index - 1]), float(grid[index])
    probability = (abs(value) - lower) / (upper - lower)
    standard_error = (upper - lower) * math.sqrt(probability * (1 - probability) / count)
    # Statistical sampling, unlike the deterministic casting checks, needs a confidence interval.
    assert abs(actual.double().mean().item() - value) <= 4 * standard_error


@pytest.mark.parametrize("sr_bits", (1, 2, 8, 16, 32))
def test_stochastic_explicit_noise_and_precision(sr_bits: int) -> None:
    x = torch.tensor([1.125, -1.125, 1.375, -1.375])
    noise = torch.tensor([2**30, 2**30, 3 * 2**30, 3 * 2**30], dtype=torch.int64)
    u = [Fraction(int(n) >> (32 - sr_bits), 2**sr_bits) for n in noise]
    fractions = (Fraction(1, 4), Fraction(1, 4), Fraction(3, 4), Fraction(3, 4))
    expected = [math.copysign(1.5 if draw < fraction else 1.0, value)
                for value, draw, fraction in zip(x.tolist(), u, fractions, strict=True)]
    actual = round_to_format(x, FP4_E2M1, Rounding.SR, noise=noise, sr_bits=sr_bits)
    assert bit_equal(actual, torch.tensor(expected))


@pytest.mark.parametrize("sr_bits", (0, 33))
def test_stochastic_rejects_invalid_precision(sr_bits: int) -> None:
    with pytest.raises(ValueError, match="sr_bits"):
        round_to_format(torch.ones(1), FP4_E2M1, Rounding.SR, sr_bits=sr_bits)


@pytest.mark.parametrize("fmt", (*SMALL_FORMATS, FP8_E5M2, FP8_E4M3FNUZ, FP8_E5M2FNUZ, E8M0), ids=str)
def test_decode_reconstructs_entire_grid(fmt: Format) -> None:
    grid = list(_positive_grid(fmt))
    if fmt.signed:
        grid += [-value for value in grid if value]
    values = torch.tensor([float(value) for value in grid], dtype=torch.float32)
    negative, exponent, significand, radix = decode(values, fmt)
    restored = torch.ldexp(significand.double(), exponent - radix)
    restored = torch.where(negative, -restored, restored).float()
    assert bit_equal(restored, values)
    assert exponent.dtype == significand.dtype == torch.int64
    assert torch.equal(significand == 0, values == 0)
    if isinstance(fmt, IntFormat):
        assert radix == fmt.frac_bits
        assert torch.equal(exponent, torch.zeros_like(exponent))
    elif isinstance(fmt, Pow2Format):
        assert radix == 0
        assert torch.equal(significand, torch.ones_like(significand))
    else:
        assert radix == fmt.mbits
        normal = values.abs() >= fmt.min_normal
        assert bool(((significand[normal] >= 2**radix) & (significand[normal] < 2**(radix + 1))).all())


@pytest.mark.parametrize("fmt,value", ((FP4_E2M1, 1.25), (INT4, 0.5), (E8M0, 1.5),
                                       (FP4_E2M1, math.nan), (FP4_E2M1, math.inf)))
def test_decode_rejects_non_grid_values(fmt: Format, value: float) -> None:
    with pytest.raises(ValueError):
        decode(torch.tensor([value]), fmt)


@pytest.mark.parametrize("fmt,value", ((FP4_E2M1, 8.0), (INT4, 8.0), (UINT4, -1.0),
                                       (E8M0, -1.0), (E8M0, 0.0),
                                       (replace(FP6_E3M2, subnormals=False), 0.0625)))
def test_decode_rejects_values_outside_format_grid(fmt: Format, value: float) -> None:
    with pytest.raises(ValueError):
        decode(torch.tensor([value]), fmt)


def test_stochastic_rounding_requires_an_explicit_random_source() -> None:
    x = torch.tensor([.125, .375, 1.0625])
    state = torch.random.get_rng_state()
    with pytest.raises(ValueError, match="explicit noise or generator"):
        round_to_format(x, FP4_E2M1, Rounding.SR)
    first = round_to_format(x, FP4_E2M1, Rounding.SR,
                            generator=torch.Generator().manual_seed(42))
    second = round_to_format(x, FP4_E2M1, Rounding.SR,
                             generator=torch.Generator().manual_seed(42))
    assert bit_equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), state)
