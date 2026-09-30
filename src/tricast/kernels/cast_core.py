"""Integer-bit casts of fp32 blocks, including gradual underflow.

Only the kept/remainder shifts are capped at 25. SR retains the original
quantum distance: even a very small nonzero fraction rounds up for noise zero.
No floating rescaling is used to construct subnormal results.
"""

from __future__ import annotations

import struct

import triton
import triton.language as tl

from ..formats import FloatFormat, Format, IntFormat, Pow2Format, get_format
from ..rounding import Rounding


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fmt_constexprs(
    fmt: Format | str, rounding: Rounding | str = Rounding.RNE, saturate: bool = True,
) -> dict:
    """Keyword constants for ``round_to_format`` and the standalone cast kernel.

    FMT packs kind, mbits, emin, emax, max/min fp32 bits, signed, subnormals,
    special, negative integer bound bits, and frac_bits (in that order).
    """
    fmt, rounding = get_format(fmt), Rounding.parse(rounding)
    if isinstance(fmt, Pow2Format) and rounding is Rounding.SR:
        raise ValueError("stochastic rounding is not defined for power-of-two formats")
    if isinstance(fmt, FloatFormat):
        packed = (0, fmt.mbits, fmt.emin, fmt.emax, _bits(fmt.max_normal), _bits(fmt.min_normal),
                  fmt.signed, fmt.subnormals, fmt.special, 0, 0)
    elif isinstance(fmt, IntFormat):
        packed = (1, 0, 0, 0, _bits(fmt.max_normal), _bits(fmt.min_normal),
                  fmt.signed, True, "none", _bits(-fmt.qmin * 2.0**-fmt.frac_bits), fmt.frac_bits)
    else:
        packed = (2, 0, fmt.emin, fmt.emax, _bits(fmt.max_normal), _bits(fmt.min_normal),
                  False, True, "none", 0, 0)
    return {"FMT": packed, "ROUNDING": rounding.code, "SATURATE": saturate}


@triton.jit
def _ilog2(value):
    """Integer floor log2 for uint32, with zero mapped to zero."""
    rest = value.to(tl.uint32)
    result = tl.full(value.shape, 0, tl.int32)
    for step in tl.static_range(4, -1, -1):
        shift = 1 << step
        active = rest >= (1 << shift)
        rest = tl.where(active, rest >> shift, rest)
        result += tl.where(active, shift, 0)
    return result


@triton.jit
def _float_parts(abs_bits):
    field = (abs_bits >> 23).to(tl.int32)
    sig = (abs_bits & 0x7FFFFF) | tl.where(field != 0, 0x800000, 0).to(tl.uint32)
    exponent = tl.where(field != 0, field - 127, -126)
    return sig, exponent


@triton.jit
def _floor_log2_bits(abs_bits):
    field = (abs_bits >> 23).to(tl.int32)
    return tl.where(field != 0, field - 127, _ilog2(abs_bits) - 149)


@triton.jit
def _encode(magnitude, quantum):
    """Encode an exact unsigned integer times 2**quantum as fp32 bits."""
    magnitude = magnitude.to(tl.uint32)
    leading = _ilog2(magnitude)
    biased = leading + quantum + 127
    mantissa = (magnitude << tl.minimum(tl.maximum(23 - leading, 0), 31)) >> tl.minimum(
        tl.maximum(leading - 23, 0), 31,
    )
    normal = (biased.to(tl.uint32) << 23) | (mantissa & 0x7FFFFF)
    subshift = quantum + 149
    sub = tl.where(
        subshift >= 0,
        magnitude << tl.minimum(tl.maximum(subshift, 0), 31),
        magnitude >> tl.minimum(tl.maximum(-subshift, 0), 31),
    )
    sub = tl.where(subshift < -31, 0, sub)
    bits = tl.where(biased <= 0, sub, normal)
    bits = tl.where(biased >= 255, 0x7F800000, bits)
    return tl.where(magnitude == 0, 0, bits).to(tl.uint32)


@triton.jit
def _increment(kept, rem, shift, negative, noise, ROUNDING: tl.constexpr, SR_BITS: tl.constexpr):
    half = 1 << tl.minimum(tl.maximum(shift - 1, 0), 24)
    if ROUNDING == 0:
        inc = (shift <= 24) & ((rem > half) | ((rem == half) & ((kept & 1) != 0)))
    elif ROUNDING == 1:
        inc = (shift <= 24) & (rem >= half)
    elif ROUNDING == 2:
        inc = tl.full(rem.shape, False, tl.int1)
    elif ROUNDING == 3:
        inc = (rem != 0) & ~negative
    elif ROUNDING == 4:
        inc = (rem != 0) & negative
    else:
        random = (noise.to(tl.uint32) >> (32 - SR_BITS)).to(tl.uint64)
        left = rem.to(tl.uint64) << tl.minimum(tl.maximum(SR_BITS - shift, 0), 32)
        right = (rem.to(tl.uint64) - 1) >> tl.minimum(tl.maximum(shift - SR_BITS, 0), 24)
        inc = (rem != 0) & tl.where(shift <= SR_BITS, random < left, random <= right)
    return (shift > 0) & inc


@triton.jit
def round_to_format(
    x, noise, FMT: tl.constexpr, ROUNDING: tl.constexpr, SATURATE: tl.constexpr,
    SR_BITS: tl.constexpr = 32,
):
    """Cast fp32 values using bits, not dtype round trips (ENGINE §2).

    Pow2 positive infinity follows the contract's overflow rule. Negative
    integer inputs preserve the reference's signed zero on an unsigned grid.
    """
    bits = x.to(tl.uint32, bitcast=True)
    absolute = bits & 0x7FFFFFFF
    sign = bits & 0x80000000
    negative = sign != 0
    isnan = absolute > 0x7F800000
    if FMT[0] == 2:
        exponent = _floor_log2_bits(absolute)
        sig, _ = _float_parts(absolute)
        leading = _ilog2(sig)
        normalized = sig << tl.minimum(tl.maximum(23 - leading, 0), 31)
        fraction = normalized & 0x7FFFFF
        if ROUNDING == 0 or ROUNDING == 1:
            up = fraction >= 0x400000
        elif ROUNDING == 3:
            up = fraction != 0
        else:
            up = tl.full(x.shape, False, tl.int1)
        exponent = tl.maximum(exponent + up.to(tl.int32), FMT[2])
        over = (exponent > FMT[3]) | (absolute == 0x7F800000)
        result = _encode(tl.full(x.shape, 1, tl.uint32), tl.minimum(exponent, FMT[3]))
        result = tl.where(absolute == 0, FMT[5], result)
        result = tl.where(over, FMT[4] if SATURATE else 0x7FC00000, result)
        result = tl.where(isnan | (negative & (absolute != 0)), 0x7FC00000, result)
    else:
        if FMT[0] == 0 and not FMT[7]:
            absolute = tl.where(absolute < FMT[5], 0, absolute)
        sig, exponent = _float_parts(absolute)
        if FMT[0] == 0:
            quantum = tl.maximum(exponent, FMT[2]) - FMT[1]
        else:
            quantum = tl.full(x.shape, -FMT[10], tl.int32)
        shift = quantum - exponent + 23
        capped = tl.minimum(tl.maximum(shift, 0), 25)
        kept = sig >> capped
        rem = sig & ((1 << capped) - 1).to(tl.uint32)
        inc = _increment(kept, rem, shift, negative, noise, ROUNDING, SR_BITS)
        magnitude = _encode(kept + inc.to(tl.uint32), quantum)
        magnitude = tl.where(shift <= 0, absolute, magnitude)
        if FMT[0] == 1:
            bound = tl.where(negative, FMT[9], FMT[4]).to(tl.uint32)
            magnitude = tl.minimum(magnitude, bound)
            magnitude = tl.where(absolute == 0x7F800000, bound, magnitude)
            if not FMT[6]:
                magnitude = tl.where(negative, 0, magnitude)
        else:
            over = (magnitude > FMT[4]) | (absolute == 0x7F800000)
            if SATURATE or FMT[8] == "none":
                overflow = tl.full(x.shape, FMT[4], tl.uint32)
            elif FMT[8] == "fn" or FMT[8] == "fnuz":
                overflow = tl.full(x.shape, 0x7FC00000, tl.uint32)
            else:
                if ROUNDING == 2:
                    away = tl.full(x.shape, False, tl.int1)
                elif ROUNDING == 3:
                    away = ~negative
                elif ROUNDING == 4:
                    away = negative
                else:
                    away = tl.full(x.shape, True, tl.int1)
                overflow = tl.where(away, 0x7F800000, FMT[4]).to(tl.uint32)
            magnitude = tl.where(over, overflow, magnitude)
        if FMT[0] == 1 and not FMT[6]:
            result = magnitude  # unsigned integer grids have no -0
        else:
            result = magnitude | sign
        if FMT[0] == 0:
            if FMT[8] == "fnuz" or not FMT[6]:
                result = tl.where(magnitude == 0, 0, result)
            if not FMT[6]:
                result = tl.where(negative & (absolute != 0), 0 if SATURATE else 0x7FC00000, result)
        result = tl.where(isnan, bits, result)
    return result.to(tl.uint32).to(tl.float32, bitcast=True)
