"""Reference casting: round any tensor onto a format grid, and decode grid values.

This module is the executable definition of TriCast's casting semantics
(``docs/design/ENGINE.md`` §2). It favours clarity over speed: inputs are taken
to fp64, where every fp32 value and every power-of-two rescaling is exact, so
each step below is an exact operation followed by one explicit rounding
decision. Triton kernels must reproduce these results bit for bit.
"""

from __future__ import annotations

import torch

from ..formats import FloatFormat, Format, IntFormat, Pow2Format, get_format
from ..rounding import Rounding


def round_to_format(
    x: torch.Tensor,
    fmt: Format | str,
    rounding: Rounding | str = Rounding.RNE,
    *,
    saturate: bool = True,
    noise: torch.Tensor | None = None,
    sr_bits: int = 32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Round ``x`` onto the grid of ``fmt``; returns fp32 holding grid values.

    Overflow: ``saturate=True`` clamps to ``±max_normal``. Otherwise IEEE rules
    apply: formats with Inf return Inf when the rounding direction points away
    from zero and ``max_normal`` otherwise; ``fn``/``fnuz`` formats return NaN;
    formats without Inf/NaN always saturate. Inf inputs overflow the same way;
    NaN inputs stay NaN. Formats without subnormals flush ``|x| < min_normal``
    to zero before rounding (microxcaling ``allow_denorm=False``).

    Stochastic rounding (``Rounding.SR``) rounds away from zero when
    ``u < discarded_fraction`` with ``u = floor(noise / 2**(32 - sr_bits)) / 2**sr_bits``;
    ``noise`` is a uint32-valued tensor (int64 accepted) broadcastable to ``x``.
    Passing the same ``noise`` makes every backend produce identical results.
    Without ``noise``, SR requires an explicit ``generator``; the global RNG is never used.
    """
    fmt = get_format(fmt)
    rounding = Rounding.parse(rounding)
    xd = x.to(torch.float64)
    if isinstance(fmt, Pow2Format):
        return _round_pow2(xd, fmt, rounding, saturate)
    if isinstance(fmt, FloatFormat):
        if not fmt.subnormals:
            xd = torch.where(xd.abs() < fmt.min_normal, torch.zeros_like(xd).copysign(xd), xd)
        exp = _floor_log2(xd.abs()).clamp(min=fmt.emin)
        quantum = exp - fmt.mbits
    elif isinstance(fmt, IntFormat):
        quantum = torch.full_like(xd, -fmt.frac_bits, dtype=torch.int64)
    else:
        raise TypeError(f"unsupported format {fmt!r}")

    ax = xd.abs()
    finite = torch.isfinite(xd)
    y = _ldexp(torch.where(finite, ax, torch.zeros_like(ax)), -quantum)
    k = torch.floor(y)
    frac = y - k
    inc = _round_up(frac, k, xd < 0, rounding, noise, sr_bits, generator)
    mag = _ldexp(k + inc.to(k.dtype), quantum)

    if isinstance(fmt, IntFormat):
        lo = -fmt.qmin * 2.0**-fmt.frac_bits  # magnitude bound for negatives
        mag = torch.where(xd < 0, mag.clamp(max=lo), mag.clamp(max=fmt.max_normal))
        if not fmt.signed:
            mag = torch.where(xd < 0, torch.zeros_like(mag), mag)
        mag = torch.where(finite, mag, torch.where(torch.isnan(xd), xd.abs(), _int_inf_bound(xd, fmt)))
        out = mag.copysign(xd) if fmt.signed else mag  # unsigned grids have no -0
        return torch.where(torch.isnan(xd), xd, out).to(torch.float32)

    over = (mag > fmt.max_normal) | torch.isinf(xd)
    mag = torch.where(over, _overflow_value(xd, fmt, rounding, saturate), mag)
    out = torch.where(torch.isnan(xd), xd, mag.copysign(xd))
    if fmt.special == "fnuz" or not fmt.signed:
        out = torch.where(out == 0, torch.zeros_like(out), out)
    if not fmt.signed:
        out = torch.where(xd < 0, torch.full_like(out, float("nan")) if not saturate
                          else torch.zeros_like(out), out)
    return out.to(torch.float32)


def _floor_log2(ax: torch.Tensor) -> torch.Tensor:
    """Exact ``floor(log2(ax))`` for positive finite fp64; 0 elsewhere."""
    _, e = torch.frexp(ax)
    valid = (ax > 0) & torch.isfinite(ax)
    return torch.where(valid, e.to(torch.int64) - 1, torch.zeros_like(e, dtype=torch.int64))


def _ldexp(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """``x * 2**e`` in fp64, exact whenever the result is representable and ``|e| <= 2046``.

    ``torch.ldexp`` multiplies by ``pow(2, e)``, which is inexact on CUDA (1.3984375 * 2**7 gave
    178.99999999999997 on an A100) and on the CPU once ``2**e`` is subnormal. Here each power of
    two is a normal fp64 built from its bit pattern, and the second factor covers shifts beyond one
    exponent range. A result below the subnormal grid may be rounded twice; the callers here only
    produce representable results.
    """
    e = e.to(torch.int64)
    first = e.clamp(-1022, 1023)
    return x.double() * _pow2(first) * _pow2((e - first).clamp(-1022, 1023))


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """``2.0**e`` for int64 ``e`` in [-1022, 1023], from the fp64 encoding."""
    return ((e + 1023) << 52).view(torch.float64)


def _round_up(frac, k, negative, rounding, noise, sr_bits, generator) -> torch.Tensor:
    """Whether to add one unit of the target quantum to the truncated magnitude."""
    half = frac == 0.5
    if rounding is Rounding.RNE:
        return (frac > 0.5) | (half & (torch.remainder(k, 2) == 1))
    if rounding is Rounding.RNA:
        return frac >= 0.5
    if rounding is Rounding.RTZ:
        return torch.zeros_like(frac, dtype=torch.bool)
    if rounding is Rounding.RUP:
        return (frac > 0) & ~negative
    if rounding is Rounding.RDN:
        return (frac > 0) & negative
    if rounding is Rounding.SR:
        if not 1 <= sr_bits <= 32:
            raise ValueError("sr_bits must be in [1, 32]")
        if noise is None:
            if generator is None:
                raise ValueError("stochastic rounding requires explicit noise or generator")
            noise = torch.randint(0, 2**32, frac.shape, generator=generator, device="cpu",
                                  dtype=torch.int64).to(frac.device)
        u = torch.floor(noise.to(torch.float64) / 2.0 ** (32 - sr_bits)) / 2.0**sr_bits
        return u < frac
    raise ValueError(f"unsupported rounding {rounding}")


def _overflow_value(xd, fmt: FloatFormat, rounding: Rounding, saturate: bool) -> torch.Tensor:
    maxv = torch.full_like(xd, fmt.max_normal)
    if saturate or fmt.special == "none":
        return maxv
    if fmt.special in ("fn", "fnuz"):
        return torch.full_like(xd, float("nan"))
    negative = xd < 0
    away = {Rounding.RTZ: torch.zeros_like(negative), Rounding.RUP: ~negative,
            Rounding.RDN: negative}.get(rounding, torch.ones_like(negative))
    return torch.where(away, torch.full_like(xd, float("inf")), maxv)


def _int_inf_bound(xd, fmt: IntFormat) -> torch.Tensor:
    lo = -fmt.qmin * 2.0**-fmt.frac_bits
    bound = torch.where(xd < 0, torch.full_like(xd, lo), torch.full_like(xd, fmt.max_normal))
    return bound if fmt.signed else torch.where(xd < 0, torch.zeros_like(xd), bound)


def _round_pow2(xd, fmt: Pow2Format, rounding: Rounding, saturate: bool) -> torch.Tensor:
    """Round positive values to a power of two. Rounding is applied to the value
    (not the exponent): RNE/RNA pick the nearer of 2**e and 2**(e+1), ties up.
    Values below 2**emin clamp to 2**emin; NaN, negatives and (non-saturating)
    overflow give NaN (E8M0 has no zero, sign or Inf)."""
    ax = xd.abs()
    e = _floor_log2(ax)
    lower = _ldexp(torch.ones_like(ax), e)
    frac = torch.where(torch.isfinite(ax), ax / lower - 1.0, torch.zeros_like(ax))  # in [0, 1)
    if rounding in (Rounding.RNE, Rounding.RNA):
        up = frac >= 0.5
    elif rounding in (Rounding.RTZ, Rounding.RDN):
        up = torch.zeros_like(frac, dtype=torch.bool)
    elif rounding is Rounding.RUP:
        up = frac > 0
    else:
        raise ValueError("stochastic rounding is not defined for power-of-two formats")
    e = (e + up.to(torch.int64)).clamp(min=fmt.emin)
    over = (e > fmt.emax) | torch.isinf(ax)
    out = _ldexp(torch.ones_like(ax), e.clamp(max=fmt.emax))
    out = torch.where(over, torch.full_like(out, fmt.max_normal), out)
    out = torch.where(ax == 0, torch.full_like(out, fmt.min_normal), out)
    bad = torch.isnan(xd) | (xd < 0) | (over & (not saturate))
    return torch.where(bad, torch.full_like(out, float("nan")), out).to(torch.float32)


# --------------------------------------------------------------------------
# Decoding grid values for the MMA datapath
# --------------------------------------------------------------------------

def decode(v: torch.Tensor, fmt: Format | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Split grid values into ``(negative, exponent, significand, radix)``.

    ``value = (-1)**negative * significand * 2**(exponent - radix)`` exactly, with
    the format-native (unnormalised) convention NADPE uses:

    * FloatFormat: ``exponent = max(floor(log2|v|), emin)``, ``radix = mbits``;
      normals have ``significand`` in ``[2**mbits, 2**(mbits+1))``, subnormals below.
    * IntFormat:   ``exponent = 0``, ``radix = frac_bits``, ``significand = |k|``.
    * Pow2Format:  ``exponent = log2 v``, ``radix = 0``, ``significand = 1``.

    Zeros decode to ``significand = 0``. Values must already lie on the grid
    (``ValueError`` otherwise). NaN/Inf are rejected; callers handle specials.
    """
    fmt = get_format(fmt)
    vd = v.to(torch.float64)
    if not torch.isfinite(vd).all():
        raise ValueError("decode() takes finite grid values; handle NaN/Inf before decoding")
    negative = vd < 0
    ax = vd.abs()
    if isinstance(fmt, FloatFormat):
        exp = _floor_log2(ax).clamp(min=fmt.emin)
        radix = fmt.mbits
    elif isinstance(fmt, IntFormat):
        exp = torch.zeros_like(ax, dtype=torch.int64)
        radix = fmt.frac_bits
    else:
        exp = _floor_log2(ax)
        radix = 0
    sig_f = _ldexp(ax, radix - exp)
    if not torch.equal(sig_f, torch.floor(sig_f)):
        raise ValueError(f"values are not on the {fmt} grid")
    if bool(_out_of_range(vd, ax, exp, sig_f, fmt).any()):
        raise ValueError(f"values are outside the representable range of {fmt}")
    sig = sig_f.to(torch.int64)
    exp = torch.where(sig == 0, torch.zeros_like(exp), exp)
    return negative, exp, sig, radix


def _out_of_range(vd, ax, exp, sig_f, fmt: Format) -> torch.Tensor:
    """Grid-aligned values the format still cannot encode (beyond max, disabled
    subnormals, negatives on unsigned grids, zero or non-powers on E8M0)."""
    if isinstance(fmt, FloatFormat):
        bad = ax > fmt.max_normal
        if not fmt.subnormals:
            bad |= (ax > 0) & (ax < fmt.min_normal)
        return bad | ((vd < 0) & (not fmt.signed))
    if isinstance(fmt, IntFormat):
        k = torch.where(vd < 0, -sig_f, sig_f)
        return (k < fmt.qmin) | (k > fmt.qmax)
    return (vd <= 0) | (sig_f != 1) | (exp < fmt.emin) | (exp > fmt.emax)
