"""Exact integer FDA/GDFS datapaths and IEEE FMA chains (ENGINE.md §4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..formats import FP32, Format, IntFormat, Pow2Format, container_dtype
from ..mma.operand import Operand
from ..mma.spec import MMASpec
from .cast import decode, round_to_format


@dataclass
class _Terms:
    negative: torch.Tensor
    exponent: torch.Tensor
    significand: torch.Tensor
    zero: torch.Tensor
    nan: torch.Tensor
    inf: torch.Tensor


def _shift(x: torch.Tensor, amount: int | torch.Tensor) -> torch.Tensor:
    """Signed shift of a nonnegative integer; over-wide shifts contribute zero."""
    if isinstance(amount, int):
        if abs(amount) >= 64:
            return torch.zeros_like(x)
        return x << amount if amount >= 0 else x >> -amount
    left = x << amount.clamp(0, 63)
    right = x >> (-amount).clamp(0, 63)
    return torch.where(amount.abs() >= 64, 0, torch.where(amount >= 0, left, right))


def _log2(x: torch.Tensor) -> torch.Tensor:
    """Integer floor-log2, without an inexact int64-to-double conversion."""
    out = torch.zeros_like(x)
    for step in (32, 16, 8, 4, 2, 1):
        high = x >= 1 << step
        out = out + high.to(torch.int64) * step
        x = torch.where(high, x >> step, x)
    return out


def _rounded_shift(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    kept = _shift(x, amount)
    discard = (-amount).clamp(0, 63)
    remainder = x - (kept << discard)
    half = torch.ones_like(x) << (discard - 1).clamp(0, 62)
    up = (remainder > half) | ((remainder == half) & ((kept & 1) != 0))
    return kept + ((amount < 0) & (amount > -64) & up).to(torch.int64)


def _mul_shift(a: torch.Tensor, b: torch.Tensor, amount: int) -> torch.Tensor:
    """Exact multiply then shift, including products wider than int64.

    Inputs and the final result have validated 62-bit headroom. Four base-2^31
    limbs avoid overflowing the intermediate product of two fp32 scales.
    """
    if amount >= 0:
        return (a * b) << amount
    mask = (1 << 31) - 1
    a0, a1, b0, b1 = a & mask, a >> 31, b & mask, b >> 31
    low = a0 * b0
    middle = (low >> 31) + a1 * b0 + a0 * b1
    high = a1 * b1 + (middle >> 31)
    limbs = (low & mask, middle & mask, high & mask, high >> 31)
    out = torch.zeros_like(low)
    for i, limb in enumerate(limbs):
        out = out | _shift(limb, amount + 31 * i)
    return out


def _decoded(values: torch.Tensor, fmt: Format) -> tuple[_Terms, int]:
    finite = torch.isfinite(values)
    # NaN/Inf slots are flagged below; decode a valid grid value in their place
    # (0 is not on every grid — E8M0 has no zero).
    placeholder = torch.full_like(values, fmt.min_normal)
    negative, exponent, significand, radix = decode(torch.where(finite, values, placeholder), fmt)
    return _Terms(torch.signbit(values) | negative, exponent, significand, values == 0,
                  torch.isnan(values), torch.isinf(values)), radix


def _fp32_terms(values: torch.Tensor, radix: int) -> _Terms:
    terms, _ = _decoded(values, FP32)
    # Unlike native-format operands, the running fp32 register normalizes subnormals.
    leading = _log2(terms.significand)
    terms.exponent = terms.exponent + leading - 23
    terms.significand = _shift(terms.significand, radix - leading)
    return terms


def _combine(a: _Terms, b: _Terms, significand: torch.Tensor) -> _Terms:
    nan = a.nan | b.nan | (a.inf & b.zero) | (b.inf & a.zero)
    return _Terms(a.negative ^ b.negative, a.exponent + b.exponent, significand,
                  a.zero | b.zero, nan, (a.inf | b.inf) & ~nan)


def _slice(terms: _Terms, index: tuple) -> _Terms:
    return _Terms(*(getattr(terms, field)[index] for field in _Terms.__dataclass_fields__))


def _sum_terms(terms: _Terms) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = ~(terms.zero | terms.nan | terms.inf)
    exponent = torch.where(active, terms.exponent, -4096).amax(dim=-1)
    magnitude = _shift(terms.significand, terms.exponent - exponent.unsqueeze(-1))
    signed = torch.where(terms.negative, -magnitude, magnitude)
    total = torch.where(active, signed, 0).sum(dim=-1)
    pos = (terms.inf & ~terms.negative).any(dim=-1)
    neg = (terms.inf & terms.negative).any(dim=-1)
    nan = terms.nan.any(dim=-1) | (pos & neg)
    special = torch.where(neg, -float("inf"), float("inf"))
    special = torch.where(nan, float("nan"), special)
    return total, exponent, nan | pos | neg, special


def _fixed_to_fp32(
    total: torch.Tensor, exponent: torch.Tensor, radix: int, norm: str, *, fda: bool = True,
) -> torch.Tensor:
    magnitude = total.abs()
    leading = _log2(magnitude)
    biased = leading + exponent - radix + 127
    precision = min(radix, 23) if fda else 23
    amount = precision - leading
    kept = _rounded_shift(magnitude, amount) if norm == "rne" else _shift(magnitude, amount)
    carry = kept >= 1 << (precision + 1)
    kept = torch.where(carry, kept >> 1, kept)
    normal_exp = biased + carry.to(torch.int64)
    normal = (normal_exp << 23) | ((kept << (23 - precision)) & 0x7FFFFF)
    normal = torch.where(normal_exp >= 255, 0x7F800000, normal)
    sub_shift = 149 + exponent - radix
    # FDA's subnormal path retains all available bits, but always truncates.
    sub = _shift(magnitude, sub_shift) if fda else _rounded_shift(magnitude, sub_shift)
    bits = torch.where(biased <= 0, sub, normal) | ((total < 0).to(torch.int64) << 31)
    bits = torch.where(total == 0, 0, bits)
    return bits.to(torch.int32).contiguous().view(torch.float32)


def _fda(terms: _Terms, c: torch.Tensor, radix: int, norm: str) -> torch.Tensor:
    accumulator = _fp32_terms(c.unsqueeze(-1), radix)
    # Each term field may be broadcast along M or N; expand before appending c.
    fields = []
    for field in _Terms.__dataclass_fields__:
        t = getattr(terms, field).expand(*c.shape, terms.significand.shape[-1])
        fields.append(torch.cat((t, getattr(accumulator, field)), dim=-1))
    joined = _Terms(*fields)
    total, exponent, exceptional, special = _sum_terms(joined)
    result = _fixed_to_fp32(total, exponent, radix, norm)
    result = torch.where(joined.zero.all(dim=-1), c, result)
    return torch.where(exceptional, special, result)


def fp32_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """IEEE binary32 FMA: exact double product, TwoSum residual, one rounding."""
    product, addend = a.double() * b.double(), c.double()
    value = product + addend
    z = value - product
    residual = (product - (value - z)) + (addend - z)
    rounded = value.float()
    nearest = rounded.double()
    direction = torch.where(value > nearest, float("inf"), -float("inf"))
    neighbor = torch.nextafter(rounded, direction).double()
    midpoint = (value - nearest).abs() == (neighbor - nearest).abs() * 0.5
    midpoint |= value.abs() == 2.0**128 - 2.0**103  # finite/Inf rounding boundary
    correction = torch.where(residual > 0, float("inf"), -float("inf")).double()
    corrected = torch.nextafter(value, correction).float()
    return torch.where(midpoint & (residual != 0) & torch.isfinite(value), corrected, rounded)


def _integer_bits(fmt: Format) -> int:
    if isinstance(fmt, IntFormat):
        return max(abs(fmt.qmin), fmt.qmax).bit_length() + max(0, -fmt.frac_bits)
    return 1


def _headroom(radix: int, integer_bits: int, width: int, label: str) -> None:
    bits = radix + integer_bits + width.bit_length()  # ceil(log2(width + 1))
    if bits > 62:
        raise ValueError(f"{label} int64 headroom exceeded: {radix} + {integer_bits} + "
                         f"ceil(log2({width}+1)) = {bits} > 62")


def resolve_scale_apply(spec: MMASpec, a: Operand, b: Operand) -> str:
    """Resolve §4.6, rejecting incompatible modes and crossing scale domains."""
    k_scaled = [op for op in (a, b) if op.scale_kind == "k"]
    expected = {"gdfs": "group", "cofda": "promote" if spec.promote_interval else "product",
                "fp32_fma": "operand", "fp64": "operand", "int_exact": "epilogue"}[spec.algorithm]
    mode = expected if k_scaled else "epilogue"
    if spec.scale_apply != "auto":
        mode = spec.scale_apply
        if mode not in ("epilogue", expected):
            raise ValueError(f"scale_apply={mode!r} is incompatible with {spec.algorithm} "
                             f"and promote_interval={spec.promote_interval}; expected {expected!r}")
    if k_scaled and (mode == "epilogue" or spec.algorithm == "int_exact"):
        raise ValueError(f"{spec.algorithm}: K-varying scales cannot be applied in the epilogue")
    span = spec.group_size if mode == "group" else spec.promote_interval if mode == "promote" else 0
    for op in k_scaled:
        if span and any(start // op.k_domain != (min(start + span, op.K) - 1) // op.k_domain
                        for start in range(0, op.K, span)):
            raise ValueError(f"{mode} interval/group size {span} crosses K scale domain {op.k_domain}")
    return mode


def _validate(a: Operand, b: Operand, spec: MMASpec, mode: str, bias: torch.Tensor | None) -> None:
    if a.K != b.K:
        raise ValueError(f"operand K dimensions differ: {a.K} != {b.K}")
    if a.values.device != b.values.device:
        raise ValueError("operands must be on the same device")
    for op in (a, b):
        if op.scale_kind not in ("none", "tensor", "row", "k"):
            raise ValueError(f"unknown scale_kind {op.scale_kind!r}")
        if op.scale is not None:
            shape = {"tensor": (), "row": (op.rows, 1),
                     "k": (op.rows, (op.K + max(op.k_domain, 1) - 1) // max(op.k_domain, 1))}[op.scale_kind]
            if tuple(op.scale.shape) != shape:
                raise ValueError(f"{op.scale_kind} scale shape must be {shape}")
            if op.scale.device != op.values.device:
                raise ValueError("scale and operand must be on the same device")
        if op.alpha is not None and (op.alpha.numel() != 1 or op.alpha.device != op.values.device):
            raise ValueError("alpha must be a scalar on the operand device")
    if bias is not None and (bias.shape != (b.rows,) or bias.device != a.values.device):
        raise ValueError("bias must have shape [N] on the operand device")
    bits = _integer_bits(a.fmt) + _integer_bits(b.fmt)
    scale_bits = sum(_integer_bits(op.scale_fmt or FP32) for op in (a, b)
                     if op.scale_kind == "k" and not isinstance(op.scale_fmt, Pow2Format))
    if spec.algorithm == "cofda":
        _headroom(spec.f_bits, bits + (scale_bits if mode == "product" else 0), spec.chunk_size, "CoFDA")
        if spec.c_mode == "decoupled" and not spec.promote_interval:
            _headroom(spec.f2_bits, 1, 1, "CoFDA F2")
    elif spec.algorithm == "gdfs":
        _headroom(spec.g_bits, bits, spec.group_size, "GDFS group")
        group_bits = bits + (spec.group_size - 1).bit_length() + scale_bits
        _headroom(spec.f_bits, group_bits, spec.groups_per_tile, "GDFS tile")
    elif spec.algorithm == "int_exact":
        if not isinstance(a.fmt, IntFormat) or not isinstance(b.fmt, IntFormat):
            raise ValueError("int_exact requires two integer-format operands")
        _headroom(0, bits, a.K, "int_exact")


def _scale_terms(op: Operand, start: int, end: int, side: str) -> tuple[_Terms, int] | None:
    if op.scale_kind != "k":
        return None
    index = torch.arange(start, end, device=op.values.device) // op.k_domain
    scale = op.scale[:, index].float()
    fmt = op.scale_fmt or FP32
    terms, radix = _decoded(scale, fmt)
    if isinstance(fmt, Pow2Format) and fmt.ebits == 8 and fmt.bias == 127:
        terms.zero = terms.zero | (scale == 2.0**-127)
        terms.significand = torch.where(terms.zero, 0, terms.significand)
    return _slice(terms, (slice(None), None, slice(None)) if side == "a"
                  else (None, slice(None), slice(None))), radix


def _apply_scales(
    terms: _Terms, radix: int, target: int, a: Operand, b: Operand, start: int, end: int,
    *, group: bool = False,
) -> _Terms:
    factors = [_scale_terms(op, start, end, side) for op, side in ((a, "a"), (b, "b"))]
    factors = [factor for factor in factors if factor is not None]
    if not factors:
        terms.significand = _shift(terms.significand, target - radix)
        return terms
    scale, sr = factors[0]
    if len(factors) == 2:
        other, other_radix = factors[1]
        scale = _combine(scale, other, scale.significand * other.significand)
        sr += other_radix
    result = _combine(terms, scale, _mul_shift(terms.significand, scale.significand, target - radix - sr))
    if group:
        # GDFS's explicit zero rule precedes Inf multiplication, but never masks NaN.
        nan, zero = terms.nan, terms.zero
        for factor, _ in factors:
            nan, zero = nan | factor.nan, zero | factor.zero
        result.nan, result.zero = nan, zero & ~nan
        result.inf = (terms.inf | scale.inf) & ~(nan | zero)
    return result


def _products(a: _Terms, b: _Terms, start: int, end: int) -> _Terms:
    at = _slice(a, (slice(None), None, slice(start, end)))
    bt = _slice(b, (None, slice(None), slice(start, end)))
    return _combine(at, bt, at.significand * bt.significand)


def _cofda(a: Operand, b: Operand, spec: MMASpec, mode: str, c: torch.Tensor) -> torch.Tensor:
    at, ar = _decoded(a.values, a.fmt)
    bt, br = _decoded(b.values, b.fmt)
    interval = spec.promote_interval or max(a.K, 1)
    acc = c
    for begin in range(0, a.K, interval):
        partial = torch.zeros_like(c) if spec.promote_interval else acc
        for start in range(begin, min(begin + interval, a.K), spec.chunk_size):
            end = min(start + spec.chunk_size, begin + interval, a.K)
            terms = _products(at, bt, start, end)
            if mode == "product":
                terms = _apply_scales(terms, ar + br, spec.f_bits, a, b, start, end)
            else:
                terms.significand = _shift(terms.significand, spec.f_bits - ar - br)
            if spec.c_mode == "decoupled" and not spec.promote_interval:
                p = _fda(terms, torch.zeros_like(c), spec.f_bits, spec.norm_rounding)
                partial = _fda(_fp32_terms(p.unsqueeze(-1), spec.f2_bits), partial,
                               spec.f2_bits, spec.norm_rounding)
            else:
                partial = _fda(terms, partial, spec.f_bits, spec.norm_rounding)
        if spec.promote_interval:
            weight = torch.ones_like(c)
            if a.scale_kind == "k":
                weight = weight * a.scale[:, begin // a.k_domain, None].float()
            if b.scale_kind == "k":
                weight = weight * b.scale[None, :, begin // b.k_domain].float()
            acc = fp32_fma(partial, weight, acc)
        else:
            acc = partial
    return acc


def _gdfs(a: Operand, b: Operand, spec: MMASpec, c: torch.Tensor) -> torch.Tensor:
    at, ar = _decoded(a.values, a.fmt)
    bt, br = _decoded(b.values, b.fmt)
    for tile in range(0, a.K, spec.k_tile):
        groups = []
        for start in range(tile, min(tile + spec.k_tile, a.K), spec.group_size):
            end = min(start + spec.group_size, a.K)
            products = _products(at, bt, start, end)
            products.significand = _shift(products.significand, spec.g_bits - ar - br)
            total, exponent, exceptional, special = _sum_terms(products)
            group = _Terms(total < 0, exponent, total.abs(), (total == 0) & ~exceptional,
                           torch.isnan(special) & exceptional, torch.isinf(special) & exceptional)
            group.negative = torch.where(group.inf, torch.signbit(special), group.negative)
            group = _slice(group, (..., None))
            groups.append(_apply_scales(group, spec.g_bits, spec.f_bits, a, b, start, start + 1, group=True))
        fields = [torch.cat([getattr(g, field).expand(*c.shape, 1) for g in groups], dim=-1)
                  for field in _Terms.__dataclass_fields__]
        c = _fda(_Terms(*fields), c, spec.f_bits, spec.norm_rounding)
    return c


def gemm_reference(a: Operand, b: Operand, spec: MMASpec, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ``a @ b.T`` using the specified arithmetic, independently per output."""
    mode = resolve_scale_apply(spec, a, b)
    _validate(a, b, spec, mode, bias)
    acc = torch.zeros((a.rows, b.rows), dtype=torch.float32, device=a.values.device)
    if spec.algorithm == "cofda":
        acc = _cofda(a, b, spec, mode, acc)
    elif spec.algorithm == "gdfs":
        acc = _gdfs(a, b, spec, acc)
    elif spec.algorithm == "int_exact":
        at, _ = _decoded(a.values, a.fmt)
        bt, _ = _decoded(b.values, b.fmt)
        if (at.nan | at.inf).any() or (bt.nan | bt.inf).any():
            raise ValueError("int_exact requires finite integer operands")
        total = torch.zeros_like(acc, dtype=torch.int64)
        for start in range(0, a.K, 32):
            terms = _products(at, bt, start, min(start + 32, a.K))
            total += torch.where(terms.negative, -terms.significand, terms.significand).sum(-1)
        exponent = torch.full_like(total, -a.fmt.frac_bits - b.fmt.frac_bits)
        acc = _fixed_to_fp32(total, exponent, 0, "rne", fda=False)
    else:
        av, bv = a.values.float(), b.values.float()
        if mode == "operand":
            if a.scale_kind == "k":
                av = av * a.scale_per_element().float()
            if b.scale_kind == "k":
                bv = bv * b.scale_per_element().float()
        if spec.algorithm == "fp64":
            acc = acc.double()
        for k in range(a.K):
            ak, bk = av[:, k, None], bv[None, :, k]
            acc = fp32_fma(ak, bk, acc) if spec.algorithm == "fp32_fma" else ak.double() * bk.double() + acc
        acc = acc.float()
    if b.scale_kind in ("tensor", "row"):
        acc = b.scale.float().reshape(1, -1) * acc
    if a.scale_kind in ("tensor", "row"):
        acc = a.scale.float().reshape(-1, 1) * acc
    alpha = None
    for op in (a, b):
        if op.alpha is not None:
            alpha = op.alpha.float() if alpha is None else alpha * op.alpha.float()
    if alpha is not None:
        acc = alpha * acc
    if bias is not None:
        acc = acc + bias.float()
    return round_to_format(acc, spec.out_format, "rne", saturate=False).to(container_dtype(spec.out_format))
