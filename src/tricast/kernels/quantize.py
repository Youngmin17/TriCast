"""Triton scale-domain reductions and integer-bit element rounding.

MSE, percentile and zero-point scale selection call the reference on GPU
tensors; their element rounding still runs in Triton. Two-level tensor amax
and d2 use torch fp32 arithmetic before domain kernels (as tricast.reference.quantize).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..formats import Format, container_dtype
from ..quant.qtensor import QTensor
from ..quant.spec import QuantSpec
from ..rounding import Rounding
from .cast_core import (
    _bits,
    _encode,
    _float_parts,
    _floor_log2_bits,
    _ilog2,
    fmt_constexprs,
    round_to_format,
)
from .ieee import add_rn, div_rn, mul_rn


@triton.jit
def _cast_kernel(
    X, Y, Noise, N, HAS_NOISE: tl.constexpr, SEED: tl.constexpr,
    FMT: tl.constexpr, ROUNDING: tl.constexpr, SATURATE: tl.constexpr,
    SR_BITS: tl.constexpr, B: tl.constexpr,
):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + offset, offset < N, 0)
    if HAS_NOISE:
        noise = tl.load(Noise + offset, offset < N, 0).to(tl.uint32)
    else:
        noise = tl.randint(SEED, offset)
    y = round_to_format(x, noise, FMT, ROUNDING, SATURATE, SR_BITS)
    tl.store(Y + offset, y, offset < N)


@triton.jit
def _domain_amax_kernel(
    X, Partial, ROWS, K, DR, DC,
    NC, ND, NP, BD: tl.constexpr, B: tl.constexpr,
):
    part = tl.program_id(0) % NP
    domain = (tl.program_id(0) // NP).to(tl.int64) * BD + tl.arange(0, BD)
    element = part.to(tl.int64) * B + tl.arange(0, B)
    row = (domain // NC)[:, None] * DR + (element // DC)[None, :]
    col = (domain % NC)[:, None] * DC + (element % DC)[None, :]
    valid = (domain[:, None] < ND) & (element[None, :] < DR * DC) & (row < ROWS) & (col < K)
    x = tl.load(X + row * K + col, valid, 0)
    # Positive fp32 bit order preserves subnormals and propagates NaNs in max.
    absolute = x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    maximum = tl.max(absolute, 1)
    tl.store(Partial + domain * NP + part, maximum, domain < ND)


@triton.jit
def _scale_kernel(
    Partial, Scale, Global, ScaleNoise, ND, NP,
    SFMT: tl.constexpr, SRND: tl.constexpr, METHOD: tl.constexpr,
    MAX_ELEM: tl.constexpr, MAX_ELEM_SIG: tl.constexpr, EMAX_ELEM: tl.constexpr,
    MIN_SCALE_BITS: tl.constexpr, TWO_LEVEL: tl.constexpr, HAS_SCALE_NOISE: tl.constexpr,
    BD: tl.constexpr, B: tl.constexpr,
):
    domain = tl.program_id(0) * BD + tl.arange(0, BD)
    part = tl.arange(0, B)
    maxima = tl.full((BD, B), 0, tl.uint32)
    for start in range(0, NP, B):
        value = tl.load(Partial + domain[:, None] * NP + start + part[None, :],
                        (domain[:, None] < ND) & (start + part[None, :] < NP), 0)
        maxima = tl.maximum(maxima, value)
    abits = tl.max(maxima, 1)
    absolute = abits & 0x7FFFFFFF
    maximum = abits.to(tl.float32, bitcast=True)
    if METHOD == "absmax" or TWO_LEVEL:  # two-level: every method is (amax / M) / d2
        raw = div_rn(maximum, tl.full(maximum.shape, MAX_ELEM, tl.float32))
        if TWO_LEVEL:
            raw = div_rn(raw, tl.broadcast_to(tl.load(Global), raw.shape))
        noise = tl.full((BD,), 0, tl.uint32)
        if HAS_SCALE_NOISE:
            noise = tl.load(ScaleNoise + domain, domain < ND, 0).to(tl.uint32)
        scale = round_to_format(raw, noise, SFMT, SRND, True, 32)
        sbits = scale.to(tl.uint32, bitcast=True)
        sbits = tl.where((sbits & 0x7FFFFFFF) == 0, MIN_SCALE_BITS, sbits)
        sbits = tl.where(absolute == 0, 0x3F800000, sbits)
    else:
        # Zero uses the smallest normal fp32 as in microxcaling.
        adjusted = tl.where(absolute == 0, 0x00800000, absolute).to(tl.uint32)
        exponent = _floor_log2_bits(adjusted) - EMAX_ELEM
        if METHOD == "pow2_ceil":
            sig, _ = _float_parts(adjusted)
            normalized = sig << tl.minimum(tl.maximum(23 - _ilog2(sig), 0), 31)
            exponent += ((normalized > MAX_ELEM_SIG) & ((abits & 0x80000000) == 0)).to(tl.int32)
        overflow = (exponent > SFMT[3]) | (absolute >= 0x7F800000)
        exponent = tl.minimum(tl.maximum(exponent, SFMT[2]), SFMT[3])
        sbits = _encode(tl.full((BD,), 1, tl.uint32), exponent)
        sbits = tl.where(overflow, 0x7FC00000, sbits)
    tl.store(Scale + domain, sbits.to(tl.uint32).to(tl.float32, bitcast=True), domain < ND)


@triton.jit
def _quantize_kernel(
    X, Values, Scale, Zero, Global, Noise, N, K,
    DR, DC, NC,
    HAS_SCALE: tl.constexpr, HAS_ZERO: tl.constexpr, FLOAT_ZERO: tl.constexpr, TWO_LEVEL: tl.constexpr,
    HAS_NOISE: tl.constexpr, SEED: tl.constexpr, FMT: tl.constexpr,
    ROUNDING: tl.constexpr, SATURATE: tl.constexpr, SR_BITS: tl.constexpr,
    QMIN: tl.constexpr, QMAX: tl.constexpr, B: tl.constexpr,
):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + offset, offset < N, 0)
    domain = (offset // K // DR) * NC + (offset % K // DC)
    if HAS_ZERO:
        zero = tl.load(Zero + domain, offset < N, 0)
        if FLOAT_ZERO:
            x = add_rn(x, -zero)
    if HAS_SCALE:
        scale = tl.load(Scale + domain, offset < N, 1)
        if TWO_LEVEL:
            scale = mul_rn(scale, tl.broadcast_to(tl.load(Global), scale.shape))
        x = div_rn(x, scale)
    if HAS_NOISE:
        noise = tl.load(Noise + offset, offset < N, 0).to(tl.uint32)
    else:
        noise = tl.randint(SEED, offset)
    if HAS_ZERO:
        # Integer zero points are added only after rounding on the unbounded grid.
        wide: tl.constexpr = (1, 0, 0, 0, 0x7F7FFFFF, 0x3F800000, True, True, "none", 0x7F7FFFFF, 0)
        rounded = round_to_format(x, noise, wide, ROUNDING, True, SR_BITS)
        rounded = tl.where((x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) == 0x7F800000, x, rounded)
        if not FLOAT_ZERO:
            rounded = rounded.to(tl.float64) + zero.to(tl.float64)
        values = tl.minimum(tl.maximum(rounded, QMIN, propagate_nan=tl.PropagateNan.ALL),
                            QMAX, propagate_nan=tl.PropagateNan.ALL).to(tl.float32)
    else:
        values = round_to_format(x, noise, FMT, ROUNDING, SATURATE, SR_BITS)
    tl.store(Values + offset, values, offset < N)


def _input(x: torch.Tensor) -> torch.Tensor:
    if x.numel() >= 2**31:
        raise ValueError("Triton quantization requires fewer than 2^31 input elements (int32 indexing)")
    if not x.is_cuda:
        raise ValueError("the Triton backend requires a CUDA tensor")
    return x.to(torch.float32).contiguous()


def _noise(noise: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
    if noise is None:
        return None
    return torch.broadcast_to(noise.to(device=x.device, dtype=torch.int64), x.shape).contiguous()


def round_to_format_triton(
    x: torch.Tensor, fmt: Format | str, rounding: Rounding | str = Rounding.RNE, *,
    saturate: bool = True, noise: torch.Tensor | None = None, sr_bits: int = 32,
) -> torch.Tensor:
    """Cast an fp32 input block; supplied uint32 noise is broadcast to its shape."""
    constants = fmt_constexprs(fmt, rounding, saturate)
    if not 1 <= sr_bits <= 32:
        raise ValueError("sr_bits must be in [1, 32]")
    x = _input(x)
    noise = _noise(noise, x)
    out = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            _cast_kernel[(triton.cdiv(x.numel(), 256),)](
                x, out, noise, x.numel(), noise is not None, 0,
                **constants, SR_BITS=sr_bits, B=256, enable_fp_fusion=False,
            )
    return out


def _domain_layout(spec: QuantSpec, rows: int, k: int) -> tuple[int, int, tuple[int, ...]]:
    if spec.granularity == "tensor":
        return rows, k, ()
    if spec.granularity == "row":
        return 1, k, (rows, 1)
    if spec.granularity == "group":
        return 1, spec.group_size, (rows, triton.cdiv(k, spec.group_size))
    br, bc = spec.block
    return br, bc, (triton.cdiv(rows, br), triton.cdiv(k, bc))


def quantize_triton(
    x: torch.Tensor, spec: QuantSpec, *, amax: torch.Tensor | None = None,
    noise: torch.Tensor | None = None, seed: int = 0, scale_noise: torch.Tensor | None = None,
) -> QTensor:
    """Quantize domains with Triton; reference scale search stays on the GPU.

    MSE/percentile/zero-point selection uses ``compute_scale``. No reference
    element cast is used on the normal absmax/pow2 paths. Scale SR requires a
    separate explicit ``scale_noise`` tensor broadcastable to the scale layout.
    """
    if x.ndim == 0 or x.numel() == 0:
        raise ValueError("quantization needs a nonempty tensor of shape [..., K]")
    from ..reference.quantize import _validate_scale_spec

    _validate_scale_spec(spec, scale_noise)
    shape = tuple(x.shape)
    x = _input(x)
    noise = _noise(noise, x)
    x2d = x.reshape(-1, shape[-1])
    rows, k = x2d.shape
    dr, dc, layout = _domain_layout(spec, rows, k)
    if dr * dc >= 2**31:
        raise ValueError("Triton quantization requires domains smaller than 2^31 elements (int32 indexing)")
    nc = triton.cdiv(k, dc)
    nd = triton.cdiv(rows, dr) * nc
    if amax is not None:
        if spec.granularity != "tensor" or amax.numel() != 1:
            raise ValueError("amax override requires tensor granularity and a scalar")
        amax = amax.to(device=x.device, dtype=torch.float32).reshape(())
    scale = zero = global_scale = None
    if spec.scale is not None and scale_noise is not None:
        scale_noise = torch.broadcast_to(scale_noise.to(device=x.device, dtype=torch.int64),
                                         layout).contiguous()
    sf = spec.scale
    with torch.cuda.device(x.device):
        if sf is not None:
            if sf.method in ("mse", "percentile") or spec.zero_point != "none":
                from ..reference.quantize import compute_scale

                scale, zero, global_scale = compute_scale(
                    x2d, spec, amax, noise=None if noise is None else noise.reshape(x2d.shape),
                    scale_noise=scale_noise,
                )
            else:
                if sf.two_level:
                    # fp64 then one rounding = IEEE fp32 division (CUDA tensor/scalar is inexact).
                    tensor_amax = x2d.abs().amax() if amax is None else amax
                    denominator = spec.format.max_normal * sf.format.max_normal
                    global_scale = (tensor_amax.double() / denominator).float()
                    global_scale = torch.where(global_scale == 0, 1.0, global_scale)
                if amax is None:
                    b = min(triton.next_power_of_2(dr * dc), 1024)
                    npart = triton.cdiv(dr * dc, b)
                    if nd * npart >= 2**31:
                        raise ValueError("Triton amax requires fewer than 2^31 partials (int32 indexing)")
                    partial = torch.empty((nd, npart), dtype=torch.uint32, device=x.device)
                    _domain_amax_kernel[(triton.cdiv(nd, 4) * npart,)](
                        x2d, partial, rows, k, dr, dc, nc, nd, npart, 4, b,
                    )
                else:
                    partial, npart = amax.reshape(1, 1).view(torch.uint32), 1
                scale = torch.empty(layout, dtype=torch.float32, device=x.device)
                sfmt = fmt_constexprs(sf.format, sf.rounding)["FMT"]
                minimum = getattr(sf.format, "min_subnormal", sf.format.min_normal)
                max_bits = _bits(spec.format.max_normal)
                max_sig = (max_bits & 0x7FFFFF) | (0x800000 if max_bits >> 23 else 0)
                max_sig <<= max(23 - (max_sig.bit_length() - 1), 0)
                _scale_kernel[(triton.cdiv(nd, 4),)](
                    partial, scale, global_scale, scale_noise, nd, npart, sfmt, sf.rounding.code, sf.method,
                    spec.format.max_normal, max_sig, spec.emax_elem,
                    _bits(minimum), sf.two_level, scale_noise is not None, 4,
                    min(triton.next_power_of_2(npart), 256),
                    enable_fp_fusion=False,
                )
        values = torch.empty((rows, k), device=x.device, dtype=container_dtype(spec.format))
        constants = fmt_constexprs(spec.format, spec.rounding, spec.saturate)
        _quantize_kernel[(triton.cdiv(x.numel(), 256),)](
            x2d, values, scale, zero, global_scale, noise, x.numel(), k, dr, dc, nc,
            scale is not None, zero is not None, spec.zero_point == "float", global_scale is not None,
            noise is not None, seed,
            **constants, SR_BITS=spec.sr_bits, QMIN=getattr(spec.format, "qmin", 0),
            QMAX=getattr(spec.format, "qmax", 0), B=256, enable_fp_fusion=False,
        )
    return QTensor(values, scale, zero, global_scale, spec, shape)
