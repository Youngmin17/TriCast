"""Reference quantization with explicit fp32 scale arithmetic."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch

from ..formats import container_dtype
from ..quant.qtensor import QTensor
from ..quant.spec import QuantSpec
from ..rounding import Rounding
from .cast import _ldexp, _round_up, round_to_format


def _div(a, b) -> torch.Tensor:
    """IEEE binary32 a / b on any device: fp64 then one rounding to fp32 is exact for
    fp32 operands (53 >= 2*24 + 2). CUDA's tensor/scalar division multiplies by a
    reciprocal, so a plain fp32 ``/`` would make the reference device-dependent."""
    a = torch.as_tensor(a)
    return (a.double() / (b.double() if torch.is_tensor(b) else float(b))).float()


def _mul(a, b) -> torch.Tensor:
    """IEEE binary32 a * b on any device (see ``_div``)."""
    a = torch.as_tensor(a)
    return (a.double() * (b.double() if torch.is_tensor(b) else float(b))).float()


def _sub(a: torch.Tensor | float, b: torch.Tensor | float) -> torch.Tensor:
    """IEEE binary32 a - b on any device (see ``_div``)."""
    a = torch.as_tensor(a)
    return (a.double() - (b.double() if torch.is_tensor(b) else float(b))).float()


def _add(a: torch.Tensor | float, b: torch.Tensor | float) -> torch.Tensor:
    """IEEE binary32 a + b on any device (see ``_div``)."""
    a = torch.as_tensor(a)
    return (a.double() + (b.double() if torch.is_tensor(b) else float(b))).float()


def view_2d(x: torch.Tensor) -> torch.Tensor:
    """View the leading dimensions as rows and the final dimension as K."""
    if x.ndim == 0 or x.numel() == 0:
        raise ValueError("quantization needs a nonempty tensor of shape [..., K]")
    return x.to(torch.float32).reshape(-1, x.shape[-1])


def _layout(spec: QuantSpec, rows: int, K: int) -> tuple[int, ...]:
    if spec.granularity == "tensor":
        return ()
    if spec.granularity == "row":
        return rows, 1
    if spec.granularity == "group":
        return rows, (K + spec.group_size - 1) // spec.group_size
    br, bc = spec.block
    return (rows + br - 1) // br, (K + bc - 1) // bc


def _domains(spec: QuantSpec, rows: int, K: int) -> Iterator[tuple[tuple, tuple]]:
    if spec.granularity == "tensor":
        yield (), (slice(None), slice(None))
        return
    br, bc = {"row": (1, K), "group": (1, spec.group_size), "block": spec.block}[spec.granularity]
    for r in range(0, rows, br):
        for k in range(0, K, bc):
            yield (r // br, k // bc), (slice(r, r + br), slice(k, k + bc))


def _domain_view(x2d: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack domains in row-major order; the mask excludes ragged-edge padding."""
    rows, k = x2d.shape
    if spec.granularity == "tensor":
        data = x2d.reshape(1, -1)
        return data, torch.ones_like(data, dtype=torch.bool)
    br, bc = {"row": (1, k), "group": (1, spec.group_size), "block": spec.block}[spec.granularity]
    br, bc = min(br, rows), min(bc, k)
    nr, nc = (rows + br - 1) // br, (k + bc - 1) // bc
    padding = (0, nc * bc - k, 0, nr * br - rows)
    padded = torch.nn.functional.pad(x2d, padding)
    valid = torch.nn.functional.pad(torch.ones_like(x2d, dtype=torch.bool), padding)
    shape = (nr, br, nc, bc)
    return (padded.reshape(shape).permute(0, 2, 1, 3).reshape(nr * nc, br * bc),
            valid.reshape(shape).permute(0, 2, 1, 3).reshape(nr * nc, br * bc))


def _domain_extrema(x2d: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch at most four rectangle shapes without changing domain reduction strides.

    Padding or copying domains changes amin/amax's signed-zero tie behavior.
    Interior, right edge, bottom edge and corner retain their original strides.
    """
    rows, k = x2d.shape
    br, bc = ((rows, k) if spec.granularity == "tensor" else
              {"row": (1, k), "group": (1, spec.group_size), "block": spec.block}[spec.granularity])
    br, bc = min(br, rows), min(bc, k)
    full_rows, tail_rows = divmod(rows, br)
    full_cols, tail_cols = divmod(k, bc)
    lo = x2d.new_empty((full_rows + bool(tail_rows), full_cols + bool(tail_cols)))
    hi = torch.empty_like(lo)
    sr, sc = x2d.stride()
    row_classes = ((0, full_rows, br), (full_rows * br, int(tail_rows > 0), tail_rows))
    col_classes = ((0, full_cols, bc), (full_cols * bc, int(tail_cols > 0), tail_cols))
    for r0, nr, height in row_classes:
        for c0, nc, width in col_classes:
            if nr == 0 or nc == 0:
                continue
            domains = x2d.as_strided(
                (nr, nc, height, width), (br * sr, bc * sc, sr, sc),
                storage_offset=x2d.storage_offset() + r0 * sr + c0 * sc,
            )
            index = (slice(r0 // br, r0 // br + nr), slice(c0 // bc, c0 // bc + nc))
            lo[index], hi[index] = domains.amin(dim=(2, 3)), domains.amax(dim=(2, 3))
    return lo.reshape(-1), hi.reshape(-1)


def compute_amax(x2d: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Maximum magnitude per scale domain, propagating NaNs."""
    data, valid = _domain_view(x2d.float(), spec)
    return data.abs().masked_fill(~valid, 0).amax(dim=1).reshape(_layout(spec, *x2d.shape))


def expand_scale(t: torch.Tensor, spec: QuantSpec, rows: int, K: int) -> torch.Tensor:
    """Expand a scale or zero-point layout to one value per element."""
    if spec.granularity in ("tensor", "row"):
        return t.expand(rows, K)
    if spec.granularity == "group":
        return t.repeat_interleave(spec.group_size, dim=1)[:, :K]
    br, bc = spec.block
    return t.repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)[:rows, :K]


def _absmax_scale(
    amax: torch.Tensor, spec: QuantSpec, global_scale: torch.Tensor | None,
    scale_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    sf = spec.scale
    t = _div(amax, spec.format.max_normal)
    if global_scale is not None:
        t = _div(t, global_scale)
    s = round_to_format(t, sf.format, sf.rounding, saturate=True, noise=scale_noise)
    minimum = getattr(sf.format, "min_subnormal", sf.format.min_normal)
    s = torch.where(s == 0, minimum, s)
    return torch.where(amax == 0, 1.0, s)


def _pow2_scale(amax: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    sf = spec.scale
    a = torch.where(amax == 0, 2.0**-126, amax).double()
    _, exponent = torch.frexp(a)
    e = exponent.to(torch.int64) - 1 - spec.emax_elem
    if sf.method == "pow2_ceil":
        too_small = _ldexp(torch.full_like(a, spec.format.max_normal), e) < a
        e = e + too_small.to(torch.int64)
    overflow = (e > sf.format.emax) | ~torch.isfinite(a)
    e = e.clamp(min=sf.format.emin, max=sf.format.emax)
    s = _ldexp(torch.ones_like(a), e).float()
    return torch.where(overflow, float("nan"), s)


def _signed_zero_extrema(x2d: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Domain min and max with -0 ordered below +0, so a zero bound's sign never
    depends on the reduction order."""
    lo, hi = _domain_extrema(x2d, spec)
    data, valid = _domain_view(x2d, spec)
    zero = (data == 0) & valid
    negative_zero = (zero & torch.signbit(data)).any(dim=1).reshape(lo.shape)
    positive_zero = (zero & ~torch.signbit(data)).any(dim=1).reshape(lo.shape)
    minus, plus = torch.full_like(lo, -0.0), torch.zeros_like(lo)
    lo = torch.where(lo == 0, torch.where(negative_zero, minus, plus), lo)
    hi = torch.where(hi == 0, torch.where(positive_zero, plus, minus), hi)
    return lo, hi


def _zero_point_scale(
    domain: torch.Tensor, spec: QuantSpec, *, extrema: tuple[torch.Tensor, torch.Tensor] | None = None,
    scale_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fmt, sf = spec.format, spec.scale
    lo, hi = (domain.amin(), domain.amax()) if extrema is None else extrema
    if spec.zero_point == "int":
        lo, hi = lo.clamp(max=0), hi.clamp(min=0)
    raw = _div(_sub(hi, lo), fmt.qmax - fmt.qmin)
    s = round_to_format(raw, sf.format, sf.rounding, saturate=True, noise=scale_noise)
    minimum = getattr(sf.format, "min_subnormal", sf.format.min_normal)
    s = torch.where(s == 0, minimum, s)
    s = torch.where(hi == lo, 1.0, s)
    z = lo
    if spec.zero_point == "int":
        z = torch.round(_sub(fmt.qmin, _div(lo, s))).clamp(fmt.qmin, fmt.qmax)
    return s, z


def _global_scale(amax: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Round a division by the exact product M*SF.max once to binary32."""
    value = amax.item()
    if not math.isfinite(value):
        return amax.clone()
    if value == 0:
        return torch.ones_like(amax)
    numerator, denominator = value.as_integer_ratio()
    product = spec.format.max_normal * spec.scale.format.max_normal
    pn, pd = product.as_integer_ratio()
    numerator, denominator = numerator * pd, denominator * pn
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        exponent -= numerator < denominator << exponent
    else:
        exponent -= numerator << -exponent < denominator
    quantum = max(exponent, -126) - 23
    if quantum < 0:
        numerator <<= -quantum
    else:
        denominator <<= quantum
    kept, remainder = divmod(numerator, denominator)
    kept += 2 * remainder > denominator or (2 * remainder == denominator and kept % 2 == 1)
    result = amax.new_tensor(math.ldexp(kept, quantum))
    return torch.where(result == 0, 1.0, result)


def _validate_scale_spec(spec: QuantSpec, scale_noise: torch.Tensor | None) -> None:
    sf = spec.scale
    if sf is None:
        return
    if sf.rounding is Rounding.SR:
        if sf.format.kind == "pow2" or sf.method in ("pow2_floor", "pow2_ceil"):
            raise ValueError("stochastic rounding is not supported for power-of-two scales")
        if scale_noise is None:
            raise ValueError("scale stochastic rounding requires explicit scale_noise")


def _percentile(data: torch.Tensor, valid: torch.Tensor, percentile: float) -> torch.Tensor:
    """Linear quantile without torch.quantile's 2**24-element size restriction."""
    ordered = data.abs().masked_fill(~valid, float("inf")).sort(dim=1).values
    rank = (valid.sum(dim=1) - 1).double() * (percentile / 100)
    below, above = rank.floor().long(), rank.ceil().long()
    lo = ordered.gather(1, below[:, None]).squeeze(1).double()
    hi = ordered.gather(1, above[:, None]).squeeze(1).double()
    value = torch.lerp(lo, hi, rank - below).float()
    return value.masked_fill(torch.isnan(data).any(dim=1), float("nan"))


def compute_scale(
    x2d: torch.Tensor, spec: QuantSpec, amax: torch.Tensor | None = None, *,
    noise: torch.Tensor | None = None, scale_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Choose every domain's scale in parallel; ties keep the first MSE candidate.

    Element ``noise`` is reused for every MSE candidate. ``scale_noise`` is a
    separate uint32 stream broadcastable to the scale layout, never a global RNG.
    """
    x2d = x2d.float()
    if amax is not None:
        if spec.granularity != "tensor" or amax.numel() != 1:
            raise ValueError("amax override requires tensor granularity and a scalar")
        amax = amax.to(device=x2d.device, dtype=torch.float32).reshape(())
    if spec.scale is None:
        return None, None, None
    _validate_scale_spec(spec, scale_noise)
    sf = spec.scale
    layout = _layout(spec, *x2d.shape)
    if scale_noise is not None:
        scale_noise = torch.broadcast_to(scale_noise.to(x2d.device), layout).reshape(-1)
    if spec.zero_point != "none":
        scales, zeros = _zero_point_scale(x2d, spec, extrema=_signed_zero_extrema(x2d, spec),
                                          scale_noise=scale_noise)
        return scales.reshape(layout), zeros.reshape(layout), None
    data, valid = _domain_view(x2d, spec)
    maxima = data.abs().masked_fill(~valid, 0).amax(dim=1) if amax is None else amax.reshape(1)
    global_scale = None
    if sf.two_level:
        global_scale = _global_scale(x2d.abs().amax() if amax is None else amax, spec)
    if not sf.two_level and sf.method in ("pow2_floor", "pow2_ceil"):
        return _pow2_scale(maxima, spec).reshape(layout), None, global_scale
    if amax is None and sf.method == "percentile":
        maxima = _percentile(data, valid, sf.percentile)
    if amax is None and sf.method == "mse":
        if spec.rounding is Rounding.SR and noise is None:
            raise ValueError("MSE with stochastic rounding requires explicit noise")
        domain_noise = None
        if noise is not None:
            domain_noise, _ = _domain_view(torch.broadcast_to(noise.to(x2d.device), x2d.shape), spec)
        ratios = (torch.tensor(sf.search, dtype=torch.float32, device="cpu").tolist() if sf.search else
                  torch.linspace(1.0, 0.5, sf.mse_grid, dtype=torch.float32, device="cpu").tolist())
        best_error = None
        for ratio in ratios:
            # Ratios are rounded to fp32 above without per-candidate host/device copies.
            candidate = _absmax_scale(_mul(maxima, ratio), spec,
                                      global_scale, scale_noise)
            q = quantize_elements(data, spec, candidate[:, None], global_scale=global_scale,
                                  noise=domain_noise)
            restored = _mul(q, candidate[:, None])
            if global_scale is not None:
                restored = _mul(restored, global_scale)
            error = (data.double() - restored.double()).square().masked_fill(~valid, 0).sum(dim=1)
            if best_error is None:
                scales, best_error = candidate, error
            else:
                better = error < best_error
                scales = torch.where(better, candidate, scales)
                best_error = torch.where(better, error, best_error)
    else:
        scales = _absmax_scale(maxima, spec, global_scale, scale_noise)
    return scales.reshape(layout), None, global_scale


def quantize_elements(
    x2d: torch.Tensor,
    spec: QuantSpec,
    scale_pe: torch.Tensor | None,
    zp_pe: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Round elements with prescribed scales (also used by GPTQ)."""
    x = x2d.float()
    if zp_pe is not None and spec.zero_point == "float":
        x = _sub(x, zp_pe)
    if scale_pe is not None:
        effective = scale_pe.float()
        if global_scale is not None:
            effective = _mul(effective, global_scale)
        x = _div(x, effective)
    if zp_pe is not None:
        magnitude = x.double().abs()
        integer = magnitude.floor()
        inc = _round_up(magnitude - integer, integer, x < 0, spec.rounding, noise, spec.sr_bits, None)
        rounded = (integer + inc).copysign(x.double())
        if spec.zero_point == "int":
            rounded = rounded + zp_pe.double()
        return rounded.clamp(spec.format.qmin, spec.format.qmax).float()
    return round_to_format(x, spec.format, spec.rounding, saturate=spec.saturate,
                           noise=noise, sr_bits=spec.sr_bits)


def quantize_reference(
    x: torch.Tensor, spec: QuantSpec, *, amax: torch.Tensor | None = None, noise: torch.Tensor | None = None,
    scale_noise: torch.Tensor | None = None,
) -> QTensor:
    """Quantize a tensor without native low-precision arithmetic."""
    x2d = view_2d(x)
    if noise is not None:
        noise = torch.broadcast_to(noise.to(x.device), x.shape).reshape(x2d.shape)
    scale, zero_point, global_scale = compute_scale(x2d, spec, amax, noise=noise, scale_noise=scale_noise)
    scale_pe = None if scale is None else expand_scale(scale, spec, *x2d.shape)
    zp_pe = None if zero_point is None else expand_scale(zero_point, spec, *x2d.shape)
    values = quantize_elements(x2d, spec, scale_pe, zp_pe, global_scale, noise)
    return QTensor(values.to(container_dtype(spec.format)), scale, zero_point, global_scale,
                   spec, tuple(x.shape))
