"""Exact sub-byte and narrow-float bit packing shared by ``tricast.pack`` and ``tricast.onnx_export``.

Every supported low-bit format (1..8-bit signed/unsigned/fixed-point integers, 4..8-bit custom or
standard floats such as FP4/FP6/FP8/UE4M3, 8-bit E8M0 power-of-two scales, NF4, and RaZeR NVFP4)
maps losslessly onto unsigned ``bits``-bit integer codes and packs eight codes into ``bits`` bytes
(LSB-first), reconstructing the exact FP32/BF16 operand TriCast multiplies with (signed zero aside).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .formats import FP4_E2M1, FloatFormat, Format, IntFormat, Pow2Format, get_format


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Unsigned ``bits``-bit codes (any shape, values < 2**bits) as a flat uint8 stream, eight codes in
    ``bits`` bytes, little-endian, the last group padded with zeros."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be 1..8, got {bits}")
    flat = codes.reshape(-1).to(torch.int64)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= 1 << bits):
        raise ValueError(f"codes do not fit in {bits} bits")
    if bits == 8:
        return flat.to(torch.uint8)
    flat = F.pad(flat, (0, -flat.numel() % 8)).reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=flat.device) * bits
    word = (flat << shifts).sum(dim=1)
    out = torch.stack([(word >> (8 * j)) & 0xFF for j in range(bits)], dim=1)
    return out.reshape(-1).to(torch.uint8)


def unpack_bits(data: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """The first ``count`` codes of a ``pack_bits`` stream, as uint8."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be 1..8, got {bits}")
    if bits == 8:
        return data[:count].clone()
    groups = data.to(torch.int64).reshape(-1, bits)
    shifts = torch.arange(bits, dtype=torch.int64, device=data.device) * 8
    word = (groups << shifts).sum(dim=1)
    codes = torch.stack([(word >> (bits * i)) & ((1 << bits) - 1) for i in range(8)], dim=1)
    return codes.reshape(-1)[:count].to(torch.uint8)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Two fp32 tensors with the same bit pattern in every element, the sign of a zero aside: a weight that
    rounds to -0.0 decodes to +0.0 (codes carry no sign of zero), which changes no product sum that starts
    from +0.0 — TriCast's and PyTorch's both do."""
    if a.shape != b.shape or a.dtype != torch.float32 or b.dtype != torch.float32:
        return False
    return torch.equal((a + 0.0).view(torch.int32), (b + 0.0).view(torch.int32))


def int_codebook(fmt: IntFormat) -> torch.Tensor:
    """FP32 grid value for each unsigned code ``0 .. 2**fmt.bits - 1`` of ``fmt``."""
    if not 1 <= fmt.bits <= 8:
        raise ValueError(f"{fmt.name}: bitpack codebook supports 1..8 bits, got {fmt.bits}")
    codes = torch.arange(1 << fmt.bits, dtype=torch.float64) + fmt.qmin
    return (codes * (2.0**-fmt.frac_bits)).to(torch.float32)


def float_codebook(fmt: FloatFormat) -> torch.Tensor:
    """FP32 value for each bit pattern ``0 .. 2**fmt.bits - 1`` of a narrow ``FloatFormat`` (bits <= 8).

    Bit layout: ``[sign | exponent (ebits) | mantissa (mbits)]`` when ``fmt.signed`` else
    ``[exponent (ebits) | mantissa (mbits)]``. Any zero encoding decodes to ``+0.0``.
    """
    if not 1 <= fmt.bits <= 8:
        raise ValueError(f"{fmt.name}: float codebook requires bits in 1..8, got {fmt.bits}")
    ebits, mbits, bias = fmt.ebits, fmt.mbits, fmt.bias
    emax_field = (1 << ebits) - 1
    mmax_field = (1 << mbits) - 1
    values: list[float] = []
    for code in range(1 << fmt.bits):
        sign = ((code >> (ebits + mbits)) & 1) if fmt.signed else 0
        exp_f = (code >> mbits) & emax_field
        man_f = code & mmax_field
        if fmt.special == "ieee" and exp_f == emax_field:
            val = (float("-inf") if sign else float("inf")) if man_f == 0 else float("nan")
        elif fmt.special == "fn" and exp_f == emax_field and man_f == mmax_field:
            val = float("nan")
        elif fmt.special == "fnuz" and sign == 1 and exp_f == 0 and man_f == 0:
            val = float("nan")
        elif exp_f == 0:
            if not fmt.subnormals or man_f == 0:
                val = 0.0
            else:
                mag = math.ldexp(man_f, (1 - bias) - mbits)
                val = -mag if sign else mag
        else:
            mag = math.ldexp((1 << mbits) + man_f, (exp_f - bias) - mbits)
            val = -mag if sign else mag
        values.append(val)
    return torch.tensor(values, dtype=torch.float32)


def pow2_codebook(fmt: Pow2Format) -> torch.Tensor:
    """FP32 value for each exponent code ``0 .. 2**fmt.ebits - 1`` of ``fmt`` (e.g. E8M0)."""
    if not 1 <= fmt.ebits <= 8:
        raise ValueError(f"{fmt.name}: pow2 codebook requires ebits in 1..8, got {fmt.ebits}")
    top = (1 << fmt.ebits) - 1
    values = [
        float("nan") if (fmt.has_nan and code == top) else math.ldexp(1.0, code - fmt.bias)
        for code in range(1 << fmt.ebits)
    ]
    return torch.tensor(values, dtype=torch.float32)


def encode_float_codes(values: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Map grid-aligned FP32 ``values`` onto unsigned codes ``0 .. 2**fmt.bits - 1`` (uint8)."""
    table = float_codebook(fmt)
    # Canonical first-seen code per finite value so +0.0/-0.0 (and flushed subnormals) map to code 0.
    finite_map: dict[float, int] = {}
    nan_code: int | None = None
    for code, item in enumerate(table.tolist()):
        if math.isnan(item):
            if nan_code is None:
                nan_code = code
        else:
            key = 0.0 if item == 0.0 else item
            finite_map.setdefault(key, code)
    pairs = sorted(finite_map.items(), key=lambda kv: kv[0])
    device = values.device
    grid_vals = torch.tensor([kv[0] for kv in pairs], dtype=torch.float32, device=device)
    grid_codes = torch.tensor([kv[1] for kv in pairs], dtype=torch.uint8, device=device)
    clean = values.to(torch.float32) + 0.0
    is_nan = torch.isnan(clean)
    safe = torch.where(is_nan, grid_vals[0], clean)
    idx = torch.searchsorted(grid_vals, safe).clamp(0, grid_vals.numel() - 1)
    matched = (grid_vals[idx] == safe) | is_nan
    if not bool(matched.all()):
        bad = float(clean[~matched][0].item())
        raise ValueError(f"value {bad!r} is not on the {fmt.name} grid")
    out = grid_codes[idx]
    if bool(is_nan.any()):
        if nan_code is None:
            raise ValueError(f"{fmt.name} cannot encode NaN")
        out = torch.where(is_nan, torch.tensor(nan_code, dtype=torch.uint8, device=device), out)
    return out


def encode_int_codes(values: torch.Tensor, fmt: IntFormat) -> torch.Tensor:
    """Map ``IntFormat`` grid values ``k * 2**-frac_bits`` onto unsigned codes ``0 .. 2**fmt.bits - 1``."""
    if not 1 <= fmt.bits <= 8:
        raise ValueError(f"{fmt.name}: bitpack supports 1..8 bits, got {fmt.bits}")
    scaled = values.to(torch.float64) * (2.0**fmt.frac_bits)
    k = torch.round(scaled)
    if not torch.equal(k, scaled):
        raise ValueError(f"values are not on the {fmt.name} grid")
    ki = k.to(torch.int64)
    if ki.numel() and (int(ki.min()) < fmt.qmin or int(ki.max()) > fmt.qmax):
        raise ValueError(f"integer grid values outside [{fmt.qmin}, {fmt.qmax}] for {fmt.name}")
    return (ki - fmt.qmin).to(torch.uint8)


def encode_pow2_codes(values: torch.Tensor, fmt: Pow2Format) -> torch.Tensor:
    """Map ``Pow2Format`` values ``2**e`` onto unsigned codes ``0 .. 2**fmt.ebits - 1`` (uint8)."""
    if not 1 <= fmt.ebits <= 8:
        raise ValueError(f"{fmt.name}: bitpack supports ebits in 1..8, got {fmt.ebits}")
    vd = values.to(torch.float64)
    is_nan = torch.isnan(vd)
    safe = torch.where(is_nan, 1.0, vd)
    if bool((safe <= 0).any() or torch.isinf(safe).any()):
        raise ValueError(f"{fmt.name} scales must be positive powers of two")
    mant, exp = torch.frexp(safe)
    if bool((mant != 0.5).any()):
        raise ValueError(f"values are not powers of two for {fmt.name}")
    e = exp.to(torch.int64) - 1
    if bool((((e < fmt.emin) | (e > fmt.emax)) & ~is_nan).any()):
        raise ValueError(f"exponent outside [{fmt.emin}, {fmt.emax}] for {fmt.name}")
    codes = (e + fmt.bias).clamp(0, (1 << fmt.ebits) - 1).to(torch.uint8)
    if bool(is_nan.any()):
        if not fmt.has_nan:
            raise ValueError(f"{fmt.name} cannot encode NaN")
        top = torch.tensor((1 << fmt.ebits) - 1, dtype=torch.uint8, device=values.device)
        codes = torch.where(is_nan, top, codes)
    return codes


def narrow_float_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Store ``tensor`` in the narrowest torch float dtype (fp16, bf16, or fp32) that is bit-exact."""
    f32 = tensor.to(torch.float32).cpu()
    for dtype, name in ((torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
        narrow = f32.to(dtype)
        if torch.equal(narrow.float(), f32):
            return narrow, name
    return f32, "float32"


def encode_scale_tensor(scale: torch.Tensor, fmt: Format) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode a scale tensor at its true bit-width when ``fmt`` is <=8-bit (E8M0, UE4M3, FP8), else narrow
    float (fp16 / bf16 / fp32)."""
    cpu_scale = scale.detach().cpu()
    if isinstance(fmt, Pow2Format) and 1 <= fmt.ebits <= 8:
        codes = encode_pow2_codes(cpu_scale, fmt)
        return pack_bits(codes, fmt.ebits), {
            "scale_shape": list(cpu_scale.shape),
            "scale_dtype": "packed_pow2",
            "scale_format": fmt.name,
            "scale_bits": fmt.ebits,
            "scale_count": cpu_scale.numel(),
        }
    if isinstance(fmt, FloatFormat) and 1 <= fmt.bits <= 8:
        codes = encode_float_codes(cpu_scale, fmt)
        return pack_bits(codes, fmt.bits), {
            "scale_shape": list(cpu_scale.shape),
            "scale_dtype": "packed_float",
            "scale_format": fmt.name,
            "scale_bits": fmt.bits,
            "scale_count": cpu_scale.numel(),
        }
    narrow, dtype_name = narrow_float_tensor(cpu_scale)
    return narrow, {"scale_shape": list(cpu_scale.shape), "scale_dtype": dtype_name}


def decode_scale_tensor(data: torch.Tensor, meta: Mapping[str, Any]) -> torch.Tensor:
    """Reconstruct the FP32 scale tensor stored by ``encode_scale_tensor``."""
    kind = meta["scale_dtype"]
    shape = tuple(meta["scale_shape"])
    if kind == "packed_pow2":
        fmt = get_format(meta["scale_format"])
        assert isinstance(fmt, Pow2Format)
        codes = unpack_bits(data, int(meta["scale_bits"]), int(meta["scale_count"]))
        lut = pow2_codebook(fmt).to(data.device)
        return lut[codes.long()].reshape(shape)
    if kind == "packed_float":
        fmt = get_format(meta["scale_format"])
        assert isinstance(fmt, FloatFormat)
        codes = unpack_bits(data, int(meta["scale_bits"]), int(meta["scale_count"]))
        lut = float_codebook(fmt).to(data.device)
        return lut[codes.long()].reshape(shape)
    return data.float().reshape(shape)


def encode_razer_weight(
    weight: torch.Tensor, groupsize: int = 16
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Encode ``razer_nvfp4`` into 4-bit codes (code 8 = ±5 special), 8-bit E4M3 block scales (MSB = +5 flag)
    and one FP32 tensor scale."""
    from .weight_quant.razer import E4M3_MAX, FP4_MAN_BITS, FP4_MAX, INLIER

    if weight.numel() % groupsize != 0:
        raise ValueError(f"RaZeR weight numel {weight.numel()} must be divisible by groupsize {groupsize}")
    blocks = weight.detach().reshape(-1, groupsize).to(torch.float32)
    tensor_scale = blocks.abs().amax() / (FP4_MAX * E4M3_MAX)
    if float(tensor_scale.item()) == 0.0:
        codes = torch.zeros(blocks.shape, dtype=torch.uint8)
        scale_u8 = torch.zeros((blocks.shape[0],), dtype=torch.uint8)
        return (
            {
                "codes": pack_bits(codes, 4),
                "scale": scale_u8,
                "global_scale": tensor_scale.cpu().reshape(()),
            },
            {"bits": 4, "count": weight.numel(), "group_size": groupsize},
        )
    scaled = blocks / tensor_scale
    block_max = scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_f8 = (block_max / FP4_MAX).clamp(max=E4M3_MAX, min=2 ** (-9)).to(torch.float8_e4m3fn)
    block_scale = block_scale_f8.to(torch.float32)
    scaled = scaled / block_scale
    exponent = torch.floor(torch.log2(torch.abs(scaled) + (scaled == 0).type(scaled.dtype))).clamp(min=0)
    mantissa = scaled / (2**exponent) * (2**FP4_MAN_BITS)
    mantissa = torch.sign(mantissa) * torch.floor(torch.abs(mantissa) + 0.5)
    fp4 = mantissa * (2**exponent) / (2**FP4_MAN_BITS)

    # Pick -5 vs +5 exactly as razer_nvfp4 does: tie keeps -5 (pos_flag = 0).
    cand_neg = torch.where((scaled - fp4).abs() < (scaled - (-INLIER)).abs(), fp4, -INLIER)
    err_neg = (cand_neg - block_scale).pow(2).mean(-1)
    cand_pos = torch.where((scaled - fp4).abs() < (scaled - INLIER).abs(), fp4, INLIER)
    err_pos = (cand_pos - block_scale).pow(2).mean(-1)
    use_pos = torch.lt(err_pos, err_neg)
    special_val = torch.where(use_pos[:, None], INLIER, -INLIER)
    take_special = (scaled - fp4).abs() >= (scaled - special_val).abs()

    fp4_codes = encode_float_codes(fp4.cpu(), FP4_E2M1)
    # Code 8 is -0.0 in FP4 E2M1 (sign=1, exp=0, man=0), unused by encode_float_codes (+0 -> 0).
    codes = torch.where(take_special.cpu(), torch.tensor(8, dtype=torch.uint8), fp4_codes)
    raw_scale_u8 = block_scale_f8.cpu().reshape(-1).view(torch.uint8)
    sign_flag = torch.where(
        use_pos.cpu(), torch.tensor(0x80, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8)
    )
    scale_u8 = raw_scale_u8 | sign_flag
    tensors = {
        "codes": pack_bits(codes, 4),
        "scale": scale_u8.contiguous(),
        "global_scale": tensor_scale.cpu().to(torch.float32).reshape(()),
    }
    meta = {"bits": 4, "count": weight.numel(), "group_size": groupsize}
    return tensors, meta


def decode_razer_weight(
    tensors: Mapping[str, torch.Tensor], meta: Mapping[str, Any], shape: tuple[int, ...]
) -> torch.Tensor:
    """Decode ``encode_razer_weight`` back to the BF16->FP32 weight ``razer_nvfp4`` produces."""
    from .weight_quant.razer import INLIER

    groupsize = int(meta["group_size"])
    count = int(meta["count"])
    device = tensors["codes"].device
    codes = unpack_bits(tensors["codes"], 4, count).reshape(-1, groupsize)
    lut = float_codebook(FP4_E2M1).to(device)
    elem = lut[codes.long()]
    scale_u8 = tensors["scale"].to(torch.uint8)
    use_pos = (scale_u8 & 0x80) != 0
    block_scale = (scale_u8 & 0x7F).view(torch.float8_e4m3fn).to(torch.float32)[:, None]
    special = torch.where(
        use_pos[:, None], torch.tensor(INLIER, device=device), torch.tensor(-INLIER, device=device)
    )
    chosen = torch.where(codes == 8, special, elem)
    tensor_scale = tensors["global_scale"].to(device=device, dtype=torch.float32)
    return (chosen * block_scale * tensor_scale).reshape(shape).to(torch.bfloat16).float()
