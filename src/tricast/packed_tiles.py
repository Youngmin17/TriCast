"""Inference-only row tiles for the original TriCast affine, NF4 and declared dense tensors.

Rows, not K, are partitioned: every dot product still sees its full K dimension.
No quantized projection is ever expanded in its entirety, even if tile_rows >= N.
A GEMM on a narrower output shape need not accumulate in exactly the same order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .bitpack import unpack_bits
from .weight_quant.nf4 import NF4_CODE


def _stream(t: torch.Tensor, bits: int, count: int) -> bool:
    return t.dtype == torch.uint8 and t.shape == ((count + 7) // 8 * bits,)


def validate(entry: dict[str, Any], tensors: Mapping[str, torch.Tensor], spec: Any) -> None:
    n, k = entry["shape"]
    kind, bits = entry["kind"], entry["bits"]
    if (
        any(type(d) is not int or d <= 0 for d in (n, k, bits))
        or spec.activation is not None
        or spec.sparsity.kind != "none"
        or spec.outliers is not None
        or getattr(getattr(spec.mma, "out_format", None), "name", None) not in (None, "bf16", "fp16", "fp32")
    ):
        raise ValueError("tiled mode accepts only weight-only inference with a native output dtype")
    trans = entry.get("transform") or {"kind": "none"}
    if trans["kind"] != spec.transform.kind:
        raise ValueError("stored input transform disagrees with the recipe")
    has_diag = trans["kind"] in ("random_hadamard", "awq", "smoothquant")
    extra = {"diag"} if has_diag else set()
    if has_diag and ("diag" not in tensors or tuple(tensors["diag"].shape) != (k,)):
        raise ValueError("invalid input transform diagonal")
    if has_diag:
        diag = tensors["diag"]
        if trans["kind"] == "random_hadamard":
            if diag.dtype != torch.int8 or not bool(((diag == 1) | (diag == -1)).all()):
                raise ValueError("random Hadamard requires a stored +/-1 int8 sign diagonal")
        elif diag.dtype != torch.float32 or not bool((torch.isfinite(diag) & (diag > 0)).all()):
            raise ValueError("input scaling requires a finite positive FP32 diagonal")
    if trans["kind"] in ("random_hadamard", "hadamard"):
        b = trans.get("block", 0)
        # fit_transform resolves a zero block to the largest power of two <= 128 dividing K.
        expected = spec.transform.block or min(k & -k, 128)
        if type(b) is not int or b != expected or b < 1 or b & (b - 1) or k % b:
            raise ValueError("invalid Hadamard block")
    if kind == "dense":
        if (
            tensors.keys() != extra | {"weight"}
            or tuple(tensors["weight"].shape) != (n, k)
            or (bits, tensors["weight"].dtype) not in ((16, torch.bfloat16), (32, torch.float32))
        ):
            raise ValueError("invalid declared dense weight")
        return
    if n <= 1:
        raise ValueError("single-row compressed projection cannot be partitioned along rows")
    if kind not in ("nf4", "affine"):
        raise ValueError(f"no bounded tile decoder for {kind!r}")
    if entry["count"] != n * k or bits not in (2, 3, 4) or not _stream(tensors["codes"], bits, n * k):
        raise ValueError("invalid packed code stream for a tiled projection")
    if kind == "nf4":
        b = entry["blocksize"]
        if (
            bits != 4
            or b != 64
            or k % b
            or spec.weight_algo.kind != "nf4"
            or tensors.keys() != extra | {"codes", "absmax"}
            or tensors["absmax"].dtype != torch.float32
            or tensors["absmax"].shape != (n * k // b,)
        ):
            raise ValueError("unsupported NF4 tile layout")
        return
    group, q = entry["group_size"], spec.weight
    if (
        q is None
        or q.granularity != "group"
        or entry["granularity"] != "group"
        or group != q.group_size
        or group <= 0
        or k % group
        or q.zero_point != "int"
        or q.mma_input != "dequant"
        or q.dequant_format.name != "bf16"
        or q.format.name != f"uint{bits}"
        or entry["qmin"] != 0
        or entry.get("frac_bits", 0)
        or entry.get("zero_kind", "int") != "int"
        or entry.get("zero_bits") != bits
        or entry.get("scale_dtype") != "float16"
        or tensors.keys() != extra | {"codes", "zero", "scale"}
    ):
        raise ValueError(
            "unsupported affine layout (expected grouped UINT2/3/4, integer ZP, FP16 scale, BF16 dequant)"
        )
    shape = (n, k // group)
    if (
        tuple(tensors["scale"].shape) != shape
        or tensors["scale"].dtype != torch.float16
        or any(tuple(entry[key]) != shape for key in ("zero_shape", "scale_shape"))
        or entry["zero_count"] != n * k // group
        or not _stream(tensors["zero"], bits, n * k // group)
    ):
        raise ValueError("invalid tiled group scales/zero-points")


def _unpack(data: torch.Tensor, bits: int, start: int, stop: int) -> torch.Tensor:
    """Unpack just enclosing 8-code groups; first or last group can straddle a tile boundary."""
    first, last = start // 8, (stop + 7) // 8
    block = data[first * bits : last * bits]
    # The common aligned 4-bit and 2-bit paths need no wide reductions.
    if bits in (2, 4):
        mask = (1 << bits) - 1
        values = torch.stack([(block >> s) & mask for s in range(0, 8, bits)], dim=-1).reshape(-1)
    else:
        values = unpack_bits(block, bits, (last - first) * 8)
    offset = start - first * 8
    return values[offset : offset + stop - start]


def tile(
    entry: dict[str, Any],
    tensors: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
    nf4_lut: torch.Tensor | None,
) -> torch.Tensor:
    """FP32 operand values, rounded as in pack.decode; a typed dense source can be a plain view."""
    n, k = entry["shape"]
    if not 0 <= start < stop <= n or (stop - start == n and entry["kind"] != "dense"):
        raise ValueError("requested rows exceed the permitted decoded projection tile")
    if entry["kind"] == "dense":
        return tensors["weight"][start:stop].float()
    bits, rows = entry["bits"], stop - start
    codes = _unpack(tensors["codes"], bits, start * k, stop * k)
    if entry["kind"] == "nf4":
        if nf4_lut is None:
            raise ValueError("NF4 codebook missing")
        b = entry["blocksize"]
        absmax = tensors["absmax"][start * (k // b) : stop * (k // b)]
        values = nf4_lut[codes.long()].reshape(-1, b) * absmax[:, None]
    else:
        g = entry["group_size"]
        zeros = _unpack(tensors["zero"], bits, start * (k // g), stop * (k // g))
        # The product of a UINT4 difference and an FP16 scale is exact in FP32.
        values = (codes.reshape(rows, k // g, g).float() - zeros.reshape(rows, k // g, 1).float()) * tensors[
            "scale"
        ][start:stop, :, None].float()
    return values.reshape(rows, k).to(torch.bfloat16).float()


def codebook(device: torch.device) -> torch.Tensor:
    return torch.tensor(NF4_CODE, device=device, dtype=torch.float32)
