"""Operand construction and backend dispatch for emulated linear GEMMs."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ..formats import container_dtype, format_of_dtype
from .operand import Operand
from .spec import MMASpec, get_preset

if TYPE_CHECKING:
    from ..quant.qtensor import QTensor


def as_operand(x: QTensor | torch.Tensor | Operand, *, compact: bool = False) -> Operand:
    """View an input as rows by K, preserving its quantization scale domains.

    ``compact`` stores weight grids in the smallest exact bf16/fp16/fp32
    container. The default fp32 representation remains the public API contract;
    packing a compact weight for Triton is transient, not another resident cache.
    """
    if isinstance(x, Operand):
        return x
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            raise ValueError("an MMA input needs at least one dimension")
        fmt = format_of_dtype(x.dtype)
        dtype = container_dtype(fmt) if compact else torch.float32
        values = x.reshape(math.prod(x.shape[:-1]), x.shape[-1]).to(dtype)
        return Operand(values, fmt)

    from ..quant.qtensor import QTensor

    if not isinstance(x, QTensor):
        raise TypeError("an MMA input must be a Tensor, QTensor, or Operand")
    if len(x.shape) == 0:
        raise ValueError("an MMA input needs at least one dimension")
    spec = x.spec
    if spec.mma_input == "dequant":
        dequantized = x.mma_operand()
        values = dequantized.values if isinstance(dequantized, Operand) else dequantized
        dtype = container_dtype(spec.dequant_format) if compact else torch.float32
        values = values.reshape(math.prod(x.shape[:-1]), x.shape[-1]).to(dtype)
        return Operand(values, spec.dequant_format)

    values = x.values.to(container_dtype(spec.format) if compact else torch.float32)
    if x.scale is None:
        return Operand(values, spec.format, alpha=x.global_scale)
    scale = x.scale.to(torch.float32)
    kind, domain = spec.granularity, 0
    if kind == "group":
        kind, domain = "k", spec.group_size
    elif kind == "block":
        kind, domain = "k", spec.block[1]
        scale = scale.repeat_interleave(spec.block[0], dim=0)[: values.shape[0]]
    return Operand(values, spec.format, scale, spec.scale.format, kind, domain, x.global_scale)


def gemm(
    a: QTensor | torch.Tensor | Operand,
    b: QTensor | torch.Tensor | Operand,
    spec: MMASpec | str | dict,
    *,
    bias: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Multiply activations ``[..., K]`` by weights ``[N, K]`` to yield ``[..., N]``."""
    if backend not in ("auto", "reference", "triton"):
        raise ValueError(f"unknown MMA backend {backend!r}")
    if isinstance(spec, dict):
        spec = MMASpec.from_dict(spec)
    elif isinstance(spec, str):
        spec = get_preset(spec)
    if not isinstance(spec, MMASpec):
        raise TypeError("spec must be an MMASpec, preset name, or mapping")

    a_op, b_op = as_operand(a), as_operand(b)
    a_shape = a.values.shape if isinstance(a, Operand) else a.shape
    b_shape = b.values.shape if isinstance(b, Operand) else b.shape
    if len(b_shape) != 2:
        raise ValueError("MMA weights must have shape [N, K]")
    if a_op.K != b_op.K:
        raise ValueError("MMA operands must have the same reduction dimension K")
    if a_op.values.device != b_op.values.device:
        raise ValueError("MMA operands must be on the same device")

    kernel = None
    if backend == "triton" or (backend == "auto" and a_op.values.is_cuda):
        try:
            from ..kernels.mma import gemm_triton
        except ImportError:
            if backend == "triton":
                raise
        else:
            kernel = gemm_triton
            # The kernels still load fp32 K-major inputs. Keep that expansion
            # local to this GEMM so a narrow resident weight never retains both.
            a_op = _kernel_operand(a_op)
            b_op = _kernel_operand(b_op)
    if kernel is None:
        from ..reference.mma import gemm_reference

        kernel = gemm_reference
    out = kernel(a_op, b_op, spec, bias)
    return out.reshape(*a_shape[:-1], b_op.rows)


def _kernel_operand(operand: Operand) -> Operand:
    """Widen only for a kernel call, preserving an already-packed weight layout."""
    if operand.values.dtype == torch.float32:
        return operand
    return replace(operand, values=operand.values.float())
