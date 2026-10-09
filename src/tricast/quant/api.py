"""Quantization dispatch and clipped straight-through fake quantization."""

from __future__ import annotations

from importlib import import_module

import torch

from ..formats import IntFormat
from ..rounding import Rounding
from .qtensor import QTensor
from .spec import QuantSpec, get_scheme


def resolve_backend(backend: str, *tensors: torch.Tensor | None) -> str:
    """Select Triton only when every supplied tensor is on CUDA."""
    if backend not in ("auto", "reference", "triton"):
        raise ValueError("backend must be 'auto', 'reference', or 'triton'")
    present = [t for t in tensors if t is not None]
    cuda = bool(present) and all(t.is_cuda for t in present)
    if backend == "triton" and not cuda:
        raise ValueError("the triton backend requires CUDA tensors")
    if backend != "auto":
        return backend
    if cuda:
        try:
            import_module("tricast.kernels")
        except ImportError:
            return "reference"
        return "triton"
    return "reference"


def quantize(
    x: torch.Tensor, spec: QuantSpec | str, *, backend: str = "auto",
    amax: torch.Tensor | None = None, noise: torch.Tensor | None = None,
    scale_noise: torch.Tensor | None = None,
) -> QTensor:
    """Quantize with a named scheme or an explicit QuantSpec.

    SR requires explicit uint32 element ``noise``; scale SR additionally requires
    an independent ``scale_noise`` tensor broadcastable to the scale layout.
    """
    spec = get_scheme(spec)
    if spec.rounding is Rounding.SR and noise is None:
        raise ValueError("stochastic rounding requires explicit noise")
    backend = resolve_backend(backend, x, amax, noise, scale_noise)
    scale_kwargs = {} if scale_noise is None else {"scale_noise": scale_noise}
    if backend == "triton":
        from ..kernels.quantize import quantize_triton

        return quantize_triton(x, spec, amax=amax, noise=noise, **scale_kwargs)
    from ..reference.quantize import quantize_reference

    return quantize_reference(x, spec, amax=amax, noise=noise, **scale_kwargs)


class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spec, backend, amax, noise, scale_noise):
        from ..reference.quantize import expand_scale

        qt = quantize(x, spec, backend=backend, amax=amax, noise=noise, scale_noise=scale_noise)
        scale = qt.scale_per_element()
        normalized = x.float().reshape(qt.rows, qt.K)
        zero = None
        if qt.zero_point is not None:
            zero = expand_scale(qt.zero_point, spec, qt.rows, qt.K)
            if spec.zero_point == "float":
                normalized = normalized - zero
        if scale is not None:
            if qt.global_scale is not None:
                scale = (scale.double() * qt.global_scale.double()).float()
            normalized = (normalized.double() / scale.double()).float()
        if zero is not None and spec.zero_point == "int":
            normalized = normalized + zero
        fmt = spec.format
        if isinstance(fmt, IntFormat):
            quantum = 2.0**-fmt.frac_bits if zero is None else 1.0
            lo, hi = fmt.qmin * quantum, fmt.qmax * quantum
        else:
            lo, hi = -fmt.max_normal if fmt.signed else 0.0, fmt.max_normal
        ctx.save_for_backward(((normalized >= lo) & (normalized <= hi)).reshape(x.shape))
        out = qt.mma_operand().values.reshape(x.shape) if spec.mma_input == "dequant" else qt.dequantize()
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return torch.where(mask, grad_output, 0), None, None, None, None, None


def fake_quant(
    x: torch.Tensor, spec: QuantSpec | str, *, backend: str = "auto",
    amax: torch.Tensor | None = None, noise: torch.Tensor | None = None,
    scale_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """MMA-equivalent values in the input dtype, with a clipped STE."""
    return _FakeQuant.apply(x, get_scheme(spec), backend, amax, noise, scale_noise)
