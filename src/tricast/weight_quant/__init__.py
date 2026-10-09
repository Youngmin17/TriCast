"""Weight rounding with optional Hessian-guided error compensation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..quant.spec import QuantSpec, WeightAlgoSpec

if TYPE_CHECKING:
    from ..quant.qtensor import QTensor


def quantize_weight(
    W: torch.Tensor,
    spec: QuantSpec,
    algo: WeightAlgoSpec,
    *,
    hessian: torch.Tensor | None = None,
    backend: str = "reference",
) -> QTensor:
    """Dispatch RTN to the quantizer or GPTQ to the reference calibration algorithm."""
    if backend not in ("auto", "reference", "triton"):
        raise ValueError(f"unknown backend {backend!r}")
    if algo.kind == "rtn":
        from ..quant.api import quantize

        return quantize(W, spec, backend=backend)
    if hessian is None:
        raise ValueError("GPTQ requires a calibration Hessian")
    from .gptq import gptq

    return gptq(W, hessian, spec, block_size=algo.block_size, damp=algo.damp, act_order=algo.act_order)


__all__ = ["quantize_weight"]
