"""Weight rounding with optional Hessian-guided error compensation, and whole-weight replacement methods."""

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


def replace_weight(W: torch.Tensor, algo: WeightAlgoSpec) -> torch.Tensor:
    """The BF16 weight that a ``tricast.recipe.WEIGHT_METHODS`` algorithm puts in place of ``W``: NF4
    (``nf4``, tricast_native), official HQQ 4-bit (``hqq``, official_adapter) or RaZeR NVFP4-E4M3 (``razer``,
    tricast_native)."""
    if algo.kind == "nf4":
        from .nf4 import nf4

        return nf4(W)
    if algo.kind == "hqq":
        from .hqq import hqq_int4

        return hqq_int4(W)
    if algo.kind == "razer":
        from .razer import razer_nvfp4

        return razer_nvfp4(W)
    raise ValueError(f"weight algorithm {algo.kind!r} does not replace the weight")


__all__ = ["quantize_weight", "replace_weight"]
