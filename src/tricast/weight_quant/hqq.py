"""HQQ 4 / 3 / 2-bit weights through the official package (dropbox/hqq, Apache-2.0; official_adapter).

``hqq`` builds the official ``HQQLinear`` with ``BaseQuantizeConfig(nbits=nbits, group_size=64)`` — HQQ's
defaults otherwise: axis 1 (groups of 64 along the input dim), ``optimize=True`` (half-quadratic proximal
solver), ``round_zero`` for the initial zero point only at 4 bits (``hqq_base_quant_config``: ``nbits == 4``),
scales and zeros not quantized — and returns its ``dequantize()``: ``(W_q - zero) * scale`` with W_q of
``nbits`` and one BF16 scale and one BF16 zero per group (``compute_dtype=torch.bfloat16``, the Quantizer's
dequant format). Calibration-free. HQQ runs its solver in fp32 on CPU and in fp16 on CUDA
(hqq/core/optimize.py), so the two devices give different weights.
"""

from __future__ import annotations

from importlib.metadata import version

import torch

HQQ_VERSION = "0.2.8.post1"  # the pip release this adapter was checked against; another one is refused
NBITS = 4
GROUP_SIZE = 64
# tricast.recipe.WEIGHT_METHODS kinds of this adapter -> HQQ nbits
KIND_NBITS = {"hqq": 4, "hqq3": 3, "hqq2": 2}


def hqq(x: torch.Tensor, nbits: int = NBITS) -> torch.Tensor:
    """Official HQQ ``nbits`` quantize -> dequantize of a 2-D weight ``x`` on BF16."""
    if nbits not in KIND_NBITS.values():
        raise ValueError(f"nbits: expected one of {sorted(KIND_NBITS.values())}, got {nbits!r}")
    installed = version("hqq")
    if installed != HQQ_VERSION:
        raise RuntimeError(f"hqq {installed} is installed; this adapter is pinned to hqq=={HQQ_VERSION}")
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    config = BaseQuantizeConfig(nbits=nbits, group_size=GROUP_SIZE)
    layer = HQQLinear.from_weights(x.detach().clone(), None, config, compute_dtype=torch.bfloat16,
                                   device=str(x.device))
    return layer.dequantize().clone()


def hqq_int4(x: torch.Tensor) -> torch.Tensor:
    """Official HQQ 4-bit quantize -> dequantize of a 2-D weight ``x`` on BF16."""
    return hqq(x, NBITS)


__all__ = ["GROUP_SIZE", "HQQ_VERSION", "KIND_NBITS", "NBITS", "hqq", "hqq_int4"]
