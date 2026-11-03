"""HQQ 4-bit weights through the official package (dropbox/hqq, Apache-2.0; official_adapter).

``hqq_int4`` builds the official ``HQQLinear`` with ``BaseQuantizeConfig(nbits=4, group_size=64)`` — HQQ's
defaults otherwise: axis 1 (groups of 64 along the input dim), ``optimize=True`` (half-quadratic proximal
solver), ``round_zero=True`` for the initial zero point, scales and zeros not quantized — and returns its
``dequantize()``: ``(W_q - zero) * scale`` with W_q uint4 and one BF16 scale and one BF16 zero per group
(``compute_dtype=torch.bfloat16``, the Quantizer's dequant format). Calibration-free. HQQ runs its solver in
fp32 on CPU and in fp16 on CUDA (hqq/core/optimize.py), so the two devices give different weights.
"""

from __future__ import annotations

from importlib.metadata import version

import torch

HQQ_VERSION = "0.2.8.post1"  # the pip release this adapter was checked against; another one is refused
NBITS = 4
GROUP_SIZE = 64


def hqq_int4(x: torch.Tensor) -> torch.Tensor:
    """Official HQQ quantize -> dequantize of a 2-D weight ``x`` on BF16."""
    installed = version("hqq")
    if installed != HQQ_VERSION:
        raise RuntimeError(f"hqq {installed} is installed; this adapter is pinned to hqq=={HQQ_VERSION}")
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    config = BaseQuantizeConfig(nbits=NBITS, group_size=GROUP_SIZE)
    layer = HQQLinear.from_weights(x.detach().clone(), None, config, compute_dtype=torch.bfloat16,
                                   device=str(x.device))
    return layer.dequantize().clone()


__all__ = ["GROUP_SIZE", "HQQ_VERSION", "NBITS", "hqq_int4"]
