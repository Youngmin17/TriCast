"""RaZeR NVFP4 with an E4M3 block scale (redundant-zero remapping), re-implemented in torch (tricast_native).

Source: ``quant_nvfp4_razer_e4m3`` in quantize/quantizer.py of the pinned official clone
(``tricast.official.repos.REPOS["razer"]``); this module repeats its fp32 operations in the same order, so the
result is bit-identical on the same device (tests/test_weight_methods.py checks it against the clone).

- blocks of ``GROUP_SIZE`` = 16 consecutive elements of the flattened tensor (the official scripts'
  ``w_groupsize`` / ``a_groupsize``; along the input dim of a ``[out, in]`` weight with ``in % 16 == 0``)
- tensor scale = amax / (6 * 448) in fp32; block scale = (block amax / 6) clamped to [2^-9, 448] and cast to
  float8_e4m3fn (round to nearest even), as NVFP4
- elements: E2M1 with the mantissa rounded half away from zero (``floor(|m| + 0.5)``)
- remap: the FP4 code of -0 (redundant with +0) stands for one special value per block, -5 or +5 (the E4M3
  scale's unused sign bit says which); an element takes it where ``|x - s| <= |x - q_fp4|``. The block
  keeps the candidate with the smaller ``mean((q - block_scale)^2)`` — the official criterion as written (it
  compares the quantized values with the block scale, not with the input); a tie keeps -5
- dequant = ``q * block_scale * tensor_scale`` in fp32, then BF16 (as the official function returns)

In the official W4A4 runs (scripts/*.sh) this E4M3 variant quantizes activations; weights use
``quant_nvfp4_razer_e3m3`` (E3M3 scale, special values ±5, ±8).
"""

from __future__ import annotations

import torch

GROUP_SIZE = 16
FP4_MAX = 6.0
FP4_MAN_BITS = 1
E4M3_MAX = 448
INLIER = 5.0  # the special values are -INLIER and +INLIER


@torch.no_grad()
def razer_nvfp4(x: torch.Tensor, groupsize: int = GROUP_SIZE) -> torch.Tensor:
    """RaZeR-E4M3 quantize -> dequantize of ``x`` on BF16, in ``x``'s shape."""
    blocks = x.reshape(-1, groupsize).to(torch.float32)
    tensor_scale = blocks.abs().amax() / (FP4_MAX * E4M3_MAX)
    scaled = blocks / tensor_scale
    block_max = scaled.abs().amax(dim=-1, keepdim=True)
    block_scale = (block_max / FP4_MAX).clamp(max=E4M3_MAX, min=2 ** (-9)).to(torch.float8_e4m3fn)
    block_scale = block_scale.to(torch.float32)
    scaled = scaled / block_scale
    exponent = torch.floor(torch.log2(torch.abs(scaled) + (scaled == 0).type(scaled.dtype))).clamp(min=0)
    mantissa = scaled / (2 ** exponent) * (2 ** FP4_MAN_BITS)
    mantissa = torch.sign(mantissa) * torch.floor(torch.abs(mantissa) + 0.5)
    fp4 = mantissa * (2 ** exponent) / (2 ** FP4_MAN_BITS)
    error = torch.full([blocks.shape[0]], float("inf"), dtype=blocks.dtype, device=blocks.device)
    chosen = torch.zeros_like(blocks)
    for special in (-INLIER, INLIER):
        candidate = torch.where((scaled - fp4).abs() < (scaled - special).abs(), fp4, special)
        candidate_error = (candidate - block_scale).pow(2).mean(-1)
        better = torch.lt(candidate_error, error)
        error[better] = candidate_error[better]
        chosen[better] = candidate[better]
    return (chosen * block_scale * tensor_scale).view(x.shape).to(torch.bfloat16)


__all__ = ["GROUP_SIZE", "INLIER", "razer_nvfp4"]
