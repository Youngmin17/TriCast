"""NormalFloat-4 (QLoRA, bitsandbytes ``quant_type="nf4"``) in torch on CPU and GPU (tricast_native).

The definition followed is bitsandbytes 0.50.2's CUDA path (csrc/kernels.cu ``kQuantizeBlockwise<NF4>`` /
``kDequantizeBlockwise<NF4>``, called by ``bitsandbytes.functional.quantize_4bit`` / ``dequantize_4bit``):

- blocks of ``blocksize`` consecutive elements of the flattened tensor (a ``[out, in]`` weight with
  ``in % blocksize == 0``: blocks along the input dim of each row); a last partial block has its own absmax
- ``absmax`` = max |x| of the block in fp32, stored as is (no double quantization)
- ``scaled = x * (1.0f / absmax)`` in fp32
- code index = ``dQuantizeNF4(scaled)``: a tree of strict ``>`` tests on NF4_THRESHOLDS, i.e. the number of
  thresholds strictly below ``scaled``; a value equal to a threshold takes the lower code, and so does NaN
  (an all-zero block: 0 * inf), which dequantizes to -0.0
- dequant = ``NF4_CODE[index] * absmax`` in fp32, then BF16 (round to nearest even), the Quantizer's dequant
  format (bitsandbytes returns the input dtype: the same for a BF16 weight)

Not modelled: bitsandbytes builds its CUDA kernels with ``--use_fast_math`` (CMakeLists.txt), so its
``1.0f / absmax`` is an approximate reciprocal; here it is the IEEE one.
"""

from __future__ import annotations

import torch

BLOCKSIZE = 64
# bitsandbytes nf4_dequantization_lut (kernels.cu) = functional.get_4bit_type("nf4"); every value is an fp32.
NF4_CODE = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
            -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
            0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
            0.7229568362236023, 1.0)
# dQuantizeNF4's float literals (kernels.cu) as fp32, the value nearest each decimal: threshold i separates
# codes i and i + 1. Six are the exact midpoint of their codes (a value equal to it is a tie: the lower code);
# five are the fp32 just below a midpoint that is not an fp32, so they act as the exact midpoint; four (i = 1,
# 3, 4, 8) are the fp32 just above it, so at exactly that value bitsandbytes takes the lower code though the
# upper one is nearer.
NF4_THRESHOLDS = (-0.8480964303016663, -0.6106328964233398, -0.4599952697753906, -0.33967941999435425,
                  -0.23460739850997925, -0.13791173696517944, -0.045525018125772476, 0.03979014977812767,
                  0.1202552542090416, 0.2035212516784668, 0.2920137643814087, 0.3893125355243683,
                  0.5016633868217468, 0.6427869200706482, 0.8614783883094788)


def _blocks(x: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, int]:
    """The flattened fp32 tensor as rows of ``blocksize`` (a last partial block padded with zeros, which do
    not change its absmax) and the number of real elements."""
    if blocksize < 1:
        raise ValueError(f"blocksize must be positive, got {blocksize}")
    flat = x.reshape(-1).float()
    pad = -flat.numel() % blocksize
    return torch.nn.functional.pad(flat, (0, pad)).reshape(-1, blocksize), flat.numel()


def nf4_quantize(x: torch.Tensor, blocksize: int = BLOCKSIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """(code indices as uint8 in ``x``'s shape, fp32 absmax per block)."""
    blocks, n = _blocks(x, blocksize)
    absmax = blocks.abs().amax(dim=1)
    scaled = blocks * (1.0 / absmax)[:, None]
    thresholds = torch.tensor(NF4_THRESHOLDS, dtype=torch.float32, device=x.device)
    index = (scaled.unsqueeze(-1) > thresholds).sum(dim=-1).to(torch.uint8)
    return index.reshape(-1)[:n].reshape(x.shape), absmax


def nf4_dequantize(index: torch.Tensor, absmax: torch.Tensor, blocksize: int = BLOCKSIZE) -> torch.Tensor:
    """``NF4_CODE[index] * absmax`` in fp32, in ``index``'s shape."""
    code = torch.tensor(NF4_CODE, dtype=torch.float32, device=index.device)
    values, n = _blocks(code[index.long()], blocksize)
    return (values * absmax[:, None]).reshape(-1)[:n].reshape(index.shape)


def nf4(x: torch.Tensor, blocksize: int = BLOCKSIZE) -> torch.Tensor:
    """NF4 quantize -> dequantize of ``x`` on BF16."""
    return nf4_dequantize(*nf4_quantize(x, blocksize), blocksize).to(torch.bfloat16)


__all__ = ["BLOCKSIZE", "NF4_CODE", "NF4_THRESHOLDS", "nf4", "nf4_dequantize", "nf4_quantize"]
