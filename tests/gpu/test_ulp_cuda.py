"""ULP statistics of CUDA tensors equal those of the same values on the CPU.

The reference decode behind the distances is exact on the CPU; on CUDA it rejected grid values
of Qwen3-0.6B bf16 outputs, so the analysis moves values to the CPU first.
"""

from __future__ import annotations

import pytest
import torch

from tricast.analysis import ulp_distance, ulp_error

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype,fmt", [(torch.bfloat16, "bf16"), (torch.float16, "fp16"),
                                       (torch.float32, "fp32")])
def test_ulp_of_cuda_tensors_matches_cpu(dtype: torch.dtype, fmt: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    gen = torch.Generator().manual_seed(42)
    bits = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[dtype]
    width = 16 if bits is torch.int16 else 32
    raw = torch.randint(-(2 ** (width - 1)), 2 ** (width - 1), (2, 50000), generator=gen,
                        dtype=torch.int64).to(bits)
    values = raw.view(dtype).float()
    finite = torch.isfinite(values).all(0)
    actual, expected = values[0, finite], values[1, finite]  # subnormals and both signs included
    cpu = ulp_error(actual, expected, fmt=fmt)
    assert ulp_error(actual.cuda(), expected.cuda(), fmt=fmt) == cpu
    assert torch.equal(ulp_distance(actual.cuda(), expected.cuda(), fmt=fmt),
                       ulp_distance(actual, expected, fmt=fmt))
