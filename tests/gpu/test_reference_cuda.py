"""The reference backend gives the same bits on CUDA tensors as on the CPU.

With ``torch.ldexp`` (``pow(2, e)`` on CUDA is inexact) a few casts differed on an A100 (1 of
~500k bf16 values, up to 7 tf32) and every decode, hence every reference GEMM, raised on CUDA.
"""

from __future__ import annotations

import math

import pytest
import torch

from tricast.mma.api import gemm
from tricast.quant.spec import get_scheme
from tricast.reference.cast import _ldexp, decode, round_to_format
from tricast.reference.quantize import quantize_reference

pytestmark = pytest.mark.gpu


def _same(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Identical bits, -0 != +0; NaN payloads are not compared."""
    actual, expected = actual.cpu().contiguous(), expected.contiguous()
    width = {8: torch.int64, 4: torch.int32, 2: torch.int16}[expected.element_size()]
    same = actual.view(width) == expected.view(width)
    return bool((same | (torch.isnan(actual) & torch.isnan(expected))).all())


def _samples() -> torch.Tensor:
    gen = torch.Generator().manual_seed(42)
    raw = torch.randint(-(2**31), 2**31, (300000,), generator=gen, dtype=torch.int64)
    return torch.cat([raw.to(torch.int32).view(torch.float32),  # all binades, subnormals, specials
                      torch.randn(100000, generator=gen) * 0.02, torch.randn(100000, generator=gen) * 300])


def test_ldexp_on_cuda_is_exact():
    gen = torch.Generator().manual_seed(7)
    m = torch.randint(-(2**53) + 1, 2**53, (20000,), generator=gen, dtype=torch.int64)
    e = torch.randint(-1100, 960, (20000,), generator=gen)
    expected = torch.tensor([math.ldexp(float(a), b) for a, b in zip(m.tolist(), e.tolist(), strict=True)],
                            dtype=torch.float64)
    assert _same(_ldexp(m.double().cuda(), e.cuda()), expected)


@pytest.mark.parametrize("rounding", ["rne", "rtz", "rup"])
@pytest.mark.parametrize("fmt", ["bf16", "fp16", "tf32", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "int8", "e8m0"])
def test_cast_on_cuda_matches_cpu(fmt, rounding):
    x = _samples()
    assert _same(round_to_format(x.cuda(), fmt, rounding), round_to_format(x, fmt, rounding))


@pytest.mark.parametrize("fmt", ["bf16", "fp16", "fp8_e4m3", "fp4_e2m1"])
def test_decode_on_cuda_matches_cpu(fmt):
    grid = round_to_format(_samples(), fmt, "rne")
    grid = grid[torch.isfinite(grid)]
    expected, actual = decode(grid, fmt), decode(grid.cuda(), fmt)
    for got, want in zip(actual[:3], expected[:3], strict=True):
        assert torch.equal(got.cpu(), want)
    assert actual[3] == expected[3]


@pytest.mark.parametrize("scheme", ["nvfp4", "mxfp4", "mxfp8_e4m3", "fp8_row", "fp8_block128",
                                    "int4_g128_zp"])
def test_quantize_on_cuda_matches_cpu(scheme):
    gen = torch.Generator().manual_seed(3)
    w = torch.randn(64, 512, generator=gen) * 0.02
    spec = get_scheme(scheme)
    expected, actual = quantize_reference(w, spec), quantize_reference(w.cuda(), spec)
    assert _same(actual.values, expected.values)
    assert (actual.scale is None) == (expected.scale is None)
    if expected.scale is not None:
        assert _same(actual.scale, expected.scale)


@pytest.mark.parametrize("scheme,preset", [("fp8_row", "nvidia_hopper_fp8"), ("fp8_tensor", "nvidia_ada_fp8"),
                                           ("nvfp4", "nvidia_blackwell_fp4"),
                                           ("mxfp8_e4m3", "nvidia_blackwell_fp8")])
def test_reference_gemm_on_cuda_matches_cpu(scheme, preset):
    gen = torch.Generator().manual_seed(11)
    a, w = torch.randn(16, 256, generator=gen), torch.randn(32, 256, generator=gen) * 0.02
    spec = get_scheme(scheme)
    expected = gemm(quantize_reference(a, spec), quantize_reference(w, spec), preset, backend="reference")
    actual = gemm(quantize_reference(a.cuda(), spec), quantize_reference(w.cuda(), spec), preset,
                  backend="reference")
    assert _same(actual, expected)
