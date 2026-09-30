"""The small-M MMA kernel (`_gemv`, M <= 16) against the exact reference, Inf/NaN included.

The special-value cases in test_triton_mma.py use FP32 operands, which never take the integer
fast path and so never reach `_gemv`. These cases use BF16/FP16 operands (fast path) with
Inf/NaN/zero values, Inf/NaN/zero scales, E8M0 field 0 and an fp32 overflow of the running sum,
and check that `_gemv` is the kernel that ran.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.conftest import bit_equal
from tricast.formats import BF16, E8M0, FP16, FP32, UE4M3
from tricast.mma.operand import Operand
from tricast.mma.spec import MMASpec
from tricast.reference.cast import round_to_format

pytestmark = pytest.mark.gpu

K = 45  # a partial last chunk, group and tile
CASES = {
    # name: (spec, value format, scale format or None, scale domain)
    "cofda-fused": (MMASpec("cofda", f_bits=13, chunk_size=8, out_format=FP32), BF16, None, 0),
    "cofda-decoupled": (MMASpec("cofda", f_bits=7, chunk_size=16, c_mode="decoupled", f2_bits=23,
                                out_format=FP32), FP16, None, 0),
    "cofda-promote": (MMASpec("cofda", f_bits=13, chunk_size=8, promote_interval=32, out_format=FP32),
                      BF16, FP32, 32),
    "cofda-e8m0-product": (MMASpec("cofda", f_bits=25, chunk_size=16, out_format=FP32), BF16, E8M0, 16),
    "gdfs": (MMASpec("gdfs", f_bits=25, g_bits=16, group_size=8, k_tile=32, out_format=FP32),
             BF16, None, 0),
    "gdfs-ue4m3-group": (MMASpec("gdfs", f_bits=25, g_bits=16, group_size=8, k_tile=32,
                                 out_format=FP32), FP16, UE4M3, 16),
}


@pytest.fixture
def backends():
    reference = pytest.importorskip("tricast.reference.mma")
    kernel = pytest.importorskip("tricast.kernels.mma")
    return reference, kernel


def _operand(rows: int, fmt, scale_fmt, domain: int, gen: torch.Generator) -> Operand:
    values = round_to_format(torch.randn(rows, K, generator=gen) * 2, fmt)
    flat = values.view(-1)
    for special in (float("inf"), -float("inf"), float("nan"), 0.0, -0.0):
        flat[torch.randint(0, flat.numel(), (2,), generator=gen)] = special
    if fmt is BF16:
        values[0, :8] = 2.0**100  # row 0 x row 0 overflows fp32 in the first chunk
    if scale_fmt is None:
        return Operand(values, fmt)
    scales = round_to_format(0.25 + torch.rand(rows, (K + domain - 1) // domain, generator=gen) * 2,
                             scale_fmt)
    flat = scales.view(-1)
    # E8M0 has no zero: its field 0 (2^-127) is the zero-contribution code.
    for special in (float("nan"), 2.0**-127 if scale_fmt is E8M0 else 0.0):
        flat[torch.randint(0, flat.numel(), (1,), generator=gen)] = special
    return Operand(values, fmt, scales, scale_fmt, "k", domain)


def _cuda(op: Operand) -> Operand:
    return replace(op, values=op.values.cuda(), scale=None if op.scale is None else op.scale.cuda())


@pytest.mark.parametrize("m", [1, 5, 16])
@pytest.mark.parametrize("case", sorted(CASES))
def test_small_m_kernel_matches_reference_with_specials(backends, monkeypatch, case, m):
    reference, kernel = backends
    spec, fmt, scale_fmt, domain = CASES[case]
    gen = torch.Generator().manual_seed(42 + m)
    a, b = _operand(m, fmt, scale_fmt, domain, gen), _operand(9, fmt, scale_fmt, domain, gen)
    expected = reference.gemm_reference(a, b, spec)
    launched = []
    original = kernel._gemv

    class Recorder:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                compiled = original[grid](*args, **kwargs)
                launched.append(compiled)
                return compiled
            return run

    monkeypatch.setattr(kernel, "_gemv", Recorder())
    actual = kernel.gemm_triton(_cuda(a), _cuda(b), spec).cpu()
    assert [compiled.name for compiled in launched] == ["_gemv"]
    assert not torch.isfinite(expected).all()  # the specials reached the output
    assert bit_equal(actual, expected), f"{case} m={m}:\nactual={actual}\nexpected={expected}"
