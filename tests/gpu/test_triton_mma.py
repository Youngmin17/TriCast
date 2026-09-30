"""Bit-exact GPU gates for every MMA mode; no tolerance and no model downloads.

Set TRICAST_MMA_PERF=1 for the opt-in 1024^3 smoke benchmark. It reports emulated
MACs, not native Tensor Core throughput. Compilation/autotuning precede timing.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests.conftest import bit_equal
from tricast.formats import (
    BF16,
    E8M0,
    FP4_E2M1,
    FP6_E2M3,
    FP6_E3M2,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    FP32,
    INT8,
    MXINT8,
    UE4M3,
    IntFormat,
)
from tricast.mma.operand import Operand
from tricast.mma.spec import MMASpec, get_preset
from tricast.reference.cast import decode, round_to_format

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

pytestmark = pytest.mark.gpu
FORMATS = (FP8_E4M3, FP8_E5M2, FP6_E3M2, FP6_E2M3, FP4_E2M1, BF16, FP16, INT8, MXINT8)
LAYOUTS = ("none", "tensor", "row", "mx", "nv", "block", "mixed")


@pytest.fixture
def backends():
    reference = pytest.importorskip("tricast.reference.mma")
    kernel = pytest.importorskip("tricast.kernels.mma")
    return reference, kernel


@pytest.fixture(autouse=True)
def deterministic(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _cuda(op: Operand) -> Operand:
    return replace(op, values=op.values.cuda(), scale=None if op.scale is None else op.scale.cuda(),
                   alpha=None if op.alpha is None else op.alpha.cuda())


def _operand(rows, k, fmt, layout, generator):
    values = round_to_format(torch.randn(rows, k, generator=generator) * 2, fmt)
    if layout == "none":
        return Operand(values, fmt)
    if layout in ("tensor", "row"):
        shape = () if layout == "tensor" else (rows, 1)
        return Operand(values, fmt, 0.5 + torch.rand(shape, generator=generator), FP32, layout)
    domain = 64 if layout == "block" else 32
    domains = (k + domain - 1) // domain
    # Block scales are expanded along rows, as required by Operand's public layout.
    scale_rows = (rows + 3) // 4 if layout == "block" else rows
    raw = 0.25 + torch.rand((scale_rows, domains), generator=generator) * 2
    scale_fmt = E8M0 if layout == "mx" else UE4M3 if layout == "nv" else FP32
    scales = round_to_format(raw, scale_fmt)
    if layout == "block":
        scales = scales.repeat_interleave(4, dim=0)[:rows]
    alpha = torch.tensor(1.001234) if layout == "nv" else None
    return Operand(values, fmt, scales, scale_fmt, "k", domain, alpha)


def _compare(backends, a, b, spec, bias=None):
    reference, kernel = backends
    expected = reference.gemm_reference(a, b, spec, bias)
    actual = kernel.gemm_triton(_cuda(a), _cuda(b), spec, None if bias is None else bias.cuda()).cpu()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert bit_equal(actual, expected), f"{spec}:\nactual={actual}\nexpected={expected}"
    return actual


def _cases():
    cases = []
    for ai, algorithm in enumerate(("cofda", "gdfs", "fp32_fma", "fp64")):
        for fi, fmt in enumerate(FORMATS):
            for variation in range(2):
                i = fi * 2 + variation
                layout = LAYOUTS[(i + ai) % len(LAYOUTS)]
                cs = (8, 16, 32)[i % 3]
                pi = 32 if algorithm == "cofda" and i % 4 == 0 else 0
                spec = MMASpec(
                    algorithm, f_bits=(3, 7, 13, 23, 25, 35)[i % 6], chunk_size=cs,
                    c_mode="decoupled" if variation else "fused", f2_bits=(7, 23, 35)[fi % 3],
                    g_bits=16, group_size=8 if variation else 16, k_tile=64,
                    norm_rounding="rne" if variation else "rtz", promote_interval=pi,
                    out_format=(FP32, BF16, FP16, FP8_E4M3, FP6_E3M2, FP4_E2M1)[(i + ai) % 6],
                )
                cases.append(pytest.param(fmt, layout, spec, i + ai * 18,
                                          id=f"{algorithm}-{fmt.name}-{layout}-{i}"))
    for fmt in (INT8, MXINT8):
        for layout in ("none", "tensor", "row"):
            cases.append(pytest.param(fmt, layout, MMASpec("int_exact", out_format=FP32), 42,
                                      id=f"int_exact-{fmt.name}-{layout}"))
    return cases


@pytest.mark.parametrize("fmt,layout,spec,index", _cases())
def test_representative_modes(backends, fmt, layout, spec, index):
    gen = torch.Generator().manual_seed(42)
    shape_gen = torch.Generator().manual_seed(42 + index)
    m, n = torch.randint(1, 20, (2,), generator=shape_gen).tolist()
    k = int(torch.randint(1, 100, (), generator=shape_gen))
    la, lb = ("row", "mx") if layout == "mixed" else (layout, layout)
    a, b = _operand(m, k, fmt, la, gen), _operand(n, k, fmt, lb, gen)
    bias = torch.randn(n, generator=gen) / 7
    _compare(backends, a, b, spec, bias)


@pytest.mark.parametrize("algorithm", ["cofda", "gdfs", "fp32_fma", "fp64"])
@pytest.mark.parametrize("case", ["subnormal", "special", "cancel", "wide_alignment"])
def test_adversarial(backends, algorithm, case):
    if case == "subnormal":
        av = [2.0**-149, -2.0**-148, 2.0**-126, 2.0**-130, 0.0]
        bv = [1.0, 1.5, 0.125, 0.375, -0.0]
    elif case == "special":
        av = [float("inf"), -float("inf"), float("nan"), 0.0, 1.0]
        bv = [0.0, 1.0, -1.0, float("inf"), 1.0]
    elif case == "cancel":
        av, bv = [1.125, -1.125, 0.0, -0.0, 1.0], [1.0, 1.0, -0.0, 0.0, 0.0]
    else:
        av, bv = [2.0**127, 2.0**64, 2.0**63, 2.0**62, 2.0**-149], [1.0] * 5
    a = Operand(torch.tensor([av]), FP32)
    # The first row exercises aggregation, subsequent rows isolate each special.
    b = Operand(torch.cat((torch.tensor([bv]), torch.diag(torch.tensor(bv)))), FP32)
    spec = MMASpec(algorithm, f_bits=25, chunk_size=8, g_bits=25, group_size=8, k_tile=32,
                   out_format=FP32)
    _compare(backends, a, b, spec)


@pytest.mark.parametrize("algorithm", ["cofda", "gdfs", "fp32_fma", "fp64"])
def test_special_singletons(backends, algorithm):
    a = Operand(torch.tensor([[-float("inf")], [float("inf")], [float("nan")], [0.0], [-0.0], [1.0]]), FP32)
    b = Operand(torch.tensor([[-float("inf")], [0.0], [float("inf")], [1.0], [-1.0]]), FP32)
    spec = MMASpec(algorithm, f_bits=25, chunk_size=8, g_bits=25, group_size=8, k_tile=32,
                   out_format=FP32)
    _compare(backends, a, b, spec)


@pytest.mark.parametrize("f_bits", [3, 7, 13, 23, 25, 35])
def test_short_promote_subnormal(backends, f_bits):
    a = Operand(torch.tensor([[1.5 * 2.0**-65]]), BF16)
    b = Operand(torch.tensor([[1.421875 * 2.0**-65]]), BF16)
    spec = MMASpec(f_bits=f_bits, chunk_size=8, promote_interval=32, out_format=FP32)
    _compare(backends, a, b, spec)


@pytest.mark.parametrize("algorithm", ["cofda", "gdfs"])
@pytest.mark.parametrize("case", ["wide_product", "scale_subnormal", "zero_nan", "zero_inf",
                                  "inf_zero", "e8m0_field0"])
def test_scale_edges(backends, algorithm, case):
    spec = MMASpec(algorithm, f_bits=35, chunk_size=8, g_bits=35, group_size=8, k_tile=32,
                   out_format=FP32)
    if case == "wide_product":
        values = torch.tensor([[1.9999998807907104, -1.5, 0.25, 1.125, -0.5]])
        scale, fmt = torch.tensor([[1.9999998807907104]]), FP32
    elif case == "scale_subnormal":
        values = torch.tensor([[2.0**60, 2.0**59, -2.0**60, 2.0**57, 0.0]])
        scale, fmt = torch.tensor([[2.0**-127]]), BF16
    elif case in ("zero_nan", "zero_inf"):
        values = torch.zeros(1, 5)
        scale, fmt = torch.tensor([[float("nan" if case == "zero_nan" else "inf")]]), FP32
    elif case == "inf_zero":
        values = torch.full((1, 5), float("inf"))
        scale, fmt = torch.zeros(1, 1), FP32
    else:
        values = torch.full((1, 5), 2.0**64)
        scale, fmt = torch.tensor([[2.0**-127]]), E8M0
    a = Operand(values, FP32, scale, fmt, "k", 8)
    b = Operand(values, FP32, torch.ones_like(scale), fmt, "k", 8)
    _compare(backends, a, b, spec)


@pytest.mark.parametrize("shape", [(0, 7, 3), (3, 0, 7), (3, 5, 0), (67, 81, 511)])
def test_shapes(backends, shape):
    gen = torch.Generator().manual_seed(42)
    m, n, k = shape
    a, b = _operand(m, k, FP8_E4M3, "none", gen), _operand(n, k, FP8_E4M3, "none", gen)
    _compare(backends, a, b, MMASpec(f_bits=13, chunk_size=32, out_format=FP32))


@pytest.mark.parametrize("algorithm", ["cofda", "gdfs", "fp32_fma", "fp64", "int_exact"])
def test_tile_invariance_and_no_reference_fallback(backends, monkeypatch, algorithm):
    reference, kernel = backends
    gen = torch.Generator().manual_seed(42)
    fmt = INT8 if algorithm == "int_exact" else BF16
    a, b = _operand(33, 37, fmt, "none", gen), _operand(35, 37, fmt, "none", gen)
    spec = MMASpec(algorithm, f_bits=13, chunk_size=8, g_bits=13, group_size=8, k_tile=32,
                   out_format=FP32)
    expected = reference.gemm_reference(a, b, spec)
    launch = kernel._gemm.fn

    def forbidden(*args, **kwargs):
        raise AssertionError("reference fallback was invoked")

    monkeypatch.setattr(reference, "gemm_reference", forbidden)
    launches = []

    class FixedTile:
        def __init__(self, bm, bn, warps):
            self.bm, self.bn, self.warps = bm, bn, warps

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                compiled = launch[grid](*args, **kwargs, BM=self.bm, BN=self.bn, num_warps=self.warps)
                launches.append(compiled)
                return compiled
            return run

    for bm, bn, warps in ((16, 16, 2), (32, 32, 4), (64, 16, 8)):
        monkeypatch.setattr(kernel, "_gemm", FixedTile(bm, bn, warps))
        actual = kernel.gemm_triton(_cuda(a), _cuda(b), spec).cpu()
        assert bit_equal(actual, expected)
        assert torch.isfinite(actual).all()
    assert len(launches) == 3
    for compiled in launches:
        assert "_gemm" in compiled.name
        assert "mma.sync" not in compiled.asm["ptx"] and "wgmma" not in compiled.asm["ptx"]
    print(f"MMA_KERNEL mode={algorithm} symbol={launches[-1].name} "
          "y_finite=True max_diff=0 bypass_detected=False")


if triton is not None:
    from tricast.kernels.mma_core import c_operand, decode_f32, fixed_to_f32, rshift

    @triton.jit
    def _decode_probe(X, NEG, EXP, MAG, MB: tl.constexpr, EMIN: tl.constexpr,
                      INT: tl.constexpr, FRAC: tl.constexpr, C: tl.constexpr, F: tl.constexpr):
        idx = tl.arange(0, 32)
        x = tl.load(X + idx)
        if C:
            neg, e, m, _, _, _ = c_operand(x, F)
        else:
            neg, e, m = decode_f32(x, MB, EMIN, INT, FRAC)
        tl.store(NEG + idx, neg)
        tl.store(EXP + idx, e)
        tl.store(MAG + idx, m)

    @triton.jit
    def _fixed_probe(S, E, OUT, SHIFT_OUT, F: tl.constexpr, RNE: tl.constexpr):
        i = tl.arange(0, 32)
        s, e = tl.load(S + i), tl.load(E + i)
        tl.store(OUT + i, fixed_to_f32(s, e, F, RNE))
        tl.store(SHIFT_OUT + i, rshift(tl.abs(s), 48 + i))


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name)
def test_native_decode(backends, fmt):
    gen = torch.Generator().manual_seed(42)
    values = round_to_format(torch.randn(32, generator=gen) * 5, fmt)
    values[0] = 0
    if not isinstance(fmt, IntFormat):
        values[1:4] = torch.tensor([fmt.min_subnormal, -fmt.min_subnormal, fmt.min_normal])
    negative, exponent, mantissa, _ = decode(values, fmt)
    actual = [torch.empty(32, dtype=d, device="cuda") for d in (torch.bool, torch.int32, torch.int64)]
    descriptor = backends[1]._format(fmt)
    _decode_probe[(1,)](values.cuda(), *actual, *descriptor[:4], False, 23)
    assert torch.equal(actual[0].cpu(), negative)
    assert torch.equal(actual[1].cpu(), exponent)
    assert torch.equal(actual[2].cpu(), mantissa)


@pytest.mark.parametrize("f_bits", [3, 7, 13, 23, 25, 35])
@pytest.mark.parametrize("rounding", ["rtz", "rne"])
def test_fixed_and_running_subnormals(backends, f_bits, rounding):
    reference, _ = backends
    s = torch.tensor([0, 1, -1, (1 << 24) + 1, (1 << 24) + 3, (1 << 25) - 1,
                      (1 << 60) - 1, -((1 << 60) - 1)] * 4, dtype=torch.int64)
    e = torch.tensor([f_bits - 151, f_bits - 149, f_bits - 130, f_bits + 100] * 8)
    expected = reference._fixed_to_fp32(s, e, f_bits, rounding)
    actual = torch.empty(32, dtype=torch.float32, device="cuda")
    shifted = torch.empty(32, dtype=torch.int64, device="cuda")
    _fixed_probe[(1,)](s.cuda(), e.cuda(), actual, shifted, f_bits, rounding == "rne")
    assert bit_equal(actual.cpu(), expected)
    assert shifted.cpu().tolist() == [abs(int(v)) >> (48 + i) for i, v in enumerate(s)]
    bits = torch.tensor([0, 1, 2, 3, 0x7FFFFF, 0x800000, 0x807FFFFF, 0x80000001] * 4)
    values = bits.to(torch.int32).view(torch.float32)
    terms = reference._fp32_terms(values, f_bits)
    decoded = [torch.empty(32, dtype=d, device="cuda") for d in (torch.bool, torch.int32, torch.int64)]
    _decode_probe[(1,)](values.cuda(), *decoded, 23, -126, False, 0, True, f_bits)
    assert torch.equal(decoded[0].cpu(), terms.negative)
    assert torch.equal(decoded[1].cpu()[values != 0], terms.exponent[values != 0])
    assert torch.equal(decoded[2].cpu(), terms.significand)


def test_invalid_integer_specials(backends):
    for value in (float("nan"), float("inf")):
        a = Operand(torch.tensor([[value]], device="cuda"), INT8)
        b = Operand(torch.ones(1, 1, device="cuda"), INT8)
        with pytest.raises(ValueError, match="finite"):
            backends[1].gemm_triton(a, b, MMASpec("int_exact"))


def test_int_exact_subnormal_rounding(backends):
    fmt = IntFormat("fixed_subnormal", 8, frac_bits=75)
    a = Operand(torch.tensor([[3 * 2.0**-75]]), fmt)
    b = Operand(torch.tensor([[2.0**-75]]), fmt)
    _compare(backends, a, b, MMASpec("int_exact", out_format=FP32))


def test_e8m0_nan_scale(backends):
    a = Operand(torch.ones(1, 1), FP4_E2M1, torch.tensor([[float("nan")]]), E8M0, "k", 8)
    b = Operand(torch.ones(1, 1), FP4_E2M1)
    for algorithm in ("cofda", "gdfs"):
        spec = MMASpec(algorithm, f_bits=13, chunk_size=8, group_size=8, k_tile=32, out_format=FP32)
        _compare(backends, a, b, spec)


@pytest.mark.skipif(os.environ.get("TRICAST_MMA_PERF") != "1", reason="set TRICAST_MMA_PERF=1")
def test_hopper_performance_smoke(backends, tmp_path):
    _, kernel = backends
    gen = torch.Generator().manual_seed(42)
    a = _cuda(_operand(1024, 1024, FP8_E4M3, "none", gen))
    b = _cuda(_operand(1024, 1024, FP8_E4M3, "none", gen))
    spec = get_preset("nvidia_hopper_fp8")
    root = Path(__file__).resolve().parents[2]
    env = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
        "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "gpu_id": torch.cuda.current_device(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "seed": 42,
        "warmup": 3, "repeat": 5, "shape": [1024, 1024, 1024], "preset": spec.name,
        "model_sha": None, "dataset_sha": None,
    }
    # Includes K-major packing, allocation and epilogue; no GEMM-only speed claim.
    for _ in range(3):
        kernel.gemm_triton(a, b, spec)
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        kernel.gemm_triton(a, b, spec)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    median, mean = statistics.median(times), statistics.mean(times)
    p99 = float(torch.quantile(torch.tensor(times, dtype=torch.float64), 0.99))
    result = {"seconds": times, "median": median, "mean": mean, "p99": p99,
              "emulated_tmac_s": 1024**3 / median / 1e12}
    (tmp_path / "env.json").write_text(json.dumps(env, indent=2))
    (tmp_path / "result.json").write_text(json.dumps(result, indent=2))
    print(f"emulated TMAC/s={result['emulated_tmac_s']:.6f} median={median:.6f}s "
          f"mean={mean:.6f}s p99={p99:.6f}s artifacts={tmp_path}")


@pytest.mark.parametrize("algorithm", ["cofda", "gdfs"])
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("format_name", ["e8m0", "e8m0:bias=128", "e7m0:bias=127"])
@pytest.mark.parametrize("exponent", [-127, -126])
def test_pow2_field_zero_is_specific_to_standard_e8m0(
    backends, algorithm: str, side: str, format_name: str, exponent: int,
) -> None:
    from tricast.formats import get_format

    values = torch.ones((1, 8), dtype=torch.float32)
    scaled_op = Operand(values, FP4_E2M1, torch.tensor([[2.0**exponent]]),
                        get_format(format_name), "k", 8)
    unscaled_op = Operand(values, FP4_E2M1)
    a, b = (scaled_op, unscaled_op) if side == "a" else (unscaled_op, scaled_op)
    spec = MMASpec(algorithm, f_bits=25, chunk_size=8, g_bits=6, group_size=8,
                   k_tile=8, out_format=FP32)
    actual = _compare(backends, a, b, spec)
    # Check the exact mathematical answer too, not just shared backend agreement.
    expected = 0 if format_name == "e8m0" and exponent == -127 else (exponent + 130) << 23
    assert actual.view(torch.int32).item() == expected


@pytest.mark.parametrize("shape", [(2**16, 1, 2**15), (1, 2**16, 2**15), (2**15, 2**16, 1)])
def test_mma_rejects_int32_index_overflow(backends: tuple, shape: tuple[int, int, int]) -> None:
    _, kernel = backends
    m, n, k = shape
    value = torch.zeros((), device="cuda")
    a = Operand(value.expand(m, k), FP32)
    b = Operand(value.expand(n, k), FP32)
    # Reject before contiguous K-major packing or output allocation, for both
    # input strides and the output's row * N address calculation.
    with pytest.raises(ValueError, match=r"fewer than 2\^31 input/output elements"):
        kernel.gemm_triton(a, b, MMASpec("fp32_fma", out_format=FP32))
