"""Exact arithmetic oracles and layout invariants for the MMA reference."""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from types import ModuleType

import pytest
import torch

from tricast.formats import BF16, FP8_E4M3, FP16, FP32, INT8, MXINT8, UE4M3, IntFormat
from tricast.mma.api import as_operand, gemm
from tricast.mma.operand import Operand
from tricast.mma.spec import MMASpec
from tricast.quant.spec import QuantSpec, ScaleSpec
from tricast.reference.cast import round_to_format

from .conftest import bit_equal, wide_fp32


def _pow2(exponent: int) -> Fraction:
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)


def _binary_round(value: Fraction, *, precision: int = 24, rtz: bool = False) -> float:
    """Round an exact rational directly to binary32 or binary64, without double rounding."""
    if not value:
        return 0.0
    negative = value < 0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _pow2(exponent):
        exponent -= 1
    emin, emax = (-126, 127) if precision == 24 else (-1022, 1023)
    quantum = max(exponent, emin) - precision + 1
    units = value / _pow2(quantum)
    mantissa, remainder = divmod(units.numerator, units.denominator)
    if not rtz and (2 * remainder > units.denominator
                    or (2 * remainder == units.denominator and mantissa % 2)):
        mantissa += 1
    magnitude = mantissa * _pow2(quantum)
    if magnitude >= _pow2(emax + 1):
        result = float("inf")
    else:
        result = math.ldexp(float(mantissa), quantum)
    return -result if negative else result


def _dot_exact(a: torch.Tensor, b: torch.Tensor, *, chain_precision: int | None = None,
               rtz: bool = False) -> torch.Tensor:
    rows = []
    for a_row in a.tolist():
        row = []
        for b_row in b.tolist():
            acc = Fraction(0)
            for av, bv in zip(a_row, b_row, strict=True):
                acc += Fraction(av) * Fraction(bv)
                if chain_precision is not None:
                    acc = Fraction(_binary_round(acc, precision=chain_precision))
            row.append(_binary_round(acc, rtz=rtz))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def _reference(a: Operand, b: Operand, spec: MMASpec) -> torch.Tensor:
    mma = pytest.importorskip("tricast.reference.mma")
    return mma.gemm_reference(a, b, spec)


@pytest.mark.parametrize("seed", [7, 42, 91])
def test_wide_cofda_matches_fraction_sum_rtz(seed):
    generator = torch.Generator().manual_seed(seed)
    grid = torch.tensor([-448, -13, -1.875, -0.001953125, 0, 0.015625, 0.75, 3.5, 128, 448])
    a = grid[torch.randint(grid.numel(), (3, 31), generator=generator)]
    b = grid[torch.randint(grid.numel(), (4, 31), generator=generator)]
    spec = MMASpec(f_bits=40, chunk_size=32, out_format=FP32)
    actual = _reference(Operand(a, FP8_E4M3), Operand(b, FP8_E4M3), spec)
    assert bit_equal(actual, _dot_exact(a, b, rtz=True))


def test_reducing_fraction_bits_increases_mean_error(gen):
    # Positive products avoid cancellation; this checks the aggregate trend, not per-dot monotonicity.
    a = round_to_format(torch.rand((6, 29), generator=gen) * 16, FP8_E4M3)
    b = round_to_format(torch.rand((7, 29), generator=gen) * 16, FP8_E4M3)
    exact = _dot_exact(a, b)
    errors = []
    for bits in (40, 13, 7, 3):
        actual = _reference(Operand(a, FP8_E4M3), Operand(b, FP8_E4M3),
                            MMASpec(f_bits=bits, chunk_size=32, out_format=FP32))
        errors.append((actual.double() - exact.double()).abs().mean().item())
    assert errors == sorted(errors)
    assert errors[-1] > errors[0]


@pytest.mark.parametrize("algorithm,precision", [("fp32_fma", 24), ("fp64", 53)])
def test_fma_chains_match_fraction(algorithm, precision, gen):
    a = wide_fp32(3 * 17, gen, 100, 155).reshape(3, 17)
    b = wide_fp32(4 * 17, gen, 100, 155).reshape(4, 17)
    actual = _reference(Operand(a, FP32), Operand(b, FP32),
                        MMASpec(algorithm=algorithm, out_format=FP32))
    assert bit_equal(actual, _dot_exact(a, b, chain_precision=precision))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_fp32_fma_does_not_double_round(sign):
    # The second product is a binary32 midpoint. The tiny first addend selects its upper neighbour.
    a = torch.tensor([[sign * 2.0**-80, sign * (1 + 2.0**-12)]])
    b = torch.tensor([[1.0, 1 + 2.0**-12]])
    expected = _dot_exact(a, b, chain_precision=24)
    naive = (a.double() * b.double()).sum(dim=1).float().reshape(1, 1)
    assert not bit_equal(expected, naive)
    actual = _reference(Operand(a, FP32), Operand(b, FP32),
                        MMASpec(algorithm="fp32_fma", out_format=FP32))
    assert bit_equal(actual, expected)


@pytest.mark.parametrize("fmt", [INT8, MXINT8])
def test_int_exact_matches_integer_sum(fmt, gen):
    a_int = torch.randint(-127, 128, (3, 97), generator=gen, dtype=torch.int64)
    b_int = torch.randint(-127, 128, (4, 97), generator=gen, dtype=torch.int64)
    a, b = a_int.float() * 2.0**-fmt.frac_bits, b_int.float() * 2.0**-fmt.frac_bits
    actual = _reference(Operand(a, fmt), Operand(b, fmt),
                        MMASpec(algorithm="int_exact", out_format=FP32))
    expected = ((a_int @ b_int.T).double() * 2.0**(-2 * fmt.frac_bits)).float()
    assert bit_equal(actual, expected)


def test_int_exact_rounds_large_integer_sum_once():
    fmt = IntFormat("uint24", 24, signed=False)
    values = torch.tensor([[2.0**23] * 256 + [2.0**15, 1.0]])
    # S = 2^54 + 2^30 + 1 is just above an fp32 midpoint, but fp64 loses its sticky bit.
    exact = Fraction(2**54 + 2**30 + 1)
    expected = torch.tensor([[_binary_round(exact)]])
    naive = torch.tensor([[float(exact)]], dtype=torch.float64).float()
    assert not bit_equal(expected, naive)
    actual = _reference(Operand(values, fmt), Operand(values, fmt),
                        MMASpec(algorithm="int_exact", out_format=FP32))
    assert bit_equal(actual, expected)


@pytest.mark.parametrize("algorithm", ["cofda", "gdfs", "fp32_fma", "fp64", "int_exact"])
def test_outputs_are_independent_under_row_column_permutation(algorithm, gen):
    a = torch.randint(-4, 5, (4, 19), generator=gen).float()
    b = torch.randint(-4, 5, (5, 19), generator=gen).float()
    fmt = INT8 if algorithm == "int_exact" else FP8_E4M3
    spec = MMASpec(algorithm=algorithm, f_bits=13, g_bits=12, group_size=4,
                   k_tile=8, chunk_size=7, out_format=FP32)
    expected = _reference(Operand(a, fmt), Operand(b, fmt), spec)
    perm_a, perm_b = torch.tensor([2, 0, 3, 1]), torch.tensor([4, 1, 3, 0, 2])
    actual = _reference(Operand(a[perm_a], fmt), Operand(b[perm_b], fmt), spec)
    assert bit_equal(actual, expected[perm_a][:, perm_b])


@pytest.mark.parametrize("dtype,fmt", [(torch.float32, FP32), (torch.float16, FP16),
                                      (torch.bfloat16, BF16)])
def test_plain_tensor_operand_flattens_leading_dimensions(dtype, fmt):
    x = torch.arange(24).reshape(2, 3, 4).to(dtype).transpose(0, 1)
    operand = as_operand(x)
    assert operand.fmt == fmt
    assert operand.values.dtype == torch.float32
    assert torch.equal(operand.values, x.reshape(6, 4).float())
    assert operand.scale is None
    assert as_operand(operand) is operand


def _qtensor(values, spec, scale=None, global_scale=None, zero_point=None, shape=None):
    qt = pytest.importorskip("tricast.quant.qtensor")
    return qt.QTensor(values=values, scale=scale, zero_point=zero_point,
                      global_scale=global_scale, spec=spec,
                      shape=tuple(values.shape) if shape is None else shape)


@pytest.mark.parametrize("granularity,options,scale,kind,domain", [
    ("tensor", {}, torch.tensor(2.0), "tensor", 0),
    ("row", {}, torch.tensor([[2.0], [3.0], [4.0]]), "row", 0),
    ("group", {"group_size": 3}, torch.tensor([[2.0, 3], [4, 5], [6, 7]]), "k", 3),
    ("block", {"block": (2, 3)}, torch.tensor([[2.0, 3], [4, 5]]), "k", 3),
])
def test_qtensor_scale_layout(granularity, options, scale, kind, domain):
    spec = QuantSpec(FP8_E4M3, granularity=granularity, **options)
    values = torch.ones((3, 5), dtype=torch.bfloat16)
    operand = as_operand(_qtensor(values, spec, scale))
    expected = scale.repeat_interleave(2, dim=0)[:3] if granularity == "block" else scale
    assert operand.scale_kind == kind
    assert operand.k_domain == domain
    assert operand.scale_fmt == FP32
    assert torch.equal(operand.scale, expected)
    assert operand.values.dtype == torch.float32
    assert torch.equal(operand.values, values.float())
    expanded = expected.repeat_interleave(3, dim=1)[:, :5] if kind == "k" else expected.expand(3, 5)
    assert torch.equal(operand.scale_per_element(), expanded)


def test_qtensor_two_level_keeps_alpha_separate():
    spec = QuantSpec(FP8_E4M3, "group", group_size=2, scale=ScaleSpec(UE4M3, two_level=True))
    scale = torch.tensor([[1.0, 2.0]])
    alpha = torch.tensor(0.125)
    operand = as_operand(_qtensor(torch.ones(1, 3), spec, scale, alpha))
    assert operand.alpha is alpha
    assert torch.equal(operand.scale, scale)
    assert operand.scale_fmt == UE4M3


def test_qtensor_direct_cast_has_no_scale():
    values = torch.tensor([[1.0, 1.5]])
    operand = as_operand(_qtensor(values, QuantSpec(BF16, scale=None)))
    assert operand.fmt == BF16
    assert operand.scale_kind == "none"
    assert operand.alpha is None
    assert torch.equal(operand.values, values)


def test_dequantized_qtensor_carries_no_scales():
    spec = QuantSpec(INT8, "row", mma_input="dequant", dequant_format=BF16, zero_point="int")
    qt = _qtensor(torch.tensor([[3.0, -2.0]]), spec,
                  scale=torch.tensor([[0.1]]), zero_point=torch.tensor([[1.0]]))
    operand = as_operand(qt)
    expected = round_to_format(torch.tensor([[0.2, -0.3]]), BF16, saturate=False)
    assert operand.fmt == BF16
    assert operand.scale is None
    assert operand.alpha is None
    assert torch.equal(operand.values, expected)


@pytest.mark.parametrize("spec", ["fp32_fma", {"algorithm": "fp32_fma", "out_format": "fp32"},
                                  MMASpec(algorithm="fp32_fma", out_format=FP32)])
def test_gemm_preserves_activation_shape(spec):
    pytest.importorskip("tricast.reference.mma")
    a, b = torch.arange(24).reshape(2, 3, 4).float(), torch.eye(4)
    actual = gemm(a, b, spec, backend="reference")
    assert actual.shape == (2, 3, 4)
    assert bit_equal(actual, a)
    assert bit_equal(gemm(a[0, 0], b, spec), a[0, 0])
    assert bit_equal(gemm(as_operand(a), as_operand(b), spec), a.reshape(6, 4))


def test_gemm_qtensor_restores_original_shape():
    pytest.importorskip("tricast.reference.mma")
    values = torch.arange(24).reshape(6, 4).float()
    qt = _qtensor(values, QuantSpec(FP32, scale=None), shape=(2, 3, 4))
    actual = gemm(qt, torch.eye(4), {"algorithm": "fp32_fma", "out_format": "fp32"})
    assert actual.shape == qt.shape
    assert bit_equal(actual, values.reshape(qt.shape))


def test_gemm_vector_qtensor_restores_original_shape():
    pytest.importorskip("tricast.reference.mma")
    values = torch.tensor([[1.0, 2.0, 3.0]])
    qt = _qtensor(values, QuantSpec(FP32, scale=None), shape=(3,))
    actual = gemm(qt, torch.eye(3), {"algorithm": "fp32_fma", "out_format": "fp32"})
    assert actual.shape == (3,)
    assert bit_equal(actual, values[0])


@pytest.mark.parametrize("mma_input", ["scaled", "dequant"])
def test_scalar_qtensor_is_not_an_mma_operand(mma_input):
    qt = _qtensor(torch.ones(1, 1), QuantSpec(FP32, scale=None, mma_input=mma_input), shape=())
    with pytest.raises(ValueError, match="dimension"):
        as_operand(qt)


@pytest.mark.parametrize("cuda,backend,expected", [(False, "auto", "reference"),
                                                   (True, "auto", "triton"),
                                                   (True, "reference", "reference"),
                                                   (False, "triton", "triton")])
def test_backend_dispatch(monkeypatch, cuda, backend, expected):
    calls = []
    for name in ("reference", "kernels"):
        module = ModuleType(f"tricast.{name}.mma")
        label = "reference" if name == "reference" else "triton"

        def run(a, b, spec, bias, label=label):
            calls.append((label, bias))
            return torch.zeros((a.rows, b.rows))

        setattr(module, "gemm_reference" if name == "reference" else "gemm_triton", run)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: cuda))
    bias = torch.zeros(2)
    out = gemm(torch.ones(3, 4), torch.ones(2, 4), "fp32_fma", bias=bias, backend=backend)
    assert out.shape == (3, 2)
    assert calls == [(expected, bias)]


def test_auto_falls_back_when_triton_import_is_unavailable(monkeypatch):
    pytest.importorskip("tricast.reference.mma")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setitem(sys.modules, "tricast.kernels.mma", None)
    assert gemm(torch.ones(1, 2), torch.ones(1, 2), "fp32_fma").item() == 2
    with pytest.raises(ImportError):
        gemm(torch.ones(1, 2), torch.ones(1, 2), "fp32_fma", backend="triton")


@pytest.mark.parametrize("a,b,backend,message", [
    (torch.ones(2, 3), torch.ones(4, 3), "bad", "backend"),
    (torch.ones(2, 3), torch.ones(4, 2), "reference", "reduction"),
    (torch.ones(2, 3), torch.ones(2, 4, 3), "reference", "weights"),
    (torch.tensor(1.0), torch.ones(1, 1), "reference", "dimension"),
])
def test_api_rejects_invalid_inputs(a, b, backend, message):
    with pytest.raises(ValueError, match=message):
        gemm(a, b, "fp32_fma", backend=backend)


@pytest.mark.parametrize("f_bits", [23, 32, 40])
def test_cofda_fp32_product_scales_preserve_wide_intermediate(f_bits):
    value = 2.0 - 2.0**-23
    scale = torch.tensor([[value]])
    a = Operand(torch.tensor([[value], [-value]]), FP32, scale.expand(2, 1), FP32, "k", 1)
    b = Operand(torch.tensor([[value]]), FP32, scale, FP32, "k", 1)
    exact = Fraction(value) ** 4
    assert exact.numerator.bit_length() == 96
    # Four 24-bit significands are multiplied before the one radix-F truncation.
    fixed = Fraction(int(exact * 2**f_bits), 2**f_bits)
    expected = torch.tensor([[_binary_round(fixed, rtz=True)], [_binary_round(-fixed, rtz=True)]])
    actual = _reference(a, b, MMASpec(f_bits=f_bits, chunk_size=1, out_format=FP32))
    assert torch.isfinite(actual).all()
    assert bit_equal(actual, expected)


def test_gdfs_fp32_scales_preserve_wide_group_intermediate():
    scale_value = 2.0 - 2.0**-23
    scale = torch.tensor([[scale_value]])
    a = Operand(torch.tensor([[1.5, 1.75]]), FP8_E4M3, scale, FP32, "k", 2)
    b = Operand(torch.tensor([[1.5, 1.25]]), FP8_E4M3, scale, FP32, "k", 2)
    group_sum = Fraction(1.5) ** 2 + Fraction(1.75) * Fraction(1.25)
    group_significand = int(group_sum * 2**40)
    scale_significand = int(Fraction(scale_value) * 2**23)
    assert (group_significand * scale_significand**2).bit_length() > 64
    # The G=40 group sum is exact; only the final scaled group is truncated to F=35.
    exact = group_sum * Fraction(scale_value) ** 2
    fixed = Fraction(int(exact * 2**35), 2**35)
    expected = torch.tensor([[_binary_round(fixed, rtz=True)]])
    spec = MMASpec(algorithm="gdfs", f_bits=35, g_bits=40, group_size=2, k_tile=2, out_format=FP32)
    actual = _reference(a, b, spec)
    assert torch.isfinite(actual).all()
    assert bit_equal(actual, expected)


def test_int_exact_subnormal_midpoints_and_sticky_bits():
    fmt = IntFormat("fixed24_q80", 24, frac_bits=80)
    integers = torch.tensor([[1024, 0], [1024, 1], [3072, 0], [3072, -1],
                             [-1024, 0], [-1024, -1], [-3072, 0], [-3072, 1]])
    a = Operand(integers.float() * 2.0**-80, fmt)
    b = Operand(torch.ones(1, 2) * 2.0**-80, fmt)
    # One subnormal ulp is 2^11 integer units: exercise even/odd ties and ±1 sticky offsets.
    expected = torch.tensor([[_binary_round(Fraction(int(row.sum()), 2**160))] for row in integers])
    actual = _reference(a, b, MMASpec(algorithm="int_exact", out_format=FP32))
    assert bit_equal(actual, expected)
    assert (actual.abs() < torch.finfo(torch.float32).tiny).all()
