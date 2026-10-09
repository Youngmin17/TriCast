"""Hand-derived scale, layout, reconstruction and STE checks."""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.conftest import bit_equal
from tricast.formats import E8M0, FP4_E2M1, FP16, FP32, UE4M3, container_dtype
from tricast.quant.api import fake_quant, quantize, resolve_backend
from tricast.quant.qtensor import QTensor
from tricast.quant.spec import SCHEMES, QuantSpec, ScaleSpec
from tricast.reference.cast import round_to_format
from tricast.reference.quantize import (
    compute_amax,
    compute_scale,
    expand_scale,
    quantize_elements,
    quantize_reference,
    view_2d,
)
from tricast.rounding import Rounding

LAYOUTS = {
    "tensor": {}, "row": {}, "group": {"group_size": 2}, "block": {"block": (2, 3)},
}
X = torch.tensor([[1., 2., 3., 4., 5.], [-6., 0., 1., 2., 3.], [0., 4., 8., 12., 16.]])
AMAX = {
    "tensor": 16., "row": [[5.], [6.], [16.]],
    "group": [[2., 4., 5.], [6., 2., 3.], [4., 12., 16.]], "block": [[6., 5.], [8., 16.]],
}
MEDIAN = {
    "tensor": 3., "row": [[3.], [2.], [8.]],
    "group": [[1.5, 3.5, 5.], [3., 1.5, 3.], [2., 10., 16.]], "block": [[1.5, 3.5], [4., 14.]],
}
FLOOR = {
    "tensor": 4., "row": [[1.], [1.], [4.]],
    "group": [[.5, 1., 1.], [1., .5, .5], [1., 2., 4.]], "block": [[1., 1.], [2., 4.]],
}
CEIL = {
    "tensor": 4., "row": [[1.], [1.], [4.]],
    "group": [[.5, 1., 1.], [1., .5, .5], [1., 2., 4.]], "block": [[1., 1.], [2., 4.]],
}


def spec_for(granularity, scale=None, **kwargs):
    return QuantSpec(FP4_E2M1, granularity, **LAYOUTS[granularity],
                     scale=scale if scale is not None else ScaleSpec(), **kwargs)


@pytest.mark.parametrize("granularity", LAYOUTS)
@pytest.mark.parametrize("method", ["absmax", "pow2_floor", "pow2_ceil", "percentile", "mse"])
def test_hand_scales(granularity, method):
    sf = E8M0 if method.startswith("pow2") else FP32
    spec = spec_for(granularity, ScaleSpec(sf, method, percentile=50, search=(1.,)))
    assert bit_equal(compute_amax(X, spec), torch.tensor(AMAX[granularity]))
    scale, zp, global_scale = compute_scale(X, spec)
    expected = {"pow2_floor": FLOOR, "pow2_ceil": CEIL, "percentile": MEDIAN}.get(method, AMAX)
    expected = torch.tensor(expected[granularity])
    if not method.startswith("pow2"):
        expected = expected / 6
    assert bit_equal(scale, expected)
    assert zp is None and global_scale is None


@pytest.mark.parametrize("granularity", LAYOUTS)
def test_layout_expansion_and_nan_domains(granularity):
    spec = spec_for(granularity)
    maxima = compute_amax(X, spec)
    expanded = expand_scale(maxima, spec, *X.shape)
    assert expanded.shape == X.shape
    q = quantize_reference(X, spec)
    assert bit_equal(q.scale_per_element(), expand_scale(q.scale, spec, *X.shape))
    op = q.mma_operand()
    assert op.values.dtype == torch.float32
    assert bit_equal(op.scale_per_element(), q.scale_per_element())
    assert op.scale_kind == ("k" if granularity in ("group", "block") else granularity)
    if granularity in ("group", "block"):
        assert op.k_domain == (2 if granularity == "group" else 3)
    bad = X.clone()
    bad[0, 0] = float("nan")
    qbad = quantize_reference(bad, spec)
    mask = expand_scale(torch.isnan(compute_amax(bad, spec)), spec, *X.shape)
    assert torch.equal(torch.isnan(qbad.dequantize()), mask)


@pytest.mark.parametrize("granularity", LAYOUTS)
@pytest.mark.parametrize("method", ["absmax", "percentile", "mse", "pow2_floor", "pow2_ceil"])
def test_two_level_order(granularity, method):
    spec = spec_for(granularity, ScaleSpec(UE4M3, method, two_level=True, percentile=50, search=(1.,)))
    q = quantize_reference(X, spec)
    expected_d2 = X.new_tensor(16.) / (6 * 448)
    maxima = torch.tensor((MEDIAN if method == "percentile" else AMAX)[granularity])
    # Native fp8 E4M3 has the same positive scale grid as UE4M3.
    expected_s = ((maxima / 6) / expected_d2).clamp(max=448).to(torch.float8_e4m3fn).float()
    assert bit_equal(q.global_scale, expected_d2)
    assert bit_equal(q.scale, expected_s)
    assert bit_equal(q.dequantize(), (q.values.float() * q.scale_per_element()) * expected_d2)
    assert bit_equal(q.mma_operand().alpha, expected_d2)


@pytest.mark.parametrize("element,scale_format,maximum", [
    (FP32, UE4M3, 3e38), (FP4_E2M1, E8M0, 6.), (FP4_E2M1, FP32, 6.),
    ("mxint4", E8M0, 1.), (FP4_E2M1, UE4M3, 17.),
])
def test_global_scale_exact_product_no_denominator_overflow(element, scale_format, maximum):
    method = "pow2_floor" if scale_format == E8M0 else "absmax"
    spec = QuantSpec(element, scale=ScaleSpec(scale_format, method, two_level=True))
    x = torch.tensor([[maximum]])
    scale, _, d2 = compute_scale(x, spec)
    exact = Fraction(x.item()) / (Fraction(spec.format.max_normal) * Fraction(scale_format.max_normal))
    error = abs(Fraction(d2.item()) - exact)
    for toward in (0., float("inf")):
        neighbor = torch.nextafter(d2, torch.tensor(toward))
        assert error <= abs(Fraction(neighbor.item()) - exact)
    assert d2 > 0 and torch.isfinite(d2)
    expected_scale = round_to_format((x.amax() / spec.format.max_normal) / d2, scale_format)
    assert bit_equal(scale, expected_scale)


def test_two_level_observer_amax_freezes_both_levels():
    spec = QuantSpec(FP4_E2M1, scale=ScaleSpec(UE4M3, two_level=True))
    for value in (6., 12.):
        q = quantize_reference(torch.tensor([[value]]), spec, amax=torch.tensor(2688.))
        assert q.global_scale == 1 and q.scale == 448


def test_mxfp4_explicit_elements_and_short_group():
    spec = QuantSpec(FP4_E2M1, "group", group_size=4, scale=ScaleSpec(E8M0, "pow2_floor"))
    x = torch.tensor([[0., 1., 3., 7., -8., 12.]])
    q = quantize_reference(x, spec)
    assert bit_equal(q.scale, torch.tensor([[1., 2.]]))
    assert bit_equal(q.values, torch.tensor([[0., 1., 3., 6., -4., 6.]]))
    assert bit_equal(q.dequantize(), torch.tensor([[0., 1., 3., 6., -8., 12.]]))


def test_pow2_ceil_exact_threshold():
    x = torch.tensor([[6., 7., 8., 12., 13.]])
    spec = QuantSpec(FP4_E2M1, "group", group_size=1, scale=ScaleSpec(E8M0, "pow2_ceil"))
    assert bit_equal(compute_scale(x, spec)[0], torch.tensor([[1., 2., 2., 2., 4.]]))
    below = torch.nextafter(torch.tensor(8.), torch.tensor(0.))
    floor = replace(spec, scale=ScaleSpec(E8M0, "pow2_floor"))
    assert compute_scale(below.reshape(1, 1), floor)[0].item() == 1.


@pytest.mark.parametrize("method", ["pow2_floor", "pow2_ceil"])
def test_pow2_clamps_zero_underflow_overflow_and_nan(method):
    # int2 max=1 lets the scale itself overflow E8M0's exponent range via +Inf.
    spec = QuantSpec("int2", "group", group_size=1, scale=ScaleSpec(E8M0, method))
    x = torch.tensor([[0., 2.**-149, 2.**127, float("inf"), float("nan")]])
    q = quantize_reference(x, spec)
    assert bit_equal(q.scale, torch.tensor([[2.**-126, 2.**-127, 2.**127, float("nan"), float("nan")]]))
    assert torch.isnan(q.values[0, 3:]).all()
    tiny_sf = replace(E8M0, name="e3m0", ebits=3, bias=3)
    tiny = replace(spec, scale=ScaleSpec(tiny_sf, method))
    assert torch.isnan(quantize_reference(torch.tensor([[16.]]), tiny).dequantize()).all()


@pytest.mark.parametrize("two_level", [False, True])
@pytest.mark.parametrize("granularity", LAYOUTS)
def test_zero_domain_guards(granularity, two_level):
    spec = spec_for(granularity, ScaleSpec(UE4M3, two_level=two_level))
    q = quantize_reference(torch.zeros(3, 5), spec)
    assert (q.scale == 1).all()
    assert (q.dequantize() == 0).all()
    if two_level:
        assert q.global_scale == 1


def test_absmax_underflow_guard():
    spec = QuantSpec("int4", scale=ScaleSpec(FP16))
    q = quantize_reference(torch.tensor([2.**-30]), spec)
    assert q.scale == 2.**-24
    assert q.values.item() == 0


def test_mse_search_default_and_first_tie():
    x = torch.tensor([[4.5] * 10 + [6.]])
    spec = spec_for("tensor", ScaleSpec(method="mse", mse_grid=3))
    q = quantize_reference(x, spec)
    # r=1: SSE=10*(4.5-4)^2=2.5; r=.75: SSE=(6-4.5)^2=2.25; r=.5: SSE=31.5.
    assert q.scale == .75
    assert (q.values == 6).all()
    tie = spec_for("tensor", ScaleSpec(method="mse", search=(1., 2.)))
    assert quantize_reference(torch.tensor([[0., 6.]]), tie).scale == 1.
    reversed_tie = replace(tie, scale=replace(tie.scale, search=(2., 1.)))
    assert quantize_reference(torch.tensor([[0., 6.]]), reversed_tie).scale == 2.


def test_four_over_six_and_percentile():
    spec = spec_for("tensor", ScaleSpec(method="mse", search=(1., 1.5)))
    q = quantize_reference(torch.tensor([[4.5, 6.]]), spec)
    assert q.scale == 1.5
    assert bit_equal(q.values, torch.tensor([[3., 4.]]))
    p = spec_for("tensor", ScaleSpec(method="percentile", percentile=50))
    qp = quantize_reference(torch.tensor([[0., 3., 6., 30.]]), p)
    assert qp.scale == .75
    assert bit_equal(qp.dequantize(), torch.tensor([[0., 3., 4.5, 4.5]]))


@pytest.mark.parametrize("granularity", LAYOUTS)
@pytest.mark.parametrize("zero_point", ["int", "float"])
@pytest.mark.parametrize("method", ["absmax", "percentile", "mse"])
def test_zero_points(granularity, zero_point, method):
    spec = QuantSpec("uint4", granularity, **LAYOUTS[granularity], scale=ScaleSpec(method=method),
                     zero_point=zero_point, mma_input="dequant", dequant_format=FP32)
    # Every domain spans 15: s=1, integer z=3, floating offset z=-3.
    x = torch.tensor([[-3., 12., -3., 12., -3., 12.]]).expand(2, 6).clone()
    q = quantize_reference(x, spec)
    assert (q.scale == 1).all()
    assert (q.zero_point == (3 if zero_point == "int" else -3)).all()
    assert bit_equal(q.dequantize(), x)
    assert bit_equal(q.mma_operand().values, x)


def test_zero_point_round_then_add_without_early_clipping():
    spec = QuantSpec("uint4", zero_point="int", mma_input="dequant", dequant_format=FP32)
    x = torch.tensor([[-5., -.5, .5, 10.]])
    q = quantize_reference(x, spec)
    assert q.scale == 1 and q.zero_point == 5
    assert bit_equal(q.values, torch.tensor([[0., 5., 5., 15.]]))
    assert bit_equal(q.dequantize(), torch.tensor([[-5., 0., 0., 10.]]))


def test_float_zero_point_does_not_drop_offset():
    spec = QuantSpec("uint4", zero_point="float", mma_input="dequant", dequant_format=FP32)
    x = torch.tensor([[-1.1, 0., 1., 13.9]])
    q = quantize_reference(x, spec)
    assert q.scale == 1
    assert q.zero_point == x.new_tensor(-1.1)
    # x-lo rounds to [0, 1.1, 2.1, 15], so stored values remain integer codes.
    assert bit_equal(q.values, torch.tensor([[0., 1., 2., 15.]]))
    expected = (torch.tensor([[0., 1., 2., 15.]], dtype=torch.float64) + x[0, 0].double()).float()
    assert bit_equal(q.dequantize(), expected)
    assert bit_equal(q.mma_operand().values, expected)
    assert q.values.dtype == container_dtype(spec.format)


@pytest.mark.parametrize("data,offset", [
    ([-1., .1, .5, 2.], -1.), ([2., 3., 4., 5.], 2.), ([-5., -4., -3., -2.], -5.),
])
def test_float_zero_point_hand_derived_uint2(data: list[float], offset: float) -> None:
    spec = SCHEMES["kivi2"].with_(group_size=4)
    q = quantize_reference(torch.tensor([data]), spec)
    # The range is three, independent of zero; the halfway code 1.5 rounds to two.
    assert q.scale.item() == 1. and q.zero_point.item() == offset
    assert bit_equal(q.values, torch.tensor([[0., 1., 2., 3.]]))
    expected = torch.tensor([[offset, offset + 1, offset + 2, offset + 3]])
    assert bit_equal(q.dequantize(), expected)
    assert bit_equal(q.mma_operand().values, expected)


@pytest.mark.parametrize("value", [-2., 0., 2., 2.**-149])
def test_float_zero_point_constant_and_short_group(value: float) -> None:
    spec = SCHEMES["kivi2"].with_(group_size=4)
    x = torch.full((2, 5), value)
    q = quantize_reference(x, spec)
    # Both full and one-element tail domains have hi=lo, hence scale one and code zero.
    assert bit_equal(q.scale, torch.ones(2, 2))
    assert bit_equal(q.zero_point, torch.full((2, 2), value))
    assert bit_equal(q.values, torch.zeros(2, 5))
    assert bit_equal(q.dequantize(), x)


@pytest.mark.parametrize("rounding,expected", [
    ("rne", [0., 0., 2., 2., 3.]), ("rna", [0., 1., 2., 3., 3.]),
    ("rtz", [0., 0., 1., 2., 3.]), ("rup", [0., 1., 2., 3., 3.]),
    ("rdn", [0., 0., 1., 2., 3.]),
])
def test_float_zero_point_rounds_shifted_values(rounding: str, expected: list[float]) -> None:
    spec = SCHEMES["kivi2"].with_(rounding=rounding)
    # lo=2 and s=1: rounding operates on [0, .5, 1.5, 2.5, 3], not on x itself.
    q = quantize_reference(torch.tensor([[2., 2.5, 3.5, 4.5, 5.]]), spec)
    assert bit_equal(q.values, torch.tensor([expected]))


@pytest.mark.parametrize("sr_bits", [2, 8, 16, 32])
def test_float_zero_point_explicit_sr(sr_bits: int) -> None:
    spec = SCHEMES["kivi2"].with_(rounding=Rounding.SR, sr_bits=sr_bits)
    x = torch.tensor([[2., 2.25, 2.25, 2.25, 2.25, 5.]])
    noise = torch.tensor([[0, 0, 2**30 - 1, 2**30, 2**32 - 1, 0]])
    # The shifted fraction is 1/4; only noise strictly below 1/4 increments the code.
    q = quantize_reference(x, spec, noise=noise)
    assert bit_equal(q.values, torch.tensor([[0., 1., 1., 0., 0., 3.]]))
    assert bit_equal(q.dequantize(), torch.tensor([[2., 3., 3., 2., 2., 5.]]))
    assert bit_equal(q.values, quantize_reference(x, spec, noise=noise).values)


def test_float_zero_point_subtract_rounds_before_divide() -> None:
    spec = SCHEMES["kivi2"]
    # RN(1 + 2^-24)=1, then /2 is an exact tie rounding to zero, not one.
    q = quantize_elements(torch.tensor([[1.]]), spec, torch.tensor(2.), torch.tensor(-2.**-24))
    assert q.item() == 0.


def test_float_zero_point_dequant_multiply_rounds_before_add() -> None:
    spec = SCHEMES["kivi2"].with_(scale=ScaleSpec(FP32), dequant_format=FP32)
    # RN(3*(1+2^-23))=3+2^-21; a fused multiply-add would instead give 3*2^-23.
    q = QTensor(torch.tensor([[3.]]), torch.tensor([[1. + 2.**-23]]), torch.tensor([[-3.]]),
                None, spec, (1, 1))
    assert q.dequantize().item() == 2.**-21
    assert q.mma_operand().values.item() == 2.**-21


@pytest.mark.parametrize("sign", [-1., 1.])
def test_float_zero_point_subnormal_arithmetic(sign: float) -> None:
    spec = SCHEMES["kivi2"].with_(scale=ScaleSpec(FP32), dequant_format=FP32)
    x = torch.tensor([[sign * k * 2.**-149 for k in (1, 2, 3, 4)]])
    q = quantize_reference(x, spec)
    # The range is three subnormal ULPs, so s=2^-149 and every input is exactly representable.
    assert q.scale.item() == 2.**-149
    assert q.zero_point.item() == x.min().item()
    assert bit_equal(q.values, torch.tensor([[0., 1., 2., 3.] if sign > 0 else [3., 2., 1., 0.]]))
    assert bit_equal(q.dequantize(), x)
    assert bit_equal(q.mma_operand().values, x)


@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 5)])
@pytest.mark.parametrize("name", SCHEMES)
def test_all_schemes_shape_container_and_device(name, shape):
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(shape, generator=gen)
    qt = quantize(x, name)
    assert qt.shape == shape and qt.K == 5 and qt.rows == x.numel() // 5
    assert qt.values.dtype == container_dtype(qt.spec.format)
    assert qt.dequantize().shape == shape
    assert torch.isfinite(qt.dequantize()).all()
    assert qt.dequantize(torch.float16).dtype == torch.float16
    moved = qt.to("cpu")
    for field in ("values", "scale", "zero_point", "global_scale"):
        a, b = getattr(qt, field), getattr(moved, field)
        assert b is None if a is None else bit_equal(a, b)
    assert moved.spec == qt.spec and moved.shape == shape
    if qt.spec.mma_input == "dequant":
        op = qt.mma_operand()
        assert op.scale is None and op.alpha is None and op.fmt == qt.spec.dequant_format
        assert bit_equal(op.values.reshape(shape), round_to_format(qt.dequantize(), op.fmt, saturate=False))


def test_noncontiguous_view_and_noise():
    x = torch.tensor([[[.125, .375], [.625, .875], [1.125, 1.375]]]).transpose(1, 2)
    spec = QuantSpec(FP4_E2M1, scale=None, rounding=Rounding.SR)
    noise = torch.tensor([[[0], [2**32 - 1]]])
    q = quantize_reference(x, spec, noise=noise)
    expected = round_to_format(x, FP4_E2M1, Rounding.SR, noise=noise)
    assert bit_equal(q.dequantize(), expected)
    assert torch.equal(view_2d(x), x.reshape(-1, 3))


def test_dequant_operation_order_not_folded():
    # These factors distinguish (q*s)*d2 from q*(s*d2) by one binary32 ULP.
    qv = torch.tensor([[3.]])
    s = torch.tensor([[176.]])
    d2 = torch.tensor(.43400341272354126)
    spec = spec_for("group", ScaleSpec(two_level=True))
    qt = QTensor(qv, s, None, d2, spec, (1, 1))
    assert not bit_equal((qv * s) * d2, qv * (s * d2))
    assert bit_equal(qt.dequantize(), (qv * s) * d2)


def test_observer_amax_override_not_searched_twice():
    x = torch.tensor([[1., 2., 6.]])
    for method in ("absmax", "percentile", "mse"):
        spec = spec_for("tensor", ScaleSpec(method=method, percentile=1, search=(.1,)))
        q = quantize_reference(x, spec, amax=torch.tensor(3.))
        assert q.scale == .5
    with pytest.raises(ValueError, match="tensor granularity"):
        quantize_reference(x, spec_for("row"), amax=torch.tensor(3.))
    with pytest.raises(ValueError, match="scalar"):
        quantize_reference(x, spec_for("tensor"), amax=torch.ones(2))


@pytest.mark.parametrize("dequant", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fake_quant_value_and_clipped_ste(dequant, dtype):
    spec = QuantSpec("int4", mma_input="dequant" if dequant else "scaled")
    x = torch.tensor([-15., -14., -2.5, 0., 2.5, 14., 15.], dtype=dtype, requires_grad=True)
    amax = torch.tensor(14., requires_grad=True)
    q = quantize(x.detach(), spec, amax=amax.detach())
    y = fake_quant(x, spec, amax=amax)
    expected = q.mma_operand().values if dequant else q.dequantize()
    assert y.dtype == dtype and bit_equal(y, expected.reshape(x.shape).to(dtype))
    y.sum().backward()
    assert bit_equal(x.grad, torch.tensor([0., 1., 1., 1., 1., 1., 0.]))
    assert amax.grad is None


def test_fake_quant_direct_cast_and_scheme():
    x = torch.tensor([-7., -6., 0., 6., 7.], requires_grad=True)
    y = fake_quant(x, QuantSpec(FP4_E2M1, scale=None))
    y.sum().backward()
    assert bit_equal(x.grad, torch.tensor([0., 1., 1., 1., 0.]))
    assert bit_equal(fake_quant(x.detach(), "bf16"), x.detach().bfloat16().float())


def test_backend_selection_and_dispatch(monkeypatch):
    import tricast.quant.api as api

    assert resolve_backend("auto", torch.ones(1)) == "reference"
    assert resolve_backend("reference", torch.ones(1)) == "reference"
    with pytest.raises(ValueError, match="CUDA"):
        quantize(torch.ones(1), "bf16", backend="triton")
    with pytest.raises(ValueError, match="backend"):
        resolve_backend("bad", torch.ones(1))
    fake_cuda = SimpleNamespace(is_cuda=True)
    calls = []
    monkeypatch.setattr(api, "import_module", lambda name: calls.append(name))
    assert resolve_backend("auto", fake_cuda) == "triton"
    assert calls == ["tricast.kernels"]
    assert resolve_backend("auto", fake_cuda, torch.ones(1)) == "reference"

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(api, "import_module", missing)
    assert resolve_backend("auto", fake_cuda) == "reference"


def test_elements_given_scales_and_explicit_sr():
    spec = QuantSpec(FP4_E2M1, rounding=Rounding.SR)
    x = torch.tensor([[.25, .25, .25, .25]])
    noise = torch.tensor([[0, 2**31 - 1, 2**31, 2**32 - 1]])
    out = quantize_elements(x, spec, torch.tensor(1.), noise=noise)
    assert bit_equal(out, torch.tensor([[.5, .5, 0., 0.]]))


def test_dispatch_forwards_spec_amax_and_noise(monkeypatch):
    import tricast.quant.api as api

    module = ModuleType("tricast.kernels.quantize")
    calls = []
    result = object()

    def kernel(x, spec, *, amax, noise):
        calls.append((x, spec, amax, noise))
        return result

    module.quantize_triton = kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(api, "resolve_backend", lambda backend, *tensors: "triton")
    x, amax, noise = torch.ones(3), torch.tensor(2.), torch.zeros(3, dtype=torch.int64)
    assert quantize(x, "bf16", amax=amax, noise=noise) is result
    assert len(calls) == 1
    assert calls[0][0] is x and calls[0][1] == SCHEMES["bf16"]
    assert calls[0][2] is amax and calls[0][3] is noise


@pytest.mark.parametrize("granularity", LAYOUTS)
@pytest.mark.parametrize("method", ["absmax", "pow2_floor", "pow2_ceil", "percentile", "mse"])
@pytest.mark.parametrize("two_level", [False, True])
@pytest.mark.parametrize("special", [False, True])
def test_vectorized_scales_match_domain_loop(
    granularity: str, method: str, two_level: bool, special: bool,
) -> None:
    from tricast.reference.quantize import (
        _absmax_scale,
        _domains,
        _global_scale,
        _layout,
        _mul,
        _pow2_scale,
    )

    gen = torch.Generator().manual_seed(42)
    x = torch.randn((5, 7), generator=gen) * 7
    if special:
        x[0] = 0
        x[1, 0], x[2, 1], x[3, 2] = float("nan"), float("inf"), 2.**-149
    fmt = UE4M3 if two_level else E8M0 if method.startswith("pow2") else FP16
    spec = spec_for(granularity, ScaleSpec(fmt, method, two_level=two_level,
                                         percentile=63.25, search=(1., .875, .5, 1.5)))
    maxima = x.new_empty(_layout(spec, *x.shape))
    for index, domain in _domains(spec, *x.shape):
        maxima[index] = x[domain].abs().amax()
    assert bit_equal(compute_amax(x, spec), maxima)
    global_scale = _global_scale(x.abs().amax(), spec) if two_level else None
    expected = torch.empty_like(maxima)
    for index, domain in _domains(spec, *x.shape):
        data, maximum = x[domain], maxima[index]
        if method.startswith("pow2") and not two_level:
            expected[index] = _pow2_scale(maximum, spec)
        elif method == "percentile":
            p = torch.quantile(data.abs().double().reshape(-1), spec.scale.percentile / 100).float()
            expected[index] = _absmax_scale(p, spec, global_scale)
        elif method == "mse":
            best_error = None
            for ratio in spec.scale.search:
                candidate = _absmax_scale(maximum * ratio, spec, global_scale)
                q = quantize_elements(data, spec, candidate, global_scale=global_scale)
                restored = _mul(q, candidate)
                if global_scale is not None:
                    restored = _mul(restored, global_scale)
                error = (data.double() - restored.double()).square().sum()
                if best_error is None or error < best_error:
                    expected[index], best_error = candidate, error
        else:
            expected[index] = _absmax_scale(maximum, spec, global_scale)
    actual, zero, actual_global = compute_scale(x, spec)
    assert bit_equal(actual, expected)
    assert zero is None
    assert actual_global is None if global_scale is None else bit_equal(actual_global, global_scale)


@pytest.mark.parametrize("granularity", LAYOUTS)
@pytest.mark.parametrize("zero_point", ["int", "float"])
@pytest.mark.parametrize("special", [False, True])
def test_vectorized_zero_points_match_domain_loop(
    granularity: str, zero_point: str, special: bool,
) -> None:
    from tricast.reference.quantize import _domains, _layout, _zero_point_scale

    gen = torch.Generator().manual_seed(42)
    x = torch.randn((5, 7), generator=gen) * 7
    if special:
        x[0] = 0
        x[1, 0], x[2, 1], x[3, 2] = float("nan"), float("inf"), 2.**-149
    spec = QuantSpec("uint4", granularity, **LAYOUTS[granularity], scale=ScaleSpec(FP16),
                     zero_point=zero_point, mma_input="dequant")
    expected_scale = x.new_empty(_layout(spec, *x.shape))
    expected_zero = torch.empty_like(expected_scale)
    for index, domain in _domains(spec, *x.shape):
        expected_scale[index], expected_zero[index] = _zero_point_scale(x[domain], spec)
    scale, zero, global_scale = compute_scale(x, spec)
    assert bit_equal(scale, expected_scale)
    assert bit_equal(zero, expected_zero)
    assert global_scale is None


@pytest.mark.parametrize("method,zero_point", [
    ("absmax", "none"), ("pow2_floor", "none"), ("pow2_ceil", "none"),
    ("percentile", "none"), ("mse", "none"), ("absmax", "int"), ("absmax", "float"),
])
def test_scale_selection_has_no_domain_iteration_or_scalar_extraction(
    method: str, zero_point: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tricast.reference.quantize as reference

    x = torch.arange(35, dtype=torch.float32).reshape(5, 7) - 17
    fmt = E8M0 if method.startswith("pow2") else FP16
    spec = QuantSpec("uint4" if zero_point != "none" else FP4_E2M1, "block", block=(2, 3),
                     scale=ScaleSpec(fmt, method, mse_grid=3), zero_point=zero_point,
                     mma_input="dequant")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scale selection must not iterate domains or extract tensor scalars")

    with monkeypatch.context() as patch:
        patch.setattr(reference, "_domains", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        maximum = compute_amax(x, spec)
        scale, _, _ = compute_scale(x, spec)
    assert maximum.shape == scale.shape == (3, 3)


@pytest.mark.parametrize("percentile", [1., 50., 63.25, 99.9, 100.])
def test_percentile_rank_selection_does_not_call_quantile(
    percentile: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tricast.reference.quantize import _absmax_scale, _domains, _layout

    x = torch.randn((5, 7), generator=torch.Generator().manual_seed(42))
    spec = QuantSpec(FP4_E2M1, "block", block=(2, 3),
                     scale=ScaleSpec(method="percentile", percentile=percentile))
    expected = x.new_empty(_layout(spec, *x.shape))
    for index, domain in _domains(spec, *x.shape):
        maximum = torch.quantile(x[domain].abs().double().reshape(-1), percentile / 100).float()
        expected[index] = _absmax_scale(maximum, spec, None)

    def unsupported_quantile(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch.quantile rejects domains larger than 2**24")

    monkeypatch.setattr(torch, "quantile", unsupported_quantile)
    scale, _, _ = compute_scale(x, spec)
    assert bit_equal(scale, expected)


@pytest.mark.parametrize("granularity", LAYOUTS)
def test_mse_scale_candidates_consume_explicit_element_noise(granularity: str) -> None:
    from tricast.reference.quantize import _absmax_scale, _domains, _layout, _mul

    gen = torch.Generator().manual_seed(42)
    x = torch.randn((5, 7), generator=gen) * 3
    noise = torch.randint(0, 2**32, x.shape, generator=gen)
    spec = spec_for(granularity, ScaleSpec(FP16, "mse", search=(1., .875, .5, 1.5)),
                    rounding=Rounding.SR)
    expected = x.new_empty(_layout(spec, *x.shape))
    for index, domain in _domains(spec, *x.shape):
        data, best_error = x[domain], None
        for ratio in spec.scale.search:
            candidate = _absmax_scale(data.abs().amax() * ratio, spec, None)
            q = quantize_elements(data, spec, candidate, noise=noise[domain])
            error = (data.double() - _mul(q, candidate).double()).square().sum()
            if best_error is None or error < best_error:
                expected[index], best_error = candidate, error
    state = torch.random.get_rng_state()
    first = quantize_reference(x, spec, noise=noise)
    second = quantize_reference(x, spec, noise=noise)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert bit_equal(first.scale, expected)
    assert bit_equal(first.scale, second.scale)
    assert bit_equal(first.values, second.values)


def test_explicit_scale_noise_controls_scale_rounding_without_global_rng() -> None:
    x = torch.tensor([[.15, 1.05, 6.003], [.15, 1.05, 6.003]])
    spec = QuantSpec(FP4_E2M1, "row", scale=ScaleSpec(FP16, rounding=Rounding.SR))
    noise = torch.tensor([[0], [2**32 - 1]])
    expected_scale = torch.tensor([[1. + 2.**-10], [1.]])
    expected_values = quantize_elements(x, spec, expected_scale)
    state = torch.random.get_rng_state()
    direct = quantize_reference(x, spec, scale_noise=noise)
    public = quantize(x, spec, backend="reference", scale_noise=noise)
    fake = fake_quant(x, spec, backend="reference", scale_noise=noise)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert bit_equal(direct.scale, expected_scale)
    assert bit_equal(direct.values, expected_values)
    assert bit_equal(public.scale, expected_scale)
    assert bit_equal(public.values, expected_values)
    assert bit_equal(fake, direct.dequantize())


@pytest.mark.parametrize("operation", [quantize_reference, quantize, fake_quant])
def test_scale_sr_requires_explicit_scale_noise(operation: object) -> None:
    spec = QuantSpec(FP4_E2M1, "row", scale=ScaleSpec(FP16, rounding=Rounding.SR))
    with pytest.raises(ValueError, match="scale_noise"):
        operation(torch.ones(2, 3), spec)


@pytest.mark.parametrize("operation", [quantize_reference, quantize, fake_quant])
@pytest.mark.parametrize("method", ["absmax", "mse"])
def test_element_sr_requires_explicit_noise(operation: object, method: str) -> None:
    spec = QuantSpec(FP4_E2M1, rounding=Rounding.SR, scale=ScaleSpec(method=method))
    with pytest.raises(ValueError, match="noise"):
        operation(torch.ones(2, 3), spec)


@pytest.mark.parametrize("method", ["pow2_floor", "pow2_ceil"])
@pytest.mark.parametrize("scale_format", [E8M0, FP16])
def test_pow2_scale_method_rejects_stochastic_rounding(method: str, scale_format: object) -> None:
    spec = QuantSpec(FP4_E2M1, scale=ScaleSpec(scale_format, method, rounding=Rounding.SR))
    with pytest.raises(ValueError, match="(?i)(stochastic|sr|power.of.two|pow2)"):
        quantize_reference(torch.ones(2, 3), spec, scale_noise=torch.tensor(0))


@pytest.mark.parametrize("method", ["percentile", "mse", "pow2_floor", "pow2_ceil"])
@pytest.mark.parametrize("zero_point", ["int", "float"])
def test_zero_point_warns_that_scale_method_is_unused(method: str, zero_point: str) -> None:
    with pytest.warns(UserWarning, match=r"\[min, max\]"):
        spec = QuantSpec("uint4", zero_point=zero_point, mma_input="dequant", scale=ScaleSpec(method=method))
    absmax = QuantSpec("uint4", zero_point=zero_point, mma_input="dequant")
    x = torch.tensor([[-3., 12., 0.5, 7.25]])
    got, want = quantize_reference(x, spec), quantize_reference(x, absmax)
    for name in ("values", "scale", "zero_point"):
        assert bit_equal(getattr(got, name), getattr(want, name)), name


@pytest.mark.parametrize("seed", [2, 3, 7])
@pytest.mark.parametrize("view", ["plain", "transpose", "strided"])
def test_zero_point_preserves_signed_zero_of_strided_domain_reduction(seed: int, view: str) -> None:
    from tricast.reference.quantize import _domains, _layout

    shape = (19, 68) if view == "strided" else (17, 33)
    signs = torch.randint(0, 2, shape, generator=torch.Generator().manual_seed(seed))
    x = (signs * -(2**31)).to(torch.int32).view(torch.float32)
    if view == "transpose":
        x = x.T
    elif view == "strided":
        # A nonzero storage offset and nonunit column stride must both survive.
        x = x[1:18, 1:67:2]
    spec = QuantSpec("uint4", "block", block=(16, 16), zero_point="float", mma_input="dequant")
    expected = x.new_empty(_layout(spec, *x.shape))
    for index, domain in _domains(spec, *x.shape):
        values = x[domain]  # ENGINE §3.5: -0 is ordered below +0
        minimum = values.amin()
        negative_zero = bool(((values == 0) & torch.signbit(values)).any())
        expected[index] = (-0.0 if negative_zero else 0.0) if minimum == 0 else minimum
    _, actual, _ = compute_scale(x, spec)
    # Numeric equality would hide the old padded-reduction regression at ragged edges.
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_zero_point_preserves_signed_zero_for_transposed_offset_rows() -> None:
    generator = torch.Generator().manual_seed(42)
    base = torch.randint(0, 2, (24, 30), generator=generator).float().mul_(-0.)
    base = torch.where(torch.rand(base.shape, generator=generator) < .5, base, -base)
    x = base[2:21, 3:26].T
    spec = QuantSpec("uint4", "row", zero_point="float", mma_input="dequant")
    has_negative_zero = ((x == 0) & torch.signbit(x)).any(dim=1)
    minimum = x.amin(dim=1)
    expected = torch.where(minimum == 0, torch.where(has_negative_zero, -0.0, 0.0), minimum).reshape(-1, 1)
    _, actual, _ = compute_scale(x, spec)
    # Shape (23, 19), stride (1, 30), offset 63 exposes a batched reduction tie change.
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
