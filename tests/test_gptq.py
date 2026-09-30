"""GPTQ grids, lazy updates, and calibration-output reconstruction."""

from __future__ import annotations

import importlib

import pytest
import torch

from tests.conftest import bit_equal
from tricast.formats import container_dtype
from tricast.quant.spec import QuantSpec, ScaleSpec, WeightAlgoSpec, get_scheme
from tricast.reference.cast import decode, round_to_format
from tricast.rounding import Rounding
from tricast.weight_quant import quantize_weight

quant_api = pytest.importorskip("tricast.quant.api")
reference = pytest.importorskip("tricast.reference.quantize")
gptq_module = importlib.import_module("tricast.weight_quant.gptq")
gptq = gptq_module.gptq


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _same_qtensor(actual, expected):
    assert actual.shape == expected.shape
    assert actual.spec == expected.spec
    for name in ("values", "scale", "zero_point", "global_scale"):
        a, b = getattr(actual, name), getattr(expected, name)
        if a is None or b is None:
            assert a is b
        else:
            assert a.shape == b.shape
            assert a.dtype == b.dtype
            assert bit_equal(a, b), name


def _assert_grid(q):
    assert q.values.dtype == container_dtype(q.spec.format)
    assert torch.isfinite(q.dequantize()).all()
    sign, exponent, significand, radix = decode(q.values, q.spec.format)
    restored = torch.ldexp(significand.double(), exponent - radix)
    restored = torch.where(sign, -restored, restored)
    assert bit_equal(restored, q.values)
    assert bit_equal(round_to_format(q.values, q.spec.format), q.values)


@pytest.mark.parametrize("scheme", ["int4_g32", "mxfp4", "nvfp4", "fp8_row"])
def test_reconstruction_improves_over_rtn(scheme, gen):
    spec = (get_scheme("int4_g128").with_(group_size=32) if scheme == "int4_g32"
            else get_scheme(scheme))
    W = torch.randn(64, 256, generator=gen) / 16
    X = torch.randn(384, 256, generator=gen, dtype=torch.float64)
    X *= torch.logspace(-1, 1, 256, dtype=torch.float64)
    H = 2 * X.T @ X / X.shape[0]
    baseline = quantize_weight(W, spec, WeightAlgoSpec("rtn"))
    result = quantize_weight(W, spec, WeightAlgoSpec("gptq"), hessian=H)
    error = ((W.double() - result.dequantize().double()) @ X.T).square().sum()
    rtn_error = ((W.double() - baseline.dequantize().double()) @ X.T).square().sum()
    assert error < rtn_error, (scheme, error.item(), rtn_error.item())
    assert not bit_equal(result.values, baseline.values)
    _assert_grid(result)


def test_error_compensation_is_not_rtn():
    W = torch.tensor([[0.4, 0.4]])
    H = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float64)
    spec = QuantSpec("int4", scale=None)
    q = gptq(W, H, spec, damp=0)
    assert bit_equal(q.values, torch.tensor([[0.0, 1.0]]))
    assert bit_equal(quant_api.quantize(W, spec, backend="reference").values, torch.zeros_like(W))


@pytest.mark.parametrize("scheme", ["bf16", "int8_tensor", "fp8_row", "nvfp4", "int4_g128_zp"])
def test_diagonal_hessian_matches_rtn(scheme, gen):
    W = torch.randn(5, 37, generator=gen)
    spec = get_scheme(scheme)
    result = gptq(W, torch.eye(37), spec, block_size=11)
    _same_qtensor(result, quant_api.quantize(W, spec, backend="reference"))


@pytest.mark.parametrize("scheme", ["int8_tensor", "fp8_row", "mxfp4", "nvfp4", "int4_g128_zp"])
def test_act_order_static_scales_and_determinism(scheme, gen):
    W = torch.randn(4, 37, generator=gen)
    X = torch.randn(48, 37, generator=gen, dtype=torch.float64) * torch.arange(1, 38)
    H = 2 * X.T @ X / X.shape[0]
    spec = get_scheme(scheme)
    if spec.granularity == "group":
        spec = spec.with_(group_size=8)
    result = gptq(W, H, spec, act_order=True, block_size=7)
    repeated = gptq(W, H, spec, act_order=True, block_size=7)
    _same_qtensor(result, repeated)
    rtn = quant_api.quantize(W, spec, backend="reference")
    for name in ("scale", "zero_point", "global_scale"):
        a, b = getattr(result, name), getattr(rtn, name)
        assert (a is b) if a is None else bit_equal(a, b)
    _assert_grid(result)


def _oracle_group_scale(weights, spec, global_scale):
    if global_scale is None:
        s, z, _ = reference.compute_scale(weights, spec)
        return s, z
    # Scalar domain oracle uses the fixed original d2, never a slice's own amax_tensor.
    sf = spec.scale
    scales = torch.empty(weights.shape[0], 1)
    for row, data in enumerate(weights.float()):
        maximum = data.abs().max()
        if sf.method == "percentile":
            maximum = torch.quantile(data.abs().double(), sf.percentile / 100).float()
        ratios = (sf.search or torch.linspace(1, 0.5, sf.mse_grid).tolist()) if sf.method == "mse" else [1]
        candidates = []
        for ratio in ratios:
            amax = maximum * ratio
            raw = (amax / spec.format.max_normal) / global_scale
            s = round_to_format(raw, sf.format, sf.rounding)
            if s == 0:
                s = torch.tensor(getattr(sf.format, "min_subnormal", sf.format.min_normal))
            if amax == 0:
                s = torch.tensor(1.0)
            q = reference.quantize_elements(data[None, :], spec, s, global_scale=global_scale)
            restored = (q * s) * global_scale
            loss = (restored.double() - data.double()).square().sum()
            candidates.append((loss, s))
        scales[row, 0] = min(candidates, key=lambda candidate: candidate[0])[1]
    return scales, None


def _dense_columns(W, H, spec, damp, act_order):
    """Independent eager-column oracle; no batched updates or cached weight slices."""
    scale, zp, global_scale = reference.compute_scale(W, spec)
    work, hessian = W.double().clone(), H.double().clone()
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    work[:, dead] = 0
    order = (torch.argsort(hessian.diagonal(), descending=True, stable=True) if act_order
             else torch.arange(W.shape[1]))
    work, hessian = work[:, order], hessian[order][:, order]
    hessian.diagonal().add_(damp * hessian.diagonal().mean())
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)
    values = torch.empty_like(W)
    for j, original in enumerate(order.tolist()):
        if spec.granularity == "group":
            group = original // spec.group_size
            if not act_order and j % spec.group_size == 0:
                s, z = _oracle_group_scale(work[:, j:j + spec.group_size], spec, global_scale)
                scale[:, group:group + 1] = s
                if zp is not None:
                    zp[:, group:group + 1] = z
            s = scale[:, group:group + 1]
            z = None if zp is None else zp[:, group:group + 1]
        else:
            s, z = scale, zp
        q = reference.quantize_elements(work[:, j:j + 1], spec, s, z, global_scale)
        values[:, original:original + 1] = q
        restored = q.float()
        if z is not None and spec.zero_point != "float":
            restored = (restored.double() - z.double()).float()
        if s is not None:
            restored = (restored.double() * s.double()).float()
        if global_scale is not None:
            restored = (restored.double() * global_scale.double()).float()
        if z is not None and spec.zero_point == "float":
            restored = (restored.double() + z.double()).float()
        error = (work[:, j:j + 1] - restored.double()) / upper[j, j]
        work[:, j:] -= error * upper[j, j:]
    return values, scale, zp, global_scale


@pytest.mark.parametrize("block_size", [1, 7, 13, 128])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("scheme", ["int4_g128_zp", "nvfp4", "mxfp4"])
def test_lazy_batches_match_eager_column_updates(block_size, act_order, scheme, gen):
    W = torch.randn(3, 19, generator=gen)
    X = torch.randn(32, 19, generator=gen, dtype=torch.float64)
    H = 2 * X.T @ X / X.shape[0]
    spec = get_scheme(scheme).with_(group_size=5)
    expected = _dense_columns(W, H, spec, 0.01, act_order)
    result = gptq(W, H, spec, block_size=block_size, act_order=act_order)
    for name, b in zip(("values", "scale", "zero_point", "global_scale"), expected, strict=True):
        a = getattr(result, name)
        assert (a is b) if a is None else bit_equal(a, b), name


@pytest.mark.parametrize("two_level", [False, True])
@pytest.mark.parametrize("method", ["absmax", "percentile", "mse"])
def test_scale_search_and_float_zero_points(two_level, method, gen):
    spec = (get_scheme("nvfp4") if two_level else get_scheme("int4_g128_zp").with_(zero_point="float"))
    spec = spec.with_(group_size=5, scale=ScaleSpec(spec.scale.format, method, two_level=two_level,
                                                 mse_grid=3, percentile=90))
    W = torch.randn(3, 19, generator=gen)
    X = torch.randn(30, 19, generator=gen, dtype=torch.float64)
    H = X.T @ X
    expected = _dense_columns(W, H, spec, 0.01, False)
    result = gptq(W, H, spec, block_size=7)
    for name, b in zip(("values", "scale", "zero_point", "global_scale"), expected, strict=True):
        a = getattr(result, name)
        assert (a is b) if a is None else bit_equal(a, b), name
    assert torch.isfinite(result.dequantize()).all()


def test_damping_dead_columns_and_input_ownership(gen):
    W = torch.randn(4, 9, generator=gen, requires_grad=True)
    X = torch.randn(4, 9, generator=gen, dtype=torch.float64)
    X[:, 2] = 0
    H = X.T @ X
    before_w, before_h = W.detach().clone(), H.clone()
    spec = get_scheme("fp8_row")
    with pytest.raises(ValueError, match="positive definite"):
        gptq(W, H, spec, damp=0)
    result = gptq(W, H, spec, damp=0.1)
    _assert_grid(result)
    assert bit_equal(result.values[:, 2], torch.zeros(4))
    assert bit_equal(W, before_w)
    assert torch.equal(H, before_h)
    assert not result.values.requires_grad
    changed_damp = gptq(W, H, spec, damp=1.0)
    assert not bit_equal(result.values, changed_damp.values)


def test_all_dead_columns(gen):
    W = torch.randn(3, 9, generator=gen)
    result = gptq(W, torch.zeros(9, 9), get_scheme("nvfp4"), act_order=True)
    assert bit_equal(result.dequantize(), torch.zeros_like(W))


def test_two_level_global_scale_comes_from_original_weights(gen):
    W = torch.randn(3, 19, generator=gen)
    X = torch.randn(30, 19, generator=gen, dtype=torch.float64)
    spec = get_scheme("nvfp4").with_(group_size=5)
    result = gptq(W, X.T @ X, spec, block_size=7)
    baseline = quant_api.quantize(W, spec, backend="reference")
    assert bit_equal(result.global_scale, baseline.global_scale)


def test_stochastic_rounding_reproducible_with_caller_seed():
    spec = QuantSpec("int4", scale=None, rounding=Rounding.SR)
    W = torch.full((2, 12), 0.3)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        first = gptq(W, torch.eye(12), spec)
        torch.manual_seed(42)
        second = gptq(W, torch.eye(12), spec)
    _same_qtensor(first, second)
    _assert_grid(first)


@pytest.mark.parametrize("kwargs,match", [({"block_size": 0}, "block_size"),
                                        ({"damp": -1}, "damp"), ({"damp": float("nan")}, "damp")])
def test_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        gptq(torch.ones(2, 3), torch.eye(3), get_scheme("fp8_row"), **kwargs)


def test_invalid_shapes_and_specs():
    with pytest.raises(ValueError, match="W must"):
        gptq(torch.ones(3), torch.eye(3), get_scheme("fp8_row"))
    with pytest.raises(ValueError, match="H must have shape"):
        gptq(torch.ones(2, 3), torch.eye(2), get_scheme("fp8_row"))
    with pytest.raises(ValueError, match="2-D block"):
        gptq(torch.ones(2, 3), torch.eye(3), get_scheme("fp8_block128"))
    with pytest.raises(ValueError, match="finite"):
        gptq(torch.full((2, 3), float("inf")), torch.eye(3), get_scheme("fp8_row"))
    with pytest.raises(ValueError, match="symmetric"):
        gptq(torch.ones(2, 2), torch.tensor([[1.0, 1.0], [0.0, 1.0]]), get_scheme("fp8_row"))


def test_dispatch_requires_hessian_and_forwards_backend(monkeypatch, gen):
    W = torch.randn(3, 7, generator=gen)
    spec = get_scheme("fp8_row")
    with pytest.raises(ValueError, match="Hessian"):
        quantize_weight(W, spec, WeightAlgoSpec("gptq"))
    with pytest.raises(ValueError, match="backend"):
        quantize_weight(W, spec, WeightAlgoSpec(), backend="missing")
    calls = []
    quantize = quant_api.quantize

    def tracked(x, spec, *, backend):
        calls.append(backend)
        return quantize(x, spec, backend=backend)

    monkeypatch.setattr(quant_api, "quantize", tracked)
    result = quantize_weight(W, spec, WeightAlgoSpec(), backend="auto")
    assert calls == ["auto"]
    _same_qtensor(result, quantize(W, spec, backend="reference"))


@pytest.mark.parametrize("act_order", [False, True])
def test_float_zero_point_compensation_uses_post_scale_offset(act_order, gen):
    W = torch.randn(3, 7, generator=gen) + 1.25
    X = torch.randn(12, 7, dtype=torch.float64, generator=gen)
    H = X.T @ X
    spec = QuantSpec("uint2", granularity="group", group_size=3, zero_point="float",
                     mma_input="dequant")
    result = gptq(W, H, spec, block_size=2, act_order=act_order)
    expected = _dense_columns(W, H, spec, 0.01, act_order)
    for name, tensor in zip(("values", "scale", "zero_point", "global_scale"), expected, strict=True):
        actual = getattr(result, name)
        assert (actual is tensor) if actual is None else bit_equal(actual, tensor), name
    _assert_grid(result)


def test_float_zero_point_column_reconstruction():
    q = torch.tensor([[0.0, 1.0, 3.0]])
    scale = torch.tensor([[0.5]])
    offset = torch.tensor([[-2.0]])
    actual = gptq_module._dequantize_column(q, scale, offset, None, float_zero_point=True)
    assert bit_equal(actual, torch.tensor([[-2.0, -1.5, -0.5]]))
    assert not bit_equal(actual, (q - offset) * scale)


@pytest.mark.parametrize("kind", ["smoothquant", "awq"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_gptq_replays_exact_fp32_hessian_without_spooling(kind, dtype, monkeypatch):
    import tempfile

    from torch import nn

    import tricast.nn.linear as linear_module
    from tricast.mma.api import as_operand
    from tricast.nn import patch_model
    from tricast.recipe import load_recipe

    generator = torch.Generator().manual_seed(42)
    model = nn.Sequential(nn.Linear(5, 3, bias=False, dtype=dtype))
    with torch.no_grad():
        model[0].weight.copy_(torch.randn(3, 5, generator=generator))
    recipe = load_recipe({"name": "diagonal_gptq", "backend": "reference", "defaults": {
        "weight": {"format": "int4", "granularity": "row"}, "weight_algo": "gptq",
        "transform": {"kind": kind, "grid": 3},
    }})
    patch_model(model, recipe)
    layer = model[0]
    batches = [torch.randn(rows, 5, dtype=dtype, generator=generator) for rows in (17, 11, 23)]

    def no_spool(*args, **kwargs):
        raise AssertionError("original calibration rows must not be spooled")

    monkeypatch.setattr(tempfile, "TemporaryFile", no_spool)
    hessians = []
    original = linear_module.quantize_weight

    def capture(*args, **kwargs):
        hessians.append(kwargs["hessian"].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(linear_module, "quantize_weight", capture)
    layer.begin_calibration()
    for batch in batches:
        layer(batch)
    assert layer._hessian is None and layer._calibration_inputs is None
    with pytest.raises(RuntimeError, match="exact input replay"):
        layer.finish_calibration()
    expected = None
    for batch in batches:
        values = layer.transform.apply_activation(batch.float()).double()
        gram = values.T @ values
        expected = gram if expected is None else expected + gram
        layer(batch)
    expected *= 2.0 / sum(batch.shape[0] for batch in batches)
    expected_weight = as_operand(original(layer.transform.apply_weight(layer.weight),
        layer.spec.weight, layer.spec.weight_algo, hessian=expected, backend="reference"))
    layer.finish_calibration()
    assert len(hessians) == 1 and torch.equal(hessians[0], expected)
    assert torch.equal(layer._weight_operand.values.float(), expected_weight.values)
    assert torch.equal(layer._weight_operand.scale, expected_weight.scale)
    assert layer._hessian is None and layer._hessian_accumulator is None


def test_shared_hessian_accumulator_counts_each_input_once():
    from torch import nn

    from tricast.nn import patch_model
    from tricast.nn.linear import HessianAccumulator
    from tricast.recipe import load_recipe

    generator = torch.Generator().manual_seed(42)
    model = nn.ModuleList([nn.Linear(4, 3, bias=False) for _ in range(3)])
    recipe = load_recipe({"name": "shared_hessian", "backend": "reference", "defaults": {
        "weight": {"format": "int4", "granularity": "row"}, "weight_algo": "gptq",
    }})
    patch_model(model, recipe)
    accumulator = HessianAccumulator()
    for index, layer in enumerate(model):
        layer.begin_calibration(hessian_accumulator=accumulator, hessian_owner=index == 0)
    batches = [torch.randn(7, 4, generator=generator), torch.randn(11, 4, generator=generator)]
    for batch in batches:
        for layer in model:
            layer(batch)
    expected = sum(batch.double().T @ batch.double() for batch in batches)
    assert torch.equal(accumulator.gram, expected)
    assert len({layer._hessian.data_ptr() for layer in model}) == 1
    assert accumulator.gram.numel() * accumulator.gram.element_size() == 128
    for layer in model:
        layer.finish_calibration()
        assert layer._hessian_accumulator is None
