"""Hand-computed calibration and delayed-scaling contracts."""

from __future__ import annotations

import pytest
import torch

from tricast.quant.api import fake_quant, quantize
from tricast.quant.observer import ObserverState
from tricast.quant.spec import ObserverSpec, QuantSpec, ScaleSpec


def _tensor(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _quant_spec(**scale_kwargs) -> QuantSpec:
    return QuantSpec("int2", scale=ScaleSpec(**scale_kwargs))


def test_minmax_calibrates_dynamically_then_freezes_running_max() -> None:
    observer = ObserverState(ObserverSpec("minmax"))
    assert observer.mode == "calibrate"
    assert observer.static_amax is None
    for values, expected in [((-2, 1), 2), ((-8, 3), 8), ((-1, 4), 4)]:
        used = observer.observe(_tensor(*values))
        assert used.shape == torch.Size([])
        assert used.dtype == torch.float32
        assert used == expected
    assert observer.amax == 8
    assert observer.count == 3
    observer.freeze(_quant_spec())
    assert observer.mode == "frozen"
    assert observer.static_amax == 8
    assert observer.observe(_tensor(100)) == 8
    assert observer.amax == 8
    assert observer.count == 3


def test_ema_first_call_initializes_and_fp32_formula_is_exact() -> None:
    observer = ObserverState(ObserverSpec("ema", decay=0.75))
    observer.observe(_tensor(-8, 1))
    assert observer.amax == 8
    assert observer.observe(_tensor(4)) == 4
    assert observer.amax == 7
    assert observer.observe(_tensor(3)) == 3
    assert observer.amax == 6
    observer.freeze(_quant_spec())
    assert observer.static_amax == 6
    assert observer.observe(_tensor(100)) == 6


def test_ema_nonbinary_decay_uses_separate_fp32_operations() -> None:
    observer = ObserverState(ObserverSpec("ema", decay=0.99))
    previous, current = _tensor(1.234567)[0], _tensor(9.876543)[0]
    observer.observe(previous)
    observer.observe(current)
    assert observer.amax == previous * 0.99 + current * 0.01


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [("max", [2, 2, 8, 8, 4]), ("most_recent", [2, 2, 8, 4, 1])],
)
def test_history_is_delayed_bounded_and_keeps_updating_when_frozen(reduce, expected) -> None:
    observer = ObserverState(ObserverSpec("history", history_len=2, reduce=reduce))
    for index, value in enumerate([2, 8, 4, 1, 16]):
        assert observer.observe(_tensor(value)) == expected[index]
        assert len(observer.history) <= 2
        if index == 1:
            observer.freeze(_quant_spec())
            assert observer.mode == "frozen"
            assert observer.static_amax == 8
    assert observer.count == 5
    assert torch.equal(torch.stack(observer.history), _tensor(1, 16))


def test_history_length_one_uses_exactly_previous_call() -> None:
    observer = ObserverState(ObserverSpec("history", history_len=1))
    assert [observer.observe(_tensor(v)).item() for v in [2, 8, 1, 4]] == [2, 2, 8, 1]


def test_percentile_freezes_fp64_linear_interpolation_of_absolute_values() -> None:
    observer = ObserverState(ObserverSpec("percentile", percentile=25, max_samples=20))
    assert observer.observe(_tensor(-8, 0)) == 8
    assert observer.observe(_tensor(-2, 4)) == 4
    observer.freeze(_quant_spec())
    # Sorted |x| = [0, 2, 4, 8], fractional index 0.75 gives 1.5.
    assert observer.static_amax == 1.5
    assert observer.observe(_tensor(1000)) == 1.5
    assert torch.equal(observer.samples, _tensor(8, 0, 2, 4))


def test_percentile_interpolates_in_fp64_before_rounding_to_fp32() -> None:
    observer = ObserverState(ObserverSpec("percentile", percentile=33.3))
    x = _tensor(0.03125, 1.234567, 8.765432, 32)
    observer.observe(x)
    observer.freeze(_quant_spec())
    expected = torch.quantile(x.double(), 0.333).float()
    assert observer.static_amax == expected


def test_reservoir_is_bounded_seeded_and_uniform_over_the_entire_stream() -> None:
    spec = ObserverSpec("percentile", max_samples=8)
    x = torch.arange(1, 65, dtype=torch.float32)
    observer = ObserverState(spec)
    for chunk in x.split(7):
        observer.observe(chunk)
        assert observer.samples.numel() <= 8
    generator = torch.Generator().manual_seed(0)
    keys = torch.rand(x.numel(), dtype=torch.float64, generator=generator)
    expected = x[keys.topk(8, largest=False).indices]
    assert torch.equal(observer.samples, expected)
    whole = ObserverState(spec)
    whole.observe(x)
    assert torch.equal(observer.samples, whole.samples)
    assert observer.samples.min() < 32 < observer.samples.max()


def test_mse_freeze_chooses_clipped_amax_and_static_quantization_uses_it() -> None:
    quantize_reference = pytest.importorskip("tricast.reference.quantize").quantize_reference
    spec = _quant_spec(method="mse", search=(1.0, 0.5))
    observer = ObserverState(ObserverSpec("mse"))
    x = _tensor(-0.5, 1, -1, 1, 2)
    assert observer.observe(x) == 2
    observer.freeze(spec)
    # int2 has grid [-1, 0, 1]. Scale 2 gives SSE 3.25; scale 1 gives SSE 1.25.
    assert observer.static_amax == 1
    actual = quantize_reference(x, spec, amax=observer.observe(x)).dequantize()
    assert torch.equal(actual, _tensor(0, 1, -1, 1, 1))


def test_mse_default_grid_and_ties_keep_first_ratio() -> None:
    pytest.importorskip("tricast.reference.quantize")
    observer = ObserverState(ObserverSpec("mse"))
    observer.observe(_tensor(0.5, 1, 1, 1, 2))
    observer.freeze(_quant_spec(method="mse", mse_grid=2))
    assert observer.static_amax == 1
    ties = ObserverState(ObserverSpec("mse"))
    ties.observe(_tensor(1))
    ties.freeze(_quant_spec(method="mse", search=(0.5, 1.5)))
    # Both scales have SSE 0.25, so retain r = 0.5 rather than r = 1.5.
    assert ties.static_amax == 0.5


@pytest.mark.parametrize("kind", ["minmax", "ema", "history", "percentile", "mse"])
def test_state_dict_roundtrip_resumes_calibration_and_frozen_behavior(kind) -> None:
    if kind == "mse":
        pytest.importorskip("tricast.reference.quantize")
    spec = ObserverSpec(kind, max_samples=5, history_len=2, percentile=50)
    original = ObserverState(spec)
    original.observe(_tensor(-1, 2, 3, 4, 5, 6))
    state = original.state_dict()
    restored = ObserverState(spec)
    restored.load_state_dict(state)
    x = _tensor(7, 8, 9, -10)
    assert original.observe(x) == restored.observe(x)
    assert original.amax == restored.amax
    assert original.count == restored.count
    assert torch.equal(original.samples, restored.samples)
    assert torch.equal(original._priorities, restored._priorities)
    quant_spec = _quant_spec(method="mse", search=(1.0, 0.5)) if kind == "mse" else _quant_spec()
    original.freeze(quant_spec)
    restored.freeze(quant_spec)
    assert original.static_amax == restored.static_amax
    frozen = ObserverState(spec)
    frozen.load_state_dict(restored.state_dict())
    assert frozen.mode == "frozen"
    assert frozen.observe(_tensor(32)) == restored.observe(_tensor(32))
    assert original.amax.data_ptr() != state["amax"].data_ptr()
    if state["samples"].numel():
        saved = original.samples.clone()
        state["samples"].fill_(1000)
        assert torch.equal(original.samples, saved)


def test_observer_does_not_keep_autograd_graph() -> None:
    observer = ObserverState(ObserverSpec("percentile"))
    used = observer.observe(_tensor(-1, 2).requires_grad_())
    assert not used.requires_grad
    assert not observer.amax.requires_grad
    assert not observer.samples.requires_grad


@pytest.mark.parametrize("kind", ["minmax", "ema", "history", "percentile", "mse"])
def test_nan_propagates_to_frozen_amax(kind) -> None:
    if kind == "mse":
        pytest.importorskip("tricast.reference.quantize")
    observer = ObserverState(ObserverSpec(kind))
    assert torch.isnan(observer.observe(_tensor(1, float("nan"))))
    observer.freeze(_quant_spec())
    assert torch.isnan(observer.static_amax)


def test_empty_or_uncalibrated_observer_rejects_invalid_lifecycle() -> None:
    observer = ObserverState(ObserverSpec("minmax"))
    with pytest.raises(ValueError, match="empty"):
        observer.observe(torch.empty(0))
    with pytest.raises(RuntimeError, match="calibration"):
        observer.freeze(_quant_spec())
    observer.mode = "frozen"
    with pytest.raises(RuntimeError, match="freeze"):
        observer.observe(_tensor(1))


@pytest.mark.parametrize("quant_spec", [QuantSpec("bf16", scale=None), QuantSpec("int4", "row")])
def test_freeze_rejects_quant_specs_without_static_tensor_scales(quant_spec) -> None:
    observer = ObserverState(ObserverSpec("minmax"))
    observer.observe(_tensor(1))
    with pytest.raises(ValueError, match="per-tensor"):
        observer.freeze(quant_spec)


def test_mse_preserves_sample_sign_for_directed_rounding() -> None:
    spec = QuantSpec("int2", rounding="rup", scale=ScaleSpec(method="mse", search=(1.0, 0.5)))
    observer = ObserverState(ObserverSpec("mse"))
    x = _tensor(-2, -0.75, -0.75)
    observer.observe(x)
    assert torch.equal(observer.samples, x)
    errors = []
    for maximum in (2, 1):
        restored = quantize(x, spec, amax=torch.tensor(maximum)).dequantize()
        errors.append((x.double() - restored.double()).square().sum().item())
    # RUP maps each -0.75 to zero at both scales; clipping -2 adds 1 only at scale 1.
    assert errors == [1.125, 2.125]
    observer.freeze(spec)
    assert observer.static_amax == 2


@pytest.mark.parametrize("kind", ["mse", "percentile", "history"])
def test_clear_samples_releases_reservoir_and_preserves_frozen_statistics(kind: str) -> None:
    observer = ObserverState(ObserverSpec(kind))
    observer.observe(_tensor(-2, 1))
    observer.freeze(_quant_spec())
    maximum = observer.static_amax.clone()
    observer.clear_samples()
    assert observer.samples.numel() == observer._priorities.numel() == 0
    assert observer.static_amax == maximum
    assert observer.observe(_tensor(8)) == maximum
    if kind == "history":
        assert observer.observe(_tensor(1)) == 8


@pytest.mark.parametrize(
    ("fmt", "values", "expected", "grad"),
    [
        ("uint8", [-1, 1], [0, 1], [0, 1]),
        ("ue4m3", [-1, 0, 1, 448, 512], [0, 0, 1, 448, 448], [0, 1, 1, 1, 0]),
        ("int4:full:frac=2", [-2.25, -2, 1.75, 2], [-2, -2, 1.75, 1.75], [0, 1, 1, 0]),
    ],
)
def test_clipped_ste_uses_unsigned_and_fixed_point_bounds(fmt, values, expected, grad) -> None:
    x = _tensor(*values).requires_grad_()
    y = fake_quant(x, QuantSpec(fmt, scale=None))
    assert torch.equal(y, _tensor(*expected))
    y.sum().backward()
    # Include both representable endpoints; saturated inputs have zero gradients.
    assert torch.equal(x.grad, _tensor(*grad))


@pytest.mark.parametrize(
    ("granularity", "kwargs"),
    [("tensor", {}), ("row", {}), ("group", {"group_size": 2}), ("block", {"block": (1, 2)})],
)
def test_clipped_ste_shifts_integer_zero_point_after_scaling(granularity: str, kwargs: dict) -> None:
    spec = QuantSpec("uint2", granularity, **kwargs, scale=ScaleSpec("fp4", rounding="rtz"),
                     zero_point="int", mma_input="dequant", dequant_format="fp32")
    x = _tensor(-1, 3, -1, 3).repeat(2, 1).requires_grad_()
    y = fake_quant(x, spec)
    # Rounded scale = 1 and z = 1 give normalized codes [0, 4], clipped to [0, 3].
    assert torch.equal(y, _tensor(-1, 2, -1, 2).repeat(2, 1))
    y.sum().backward()
    assert torch.equal(x.grad, _tensor(1, 0, 1, 0).repeat(2, 1))


@pytest.mark.parametrize(
    ("granularity", "kwargs"),
    [("tensor", {}), ("row", {}), ("group", {"group_size": 3}), ("block", {"block": (1, 3)})],
)
def test_clipped_ste_subtracts_float_zero_point_before_scaling(granularity: str, kwargs: dict) -> None:
    spec = QuantSpec("uint2", granularity, **kwargs, scale=ScaleSpec("fp4", rounding="rtz"),
                     zero_point="float", mma_input="dequant", dequant_format="fp32")
    x = _tensor(5, 6, 9, 5, 6, 9).repeat(2, 1).requires_grad_()
    y = fake_quant(x, spec)
    # Rounded scale = 1 and offset = 5 give codes [0, 1, 4], clipped to [0, 1, 3].
    assert torch.equal(y, _tensor(5, 6, 8, 5, 6, 8).repeat(2, 1))
    y.sum().backward()
    assert torch.equal(x.grad, _tensor(1, 1, 0, 1, 1, 0).repeat(2, 1))
