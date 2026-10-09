"""Real-valued transform invariants use tolerances; seeded grids use exact equality."""

from __future__ import annotations

import pytest
import torch

from tricast.formats import IntFormat
from tricast.quant.spec import QuantSpec, TransformSpec
from tricast.transforms import (
    LinearTransform,
    StatsCollector,
    fit_transform,
    fit_transform_group,
    fwht,
    hadamard,
    random_sign_diag,
)


def _stats(x: torch.Tensor, *, samples: bool = True):
    collector = StatsCollector(x.shape[-1], want_xtx=True, want_samples=samples)
    collector.update(x)
    return collector.result()


@pytest.mark.parametrize("block", [1, 2, 4, 8, 32, 128])
def test_hadamard_is_orthogonal(block):
    h = hadamard(block)
    # Normalization by sqrt(block) is real arithmetic, not a format-grid operation.
    torch.testing.assert_close(h @ h.T, torch.eye(block, dtype=torch.float64), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("block", [1, 2, 8, 16])
def test_fwht_matches_block_matrix(block, gen):
    x = torch.randn(2, 3, 32, dtype=torch.float64, generator=gen)
    transform = LinearTransform("hadamard", block)
    expected = (x.reshape(2, 3, -1, block) @ hadamard(block)).reshape(x.shape)
    torch.testing.assert_close(transform.apply_activation(x), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(transform.apply_activation(transform.apply_activation(x)),
                               x, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fwht_is_fp32(dtype):
    # Integer butterfly sums are exact; block 4 normalization is exactly 1/2.
    x = torch.arange(24, dtype=dtype).reshape(3, 8)
    result = fwht(x, 4)
    expected = (x.float().reshape(3, 2, 4) @ hadamard(4).float()).reshape(x.shape)
    assert result.dtype == torch.float32
    assert torch.equal(result, expected)


@pytest.mark.parametrize("kind", ["none", "hadamard", "random_hadamard", "smoothquant", "awq"])
def test_transform_preserves_linear_function(kind, gen):
    x = torch.randn(9, 32, dtype=torch.float64, generator=gen)
    w = torch.randn(7, 32, dtype=torch.float64, generator=gen)
    transform = fit_transform(TransformSpec(kind=kind, block=8, grid=5), w, _stats(x))
    actual = transform.apply_activation(x) @ transform.apply_weight(w).T
    # T is a real-valued change of basis, so fp64 arithmetic roundoff is allowed.
    torch.testing.assert_close(actual, x @ w.T, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind", ["none", "hadamard", "random_hadamard", "smoothquant", "awq"])
def test_operational_transform_outputs_fp32(dtype, kind, gen):
    x = torch.randn(3, 8, generator=gen).to(dtype)
    w = torch.randn(2, 8, generator=gen).to(dtype)
    transform = fit_transform(TransformSpec(kind=kind, grid=3), w, _stats(x))
    assert transform.apply_activation(x).dtype == torch.float32
    assert transform.apply_weight(w).dtype == torch.float32


def test_random_hadamard_seed_and_order(gen):
    state = torch.random.get_rng_state()
    d = random_sign_diag(32, 7)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(d, random_sign_diag(32, 7))
    assert not torch.equal(d, random_sign_diag(32, 8))
    assert bool(((d == 1) | (d == -1)).all())
    w = torch.randn(3, 32, dtype=torch.float64, generator=gen)
    transform = fit_transform(TransformSpec(kind="random_hadamard", block=8, seed=7), w, None)
    expected = ((w * d).reshape(3, 4, 8) @ hadamard(8)).reshape(w.shape)
    torch.testing.assert_close(transform.apply_activation(w), expected, rtol=1e-10, atol=1e-10)
    assert torch.equal(transform.apply_activation(w), transform.apply_weight(w))


@pytest.mark.parametrize(("K", "expected"), [(1, 1), (7, 1), (12, 4), (96, 32), (128, 128), (512, 128)])
def test_automatic_block(K, expected):
    assert fit_transform(TransformSpec(kind="hadamard"), torch.ones(2, K), None).block == expected


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_smoothquant_formula(alpha):
    x = torch.tensor([[1.0, -2.0, 4.0], [-3.0, 1.0, 8.0]], dtype=torch.float64)
    w = torch.tensor([[2.0, -4.0, 1.0], [-1.0, 2.0, 0.5]], dtype=torch.float64)
    transform = fit_transform(TransformSpec(kind="smoothquant", alpha=alpha), w, _stats(x))
    expected = (x.abs().amax(0).pow(alpha) / w.abs().amax(0).pow(1 - alpha)).clamp(1e-5, 1e5)
    assert torch.equal(transform.diag, expected)
    assert torch.equal(transform.apply_activation(x), x / expected)
    assert torch.equal(transform.apply_weight(w), w * expected)


def test_smoothquant_zero_channels_are_invertible():
    x = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    w = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    transform = fit_transform(TransformSpec(kind="smoothquant"), w, _stats(x))
    assert torch.equal(transform.diag, torch.tensor([1.0, 1e5, 1e-5], dtype=torch.float64))


def test_stats_collector_statistics():
    x = torch.tensor([[1, -2, 0], [3, 4, -1], [-2, 0, 5], [0, 2, -3]], dtype=torch.float64)
    collector = StatsCollector(3, want_xtx=True, want_samples=True)
    collector.update(x[:1])
    collector.update(x[1:])
    stats = collector.result()
    assert stats.n_tokens == 4
    assert torch.equal(stats.absmax, x.abs().amax(0))
    assert torch.equal(stats.absmean, x.abs().mean(0))
    assert stats.xtx.dtype == torch.float64
    assert torch.equal(stats.xtx, x.T @ x)
    assert sorted(stats.samples.tolist()) == sorted(x.tolist())
    stats.absmax.zero_()
    assert torch.equal(collector.result().absmax, x.abs().amax(0))


def test_reservoir_is_bounded_uniform_seeded_and_batch_independent():
    x = torch.arange(10_000, dtype=torch.float32).reshape(-1, 1)
    whole = StatsCollector(1, want_xtx=False, want_samples=True)
    split = StatsCollector(1, want_xtx=False, want_samples=True)
    whole.update(x)
    for chunk in x.split(137):
        split.update(chunk)
    samples = whole.result().samples
    assert samples.shape == (4096, 1)
    assert torch.equal(samples, split.result().samples)
    priorities = torch.rand(10_000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    expected = x[priorities.topk(4096, largest=False).indices]
    assert torch.equal(samples, expected)
    assert samples.unique().numel() == 4096


def test_stats_optional_buffers_and_empty_update():
    collector = StatsCollector(4, want_xtx=False, want_samples=False)
    collector.update(torch.empty(0, 4))
    stats = collector.result()
    assert stats.n_tokens == 0
    assert torch.equal(stats.absmean, torch.zeros(4, dtype=torch.float64))
    assert stats.xtx is None
    assert stats.samples is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("quant_weight", "quant_activation"), [(True, False), (False, True), (True, True)])
def test_awq_selects_grid_minimum(quant_weight, quant_activation, dtype, gen):
    api = pytest.importorskip("tricast.quant.api")
    x = torch.randn(23, 16, dtype=dtype, generator=gen) * torch.linspace(0.1, 8, 16)
    w = torch.randn(7, 16, dtype=dtype, generator=gen)
    x[:, 0] = 0
    x[:, 1] *= 1e-8
    stats = _stats(x)
    spec = QuantSpec(IntFormat("int4", 4), granularity="row")
    weight_spec = spec if quant_weight else None
    act_spec = spec if quant_activation else None
    transform = fit_transform(TransformSpec(kind="awq", grid=9), w, stats,
                              weight_spec=weight_spec, act_spec=act_spec)
    target = w.double() @ stats.samples.double().T

    def error(t):
        qw, qx = t.apply_weight(w), t.apply_activation(stats.samples)
        if weight_spec is not None:
            qw = api.fake_quant(qw, weight_spec, backend="reference")
        if act_spec is not None:
            qx = api.fake_quant(qx, act_spec, backend="reference")
        return (qw.double() @ qx.double().T - target).square().sum()

    candidates = []
    for alpha in torch.linspace(0, 1, 9, dtype=torch.float64):
        s = stats.absmean.pow(float(alpha))
        s = torch.where(s == 0, 1e-5, s)
        s /= (s.max() * s.min()).sqrt()
        candidates.append(LinearTransform("awq", diag=s))
    errors = torch.stack([error(t) for t in candidates])
    assert torch.equal(transform.diag, candidates[int(errors.argmin())].diag)
    assert error(transform) == errors.min()
    assert error(transform) <= errors[0]


def test_awq_zero_channels_use_finite_scales():
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=torch.float64)
    w = torch.ones(2, 3, dtype=torch.float64)
    transform = fit_transform(TransformSpec(kind="awq", grid=3), w, _stats(x))
    assert bool(torch.isfinite(transform.diag).all())
    assert bool((transform.diag > 0).all())
    torch.testing.assert_close(transform.apply_activation(x) @ transform.apply_weight(w).T,
                               x @ w.T, rtol=1e-10, atol=1e-10)


def test_invalid_transform_inputs():
    with pytest.raises(ValueError, match="positive"):
        hadamard(0)
    with pytest.raises(ValueError, match="power of two"):
        fwht(torch.ones(2, 12), 3)
    with pytest.raises(ValueError, match="divisible"):
        fit_transform(TransformSpec(kind="hadamard", block=8), torch.ones(2, 12), None)
    with pytest.raises(ValueError, match="calibration"):
        fit_transform(TransformSpec(kind="smoothquant"), torch.ones(2, 12), None)
    with pytest.raises(ValueError, match="samples"):
        fit_transform(TransformSpec(kind="awq"), torch.ones(2, 3), _stats(torch.ones(2, 3), samples=False))
    with pytest.raises(ValueError, match="grid"):
        fit_transform(TransformSpec(kind="awq", grid=0), torch.ones(2, 3), _stats(torch.ones(2, 3)))
    with pytest.raises(ValueError, match="expected"):
        StatsCollector(3, False, False).update(torch.ones(2, 4))


def test_shared_smoothquant_pools_activation_and_weight_maxima():
    xs = [torch.tensor([[1.0, -8.0, 2.0]]), torch.tensor([[-4.0, 2.0, 1.0], [2.0, 1.0, 2.0]])]
    weights = [torch.tensor([[4.0, -1.0, 2.0]]), torch.tensor([[1.0, 4.0, -8.0], [2.0, -1.0, 2.0]])]
    spec = TransformSpec(kind="smoothquant", alpha=0.5)
    stats = [_stats(x) for x in xs]
    transform = fit_transform_group(spec, weights, stats)
    amax = torch.cat(xs).double().abs().amax(0)
    wmax = torch.cat(weights).double().abs().amax(0)
    expected = amax.sqrt() / wmax.sqrt()
    assert torch.equal(transform.diag, expected)
    for w, x in zip(weights, xs, strict=True):
        assert torch.equal(transform.apply_weight(w), w * expected.float())
        assert torch.equal(transform.apply_activation(x), x / expected.float())
    assert not torch.equal(transform.diag, fit_transform(spec, weights[0], stats[0]).diag)


@pytest.mark.parametrize("kind", ["smoothquant", "awq"])
def test_shared_singleton_is_exactly_the_independent_transform(kind, gen):
    weights = torch.randn(3, 5, generator=gen)
    stats = _stats(torch.randn(7, 5, generator=gen))
    spec = TransformSpec(kind=kind, grid=5)
    qspec = QuantSpec("int4", granularity="row")
    expected = fit_transform(spec, weights, stats, weight_spec=qspec)
    actual = fit_transform_group(spec, [weights], [stats], weight_specs=[qspec])
    assert torch.equal(actual.diag, expected.diag)


def test_shared_awq_minimizes_summed_signed_errors_with_member_specs(gen):
    api = pytest.importorskip("tricast.quant.api")
    x = torch.randn(7, 4, generator=gen) * torch.tensor([0.25, 1.0, 3.0, 10.0])
    weights = [torch.randn(2, 4, generator=gen), torch.randn(3, 4, generator=gen)]
    stats = [_stats(x), _stats(x)]
    wspecs = [QuantSpec("int2", rounding="rup"), QuantSpec("int4", granularity="row")]
    aspecs = [None, QuantSpec("int2", granularity="row", rounding="rdn")]
    spec = TransformSpec(kind="awq", grid=7)
    actual = fit_transform_group(spec, weights, stats, weight_specs=wspecs, act_specs=aspecs)
    repeated = fit_transform_group(spec, weights, stats, weight_specs=wspecs, act_specs=aspecs)
    assert torch.equal(actual.diag, repeated.diag)
    pooled = sum(s.absmean * s.n_tokens for s in stats) / sum(s.n_tokens for s in stats)
    candidates, errors, unsigned_errors = [], [], []
    for alpha in torch.linspace(0, 1, spec.grid, dtype=torch.float64):
        scale = pooled.pow(float(alpha))
        scale /= (scale.amax() * scale.amin()).sqrt()
        transform = LinearTransform("awq", diag=scale)
        candidates.append(transform)
        signed_loss, unsigned_loss = 0.0, 0.0
        for w, s, wspec, aspec in zip(weights, stats, wspecs, aspecs, strict=True):
            qw = api.fake_quant(transform.apply_weight(w), wspec, backend="reference")
            for samples, losses in ((s.samples, "signed"), (s.samples.abs(), "unsigned")):
                qx = transform.apply_activation(samples)
                if aspec is not None:
                    qx = api.fake_quant(qx, aspec, backend="reference")
                target = w.double() @ samples.double().T
                loss = float((qw.double() @ qx.double().T - target).square().sum())
                if losses == "signed":
                    signed_loss += loss
                else:
                    unsigned_loss += loss
        errors.append(signed_loss)
        unsigned_errors.append(unsigned_loss)
    best = min(range(spec.grid), key=errors.__getitem__)
    assert torch.equal(actual.diag, candidates[best].diag)
    assert best != min(range(spec.grid), key=unsigned_errors.__getitem__)
    independent = fit_transform(spec, weights[0], stats[0], weight_spec=wspecs[0])
    assert not torch.equal(actual.diag, independent.diag)


def test_shared_awq_pools_means_weighted_by_token_count():
    xs = [torch.tensor([[-1.0, 4.0]]), torch.tensor([[2.0, -1.0], [-6.0, 2.0], [3.0, -5.0]])]
    weights = [torch.ones(1, 2), torch.ones(2, 2)]
    stats = [_stats(x) for x in xs]
    spec = TransformSpec(kind="awq", grid=2)
    qspec = QuantSpec("int2", scale=None)
    actual = fit_transform_group(spec, weights, stats, weight_specs=[qspec, qspec])
    pooled = torch.cat(xs).double().abs().mean(0)
    choices = [torch.ones(2, dtype=torch.float64), pooled / (pooled.max() * pooled.min()).sqrt()]
    api = pytest.importorskip("tricast.quant.api")
    errors = []
    for scale in choices:
        transform = LinearTransform("awq", diag=scale)
        error = 0.0
        for w, x in zip(weights, xs, strict=True):
            qw = api.fake_quant(transform.apply_weight(w), qspec, backend="reference")
            qx = transform.apply_activation(x)
            error += float((qw.double() @ qx.double().T - w.double() @ x.double().T).square().sum())
        errors.append(error)
    assert torch.equal(actual.diag, choices[min(range(2), key=errors.__getitem__)])


def test_shared_awq_ties_choose_first_alpha():
    stats = _stats(torch.tensor([[-2.0, 8.0], [4.0, -1.0]]))
    transform = fit_transform_group(TransformSpec(kind="awq", grid=5), [torch.zeros(1, 2)] * 2,
                                    [stats, stats])
    assert torch.equal(transform.diag, torch.ones(2, dtype=torch.float64))


def test_shared_transform_rejects_mismatched_group_inputs():
    spec = TransformSpec(kind="smoothquant")
    weights = [torch.ones(1, 2)] * 2
    stats = [_stats(torch.ones(1, 2))] * 2
    with pytest.raises(ValueError, match="nonzero lengths"):
        fit_transform_group(spec, [], [])
    with pytest.raises(ValueError, match="nonzero lengths"):
        fit_transform_group(spec, weights, stats[:1])
    with pytest.raises(ValueError, match="group size"):
        fit_transform_group(spec, weights, stats, weight_specs=[None])
    with pytest.raises(ValueError, match="share K"):
        fit_transform_group(spec, [weights[0], torch.ones(2, 3)], stats)
    with pytest.raises(ValueError, match="match K"):
        fit_transform_group(spec, weights, [stats[0], _stats(torch.ones(1, 3))])
    with pytest.raises(ValueError, match="smoothquant or awq"):
        fit_transform_group(TransformSpec(kind="hadamard"), weights, stats)


@pytest.mark.parametrize("kind", ["smoothquant", "awq"])
@pytest.mark.parametrize("observer_kind", ["minmax", "ema", "history", "percentile", "mse"])
def test_diagonal_observers_match_row_replay_without_spooling(kind, observer_kind):
    import tempfile
    from unittest.mock import patch

    from torch import nn

    from tricast.nn import patch_model
    from tricast.quant.observer import ObserverState
    from tricast.recipe import load_recipe

    generator = torch.Generator().manual_seed(42)
    model = nn.Sequential(nn.Linear(5, 3, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.randn(3, 5, generator=generator))
    recipe = load_recipe({"name": "diagonal_observer", "backend": "reference", "defaults": {
        "transform": {"kind": kind, "grid": 3},
        "activation": {"scheme": "fp8_tensor", "observer": {
            "kind": observer_kind, "max_samples": 31, "percentile": 87, "history_len": 2,
        }},
    }})
    patch_model(model, recipe)
    layer = model[0]
    batches = [torch.randn(rows, 5, generator=generator) for rows in (17, 11, 23)]
    with patch.object(tempfile, "TemporaryFile", side_effect=AssertionError("row spool")):
        layer.begin_calibration()
        for batch in batches:
            layer(batch)
        summary = layer._diagonal_observer
        assert layer._calibration_inputs is None
        assert sum(maximum.numel() for maximum in summary.maxima) == len(batches) * 5
        assert summary.samples.numel() <= 31
        layer.prepare_calibration()
        expected = ObserverState(layer.spec.activation.observer)
        for batch in batches:
            expected.observe(layer.transform.apply_activation(batch))
        expected.freeze(layer.spec.activation)
        assert torch.equal(layer.observer.samples, expected.samples)
        assert torch.equal(layer.observer._priorities, expected._priorities)
        assert torch.equal(layer.observer._generator.get_state(), expected._generator.get_state())
        layer.finish_calibration()
    assert torch.equal(layer.observer.static_amax, expected.static_amax)
    assert torch.equal(layer.observer.amax, expected.amax)
    assert layer.observer.count == expected.count == len(batches)
    assert len(layer.observer.history) == len(expected.history)
    assert all(torch.equal(a, b) for a, b in zip(layer.observer.history, expected.history, strict=True))
    assert layer._diagonal_observer is None


def test_diagonal_observer_reservoir_is_bounded_and_retains_channel_identity():
    from tricast.nn.linear import _DiagonalObserverStats
    from tricast.quant.observer import ObserverState
    from tricast.quant.spec import ObserverSpec

    generator = torch.Generator().manual_seed(42)
    rows = torch.randn(1001, 9, generator=generator)
    diagonal = torch.linspace(0.13, 3.71, 9, dtype=torch.float64)
    transform = LinearTransform("smoothquant", diag=diagonal)
    spec = ObserverSpec("mse", max_samples=41)
    summary = _DiagonalObserverStats(spec)
    expected = ObserverState(spec)
    for batch in rows.split(73):
        summary.update(batch)
        expected.observe(transform.apply_activation(batch))
        assert summary.samples.numel() <= 41
        assert summary.channels.numel() == summary.samples.numel()
    actual = summary.restore(transform, rows.device)
    assert torch.equal(actual.samples, expected.samples)
    assert torch.equal(actual._priorities, expected._priorities)
    assert torch.equal(actual.amax, expected.amax)
