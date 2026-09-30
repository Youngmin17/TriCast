"""Streaming boundaries and KV values match quantize-based simulations exactly."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.conftest import bit_equal

pytest.importorskip("transformers")

from tricast.kv import TriCastKVCache
from tricast.quant.api import quantize
from tricast.quant.spec import KVSpec, QuantSpec, get_kv_spec, get_scheme


def _dequant(x: torch.Tensor, spec: QuantSpec, axis: str) -> torch.Tensor:
    if axis == "token" and spec.scale is not None and spec.granularity == "tensor":
        spec = spec.with_(granularity="row")
    view = x.transpose(2, 3) if axis == "channel" else x
    result = quantize(view.reshape(-1, view.shape[-1]), spec).mma_operand().values.reshape(view.shape)
    return result.transpose(2, 3) if axis == "channel" else result


def _step(
    chunks: list[torch.Tensor], residual: torch.Tensor, new: torch.Tensor,
    spec: QuantSpec | None, axis: str, keep: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    residual = torch.cat((residual, new), dim=2)
    visible = torch.cat([*chunks, residual], dim=2).to(new.dtype)
    if spec is None:
        return visible, chunks, residual
    move = residual.shape[2] // keep * keep if axis == "channel" else max(0, residual.shape[2] - keep)
    width = spec.group_size if axis == "channel" and spec.granularity == "group" else 1
    if move == 0:
        return visible, chunks, residual
    additions = [_dequant(chunk, spec, axis) for chunk in residual[:, :, :move].split(width, dim=2)]
    return visible, chunks + additions, residual[:, :, move:]


@pytest.mark.parametrize("axes", [("channel", "token"), ("token", "channel"), ("channel", "channel")])
@pytest.mark.parametrize("keep", [0, 1, 4, 8])
@pytest.mark.parametrize("sides", [(True, True), (True, False), (False, True)])
def test_stream_matches_quantize_simulation(
    axes: tuple[str, str], keep: int, sides: tuple[bool, bool],
) -> None:
    spec = replace(get_scheme("kivi2"), group_size=4)
    kwargs = dict(key=spec if sides[0] else None, value=spec if sides[1] else None,
                  key_axis=axes[0], value_axis=axes[1], residual=keep)
    if any(selected and axis == "channel" for selected, axis in zip(sides, axes, strict=True)) and (
        keep == 0 or keep % spec.group_size
    ):
        with pytest.raises(ValueError, match="residual"):
            KVSpec(**kwargs)
        return
    kv = KVSpec(**kwargs)
    cache = TriCastKVCache(kv, num_hidden_layers=2)
    gen = torch.Generator().manual_seed(42)
    chunks: list[list[torch.Tensor]] = [[], []]
    residuals = [torch.empty(2, 2, 0, 5), torch.empty(2, 2, 0, 5)]
    total = 0
    for tokens in (7, 1, 1, 4, 1):
        incoming = [torch.randn(2, 2, tokens, 5, generator=gen) for _ in range(2)]
        expected = []
        for i, side_spec in enumerate((kv.key, kv.value)):
            visible, chunks[i], residuals[i] = _step(
                chunks[i], residuals[i], incoming[i], side_spec, axes[i], keep,
            )
            expected.append(visible)
        returned = cache.update(*incoming, layer_idx=0)
        total += tokens
        assert cache.get_seq_length() == total
        assert cache.get_seq_length(1) == 0
        assert cache.get_mask_sizes(torch.arange(total, total + 2), 0) == (total + 2, 0)
        assert cache.get_max_cache_shape() == -1
        for i, side in enumerate(("keys", "values")):
            assert bit_equal(returned[i], expected[i])
            assert bit_equal(returned[i][..., -tokens:, :], incoming[i])
            stored = getattr(cache.layers[0], f"quantized_{side}")
            remainder = getattr(cache.layers[0], f"residual_{side}")
            assert bit_equal(remainder, residuals[i])
            if chunks[i]:
                assert bit_equal(stored, torch.cat(chunks[i], dim=2))
            else:
                assert stored is None
            expected_count = 0 if not sides[i] else (
                total // keep * keep if axes[i] == "channel" else max(0, total - keep)
            )
            assert (0 if stored is None else stored.shape[2]) == expected_count


@pytest.mark.parametrize("length", [2, 3, 4, 6, 7, 8, 11])
def test_prefill_group_boundaries(length: int) -> None:
    spec = replace(get_scheme("kivi4"), group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, residual=4))
    x = torch.randn(1, 1, length, 5, generator=torch.Generator().manual_seed(42))
    key, value = cache.update(x, x, 0)
    assert bit_equal(key, x) and bit_equal(value, x)
    count = length // 4 * 4
    layer = cache.layers[0]
    assert layer.residual_keys.shape[2] == length - count
    assert layer.residual_values.shape[2] == length
    if count:
        expected = torch.cat([_dequant(x[..., i:i + 4, :], spec, "channel")
                              for i in range(0, count, 4)], dim=2)
        assert bit_equal(layer.quantized_keys, expected)
    else:
        assert layer.quantized_keys is None


@pytest.mark.parametrize("name", ["kivi2", "kivi4", "kv_fp8"])
def test_unmodified_presets(name: str) -> None:
    kv = get_kv_spec(name)
    cache = TriCastKVCache(kv)
    x = torch.randn(1, 1, 161, 5, generator=torch.Generator().manual_seed(42))
    cache.update(x, x, 0)
    for side, spec, axis in (("keys", kv.key, kv.key_axis), ("values", kv.value, kv.value_axis)):
        _, chunks, residual = _step([], x[..., :0, :], x, spec, axis, kv.residual)
        assert bit_equal(getattr(cache.layers[0], f"quantized_{side}"), torch.cat(chunks, dim=2))
        assert bit_equal(getattr(cache.layers[0], f"residual_{side}"), residual)
    next_token = x[..., :1, :]
    for output in cache.update(next_token, next_token, 0):
        assert torch.isfinite(output).all()
        assert bit_equal(output[..., -1:, :], next_token)


@pytest.mark.parametrize("axis", ["token", "channel"])
@pytest.mark.parametrize("granularity", ["tensor", "row", "block"])
def test_non_group_domains_and_bf16_storage(axis: str, granularity: str) -> None:
    spec = QuantSpec("fp8_e4m3", granularity=granularity, block=(2, 3),
                     mma_input="dequant", dequant_format="fp16")
    if axis == "channel" or granularity == "block":
        # Dynamic channel domains need fixed token groups; token blocks cannot
        # share scales across two tokens (ENGINE §3.13 domain independence).
        with pytest.raises(ValueError, match="group|block"):
            KVSpec(key=spec, value=spec, key_axis=axis, value_axis=axis, residual=4)
        return
    cache = TriCastKVCache(KVSpec(key=spec, value=spec, key_axis=axis, value_axis=axis, residual=1))
    x = torch.randn(2, 2, 7, 5, generator=torch.Generator().manual_seed(42)).bfloat16()
    returned = cache.update(x, x, 0)
    expected = _dequant(x[..., :6, :], spec, axis)
    for tensor in returned:
        assert tensor.dtype == x.dtype and bit_equal(tensor, x)
    for side in ("keys", "values"):
        assert getattr(cache.layers[0], f"quantized_{side}").dtype == torch.float32
        assert bit_equal(getattr(cache.layers[0], f"quantized_{side}"), expected)
    y = torch.randn(2, 2, 1, 5, generator=torch.Generator().manual_seed(43)).bfloat16()
    expected_visible = torch.cat((expected, x[..., -1:, :], y), dim=2).to(x.dtype)
    assert bit_equal(cache.update(y, y, 0)[0], expected_visible)


def test_layer_selection_and_cache_protocol() -> None:
    kv = replace(get_kv_spec("kivi2"), key=replace(get_scheme("kivi2"), group_size=4), residual=4)
    cache = TriCastKVCache(kv, layers={1}, num_hidden_layers=2)
    gen = torch.Generator().manual_seed(42)
    key = torch.randn(2, 1, 9, 5, generator=gen)
    value = torch.randn(2, 1, 9, 5, generator=gen)
    cache.update(key, value, 0)
    cache.update(key, value, 1)
    assert cache.layers[0].quantized_keys is None
    assert cache.layers[0].quantized_values is None
    assert bit_equal(cache[0][0], key)
    assert bit_equal(cache[0][1], value)
    before = list(cache)
    cache.reorder_cache(torch.tensor([1, 0, 1]))
    for i in range(2):
        for side in range(2):
            assert bit_equal(cache[i][side], before[i][side][[1, 0, 1]])
    cache.batch_repeat_interleave(2)
    cache.batch_select_indices(torch.tensor([5, 0]))
    for i in range(2):
        for side in range(2):
            expected = before[i][side][[1, 0, 1]].repeat_interleave(2, dim=0)[[5, 0]]
            assert bit_equal(cache[i][side], expected)
    before = list(cache)
    for target in (-3, 2):
        # Even an initial prefill cannot restore a flushed token's original bits.
        with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
            cache.crop(target)
        assert cache.get_seq_length() == 9
        for i in range(2):
            for side in range(2):
                assert bit_equal(cache[i][side], before[i][side])
    cache.reset()
    assert cache.get_seq_length() == cache.get_seq_length(1) == 0
    assert cache.layers[0].keys is None and cache.layers[1].values is None
    assert bit_equal(cache.update(key[..., :1, :], value[..., :1, :], 1)[0], key[..., :1, :])


def test_empty_update_and_validation() -> None:
    kv = get_kv_spec("kv_fp8")
    cache = TriCastKVCache(kv)
    empty = torch.empty(1, 1, 0, 5)
    assert cache.update(empty, empty, 2)[0].shape == empty.shape
    assert cache.get_seq_length(2) == 0
    with pytest.raises(ValueError, match="nonnegative"):
        cache.update(empty, empty, -1)
    with pytest.raises(ValueError, match="KV states"):
        cache.update(torch.zeros(1, 2), torch.zeros(1, 2), 0)
    with pytest.raises(ValueError, match="matching"):
        cache.update(torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 1, 3), 0)
    with pytest.raises(ValueError, match="mode='cache'"):
        TriCastKVCache(replace(kv, mode="fakequant"))
    with pytest.raises(ValueError, match="positive"):
        TriCastKVCache(kv, num_hidden_layers=0)


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("axis", ["channel", "token"])
@pytest.mark.parametrize("target", [2, 4, -4, -2])
def test_crop_rejects_lost_residual_without_mutation(side: str, axis: str, target: int) -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(**{side: spec}, key_axis=axis, value_axis=axis, residual=4))
    initial = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    trial = torch.tensor([100.0, -100.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    cache.update(trial, trial, 0)
    before = tuple(t.clone() for t in cache[0])
    endpoint = target if target >= 0 else 6 + target
    if axis == "channel" and endpoint == 4:
        cache.crop(target)
        assert cache.get_seq_length() == endpoint
        assert all(bit_equal(actual, expected[..., :endpoint, :])
                   for actual, expected in zip(cache[0], before, strict=True))
        assert getattr(cache.layers[0], f"residual_{side}s").shape[-2] == 0
    else:
        with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
            cache.crop(target)
        assert cache.get_seq_length() == 6
        assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))


def test_crop_unflushed_candidates_restores_cache_exactly() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, value=spec, value_axis="channel", residual=4))
    initial = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, 1, 5, 1)
    cache.update(initial, initial, 0)
    before = tuple(t.clone() for t in cache[0])
    trial = torch.tensor([100.0, -100.0]).reshape(1, 1, 2, 1)
    cache.update(trial, trial, 0)
    cache.crop(-2)
    assert cache.get_seq_length() == 5
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))
    assert bit_equal(cache.layers[0].residual_keys, initial[..., -1:, :])
    assert bit_equal(cache.layers[0].residual_values, initial[..., -1:, :])


def test_crop_residual_tail_after_flush_matches_accepted_tokens() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    kv = KVSpec(key=spec, value=spec, value_axis="channel", residual=4)
    cache, accepted = TriCastKVCache(kv), TriCastKVCache(kv)
    initial = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    trial = torch.tensor([100.0, -100.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    for current in (cache, accepted):
        current.update(initial, initial, 0)
    cache.update(trial, trial, 0)
    accepted.update(trial[..., :3, :], trial[..., :3, :], 0)
    cache.crop(5)
    assert cache.get_seq_length() == accepted.get_seq_length() == 5
    for side in ("quantized_keys", "quantized_values", "residual_keys", "residual_values"):
        assert bit_equal(getattr(cache.layers[0], side), getattr(accepted.layers[0], side))


def test_crop_preflights_all_layers_before_mutating_any() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, residual=4), layers={1}, num_hidden_layers=2)
    initial = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    trial = torch.tensor([100.0, -100.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    for layer in range(2):
        cache.update(initial, initial, layer)
        cache.update(trial, trial, layer)
    before = [tuple(t.clone() for t in layer) for layer in cache]
    with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
        cache.crop(2)
    for layer in range(2):
        assert cache.get_seq_length(layer) == 6
        assert all(bit_equal(actual, expected)
                   for actual, expected in zip(cache[layer], before[layer], strict=True))


@pytest.mark.parametrize("clear", ["reset", "crop"])
def test_clearing_cache_discards_crop_rollback_guard(clear: str) -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, residual=4))
    initial = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    trial = torch.tensor([100.0, -100.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    cache.update(trial, trial, 0)
    if clear == "reset":
        cache.reset()
    else:
        cache.crop(0)
    assert cache.get_seq_length() == 0
    cache.update(initial, initial, 0)
    cache.update(trial[..., :1, :], trial[..., :1, :], 0)
    cache.crop(2)
    assert bit_equal(cache[0][0], initial)
    assert bit_equal(cache[0][1], initial)


def test_crop_keeps_flush_guard_across_nonflushing_updates() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, residual=4))
    initial = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    trial = torch.tensor([100.0, -100.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    cache.update(trial, trial, 0)
    before = tuple(t.clone() for t in cache[0])
    cache.update(initial[..., :1, :], initial[..., :1, :], 0)
    with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
        cache.crop(2)
    assert cache.get_seq_length() == 7
    cache.crop(6)
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))
    cache.crop(4)
    assert cache.get_seq_length() == 4
    assert all(bit_equal(actual, expected[..., :4, :])
               for actual, expected in zip(cache[0], before, strict=True))


def test_crop_checks_independent_key_and_value_flush_boundaries() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, value=spec, residual=4))
    initial = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    trial = torch.tensor([100.0, -100.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    cache.update(trial, trial, 0)
    assert cache.layers[0].quantized_keys.shape[-2] == 4
    assert cache.layers[0].quantized_values.shape[-2] == 2
    before = tuple(t.clone() for t in cache[0])
    with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
        cache.crop(5)
    assert cache.get_seq_length() == 6
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))


@pytest.mark.parametrize("axis", ["channel", "token"])
@pytest.mark.parametrize("sides", [(True, False), (False, True), (True, True)])
def test_crop_restores_independent_quantized_append(axis: str, sides: tuple[bool, bool]) -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(
        key=spec if sides[0] else None, value=spec if sides[1] else None,
        key_axis=axis, value_axis=axis, residual=4 if axis == "channel" else 0,
    ))
    initial = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    trial = torch.tensor([100.0, -100.0, 5.0, 6.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    before = tuple(t.clone() for t in cache[0])
    cache.update(trial, trial, 0)
    cache.crop(-4)
    assert cache.get_seq_length() == 4
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))
    if axis == "channel":
        with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
            cache.crop(2)
        assert all(bit_equal(actual, expected)
                   for actual, expected in zip(cache[0], before, strict=True))
    else:
        cache.crop(2)
        assert all(bit_equal(actual, expected[..., :2, :])
                   for actual, expected in zip(cache[0], before, strict=True))


def test_crop_safe_checkpoint_restores_earlier_flush_guard() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, residual=4))
    initial = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    trial = torch.tensor([100.0, -100.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    cache.update(trial[..., :2, :], trial[..., :2, :], 0)
    before = tuple(t.clone() for t in cache[0])
    cache.update(trial, trial, 0)
    cache.crop(4)
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))
    with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
        cache.crop(2)


def test_crop_checkpoint_requires_empty_residual_on_both_quantized_sides() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(
        key=spec, value=spec.with_(group_size=8), value_axis="token", residual=4,
    ))
    initial = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    trial = torch.tensor([100.0, -100.0, 5.0, 6.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    assert cache.layers[0].residual_keys.shape[-2] == 0
    assert cache.layers[0].residual_values.shape[-2] == 4
    cache.update(trial, trial, 0)
    before = tuple(t.clone() for t in cache[0])
    with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
        cache.crop(4)
    assert cache.get_seq_length() == 8
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))


def test_crop_checkpoint_survives_nonflushing_candidate_updates() -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, residual=4))
    initial = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    trial = torch.tensor([100.0, -100.0, 5.0, 6.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    before = tuple(t.clone() for t in cache[0])
    cache.update(trial[..., :1, :], trial[..., :1, :], 0)
    cache.update(trial[..., 1:, :], trial[..., 1:, :], 0)
    cache.crop(4)
    assert cache.get_seq_length() == 4
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))


@pytest.mark.parametrize("clear", ["reset", "crop"])
def test_clearing_cache_discards_exact_rollback_checkpoint(clear: str) -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    cache = TriCastKVCache(KVSpec(key=spec, residual=4))
    initial = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4, 1)
    trial = torch.tensor([100.0, -100.0, 5.0, 6.0]).reshape(1, 1, 4, 1)
    cache.update(initial, initial, 0)
    with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
        cache.crop(2)
    cache.update(trial, trial, 0)
    if clear == "reset":
        cache.reset()
    else:
        cache.crop(0)
    cache.update(initial[..., :1, :], initial[..., :1, :], 0)
    cache.update(trial[..., :3, :], trial[..., :3, :], 0)
    before = tuple(t.clone() for t in cache[0])
    with pytest.raises(NotImplementedError, match="speculative/assisted generation"):
        cache.crop(2)
    assert cache.get_seq_length() == 4
    assert all(bit_equal(actual, expected) for actual, expected in zip(cache[0], before, strict=True))


@pytest.mark.parametrize("domain", ["channel_group", "token_group", "token_tensor", "token_row",
                                    "token_block", "channel_static", "kv_fp8"])
@pytest.mark.parametrize("keep", [0, 2, 5, 4, 8])
def test_kv3_chunking_returns_and_stores_identical_bits(domain: str, keep: int) -> None:
    """I3: stored bits are chunk invariant; attention sees pre-write state."""
    if domain.startswith("channel") and (keep == 0 or (domain == "channel_group" and keep % 4)):
        spec = (QuantSpec("fp8_e4m3", scale=None, mma_input="dequant")
                if domain == "channel_static" else get_scheme("kivi2").with_(group_size=4))
        with pytest.raises(ValueError, match="residual"):
            KVSpec(key=spec, value=spec, key_axis="channel", value_axis="channel", residual=keep)
        return
    if domain == "kv_fp8":
        kv = replace(get_kv_spec("kv_fp8"), residual=keep)
    elif domain == "channel_static":
        spec = QuantSpec("fp8_e4m3", scale=None, mma_input="dequant")
        kv = KVSpec(key=spec, value=spec, key_axis="channel", value_axis="channel", residual=keep)
    else:
        axis, granularity = domain.split("_")
        spec = get_scheme("kivi2").with_(granularity=granularity, group_size=4, block=(1, 3))
        kv = KVSpec(key=spec, value=spec, key_axis=axis, value_axis=axis, residual=keep)
    generator = torch.Generator().manual_seed(42)
    key = torch.randn(2, 2, 19, 7, generator=generator)
    value = torch.randn(2, 2, 19, 7, generator=generator)
    # Distinct dynamic ranges expose accidental per-call, batch, or head scales.
    key *= torch.tensor([0.125, 8.0]).reshape(2, 1, 1, 1)
    value *= torch.tensor([0.5, 4.0]).reshape(1, 2, 1, 1)
    for chunks in ((19,), (1,) * 19, (3, 1, 5, 2, 7, 1)):
        cache = TriCastKVCache(kv)
        endpoint = 0
        for count in chunks:
            start, endpoint = endpoint, endpoint + count
            incoming = key[..., start:endpoint, :], value[..., start:endpoint, :]
            expected_visible = tuple(
                new if start == 0 else torch.cat((stored, new), dim=-2)
                for stored, new in zip(cache[0], incoming, strict=True)
            )
            returned = cache.update(*incoming, 0)
            for actual, reference in zip(returned, expected_visible, strict=True):
                assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
            prefill = TriCastKVCache(kv)
            raw_prefill = prefill.update(key[..., :endpoint, :], value[..., :endpoint, :], 0)
            for actual, raw in zip(raw_prefill, (key, value), strict=True):
                assert torch.equal(actual.view(torch.int32), raw[..., :endpoint, :].view(torch.int32))
            for side in ("quantized_keys", "quantized_values", "residual_keys", "residual_values"):
                actual = getattr(cache.layers[0], side)
                reference = getattr(prefill.layers[0], side)
                if actual is None or reference is None:
                    assert actual is reference
                else:
                    assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
            for side, axis in (("keys", kv.key_axis), ("values", kv.value_axis)):
                expected_count = endpoint // keep * keep if axis == "channel" else max(0, endpoint - keep)
                quantized = getattr(cache.layers[0], f"quantized_{side}")
                residual = getattr(cache.layers[0], f"residual_{side}")
                assert (0 if quantized is None else quantized.shape[-2]) == expected_count
                assert residual.shape[-2] == endpoint - expected_count


@pytest.mark.parametrize("domain", ["channel_group", "token_group", "token_tensor", "token_row",
                                    "token_block", "kv_fp8"])
def test_kv3_batch_rows_and_heads_have_independent_scales(domain: str) -> None:
    """I4: changing a different batch row or head cannot change this KV slice."""
    if domain == "kv_fp8":
        kv = get_kv_spec("kv_fp8")
    else:
        axis, granularity = domain.split("_")
        spec = get_scheme("kivi2").with_(granularity=granularity, group_size=4, block=(1, 3))
        kv = KVSpec(key=spec, value=spec, key_axis=axis, value_axis=axis,
                    residual=4 if axis == "channel" else 2)
    generator = torch.Generator().manual_seed(42)
    key = torch.randn(2, 2, 13, 7, generator=generator)
    value = torch.randn(2, 2, 13, 7, generator=generator)
    solo_cache, original_cache, changed_cache = (TriCastKVCache(kv) for _ in range(3))
    solo = solo_cache.update(key[:1, :1], value[:1, :1], 0)
    original = original_cache.update(key, value, 0)
    altered_key, altered_value = key.clone(), value.clone()
    for altered in (altered_key, altered_value):
        altered[1] = altered[1] * 1000 + 700
        altered[0, 1] = altered[0, 1] * 300 - 400
    changed = changed_cache.update(altered_key, altered_value, 0)
    for views in ((solo, original, changed), (solo_cache[0], original_cache[0], changed_cache[0])):
        for reference, batch, modified in zip(*views, strict=True):
            assert torch.equal(reference.view(torch.int32), batch[:1, :1].view(torch.int32))
            assert torch.equal(reference.view(torch.int32), modified[:1, :1].view(torch.int32))
    for cache, raw in ((solo_cache, key[:1, :1]), (original_cache, key), (changed_cache, altered_key)):
        assert not bit_equal(cache[0][0], raw)


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("granularity", ["tensor", "row", "block"])
def test_kv3_dynamic_channel_domains_without_fixed_groups_are_rejected(
    side: str, granularity: str,
) -> None:
    """I3: construction rejects domains whose statistics depend on update chunks."""
    spec = QuantSpec("fp8_e4m3", granularity=granularity, block=(1, 3), mma_input="dequant")
    with pytest.raises(ValueError, match="channel.*group|group.*channel"):
        KVSpec(**{side: spec}, key_axis="channel", value_axis="channel")


@pytest.mark.parametrize("side", ["key", "value"])
def test_kv3_token_blocks_cannot_span_tokens_or_heads(side: str) -> None:
    """I3/I4: a two-row token block would couple tokens or heads, so reject it."""
    spec = QuantSpec("fp8_e4m3", granularity="block", block=(2, 3), mma_input="dequant")
    with pytest.raises(ValueError, match="block"):
        KVSpec(**{side: spec}, key_axis="token", value_axis="token")


def test_kv3_fp8_preset_uses_static_unit_scale() -> None:
    """I5: default uncalibrated vLLM FP8 KV semantics are direct E4M3 casts."""
    kv = get_kv_spec("kv_fp8")
    assert kv.residual == 0
    for spec in (kv.key, kv.value):
        assert spec.scale is None
        assert spec.format.name == "fp8_e4m3"
        assert spec.zero_point == "none"
    # Explicit round-to-nearest-even ties and saturation; dynamic amax cannot
    # reproduce these values when the same vector also contains 500.0.
    values = torch.tensor([0.0, 1.0625, 1.1875, 1.3125, 448.0, 500.0, -1.0625])
    expected = torch.tensor([0.0, 1.0, 1.25, 1.25, 448.0, 448.0, -1.0]).reshape(1, 1, 1, 7)
    states = values.reshape(1, 1, 1, 7)
    cache = TriCastKVCache(kv)
    for actual in cache.update(states, states, 0):
        assert torch.equal(actual.view(torch.int32), states.view(torch.int32))
    for actual in cache[0]:
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    for actual in cache.update(states, states, 0):
        assert bit_equal(actual, torch.cat((expected, states), dim=-2))
    for side in ("keys", "values"):
        assert getattr(cache.layers[0], f"residual_{side}").shape[-2] == 0
        assert bit_equal(getattr(cache.layers[0], f"quantized_{side}"),
                         torch.cat((expected, expected), dim=-2))


@pytest.mark.parametrize("length", [2, 3, 5, 6, 9, 10, 18, 19])
def test_kv3_returned_values_apply_exact_postflush_boundaries(length: int) -> None:
    """I3: chunk invariance is not an identity fallback; flushed bits match quantize."""
    key_spec = get_scheme("kivi2").with_(group_size=4)
    value_spec = get_scheme("kivi2").with_(group_size=3)
    kv = KVSpec(key=key_spec, value=value_spec, residual=4)
    generator = torch.Generator().manual_seed(42)
    states = torch.randn(2, 2, length, 7, generator=generator)
    cache = TriCastKVCache(kv)
    returned = cache.update(states, states, 0)
    for actual, side, spec, axis, width in (
        (returned[0], "keys", key_spec, "channel", 4),
        (returned[1], "values", value_spec, "token", 1),
    ):
        count = length // kv.residual * kv.residual if axis == "channel" else max(0, length - kv.residual)
        quantized = getattr(cache.layers[0], f"quantized_{side}")
        residual = getattr(cache.layers[0], f"residual_{side}")
        assert residual.shape[-2] == length - count
        expected = states.clone()
        if count:
            expected_prefix = torch.cat([
                _dequant(states[..., start:start + width, :], spec, axis)
                for start in range(0, count, width)
            ], dim=2)
            assert torch.equal(quantized.view(torch.int32), expected_prefix.view(torch.int32))
            assert not torch.equal(expected_prefix, states[..., :count, :])
            expected[..., :count, :] = expected_prefix
        else:
            assert quantized is None
        assert torch.equal(actual.view(torch.int32), states.view(torch.int32))
        assert torch.equal(getattr(cache.layers[0], side).view(torch.int32), expected.view(torch.int32))
        assert torch.equal(residual.view(torch.int32), states[..., count:, :].view(torch.int32))


@pytest.mark.parametrize("side", ["key", "value"])
@pytest.mark.parametrize("residual", [0, 1, 2, 3, 5, 6, 7])
def test_channel_residual_must_be_positive_complete_groups(side: str, residual: int) -> None:
    spec = get_scheme("kivi2").with_(group_size=4)
    with pytest.raises(ValueError, match="residual"):
        KVSpec(**{side: spec}, key_axis="channel", value_axis="channel", residual=residual)


@pytest.mark.parametrize("side", ["key", "value"])
def test_static_channel_residual_cannot_be_zero(side: str) -> None:
    spec = QuantSpec("fp8_e4m3", scale=None, mma_input="dequant")
    with pytest.raises(ValueError, match="residual"):
        KVSpec(**{side: spec}, key_axis="channel", value_axis="channel", residual=0)


@pytest.mark.parametrize("residual", [4, 8, 12])
@pytest.mark.parametrize("length", [3, 4, 5, 7, 8, 9, 11, 12, 13, 23, 24, 25])
def test_kivi_buffer_and_value_tail_exact_boundaries(residual: int, length: int) -> None:
    """KIVI keys flush each full R buffer, not each surplus G group."""
    spec = get_scheme("kivi2").with_(group_size=4)
    kv = KVSpec(key=spec, value=spec, residual=residual)
    states = torch.randn(2, 2, length, 7, generator=torch.Generator().manual_seed(42))
    cache = TriCastKVCache(kv)
    cache.update(states, states, 0)
    for side, axis, count in (("keys", "channel", length // residual * residual),
                             ("values", "token", max(0, length - residual))):
        quantized = getattr(cache.layers[0], f"quantized_{side}")
        buffer = getattr(cache.layers[0], f"residual_{side}")
        assert bit_equal(buffer, states[..., count:, :])
        if count:
            assert bit_equal(quantized, _dequant(states[..., :count, :], spec, axis))
            assert not bit_equal(quantized, states[..., :count, :])
        else:
            assert quantized is None


@pytest.mark.parametrize("granularity", ["tensor", "block"])
@pytest.mark.parametrize("shape", [(1, 6), (3, 6), (2, 5), (2, 7)])
def test_channel_helper_validates_full_mask_before_head_isolation(
    granularity: str, shape: tuple[int, int],
) -> None:
    from tricast.kv.cache import quantize_states

    spec = QuantSpec("fp8_e4m3", granularity=granularity, block=(2, 3), mma_input="dequant")
    states = torch.ones(2, 2, 6, 4)
    with pytest.raises(ValueError, match=r"\[batch, tokens\]"):
        quantize_states(states, spec, "channel", torch.ones(shape, dtype=torch.bool))
