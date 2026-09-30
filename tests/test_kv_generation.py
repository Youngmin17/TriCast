"""Config-only HF attention and generation coverage for KV emulation."""

from __future__ import annotations

from types import MethodType
from typing import Any

import pytest
import torch

from tests.conftest import bit_equal
from tricast.quant.api import quantize
from tricast.quant.spec import KVSpec, QuantSpec, get_scheme

transformers = pytest.importorskip("transformers")
kv_api = pytest.importorskip("tricast.kv")


def _model(
    family: str, *, dtype: torch.dtype = torch.float32, attention: str = "sdpa",
) -> torch.nn.Module:
    config_cls = transformers.LlamaConfig if family == "llama" else transformers.Qwen3Config
    model_cls = transformers.LlamaForCausalLM if family == "llama" else transformers.Qwen3ForCausalLM
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = attention
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return model_cls(config).to(dtype=dtype).eval()


def _simulate(states: torch.Tensor, spec: QuantSpec | None, axis: str) -> torch.Tensor:
    if spec is None:
        return states
    view = states.transpose(-1, -2) if axis == "channel" else states
    flat = view.reshape(-1, view.shape[-1])
    values = quantize(flat, spec, backend="auto").mma_operand().values
    out = values.reshape(view.shape).to(states.dtype)
    return out.transpose(-1, -2) if axis == "channel" else out


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize(
    ("key_axis", "value_axis", "use_key", "use_value"),
    [
        ("channel", "token", True, True),
        ("token", "channel", True, True),
        ("channel", "token", True, False),
        ("token", "channel", False, True),
    ],
)
def test_fakequant_attention_receives_axis_simulation(
    monkeypatch: pytest.MonkeyPatch, family: str, key_axis: str, value_axis: str,
    use_key: bool, use_value: bool,
) -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = _model(family)
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capture(
        module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attention_mask: torch.Tensor | None, **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        captured[module.layer_idx] = (key.clone(), value.clone())
        return original(module, query, key, value, attention_mask, **kwargs)

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", capture)
    ids = torch.tensor([[1, 3, 5, 7, 11]])
    with torch.inference_mode():
        baseline_logits = model(ids, use_cache=False).logits
    baseline = dict(captured)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv = KVSpec(
        key=spec if use_key else None,
        value=spec if use_value else None,
        key_axis=key_axis,
        value_axis=value_axis,
        mode="fakequant",
        residual=1000,
    )
    names = kv_api.apply_kv(model, kv, layers="1-1")
    assert names == ["model.layers.1.self_attn"]
    captured.clear()
    with torch.inference_mode():
        quantized_logits = model(ids, use_cache=False).logits
    assert set(captured) == {0, 1}
    assert bit_equal(captured[0][0], baseline[0][0])
    assert bit_equal(captured[0][1], baseline[0][1])
    # ENGINE 3.13 v2: no query has consumed R real tokens, so no stored token
    # has flushed; fakequant must retain both axes and disabled operands exactly.
    assert bit_equal(captured[1][0], baseline[1][0])
    assert bit_equal(captured[1][1], baseline[1][1])
    assert bit_equal(quantized_logits, baseline_logits)

    # The same selected layer/axes must become active after a valid R boundary.
    kv_api.apply_kv(model, KVSpec(
        key=kv.key, value=kv.value, key_axis=key_axis, value_axis=value_axis,
        mode="fakequant", residual=4,
    ), layers="1-1")
    with torch.inference_mode():
        flushed_logits = model(ids, use_cache=False).logits
    assert not bit_equal(flushed_logits, baseline_logits)
    assert bit_equal(captured[0][0], baseline[0][0])
    assert bit_equal(captured[0][1], baseline[0][1])

    kv_api.remove_kv(model)
    assert model.config._attn_implementation == "sdpa"
    with torch.inference_mode():
        restored_logits = model(ids, use_cache=False).logits
    assert bit_equal(restored_logits, baseline_logits)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cache_prefill_is_full_precision_and_decode_changes_logits(
    family: str, attention: str, dtype: torch.dtype,
) -> None:
    from transformers.cache_utils import DynamicCache

    model = _model(family, dtype=dtype, attention=attention)
    oracle = _model(family, dtype=dtype, attention=attention)
    prompt = torch.tensor([[1, 3, 5, 7, 11, 13]])
    next_token = torch.tensor([[17]])
    baseline_cache = DynamicCache()
    with torch.inference_mode():
        baseline_prefill = model(prompt, past_key_values=baseline_cache, use_cache=True).logits
        baseline_decode = model(next_token, past_key_values=baseline_cache, use_cache=True).logits

    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4))
    cache = kv_api.make_cache(model)
    assert isinstance(cache, kv_api.TriCastKVCache)
    with torch.inference_mode():
        prefill = model(prompt, past_key_values=cache, use_cache=True).logits
        # A native attention call over the stored prefix is the deployment oracle;
        # its appended token stays full precision even if this write causes a flush.
        stored_cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(cache):
            stored_cache.update(key.clone(), value.clone(), layer_idx)
        expected_decode = oracle(next_token, past_key_values=stored_cache, use_cache=True).logits
        decode = model(next_token, past_key_values=cache, use_cache=True).logits
    assert bit_equal(prefill, baseline_prefill)
    assert bit_equal(decode, expected_decode)
    assert not bit_equal(decode, baseline_decode)
    assert cache.get_seq_length() == prompt.shape[-1] + 1
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_generate_injects_cache_and_remove_restores_original(family: str) -> None:
    model = _model(family, dtype=torch.bfloat16)
    original_generate = model.generate
    prompt = torch.tensor([[1, 3, 5, 7, 11, 13]])
    kwargs = {
        "max_new_tokens": 3,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    with torch.inference_mode():
        baseline = model.generate(prompt, **kwargs)
    spec = get_scheme("kivi2").with_(group_size=4)
    names = kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4), layers="0-1")
    assert names == ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    with torch.inference_mode():
        generated = model.generate(prompt, **kwargs)
        repeated = model.generate(prompt, **kwargs)
    assert isinstance(generated.past_key_values, kv_api.TriCastKVCache)
    assert generated.sequences.shape == (1, prompt.shape[-1] + 3)
    assert torch.equal(generated.sequences[:, :prompt.shape[-1]], prompt)
    assert len(generated.scores) == 3
    assert all(torch.isfinite(score).all() for score in generated.scores)
    assert any(not bit_equal(a, b) for a, b in zip(generated.scores, baseline.scores, strict=True))
    assert torch.equal(generated.sequences, repeated.sequences)
    assert all(bit_equal(a, b) for a, b in zip(generated.scores, repeated.scores, strict=True))

    kv_api.remove_kv(model)
    assert model.generate == original_generate
    assert model.config._attn_implementation == "sdpa"
    with torch.inference_mode():
        restored = model.generate(prompt, **kwargs)
    assert not isinstance(restored.past_key_values, kv_api.TriCastKVCache)
    assert torch.equal(restored.sequences, baseline.sequences)
    assert all(bit_equal(a, b) for a, b in zip(restored.scores, baseline.scores, strict=True))


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_eager_fakequant_restores_attention_config_identity(family: str) -> None:
    model = _model(family, attention="eager")
    attention = model.model.layers[0].self_attn
    original_config = attention.config
    original_generate = model.generate
    ids = torch.tensor([[1, 3, 5, 7, 11]])
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, mode="fakequant", residual=4), layers="0")
    with torch.inference_mode():
        quantized = model(ids, use_cache=False).logits
    assert torch.isfinite(quantized).all()
    assert not bit_equal(quantized, baseline)
    kv_api.remove_kv(model)
    assert attention.config is original_config
    assert model.generate == original_generate
    with torch.inference_mode():
        restored = model(ids, use_cache=False).logits
    assert bit_equal(restored, baseline)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_identity_fakequant_preserves_padding_and_causal_masks(family: str, attention: str) -> None:
    model = _model(family, attention=attention)
    ids = torch.tensor([[0, 0, 1, 3, 5], [1, 3, 5, 7, 11]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    with torch.inference_mode():
        baseline = model(ids, attention_mask=mask, use_cache=False).logits
    identity = QuantSpec("fp32", scale=None, mma_input="dequant", dequant_format="fp32")
    kv_api.apply_kv(model, KVSpec(key=identity, value=identity, mode="fakequant"))
    assert model.config._attn_implementation == attention
    changed_ids = ids.clone()
    changed_ids[:, -1] = 17
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, use_cache=False).logits
        changed_suffix = model(changed_ids, attention_mask=mask, use_cache=False).logits
    assert bit_equal(actual, baseline)
    # Fully masked padding queries have no causal meaning in HF eager attention.
    real_queries = mask[:, :-1].bool()
    assert bit_equal(changed_suffix[:, :-1][real_queries], actual[:, :-1][real_queries])
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("override", ["use_cache", "generation_config", "explicit_cache"])
def test_generation_respects_explicit_cache_overrides(family: str, override: str) -> None:
    from transformers.cache_utils import DynamicCache

    model = _model(family)
    prompt = torch.tensor([[1, 3, 5, 7, 11, 13]])
    kwargs = {
        "max_new_tokens": 2,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if override == "use_cache":
        kwargs["use_cache"] = False
    elif override == "generation_config":
        config = transformers.GenerationConfig.from_model_config(model.config)
        config.use_cache = False
        kwargs["generation_config"] = config
    else:
        kwargs["past_key_values"] = DynamicCache()
    with torch.inference_mode():
        baseline = model.generate(prompt, **kwargs)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4))
    explicit_cache = DynamicCache() if override == "explicit_cache" else None
    if explicit_cache is not None:
        kwargs["past_key_values"] = explicit_cache
    with torch.inference_mode():
        actual = model.generate(prompt, **kwargs)
    assert actual.past_key_values is explicit_cache
    assert torch.equal(actual.sequences, baseline.sequences)
    assert all(bit_equal(a, b) for a, b in zip(actual.scores, baseline.scores, strict=True))
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_reapply_fakequant_to_cache_restores_original_configs(family: str) -> None:
    model = _model(family)
    configs = [layer.self_attn.config for layer in model.model.layers]
    original_prepare = model._prepare_cache_for_generation
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, mode="fakequant"))
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4), layers="1")
    assert all(layer.self_attn.config is config
               for layer, config in zip(model.model.layers, configs, strict=True))
    assert isinstance(kv_api.make_cache(model), kv_api.TriCastKVCache)
    kv_api.remove_kv(model)
    assert model._prepare_cache_for_generation == original_prepare
    assert all(layer.self_attn.config is config
               for layer, config in zip(model.model.layers, configs, strict=True))
    with pytest.raises(ValueError, match="apply_kv"):
        kv_api.make_cache(model)
    kv_api.remove_kv(model)


@pytest.mark.parametrize("layers", ["", "1-0", "2"])
def test_invalid_layer_selector_leaves_model_unchanged(layers: str) -> None:
    model = _model("llama")
    config = model.model.layers[0].self_attn.config
    prepare = model._prepare_cache_for_generation
    with pytest.raises(ValueError):
        kv_api.apply_kv(model, KVSpec(key="kivi2", mode="fakequant"), layers=layers)
    assert model.model.layers[0].self_attn.config is config
    assert model._prepare_cache_for_generation == prepare


def test_unsupported_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="no supported"):
        kv_api.apply_kv(torch.nn.Linear(2, 2), KVSpec(key="kivi2"))


def test_make_cache_requires_cache_mode() -> None:
    model = _model("llama")
    kv_api.apply_kv(model, KVSpec(key="kivi2", mode="fakequant"))
    with pytest.raises(ValueError, match="cache"):
        kv_api.make_cache(model)
    kv_api.remove_kv(model)


def test_generate_rejects_conflicting_cache_implementation() -> None:
    model = _model("llama")
    kv_api.apply_kv(model, KVSpec(key="kivi2"))
    with pytest.raises(ValueError, match="cache_implementation"), torch.inference_mode():
        model.generate(torch.tensor([[1, 3, 5]]), max_new_tokens=1, cache_implementation="static")
    kv_api.remove_kv(model)


def test_remove_restores_preexisting_instance_generation_hook() -> None:
    model = _model("llama")
    original = model._prepare_cache_for_generation
    calls: list[bool] = []

    def prepare(_self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        return original(*args, **kwargs)

    hook = MethodType(prepare, model)
    model._prepare_cache_for_generation = hook
    kv_api.apply_kv(model, KVSpec(key="kivi2"))
    prompt = torch.tensor([[1, 3, 5]])
    with torch.inference_mode():
        patched = model.generate(prompt, max_new_tokens=1, return_dict_in_generate=True)
    assert isinstance(patched.past_key_values, kv_api.TriCastKVCache)
    kv_api.remove_kv(model)
    assert model._prepare_cache_for_generation is hook
    with torch.inference_mode():
        restored = model.generate(prompt, max_new_tokens=1, return_dict_in_generate=True)
    assert not isinstance(restored.past_key_values, kv_api.TriCastKVCache)
    assert calls == [True, True]


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("padding", ["left", "right", "mixed"])
@pytest.mark.parametrize("operands", ["key", "value", "both"])
def test_fakequant_padding_contents_do_not_change_real_token_logits(
    family: str, attention: str, padding: str, operands: str,
) -> None:
    model = _model(family, attention=attention)
    masks = {
        "left": [[0, 0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1]],
        "right": [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]],
        "mixed": [[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]],
    }
    mask = torch.tensor(masks[padding])
    ids = torch.tensor([[1, 3, 5, 7, 11, 13, 17, 19], [1, 5, 7, 11, 13, 17, 19, 23]])
    ids = ids.masked_fill(~mask.bool(), 0)
    changed = ids.masked_fill(~mask.bool(), 31)
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    kwargs = {"attention_mask": mask, "position_ids": positions, "use_cache": False}
    with torch.inference_mode():
        baseline = model(ids, **kwargs).logits
        baseline_changed = model(changed, **kwargs).logits
    assert bit_equal(baseline[mask.bool()], baseline_changed[mask.bool()])
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(
        key=spec if operands != "value" else None,
        value=spec if operands != "key" else None,
        mode="fakequant", residual=4,
    ))
    with torch.inference_mode():
        actual = model(ids, **kwargs).logits
        actual_changed = model(changed, **kwargs).logits
    assert torch.isfinite(actual).all() and torch.isfinite(actual_changed).all()
    assert bit_equal(actual[mask.bool()], actual_changed[mask.bool()])
    assert not bit_equal(actual[mask.bool()], baseline[mask.bool()])
    kv_api.remove_kv(model)
    with torch.inference_mode():
        restored = model(ids, **kwargs).logits
    assert bit_equal(restored, baseline)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("positional_mask", [False, True])
def test_cache_padding_is_rejected_before_mutating_existing_cache(
    family: str, positional_mask: bool,
) -> None:
    model = _model(family)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4))
    cache = kv_api.make_cache(model)
    with torch.inference_mode():
        model(torch.tensor([[1, 3, 5, 7, 11]]), past_key_values=cache, use_cache=True)
    before = [(key.clone(), value.clone()) for key, value in cache]
    prompt = torch.tensor([[0, 13]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 1]])
    with pytest.raises(NotImplementedError, match="padding"), torch.inference_mode():
        if positional_mask:
            model(prompt, mask, past_key_values=cache, use_cache=True)
        else:
            model(prompt, attention_mask=mask, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 5
    for (key, value), (old_key, old_value) in zip(cache, before, strict=True):
        assert bit_equal(key, old_key) and bit_equal(value, old_value)
    forward_calls: list[bool] = []
    handle = model.register_forward_pre_hook(lambda *_args: forward_calls.append(True))
    with pytest.raises(NotImplementedError, match="padding"), torch.inference_mode():
        model.generate(torch.tensor([[0, 1, 3]]), attention_mask=torch.tensor([[0, 1, 1]]),
                       max_new_tokens=1, do_sample=False)
    assert forward_calls == []
    handle.remove()
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("override", ["use_cache", "explicit_cache"])
def test_cache_padding_rejection_respects_generation_overrides(
    family: str, override: str,
) -> None:
    from transformers.cache_utils import DynamicCache

    model = _model(family)
    prompt = torch.tensor([[0, 0, 1, 3, 5]])
    kwargs = {
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1]]),
        "max_new_tokens": 2,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if override == "use_cache":
        kwargs["use_cache"] = False
    else:
        kwargs["past_key_values"] = DynamicCache()
    with torch.inference_mode():
        baseline = model.generate(prompt, **kwargs)
    if override == "explicit_cache":
        kwargs["past_key_values"] = DynamicCache()
    kv_api.apply_kv(model, KVSpec(key=get_scheme("kivi2").with_(group_size=4)))
    with torch.inference_mode():
        actual = model.generate(prompt, **kwargs)
    assert torch.equal(actual.sequences, baseline.sequences)
    assert all(bit_equal(a, b) for a, b in zip(actual.scores, baseline.scores, strict=True))
    assert not isinstance(actual.past_key_values, kv_api.TriCastKVCache)
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("mode", ["fakequant", "cache"])
def test_unpadded_mask_is_bit_exact_and_mask_hooks_are_removed(family: str, mode: str) -> None:
    model = _model(family)
    ids = torch.tensor([[1, 3, 5, 7, 11, 13]])
    mask = torch.ones_like(ids)
    original_hooks = dict(model._forward_pre_hooks)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv = KVSpec(key=spec, value=spec, mode=mode, residual=4)
    for _ in range(2):
        kv_api.apply_kv(model, kv)
        first_cache = kv_api.make_cache(model) if mode == "cache" else None
        second_cache = kv_api.make_cache(model) if mode == "cache" else None
        with torch.inference_mode():
            no_mask = model(ids, past_key_values=first_cache, use_cache=mode == "cache").logits
            with_mask = model(ids, attention_mask=mask, past_key_values=second_cache,
                              use_cache=mode == "cache").logits
        assert bit_equal(no_mask, with_mask)
        if mode == "cache":
            for (key, value), (other_key, other_value) in zip(
                first_cache, second_cache, strict=True,
            ):
                assert bit_equal(key, other_key) and bit_equal(value, other_value)
        kv_api.remove_kv(model)
        assert dict(model._forward_pre_hooks) == original_hooks
    with torch.inference_mode():
        restored = model(ids, attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]),
                         use_cache=False).logits
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("axis", ["channel", "token"])
@pytest.mark.parametrize("granularity", ["tensor", "row", "group", "block"])
def test_masked_kv_quantization_excludes_padding_without_regrouping(
    axis: str, granularity: str,
) -> None:
    from tricast.kv.cache import quantize_states

    states = torch.arange(1, 97, dtype=torch.float32).reshape(2, 2, 6, 4) / 7
    mask = torch.tensor([[False, False, True, True, True, True],
                         [True, True, True, True, False, False]])
    valid = mask[:, None, :, None].expand_as(states)
    states = states.masked_fill(~valid, 1000)
    changed = states.masked_fill(~valid, -1000)
    block = (3, 4) if axis == "channel" else (1, 4)
    spec = get_scheme("kivi2").with_(granularity=granularity, group_size=4, block=block)
    expected = states.clone()
    # ENGINE 3.13 v2: real-token compression precedes grouping; every domain
    # is confined to one batch/head pair, and token-axis scales are per token.
    for batch in range(states.shape[0]):
        for head in range(states.shape[1]):
            compact = states[batch, head, mask[batch]]
            view = compact.T if axis == "channel" else compact
            reference_spec = (spec.with_(granularity="row")
                              if axis == "token" and granularity == "tensor" else spec)
            result = quantize(view, reference_spec, backend="reference").mma_operand().values
            expected[batch, head, mask[batch]] = result.T if axis == "channel" else result
    actual = quantize_states(states, spec, axis, token_mask=mask)
    altered = quantize_states(changed, spec, axis, token_mask=mask)
    assert bit_equal(actual, expected)
    assert bit_equal(actual[valid], altered[valid])
    assert bit_equal(actual[~valid], states[~valid])
    assert bit_equal(altered[~valid], changed[~valid])
    assert not bit_equal(actual[valid], states[valid])
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("axis", ["channel", "token"])
def test_masked_kv_direct_cast_keeps_elementwise_behavior(axis: str) -> None:
    from tricast.kv.cache import quantize_states

    states = torch.linspace(-1, 1, 24).reshape(1, 1, 6, 4)
    mask = torch.tensor([[False, False, True, True, True, True]])
    spec = QuantSpec("bf16", scale=None, mma_input="dequant", dequant_format="fp32")
    expected = quantize_states(states, spec, axis)
    assert bit_equal(quantize_states(states, spec, axis, token_mask=mask), expected)
    assert bit_equal(quantize_states(states, spec, axis, token_mask=torch.ones_like(mask)),
                     expected)


def test_masked_kv_two_level_scaling_fails_explicitly() -> None:
    from tricast.kv.cache import quantize_states
    from tricast.quant.spec import ScaleSpec

    states = torch.arange(1, 25, dtype=torch.float32).reshape(1, 1, 6, 4)
    mask = torch.tensor([[False, False, True, True, True, True]])
    spec = QuantSpec("fp4_e2m1", granularity="group", group_size=4,
                     scale=ScaleSpec(format="ue4m3", two_level=True), mma_input="dequant")
    with pytest.raises(NotImplementedError, match="two.level"):
        quantize_states(states, spec, "channel", token_mask=mask)
    assert bit_equal(quantize_states(states, spec, "channel"),
                     quantize_states(states, spec, "channel", token_mask=torch.ones_like(mask)))


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("case", ["disabled_cache", "positional_cache", "custom_mask"])
def test_explicit_tricast_cache_rejects_masks_even_when_cache_flag_is_disabled(
    family: str, case: str,
) -> None:
    model = _model(family)
    kv_api.apply_kv(model, KVSpec(key=get_scheme("kivi2").with_(group_size=4)))
    cache = kv_api.make_cache(model)
    prompt = torch.tensor([[0, 1, 3, 5]])
    mask = torch.tensor([[0, 1, 1, 1]])
    if case == "custom_mask":
        mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="padding"), torch.inference_mode():
        if case == "positional_cache":
            model(prompt, mask, None, cache, use_cache=False)
        else:
            model(prompt, attention_mask=mask, past_key_values=cache, use_cache=False)
    assert all(layer.get_seq_length() == 0 for layer in cache.layers)
    assert all(layer.keys is None and layer.values is None for layer in cache.layers)
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_fakequant_rejects_unsupported_additive_attention_bias(
    family: str, attention: str,
) -> None:
    model = _model(family, attention=attention)
    kv_api.apply_kv(model, KVSpec(key=get_scheme("kivi2").with_(group_size=4),
                                mode="fakequant"))
    prompt = torch.tensor([[1, 3, 5, 7]])
    bias = torch.full((1, 1, 4, 4), -1.0).triu(diagonal=1)
    with pytest.raises(NotImplementedError, match="4D mask"), torch.inference_mode():
        model(prompt, attention_mask=bias, use_cache=False)
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("assisted", ["assistant_model", "prompt_lookup", "early_exit"])
@pytest.mark.parametrize("explicit_cache", [False, True])
def test_assisted_generation_rejects_tricast_cache_before_forward(
    monkeypatch: pytest.MonkeyPatch, family: str, assisted: str, explicit_cache: bool,
) -> None:
    from tricast.kv import attention as kv_attention

    model = _model(family)
    kv_api.apply_kv(model, KVSpec(key=get_scheme("kivi2").with_(group_size=4), residual=4))
    caches: list[Any] = []
    original_make_cache = kv_attention.make_cache

    def capture_cache(module: torch.nn.Module) -> Any:
        cache = original_make_cache(module)
        caches.append(cache)
        return cache

    monkeypatch.setattr(kv_attention, "make_cache", capture_cache)
    kwargs: dict[str, Any] = {"max_new_tokens": 3, "do_sample": False}
    before: list[tuple[torch.Tensor, torch.Tensor]] = []
    if explicit_cache:
        cache = capture_cache(model)
        states = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8)
        for layer_idx in range(len(cache.layers)):
            cache.update(states, states, layer_idx)
        before = [(key.clone(), value.clone()) for key, value in cache]
        kwargs["past_key_values"] = cache
    if assisted == "assistant_model":
        kwargs["assistant_model"] = _model(family)
    elif assisted == "prompt_lookup":
        kwargs["prompt_lookup_num_tokens"] = 2
    else:
        kwargs["assistant_early_exit"] = 1
    forward_calls: list[bool] = []
    handle = model.register_forward_pre_hook(lambda *_args: forward_calls.append(True))
    with pytest.raises(NotImplementedError, match="speculative/assisted"), torch.inference_mode():
        model.generate(torch.tensor([[1, 3, 1, 3, 1, 3]]), **kwargs)
    assert forward_calls == []
    assert len(caches) == 1
    assert all(layer.get_seq_length() == (2 if explicit_cache else 0) for layer in caches[0].layers)
    if explicit_cache:
        for (key, value), (old_key, old_value) in zip(caches[0], before, strict=True):
            assert bit_equal(key, old_key) and bit_equal(value, old_value)
    handle.remove()
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_prompt_lookup_with_explicit_dynamic_cache_is_unchanged(family: str) -> None:
    from transformers.cache_utils import DynamicCache

    model = _model(family)
    prompt = torch.tensor([[1, 3, 1, 3, 1, 3]])
    kwargs = {"max_new_tokens": 3, "do_sample": False, "prompt_lookup_num_tokens": 2,
              "eos_token_id": 31, "return_dict_in_generate": True, "output_scores": True}
    with torch.inference_mode():
        baseline = model.generate(prompt, past_key_values=DynamicCache(), **kwargs)
    kv_api.apply_kv(model, KVSpec(key=get_scheme("kivi2").with_(group_size=4), residual=4))
    explicit_cache = DynamicCache()
    with torch.inference_mode():
        actual = model.generate(prompt, past_key_values=explicit_cache, **kwargs)
    assert actual.past_key_values is explicit_cache
    assert torch.equal(actual.sequences, baseline.sequences)
    assert all(bit_equal(a, b) for a, b in zip(actual.scores, baseline.scores, strict=True))
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_fakequant_preserves_padding_when_sdpa_relaxes_fully_masked_queries(
    monkeypatch: pytest.MonkeyPatch, family: str,
) -> None:
    from importlib import import_module

    model = _model(family, attention="sdpa")
    source = import_module(type(model.model).__module__)
    original_mask = source.create_causal_mask
    relaxed_rows: list[int] = []

    def relax_mask(*args: Any, **kwargs: Any) -> torch.Tensor:
        mask = original_mask(*args, **kwargs).clone()
        blocked = ~mask if mask.dtype == torch.bool else mask != 0
        fully_masked = blocked.all(dim=-1, keepdim=True)
        relaxed_rows.append(int(fully_masked.sum()))
        return mask.masked_fill(fully_masked, True if mask.dtype == torch.bool else 0)

    monkeypatch.setattr(source, "create_causal_mask", relax_mask)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, mode="fakequant"))
    ids = torch.tensor([[0, 0, 1, 3, 5, 7]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    changed = ids.masked_fill(~mask.bool(), 31)
    with torch.inference_mode():
        baseline = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        actual = model(changed, attention_mask=mask, position_ids=positions, use_cache=False).logits
    assert relaxed_rows == [2, 2]
    assert torch.isfinite(baseline).all() and torch.isfinite(actual).all()
    assert bit_equal(actual[mask.bool()], baseline[mask.bool()])
    assert model._tricast_kv_patch.token_mask is None
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_fakequant_mask_state_clears_on_success_error_and_remove(family: str) -> None:
    model = _model(family)
    original_pre_hooks = dict(model._forward_pre_hooks)
    original_hooks = dict(model._forward_hooks)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, mode="fakequant"))
    patch = model._tricast_kv_patch
    ids = torch.tensor([[0, 0, 1, 3, 5, 7]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    observed_masks: list[torch.Tensor | None] = []
    handle = model.model.register_forward_pre_hook(
        lambda *_args: observed_masks.append(patch.token_mask),
    )
    with torch.inference_mode():
        model(ids, attention_mask=mask, use_cache=False)
    assert observed_masks[-1] is mask
    assert patch.token_mask is None
    with pytest.raises(ValueError, match="exactly one"), torch.inference_mode():
        model(ids, attention_mask=mask, inputs_embeds=torch.zeros((1, 6, 16)), use_cache=False)
    assert observed_masks[-1] is mask
    assert patch.token_mask is None
    with torch.inference_mode():
        model(ids, use_cache=False)
    assert observed_masks[-1] is None
    handle.remove()
    kv_api.remove_kv(model)
    assert dict(model._forward_pre_hooks) == original_pre_hooks
    assert dict(model._forward_hooks) == original_hooks


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("length", [3, 6])
def test_fakequant_static_cache_excludes_padding_and_unfilled_capacity(
    family: str, attention: str, length: int,
) -> None:
    from transformers.cache_utils import StaticCache

    from tricast.kv.attention import _token_mask

    model = _model(family, attention=attention)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, mode="fakequant"))
    ids = torch.tensor([[0, 0, 1, 3, 5, 7]])[:, :length]
    mask = torch.ones_like(ids)
    mask[:, :2] = 0
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    extended_mask = _token_mask(mask, torch.empty((1, 1, 8, 8)))
    assert torch.equal(extended_mask, torch.cat((mask.bool(), torch.zeros((1, 8 - length),
                                                                         dtype=torch.bool)), dim=-1))
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for prompt in (ids, ids.masked_fill(~mask.bool(), 31)):
        cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=8,
                            device="cpu", dtype=torch.float32)
        with torch.inference_mode():
            prefill = model(prompt, attention_mask=mask, position_ids=positions,
                            cache_position=torch.arange(length), past_key_values=cache,
                            use_cache=True).logits
            decode = model(torch.tensor([[11]]),
                           attention_mask=torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype)),
                                                    dim=-1),
                           position_ids=mask.sum(-1, keepdim=True),
                           cache_position=torch.tensor([length]), past_key_values=cache,
                           use_cache=True).logits
        assert torch.isfinite(prefill).all() and torch.isfinite(decode).all()
        outputs.append((prefill, decode))
    assert bit_equal(outputs[0][0][mask.bool()], outputs[1][0][mask.bool()])
    assert bit_equal(outputs[0][1], outputs[1][1])
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("mode", ["fakequant", "cache"])
@pytest.mark.parametrize(("group_size", "length"), [(4, 25), (32, 97)])
def test_kv3_quantized_logits_are_bitwise_causal(
    family: str, attention: str, mode: str, group_size: int, length: int,
) -> None:
    model = _model(family, attention=attention)
    ids = torch.randint(1, 32, (1, length), generator=torch.Generator().manual_seed(42))
    changed = ids.clone()
    # Change the tail of the final scale group after several group/residual boundaries.
    cutoff = length - 2
    changed[:, cutoff:] = changed[:, cutoff:] % 31 + 1
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
    spec = get_scheme("kivi2").with_(group_size=group_size)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=group_size, mode=mode))
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for tokens in (ids, changed):
            cache = kv_api.make_cache(model) if mode == "cache" else None
            if mode == "cache":
                outputs.append(torch.cat([
                    model(tokens[:, position:position + 1], past_key_values=cache,
                          use_cache=True).logits for position in range(tokens.shape[-1])
                ], dim=1))
            else:
                outputs.append(model(tokens, use_cache=False).logits)
    assert all(torch.isfinite(output).all() for output in outputs)
    assert torch.equal(outputs[0][:, :cutoff].contiguous().view(torch.int32),
                       outputs[1][:, :cutoff].contiguous().view(torch.int32))
    # An identity/no-op path would be causal too; actual quantization must affect logits.
    assert not torch.equal(outputs[0], baseline)
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize(("group_size", "residual", "length"), [(4, 4, 25), (32, 32, 97), (4, 8, 41)])
def test_kv3_full_fakequant_matches_token_teacher_forcing(
    family: str, attention: str, group_size: int, residual: int, length: int,
) -> None:
    from dataclasses import replace

    model = _model(family, attention=attention)
    ids = torch.randint(1, 32, (1, length), generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
    spec = get_scheme("kivi2").with_(group_size=group_size)
    kv = KVSpec(key=spec, value=spec, residual=residual, mode="fakequant")
    kv_api.apply_kv(model, kv)
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
    kv_api.apply_kv(model, replace(kv, mode="cache"))
    cache = kv_api.make_cache(model)
    with torch.inference_mode():
        streamed = torch.cat([
            model(ids[:, position:position + 1], past_key_values=cache, use_cache=True).logits
            for position in range(length)
        ], dim=1)
    assert cache.get_seq_length() == length
    assert torch.isfinite(full).all() and torch.isfinite(streamed).all()
    # Batched QK/PV and token-at-a-time attention accumulate in different orders;
    # this absolute fp32 bound permits only that arithmetic noise, not different KV values.
    assert (full - streamed).abs().max().item() <= 1e-5
    assert (full - baseline).abs().max().item() > 1e-5
    assert (streamed - baseline).abs().max().item() > 1e-5
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("mode", ["fakequant", "cache"])
@pytest.mark.parametrize("preset", ["kivi2", "kv_fp8"])
def test_kv3_other_batch_rows_do_not_change_logits(
    family: str, attention: str, mode: str, preset: str,
) -> None:
    from dataclasses import replace

    from tricast.quant.spec import get_kv_spec

    model = _model(family, attention=attention)
    ids = torch.randint(1, 32, (2, 25), generator=torch.Generator().manual_seed(42))
    changed = ids.clone()
    changed[1] = changed[1] % 31 + 1
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
    kv = get_kv_spec(preset)
    if preset == "kivi2":
        spec = get_scheme("kivi2").with_(group_size=4)
        kv = KVSpec(key=spec, value=spec, residual=4)
    kv_api.apply_kv(model, replace(kv, mode=mode))
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for tokens in (ids, changed):
            cache = kv_api.make_cache(model) if mode == "cache" else None
            if mode == "cache":
                outputs.append(torch.cat([
                    model(tokens[:, position:position + 1], past_key_values=cache,
                          use_cache=True).logits for position in range(tokens.shape[-1])
                ], dim=1))
            else:
                outputs.append(model(tokens, use_cache=False).logits)
    assert torch.isfinite(outputs[0]).all() and torch.isfinite(outputs[1]).all()
    assert torch.equal(outputs[0][0].contiguous().view(torch.int32),
                       outputs[1][0].contiguous().view(torch.int32))
    assert not torch.equal(outputs[0], baseline)
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("padding", ["left", "right", "unmasked_right"])
def test_kv3_fakequant_padding_matches_unpadded_stream(
    family: str, attention: str, padding: str,
) -> None:
    from dataclasses import replace

    model = _model(family, attention=attention)
    ids = torch.randint(1, 32, (1, 21), generator=torch.Generator().manual_seed(42))
    spec = get_scheme("kivi2").with_(group_size=4)
    kv = KVSpec(key=spec, value=spec, residual=4, mode="cache")
    kv_api.apply_kv(model, kv)
    cache = kv_api.make_cache(model)
    with torch.inference_mode():
        streamed = torch.cat([
            model(ids[:, position:position + 1], past_key_values=cache, use_cache=True).logits
            for position in range(ids.shape[-1])
        ], dim=1)
    # Three pads deliberately misalign absolute positions with the four-token groups.
    pads = torch.zeros((1, 3), dtype=ids.dtype)
    padded = torch.cat((pads, ids) if padding == "left" else (ids, pads), dim=-1)
    other = torch.randint(1, 32, padded.shape, generator=torch.Generator().manual_seed(43))
    batch = torch.cat((padded, other), dim=0)
    mask = torch.ones_like(batch)
    if padding == "left":
        mask[0, :3] = 0
    else:
        mask[0, -3:] = 0
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    kv_api.apply_kv(model, replace(kv, mode="fakequant"))
    with torch.inference_mode():
        actual = model(batch, attention_mask=None if padding == "unmasked_right" else mask,
                       position_ids=positions, use_cache=False).logits
    real = actual[0, mask[0].bool()]
    assert torch.isfinite(actual).all()
    # As in the full-window/streaming test, only fp32 attention reduction order differs.
    assert (real - streamed[0]).abs().max().item() <= 1e-5
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("configuration", ["kwargs", "generation_config"])
@pytest.mark.parametrize("explicit_cache", [False, True])
def test_kv3_contrastive_search_rejected_before_first_forward(
    family: str, configuration: str, explicit_cache: bool,
) -> None:
    model = _model(family)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4))
    kwargs: dict[str, Any] = {"max_new_tokens": 2, "do_sample": False}
    if configuration == "generation_config":
        config = transformers.GenerationConfig.from_model_config(model.config)
        config.penalty_alpha = 0.6
        config.top_k = 3
        kwargs["generation_config"] = config
    else:
        kwargs.update(penalty_alpha=0.6, top_k=3)
    cache = kv_api.make_cache(model) if explicit_cache else None
    if cache is not None:
        kwargs["past_key_values"] = cache
    forward_calls: list[bool] = []
    handle = model.register_forward_pre_hook(lambda *_args: forward_calls.append(True))
    try:
        with pytest.raises(NotImplementedError, match="contrastive"), torch.inference_mode():
            model.generate(torch.tensor([[1, 3, 5, 7, 11, 13]]), **kwargs)
        assert forward_calls == []
        if cache is not None:
            assert all(layer.get_seq_length() == 0 for layer in cache.layers)
            assert all(layer.keys is None and layer.values is None for layer in cache.layers)
    finally:
        handle.remove()
        kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("mode", ["fakequant", "cache"])
@pytest.mark.parametrize("chunks", [(25,), (6, 1, 7, 2, 9)])
def test_kv3_chunked_teacher_forcing_matches_single_token(
    family: str, attention: str, mode: str, chunks: tuple[int, ...],
) -> None:
    from transformers.cache_utils import DynamicCache

    model = _model(family, attention=attention)
    oracle = _model(family, attention=attention)
    ids = torch.randint(1, 32, (1, sum(chunks)), generator=torch.Generator().manual_seed(42))
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4, mode=mode))
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for partition in ((1,) * ids.shape[-1], chunks):
            cache = kv_api.make_cache(model) if mode == "cache" else DynamicCache()
            expected_cache = DynamicCache()
            pieces: list[torch.Tensor] = []
            offset = 0
            for count in partition:
                tokens = ids[:, offset:offset + count]
                actual = model(tokens, past_key_values=cache, use_cache=True).logits
                pieces.append(actual)
                if mode == "cache":
                    # Independent native attention oracle: read the already stored
                    # prefix plus all appended FP tokens, then quantize for the next call.
                    expected = oracle(tokens, past_key_values=expected_cache, use_cache=True).logits
                    assert bit_equal(actual, expected)
                    for layer in expected_cache.layers:
                        for name, axis in (("keys", "channel"), ("values", "token")):
                            previous = offset // 4 * 4 if axis == "channel" else max(0, offset - 4)
                            stop = ((offset + count) // 4 * 4 if axis == "channel"
                                    else max(0, offset + count - 4))
                            states = getattr(layer, name)
                            middle = (_simulate(states[..., previous:stop, :], spec, axis)
                                      if stop > previous else states[..., previous:stop, :])
                            setattr(layer, name, torch.cat((states[..., :previous, :], middle,
                                                          states[..., stop:, :]), dim=-2))
                    for (key, value), (expected_key, expected_value) in zip(
                        cache, expected_cache, strict=True,
                    ):
                        assert bit_equal(key, expected_key)
                        assert bit_equal(value, expected_value)
                offset += count
                assert cache.get_seq_length() == offset
                if mode == "cache":
                    assert model._tricast_kv_patch.cache is None
                    assert all(layer.attention_keys is None and layer.attention_values is None
                               for layer in cache.layers)
            outputs.append(torch.cat(pieces, dim=1))
    assert all(torch.isfinite(output).all() for output in outputs)
    if mode == "fakequant":
        # I2: query chunking changes only fp32 attention summation order; the
        # virtual pre-write cache schedule is the same for every fakequant query.
        assert (outputs[0] - outputs[1]).abs().max().item() <= 1e-5
    else:
        # V2 deployment intentionally differs: each multi-token append attends
        # its entire appended chunk in FP, rather than simulating decode writes.
        assert not bit_equal(outputs[0], outputs[1])
    kv_api.remove_kv(model)
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
    assert (outputs[0] - baseline).abs().max().item() > 1e-5
    if mode == "cache" and len(chunks) == 1:
        assert bit_equal(outputs[1], baseline)
    else:
        assert (outputs[1] - baseline).abs().max().item() > 1e-5

@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_kv3_cache_attention_snapshots_clear_after_forward_exception(
    family: str, attention: str,
) -> None:
    model = _model(family, attention=attention)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4))
    cache = kv_api.make_cache(model)
    patch = model._tricast_kv_patch
    observed: list[bool] = []

    def fail_after_attention(_module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
        assert patch.cache is cache
        assert cache.layers[0].attention_keys is not None
        assert cache.layers[0].attention_values is not None
        observed.append(True)
        raise RuntimeError("interrupted after attention")

    handle = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(fail_after_attention)
    try:
        with pytest.raises(RuntimeError, match="interrupted after attention"), torch.inference_mode():
            model(torch.tensor([[1, 3, 5, 7, 11, 13, 17]]), past_key_values=cache, use_cache=True)
    finally:
        handle.remove()
    assert observed == [True]
    assert patch.cache is None and patch.token_mask is None
    assert all(layer.attention_keys is None and layer.attention_values is None for layer in cache.layers)
    # A failed forward must not leave stale transient state affecting an independent call.
    ids = torch.tensor([[1, 3, 5, 7, 11, 13, 17]])
    with torch.inference_mode():
        first = model(ids, past_key_values=kv_api.make_cache(model), use_cache=True).logits
        second = model(ids, past_key_values=kv_api.make_cache(model), use_cache=True).logits
    assert torch.equal(first, second)
    assert patch.cache is None and patch.token_mask is None
    kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_kv3_fakequant_static_capacity_preserves_query_flush_schedule(
    family: str, attention: str,
) -> None:
    from transformers.cache_utils import StaticCache

    model = _model(family, attention=attention)
    ids = torch.randint(1, 32, (1, 25), generator=torch.Generator().manual_seed(42))
    changed = ids.clone()
    changed[:, -2:] = changed[:, -2:] % 31 + 1
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, residual=4, mode="fakequant"))
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for tokens in (ids, changed):
            cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=32,
                                device="cpu", dtype=torch.float32)
            pieces: list[torch.Tensor] = []
            offset = 0
            for count in (6, 1, 7, 2, 9):
                stop = offset + count
                pieces.append(model(tokens[:, offset:stop], attention_mask=torch.ones((1, stop)),
                                    cache_position=torch.arange(offset, stop),
                                    past_key_values=cache, use_cache=True).logits)
                offset = stop
            outputs.append(torch.cat(pieces, dim=1))
    assert all(torch.isfinite(output).all() for output in outputs)
    # Static unused capacity changes matrix dimensions, not the query-time KV schedule;
    # the same fp32 attention-reduction allowance as streaming/full-window applies.
    assert (outputs[0] - full).abs().max().item() <= 1e-5
    assert torch.equal(outputs[0][:, :-2].contiguous().view(torch.int32),
                       outputs[1][:, :-2].contiguous().view(torch.int32))
    kv_api.remove_kv(model)
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
    assert (outputs[0] - baseline).abs().max().item() > 1e-5


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_kv3_sliding_window_preserves_fakequant_streaming_scale_domains(attention: str) -> None:
    from dataclasses import replace

    from transformers.cache_utils import DynamicCache

    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        use_sliding_window=True,
        sliding_window=5,
        layer_types=["sliding_attention", "sliding_attention"],
    )
    config._attn_implementation = attention
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = transformers.Qwen3ForCausalLM(config).float().eval()
    ids = torch.randint(1, 32, (1, 21), generator=torch.Generator().manual_seed(42))
    spec = get_scheme("kivi2").with_(group_size=4)
    kv = KVSpec(key=spec, value=spec, residual=4, mode="fakequant")
    with torch.inference_mode():
        baseline = model(ids, use_cache=False).logits
    kv_api.apply_kv(model, kv)
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        raw_cache = DynamicCache()
        fakequant_streamed = torch.cat([
            model(ids[:, position:position + 1], past_key_values=raw_cache, use_cache=True).logits
            for position in range(ids.shape[-1])
        ], dim=1)
    kv_api.apply_kv(model, replace(kv, mode="cache"))
    cache = kv_api.make_cache(model)
    with torch.inference_mode():
        streamed = torch.cat([
            model(ids[:, position:position + 1], past_key_values=cache, use_cache=True).logits
            for position in range(ids.shape[-1])
        ], dim=1)
    assert torch.isfinite(full).all() and torch.isfinite(fakequant_streamed).all()
    assert torch.isfinite(streamed).all()
    # Attention visibility is not token validity: old keys outside the sliding
    # window must retain their original scale-group alignment and flush clock.
    # Full and token-wise attention have only fp32 accumulation-order differences.
    assert (full - streamed).abs().max().item() <= 1e-5
    assert (fakequant_streamed - streamed).abs().max().item() <= 1e-5
    assert (full - baseline).abs().max().item() > 1e-5
    kv_api.remove_kv(model)


@pytest.mark.parametrize("axis", ["channel", "token"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_kv4_fakequant_selector_reads_prewrite_state(
    axis: str, enabled: bool, padded: bool,
) -> None:
    from tricast.kv.attention import _flush_mask

    spec = get_scheme("kivi2").with_(group_size=4) if enabled else None
    # R=8 deliberately contains two G=4 groups: flushing at each group boundary
    # would be early. Query 8 sees the first full buffer, but query 7 does not.
    queries = torch.tensor([[7, 8, 9, 15, 16, 17]])
    real_positions = torch.arange(20).reshape(1, -1)
    if padded:
        positions = torch.cat((torch.tensor([[-1, -1]]), real_positions), dim=-1)
    else:
        positions = real_positions
    actual = _flush_mask(positions, queries, spec, axis, residual=8)
    expected = torch.zeros_like(actual)
    for query_index, query in enumerate(queries[0].tolist()):
        boundary = query // 8 * 8 if axis == "channel" else max(0, query - 8)
        expected[0, 0, query_index] = (positions[0] >= 0) & (positions[0] < boundary) & enabled
    assert torch.equal(actual, expected)
    # Own and future K/V always stay original, including a boundary-triggering write.
    assert not (actual & (positions[:, None, None, :] >= queries[:, None, :, None])).any()
    if enabled:
        assert actual.any()
    else:
        assert not actual.any()


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("mask_type", ["bool", "additive"])
@pytest.mark.parametrize("residual", [4, 128])
def test_kv4_fakequant_rejects_noncausal_4d_mask_before_forward(
    family: str, attention: str, mask_type: str, residual: int,
) -> None:
    from transformers.cache_utils import DynamicCache

    model = _model(family, attention=attention)
    spec = get_scheme("kivi2").with_(group_size=4)
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, mode="fakequant", residual=residual))
    ids = torch.tensor([[1, 3, 5, 7, 11, 13]])
    mask = (torch.ones((1, 1, 6, 6), dtype=torch.bool) if mask_type == "bool"
            else torch.zeros((1, 1, 6, 6), dtype=torch.float32))
    cache = DynamicCache()
    forward_calls: list[bool] = []
    handle = model.model.register_forward_pre_hook(lambda *_args: forward_calls.append(True))
    try:
        with pytest.raises(NotImplementedError, match="causal 4D mask"), torch.inference_mode():
            model(ids, attention_mask=mask, past_key_values=cache, use_cache=True)
        assert forward_calls == []
        assert cache.get_seq_length() == 0
        assert all(layer.keys is None and layer.values is None for layer in cache.layers)
    finally:
        handle.remove()
        kv_api.remove_kv(model)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("mask_type", ["bool", "additive"])
@pytest.mark.parametrize("residual", [4, 128])
def test_kv4_fakequant_accepts_causal_4d_mask_without_future_leakage(
    family: str, attention: str, mask_type: str, residual: int,
) -> None:
    model = _model(family, attention=attention)
    spec = get_scheme("kivi2").with_(group_size=4)
    ids = torch.randint(1, 32, (1, 13), generator=torch.Generator().manual_seed(42))
    changed = ids.clone()
    changed[:, -2:] = changed[:, -2:] % 31 + 1
    allowed = torch.ones((1, 1, 13, 13), dtype=torch.bool).tril()
    mask = (allowed if mask_type == "bool" else torch.zeros_like(allowed, dtype=torch.float32)
            .masked_fill(~allowed, torch.finfo(torch.float32).min))
    # HF eager consumes additive masks; passing it bool directly adds 0/1
    # scores rather than blocking keys. Compare to the equivalent causal mask.
    native_mask = (torch.zeros_like(allowed, dtype=torch.float32)
                   .masked_fill(~allowed, torch.finfo(torch.float32).min)
                   if attention == "eager" else mask)
    with torch.inference_mode():
        baseline = model(ids, attention_mask=native_mask, use_cache=False).logits
    kv_api.apply_kv(model, KVSpec(key=spec, value=spec, mode="fakequant", residual=residual))
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, use_cache=False).logits
        mutated = model(changed, attention_mask=mask, use_cache=False).logits
    assert torch.isfinite(actual).all() and torch.isfinite(mutated).all()
    assert bit_equal(actual[:, :-2], mutated[:, :-2])
    if residual == 4:
        assert not bit_equal(actual, baseline)
    else:
        assert bit_equal(actual, baseline)
    kv_api.remove_kv(model)
