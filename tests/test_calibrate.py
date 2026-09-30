"""Calibration gates, shared inputs, and sequential decoder fitting."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from tricast.calibration import calibrate
from tricast.nn import iter_emulinear, patch_model
from tricast.recipe import load_recipe


@pytest.fixture(params=["llama", "qwen3"])
def tiny(request):
    transformers = pytest.importorskip("transformers")
    config_cls, model_cls = (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM) if request.param == "llama"
        else (transformers.Qwen3Config, transformers.Qwen3ForCausalLM)
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = config_cls(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        vocab_size=32, max_position_embeddings=32)
    config._attn_implementation = "eager"
    yield model_cls(config).eval()
    torch.set_num_threads(old_threads)


def _recipe(**defaults):
    return load_recipe({"name": "calibration", "defaults": {
        "mma": {"preset": "fp64", "out_format": "fp32"}, **defaults}, "backend": "reference"})


@pytest.mark.parametrize("defaults", [
    {"weight": "nvfp4", "transform": "smoothquant"},
    {"weight": "int4_g128_zp", "weight_algo": "gptq"},
    {"activation": "fp8_tensor_ema"},
])
def test_calibration_required_before_forward(tiny, defaults):
    recipe = _recipe(**defaults)
    patch_model(tiny, recipe)
    ids = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(RuntimeError, match=r"call tricast.calibrate\(\.\.\.\)"):
        tiny(ids)
    result = calibrate(tiny, recipe, input_ids=ids, samples=1, seqlen=4)
    assert result["n_tokens"] == 4 and result["mode"] == "nonsequential"
    assert torch.isfinite(tiny(ids).logits).all()
    for _, layer in iter_emulinear(tiny):
        assert layer._calibrated
        assert layer._hessian is None and layer._stats is None and layer._calibration_inputs is None
        assert getattr(layer, "_gptq_hessian", None) is None


def test_history_is_online_without_calibration(tiny):
    recipe = _recipe(activation="fp8_tensor_delayed")
    patch_model(tiny, recipe)
    assert not recipe.needs_calibration
    assert torch.isfinite(tiny(torch.tensor([[1, 2, 3, 4]])).logits).all()
    assert all(layer.observer.count == 1 for _, layer in iter_emulinear(tiny))


@pytest.mark.parametrize("kind", ["smoothquant", "awq"])
@pytest.mark.parametrize("shared", [True, False])
def test_shared_input_groups_and_scales(tiny, kind, shared):
    recipe = _recipe(weight="nvfp4", transform={"kind": kind, "grid": 3, "share_inputs": shared})
    patch_model(tiny, recipe)
    result = calibrate(tiny, recipe, input_ids=[[1, 2, 3, 4]], samples=2, seqlen=4, seed=42)
    groups = [set(group) for group in result["groups"]]
    for index in range(2):
        prefix = f"model.layers.{index}."
        expected = [[prefix + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj")],
                    [prefix + "mlp." + name for name in ("gate_proj", "up_proj")],
                    [prefix + "self_attn.o_proj"], [prefix + "mlp.down_proj"]]
        if not shared:
            expected = [[name] for group in expected for name in group]
        for group in expected:
            assert set(group) in groups
            diag = tiny.get_submodule(group[0]).transform.diag
            assert all(torch.equal(tiny.get_submodule(name).transform.diag, diag) for name in group)
    assert result["n_tokens"] == 8
    assert result["config"]["seed"] == 42 and result["source"] == "input_ids"
    assert result["dataset"] is None and result["split"] is None


@pytest.mark.parametrize("gptq", [False, True])
def test_sequential_uses_quantized_preceding_decoder_inputs(tiny, gptq):
    ids = torch.tensor([[1, 2, 3, 4]])
    recipe = _recipe(weight={"format": "int2"}, activation="fp8_tensor_ema",
                     weight_algo="gptq" if gptq else "rtn")
    captured = []
    for sequential in (False, True):
        model = copy.deepcopy(tiny)
        patch_model(model, recipe)
        layer = model.model.layers[1].self_attn.q_proj
        inputs = []

        def capture(module, args, inputs=inputs):
            if module.mode == "calibrate":
                inputs.append(args[0].detach().clone())

        handle = layer.register_forward_pre_hook(capture)
        result = calibrate(model, recipe, input_ids=ids, samples=1, seqlen=4, sequential=sequential)
        handle.remove()
        assert len(inputs) == 1
        assert result["mode"] == ("sequential" if sequential else "nonsequential")
        assert result["reason"] is None
        assert layer.observer.static_amax == inputs[0].abs().amax()
        captured.append(inputs[0])
    assert not torch.equal(*captured)


def test_gptq_hessian_released_and_not_recomputed_on_move(tiny, monkeypatch):
    import tricast.nn.linear as linear_module

    recipe = _recipe(weight="int4_g128_zp", weight_algo="gptq")
    recipe.include = ["model.layers.0.self_attn.q_proj"]
    patch_model(tiny, recipe)
    calls = []
    original = linear_module.quantize_weight

    def tracked(*args, **kwargs):
        calls.append(kwargs["hessian"].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(linear_module, "quantize_weight", tracked)
    calibrate(tiny, recipe, input_ids=[[1, 2, 3, 4]], samples=1, seqlen=4)
    layer = tiny.model.layers[0].self_attn.q_proj
    before = layer._weight_operand.values.clone()
    assert len(calls) == 1 and torch.isfinite(calls[0]).all()
    assert layer._hessian is None and getattr(layer, "_gptq_hessian", None) is None
    layer.to("cpu")
    assert len(calls) == 1 and torch.equal(before, layer._weight_operand.values)
    tiny(torch.tensor([[1, 2, 3, 4]]))
    assert len(calls) == 1
    with torch.no_grad():
        layer.weight.add_(0.1)
    layer.to("cpu")
    with pytest.raises(RuntimeError, match="call tricast.calibrate"):
        tiny(torch.tensor([[1, 2, 3, 4]]))


def test_calibrated_gptq_survives_deepcopy_and_pickle(tiny):
    import pickle

    recipe = _recipe(weight="int4_g128_zp", weight_algo="gptq")
    recipe.include = ["model.layers.0.self_attn.q_proj"]
    patch_model(tiny, recipe)
    calibrate(tiny, recipe, input_ids=[[1, 2, 3, 4]], samples=1, seqlen=4)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = tiny(ids).logits
        for clone in (copy.deepcopy(tiny), pickle.loads(pickle.dumps(tiny))):
            assert torch.equal(clone(ids).logits, expected)
            clone.model.layers[0].self_attn.q_proj.weight.add_(0.1)  # a real edit still needs calibrate()
            with pytest.raises(RuntimeError, match="call tricast.calibrate"):
                clone(ids)


def test_refresh_after_unversioned_weight_edit(tiny):
    recipe = _recipe(weight="int4_g128_zp")
    recipe.include = ["model.layers.0.self_attn.q_proj"]
    patch_model(tiny, recipe)
    layer = tiny.model.layers[0].self_attn.q_proj
    x = torch.randn(3, layer.in_features)
    with torch.no_grad():
        before = layer(x)
        layer.weight.data.mul_(2)  # .data edits bypass the version counter
        assert torch.equal(layer(x), before)
        layer.refresh()
        assert not torch.equal(layer(x), before)


def test_observer_reservoirs_released_after_calibration(tiny):
    recipe = _recipe(activation={"scheme": "fp8_tensor", "observer": {"kind": "mse", "max_samples": 32}})
    patch_model(tiny, recipe)
    calibrate(tiny, recipe, input_ids=[[1, 2, 3, 4]], samples=1, seqlen=4)
    for _, layer in iter_emulinear(tiny):
        assert layer.observer.samples.numel() == 0
        assert layer.observer._priorities.numel() == 0
        assert layer.observer.static_amax is not None


class _UnknownDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4)

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))


def test_unknown_structure_reports_nonsequential_fallback():
    model = _UnknownDecoder()
    recipe = _recipe(activation="fp8_tensor_ema")
    result = calibrate(model, recipe, input_ids=[[1, 2, 3]], samples=1, seqlen=3, sequential=True)
    assert result["mode"] == "nonsequential"
    assert "model.model.layers" in result["reason"]
    assert result["config"]["sequential"] is True


def test_failed_calibration_keeps_initial_guard(tiny, monkeypatch):
    recipe = _recipe(transform="smoothquant")
    patch_model(tiny, recipe)
    original = tiny.forward

    def fail(*args, **kwargs):
        raise ValueError("intentional failure")

    monkeypatch.setattr(tiny, "forward", fail)
    with pytest.raises(ValueError, match="intentional failure"):
        calibrate(tiny, recipe, input_ids=[[1, 2, 3, 4]], samples=1, seqlen=4)
    monkeypatch.setattr(tiny, "forward", original)
    with pytest.raises(RuntimeError, match="call tricast.calibrate"):
        tiny(torch.tensor([[1, 2, 3, 4]]))
    for _, layer in iter_emulinear(tiny):
        assert not layer._forward_pre_hooks and not layer._calibration_passthrough
        assert layer._stats is None and layer._calibration_inputs is None


def test_stochastic_calibration_repeats_and_restores_rng(tiny):
    recipe = _recipe(weight={"format": "int2", "granularity": "row", "rounding": "sr"},
                     weight_algo="gptq")
    recipe.include = ["model.layers.0.self_attn.q_proj"]
    outputs = []
    for _ in range(2):
        model = copy.deepcopy(tiny)
        patch_model(model, recipe)
        before = torch.random.get_rng_state().clone()
        calibrate(model, recipe, input_ids=[[1, 2, 3, 4]], samples=1, seqlen=4, seed=42)
        assert torch.equal(torch.random.get_rng_state(), before)
        outputs.append(model.model.layers[0].self_attn.q_proj._weight_operand.values.clone())
    assert torch.equal(*outputs)


def test_tricast_lm_calibrate_false_preserves_forward_guard(tiny):
    from tricast.eval.lmeval import TriCastLM

    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]",
    ))
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )
    adapter = TriCastLM(pretrained=tiny, tokenizer=tokenizer, recipe="nvfp4_smoothquant",
                        backend="reference", calibrate=False, batch_size=1)
    with pytest.raises(RuntimeError, match="call tricast.calibrate"):
        adapter.model(torch.tensor([[1, 2, 3, 4]]))


def test_unknown_predecoder_order_reports_fallback():
    class PreDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.pre = nn.Linear(4, 4)
            self.model = nn.ModuleDict({"layers": nn.ModuleList([nn.Linear(4, 4)])})

        def forward(self, input_ids):
            return self.model.layers[0](self.pre(self.embed(input_ids)))

    recipe = _recipe(weight={"format": "int2"}, activation="fp8_tensor_ema")
    result = calibrate(PreDecoder(), recipe, input_ids=[[1, 2, 3]], samples=1, seqlen=3, sequential=True)
    assert result["mode"] == "nonsequential"
    assert "unknown execution order" in result["reason"]


def test_shared_views_use_same_storage_identity():
    class SharedViews(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.q = nn.Linear(4, 4)
            self.k = nn.Linear(4, 4)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            return self.q(x.view_as(x)) + self.k(x.view_as(x))

    result = calibrate(SharedViews(), _recipe(transform="smoothquant"),
                       input_ids=[[1, 2, 3]], samples=1, seqlen=3)
    assert result["groups"] == [["q", "k"]]


def test_keyword_linear_input_is_calibrated():
    class KeywordInput(_UnknownDecoder):
        def forward(self, input_ids):
            return self.proj(input=self.embed(input_ids))

    model = KeywordInput()
    recipe = _recipe(transform="smoothquant")
    result = calibrate(model, recipe, input_ids=[[1, 2, 3]], samples=1, seqlen=3)
    assert result["groups"] == [["proj"]]
    assert torch.isfinite(model(torch.tensor([[1, 2, 3]]))).all()


@pytest.mark.parametrize("defaults", [{"transform": "smoothquant"},
                                      {"weight": "int4_g128_zp", "weight_algo": "gptq"}])
def test_direct_calibration_lifecycle_cannot_bypass_guard(defaults):
    model = nn.Sequential(nn.Linear(4, 4))
    patch_model(model, _recipe(**defaults))
    layer = model[0]
    layer.begin_calibration()
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert torch.equal(layer(x), torch.nn.functional.linear(x, layer.weight, layer.bias))
    layer.clear_calibration()
    with pytest.raises(RuntimeError, match="call tricast.calibrate"):
        layer(x)
    layer.begin_calibration()
    layer(x)
    layer.finish_calibration()
    assert torch.isfinite(layer(x)).all() and not layer._calibration_passthrough


@pytest.mark.parametrize("prepatched", [False, True])
def test_stochastic_setup_restores_rng(prepatched):
    model = _UnknownDecoder()
    recipe = _recipe(weight={"format": "int2", "rounding": "sr"})
    if prepatched:
        patch_model(model, recipe)
    before = torch.random.get_rng_state().clone()
    calibrate(model, recipe, input_ids=[[1, 2, 3]], samples=1, seqlen=3, seed=42, device="cpu")
    assert torch.equal(before, torch.random.get_rng_state())


def test_gptq_input_sharing_reduces_live_hessian_storage(tiny):
    recipe = _recipe(weight="int4_g128_zp", weight_algo="gptq")
    patch_model(tiny, recipe)
    snapshots = []

    def capture(module, args, output):
        layers = [layer for _, layer in iter_emulinear(module) if layer.mode == "calibrate"]
        if layers:
            grams = {id(layer._hessian_accumulator.gram): layer._hessian_accumulator.gram for layer in layers}
            nbytes = sum(h.numel() * h.element_size() for h in grams.values())
            snapshots.append((len(layers), len(grams), nbytes))

    handle = tiny.register_forward_hook(capture)
    try:
        result = calibrate(tiny, recipe, input_ids=[[1, 2, 3, 4]], samples=2, seqlen=4)
    finally:
        handle.remove()
    assert snapshots == [(14, 8, 28_672)] * 2
    assert result["hessian_unshared_bytes"] == 40_960
    assert result["hessian_bytes"] == 28_672
    assert result["full_model_forwards"] == 3
    for block in range(2):
        assert [f"model.layers.{block}.self_attn.{name}_proj" for name in ("q", "k", "v")] \
            in result["hessian_groups"]
        assert [f"model.layers.{block}.mlp.{name}_proj" for name in ("gate", "up")] \
            in result["hessian_groups"]


def test_gptq_budget_rejects_before_allocating_hessians(tiny, monkeypatch):
    from tricast.nn.linear import HessianAccumulator

    recipe = _recipe(weight="int4_g128_zp", weight_algo="gptq")
    patch_model(tiny, recipe)
    updates = []
    original = HessianAccumulator.update

    def tracked(self, rows):
        updates.append(rows.shape)
        return original(self, rows)

    monkeypatch.setattr(HessianAccumulator, "update", tracked)
    with pytest.raises(ValueError, match=r"requires 28672 bytes.*hessian_max_bytes=28671"):
        calibrate(tiny, recipe, input_ids=[[1, 2, 3, 4]], samples=2, seqlen=4, hessian_max_bytes=28_671)
    assert updates == []
    assert all(layer._hessian is None and layer._hessian_accumulator is None
               for _, layer in iter_emulinear(tiny))
    assert tiny.config.use_cache is True
    with pytest.raises(RuntimeError, match="calibration is required"):
        tiny(torch.tensor([[1, 2, 3, 4]]))


@pytest.mark.parametrize("transform", ["none", "smoothquant"])
def test_cached_sequential_matches_full_model_replay(tiny, monkeypatch, transform):
    import tricast.calibration as calibration_module

    recipe = _recipe(weight="int4_g128_zp", activation="fp8_tensor_ema", weight_algo="gptq",
                     transform=transform)
    original = calibration_module._DecoderInputs._before
    models = [copy.deepcopy(tiny), copy.deepcopy(tiny)]
    results = []
    for index, model in enumerate(models):
        if index:
            def force_fallback(self, block_index):
                hook = original(self, block_index)

                def reject(module, args, kwargs):
                    hook(module, args, kwargs)
                    self.reason = "forced legacy replay oracle"
                return reject
            monkeypatch.setattr(calibration_module._DecoderInputs, "_before", force_fallback)
        patch_model(model, recipe)
        results.append(calibrate(model, recipe, input_ids=[[1, 2, 3, 4]], samples=2, seqlen=4,
                                 sequential=True, hessian_max_bytes=14_336))
    assert results[0]["sequential_cached"] is True
    assert results[0]["full_model_forwards"] == 2
    assert results[0]["hessian_bytes"] == 14_336
    assert results[1]["sequential_cached"] is False
    assert "forced legacy replay oracle" in results[1]["reason"]
    for (_, actual), (_, expected) in zip(iter_emulinear(models[0]), iter_emulinear(models[1]), strict=True):
        assert torch.equal(actual._weight_operand.values, expected._weight_operand.values)
        if actual._weight_operand.scale is None:
            assert expected._weight_operand.scale is None
        else:
            assert torch.equal(actual._weight_operand.scale, expected._weight_operand.scale)
        assert torch.equal(actual.observer.static_amax, expected.observer.static_amax)
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(models[0](ids).logits, models[1](ids).logits)


def test_custom_decoder_kwargs_report_full_replay_fallback():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, hidden_states, *, gain):
            return self.proj(hidden_states) * gain

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.model = nn.ModuleDict({"layers": nn.ModuleList([Block(), Block()])})

        def forward(self, input_ids):
            hidden = self.embed(input_ids)
            for block in self.model.layers:
                hidden = block(hidden, gain=hidden.mean())
            return hidden

    result = calibrate(Model(), _recipe(activation="fp8_tensor_ema"), input_ids=[[1, 2, 3]],
                       samples=2, seqlen=3, sequential=True)
    assert result["mode"] == "sequential" and not result["sequential_cached"]
    assert "custom kwargs require full replay" in result["reason"]
    assert result["full_model_forwards"] == 4


def test_input_groups_do_not_share_inplace_changed_tensor():
    class Mutating(_UnknownDecoder):
        def __init__(self):
            super().__init__()
            self.other = nn.Linear(4, 4)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            first = self.proj(x)
            x.add_(1)
            return first + self.other(x)

    result = calibrate(Mutating(), _recipe(weight="int4_g128_zp", weight_algo="gptq"),
                       input_ids=[[1, 2, 3]], samples=1, seqlen=3)
    assert result["hessian_groups"] == [["proj"], ["other"]]
    assert result["hessian_bytes"] == result["hessian_unshared_bytes"] == 256


def test_changed_input_sharing_fails_without_fitting_weights():
    class Changing(_UnknownDecoder):
        def __init__(self):
            super().__init__()
            self.other = nn.Linear(4, 4)
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            x = self.embed(input_ids)
            return self.proj(x) + self.other(x if self.calls == 1 else x + 1)

    model = Changing()
    recipe = _recipe(weight="int4_g128_zp", weight_algo="gptq")
    with pytest.raises(ValueError, match="input-sharing pattern changed"):
        calibrate(model, recipe, input_ids=[[1, 2, 3]], samples=1, seqlen=3)
    for _, layer in iter_emulinear(model):
        assert not layer._calibrated and layer._hessian_accumulator is None
        assert not layer._forward_pre_hooks


def test_c4_calibration_windows_are_document_local_and_lazy():
    import random

    from tricast.calibration import _windows

    tokenized = []
    documents = ["1 2", "10 11 12 13 14 15", "20 21 22 23 24 25"]

    def tokenizer(text, return_tensors):
        tokenized.append(text)
        return {"input_ids": torch.tensor([[int(token) for token in text.split()]])}

    class Documents:
        def __len__(self):
            return len(documents)

        def __getitem__(self, index):
            return {"text": documents[index]}

        def __iter__(self):
            raise AssertionError("the entire C4 shard must not be consumed")

    first = _windows(tokenizer, Documents(), None, samples=2, seqlen=3, seed=42, document_local=True)
    second = _windows(tokenizer, Documents(), None, samples=2, seqlen=3, seed=42, document_local=True)
    rng = random.Random(42)
    expected = []
    while len(expected) < 2:
        tokens = list(map(int, documents[rng.randint(0, len(documents) - 1)].split()))
        if len(tokens) <= 3:
            continue
        start = rng.randint(0, len(tokens) - 3 - 1)
        expected.append(tokens[start:start + 3])
    assert [window[0].tolist() for window in first] == expected
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert all("\n" not in text for text in tokenized)
    assert tokenized[:len(tokenized) // 2] == tokenized[len(tokenized) // 2:]


def test_c4_dataset_uses_memory_mapped_random_access(monkeypatch):
    import sys
    from types import SimpleNamespace

    from tricast.calibration import _dataset_texts

    calls = []
    documents = [{"text": "one document"}]

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return documents

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    assert _dataset_texts("c4", "train") is documents
    assert calls[0][1]["keep_in_memory"] is False
    assert not calls[0][1].get("streaming", False)
    assert calls[0][1]["data_files"] == {"train": "en/c4-train.00000-of-01024.json.gz"}


@pytest.mark.parametrize("transform", ["none", "smoothquant"])
def test_shared_hessian_matches_independent_accumulation(tiny, monkeypatch, transform):
    from tricast.calibration import _InputGroups

    recipe = _recipe(weight="int4_g128_zp", weight_algo="gptq", transform=transform)
    models = [copy.deepcopy(tiny), copy.deepcopy(tiny)]
    original = _InputGroups.hessian_groups
    for index, model in enumerate(models):
        if index:
            def separate(self, layers):
                return [[member] for group in original(self, layers) for member in group]
            monkeypatch.setattr(_InputGroups, "hessian_groups", separate)
        calibrate(model, recipe, input_ids=[[1, 2, 3, 4]], samples=2, seqlen=4)
    for (_, actual), (_, expected) in zip(iter_emulinear(models[0]), iter_emulinear(models[1]), strict=True):
        assert torch.equal(actual._weight_operand.values, expected._weight_operand.values)
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(models[0](ids).logits, models[1](ids).logits)


def test_decoder_replay_does_not_mutate_cpu_cached_inputs():
    from tricast.calibration import _DecoderInputs

    class InPlaceBlock(nn.Module):
        def forward(self, hidden_states, *, offset):
            hidden_states.add_(offset)
            offset.add_(1)
            return hidden_states

    replay = _DecoderInputs(nn.ModuleList([InPlaceBlock()]))
    replay.close()
    hidden = [torch.zeros(1, 2, 3)]
    offset = torch.ones(())
    replay.samples = [(torch.empty(0), [(False, (), {"offset": offset})])]
    first = replay.run(0, hidden, torch.device("cpu"))
    second = replay.run(0, hidden, torch.device("cpu"))
    assert torch.equal(first[0], torch.ones(1, 2, 3))
    assert torch.equal(first[0], second[0])
    assert torch.equal(hidden[0], torch.zeros(1, 2, 3))
    assert offset == 1
