"""Reversible HF patching, operand caching, and quantized STE training."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from tricast.recipe import load_recipe


@pytest.fixture
def api():
    pytest.importorskip("tricast.transforms")
    pytest.importorskip("tricast.quant.api")
    pytest.importorskip("tricast.reference.mma")
    return pytest.importorskip("tricast.nn")


@pytest.fixture(params=["llama", "qwen3"])
def tiny_model(request):
    transformers = pytest.importorskip("transformers")
    config_cls, model_cls = (
        (transformers.LlamaConfig, transformers.LlamaForCausalLM) if request.param == "llama"
        else (transformers.Qwen3Config, transformers.Qwen3ForCausalLM)
    )
    torch.manual_seed(42)
    config = config_cls(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, intermediate_size=128, vocab_size=256,
                        max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = "eager"
    return model_cls(config).eval()


def test_tiny_hf_patch_restore(api, tiny_model):
    model = tiny_model
    tokens = torch.tensor([[1, 2, 3, 4]])
    originals = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    with torch.no_grad():
        expected = model(tokens).logits
        report = api.patch_model(model, load_recipe("fp64_reference"), backend="reference")
        actual = model(tokens).logits
    assert report.patched and report.skipped == ["lm_head"]
    assert model.lm_head is originals["lm_head"]
    assert len(list(api.iter_emulinear(model))) == len(originals) - 1
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    # The dense baseline uses a different reduction order from the fp64 FMA chain.
    assert (actual - expected).abs().max().item() < 1e-5
    api.unpatch_model(model)
    for name, module in originals.items():
        assert model.get_submodule(name) is module
    with torch.no_grad():
        assert torch.equal(model(tokens).logits, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_passthrough_shape_dtype_and_dispatch(api, dtype, monkeypatch):
    import tricast.nn.linear as linear_module

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3)).to(dtype).eval()
    calls = []
    original_gemm = linear_module.gemm

    def tracked(*args, **kwargs):
        calls.append(args[2].algorithm)
        return original_gemm(*args, **kwargs)

    monkeypatch.setattr(linear_module, "gemm", tracked)
    api.patch_model(model, load_recipe("bf16_passthrough"), backend="reference")
    x = torch.randn(2, 2, 4, dtype=dtype)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2, 3) and y.dtype == dtype
    assert torch.isfinite(y).all() and calls == ["fp32_fma"]
    assert model[0].bias.dtype == torch.float32
    assert model[0]._weight_operand.values.stride(0) == 1


def test_ste_gradients_and_refresh(api):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3, bias=True)).train()
    base = copy.deepcopy(model)
    api.patch_model(model, load_recipe("fp64_reference"), backend="reference")
    x = torch.randn(2, 4, requires_grad=True)
    xb = x.detach().clone().requires_grad_(True)
    model(x).sum().backward()
    base(xb).sum().backward()
    assert torch.equal(x.grad, xb.grad)
    assert torch.equal(model[0].weight.grad, base[0].weight.grad)
    assert torch.equal(model[0].bias.grad, base[0].bias.grad)
    cached = model[0]._weight_operand
    with torch.no_grad():
        model[0].weight.add_(1)
    model(x)
    assert model[0]._weight_operand is not cached
    assert torch.equal(model[0]._weight_operand.values, model[0].weight.float())


def test_quantized_ste_uses_quantized_operands(api):
    from tricast.mma.api import as_operand
    from tricast.quant.api import fake_quant

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3, bias=False)).train()
    recipe = load_recipe({"name": "qat", "defaults": {
        "weight": "fp8_tensor", "activation": "fp8_tensor", "mma": {"preset": "fp64", "out_format": "fp32"}}})
    api.patch_model(model, recipe, backend="reference")
    x = torch.randn(2, 4, requires_grad=True)
    model(x).sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    w2 = model[0].weight.detach().clone().requires_grad_(True)
    a = fake_quant(x2, recipe.defaults.activation, backend="reference")
    b = fake_quant(w2, recipe.defaults.weight, backend="reference")
    (a @ b.T).sum().backward()
    assert torch.equal(x.grad, x2.grad)
    assert torch.equal(model[0].weight.grad, w2.grad)
    assert as_operand(model[0]._weight_operand) is model[0]._weight_operand


def test_include_override_and_idempotence(api):
    model = nn.ModuleDict({"a": nn.Linear(4, 3), "b": nn.Linear(4, 3), "lm_head": nn.Linear(4, 3)})
    recipe = load_recipe({"name": "select", "include": ["a", "lm_head"], "defaults": {
        "mma": {"preset": "fp64", "out_format": "fp32"}},
        "overrides": [{"match": "a", "weight": "fp8_row"}]})
    report = api.patch_model(model, recipe, backend="reference")
    assert [name for name, _ in report.patched] == ["a"]
    assert report.skipped == ["b", "lm_head"]
    assert model["a"].spec.weight.granularity == "row"
    assert not api.patch_model(model, recipe, backend="reference").patched


def test_weight_cache_once_and_training_transform(api):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3)).eval()
    base = copy.deepcopy(model)
    recipe = load_recipe({"name": "rht", "defaults": {"transform": "random_hadamard",
                         "mma": {"preset": "fp64", "out_format": "fp32"}}})
    api.patch_model(model, recipe, backend="reference")
    cached = model[0]._weight_operand
    x = torch.randn(2, 4, requires_grad=True)
    with torch.no_grad():
        y = model(x)
        model(x)
    assert model[0]._weight_operand is cached
    # Orthogonal transforms and the dense baseline have different rounding sites.
    assert (y - base(x)).abs().max() < 1e-6
    model.train()
    model(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_device_dtype_change_rebuilds_cache(api):
    model = nn.Sequential(nn.Linear(4, 3)).eval()
    api.patch_model(model, load_recipe("fp64_reference"), backend="reference")
    cached = model[0]._weight_operand
    model.bfloat16()
    with torch.no_grad():
        y = model(torch.ones(2, 4, dtype=torch.bfloat16))
    assert y.dtype == torch.bfloat16 and model[0].bias.dtype == torch.float32
    assert model[0]._weight_operand is not cached
    assert torch.equal(model[0]._weight_operand.values, model[0].weight.float())
    api.unpatch_model(model)
    assert model[0].bias.dtype == torch.bfloat16
    assert model(torch.ones(2, 4, dtype=torch.bfloat16)).dtype == torch.bfloat16


def _fp64_linear(x, linear):
    """fp64_reference semantics: exact dot products rounded once to fp32, then the fp32 bias."""
    y = (x.double() @ linear.weight.double().T).float()
    return y if linear.bias is None else y + linear.bias


def test_replaced_or_edited_weight_is_requantized(api):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3, bias=False)).eval()
    fresh = nn.Sequential(nn.Linear(4, 3, bias=False)).eval()
    api.patch_model(model, load_recipe("fp64_reference"), backend="reference")
    x = torch.randn(2, 4)
    with torch.no_grad():
        model(x)
        model.load_state_dict(fresh.state_dict(), assign=True)  # new Parameter, same _version
        assert torch.equal(model(x), _fp64_linear(x, fresh[0]))
        model[0].weight.mul_(2)
        assert torch.equal(model(x), _fp64_linear(x, model[0]))


def test_patch_model_built_in_inference_mode(api):
    with torch.inference_mode():
        model = nn.Sequential(nn.Linear(4, 3)).eval()
        api.patch_model(model, load_recipe("fp64_reference"), backend="reference")
        x = torch.randn(2, 4)
        assert torch.equal(model(x), _fp64_linear(x, model[0]))


def test_operand_k_major_follows_values():
    from tricast.formats import FP32
    from tricast.mma.operand import Operand

    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    operand = Operand(values, FP32)
    assert torch.equal(operand.k_major(), values.T)
    values.add_(10)
    assert torch.equal(operand.k_major(), values.T)
    operand.values = torch.ones(3, 2)
    assert operand.k_major().shape == (2, 3)


def test_shared_linear_preserved(api):
    layer = nn.Linear(4, 3)
    model = nn.ModuleDict({"a": layer, "b": layer})
    api.patch_model(model, load_recipe("fp64_reference"), backend="reference")
    assert model["a"] is model["b"]
    api.unpatch_model(model)
    assert model["a"] is layer and model["b"] is layer


@pytest.mark.parametrize("recipe_name", [
    "bf16_passthrough", "fp64_reference", "hopper_fp8_w8a8", "ada_fp8_w8a8",
    "blackwell_fp8_w8a8", "fp8_f7_lowacc", "fp8_f7_decoupled", "deepseek_fp8_block",
    "mxfp8_w_a", "mxfp6_w_a", "mxfp4_w_a", "nvfp4_w_a", "nvfp4_4o6", "msfp12_bfp",
    "int8_row_w8a8", "w4a16_g128_zp_gptq", "fp8_ema_static", "fp8_delayed_history",
    "mxfp4_rht", "nvfp4_smoothquant",
])
def test_every_bundled_recipe_executes(api, recipe_name):
    from tricast.calibration import calibrate

    class TokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 16)
            self.proj = nn.Linear(16, 8)

        def forward(self, input_ids):
            return self.proj(self.embed(input_ids))

    torch.manual_seed(42)
    model = TokenModel().eval()
    recipe = load_recipe(recipe_name)
    api.patch_model(model, recipe, backend="reference")
    ids = torch.tensor([[1, 2, 3, 4]])
    if recipe.needs_calibration:
        calibrate(model, recipe, input_ids=ids, seqlen=4, samples=1, seed=42)
    with torch.no_grad():
        actual = model(ids)
    assert actual.shape == (1, 4, 8) and actual.dtype == torch.float32
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("defaults,message", [
    ({"weight": "fp8_tensor_ema"}, "weight.observer"),
    ({"weight_algo": "gptq"}, "weight_algo"),
])
def test_unsupported_weight_paths_fail_before_patching(api, defaults, message):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    originals = list(model)
    recipe = load_recipe({"name": "bad", "defaults": defaults})
    with pytest.raises(ValueError, match=message):
        api.patch_model(model, recipe, backend="reference")
    assert list(model) == originals


def test_layer_selectors_skip_and_report(api, tiny_model):
    recipe = load_recipe({"name": "selectors", "defaults": {"mma": "fp64"}, "overrides": [
        {"layers": "0", "modules": ["q_proj", "k_proj"], "match": "*.self_attn.*", "skip": True},
        {"layers": "0-1", "modules": ["q_proj"], "weight": "fp8_row"},
    ]})
    original = tiny_model.model.layers[0].self_attn.q_proj
    report = api.patch_model(tiny_model, recipe, backend="reference")
    entries = {entry["name"]: entry for entry in report.layers}
    assert tiny_model.model.layers[0].self_attn.q_proj is original
    assert entries["model.layers.0.self_attn.q_proj"]["skipped"] == "overrides[0].skip"
    assert entries["lm_head"]["skipped"] == "excluded"
    entry = entries["model.layers.1.self_attn.q_proj"]
    assert entry["weight"]["format"] == "fp8_e4m3" and entry["weight"]["granularity"] == "row"
    assert entry["activation"] is None and entry["mma"]["algorithm"] == "fp64"
    assert entry["transform"] == "none" and entry["weight_algo"] == "rtn"
    assert entry["skipped"] is None


def test_layer_selectors_follow_the_decoder_list_not_module_names(api):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    model = transformers.BloomForCausalLM(transformers.BloomConfig(
        vocab_size=32, hidden_size=16, n_layer=3, n_head=2)).eval()  # blocks live in transformer.h
    recipe = load_recipe({"name": "first_last", "backend": "reference",
                          "defaults": {"weight": "mxfp4", "mma": {"preset": "fp64", "out_format": "fp32"}},
                          "overrides": [{"layers": "0,-1", "skip": True},
                                        {"modules": ["down_proj"], "skip": True}]})
    with pytest.warns(UserWarning, match=r"overrides \[1\] select no Linear"):
        report = api.patch_model(model, recipe)
    assert report.decoder_layers == "transformer.h" and report.unused_overrides == [1]
    patched = {name for name, _ in report.patched}
    assert any(name.startswith("transformer.h.1.") for name in patched)
    assert not any(name.startswith(("transformer.h.0.", "transformer.h.2.")) for name in patched)
    api.unpatch_model(model)


def test_layer_selectors_refuse_an_unresolved_decoder(api):
    model = nn.Sequential(nn.Linear(4, 4))
    recipe = load_recipe({"name": "l", "defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}},
                          "overrides": [{"layers": "0", "skip": True}]})
    with pytest.raises(ValueError, match="decoder-block list"):
        api.patch_model(model, recipe, backend="reference")


def test_first_last_recipe_uses_actual_decoder_count(api):
    model = nn.ModuleDict({"model": nn.ModuleDict({"layers": nn.ModuleList([
        nn.ModuleDict({"proj": nn.Linear(4, 4)}) for _ in range(4)
    ])})})
    originals = [layer["proj"] for layer in model["model"]["layers"]]
    report = api.patch_model(model, "mixed_first_last_bf16", backend="reference")
    assert [name for name, _ in report.patched] == ["model.layers.1.proj", "model.layers.2.proj"]
    assert model["model"]["layers"][0]["proj"] is originals[0]
    assert model["model"]["layers"][3]["proj"] is originals[3]
    assert model["model"]["layers"][1]["proj"].spec.weight.format.name == "fp4_e2m1"


@pytest.mark.parametrize("mode", ["cache", "fakequant"])
def test_recipe_kv_apply_and_unpatch(api, tiny_model, mode):
    kv = pytest.importorskip("tricast.kv")
    recipe = load_recipe({"name": "kv", "defaults": {"mma": "fp64"}, "include": [],
                          "kv": {"preset": "kivi2", "mode": mode, "layers": "1"}})
    assert not recipe.needs_calibration
    report = api.patch_model(tiny_model, recipe, backend="reference")
    assert report.kv == ["model.layers.1.self_attn"]
    assert not report.patched
    assert tiny_model._tricast_kv_patch.kv == recipe.kv
    if mode == "cache":
        assert kv.make_cache(tiny_model).kv == recipe.kv
    else:
        with torch.no_grad():
            assert torch.isfinite(tiny_model(torch.tensor([[1, 2, 3, 4]])).logits).all()
    api.unpatch_model(tiny_model)
    assert not hasattr(tiny_model, "_tricast_kv_patch")
    assert not hasattr(tiny_model, "_tricast_kv_patched")


@pytest.mark.parametrize("scheme", ["bf16", "fp16", "fp8_row", "nvfp4", "int4_g128_zp"])
def test_compact_weight_operand_preserves_grid_and_gemm(scheme):
    from tricast.formats import container_dtype
    from tricast.mma.api import as_operand, gemm
    from tricast.mma.spec import MMASpec
    from tricast.quant.api import quantize
    from tricast.quant.spec import get_scheme

    gen = torch.Generator().manual_seed(42)
    weight = torch.randn(3, 8, generator=gen)
    quantized = quantize(weight, get_scheme(scheme), backend="reference")
    baseline = as_operand(quantized)
    compact = as_operand(quantized, compact=True)
    fmt = quantized.spec.dequant_format if quantized.spec.mma_input == "dequant" else quantized.spec.format
    assert compact.values.dtype == container_dtype(fmt)
    assert torch.equal(compact.values.float(), baseline.values)
    assert torch.equal(torch.signbit(compact.values), torch.signbit(baseline.values))
    for algorithm in ("fp64", "fp32_fma", "cofda"):
        spec = MMASpec(algorithm=algorithm, f_bits=23, chunk_size=4, out_format="fp32")
        activation = torch.randn(2, 8, generator=gen)
        assert torch.equal(gemm(activation, compact, spec, backend="reference"),
                           gemm(activation, baseline, spec, backend="reference"))


@pytest.mark.parametrize("recipe_name", ["bf16_passthrough", "nvfp4_w_a", "hopper_fp8_w8a8"])
def test_cached_weight_operand_half_size_without_numeric_change(api, recipe_name):
    from dataclasses import replace

    gen = torch.Generator().manual_seed(42)
    model = nn.Sequential(nn.Linear(8, 4, bias=False)).bfloat16().eval()
    api.patch_model(model, load_recipe(recipe_name), backend="reference")
    layer = model[0]
    compact = layer._weight_operand
    assert compact.values.element_size() == 2
    assert compact.values.numel() * compact.values.element_size() == layer.weight.numel() * 2
    assert compact.values.stride(0) == 1
    x = torch.randn(2, 8, generator=gen, dtype=torch.bfloat16)
    with torch.no_grad():
        actual = model(x)
        layer._weight_operand = replace(compact, values=compact.values.float())
        baseline = model(x)
        layer._weight_operand = compact
    assert torch.equal(actual, baseline)
    assert "_k_major" not in compact.__dict__


def test_compact_kernel_expansion_is_not_resident(monkeypatch):
    import sys
    from dataclasses import replace
    from types import ModuleType

    from tricast.mma.api import as_operand, gemm
    from tricast.mma.spec import MMASpec
    from tricast.reference.mma import gemm_reference

    gen = torch.Generator().manual_seed(42)
    values = torch.randn(3, 8, generator=gen).bfloat16().T.contiguous().T
    compact = as_operand(values, compact=True)
    assert compact.values.data_ptr() == values.data_ptr()
    calls = []

    def kernel(a, b, spec, bias):
        calls.append(b.values.dtype)
        assert b is not compact
        assert b.values.dtype == torch.float32
        assert b.k_major().data_ptr() == b.values.data_ptr()
        return gemm_reference(a, b, spec, bias)

    module = ModuleType("tricast.kernels.mma")
    module.gemm_triton = kernel
    monkeypatch.setitem(sys.modules, "tricast.kernels.mma", module)
    activation = torch.randn(2, 8, generator=gen)
    spec = MMASpec(algorithm="fp64", out_format="fp32")
    expected = gemm(activation, replace(compact, values=values.float()), spec, backend="reference")
    for _ in range(2):
        assert torch.equal(gemm(activation, compact, spec, backend="triton"), expected)
        assert compact.values.dtype == torch.bfloat16
        assert "_k_major" not in compact.__dict__
    assert calls == [torch.float32, torch.float32]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16])
def test_compact_plain_operand_preserves_format_and_special_values(dtype):
    from tricast.formats import container_dtype
    from tricast.mma.api import as_operand, gemm
    from tricast.mma.spec import MMASpec

    values = torch.tensor([[0.0, -0.0, 0.5, -1.0, 2.0]]).to(dtype)
    baseline = as_operand(values)
    compact = as_operand(values, compact=True)
    assert compact.fmt == baseline.fmt
    assert compact.values.dtype == container_dtype(baseline.fmt)
    assert torch.equal(compact.values.float().view(torch.int32), baseline.values.view(torch.int32))
    spec = MMASpec(algorithm="cofda", f_bits=7, chunk_size=4, out_format="fp32")
    x = torch.tensor([[2.0, 4.0, 1.0, 1.0, 1.0]])
    assert torch.equal(gemm(x, compact, spec), gemm(x, baseline, spec))
    special = as_operand(torch.tensor([[float("nan")]]).to(dtype), compact=True)
    assert torch.isnan(special.values).all()
