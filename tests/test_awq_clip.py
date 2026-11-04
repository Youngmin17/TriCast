"""weight_algo awq_clip (tricast.weight_quant.clip, AWQ's auto_clip): the search equals the pinned official
auto_clip_layer bit for bit when both use the official quantizer; EmuLinear fits it on the AWQ-transformed
weight and samples, skips q_ / k_ Linears, applies it before RTN, and the calibration cache keeps it; the
recipe spec, its checks and the schema."""

from __future__ import annotations

import ast
import gc
import json
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator
from torch import nn

from tricast.calibration import _layer_state, _restore_layer
from tricast.nn.linear import EmuLinear
from tricast.quant.api import fake_quant
from tricast.recipe import ClipSearchSpec, load_recipe
from tricast.weight_quant.clip import clip_search, clip_skipped

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "src" / "tricast" / "schemas" /
                     "recipe.schema.json").read_text(encoding="utf-8"))
INT3 = {"scheme": "int4_g128_zp", "format": "uint3"}


def _official_functions() -> dict:
    """auto_clip_layer and pseudo_quantize_tensor, compiled from the pinned clone's files as written: the
    modules themselves import the awq_inference_engine CUDA extension, so they cannot be imported here."""
    from tricast.official import repos

    root = repos.ensure("awq")
    namespace: dict = {"torch": torch, "nn": nn, "gc": gc}
    for file, name in (("awq/quantize/quantizer.py", "pseudo_quantize_tensor"),
                       ("awq/quantize/auto_clip.py", "auto_clip_layer")):
        path = root / file
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _layer_inputs(rows: int, k: int, tokens: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, k, generator=generator) * 0.02
    w[:, torch.randperm(k, generator=generator)[:4]] *= 8.0  # heavy columns: clipping helps some groups
    x = torch.randn(tokens, k, generator=generator)
    x[:, torch.randperm(k, generator=generator)[:6]] *= 10.0
    return w, x


@pytest.mark.parametrize(("rows", "n_bit", "tokens"), [(256, 3, 4096), (192, 4, 2048), (64, 2, 1024)])
def test_the_search_is_the_official_auto_clip_layer(rows: int, n_bit: int, tokens: int) -> None:
    official = _official_functions()
    w, x = _layer_inputs(rows, 512, tokens, seed=rows + n_bit)
    q_config = {"zero_point": True, "q_group_size": 128}
    reference = official["auto_clip_layer"](w.clone(), x.clone(), n_bit, q_config)
    ours = clip_search(w, x, lambda t: official["pseudo_quantize_tensor"](t, n_bit=n_bit, **q_config), 128)
    assert ours.shape == reference.shape == (rows, 4, 1)
    assert torch.equal(ours, reference)
    shrunk = (ours < w.reshape(rows, 4, 128).abs().amax(dim=-1, keepdim=True)).float().mean().item()
    assert shrunk > 0, shrunk  # the search clipped some groups


def test_a_small_calibration_set_keeps_every_token() -> None:
    w, x = _layer_inputs(64, 256, 100, seed=3)
    spec = load_recipe({"name": "q", "defaults": {"weight": INT3}}).defaults.weight

    def quantize(t: torch.Tensor) -> torch.Tensor:
        return fake_quant(t, spec, backend="reference")

    # 100 tokens < n_sample_token: stride 1 (the official stride 100 // 512 = 0 would not run)
    assert torch.equal(clip_search(w, x, quantize, 128), clip_search(w, x, quantize, 128, n_sample_token=100))
    with pytest.raises(ValueError, match="multiple of the group size"):
        clip_search(w[:, :200], x[:, :200], quantize, 128)


def _awq_layer(name: str, weight_algo: str, w: torch.Tensor, x: torch.Tensor) -> EmuLinear:
    base = nn.Linear(w.shape[1], w.shape[0], bias=False)
    base.weight.data.copy_(w)
    spec = load_recipe({"name": "c", "defaults": {"weight": INT3, "activation": None,
                                                  "transform": {"kind": "awq", "share_inputs": False},
                                                  "weight_algo": weight_algo}}).defaults
    layer = EmuLinear(base, spec, name, backend="reference")
    layer.begin_calibration()
    for chunk in x.split(512):
        layer(chunk)
    layer.finish_calibration()
    return layer


def test_emulinear_fits_the_clip_on_the_awq_transformed_weight_and_applies_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import tricast.nn.linear as linear

    calls = []

    def spy(weight: torch.Tensor, inputs: torch.Tensor, quantize, group: int, **kwargs) -> torch.Tensor:
        calls.append((weight.clone(), inputs.clone(), quantize, group, kwargs))
        return clip_search(weight, inputs, quantize, group, **kwargs)

    monkeypatch.setattr(linear, "clip_search", spy)
    w, x = _layer_inputs(128, 256, 2048, seed=5)
    clipped = _awq_layer("model.layers.0.mlp.down_proj", "awq_clip", w, x)
    plain = _awq_layer("model.layers.0.mlp.down_proj", "rtn", w, x)
    assert clipped.needs_calibration and isinstance(clipped.spec.weight_algo, ClipSearchSpec)
    assert torch.equal(clipped.transform.diag, plain.transform.diag)  # the AWQ scales are the same search
    assert plain._clip is None and clipped._clip is not None and clipped._clip.shape == (128, 2, 1)
    # the search ran on what AWQ quantizes: W diag(s) and the reservoir samples x / s (all 2048 rows here,
    # in the reservoir's order), with the layer's weight quantizer and group, at the official settings
    ((weight, inputs, quantize, group, kwargs),) = calls
    transform = clipped.transform
    assert torch.equal(weight, transform.apply_weight(w).float()) and group == 128
    expected = transform.apply_activation(x).float()
    assert inputs.shape == expected.shape
    for column in (0, 1, 255):
        assert torch.equal(inputs[:, column].sort().values, expected[:, column].sort().values)
    probe = torch.randn(4, 256)
    assert torch.equal(quantize(probe), fake_quant(probe, clipped.spec.weight, backend="reference"))
    assert kwargs == {"n_grid": 20, "max_shrink": 0.5, "n_sample_token": 512}
    with torch.no_grad():
        assert not torch.equal(clipped(x[:64]), plain(x[:64]))  # the clip reaches the quantized weight
    for skipped in ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"):
        layer = _awq_layer(skipped, "awq_clip", w, x)
        assert clip_skipped(skipped) and layer._clip is None
        with torch.no_grad():
            assert torch.equal(layer(x[:64]), plain(x[:64]))  # AWQ does not clip q / k: plain RTN
    assert not clip_skipped("model.layers.0.self_attn.v_proj") and not clip_skipped("mlp.up_proj")


def test_the_calibration_cache_state_keeps_the_clip() -> None:
    w, x = _layer_inputs(128, 256, 2048, seed=7)
    layer = _awq_layer("model.layers.1.self_attn.o_proj", "awq_clip", w, x)
    state = _layer_state(layer)
    assert torch.equal(state["clip"], layer._clip.cpu())
    base = nn.Linear(256, 128, bias=False)
    base.weight.data.copy_(w)
    fresh = EmuLinear(base, layer.spec, layer.name, backend="reference")
    _restore_layer(fresh, state)
    assert torch.equal(fresh._clip, layer._clip)
    fresh.requantize()  # a rebuilt operand keeps the clip
    with torch.no_grad():
        assert torch.equal(fresh(x[:64]), layer(x[:64]))


def test_the_recipe_spec_checks_and_schema() -> None:
    validator = Draft202012Validator(SCHEMA)
    for given in ("awq_clip", {"kind": "awq_clip", "n_grid": 10, "max_shrink": 0.3, "n_sample_token": 256}):
        data = {"name": "c", "defaults": {"weight": INT3, "weight_algo": given}}
        validator.validate(data)
        recipe = load_recipe(data)
        assert type(recipe.defaults.weight_algo) is ClipSearchSpec and recipe.needs_calibration
    assert load_recipe({"name": "c", "defaults": {"weight": INT3, "weight_algo": {
        "kind": "awq_clip", "n_grid": 10}}}).sha256 != load_recipe(
        {"name": "c", "defaults": {"weight": INT3, "weight_algo": "awq_clip"}}).sha256
    for defaults, words in (({"weight": None, "weight_algo": "awq_clip"}, "requires a weight Quant"),
                            ({"weight": "fp8_block128", "weight_algo": "awq_clip"}, "group or row weight"),
                            ({"weight": INT3, "weight_algo": "awq_clip",
                              "sparsity": {"kind": "n:m", "n": 2, "m": 4}}, "sparsity: not supported"),
                            ({"weight": INT3, "weight_algo": {"kind": "awq_clip", "max_shrink": 1.0}},
                             "max_shrink"),  # refused by the schema (and ClipSearchSpec)
                            ({"weight": INT3, "weight_algo": {"kind": "awq_clip", "n_grid": 0}},
                             "n_grid")):
        with pytest.raises(ValueError, match=words):
            load_recipe({"name": "bad", "defaults": defaults})
    assert not list(validator.iter_errors({"name": "c", "defaults": {"weight_algo": {"kind": "awq_clip",
                                                                                       "n_grid": 3}}}))
    assert list(validator.iter_errors({"name": "c", "defaults": {"weight_algo": {"kind": "awq_clip",
                                                                                   "max_shrink": 1}}}))
