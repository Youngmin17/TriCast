"""Official GPTQ (IST-DASLab/gptq ``llama.py: llama_sequential``, pinned) called unchanged on
transformers>=4.48 decoders: firing evidence, quantization levels and groups, the official data path, the
recipe block and the eval-runner hook-in. Tiny models from configs; no network.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
pytest.importorskip("datasets")  # official datautils imports it

from tests.test_official_prune import (  # noqa: E402
    KINDS,
    LINEARS,
    NSAMPLES,
    SEQLEN,
    _linears,
    _loader,
    _Rows,
    _tiny,
    _tokenizer,
)
from tricast.official import gptq as official_gptq  # noqa: E402
from tricast.official import prune as official_prune  # noqa: E402
from tricast.official import repos  # noqa: E402
from tricast.recipe import OFFICIAL_GPTQ_DEFAULTS, load_recipe  # noqa: E402

PASSES = 5  # true_sequential: 4 sub-layer groups + the final pass, each over every calibration sample


@pytest.fixture(autouse=True)
def _cpu_synchronize(monkeypatch):
    if not torch.cuda.is_available():  # official fasterquant calls torch.cuda.synchronize(): no-op on CPU
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)


def _distinct_per_group(weight: torch.Tensor, width: int) -> int:
    most = 0
    for start in range(0, weight.shape[1], width):
        values = weight[:, start:start + width].sort(dim=1).values
        most = max(most, int(((values[:, 1:] != values[:, :-1]).sum(dim=1) + 1).max()))
    return most


@pytest.mark.parametrize("kind", KINDS)
def test_official_gptq_quantizes_every_decoder_linear_once(kind, monkeypatch):
    model = _tiny(kind)
    original = [weight.clone() for weight in _linears(model)]
    gptq_class = repos.load_module("gptq", "gptq").GPTQ
    assert repos.load_module("gptq", "llama").GPTQ is gptq_class  # llama.py's `from gptq import *`
    solves = {"calls": 0}
    fasterquant = gptq_class.fasterquant

    def counted(self, *args, **kwargs):
        solves["calls"] += 1
        return fasterquant(self, *args, **kwargs)

    monkeypatch.setattr(gptq_class, "fasterquant", counted)
    report = official_gptq.quantize(model, None, 4, seqlen=SEQLEN, nsamples=NSAMPLES, loader=_loader())
    assert solves["calls"] == report["fasterquant_calls"] == 2 * LINEARS
    # every official layer call carried the position_ids the official Catcher captured
    calls = 2 * NSAMPLES * PASSES
    assert report["official_layer_calls"] == {"total": calls, "with_position_ids": calls}
    for before, weight in zip(original, _linears(model), strict=True):
        assert not torch.equal(before, weight)
        assert _distinct_per_group(weight, weight.shape[1]) <= 16  # 4-bit grid per row (groupsize -1)
    assert len(report["layers"]) == 2 * LINEARS
    assert all(layer["max_distinct"] <= 16 and layer["maxq"] == 15 for layer in report["layers"])
    for layer in report["layers"]:  # official Quantizer: one (scale, zero) per output row
        rows = model.get_submodule(layer["name"]).out_features
        assert layer["scale_shape"] == layer["zero_shape"] == [rows, 1]
    assert report["official_commit"] == repos.REPOS["gptq"].commit
    assert report["official"]["function"] == "llama.llama_sequential"
    assert report["args"]["wbits"] == 4 and report["args"]["true_sequential"] is True
    mappings = [entry for entry in report["deviations"] if isinstance(entry, dict)]
    assert [{key: entry[key] for key in ("param", "official", "used")} for entry in mappings] == [
        {"param": "seqlen", "official": 2048, "used": SEQLEN}]
    assert type(model.model.layers) is nn.ModuleList and "seqlen" not in vars(model)
    assert not hasattr(repos.load_module("gptq", "llama"), "args")
    assert gptq_class.fasterquant is counted  # the report's counter was removed again


@pytest.mark.parametrize(("act_order", "static_groups"), [(False, False), (True, True)])
def test_groupsize_gives_each_group_its_own_grid(act_order, static_groups):
    model = _tiny("llama")
    report = official_gptq.quantize(model, None, 2, groupsize=16, act_order=act_order,
                                    static_groups=static_groups, seqlen=SEQLEN, nsamples=NSAMPLES,
                                    loader=_loader())
    weights = _linears(model)
    assert all(_distinct_per_group(weight, 16) <= 4 for weight in weights)  # 2-bit grid per 16-column group
    assert max(_distinct_per_group(weight, weight.shape[1]) for weight in weights) > 4  # not one grid per row
    assert report["groups"] == "contiguous"
    assert all(layer["max_distinct"] <= 4 for layer in report["layers"])


def test_default_data_is_official_get_loaders(tmp_path, monkeypatch):
    import datasets

    _tokenizer().save_pretrained(tmp_path)  # official datautils loads the tokenizer itself, by model name
    texts = [" ".join(f"t{(row * 7 + column) % 252}" for column in range(80)) for row in range(10)]
    calls = []

    def load_dataset(path, name=None, **kwargs):
        if name == "allenai--c4":
            raise ValueError("BuilderConfig 'allenai--c4' not found.")  # datasets>=2.x on the hub repo
        calls.append((path, name, kwargs.get("split")))
        return {"text": texts} if path == "wikitext" else _Rows(texts)

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    for dataset, expected in (("wikitext2", [("wikitext", "wikitext-2-raw-v1", "train"),
                                             ("wikitext", "wikitext-2-raw-v1", "test")]),
                              # the c4 shim drops the config name (tricast.official.prune._c4_config_shim)
                              ("c4", [("allenai/c4", None, "train"), ("allenai/c4", None, "validation")])):
        calls.clear()
        model = _tiny("llama")
        model.config._name_or_path = str(tmp_path)
        report = official_gptq.quantize(model, None, 4, seqlen=SEQLEN, nsamples=NSAMPLES, dataset=dataset)
        assert calls == expected
        assert report["data"]["source"] == f"official datautils.get_loaders({dataset!r})"
        assert report["fasterquant_calls"] == 2 * LINEARS
        assert any("allenai--c4" in entry for entry in report["deviations"] if isinstance(entry, str)) == (
            dataset == "c4")
    assert datasets.load_dataset is load_dataset


def test_official_gptq_recipe_block():
    block = {"wbits": 4, "groupsize": 128}
    plain = {"name": "plain", "defaults": {}}
    recipe = load_recipe({**plain, "official_gptq": block})
    assert recipe.official_gptq == {**OFFICIAL_GPTQ_DEFAULTS, **block}
    assert load_recipe(recipe.to_dict()).sha256 == recipe.sha256 != load_recipe(plain).sha256
    # recorded before official_gptq existed: recipes without the block keep their hashes
    assert load_recipe(plain).sha256 == "0528c01b61a81889f5afc6bfdab8a4a9ad71da600865b56b724177dcd19bb36c"
    assert load_recipe("bf16_passthrough").sha256 == (
        "6ffc7a26396a544bfb67c9e93932c30644a5a40abaf079d65ebe7c8279502d64")
    assert load_recipe("w4a16_g128_zp_gptq").sha256 == (
        "637ecdc4d0e90f770340796dd1dfde39b54805648318d8cddaa1e455ee464ef5")
    assert "official_gptq" not in load_recipe(plain).to_dict()
    mapped = load_recipe({**plain, "official_gptq": {**block, "seqlen": 4096}})
    assert mapped.official_gptq["seqlen"] == 4096 and load_recipe(mapped.to_dict()).sha256 == mapped.sha256
    with pytest.raises(ValueError, match="two GPTQ paths"):
        load_recipe({"name": "x", "defaults": {"weight": "int4_g128_zp", "weight_algo": "gptq"},
                     "official_gptq": block})
    with pytest.raises(ValueError, match="quantize them twice"):
        load_recipe({"name": "x", "defaults": {"weight": "int4_g128_zp"}, "official_gptq": block})
    with pytest.raises(ValueError, match="quantize them twice"):
        load_recipe({**plain, "overrides": [{"match": "*q_proj", "weight": "int4_g128_zp"}],
                     "official_gptq": block})
    with pytest.raises(ValueError, match=r"official_gptq\.wbits"):
        load_recipe({**plain, "official_gptq": {"wbits": 16}})
    with pytest.raises(ValueError, match=r"official_gptq\.static_groups"):
        load_recipe({**plain, "official_gptq": {"wbits": 4, "static_groups": True}})
    with pytest.raises(ValueError, match="official_gptq"):
        load_recipe({**plain, "official_gptq": {**block, "blocksize": 64}})  # unknown key
    from tricast.nn.patch import patch_model

    with pytest.raises(ValueError, match="unquantized weights"):
        patch_model(_tiny("llama"), recipe)


def test_runner_prunes_then_quantizes_then_patches_and_reloads(tmp_path, monkeypatch):
    runner = pytest.importorskip("tricast.eval.runner")
    from tricast.eval.envinfo import capture_env
    from tricast.nn import patch

    model, tokenizer = _tiny("llama"), _tokenizer()
    loads = []
    monkeypatch.setattr(runner, "_load_model", lambda *args: loads.append(args) or (model, tokenizer))
    env = capture_env(None, {"model_id": "offline/tiny", "model_sha": "model-commit", "git_sha": "a" * 40,
                             "git_dirty": False, "src_sha256": None, "seed": 42, "dtype": "fp32",
                             "device": "cpu"})
    monkeypatch.setattr(runner, "capture_env", lambda *args: env.copy())
    order = []
    prune, quantize, patch_model = official_prune.prune, official_gptq.quantize, patch.patch_model

    def zeros(target):
        return float((target.model.layers[0].self_attn.q_proj.weight == 0).float().mean())

    def prune_spy(target, *args, **kwargs):
        order.append("prune")
        return prune(target, *args, loader=_loader(), **kwargs)

    def quantize_spy(target, *args, **kwargs):
        order.append(("official_gptq", zeros(target)))  # sees the pruned weights
        return quantize(target, *args, loader=_loader(), **kwargs)

    def patch_spy(target, recipe, **kwargs):
        order.append("patch")
        return patch_model(target, recipe, **kwargs)

    monkeypatch.setattr(official_prune, "prune", prune_spy)
    monkeypatch.setattr(official_gptq, "quantize", quantize_spy)
    monkeypatch.setattr(patch, "patch_model", patch_spy)
    linear = {"defaults": {"mma": {"preset": "fp64", "out_format": "fp32"}},
              "include": ["model.layers.0.self_attn.q_proj"]}
    both = {"name": "pruned_gptq", **linear, "prune": {"method": "wanda", "ratio": 0.5, "nsamples": NSAMPLES},
            "official_gptq": {"wbits": 4, "nsamples": NSAMPLES, "seqlen": SEQLEN}}
    cfg = {"model": "offline/tiny", "dtype": "fp32", "device": "cpu", "seed": 42,
           "recipes": [both, {"name": "plain", **linear}],
           "tasks": {"ppl": {"texts": ["t0 t1 t2 t3"], "seqlen": 4}}, "output_dir": str(tmp_path)}
    results = runner.run_config(cfg)
    assert [result["status"] for result in results] == ["complete", "complete"]
    assert order == ["prune", ("official_gptq", 0.5), "patch", "patch"]
    assert results[0]["official_gptq"]["fasterquant_calls"] == 2 * LINEARS
    assert "official_gptq" not in results[1]
    assert len(loads) == 2  # the model was reloaded after the run that quantized it
