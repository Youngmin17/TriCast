"""Recipe contracts, schema vocabulary, and bundled design-space examples."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import get_args, get_type_hints

import pytest
import yaml
from jsonschema import Draft202012Validator

from tricast.formats import Special
from tricast.mma.spec import PRESETS, Algorithm, MMASpec, ScaleApply
from tricast.quant.spec import (
    SCHEMES,
    Granularity,
    MMAInput,
    ObserverKind,
    ObserverSpec,
    ScaleMethod,
    TransformKind,
    WeightAlgo,
    ZeroPoint,
)
from tricast.recipe import CALIBRATION_DEFAULTS, list_recipes, load_recipe
from tricast.rounding import Rounding

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "src/tricast/schemas/recipe.schema.json").read_text())
RECIPE_PATHS = sorted((ROOT / "src/tricast/recipes").glob("*.yaml"))


@pytest.mark.parametrize("path", RECIPE_PATHS, ids=lambda p: p.stem)
def test_bundled_recipe_schema_and_roundtrip(path):
    data = yaml.safe_load(path.read_text())
    Draft202012Validator(SCHEMA).validate(data)
    recipe = load_recipe(path)
    assert recipe.name == path.stem
    assert load_recipe(path.stem).to_dict() == recipe.to_dict()
    assert load_recipe(recipe.to_dict()).sha256 == recipe.sha256
    assert recipe.spec_for("lm_head") is None
    first_spec = recipe.spec_for("model.layers.0.self_attn.q_proj", num_layers=3)
    assert (first_spec is None) == (recipe.name == "mixed_first_last_bf16" or recipe.include == [])


def test_schema_valid_and_covers_required_recipes():
    Draft202012Validator.check_schema(SCHEMA)
    assert len(RECIPE_PATHS) >= 20
    assert SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_shorthand():
    short = {"name": "short", "defaults": {"weight": "nvfp4", "activation": "fp8_tensor",
             "mma": "nvidia_hopper_fp8", "transform": "hadamard", "weight_algo": "gptq"}}
    explicit = copy.deepcopy(short)
    for key, selector in (("weight", "scheme"), ("activation", "scheme"), ("mma", "preset"),
                          ("transform", "kind"), ("weight_algo", "kind")):
        explicit["defaults"][key] = {selector: short["defaults"][key]}
    assert load_recipe(short).to_dict() == load_recipe(explicit).to_dict()
    assert short["defaults"]["weight"] == "nvfp4"


def test_include_exclude_and_first_override_deep_merge():
    recipe = load_recipe({
        "name": "match", "include": ["model.*"], "exclude": ["*.skip"],
        "defaults": {"weight": {"scheme": "fp8_tensor", "scale": {"method": "mse", "mse_grid": 7}},
                     "mma": {"preset": "nvidia_hopper_fp8", "out_format": "fp32"}},
        "overrides": [
            {"match": "*.down_proj", "weight": {"scale": {"search": [1.0, 1.5]}},
             "mma": {"f_bits": 7}},
            {"match": "*", "weight": "nvfp4"},
        ],
    })
    spec = recipe.spec_for("model.mlp.down_proj")
    assert spec.weight.scale.method == "mse"
    assert spec.weight.scale.mse_grid == 7
    assert spec.weight.scale.search == (1.0, 1.5)
    assert spec.mma.f_bits == 7
    assert spec.mma.chunk_size == 32
    assert spec.mma.out_format.name == "fp32"
    assert recipe.spec_for("model.mlp.up_proj").weight == SCHEMES["nvfp4"]
    assert recipe.spec_for("outside") is None
    assert recipe.spec_for("model.skip") is None
    assert recipe.defaults.weight.scale.search == ()
    assert recipe.defaults.mma.f_bits == 13


def test_new_preset_uses_its_defaults():
    recipe = load_recipe({"name": "presets", "defaults": {"mma": "nvidia_hopper_fp8"},
                          "overrides": [{"match": "*", "mma": "nvidia_ada_fp8"}]})
    assert recipe.spec_for("layer").mma.chunk_size == 16


@pytest.mark.parametrize("changes,expected", [
    ({"weight": "nvfp4"}, False),
    ({"activation": "fp8_tensor_ema"}, True),
    ({"activation": "fp8_tensor_delayed"}, False),
    ({"transform": "smoothquant"}, True),
    ({"transform": "awq"}, True),
    ({"transform": "random_hadamard"}, False),
    ({"weight_algo": "gptq"}, True),
])
def test_calibration_requirement(changes, expected):
    assert load_recipe({"name": "calib", "defaults": changes}).needs_calibration is expected
    recipe = load_recipe({"name": "calib", "defaults": {}, "overrides": [{"match": "*", **changes}]})
    assert recipe.needs_calibration is expected


@pytest.mark.parametrize("data,path", [
    ({"defaults": {"weight": {"scheme": "unknown"}}}, "defaults.weight.scheme"),
    ({"defaults": {"weight": {"format": "not-a-format"}}}, "defaults.weight.format"),
    ({"defaults": {"weight": {"scheme": "fp8_tensor", "sr_bits": 0}}}, "defaults.weight.sr_bits"),
    ({"defaults": {"mma": {"algorithm": "wrong"}}}, "defaults.mma.algorithm"),
    ({"defaults": {"transform": {"alpha": 2}}}, "defaults.transform.alpha"),
    ({"defaults": {}, "overrides": [{"match": "x"}, {"match": "y", "weight":
       {"scheme": "nvfp4", "scale": {"method": "wrong"}}}]}, "overrides[1].weight.scale.method"),
    ({"defaults": {"weight": {"scheme": "fp8_tensor", "scale": {"format": "wrong"}}}},
     "defaults.weight.scale.format"),
    ({"defaults": {}, "backend": "cuda"}, "backend"),
])
def test_errors_include_recipe_path(data, path):
    with pytest.raises(ValueError) as error:
        load_recipe({"name": "invalid", **data})
    assert path in str(error.value)


@pytest.mark.parametrize("key,values", [
    ("granularity_kind", get_args(Granularity)), ("scale_method", get_args(ScaleMethod)),
    ("zero_point", get_args(ZeroPoint)), ("mma_input", get_args(MMAInput)),
    ("observer_kind", get_args(ObserverKind)), ("transform_kind", get_args(TransformKind)),
    ("weight_algorithm", get_args(WeightAlgo)), ("algorithm", get_args(Algorithm)),
    ("scale_apply", get_args(ScaleApply)), ("rounding", [r.value for r in Rounding]),
    ("scheme_name", SCHEMES), ("preset_name", PRESETS),
])
def test_schema_enums_match_specs(key, values):
    assert set(SCHEMA["$defs"][key]["enum"]) == set(values)



@pytest.mark.parametrize("definition,field,values", [
    ("float_format", "special", get_args(Special)),
    ("mma_object", "c_mode", get_args(get_type_hints(MMASpec)["c_mode"])),
    ("mma_object", "norm_rounding", get_args(get_type_hints(MMASpec)["norm_rounding"])),
    ("observer", "reduce", get_args(get_type_hints(ObserverSpec)["reduce"])),
])
def test_nested_schema_enums_match_specs(definition, field, values):
    assert set(SCHEMA["$defs"][definition]["properties"][field]["enum"]) == set(values)


def test_hash_order_independent_and_sensitive():
    a = load_recipe({"name": "hash", "defaults": {"weight": "fp8_tensor", "mma": "fp64"}})
    b = load_recipe({"defaults": {"mma": "fp64", "weight": "fp8_tensor"}, "name": "hash"})
    assert a.sha256 == b.sha256
    assert len(a.sha256) == 64
    changed = a.to_dict()
    changed["defaults"]["weight"]["rounding"] = "rtz"
    assert a.sha256 != load_recipe(changed).sha256
    assert load_recipe(a).sha256 == a.sha256


def test_explicit_format_and_granularity_roundtrip():
    recipe = load_recipe({"name": "custom", "defaults": {"weight": {
        "format": {"name": "custom_e3m4", "kind": "float", "ebits": 3, "mbits": 4, "special": "none"},
        "granularity": "block:3x5", "scale": {"format": "fp32"}}}})
    assert recipe.defaults.weight.block == (3, 5)
    assert load_recipe(recipe.to_dict()).to_dict() == recipe.to_dict()
    bfp = load_recipe({"name": "bfp", "defaults": {"weight": "bfp5_b16"}})
    assert load_recipe(bfp.to_dict()).sha256 == bfp.sha256


def test_json_path_and_lookup_away_from_repository(tmp_path, monkeypatch):
    recipe = load_recipe("fp64_reference")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe.to_dict()))
    monkeypatch.chdir(tmp_path)
    assert load_recipe(path).sha256 == recipe.sha256
    assert load_recipe("fp64_reference").sha256 == recipe.sha256
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_recipe(tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError, match="unknown recipe"):
        load_recipe("missing_recipe")


def test_invalid_root_and_unknown_fields():
    with pytest.raises(ValueError, match="additional properties|Additional properties"):
        load_recipe({"name": "bad", "defaults": {}, "extra": 1})
    with pytest.raises(ValueError, match="defaults.weight"):
        load_recipe({"name": "bad", "defaults": {"weight": {}}})


@pytest.mark.parametrize("name,count", [("fp8_cofda_f_sweep", 18), ("nvfp4_gdfs_sweep", 16)])
def test_sweep_expansion(name, count):
    runner = pytest.importorskip("tricast.eval.runner")
    cfg = yaml.safe_load((ROOT / "configs/sweeps" / f"{name}.yaml").read_text())
    recipes = runner.expand_sweep(cfg)
    assert len(recipes) == count
    assert len({recipe.name for recipe in recipes}) == count
    assert len({recipe.sha256 for recipe in recipes}) == count
    if name == "fp8_cofda_f_sweep":
        assert {r.defaults.mma.f_bits for r in recipes} == {25, 21, 17, 13, 11, 9, 7, 5, 3}
        assert {r.defaults.mma.c_mode for r in recipes} == {"fused", "decoupled"}
        assert {r.defaults.mma.chunk_size for r in recipes} == {32}
    else:
        assert {r.defaults.mma.g_bits for r in recipes} == {6, 5, 4, 3}
        assert {r.defaults.mma.f_bits for r in recipes} == {35, 25, 13, 10}


def test_partial_mma_override_clears_hardware_identity():
    recipe = load_recipe({
        "name": "custom_accumulator", "defaults": {"mma": "nvidia_hopper_fp8"},
        "overrides": [{"match": "*.q_proj", "mma": {"f_bits": 7}}],
    })
    modified = recipe.spec_for("model.layers.0.self_attn.q_proj").mma
    assert modified.f_bits == 7
    assert modified.chunk_size == 32
    assert modified.name == modified.provenance == ""
    original = recipe.spec_for("model.layers.0.self_attn.k_proj").mma
    assert original.name == "nvidia_hopper_fp8"
    assert original.provenance
    assert load_recipe(recipe.to_dict()).spec_for("model.layers.0.self_attn.q_proj").mma == modified


@pytest.mark.parametrize("name,expected", [
    ("model.layers.0.self_attn.q_proj", "fp8_tensor"),
    ("model.layers.3.self_attn.k_proj", "fp8_tensor"),
    ("model.layers.27.self_attn.v_proj", "fp8_tensor"),
    ("model.layers.4.self_attn.q_proj", "bf16"),
    ("model.layers.0.self_attn.o_proj", "bf16"),
    ("model.layers.0.mlp.q_proj", "bf16"),
    ("model.q_proj", "bf16"),
    ("layers.1.self_attn.q_proj", "fp8_tensor"),
])
def test_layer_selectors_are_conjunctive(name, expected):
    recipe = load_recipe({
        "name": "selectors", "defaults": {"weight": "bf16"},
        "overrides": [{"match": "*.self_attn.*", "layers": "0-3,27",
                       "modules": ["q_proj", "k_proj", "v_proj"], "weight": "fp8_tensor"}],
    })
    assert recipe.spec_for(name).weight == SCHEMES[expected]
    assert load_recipe(recipe.to_dict()).spec_for(name).weight == SCHEMES[expected]


def test_skip_first_match_and_skip_reasons():
    recipe = load_recipe({
        "name": "skip", "include": ["model.*"], "exclude": ["*.lm_head"],
        "defaults": {"weight": "mxfp4"},
        "overrides": [{"layers": "0", "skip": True}, {"modules": ["q_proj"], "weight": "bf16"}],
    })
    assert recipe.spec_for("model.layers.0.self_attn.q_proj") is None
    assert recipe.skip_reason("model.layers.0.self_attn.q_proj") == "overrides[0].skip"
    assert recipe.spec_for("model.layers.1.self_attn.q_proj").weight == SCHEMES["bf16"]
    assert recipe.skip_reason("model.layers.1.self_attn.q_proj") is None
    assert recipe.skip_reason("outside") == "not included"
    assert recipe.skip_reason("model.lm_head") == "excluded"


@pytest.mark.parametrize("entry", [
    {"skip": True}, {"weight": "bf16"}, {"layers": "3-1"}, {"layers": ""},
    {"layers": "one"}, {"layers": "-2"}, {"modules": []},
])
def test_invalid_override_selectors_report_location(entry):
    with pytest.raises(ValueError, match=r"overrides\[0\]"):
        load_recipe({"name": "invalid", "defaults": {}, "overrides": [entry]})


def test_model_relative_first_and_last_recipe():
    recipe = load_recipe("mixed_first_last_bf16")
    for num_layers in (2, 4, 28):
        assert recipe.spec_for("model.layers.0.self_attn.q_proj", num_layers=num_layers) is None
        assert recipe.spec_for(f"model.layers.{num_layers - 1}.mlp.down_proj", num_layers=num_layers) is None
        for index in range(1, num_layers - 1):
            assert recipe.spec_for(f"model.layers.{index}.self_attn.q_proj",
                                   num_layers=num_layers).weight == SCHEMES["mxfp4"]
    with pytest.raises(ValueError, match="num_layers"):
        recipe.spec_for("model.layers.1.self_attn.q_proj")


@pytest.mark.parametrize("kv", [
    "kivi2",
    {"preset": "kivi4", "mode": "fakequant", "residual": 64, "layers": "0-3"},
    {"key": "kivi2", "value": None, "key_axis": "channel", "value_axis": "token",
     "residual": 32, "mode": "fakequant"},
])
def test_kv_configuration_roundtrip(kv):
    recipe = load_recipe({"name": "kv", "defaults": {"mma": "fp32_fma"}, "kv": kv})
    assert recipe.kv is not None
    assert not recipe.needs_calibration
    assert load_recipe(recipe.to_dict()).to_dict() == recipe.to_dict()
    assert load_recipe(recipe.to_dict()).sha256 == recipe.sha256
    if isinstance(kv, dict) and "layers" in kv:
        assert recipe.kv_layers == "0-3"
        assert recipe.kv.mode == "fakequant"
        assert recipe.kv.residual == 64


@pytest.mark.parametrize("kv", [
    "unknown", {"preset": "unknown"}, {"residual": 0},
    {"preset": "kivi2", "layers": "3-1"}, {"preset": "kivi2", "layers": "-1"},
])
def test_invalid_kv_reports_recipe_path(kv):
    with pytest.raises(ValueError, match="kv"):
        load_recipe({"name": "kv", "defaults": {}, "kv": kv})


def test_calibration_defaults_are_central_and_not_mutated():
    recipe = load_recipe({"name": "calib", "defaults": {}, "calibration": {"samples": 3}})
    assert CALIBRATION_DEFAULTS == {"dataset": "wikitext2", "split": "train", "samples": 128,
                                    "seqlen": 2048, "seed": 0, "sequential": False}
    assert recipe.calibration_options == {**CALIBRATION_DEFAULTS, "samples": 3}
    options = recipe.calibration_options
    options["samples"] = 9
    assert recipe.calibration_options["samples"] == 3
    assert CALIBRATION_DEFAULTS["samples"] == 128
    assert load_recipe("w4a16_gptq_sequential").calibration_options["sequential"] is True
    assert load_recipe("nvfp4_awq_shared").defaults.transform.share_inputs is True


def test_bundled_recipes_listed_from_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = list_recipes()
    assert names == sorted(path.stem for path in RECIPE_PATHS)
    assert {"kivi2_kv", "nvfp4_awq_shared", "w4a16_gptq_sequential",
            "mixed_first_last_bf16"}.issubset(names)
    assert all(load_recipe(name).name == name for name in names)
