"""app/quantizer_recipe.py, the Quantizer page's grid -> recipe generator, against tricast itself.

The page (app/web/js/quantizer.js) is a port of this module; their byte parity is checked in a browser by
scripts/quantizer/qz_check.mjs. Here: the presets are byte-identical to the YAML the recorded GPU runs used
and to their pins; for random grids, with and without per-method options, every decoder Linear resolves
through tricast's first-match overrides (Recipe.spec_for) to exactly the spec its method (variant) has alone;
every option value reaches the resolved spec; and what does not apply is refused instead of ignored.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr
from app.tests import quantizer_records as rec
from tricast.quant.spec import KV_PRESETS, ObserverSpec, ScaleSpec, TransformSpec, WeightAlgoSpec
from tricast.recipe import CALIBRATION_DEFAULTS, load_recipe

REPO = Path(qr.__file__).resolve().parents[1]
SCHEMA = json.loads((REPO / "src" / "tricast" / "schemas" / "recipe.schema.json").read_text(encoding="utf-8"))
DECODER = ("model.layers", qr.BLOCKS)
QUANT = [m for m in qr.METHODS if m != "bf16"]
ALONE = {m: load_recipe({"name": m, "defaults": dict(f)}).defaults for m, f in qr.METHODS.items() if f}


def spec_of(recipe, block: int, kind: str):
    return recipe.spec_for(qr.layer_name(block, kind), decoder=DECODER)


def field_value(spec, option: str):
    """The resolved LinearSpec's value for an option id (e.g. activation.observer.kind)."""
    target, *path = option.split(".")
    value = getattr(spec, target)
    if path == ["observer", "kind"] and value.observer is None:
        return "none"
    for part in path:
        value = getattr(value, part)
    return getattr(value, "value", value)  # Rounding -> "rne"


def random_grid(rng: random.Random, default: str) -> list[list[str]]:
    methods = list(qr.METHODS)
    grid = [
        [rng.choice(methods) if rng.random() < 0.3 else default for _ in qr.KINDS] for _ in range(qr.BLOCKS)
    ]
    for b in rng.sample(range(qr.BLOCKS), rng.randint(0, 3)):
        grid[b] = [rng.choice(methods)] * len(qr.KINDS)
    for k in rng.sample(range(len(qr.KINDS)), rng.randint(0, 2)):
        method = rng.choice(methods)
        for b in range(qr.BLOCKS):
            grid[b][k] = method
    return grid


FLOATS = (0, 0.0001, 0.01, 0.05, 0.25, 0.5, 0.9, 0.999, 1.0, 12.5, 50, 99.0, 99.99, 100.0)


def random_value(rng: random.Random, spec: dict):
    if spec["type"] == "choice":
        return rng.choice(spec["choices"])
    if spec["type"] == "bool":
        return rng.random() < 0.5
    if spec["type"] == "int":
        step = spec.get("multiple", 1)
        return rng.randrange(spec["min"], min(spec["max"], spec["min"] + 64 * step) + 1, step)
    while True:
        value = rng.choice(FLOATS) if rng.random() < 0.7 else round(rng.uniform(spec["min"], spec["max"]), 4)
        try:
            return qr.check_value("random", spec, value)
        except ValueError:
            continue


def random_options(rng: random.Random) -> dict:
    options = {}
    for method in QUANT:
        if rng.random() < 0.55:
            raw = {
                o: random_value(rng, qr.OPTIONS[o]) for o in qr.METHOD_OPTIONS[method] if rng.random() < 0.5
            }
            if method == "int8" and rng.random() < 0.6:  # reach the observer sub-fields often enough
                raw.update(
                    {
                        "activation.granularity": "tensor",
                        "activation.observer.kind": rng.choice(
                            qr.OPTIONS["activation.observer.kind"]["choices"]
                        ),
                    }
                )
            raw = {o: v for o, v in raw.items() if qr._applies(o, raw)}  # what the page's controls can hold
            if raw:
                options[method] = raw
    return options


# ---------- presets ----------
@pytest.mark.parametrize("preset_id", list(qr.PRESETS))
def test_presets_are_the_recorded_yaml_and_match_the_pins(preset_id: str) -> None:
    text = qr.to_yaml(qr.preset(preset_id))
    assert text + "\n" == rec.preset_yaml(preset_id)  # byte for byte what the GPU runs used
    yaml_pin, recipe_pin = ex.PINS[preset_id]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == yaml_pin
    recipe = load_recipe(yaml.safe_load(text))
    assert (
        recipe.sha256 == recipe_pin == load_recipe(qr.preset(preset_id)).sha256 == ex.preset_sha256(preset_id)
    )
    assert recipe.to_dict() == rec.fixture(preset_id)  # the normalised recipe a runner records
    assert ex.recipe_sha256(rec.fixture(preset_id)) == recipe_pin


def test_build_and_yaml_are_deterministic_and_round_trip() -> None:
    for preset_id in qr.PRESETS:
        first, second = qr.preset(preset_id), qr.preset(preset_id)
        assert first == second and qr.to_yaml(first) == qr.to_yaml(second)
        assert yaml.safe_load(qr.to_yaml(first)) == first


# ---------- grid -> recipe equivalence ----------
def test_random_grids_resolve_to_each_methods_own_spec() -> None:
    rng = random.Random(42)
    for t in range(80):
        default = rng.choice(QUANT)
        grid = random_grid(rng, default)
        data = qr.build("t", default, grid, rng.choice(list(qr.KV)))
        assert (
            yaml.safe_load(qr.to_yaml(data)) == data
            and qr.resolve(default, data.get("overrides", [])) == grid
        )
        recipe = load_recipe(data)
        for b in range(qr.BLOCKS):
            for k, kind in enumerate(qr.KINDS):
                method = grid[b][k]
                assert spec_of(recipe, b, kind) == (None if method == "bf16" else ALONE[method]), (
                    t,
                    b,
                    kind,
                    method,
                )


def test_random_option_states_resolve_to_each_variants_own_spec() -> None:
    """Grids with per-method options, calibration settings and KV options: each layer gets its
    method-variant's spec used alone, every chosen option value is in that spec, the KV residual and
    calibration settings are in the Recipe, and the calibration block is written exactly when a method needs
    it or a value was changed."""
    rng = random.Random(7)
    alone: dict[str, object] = {}
    seen: set[str] = set()
    for t in range(120):
        default = rng.choice(QUANT)
        grid, options, kv = random_grid(rng, default), random_options(rng), rng.choice(list(qr.KV))
        calibration = {
            k: random_value(rng, spec) for k, spec in qr.CALIBRATION.items() if rng.random() < 0.25
        }
        kv_options = {
            k: random_value(rng, spec) for k, spec in qr.KV_OPTIONS[kv].items() if rng.random() < 0.6
        }
        data = qr.build(
            f"o{t}", default, grid, kv, options=options, calibration=calibration, kv_options=kv_options
        )
        assert yaml.safe_load(qr.to_yaml(data)) == data
        recipe = load_recipe(data)
        chosen = {m: qr.method_options(m, o) for m, o in options.items()}
        for b in range(qr.BLOCKS):
            for k, kind in enumerate(qr.KINDS):
                method, spec = grid[b][k], spec_of(recipe, b, kind)
                if method == "bf16":
                    assert spec is None
                    continue
                key = json.dumps([method, chosen.get(method, {})], sort_keys=True)
                if key not in alone:
                    alone[key] = load_recipe(
                        {"name": "a", "defaults": qr.variant(method, chosen.get(method))}
                    ).defaults
                assert spec == alone[key], (t, b, kind, method, chosen.get(method))
                for option, value in chosen.get(method, {}).items():
                    assert field_value(spec, option) == value, (t, method, option)
                    seen.add(option)
        if kv != "none":
            residual = kv_options.get("residual", qr.KV_OPTIONS[kv]["residual"]["default"])
            assert recipe.kv.residual == residual
            seen.update({"kv.residual"} & {f"kv.{key}" for key in kv_options})
        used = {default, *(m for row in grid for m in row)}
        settings = qr.check_calibration(calibration)
        needed = any(qr.needs_calibration(m, chosen.get(m)) for m in used)
        changed = {key for key, value in settings.items() if value != qr.CALIBRATION[key]["default"]}
        assert ("calibration" in data) == (needed or bool(changed)) and recipe.needs_calibration == needed
        if "calibration" in data:
            assert all(recipe.calibration_options[key] == value for key, value in settings.items())
            seen.update(f"calibration.{key}" for key in changed)
    assert seen == {*qr.OPTIONS, "kv.residual", *(f"calibration.{key}" for key in qr.CALIBRATION)}, (
        "a random state never exercised some option"
    )


# ---------- options ----------
OPTION_VALUES = {  # option id -> (method, the options that make it apply, a non-default value)
    "weight.group_size": ("gptq_int4", {}, 64),
    "weight.scale.method": ("nvfp4", {}, "mse"),
    "weight.scale.percentile": ("int8", {"weight.scale.method": "percentile"}, 99.5),
    "weight.rounding": ("mxfp4", {}, "rtz"),
    "activation.granularity": ("int8", {}, "tensor"),
    "activation.scale.method": ("nvint4", {}, "percentile"),
    "activation.scale.percentile": ("nvfp4", {"activation.scale.method": "percentile"}, 99.0),
    "activation.rounding": ("bfp12", {}, "sr"),
    "activation.observer.kind": ("int8", {"activation.granularity": "tensor"}, "minmax"),
    "activation.observer.decay": (
        "int8",
        {"activation.granularity": "tensor", "activation.observer.kind": "ema"},
        0.9,
    ),
    "activation.observer.history_len": (
        "int8",
        {"activation.granularity": "tensor", "activation.observer.kind": "history"},
        4,
    ),
    "activation.observer.percentile": (
        "int8",
        {"activation.granularity": "tensor", "activation.observer.kind": "percentile"},
        99.9,
    ),
    "transform.alpha": ("smoothquant_nvfp4", {}, 0.8),
    "transform.grid": ("awq_nvfp4", {}, 10),
    "transform.share_inputs": ("awq_nvfp4", {}, False),
    "weight_algo.damp": ("gptq_int4", {}, 0.1),
    "weight_algo.act_order": ("gptq_int4", {}, True),
    "weight_algo.block_size": ("gptq_int4", {}, 64),
}


@pytest.mark.parametrize("option", list(qr.OPTIONS))
def test_every_option_field_reaches_the_resolved_spec(option: str) -> None:
    """Each option, set on a method used in one override cell over another default, is in that layer's
    resolved spec, and the default's layers keep their own spec."""
    method, enabling, value = OPTION_VALUES[option]
    default = "mxfp8" if method != "mxfp8" else "nvfp4"
    grid = qr.uniform(default)
    grid[5][2] = method
    recipe = load_recipe(qr.build("one", default, grid, options={method: {**enabling, option: value}}))
    assert field_value(spec_of(recipe, 5, "v_proj"), option) == value
    assert field_value(spec_of(recipe, 5, "v_proj"), option) != qr.OPTIONS[option]["default"]
    assert spec_of(recipe, 4, "v_proj") == ALONE[default]


def test_option_defaults_are_what_the_fragments_and_tricast_imply() -> None:
    assert qr.TRANSFORM_FIELDS == {f.name: f.default for f in fields(TransformSpec) if f.name != "kind"}
    assert qr.WEIGHT_ALGO_FIELDS == {f.name: f.default for f in fields(WeightAlgoSpec) if f.name != "kind"}
    observer = {f.name: f.default for f in fields(ObserverSpec)}
    for key in ("decay", "history_len", "percentile"):
        assert qr.OPTIONS[f"activation.observer.{key}"]["default"] == observer[key]
    for side in ("weight", "activation"):
        assert (
            qr.OPTIONS[f"{side}.scale.percentile"]["default"]
            == {f.name: f.default for f in fields(ScaleSpec)}["percentile"]
        )
        assert set(qr.OPTIONS[f"{side}.scale.method"]["choices"]) <= set(
            SCHEMA["$defs"]["scale_method"]["enum"]
        )
        assert set(qr.OPTIONS[f"{side}.rounding"]["choices"]) <= set(SCHEMA["$defs"]["rounding"]["enum"])
    kinds = set(qr.OPTIONS["activation.observer.kind"]["choices"]) - {"none"}
    assert kinds <= set(SCHEMA["$defs"]["observer_kind"]["enum"])
    assert set(qr.CALIBRATING_OBSERVERS) == kinds - {
        "history"
    }  # tricast: every observer but history calibrates
    assert (
        qr.CALIBRATION["dataset"]["choices"]
        == SCHEMA["$defs"]["calibration"]["properties"]["dataset"]["enum"]
    )
    for key, spec in qr.CALIBRATION.items():
        assert spec["default"] == (42 if key == "seed" else CALIBRATION_DEFAULTS[key]), key
    for kv, specs in qr.KV_OPTIONS.items():
        for key, spec in specs.items():
            assert spec["default"] == getattr(KV_PRESETS[kv], key), (kv, key)
    for (
        method,
        options,
    ) in qr.METHOD_OPTIONS.items():  # a default emits nothing because it is already the value
        for option in options:
            if qr._applies(option, {}):
                assert field_value(ALONE[method], option) == qr.OPTIONS[option]["default"], (method, option)


def test_nvint4_custom_format_leaks_into_no_other_layer() -> None:
    """NVINT4 is the nvfp4 scheme with its element format replaced; a scheme mapping replaces the previous
    spec whole, so neither the int4 elements nor NVFP4's fields cross between NVINT4 and other layers."""
    nvint4 = ALONE["nvint4"]
    assert nvint4.weight.format.name == "int4" and nvint4.weight.group_size == 16
    assert nvint4.weight.scale.format.name == "ue4m3" and nvint4.weight.scale.two_level
    assert nvint4.weight == ALONE["nvfp4"].weight.with_(format=nvint4.weight.format)
    for default, other in (("nvint4", "nvfp4"), ("nvfp4", "nvint4"), ("nvint4", "mxfp4"), ("int8", "nvint4")):
        grid = qr.uniform(default)
        grid[3] = [other] * len(qr.KINDS)
        for b in range(qr.BLOCKS):
            grid[b][qr.KINDS.index("down_proj")] = other
        grid[9][0] = "bf16"
        recipe = load_recipe(qr.build("leak", default, grid))
        for b in range(qr.BLOCKS):
            for k, kind in enumerate(qr.KINDS):
                method = grid[b][k]
                assert spec_of(recipe, b, kind) == (None if method == "bf16" else ALONE[method]), (
                    default,
                    b,
                    kind,
                )


def test_transform_and_gptq_options_do_not_leak_through_the_field_merge() -> None:
    """tricast merges an override's transform / weight_algo field by field onto the defaults: when the default
    method carries such options, every rule writes those keys completely."""
    grid = qr.uniform("awq_nvfp4")
    for b in range(qr.BLOCKS):
        grid[b][qr.KINDS.index("o_proj")] = "nvfp4"
        grid[b][qr.KINDS.index("down_proj")] = "smoothquant_nvfp4"
    grid[0] = ["gptq_int4"] * len(qr.KINDS)
    options = {"awq_nvfp4": {"transform.grid": 2, "transform.share_inputs": False}}
    data = qr.build("leak", "awq_nvfp4", grid, options=options)
    complete = {"kind", *qr.TRANSFORM_FIELDS}
    assert set(data["defaults"]["transform"]) == complete
    assert all(set(rule["transform"]) == complete for rule in data["overrides"])
    recipe = load_recipe(data)
    assert (
        spec_of(recipe, 5, "o_proj") == ALONE["nvfp4"] and spec_of(recipe, 5, "o_proj").transform.grid == 20
    )
    assert spec_of(recipe, 5, "down_proj") == ALONE["smoothquant_nvfp4"]
    assert spec_of(recipe, 0, "q_proj") == ALONE["gptq_int4"]
    assert spec_of(recipe, 5, "q_proj").transform == TransformSpec("awq", grid=2, share_inputs=False)
    gptq = qr.build(
        "leak", "gptq_int4", qr.uniform("mxfp4"), options={"gptq_int4": {"weight_algo.damp": 0.1}}
    )
    assert spec_of(load_recipe(gptq), 1, "q_proj") == ALONE["mxfp4"]  # damp does not reach RTN layers


@pytest.mark.parametrize(
    "case",
    [
        {"options": {"nvfp4": {"weight.group_size": 64}}},  # NVFP4's block 16 is fixed by the format
        {"options": {"mxfp4": {"weight.scale.method": "mse"}}},  # E8M0 scales: power-of-two only
        {"options": {"gptq_int4": {"weight.scale.method": "mse"}}},  # zero point: range from [min, max]
        {"options": {"nvfp4": {"weight.scale.percentile": 99.0}}},  # needs scale.method percentile
        {"options": {"int8": {"activation.observer.kind": "ema"}}},  # static observers need one tensor scale
        {
            "options": {
                "int8": {
                    "activation.granularity": "tensor",
                    "activation.observer.kind": "ema",
                    "activation.scale.method": "mse",
                }
            }
        },  # the observer sets amax: mse would be ignored
        {
            "options": {
                "int8": {
                    "activation.granularity": "tensor",
                    "activation.observer.kind": "ema",
                    "activation.observer.decay": 1.0,
                }
            }
        },  # decay < 1
        {"options": {"smoothquant_nvfp4": {"transform.alpha": 0.00001}}},  # prints differently in JS
        {"options": {"awq_nvfp4": {"transform.grid": 0}}},
        {"options": {"gptq_int4": {"weight_algo.damp": True}}},
        {"options": {"bf16": {"weight.rounding": "rtz"}}},
        {"options": {"nvfp5": {}}},
        {"calibration": {"samples": 513}},
        {"calibration": {"seqlen": 1}},
        {"calibration": {"dataset": "ptb"}},
        {"calibration": {"split": "test"}},
        {"kv": "kivi2", "kv_options": {"residual": 48}},
        {"kv": "kivi2", "kv_options": {"residual": 0}},
        {"kv": "none", "kv_options": {"residual": 64}},
    ],
)
def test_options_that_do_not_apply_or_are_out_of_range_are_refused(case: dict) -> None:
    case = dict(case)
    kv = case.pop("kv", "none")
    with pytest.raises(ValueError):
        qr.build("bad", "nvfp4", qr.uniform("nvfp4"), kv, **case)


def test_calibration_is_written_when_a_method_needs_it_or_a_value_changes() -> None:
    plain = qr.uniform("nvfp4")
    assert "calibration" not in qr.build("c", "nvfp4", plain)
    assert qr.build("c", "awq_nvfp4", qr.uniform("awq_nvfp4"))["calibration"] == {"seed": 42}
    static = {"int8": {"activation.granularity": "tensor", "activation.observer.kind": "ema"}}
    assert qr.build("c", "int8", qr.uniform("int8"), options=static)["calibration"] == {"seed": 42}
    online = {"int8": {"activation.granularity": "tensor", "activation.observer.kind": "history"}}
    assert "calibration" not in qr.build("c", "int8", qr.uniform("int8"), options=online)  # history is online
    assert qr.build("c", "nvfp4", plain, calibration={"seqlen": 512, "dataset": "c4"})["calibration"] == {
        "dataset": "c4",
        "seqlen": 512,
        "seed": 42,
    }
    assert "calibration" not in qr.build("c", "nvfp4", plain, calibration=dict.fromkeys(["seed"], 42))


def test_cli_from_json_takes_options_calibration_and_kv_options(tmp_path: Path) -> None:
    spec = {
        "name": "cli",
        "default": "int8",
        "grid": qr.uniform("int8"),
        "kv": "kivi4",
        "options": {
            "int8": {
                "activation.granularity": "tensor",
                "activation.observer.kind": "percentile",
                "activation.observer.percentile": 99.9,
            }
        },
        "calibration": {"samples": 16},
        "kv_options": {"residual": 256},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps([spec]), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "app.quantizer_recipe", "--from-json", str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout
    expected = qr.build(
        "cli",
        "int8",
        spec["grid"],
        "kivi4",
        options=spec["options"],
        calibration=spec["calibration"],
        kv_options=spec["kv_options"],
    )
    assert out == f"---\n{qr.to_yaml(expected)}\n"
    assert "residual: 256" in out and "percentile: 99.9" in out and "samples: 16" in out
