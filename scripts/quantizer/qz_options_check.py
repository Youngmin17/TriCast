"""tricast checks of app.quantizer_recipe's per-method options, calibration settings and KV options.

1. Every option / calibration / KV default is the value the method's fragment already implies (tricast
dataclass defaults, schema enums), so a default never emits anything. 2. 300 random (grid, options,
calibration, kv, kv_options) states: YAML round-trip == dict, load_recipe accepts it, every one of the 196
layers resolves (spec_for) to exactly its method-variant used alone as defaults, every chosen option value is
in the resolved spec (no silent ignore), the KV residual and calibration settings are in the Recipe, and the
calibration block is present exactly when needed or changed. 3. Refusals: options that do not exist, do not
apply, or are out of range raise ValueError.

usage: cd <repo> && PYTHONPATH=src:. python scripts/quantizer/qz_options_check.py   (ends with
QZ_OPTIONS_CHECK_OK)
"""

from __future__ import annotations

import json
import random
from dataclasses import fields
from pathlib import Path

import yaml
from app import quantizer_recipe as qr

from tricast.quant.spec import KV_PRESETS, ObserverSpec, ScaleSpec, TransformSpec, WeightAlgoSpec
from tricast.recipe import CALIBRATION_DEFAULTS, load_recipe

REPO = Path(qr.__file__).resolve().parents[1]
SCHEMA = json.loads((REPO / "src" / "tricast" / "schemas" / "recipe.schema.json").read_text(encoding="utf-8"))
DECODER = ("model.layers", qr.BLOCKS)


def enum(name: str) -> list:
    return SCHEMA["$defs"][name]["enum"]


def field_value(spec, option: str):
    """The resolved LinearSpec's value for an option id (e.g. activation.observer.kind)."""
    target, *path = option.split(".")
    value = getattr(spec, target)
    if path == ["observer", "kind"] and value.observer is None:
        return "none"
    for part in path:
        value = getattr(value, part)
    return getattr(value, "value", value)  # Rounding enum -> "rne"


def check_defaults() -> int:
    defaults = {f.name: f.default for f in fields(TransformSpec)}
    assert qr.TRANSFORM_FIELDS == {k: v for k, v in defaults.items() if k != "kind"}, qr.TRANSFORM_FIELDS
    assert qr.WEIGHT_ALGO_FIELDS == {f.name: f.default for f in fields(WeightAlgoSpec) if f.name != "kind"}
    observer = {f.name: f.default for f in fields(ObserverSpec)}
    scale = {f.name: f.default for f in fields(ScaleSpec)}
    assert qr.OPTIONS["activation.observer.decay"]["default"] == observer["decay"]
    assert qr.OPTIONS["activation.observer.history_len"]["default"] == observer["history_len"]
    assert qr.OPTIONS["activation.observer.percentile"]["default"] == observer["percentile"]
    for side in ("weight", "activation"):
        assert qr.OPTIONS[f"{side}.scale.percentile"]["default"] == scale["percentile"]
        assert set(qr.OPTIONS[f"{side}.scale.method"]["choices"]) <= set(enum("scale_method"))
        assert set(qr.OPTIONS[f"{side}.rounding"]["choices"]) <= set(enum("rounding"))
    assert set(qr.OPTIONS["activation.observer.kind"]["choices"]) - {"none"} <= set(enum("observer_kind"))
    assert set(qr.OPTIONS["activation.granularity"]["choices"]) <= set(enum("granularity_kind"))
    assert set(qr.CALIBRATING_OBSERVERS) == set(qr.OPTIONS["activation.observer.kind"]["choices"]) - {
        "none",
        "history",
    }
    calib = SCHEMA["$defs"]["calibration"]["properties"]
    assert qr.CALIBRATION["dataset"]["choices"] == calib["dataset"]["enum"]
    for key, spec in qr.CALIBRATION.items():
        want = 42 if key == "seed" else CALIBRATION_DEFAULTS[key]  # the page always writes seed 42
        assert spec["default"] == want, key
        if "minimum" in calib[key]:
            assert spec["min"] >= calib[key]["minimum"], key
    for kv, specs in qr.KV_OPTIONS.items():
        for key, spec in specs.items():
            assert spec["default"] == getattr(KV_PRESETS[kv], key), (kv, key)
    checked = 0
    for method, options in qr.METHOD_OPTIONS.items():
        if qr.METHODS[method] is None:
            continue
        alone = load_recipe({"name": method, "defaults": dict(qr.METHODS[method])}).defaults
        for option in options:
            if qr._applies(option, {}):
                assert field_value(alone, option) == qr.OPTIONS[option]["default"], (method, option)
                checked += 1
    return checked


FLOATS = (
    0,
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.999,
    1.0,
    2.0,
    12.5,
    50,
    99.0,
    99.9,
    99.99,
    100.0,
)


def random_value(rng: random.Random, spec: dict):
    if spec["type"] == "choice":
        return rng.choice(spec["choices"])
    if spec["type"] == "bool":
        return rng.random() < 0.5
    if spec["type"] == "int":
        step = spec.get("multiple", 1)
        return rng.randrange(spec["min"], min(spec["max"], spec["min"] + 64 * step) + 1, step)
    while True:  # a float inside the range
        value = rng.choice(FLOATS) if rng.random() < 0.7 else round(rng.uniform(spec["min"], spec["max"]), 4)
        try:
            return qr.check_value("random", spec, value)
        except ValueError:
            continue


def random_state(rng: random.Random, index: int) -> dict:
    methods = list(qr.METHODS)
    quant = [m for m in methods if m != "bf16"]
    default = rng.choice(quant)
    grid = [
        [rng.choice(methods) if rng.random() < 0.3 else default for _ in qr.KINDS] for _ in range(qr.BLOCKS)
    ]
    for b in rng.sample(range(qr.BLOCKS), rng.randint(0, 3)):
        grid[b] = [rng.choice(methods)] * len(qr.KINDS)
    for k in rng.sample(range(len(qr.KINDS)), rng.randint(0, 2)):
        m = rng.choice(methods)
        for b in range(qr.BLOCKS):
            grid[b][k] = m
    options = {}
    for method in quant:
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
            raw = {o: v for o, v in raw.items() if qr._applies(o, raw)}
            if raw:
                options[method] = raw
    calibration = {k: random_value(rng, spec) for k, spec in qr.CALIBRATION.items() if rng.random() < 0.25}
    kv = rng.choice(list(qr.KV))
    kv_options = {k: random_value(rng, spec) for k, spec in qr.KV_OPTIONS[kv].items() if rng.random() < 0.6}
    return {
        "name": f"qz_opt_{index:03d}",
        "default": default,
        "grid": grid,
        "kv": kv,
        "options": options,
        "calibration": calibration,
        "kv_options": kv_options,
    }


def check_state(state: dict, alone: dict, seen: set) -> int:
    recipe_dict = qr.build(
        state["name"],
        state["default"],
        state["grid"],
        state["kv"],
        options=state["options"],
        calibration=state["calibration"],
        kv_options=state["kv_options"],
    )
    assert yaml.safe_load(qr.to_yaml(recipe_dict)) == recipe_dict, state["name"]
    recipe = load_recipe(recipe_dict)
    chosen = {m: qr.method_options(m, o) for m, o in state["options"].items()}
    layers = 0
    for b in range(qr.BLOCKS):
        for k, kind in enumerate(qr.KINDS):
            method = state["grid"][b][k]
            spec = recipe.spec_for(qr.layer_name(b, kind), decoder=DECODER)
            if method == "bf16":
                assert spec is None, (state["name"], b, kind)
                continue
            key = json.dumps([method, chosen.get(method, {})], sort_keys=True)
            if key not in alone:
                alone[key] = load_recipe(
                    {"name": "alone", "defaults": qr.variant(method, chosen.get(method))}
                ).defaults
            assert spec == alone[key], (state["name"], b, kind, method, chosen.get(method))
            for option, value in chosen.get(method, {}).items():
                assert field_value(spec, option) == value, (state["name"], method, option, value)
                seen.add(option)
            layers += 1
    used = {state["default"], *(m for row in state["grid"] for m in row)}
    if state["kv"] != "none":
        residual = state["kv_options"].get("residual", qr.KV_OPTIONS[state["kv"]]["residual"]["default"])
        assert recipe.kv.residual == residual and recipe.kv.mode == "fakequant", state["name"]
        if "residual" in state["kv_options"]:
            seen.add("kv.residual")
    else:
        assert recipe.kv is None
    settings = qr.check_calibration(state["calibration"])
    needed = any(qr.needs_calibration(m, chosen.get(m)) for m in used)
    changed = {k for k, v in settings.items() if v != qr.CALIBRATION[k]["default"]}
    assert ("calibration" in recipe_dict) == (needed or bool(changed)), state["name"]
    assert recipe.needs_calibration == needed, (state["name"], recipe.needs_calibration, needed)
    if "calibration" in recipe_dict:
        options = recipe.calibration_options
        assert all(options[k] == v for k, v in settings.items()), (state["name"], options, settings)
        seen.update(f"calibration.{k}" for k in changed)
    return layers


def check_refusals() -> int:
    grid = qr.uniform("nvfp4")
    bad = [
        {"options": {"nvfp4": {"weight.group_size": 64}}},  # fixed by the NVFP4 definition
        {"options": {"mxfp4": {"weight.scale.method": "mse"}}},  # E8M0 scales: pow2 only
        {"options": {"gptq_int4": {"weight.scale.method": "mse"}}},  # zero point: [min, max]
        {"options": {"nvfp4": {"weight.scale.percentile": 99.0}}},  # needs scale.method percentile
        {"options": {"int8": {"activation.observer.kind": "ema"}}},  # needs tensor granularity
        {
            "options": {
                "int8": {
                    "activation.granularity": "tensor",
                    "activation.observer.kind": "ema",
                    "activation.scale.method": "mse",
                }
            }
        },  # the observer sets amax
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
    ]
    for case in bad:
        kv = case.pop("kv", "none")
        try:
            qr.build("bad", "nvfp4", grid, kv, **case)
        except ValueError:
            continue
        raise AssertionError(f"accepted {case}")
    return len(bad)


def main() -> int:
    implied = check_defaults()
    rng = random.Random(7)
    alone: dict = {}
    seen: set = set()
    layers = sum(check_state(random_state(rng, i), alone, seen) for i in range(300))
    every = {*qr.OPTIONS, "kv.residual", *(f"calibration.{k}" for k in qr.CALIBRATION)}
    missing = sorted(every - seen)
    assert not missing, f"never exercised: {missing}"
    refused = check_refusals()
    print(
        f"defaults implied by fragments {implied}, states 300, quantized layers checked {layers}, "
        f"variants {len(alone)}, option fields seen {len(seen)}/{len(every)}, refusals {refused}"
    )
    print("QZ_OPTIONS_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
