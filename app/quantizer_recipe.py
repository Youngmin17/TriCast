"""Reference recipe generator for the Studio Quantizer demo (``app/web/quantizer.html``).

The page lets people pick one method per Linear layer on a block × kind grid; this module turns that grid
into a TriCast recipe the same way ``app/web/js/quantizer.js`` does, so the two can be checked against each
other and the recorded demo runs can be traced to the exact recipe the page shows. Pure stdlib: no torch, no
tricast.

Rules are emitted most specific first, because a recipe applies the first override that matches a layer:
single layers (``match`` on the full name), then whole blocks (``layers``), then whole Linear kinds
(``match`` on ``*.<parent>.<kind>``). ``resolve`` replays that first-match order and ``build`` refuses a
result that does not reproduce the grid.

A method can carry options (``OPTIONS``, per method ``METHOD_OPTIONS``); a method with options is a
*variant* and every layer using that method gets it. Options left at their default emit nothing, so a recipe
without options is byte-identical to one built before options existed. ``calibration`` and ``kv_options``
set the recipe's ``calibration`` block and the KV residual the same way.

    python -m app.quantizer_recipe --preset qd_mixed            # print one demo preset as YAML
    python -m app.quantizer_recipe --write-presets <dir>        # write every preset to <dir>/<id>.yaml
    python -m app.quantizer_recipe --from-json grids.json        # YAML for {name, default, grid, kv, options,
                                                                 #   calibration, kv_options} specs
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PARENT = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
          "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
BLOCKS = 28  # Qwen3-0.6B decoder blocks (the recorded demo model)

# Every quantized operand is dequantized onto BF16 before the GEMM (``mma_input: dequant``, the usual
# fake-quant convention). The recipe sets no ``mma``, so that GEMM uses the recipe default (cofda F=23) on
# the BF16 operands instead of a chip preset, and block-scaled formats stay fast.
# A mapping with "scheme" replaces the layer's previous weight/activation spec as a whole when overrides are
# merged, so no field of an earlier spec leaks in.
def _q(scheme: str, **fields: Any) -> dict[str, Any]:
    return {"scheme": scheme, **fields, "mma_input": "dequant"}


# NVINT4: the NVFP4 layout (16-element blocks, UE4M3 block scale under an fp32 tensor scale) with symmetric
# INT4 elements, i.e. the nvfp4 scheme with its element format replaced.
NVINT4 = _q("nvfp4", format="int4")

# method id -> recipe fields set for a layer using it (None = keep the layer in BF16, i.e. ``skip``)
METHODS: dict[str, dict[str, Any] | None] = {
    "bf16": None,
    "gptq_int4": {"weight": "int4_g128_zp", "activation": None, "transform": "none", "weight_algo": "gptq"},
    "awq_nvfp4": {"weight": _q("nvfp4"), "activation": _q("nvfp4"),
                  "transform": {"kind": "awq", "share_inputs": True}, "weight_algo": "rtn"},
    "smoothquant_nvfp4": {"weight": _q("nvfp4"), "activation": _q("nvfp4"),
                          "transform": {"kind": "smoothquant", "alpha": 0.5}, "weight_algo": "rtn"},
    "nvfp4": {"weight": _q("nvfp4"), "activation": _q("nvfp4"), "transform": "none", "weight_algo": "rtn"},
    "nvint4": {"weight": NVINT4, "activation": NVINT4, "transform": "none", "weight_algo": "rtn"},
    "mxfp4": {"weight": _q("mxfp4"), "activation": _q("mxfp4"), "transform": "none", "weight_algo": "rtn"},
    "mxfp8": {"weight": _q("mxfp8_e4m3"), "activation": _q("mxfp8_e4m3"), "transform": "none",
              "weight_algo": "rtn"},
    "int8": {"weight": _q("int8_row"), "activation": _q("int8_row"), "transform": "none",
             "weight_algo": "rtn"},
    "bfp12": {"weight": _q("msfp12"), "activation": _q("msfp12"), "transform": "none", "weight_algo": "rtn"},
}
CALIBRATED = {"gptq_int4", "awq_nvfp4", "smoothquant_nvfp4"}
KV = {"none": None, "kivi2": {"preset": "kivi2", "mode": "fakequant"},
      "kivi4": {"preset": "kivi4", "mode": "fakequant"}, "kv_fp8": {"preset": "kv_fp8", "mode": "fakequant"}}
DESCRIPTION = "Built with the Tri-Cast Quantizer demo (one method per Linear layer)."

# ---------- options ----------
# Every option is a recipe field of src/tricast/schemas/recipe.schema.json; its id is the field's path in
# a linear spec ("weight.scale.method" = weight: {scale: {method: ...}}). Defaults are what the method's
# fragment already implies (the tricast.quant.spec dataclass defaults; the acceptance scripts check them).
# "when": [option id, values] — the option applies only while that option's value is one of these. Ranges
# are inclusive unless "min_open"/"max_open"; maxima the schema leaves open are page limits (awq grid,
# observer history, GPTQ damp). Floats are 0 or at least FLOAT_MIN in magnitude, so they print the same in
# Python and JS.
FLOAT_MIN = 0.0001
_ROUNDING = {"type": "choice", "choices": ["rne", "rtz", "sr"], "default": "rne"}
_SCALE = {"type": "choice", "choices": ["absmax", "mse", "percentile"], "default": "absmax"}
_PERCENTILE = {"type": "float", "min": 0, "min_open": True, "max": 100, "default": 99.99}
OPTIONS: dict[str, dict[str, Any]] = {
    "weight.group_size": {"type": "choice", "choices": [32, 64, 128, 256], "default": 128},
    "weight.scale.method": _SCALE,
    "weight.scale.percentile": {**_PERCENTILE, "when": ["weight.scale.method", ["percentile"]]},
    "weight.rounding": _ROUNDING,
    "activation.granularity": {"type": "choice", "choices": ["row", "tensor"], "default": "row"},
    # a static observer replaces the call's amax, so the scale method would not be used
    "activation.scale.method": {**_SCALE, "when": ["activation.observer.kind", ["none"]]},
    "activation.scale.percentile": {**_PERCENTILE, "when": ["activation.scale.method", ["percentile"]]},
    "activation.rounding": _ROUNDING,
    # QuantSpec: static observers need a scaled, per-tensor spec. "none" = dynamic (no observer field).
    "activation.observer.kind": {"type": "choice", "default": "none",
                                 "choices": ["none", "minmax", "ema", "history", "percentile"],
                                 "when": ["activation.granularity", ["tensor"]]},
    "activation.observer.decay": {"type": "float", "min": 0, "max": 1, "max_open": True, "default": 0.99,
                                  "when": ["activation.observer.kind", ["ema"]]},
    "activation.observer.history_len": {"type": "int", "min": 1, "max": 1024, "default": 16,
                                        "when": ["activation.observer.kind", ["history"]]},
    "activation.observer.percentile": {**_PERCENTILE, "when": ["activation.observer.kind", ["percentile"]]},
    "transform.alpha": {"type": "float", "min": 0, "max": 1, "default": 0.5},
    "transform.grid": {"type": "int", "min": 1, "max": 100, "default": 20},
    "transform.share_inputs": {"type": "bool", "default": True},
    "weight_algo.damp": {"type": "float", "min": 0, "max": 1, "default": 0.01},
    "weight_algo.act_order": {"type": "bool", "default": False},
    "weight_algo.block_size": {"type": "choice", "choices": [32, 64, 128, 256], "default": 128},
}
_SCALED = ("weight.scale.method", "weight.scale.percentile", "weight.rounding",
           "activation.scale.method", "activation.scale.percentile", "activation.rounding")
# The options each method offers, in the order they are written. MX / BFP / NVFP4 block sizes and scale
# formats are fixed by those formats' definitions, so they have no group size or scale method option;
# GPTQ's zero point takes its range from [min, max], so it has no scale method either.
METHOD_OPTIONS: dict[str, tuple[str, ...]] = {
    "bf16": (),
    "gptq_int4": ("weight.group_size", "weight.rounding", "weight_algo.damp", "weight_algo.act_order",
                  "weight_algo.block_size"),
    "awq_nvfp4": (*_SCALED, "transform.grid", "transform.share_inputs"),
    "smoothquant_nvfp4": (*_SCALED, "transform.alpha"),
    "nvfp4": _SCALED,
    "nvint4": _SCALED,
    "mxfp4": ("weight.rounding", "activation.rounding"),
    "mxfp8": ("weight.rounding", "activation.rounding"),
    "int8": ("weight.scale.method", "weight.scale.percentile", "weight.rounding", "activation.granularity",
             "activation.scale.method", "activation.scale.percentile", "activation.rounding",
             "activation.observer.kind", "activation.observer.decay", "activation.observer.history_len",
             "activation.observer.percentile"),
    "bfp12": ("weight.rounding", "activation.rounding"),
}
CALIBRATING_OBSERVERS = ("minmax", "ema", "percentile")  # tricast: all observers but history calibrate
# A transform / weight_algo that carries options is written as a complete mapping (every field, in dataclass
# order): tricast merges an override's transform field by field onto the defaults' one.
TRANSFORM_FIELDS = {"block": 0, "seed": 0, "alpha": 0.5, "grid": 20, "share_inputs": True}
WEIGHT_ALGO_FIELDS = {"block_size": 128, "damp": 0.01, "act_order": False}
_COMPLETE = {"transform": TRANSFORM_FIELDS, "weight_algo": WEIGHT_ALGO_FIELDS}
# KV presets: the residual (tokens kept in full precision). KIVI's channel-axis keys need residual > 0 and a
# multiple of their group size 32; vLLM-style FP8 stores every token (0).
_KIVI_RESIDUAL = {"type": "int", "min": 32, "max": 4096, "multiple": 32, "default": 128}
KV_OPTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "none": {}, "kivi2": {"residual": _KIVI_RESIDUAL}, "kivi4": {"residual": _KIVI_RESIDUAL},
    "kv_fp8": {"residual": {"type": "int", "min": 0, "max": 4096, "default": 0}}}
# The recipe's calibration block (CLI: --calib-dataset / --calib-samples / --calib-seqlen / --calib-seed /
# --sequential). Written when a used method needs calibration or a value differs from these; seed is then
# always written (the page's 42, not tricast's 0). The samples / seqlen maxima are the live server's limits.
CALIBRATION: dict[str, dict[str, Any]] = {
    "dataset": {"type": "choice", "choices": ["wikitext2", "c4", "pile"], "default": "wikitext2"},
    "samples": {"type": "int", "min": 1, "max": 512, "default": 128},
    "seqlen": {"type": "int", "min": 2, "max": 4096, "default": 2048},
    "seed": {"type": "int", "min": 0, "max": 2**31 - 1, "default": 42},
    "sequential": {"type": "bool", "default": False},
}

Grid = list[list[str]]  # grid[block][kind index] = method id
Options = dict[str, dict[str, Any]]  # method id -> {option id: value}
Fragments = dict[str, dict[str, Any] | None]  # method id -> its layers' recipe fields (None = skip)


def check_value(where: str, spec: dict[str, Any], value: Any) -> Any:
    """``value`` if it is allowed by ``spec`` (an OPTIONS / KV_OPTIONS / CALIBRATION entry), else ValueError.
    Integral floats come back as ints, so a value prints the same in Python and JS."""
    kind = spec["type"]
    if kind == "choice":
        if not any(value == choice and type(value) is type(choice) for choice in spec["choices"]):
            raise ValueError(f"{where}: {value!r} is not one of {spec['choices']}")
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true or false, got {value!r}")
        return value
    number = kind == "float" and isinstance(value, float) and math.isfinite(value)
    if isinstance(value, bool) or not (isinstance(value, int) or number):
        raise ValueError(f"{where}: expected {'an integer' if kind == 'int' else 'a number'}, got {value!r}")
    if kind == "float" and value != 0 and abs(value) < FLOAT_MIN:
        raise ValueError(f"{where}: use 0 or a magnitude of at least {FLOAT_MIN}, got {value!r}")
    low, high = spec["min"], spec["max"]
    if value < low or (value == low and spec.get("min_open")) or value > high \
            or (value == high and spec.get("max_open")):
        raise ValueError(f"{where}: {value!r} is outside {'(' if spec.get('min_open') else '['}{low}, "
                         f"{high}{')' if spec.get('max_open') else ']'}")
    if "multiple" in spec and value % spec["multiple"]:
        raise ValueError(f"{where}: {value!r} is not a multiple of {spec['multiple']}")
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _applies(option: str, chosen: dict[str, Any]) -> bool:
    """Whether ``option`` takes effect given the other chosen options (unset ones are at their default)."""
    when = OPTIONS[option].get("when")
    if when is None:
        return True
    parent = when[0]
    value = chosen.get(parent, OPTIONS[parent]["default"]) if _applies(parent, chosen) \
        else OPTIONS[parent]["default"]
    return value in when[1]


def method_options(method: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """The checked options of one method that differ from their defaults, in METHOD_OPTIONS order. An option
    that does not apply (its "when" is not met) must be at its default: nothing is silently ignored."""
    given = dict(options or {})
    allowed = METHOD_OPTIONS[method]
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise ValueError(f"{method}: unknown options {unknown}; {method} has {list(allowed) or 'none'}")
    chosen = {option: check_value(f"{method}.{option}", OPTIONS[option], given[option])
              for option in allowed if option in given}
    out = {}
    for option, value in chosen.items():
        if value == OPTIONS[option]["default"]:
            continue
        if not _applies(option, chosen):
            parent, values = OPTIONS[option]["when"]
            raise ValueError(f"{method}.{option}: applies only when {parent} is one of {values}")
        out[option] = value
    return out


def _check_options(options: Options | None) -> Options:
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("options must map method ids to {option id: value}")
    unknown = sorted(set(options) - set(METHODS))
    if unknown:
        raise ValueError(f"options for unknown methods {unknown}")
    return {method: method_options(method, value) for method, value in options.items()}


def needs_calibration(method: str, options: dict[str, Any] | None = None) -> bool:
    """GPTQ, AWQ and SmoothQuant always; any method once a static activation observer is chosen."""
    chosen = method_options(method, options)
    return method in CALIBRATED or chosen.get("activation.observer.kind") in CALIBRATING_OBSERVERS


def _complete(key: str, value: str | dict[str, Any]) -> dict[str, Any]:
    fields = {"kind": value} if isinstance(value, str) else value
    return {"kind": fields["kind"],
            **{name: fields.get(name, default) for name, default in _COMPLETE[key].items()}}


def _modified(chosen: dict[str, Any]) -> set[str]:
    return {option.split(".")[0] for option in chosen} & set(_COMPLETE)


def _variant(method: str, chosen: dict[str, Any], complete: set[str]) -> dict[str, Any] | None:
    """METHODS[method] with the (checked) options written in. Operand options go inside the operand's scheme
    mapping (``mma_input`` stays last, as in ``_q``), so a later spec never inherits them; transform /
    weight_algo keys in ``complete`` (and those with options) become complete mappings."""
    base = METHODS[method]
    if base is None:
        return None
    fragment = copy.deepcopy(base)
    for key in complete | _modified(chosen):
        fragment[key] = _complete(key, fragment[key])
    for option, value in chosen.items():
        target, *path = option.split(".")
        if target in _COMPLETE:
            fragment[target][path[0]] = value
            continue
        operand = fragment[target]
        mapping = {"scheme": operand} if isinstance(operand, str) else dict(operand)
        tail = {"mma_input": mapping.pop("mma_input")} if "mma_input" in mapping else {}
        if path == ["observer", "kind"]:
            mapping["observer"] = {"kind": value}
        elif len(path) == 2:
            mapping.setdefault(path[0], {})[path[1]] = value
        else:
            mapping[path[0]] = value
        fragment[target] = {**mapping, **tail}
    return fragment


def variant(method: str, options: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The recipe fields of ``method`` with ``options``, as when it is a recipe's default (used alone)."""
    chosen = method_options(method, options)
    return _variant(method, chosen, _modified(chosen))


def fragments(default: str, options: Options | None = None) -> Fragments:
    """method id -> the fields its layers get in a recipe whose default method is ``default``. When the
    default has transform / weight_algo options, every method writes that key completely, so none of them
    inherits the default's values through tricast's field-by-field merge."""
    checked = _check_options(options)
    complete = _modified(checked.get(default, {}))
    return {method: _variant(method, checked.get(method, {}), complete) for method in METHODS}


def check_calibration(calibration: dict[str, Any] | None) -> dict[str, Any]:
    """The checked calibration settings, every key filled with its default."""
    given = dict(calibration or {})
    unknown = sorted(set(given) - set(CALIBRATION))
    if unknown:
        raise ValueError(f"calibration: unknown keys {unknown}; known: {list(CALIBRATION)}")
    return {key: check_value(f"calibration.{key}", spec, given[key]) if key in given else spec["default"]
            for key, spec in CALIBRATION.items()}


def _kv_fields(kv: str, kv_options: dict[str, Any] | None) -> dict[str, Any] | None:
    if kv not in KV:
        raise ValueError(f"unknown KV preset {kv!r}; known: {sorted(KV)}")
    given = dict(kv_options or {})
    unknown = sorted(set(given) - set(KV_OPTIONS[kv]))
    if unknown:
        raise ValueError(f"kv {kv}: unknown options {unknown}; it has {list(KV_OPTIONS[kv]) or 'none'}")
    if KV[kv] is None:
        return None
    fields = dict(KV[kv])
    for key, spec in KV_OPTIONS[kv].items():
        if key in given and (value := check_value(f"kv.{key}", spec, given[key])) != spec["default"]:
            fields[key] = value
    return fields


def layer_name(block: int, kind: str) -> str:
    return f"model.layers.{block}.{PARENT[kind]}.{kind}"


def _blocks_text(blocks: Sequence[int]) -> str:
    """Ascending block numbers with contiguous runs folded: [0, 1, 2, 27] -> "0-2,27"."""
    parts: list[str] = []
    start = prev = None
    for block in sorted(blocks):
        if start is None:
            start = prev = block
        elif block == prev + 1:
            prev = block
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = block
    if start is not None:
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def _expand_blocks(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _rule(selector: dict[str, Any], method: str, table: Fragments) -> dict[str, Any]:
    fields = table[method]
    return {**selector, "skip": True} if fields is None else {**selector, **fields}


def overrides_for(default: str, grid: Grid, table: Fragments | None = None) -> list[dict[str, Any]]:
    """Minimal first-match rules that reproduce ``grid`` on top of ``default`` (``table``: ``fragments``)."""
    table = METHODS if table is None else table
    covered: dict[tuple[int, int], str] = {}
    block_rules: dict[str, list[int]] = {}
    for block, row in enumerate(grid):
        if len(set(row)) == 1 and row[0] != default:
            block_rules.setdefault(row[0], []).append(block)
            covered.update({(block, k): row[0] for k in range(len(KINDS))})
    kind_rules: list[tuple[int, str]] = []
    for k in range(len(KINDS)):
        rest = {grid[b][k] for b in range(len(grid)) if (b, k) not in covered}
        if len(rest) == 1 and (method := rest.pop()) != default:
            kind_rules.append((k, method))
            covered.update({(b, k): method for b in range(len(grid)) if (b, k) not in covered})
    cells = [(b, k) for b in range(len(grid)) for k in range(len(KINDS))
             if grid[b][k] != covered.get((b, k), default)]
    rules = [_rule({"match": layer_name(b, KINDS[k])}, grid[b][k], table) for b, k in cells]
    # block rules in a stable order: by first block, so "0,27 → BF16" reads before later blocks
    for method, blocks in sorted(block_rules.items(), key=lambda item: min(item[1])):
        rules.append(_rule({"layers": _blocks_text(blocks)}, method, table))
    rules += [_rule({"match": f"*.{PARENT[KINDS[k]]}.{KINDS[k]}"}, method, table) for k, method in kind_rules]
    return rules


def resolve(default: str, rules: list[dict[str, Any]], blocks: int = BLOCKS,
            table: Fragments | None = None) -> Grid:
    """Replay first-match override resolution to the method id each layer gets."""
    table = METHODS if table is None else table
    by_fields = {json.dumps(fields, sort_keys=True): method for method, fields in table.items() if fields}
    if len(by_fields) != sum(1 for fields in table.values() if fields):
        raise AssertionError("two methods have the same recipe fields")
    grid: Grid = []
    for block in range(blocks):
        row = []
        for kind in KINDS:
            name, method = layer_name(block, kind), default
            for rule in rules:
                if "match" in rule:
                    pattern = rule["match"]
                    hit = name == pattern or (pattern.startswith("*.") and name.endswith(pattern[1:]))
                else:
                    hit = block in _expand_blocks(rule["layers"])
                if hit:
                    fields = {key: value for key, value in rule.items() if key not in ("match", "layers")}
                    method = "bf16" if fields.get("skip") else by_fields[json.dumps(fields, sort_keys=True)]
                    break
            row.append(method)
        grid.append(row)
    return grid


def build(name: str, default: str, grid: Grid, kv: str = "none", description: str = DESCRIPTION, *,
          options: Options | None = None, calibration: dict[str, Any] | None = None,
          kv_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """The recipe mapping for one grid (default method, per-layer methods, KV preset), with per-method
    ``options`` ({method: {option id: value}}), ``calibration`` settings and ``kv_options``."""
    if default not in METHODS or default == "bf16":
        raise ValueError(f"default method must be a quantization method, got {default!r}")
    if len(grid) != BLOCKS or any(len(row) != len(KINDS) for row in grid):
        raise ValueError(f"grid must be {BLOCKS} blocks × {len(KINDS)} kinds")
    unknown = sorted({m for row in grid for m in row} - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}")
    kv_fields = _kv_fields(kv, kv_options)
    checked = _check_options(options)
    settings = check_calibration(calibration)
    table = fragments(default, checked)
    rules = overrides_for(default, grid, table)
    if resolve(default, rules, table=table) != grid:
        raise AssertionError("generated rules do not reproduce the grid")
    used = {default, *(m for row in grid for m in row)}
    recipe: dict[str, Any] = {"name": name, "description": description,
                              "defaults": dict(table[default] or {})}
    if rules:
        recipe["overrides"] = rules
    if kv_fields is not None:
        recipe["kv"] = kv_fields
    recipe["exclude"] = ["lm_head"]
    changed = {key for key, value in settings.items() if value != CALIBRATION[key]["default"]}
    if changed or any(needs_calibration(method, checked.get(method)) for method in used):
        recipe["calibration"] = {key: value for key, value in settings.items()
                                 if key in changed or key == "seed"}
    return recipe


def uniform(method: str) -> Grid:
    return [[method] * len(KINDS) for _ in range(BLOCKS)]


def _preset_grids() -> dict[str, tuple[str, Grid, str, str]]:
    ends = uniform("mxfp4")
    ends[0], ends[-1] = ["bf16"] * len(KINDS), ["bf16"] * len(KINDS)
    mixed = uniform("awq_nvfp4")
    for block in range(BLOCKS):
        mixed[block][KINDS.index("o_proj")] = "mxfp8"
        mixed[block][KINDS.index("down_proj")] = "gptq_int4"
    mixed[0], mixed[-1] = ["bf16"] * len(KINDS), ["bf16"] * len(KINDS)
    return {"qd_mxfp4": ("mxfp4", uniform("mxfp4"), "none", "MXFP4 전체"),
            "qd_mxfp4_ends_bf16": ("mxfp4", ends, "none", "MXFP4 + 첫·마지막 블록 BF16"),
            "qd_nvfp4": ("nvfp4", uniform("nvfp4"), "none", "NVFP4 전체"),
            "qd_nvint4": ("nvint4", uniform("nvint4"), "none", "NVINT4 전체"),
            "qd_mixed": ("awq_nvfp4", mixed, "kivi2", "혼합: AWQ+NVFP4 · GPTQ down · MXFP8 o · KIVI-2")}


PRESETS = _preset_grids()


def preset(preset_id: str) -> dict[str, Any]:
    default, grid, kv, _ = PRESETS[preset_id]
    return build(preset_id, default, grid, kv)


def to_yaml(value: Any, indent: int = 0) -> str:
    """Deterministic YAML for the recipe subset (mappings, lists of mappings, scalars), as the page emits."""
    pad = " " * indent
    lines: list[str] = []
    for key, item in value.items():
        if isinstance(item, dict) and item:
            lines.append(f"{pad}{key}:")
            lines.append(to_yaml(item, indent + 2))
        elif isinstance(item, list) and item and all(isinstance(x, dict) for x in item):
            lines.append(f"{pad}{key}:")
            for entry in item:
                body = to_yaml(entry, indent + 4).splitlines()
                lines.append(f"{pad}  - {body[0].lstrip()}")
                lines.extend(body[1:])
        else:
            lines.append(f"{pad}{key}: {_scalar(item)}")
    return "\n".join(lines)


def _scalar(item: Any) -> str:
    if item is None:
        return "null"
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)):
        return repr(item)
    if isinstance(item, list):
        return "[" + ", ".join(_scalar(x) for x in item) + "]"
    if isinstance(item, dict):
        return "{}"
    text = str(item)
    plain = text and text[0].isalpha() and all(c.isalnum() or c in "_-." for c in text) \
        and text not in ("null", "true", "false", "none", "yes", "no", "on", "off")
    return text if plain or text == "none" else json.dumps(text, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.quantizer_recipe",
                                     description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=sorted(PRESETS))
    group.add_argument("--write-presets", type=Path, metavar="DIR")
    group.add_argument("--from-json", type=Path, metavar="FILE",
                       help='{"name", "default", "grid", "kv", "options", "calibration", "kv_options"} '
                            'or a list of them -> one YAML document each')
    args = parser.parse_args(argv)
    if args.preset:
        print(to_yaml(preset(args.preset)))
        return 0
    if args.from_json:
        specs = json.loads(args.from_json.read_text(encoding="utf-8"))
        for spec in specs if isinstance(specs, list) else [specs]:
            print("---")
            print(to_yaml(build(spec["name"], spec["default"], spec["grid"], spec.get("kv", "none"),
                                options=spec.get("options"), calibration=spec.get("calibration"),
                                kv_options=spec.get("kv_options"))))
        return 0
    args.write_presets.mkdir(parents=True, exist_ok=True)
    for preset_id in PRESETS:
        path = args.write_presets / f"{preset_id}.yaml"
        path.write_text(to_yaml(preset(preset_id)) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
