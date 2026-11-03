"""The paper-track Python-only methods (app.quantizer_recipe.LOWBIT: gptq_int3, gptq_int2, hqq_int3,
hqq_int2, awq_int4, had_nvfp4, had_mxfp4) in the grid recipes that app.autotune and app.tune_agent use:
kept apart from the page (METHODS, CALIBRATED, the committed bundle) and from the pinned EXTENDED set, every
layer of a grid resolves to its method's own spec (options included), --from-json prints the same YAML, a
tiny-model calibrated Auto-tune table over them and its solve, their stored bits, and the recipe hashes
recorded before they existed.
"""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from app import autotune as at
from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr
from app import tune_agent as ta
from app.tests.test_autotune import from_json
from app.tests.test_autotune_v2 import load, make_root, run, table_run
from app.tests.test_quantizer_export import BUNDLE
from tricast.recipe import load_recipe

DECODER = ("model.layers", qr.BLOCKS)
LOW = list(qr.LOWBIT)
CALIBRATED_LOW = {"gptq_int3", "gptq_int2", "awq_int4"}
# method-alone recipe hashes of METHODS + EXTENDED as recorded at 7120c41, before LOWBIT existed
RECORDED = {
    "awq_nvfp4": "ebce2981f974928be1e7087a9133cdffbe24b0f2a1020914436737b9afaa483c",
    "bfp12": "2f86fd1b55e521ba951b7a5ec280b9f96bcf2d68ccb196a9919179764cbad69a",
    "gptq_int4": "3e9dcebf1c2083a7866598e0c7d43ad7965ff1e705cf59940f74bd54acbe1e0e",
    "hqq_int4": "f9fc0b27c173b15f3305dc0cafc195f8ed48b7789601cacbf83f390f5dd95406",
    "int8": "2e6f5a85bbb06b7190ec2b17332abb0d3ac5bc0e37e11df727e1394710f2b55e",
    "mxfp4": "31a4f801561e28ec76a5943fbd667b782724899d5e7bbdde6327daf5f4beefff",
    "mxfp8": "ba60197accde3986047fd6a1b247972c3122ca996ab5cc6d95a359183915a17f",
    "nf4": "9902e6a32701815ed96a8a8989eb0fed49d2bc3f0af3fc026bf0e2343270800a",
    "nvfp4": "8ad6416ad153bb3ff855ab40f65754c2bd75f4a55e13aee3b64f44e06b76092a",
    "nvint4": "59e2c240f8e49c568c49a13a3e44cebc7146b0d832d3bb2e8918a94c460f905f",
    "razer_nvfp4": "87e368adf87572594bdf5446fb4dc9886eba769acc1189c96533ac4f423ad9bb",
    "smoothquant_nvfp4": "b9f9732465a078eabc74b58b95265c82c42bf5f7a6eadf2db8a4b5cd7d0cdc7c",
}
OPTIONS = [None,
           {"gptq_int3": {"weight_algo.damp": 0.05, "weight.group_size": 64}},
           {"awq_int4": {"transform.grid": 10, "weight.rounding": "rtz"}},
           {"had_nvfp4": {"weight.rounding": "rtz", "activation.scale.method": "mse"},
            "gptq_int2": {"weight_algo.act_order": True}}]


def alone(method: str, options: dict | None = None):
    fields = qr.variant(method, (options or {}).get(method))
    return None if fields is None else load_recipe({"name": method, "defaults": dict(fields)}).defaults


def test_lowbit_is_apart_from_the_page_and_the_pinned_extended_set() -> None:
    assert LOW == ["gptq_int3", "gptq_int2", "hqq_int3", "hqq_int2", "awq_int4", "had_nvfp4", "had_mxfp4"]
    assert not set(LOW) & (set(qr.METHODS) | set(qr.EXTENDED))
    assert list(qr.ALL_METHODS) == [*qr.METHODS, *qr.EXTENDED]  # unchanged: the tune loop's alphabet
    assert list(qr.PYTHON_METHODS) == [*qr.ALL_METHODS, *LOW]
    assert qr.CALIBRATED == {"gptq_int4", "awq_nvfp4", "smoothquant_nvfp4"}  # the page's flags, unchanged
    assert qr.CALIBRATED_ALL == qr.CALIBRATED | CALIBRATED_LOW
    assert {m for m in qr.PYTHON_METHODS if qr.needs_calibration(m)} == qr.CALIBRATED_ALL
    # options: page keys only in the mapping; LOWBIT answers through __missing__
    assert set(qr.METHOD_OPTIONS) == set(qr.METHODS)
    assert {m: qr.METHOD_OPTIONS[m] for m in LOW} == {
        "gptq_int3": qr.METHOD_OPTIONS["gptq_int4"], "gptq_int2": qr.METHOD_OPTIONS["gptq_int4"],
        "hqq_int3": (), "hqq_int2": (),
        "awq_int4": ("weight.group_size", "weight.rounding", "transform.grid", "transform.share_inputs"),
        "had_nvfp4": qr.METHOD_OPTIONS["nvfp4"], "had_mxfp4": qr.METHOD_OPTIONS["mxfp4"]}
    with pytest.raises(ValueError, match="unknown options"):
        qr.build("t", "hqq_int3", qr.uniform("hqq_int3"), options={"hqq_int3": {"weight.rounding": "rtz"}})
    # the page data: the committed bundle is what the exporter writes now, and names no LOWBIT method
    text = BUNDLE.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["generator"] == json.loads(json.dumps(ex.generator()))
    assert {key: data[key] for key in ex.catalog()} == json.loads(json.dumps(ex.catalog()))
    assert data["generator"]["calibrated"] == sorted(qr.CALIBRATED)
    assert sorted(m["id"] for m in ex.catalog()["methods"]) == sorted(qr.METHODS)
    assert not [m for m in LOW if m in text]


def test_recipe_hashes_recorded_before_lowbit_are_unchanged() -> None:
    assert {m: load_recipe({"name": m, "defaults": dict(f)}).sha256
            for m, f in qr.ALL_METHODS.items() if f} == RECORDED
    for preset_id in qr.PRESETS:  # the page presets: YAML and recipe hash as pinned by the exporter
        assert load_recipe(qr.preset(preset_id)).sha256 == ex.PINS[preset_id][1]


def test_lowbit_specs() -> None:
    specs = {m: alone(m) for m in LOW}
    assert {m: (s.weight.format.name, s.weight.group_size, s.weight.scale.format.name, s.weight.zero_point,
                s.weight_algo.kind, s.activation) for m, s in specs.items() if m.startswith("gptq")} == {
        "gptq_int3": ("uint3", 128, "fp16", "int", "gptq", None),
        "gptq_int2": ("uint2", 128, "fp16", "int", "gptq", None)}
    assert specs["gptq_int3"].weight == alone("gptq_int4").weight.with_(format="uint3")
    hqq = {m: (specs[m].weight, specs[m].weight_algo.kind, specs[m].activation)
           for m in ("hqq_int3", "hqq_int2")}
    assert hqq == {"hqq_int3": (None, "hqq3", None), "hqq_int2": (None, "hqq2", None)}
    awq = specs["awq_int4"]
    assert (awq.weight, awq.activation, awq.transform.kind, awq.weight_algo.kind) == (
        alone("gptq_int4").weight, None, "awq", "rtn")
    for fmt in ("nvfp4", "mxfp4"):
        had, plain = specs[f"had_{fmt}"], alone(fmt)
        assert (had.weight, had.activation, had.transform.kind, had.transform.block) == (
            plain.weight, plain.activation, "hadamard", 0)


def test_stored_bits_of_the_lowbit_methods() -> None:
    assert {m: ex.WEIGHT_BITS[m] for m in LOW} == {
        "gptq_int3": (3, 19, 128), "gptq_int2": (2, 18, 128), "hqq_int3": (3, 32, 64),
        "hqq_int2": (2, 32, 64), "awq_int4": (4, 20, 128), "had_nvfp4": (4, 8, 16), "had_mxfp4": (4, 8, 32)}
    assert {m: at.weight_bits(m, 1024) for m in LOW} == {
        "gptq_int3": Fraction(403, 128), "gptq_int2": Fraction(137, 64), "hqq_int3": Fraction(7, 2),
        "hqq_int2": Fraction(5, 2), "awq_int4": Fraction(133, 32), "had_nvfp4": Fraction(9, 2),
        "had_mxfp4": Fraction(17, 4)}
    assert ta.method_bits("gptq_int3", 1024, {"gptq_int3": {"weight.group_size": 64}}) == 3 + Fraction(19, 64)


def test_lowbit_grids_resolve_to_each_methods_own_spec(tmp_path: Path) -> None:
    rng = random.Random(5)
    methods = list(qr.PYTHON_METHODS)
    seen = set()
    for t in range(32):
        options = OPTIONS[t % len(OPTIONS)]
        default = rng.choice([m for m in methods if m != "bf16"] if t % 2 else LOW)
        grid = [[rng.choice(methods) if rng.random() < 0.4 else default for _ in qr.KINDS]
                for _ in range(qr.BLOCKS)]
        grid[rng.randrange(qr.BLOCKS)] = [rng.choice(LOW)] * len(qr.KINDS)
        data = qr.build("t", default, grid, options=options)
        assert yaml.safe_load(qr.to_yaml(data)) == data
        assert qr.resolve(default, data.get("overrides", []), table=qr.fragments(default, options)) == grid
        used = {default, *(m for row in grid for m in row)}
        assert ("calibration" in data) == bool(used & qr.CALIBRATED_ALL)
        recipe = load_recipe(data)
        expected = {m: alone(m, options) for m in methods}
        for b in range(qr.BLOCKS):
            for k, kind in enumerate(qr.KINDS):
                seen.add(grid[b][k])
                assert recipe.spec_for(qr.layer_name(b, kind), decoder=DECODER) == expected[grid[b][k]], (
                    t, b, kind, grid[b][k])
    assert set(LOW) <= seen
    # --from-json prints exactly build's YAML for a LOWBIT grid with options and calibration settings
    spec = {"name": "low", "default": "gptq_int3", "grid": [["hqq_int2", "had_mxfp4", "gptq_int2", "awq_int4",
                                                            "had_nvfp4", "hqq_int3", "bf16"]] + qr.uniform(
        "gptq_int3", qr.BLOCKS - 1), "kv": "kivi2", "options": OPTIONS[1], "calibration": {"samples": 16}}
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    built = qr.build(spec["name"], spec["default"], spec["grid"], spec["kv"], options=spec["options"],
                     calibration=spec["calibration"])
    assert from_json(path) == f"---\n{qr.to_yaml(built)}\n"
    assert built["calibration"] == {"samples": 16, "seed": 42}


def test_the_agent_asks_for_calibration_of_the_calibrated_lowbit_methods() -> None:
    windows = {"dataset": "wikitext2", "samples": 128, "seqlen": 2048, "seed": 42}
    for method in qr.PYTHON_METHODS:
        if method == "bf16":
            continue
        spec = {"grid": [[method] * len(qr.KINDS)] + qr.uniform("nvfp4", qr.BLOCKS - 1)}
        reason = ta._unpredicted(spec, {"meta": {}})
        assert (reason is not None) == (method in qr.CALIBRATED_ALL), method
        assert ta._unpredicted(spec, {"meta": {"calibration_windows": windows}}) is None


# ---------- a calibrated tiny-model table over the LOWBIT methods ----------
TABLE_METHODS = ["bf16", "nvfp4", *LOW]
CALIB_ARGS = ("--calibrate", "--methods", ",".join(TABLE_METHODS), "--blocks", "1", "--calib-samples", "4",
              "--calib-seqlen", "64")


@pytest.fixture(scope="module")
def lowbit_table(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    pytest.importorskip("transformers")
    pytest.importorskip("hqq")
    root = make_root(tmp_path_factory.mktemp("lowbit"))
    rc, stdout, stderr, spy = table_run(root, root / "table.json", *CALIB_ARGS)
    assert rc == 0, stderr
    # 7 layers x 8 methods, less awq_int4 on the five shared-input kinds
    assert stdout.splitlines()[0].startswith("AUTOTUNE_TABLE_START layers=7 methods=9 entries=51 done=0 "
                                             "floors=7")
    return root, spy


def test_the_table_measures_every_lowbit_method_and_they_fire(lowbit_table: tuple[Path, dict]) -> None:
    root, spy = lowbit_table
    table = load(root / "table.json")
    assert table["meta"]["methods"] == TABLE_METHODS
    assert set(table["meta"]["method_recipe_sha256"]) == set(TABLE_METHODS[1:])
    assert "awq_int4" in table["meta"]["shared_input_rule"]
    for row in table["layers"]:
        unique = row["kind"] in at.UNIQUE_INPUT
        kl = row["kl"]
        assert kl["bf16"] == 0.0 and (kl["awq_int4"] is None) == (not unique), row["name"]
        measured = [m for m in TABLE_METHODS[1:] if kl[m] is not None]
        assert all(math.isfinite(kl[m]) and kl[m] > 0 for m in measured), row["name"]
        assert len({kl[m] for m in measured}) == len(measured), row["name"]  # every method a different weight
    entries = [json.loads(line) for line in (root / "table.entries.jsonl").read_text().splitlines()[2:]]
    calibrated = [e for e in entries if e["method"] in qr.CALIBRATED_ALL]
    assert len(calibrated) == 7 + 7 + 2 and spy["datasets"] == [("wikitext2", "train")] * 16
    for entry in calibrated:  # each one calibrated exactly its layer
        assert entry["calibration"]["layers"] == [entry["layer"]]
    assert all("calibration" not in e for e in entries if e["method"] not in qr.CALIBRATED_ALL)
    order = [(e["layer"], e["method"]) for e in entries]
    levels = {"gptq_int3": (8, 128), "gptq_int2": (4, 128), "hqq_int3": (8, 64), "hqq_int2": (4, 64)}
    for method, (most, group) in levels.items():  # the measured weight is on the n-bit grid of its groups
        state = spy["states"][order.index(("model.layers.0.self_attn.o_proj", method))]
        ((spec, weight),) = state["linears"].values()
        assert spec == alone(method)
        assert max(len(g.unique()) for g in weight.reshape(-1, group)) <= most
    for method in ("had_nvfp4", "had_mxfp4"):
        state = spy["states"][order.index(("model.layers.0.mlp.down_proj", method))]
        ((spec, _),) = state["linears"].values()
        assert spec == alone(method) and spec.transform.kind == "hadamard"


def test_solve_writes_a_lowbit_grid_that_from_json_reproduces(lowbit_table: tuple[Path, dict]) -> None:
    root, _ = lowbit_table
    out = root / "solved"
    rc, _, err = run(["solve", "--table", str(root / "table.json"), "--budget-bits", "3.25",
                      "--out", str(out)])
    assert rc == 0, err
    spec = load(out / "grid.json")
    assert from_json(out / "grid.json") == (out / "recipe.yaml").read_text(encoding="utf-8")
    chosen = [e["method"] for e in load(out / "solution.json")["layers"]]
    assert set(chosen) <= set(TABLE_METHODS) and set(chosen) & set(LOW)
    recipe = load_recipe(yaml.safe_load((out / "recipe.yaml").read_text(encoding="utf-8")))
    for e in load(out / "solution.json")["layers"]:
        assert recipe.spec_for(e["name"], decoder=DECODER) == alone(e["method"])
    assert ("calibration" in spec) == bool(set(chosen) & qr.CALIBRATED_ALL)
