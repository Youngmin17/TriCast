"""The Python-only Quantizer methods (app.quantizer_recipe.EXTENDED: nf4, hqq_int4, razer_nvfp4) in the grid
recipes that app.autotune and app.tune_agent use: every layer of a grid resolves to its method's own spec, the
methods stay off the page's catalog (AC15 covers METHODS), their stored bits, a tiny-model Auto-tune table
over them, and the agent's ranked moves to them.
"""

from __future__ import annotations

import contextlib
import io
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
from app.tests.test_autotune import QWEN, TEXT, TINY_ARGS, from_json, tiny_qwen3
from tricast.recipe import load_recipe

DECODER = ("model.layers", qr.BLOCKS)
NEW = list(qr.EXTENDED)
ALONE = {m: load_recipe({"name": m, "defaults": dict(f)}).defaults for m, f in qr.ALL_METHODS.items() if f}
TABLE_METHODS = ["bf16", "nvfp4", *NEW]


def test_extended_grids_resolve_to_each_methods_own_spec() -> None:
    rng = random.Random(3)
    methods = list(qr.ALL_METHODS)
    seen = set()
    for t in range(40):
        default = rng.choice([m for m in methods if m != "bf16"] if t % 2 else NEW)
        grid = [[rng.choice(methods) if rng.random() < 0.4 else default for _ in qr.KINDS]
                for _ in range(qr.BLOCKS)]
        grid[rng.randrange(qr.BLOCKS)] = [rng.choice(NEW)] * len(qr.KINDS)
        # gptq options make every method write weight_algo as a complete mapping (AC16)
        options = {"gptq_int4": {"weight_algo.damp": 0.05}} if default == "gptq_int4" else None
        data = qr.build("t", default, grid, options=options)
        assert yaml.safe_load(qr.to_yaml(data)) == data
        table = qr.fragments(default, options)
        assert qr.resolve(default, data.get("overrides", []), table=table) == grid
        recipe = load_recipe(data)
        for b in range(qr.BLOCKS):
            for k, kind in enumerate(qr.KINDS):
                method = grid[b][k]
                seen.add(method)
                spec = recipe.spec_for(qr.layer_name(b, kind), decoder=DECODER)
                if method == "gptq_int4" and options:
                    assert spec.weight_algo.damp == 0.05
                    continue
                assert spec == (None if method == "bf16" else ALONE[method]), (t, b, kind, method)
    assert set(NEW) <= seen
    assert {m: (ALONE[m].weight, ALONE[m].weight_algo.kind) for m in NEW} == {
        "nf4": (None, "nf4"), "hqq_int4": (None, "hqq"), "razer_nvfp4": (None, "razer")}
    assert ALONE["nf4"].activation is None and ALONE["hqq_int4"].activation is None
    assert ALONE["razer_nvfp4"].activation == ALONE["nvfp4"].activation  # plain NVFP4 activations


def test_extended_methods_stay_off_the_page_but_build_and_have_no_options() -> None:
    assert not set(qr.EXTENDED) & set(qr.METHODS) and list(qr.ALL_METHODS) == [*qr.METHODS, *qr.EXTENDED]
    assert [key for key, _ in ex.generator()["methods"]["items"]] == list(qr.METHODS)
    assert sorted(m["id"] for m in ex.catalog()["methods"]) == sorted(qr.METHODS)
    assert set(qr.METHOD_OPTIONS) == set(qr.METHODS) and all(qr.METHOD_OPTIONS[m] == () for m in NEW)
    with pytest.raises(KeyError):
        qr.METHOD_OPTIONS["nf5"]
    with pytest.raises(ValueError, match="unknown options"):
        qr.build("t", "nf4", qr.uniform("nf4"), options={"nf4": {"weight.rounding": "rtz"}})
    assert all(not qr.needs_calibration(m) for m in NEW)
    assert "calibration" not in qr.build("t", "nf4", qr.uniform("hqq_int4"))


def test_stored_bits_of_the_extended_methods() -> None:
    """nf4: 4 + fp32 absmax / 64; hqq_int4: 4 + (BF16 scale + BF16 zero) / 64; razer_nvfp4: 4 + E4M3 / 16."""
    assert {m: ex.WEIGHT_BITS[m] for m in NEW} == {"nf4": (4, 32, 64), "hqq_int4": (4, 32, 64),
                                                   "razer_nvfp4": (4, 8, 16)}
    for m in NEW:
        assert {at.weight_bits(m, n) for n in (1024, 2048, 3072)} == {Fraction(9, 2)}


def test_tiny_table_tables_the_extended_methods(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("hqq")
    (tmp_path / "model").mkdir()
    (tmp_path / "text.txt").write_text(TEXT, encoding="utf-8")
    out = tmp_path / "table.json"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        rc = at.main(["table", "--model", str(tmp_path / "model"), "--text", str(tmp_path / "text.txt"),
                      *TINY_ARGS, "--methods", ",".join(TABLE_METHODS), "--out", str(out)],
                     model_factory=tiny_qwen3)
    assert rc == 0, stderr.getvalue()
    lines = stdout.getvalue().splitlines()
    assert lines[0].startswith("AUTOTUNE_TABLE_START layers=14 methods=5 entries=56 done=0")
    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["meta"]["methods"] == TABLE_METHODS
    assert set(table["meta"]["method_recipe_sha256"]) == set(TABLE_METHODS[1:])
    for row in table["layers"]:
        kl = row["kl"]
        assert list(kl) == TABLE_METHODS and kl["bf16"] == 0.0
        assert all(math.isfinite(kl[m]) and kl[m] > 0 for m in TABLE_METHODS[1:]), row["name"]  # each fired
        assert len({kl[m] for m in TABLE_METHODS[1:]}) == 4, row["name"]  # four different weights
    for m in TABLE_METHODS[1:]:
        assert at.method_recipe(m).defaults == ALONE[m]
    solved = tmp_path / "s"
    with contextlib.redirect_stdout(io.StringIO()):
        assert at.main(["solve", "--table", str(out), "--budget-bits", "4.5", "--out", str(solved)]) == 0
    spec = json.loads((solved / "grid.json").read_text(encoding="utf-8"))
    assert from_json(solved / "grid.json") == (solved / "recipe.yaml").read_text(encoding="utf-8")
    assert {m for row in spec["grid"][:2] for m in row} <= set(TABLE_METHODS[1:])


def test_the_agent_ranks_and_builds_moves_to_the_extended_methods(tmp_path: Path) -> None:
    rng = random.Random(0)
    rows = [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
             "in_features": layer.in_features,
             "kl": {m: 0.0 if m == "bf16" else rng.random() * 1e-3 for m in TABLE_METHODS}} for layer in QWEN]
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"schema": at.TABLE_SCHEMA, "status": "complete", "objective": at.OBJECTIVE,
                                 "meta": {"methods": TABLE_METHODS, "model_blocks": qr.BLOCKS},
                                 "layers": rows}), encoding="utf-8")
    start = {"default": "nvfp4", "grid": qr.uniform("nvfp4"), "kv": "none", "options": None,
             "calibration": None, "kv_options": None}
    moves = ta.neighbors(start, table, Fraction(9, 2), 1000)
    assert {move["to"] for move in moves} == set(NEW)  # every move fits 4.5 bits: bf16 does not
    for move in moves[:6]:
        spec = {**start, "grid": move["grid"]}
        result = ta.evaluate(spec, table, Fraction(9, 2))
        assert result["avg_bits_fraction"] == "9/2" and result["predicted"]["unpredicted"] is None
        recipe = load_recipe(ta._recipe("move", spec))
        layer = next(item for item in QWEN if item.name == move["layer"])
        assert recipe.spec_for(layer.name, decoder=DECODER) == ALONE[move["to"]]
