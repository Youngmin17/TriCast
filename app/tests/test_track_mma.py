"""One GEMM accumulation per recipe (``mma``, the paper's per-track accumulation): app.quantizer_recipe.build
writes it as ``defaults.mma`` and every quantized cell inherits it; the auto-tune table (``--mma``) uses it
for the method recipes and floors and records it, solve writes it into grid.json / recipe.yaml, and the tune
loop keeps it through every proposal and refuses a grid whose mma is not the table's. CPU only."""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from app import autotune as at
from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_autotune import from_json
from app.tests.test_autotune_v2 import load, make_root, run, table_run
from app.tests.test_tune_agent import _ledger, _table
from app.tests.test_tune_loop import _measure
from tricast.recipe import load_recipe

DECODER = ("model.layers", qr.BLOCKS)
FMA = load_recipe({"name": "m", "defaults": {"mma": "fp32_fma"}}).defaults.mma


def _mixed_grid() -> qr.Grid:
    grid = qr.uniform("gptq_int3")
    grid[0] = ["bf16", "nvfp4", "rht_gptq_int2", "awq_int3", "int8_w", "hqq_int2", "gptq_int4"]
    return grid


def test_build_writes_one_mma_that_every_quantized_cell_inherits(tmp_path: Path) -> None:
    grid = _mixed_grid()
    plain = qr.build("t", "gptq_int3", grid)
    fma = qr.build("t", "gptq_int3", grid, mma="fp32_fma")
    assert "mma" not in plain["defaults"] and fma["defaults"]["mma"] == "fp32_fma"
    assert {key: value for key, value in fma["defaults"].items() if key != "mma"} == plain["defaults"]
    assert fma["overrides"] == plain["overrides"] and qr.build("t", "gptq_int3", grid, mma=None) == plain
    with_mma, without = load_recipe(fma), load_recipe(plain)
    for block in range(qr.BLOCKS):
        for k, kind in enumerate(qr.KINDS):
            name = qr.layer_name(block, kind)
            spec, ref = with_mma.spec_for(name, decoder=DECODER), without.spec_for(name, decoder=DECODER)
            if grid[block][k] == "bf16":
                assert spec is None and ref is None
                continue
            assert spec.mma == FMA and ref.mma != FMA and dataclasses.replace(spec, mma=ref.mma) == ref, name
    with pytest.raises(ValueError, match="mma: expected one of"):
        qr.build("t", "gptq_int3", grid, mma="fp33_fma")
    path = tmp_path / "grid.json"
    path.write_text(json.dumps({"name": "t", "default": "gptq_int3", "grid": grid, "mma": "fp32_fma"}))
    assert from_json(path) == f"---\n{qr.to_yaml(fma)}\n"


@pytest.fixture(scope="module")
def fma_table(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    pytest.importorskip("transformers")
    root = make_root(tmp_path_factory.mktemp("fma"))
    rc, _, stderr, spy = table_run(root, root / "table.json", "--methods", "bf16,nvfp4,int8_w",
                                   "--mma", "fp32_fma")
    assert rc == 0, stderr
    return root, spy


def test_the_table_measures_and_records_one_mma(fma_table: tuple[Path, dict]) -> None:
    root, spy = fma_table
    meta = load(root / "table.json")["meta"]
    assert meta["mma"] == "fp32_fma"
    assert meta["method_recipe_sha256"] == {m: at.method_recipe(m, None, "fp32_fma").sha256
                                            for m in ("nvfp4", "int8_w")}
    assert meta["floor_recipe_sha256"] == at.floor_recipe("fp32_fma").sha256 != at.floor_recipe().sha256
    specs = [spec for state in spy["states"] for spec, _ in state["linears"].values()]
    assert specs and all(spec.mma == FMA for spec in specs)  # method entries and floors alike
    # resuming the same out without --mma would mix accumulations: refused before any entry is measured
    rc, _, err, _ = table_run(root, root / "table.json", "--methods", "bf16,nvfp4,int8_w")
    assert rc == 2 and "recipe changed since its entries were" in err, err


def test_solve_writes_the_tables_mma_and_from_json_reproduces_it(fma_table: tuple[Path, dict]) -> None:
    root, _ = fma_table
    out = root / "solved"
    rc, _, err = run(["solve", "--table", str(root / "table.json"), "--budget-bits", "6", "--out", str(out)])
    assert rc == 0, err
    spec = load(out / "grid.json")
    recipe = yaml.safe_load((out / "recipe.yaml").read_text(encoding="utf-8"))
    assert spec["mma"] == recipe["defaults"]["mma"] == "fp32_fma"
    assert from_json(out / "grid.json") == (out / "recipe.yaml").read_text(encoding="utf-8")


def test_the_tune_loop_keeps_the_mma_and_refuses_another(tmp_path: Path) -> None:
    table = _table(tmp_path, floor=0.001)
    data = json.loads(table.read_text())
    data["meta"]["mma"] = "fp32_fma"
    table.write_text(json.dumps(data))
    ledger, record = _ledger(tmp_path)
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"default": "nvfp4", "grid": qr.uniform("nvfp4"), "mma": "fp32_fma"}))
    spec = ta.load_grid(path)
    assert spec["mma"] == "fp32_fma" and ta._recipe("x", spec)["defaults"]["mma"] == "fp32_fma"
    start = ta.propose(tmp_path / "s", "start", spec, table, Fraction(9, 2), "start", [record], ledger)
    _measure(tmp_path, ledger, {"start": (start["recipe_sha256"], 0.30)}, "m0")
    config = {"session": "s", "table": "table.json", "budget_bits": "9/2", "ledger": "ledger.jsonl"}
    reply = tl.check_reply(tl.evolution(config, tmp_path)(tl.analyze(config, tmp_path), 1, 2))
    for item in reply["proposals"]:
        tl.materialize(config, tmp_path, item, "evolution")
        child = ta.read_proposal(tmp_path / "s", item["name"])
        assert child["spec"]["mma"] == "fp32_fma", item["name"]
    for other in (None, "fp64"):
        with pytest.raises(ta.TuneError, match="is not the table's"):
            ta.evaluate({**spec, "mma": other}, table, Fraction(9, 2))
