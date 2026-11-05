"""Tables measured with method options and merged into another table: table --options reaches the method
recipes (other recipe sha256, other KL) and leaves the patch floors bit for bit; merge puts a column table's
methods in place of the base's only for tables of the same measurement (equal floors, same settings); solve on
a table with options writes them into grid.json and recipe.yaml. Tiny random Llama on CPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app import autotune as at
from app.tests.test_autotune_v2 import load, make_root, run, table_run

ARGS = ("--calibrate", "--methods", "bf16,nvfp4,gptq_int4", "--blocks", "1", "--calib-samples", "4",
        "--calib-seqlen", "64")
ACT = {"weight_algo.act_order": True, "weight_algo.damp": 0.05}


@pytest.fixture(scope="module")
def tables(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    pytest.importorskip("transformers")
    root = make_root(tmp_path_factory.mktemp("options"))
    rc, _, stderr, _ = table_run(root, root / "base.json", *ARGS)
    assert rc == 0, stderr
    (root / "options.json").write_text(json.dumps({"gptq_int4": {**ACT, "weight.rounding": "rne"}}))
    rc, _, stderr, _ = table_run(root, root / "act.json", *ARGS, "--options", str(root / "options.json"))
    assert rc == 0, stderr
    return root, load(root / "base.json"), load(root / "act.json")


def test_options_reach_the_method_recipes_and_leave_the_floors(tables: tuple[Path, dict, dict]) -> None:
    _, base, act = tables
    assert "options" not in base["meta"] and act["meta"]["options"] == {"gptq_int4": ACT}  # defaults dropped
    shas = base["meta"]["method_recipe_sha256"], act["meta"]["method_recipe_sha256"]
    assert shas[0]["nvfp4"] == shas[1]["nvfp4"] and shas[0]["gptq_int4"] != shas[1]["gptq_int4"]
    assert [row["floor"] for row in base["layers"]] == [row["floor"] for row in act["layers"]]
    assert [row["kl"]["nvfp4"] for row in base["layers"]] == [row["kl"]["nvfp4"] for row in act["layers"]]
    gptq = [[row["kl"]["gptq_int4"] for row in table["layers"]] for table in (base, act)]
    assert gptq[0] != gptq[1]  # the options fired


def test_merge_takes_the_column_tables_methods_and_solve_writes_their_options(
        tables: tuple[Path, dict, dict], tmp_path: Path) -> None:
    root, base, act = tables
    rc, stdout, stderr = run(["merge", "--base", str(root / "base.json"), "--columns", str(root / "act.json"),
                              "--out", str(tmp_path / "merged.json")])
    assert rc == 0, stderr
    assert "AUTOTUNE_MERGE_DONE" in stdout
    merged = load(tmp_path / "merged.json")
    for row, base_row, act_row in zip(merged["layers"], base["layers"], act["layers"], strict=True):
        assert row["kl"] == {**base_row["kl"], "gptq_int4": act_row["kl"]["gptq_int4"]}
    assert merged["meta"]["options"] == {"gptq_int4": ACT}
    assert [source["methods"] for source in merged["meta"]["merged_from"][1:]] == [["gptq_int4", "nvfp4"]]
    at.load_table(tmp_path / "merged.json")  # a complete table
    rc, _, stderr = run(["solve", "--table", str(tmp_path / "merged.json"), "--budget-bits", "16",
                         "--name", "s", "--out", str(tmp_path / "s")])
    assert rc == 0, stderr
    spec = load(tmp_path / "s" / "grid.json")
    used = {spec["default"], *(m for row in spec["grid"] for m in row)}
    assert spec.get("options", {}) == ({"gptq_int4": ACT} if "gptq_int4" in used else {})
    rc, _, stderr = run(["solve", "--table", str(tmp_path / "merged.json"), "--budget-bits", "9/2",
                         "--name", "low", "--out", str(tmp_path / "low")])
    assert rc == 0, stderr
    low = load(tmp_path / "low" / "grid.json")
    if "gptq_int4" in {m for row in low["grid"] for m in row} | {low["default"]}:
        assert low["options"] == {"gptq_int4": ACT}
        assert "act_order: true" in (tmp_path / "low" / "recipe.yaml").read_text()


def test_merge_refuses_another_measurement(tables: tuple[Path, dict, dict], tmp_path: Path) -> None:
    root, _, act = tables
    other = json.loads(json.dumps(act))
    other["layers"][0]["floor"] += 1e-9
    (tmp_path / "other.json").write_text(json.dumps(other))
    rc, _, stderr = run(["merge", "--base", str(root / "base.json"), "--columns",
                         str(tmp_path / "other.json"), "--out", str(tmp_path / "m.json")])
    assert rc == 2 and "patch floors differ" in stderr
    rc, _, stderr = run(["merge", "--base", str(root / "base.json"), "--columns", str(root / "act.json"),
                         str(root / "act.json"), "--out", str(tmp_path / "m.json")])
    assert rc == 2 and "in two column tables" in stderr


def test_bad_options_are_refused_before_any_model_loads(tmp_path: Path) -> None:
    (tmp_path / "o.json").write_text(json.dumps({"int8": {"weight.rounding": "sr"}}))
    with pytest.raises(at.AutotuneError, match="does not measure"):
        at.load_options(str(tmp_path / "o.json"), ["bf16", "gptq_int4"])
    (tmp_path / "o.json").write_text(json.dumps({"gptq_int4": {"weight_algo.nope": 1}}))
    with pytest.raises(at.AutotuneError, match="unknown options"):
        at.load_options(str(tmp_path / "o.json"), ["bf16", "gptq_int4"])


def test_a_grid_is_written_with_the_options_of_the_methods_it_uses(tmp_path: Path) -> None:
    from app import quantizer_recipe as qr

    grid = qr.uniform("gptq_int4")
    options = {"gptq_int4": ACT, "gptq_int3": {"weight_algo.damp": 0.1}}  # gptq_int3 unused: left out
    at.write_grid(tmp_path, "g", "gptq_int4", grid, "none", qr.check_calibration({}), None, options)
    assert load(tmp_path / "grid.json")["options"] == {"gptq_int4": ACT}
    text = (tmp_path / "recipe.yaml").read_text()
    assert "act_order: true" in text and "damp: 0.05" in text
    at.write_grid(tmp_path / "plain", "g", "gptq_int4", grid, "none", qr.check_calibration({}))
    assert "options" not in load(tmp_path / "plain" / "grid.json")
