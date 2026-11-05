"""The replan proposer (the agent's options, then the knapsack again): the session table with a column table
measured under the best grid's options merged in, solved again at the session budget, gives one proposal on
that grid with its options kept and the cells solve moves; it goes through the usual checks and measurement.
A column table measured with other options is refused. GPU and LLM are fakes; CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import autotune as at
from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_loop import _measure, _session

SR = {"nvfp4": {"weight.rounding": "sr"}}
MOVED = {(block, "q_proj") for block in range(4)}  # nvfp4 is made bad there under the options


def _setup(tmp_path: Path, column_options: dict) -> tuple[dict, Path, Path]:
    config, ledger = _session(tmp_path)
    record = tl.analyze(config, tmp_path)["best"]["kl_record"]
    item = {"name": "opt", "base": "start", "changes": [], "options": SR, "rationale": "x", "cites": [record]}
    opt = tl.materialize(config, tmp_path, item, "test")
    _measure(tmp_path, ledger, {"opt": (opt["recipe_sha256"], 0.2)}, "m1")
    column = json.loads((tmp_path / "table.json").read_text())
    column["meta"]["options"] = column_options
    column["meta"]["methods"] = ["bf16", "nvfp4"]
    for row in column["layers"]:
        bad = (row["block"], row["kind"]) in MOVED
        row["kl"] = {"bf16": 0.0, "nvfp4": 1.0 if bad else row["kl"]["nvfp4"]}
    path = tmp_path / "column.json"
    path.write_text(json.dumps(column))
    return config, ledger, path


def test_replan_keeps_the_best_grids_options_and_moves_the_cells_solve_moves(tmp_path: Path) -> None:
    config, ledger, column = _setup(tmp_path, SR)
    reply = tl.replan(config, tmp_path, [column])(tl.analyze(config, tmp_path), 7, 1)
    (item,) = reply["proposals"]
    assert item["base"] == "opt" and item["options"] is None and item["name"] == "replan_r7"
    folder = tmp_path / "s" / "loop" / "replan" / "round_7"
    merged, _, _, _ = at.load_table(folder / "table.json")
    assert merged["meta"]["options"] == SR
    grid = json.loads((folder / "solve" / "grid.json").read_text())["grid"]
    assert item["changes"]
    assert all(grid[c["block"]][qr.KINDS.index(c["kind"])] == c["method"] for c in item["changes"])
    assert {(c["block"], c["kind"]) for c in item["changes"]} >= MOVED  # every bad nvfp4 cell moves

    def measure(number: int, names: list[str]) -> Path:
        shas = {name: ta.read_proposal(tmp_path / "s", name)["recipe_sha256"] for name in names}
        return _measure(tmp_path, ledger, {name: (shas[name], 0.15) for name in names}, f"r{number}")

    result = tl.run(config, tmp_path, 1, 1, tl.replan(config, tmp_path, [column]), "knapsack-replan", measure,
                    log=lambda _: None)
    (entry,) = result["rounds"]
    assert entry["accepted"] == ["replan_r1"] and entry["improved"] and result["best"] == "replan_r1"
    record = ta.read_proposal(tmp_path / "s", "replan_r1")
    assert record["author"] == "knapsack-replan" and record["spec"]["options"] == SR


def test_a_column_measured_with_other_options_is_refused(tmp_path: Path) -> None:
    config, _, column = _setup(tmp_path, {"nvfp4": {"weight.rounding": "rtz"}})
    with pytest.raises(tl.LoopError, match="other options than opt's"):
        tl.replan(config, tmp_path, [column])(tl.analyze(config, tmp_path), 1, 1)
