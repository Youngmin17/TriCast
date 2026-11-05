"""Option-transfer rows of the Auto-tune bundle (``export_autotune_demo --transfer``): a knapsack solution
with a session's options and the same cells gets the KL of its exact recipe file (the tune loop's proposal
file) and that recipe's PPL; a grid whose cells are not its base solution's is refused, and so is a file
nobody measured. CPU only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app import autotune as at
from app import export_autotune_demo as ed
from app import quantizer_recipe as qr
from app import tune_agent as ta
from app.tests.test_autotune_demo import _kl_file, _ppl_file
from app.tests.test_tune_agent import _table

SR = {"nvfp4": {"weight.rounding": "sr"}}


def _transfer(root: Path, base: Path, name: str, grid_edit=None) -> str:
    spec = ta.load_grid(base / "grid.json")
    spec["options"] = SR
    if grid_edit:
        grid_edit(spec["grid"])
    folder = root / name
    folder.mkdir(parents=True)
    solution = json.loads((base / "grid.json").read_text())["name"]
    record = {"name": name, "base": solution, "options_from": "best", **spec}
    (folder / "grid.json").write_text(json.dumps(record))
    return hashlib.sha256((qr.to_yaml(ta._recipe(name, spec)) + "\n").encode()).hexdigest()


def test_transfer_rows_attach_to_their_exact_recipe_and_keep_the_cells(tmp_path: Path) -> None:
    table = _table(tmp_path, floor=0.001)
    run = tmp_path / "run"
    solve = ["solve", "--table", str(table), "--budget-bits", "9/2", "--out", str(run / "auto_4.50")]
    assert at.main(solve) == 0
    kl, ppl, _ = ed.measurements(run)
    solved = ed.solutions(run, kl, ppl, ed.table_bundle(table)["sha256"])
    xfer = tmp_path / "xfer"
    same = _transfer(xfer, run / "auto_4.50", "xfer_same")

    def moved(grid: list[list[str]]) -> None:
        grid[0][0] = "bf16" if grid[0][0] != "bf16" else "nvfp4"
    other = _transfer(xfer, run / "auto_4.50", "xfer_moved", moved)
    _transfer(xfer, run / "auto_4.50", "xfer_unmeasured")
    (xfer / "out_a").mkdir()
    _kl_file(xfer / "out_a" / "kl_seed43.json", [
        {"name": "xfer_same", "recipe_sha256": "r_same", "file_sha256": same, "kl": 0.15},
        {"name": "xfer_moved", "recipe_sha256": "r_moved", "file_sha256": other, "kl": 0.1}])
    _ppl_file(xfer / "out_a" / "ppl" / "xfer_same.json", "r_same", 22.0, "xfer_same")
    rows = {row["id"]: row for row in ed.transfers(xfer, solved)}
    good, bad, none = rows["xfer_same"], rows["xfer_moved"], rows["xfer_unmeasured"]
    assert good["kl"] == 0.15 and good["ppl"] == 22.0 and not good["rejected"]
    assert good["bits"] == solved[0]["bits"] and good["base"] == solved[0]["id"]
    assert good["options_changed"] == SR
    assert good["kl_record"]["source"] == "kl_seed43.json"
    assert bad["kl"] is None and "칸이 다릅니다" in bad["rejected"][0] and len(bad["rejected"]) == 2
    assert none["kl"] is None and "측정 기록 없음" in none["rejected"][0]
