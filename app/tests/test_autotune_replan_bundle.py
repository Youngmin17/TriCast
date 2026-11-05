"""A round with no LLM reply on file (the replan proposer) shows the cells its proposal moves: the bundle
takes them from the recorded grid against its parent's, the same cells the proposer returned. GPU and LLM are
fakes; CPU only."""

from __future__ import annotations

from pathlib import Path

from app import export_autotune_demo as ed
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_loop import _measure
from app.tests.test_tune_replan import SR, _setup


def test_a_replan_round_in_the_bundle_lists_the_cells_it_moves(tmp_path: Path) -> None:
    config, ledger, column = _setup(tmp_path, SR)
    returned = tl.replan(config, tmp_path, [column])(tl.analyze(config, tmp_path), 1, 1)["proposals"][0]

    def measure(number: int, names: list[str]) -> Path:
        shas = {name: ta.read_proposal(tmp_path / "s", name)["recipe_sha256"] for name in names}
        return _measure(tmp_path, ledger, {name: (shas[name], 0.15) for name in names}, f"r{number}")

    tl.run(config, tmp_path, 1, 1, tl.replan(config, tmp_path, [column]), "knapsack-replan", measure,
           log=lambda _: None)
    bundle = ed.session_bundle(tmp_path / "s", ledger)
    (proposal,) = bundle["rounds"][-1]["proposals"]
    assert proposal["name"] == "replan_r1" and proposal["author"] == "knapsack-replan"
    assert bundle["rounds"][-1]["llm"] is None
    assert proposal["changes"] and proposal["changes"] == returned["changes"]
    assert proposal["base"] == "opt"
