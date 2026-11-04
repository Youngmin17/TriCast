"""Track C's compute budget in the tune loop: proposals, neighbours and every arm keep the bit budget and the
compute budget (average MAC weight of the B200 GEMMs, BF16 = 1) and never choose an unrated method. Tables,
ledgers and GPUs are fakes; CPU only."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_agent import _grid
from app.tests.test_tune_loop import _session

UNRATED = {"nvint4", "bfp12"}  # of the fixture table's methods
FP4 = {"nvfp4", "mxfp4"}


def _grid_of(item: dict, root: Path, config: dict) -> list[list[str]]:
    grid = json.loads(json.dumps(ta.read_proposal(root / config["session"], item["base"])["spec"]["grid"]))
    for change in item["changes"]:
        grid[change["block"]][qr.KINDS.index(change["kind"])] = change["method"]
    return grid


def _fits(spec: dict, root: Path, bits: Fraction, compute: Fraction) -> dict:
    return ta.evaluate(spec, root / "table.json", bits, compute)


def test_evaluate_and_propose_bound_the_compute(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    start = ta.read_proposal(tmp_path / "s", "start")["spec"]
    computed = _fits(start, tmp_path, Fraction(9, 2), Fraction(1, 4))
    assert computed["avg_compute"] == 0.25 and computed["compute_budget_fraction"] == "1/4"
    assert "avg_compute" not in ta.evaluate(start, tmp_path / "table.json", Fraction(9, 2))
    grid = qr.uniform("nvfp4")
    grid[0][qr.KINDS.index("k_proj")] = "mxfp8"  # a BF16 -> FP8 GEMM on one 1024 x 1024 layer
    spec = ta.load_grid(_grid(tmp_path, grid))
    with pytest.raises(ta.TuneError, match="exceeds the compute budget 0.25"):
        _fits(spec, tmp_path, Fraction(5), Fraction(1, 4))
    assert _fits(spec, tmp_path, Fraction(5), Fraction(5, 16))["avg_compute"] > 0.25
    grid[0][qr.KINDS.index("k_proj")] = "nvint4"
    with pytest.raises(ta.TuneError, match="nvint4 has no sourced B200 Tensor Core rate"):
        _fits(ta.load_grid(_grid(tmp_path, grid)), tmp_path, Fraction(5), Fraction(1))
    record = tl.analyze(config, tmp_path)["best"]["kl_record"]
    with pytest.raises(ta.TuneError, match="exceeds the compute budget"):
        ta.propose(tmp_path / "s", "over", spec, tmp_path / "table.json", Fraction(5), "start KL", [record],
                   ledger, parent="start", compute_budget=Fraction(1, 4))
    kept = ta.propose(tmp_path / "s", "fits", spec, tmp_path / "table.json", Fraction(5), "start KL",
                      [record], ledger, parent="start", compute_budget=Fraction(5, 16))
    assert kept["avg_compute_fraction"] == computed_fraction(spec, tmp_path)


def computed_fraction(spec: dict, root: Path) -> str:
    _, _, layers, _ = ta.load_full_table(root / "table.json")
    work = sum((layer.params * ta.method_compute(spec["grid"][layer.block][qr.KINDS.index(layer.kind)])
                for layer in layers), Fraction(0))
    return str(work / sum(layer.params for layer in layers))


def test_neighbours_fit_both_budgets_fund_compute_and_skip_unrated(tmp_path: Path) -> None:
    _session(tmp_path)
    start = ta.read_proposal(tmp_path / "s", "start")["spec"]
    tight = ta.neighbors(start, tmp_path / "table.json", Fraction(5), 50, Fraction(1, 4))
    assert tight and all(method in FP4 for move in tight for row in move["grid"] for method in row)
    loose = ta.neighbors(start, tmp_path / "table.json", Fraction(5), 200, Fraction(5, 16))
    assert {move["to"] for move in loose} >= {"mxfp8", "int8", "bf16"}
    for move in tight + loose:
        assert not UNRATED & {method for row in move["grid"] for method in row}
        _fits({**start, "grid": move["grid"]}, tmp_path, Fraction(5), Fraction(5, 16))
    # a grid at its compute budget: an upgrade to INT8 is paid by moving another INT8 layer down to FP4
    grid = qr.uniform("nvfp4")
    for kind in ("k_proj", "v_proj"):
        grid[0][qr.KINDS.index(kind)] = "int8"
    spec = ta.load_grid(_grid(tmp_path, grid))
    budget = Fraction(computed_fraction(spec, tmp_path))
    moves = ta.neighbors(spec, tmp_path / "table.json", Fraction(5), 400, budget)
    funded = [move for move in moves if move["to"] == "int8" and move["funded_by"]]
    assert funded
    for move in moves:
        _fits({**spec, "grid": move["grid"]}, tmp_path, Fraction(5), budget)


def test_every_arm_keeps_both_budgets_and_the_analysis_shows_the_compute(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    with pytest.raises(tl.LoopError, match="drop_unrated"):
        tl.analyze({**config, "compute_budget": "5/16"}, tmp_path)
    config = {**config, "compute_budget": "5/16", "drop_unrated": True}
    analysis = tl.analyze(config, tmp_path)
    compute = analysis["compute"]
    assert compute["budget"] == "5/16" and compute["best"] == 0.25
    assert set(compute["unrated_left_out"]) == UNRATED and not UNRATED & set(compute["weights"])
    assert compute["weights"]["mxfp8"] == 0.5 and compute["weights"]["bf16"] == 1.0
    assert all(row["best_alternative"] not in UNRATED for row in analysis["loss_concentration"])
    assert all("compute_delta_per_mac" in row for row in analysis["loss_concentration"])
    arms = {"evolution": tl.evolution(config, tmp_path, seed=1),
            "evolution-table": tl.evolution(config, tmp_path, seed=1, guided=True),
            "random": tl.random_search(config, tmp_path, 1), "auto-greedy": tl.auto_greedy(config, tmp_path)}
    for label, arm in arms.items():
        reply = tl.check_reply(arm(analysis, 1, 8))
        assert reply["proposals"], label
        for item in reply["proposals"]:
            grid = _grid_of(item, tmp_path, config)
            assert not UNRATED & {method for row in grid for method in row}, label
            spec = {**ta.read_proposal(tmp_path / "s", item["base"])["spec"], "grid": grid,
                    "options": item.get("options") or None}
            _fits(spec, tmp_path, Fraction(9, 2), Fraction(5, 16))
        tl.materialize(config, tmp_path, reply["proposals"][0], label)  # propose checks the compute too


def test_tpe_keeps_both_budgets(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    config, _ = _session(tmp_path)
    config = {**config, "compute_budget": "5/16", "drop_unrated": True}
    reply = tl.check_reply(tl.tpe_search(config, tmp_path, 0)(tl.analyze(config, tmp_path), 1, 8))
    assert reply["proposals"]
    for item in reply["proposals"]:
        grid = _grid_of(item, tmp_path, config)
        assert not UNRATED & {method for row in grid for method in row}
        spec = {**ta.read_proposal(tmp_path / "s", item["base"])["spec"], "grid": grid,
                "options": item.get("options") or None}
        _fits(spec, tmp_path, Fraction(9, 2), Fraction(5, 16))
