"""The information-matched controls of the tune loop (protocol §6): evolution whose mutations read the table
(``evolution-table``) and Claude proposals that the table selects (``claude-table-select``). Tables, ledgers
and the LLM are fakes; CPU only."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_loop import _session


def _predicted(config: dict, root: Path, item: dict) -> float:
    spec = json.loads(json.dumps(ta.read_proposal(root / config["session"], item["base"])["spec"]))
    for change in item["changes"]:
        spec["grid"][change["block"]][qr.KINDS.index(change["kind"])] = change["method"]
    return ta.evaluate(spec, root / config["table"], Fraction(config["budget_bits"]))["predicted"][
        "predicted_kl_sum_minus_floor"]


def test_table_guided_evolution_takes_tabled_moves_and_passes_the_checks(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    guided = tl.check_reply(tl.evolution(config, tmp_path, seed=2, guided=True)(analysis, 1, 12))
    plain = tl.check_reply(tl.evolution(config, tmp_path, seed=2)(analysis, 1, 12))
    assert len(guided["proposals"]) == len(plain["proposals"]) == 12
    assert guided["analysis"] == "evolution-table" and plain["analysis"] == "evolution"
    assert all(item["name"].startswith("evolution-table_r1_") for item in guided["proposals"])
    again = tl.evolution(config, tmp_path, seed=2, guided=True)(analysis, 1, 12)["proposals"]
    assert again == guided["proposals"]  # seeded
    start = ta.read_proposal(tmp_path / "s", "start")["spec"]
    tabled = [move["grid"] for move in ta.neighbors(start, tmp_path / "table.json", Fraction(9, 2),
                                                    tl.GUIDED_MOVES)]
    for item in guided["proposals"]:
        grid = json.loads(json.dumps(start["grid"]))
        for change in item["changes"]:
            grid[change["block"]][qr.KINDS.index(change["kind"])] = change["method"]
        assert grid in tabled, item["name"]  # the cells are one tabled move (options aside)
    # the table's information shows: lower predicted KL than random switches from the same parent
    ours = sorted(_predicted(config, tmp_path, item) for item in guided["proposals"] if not item["options"])
    theirs = sorted(_predicted(config, tmp_path, item) for item in plain["proposals"] if not item["options"])
    assert ours and theirs and ours[len(ours) // 2] < theirs[len(theirs) // 2]
    for item in guided["proposals"][:4]:  # every check of propose passes
        row = tl.materialize(config, tmp_path, item, "evolution-table")
        assert Fraction(row["avg_bits_fraction"]) <= Fraction(9, 2)


def test_claude_table_select_measures_the_tables_best_and_keeps_the_rest_last(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    cite = [analysis["best"]["kl_record"]]

    def item(name: str, changes: list, options: dict | None = None) -> dict:
        return {"name": name, "base": "start", "changes": changes, "options": options, "cites": cite,
                "rationale": "start KL"}

    cells = [{"block": b, "kind": "q_proj", "method": "mxfp4"} for b in range(4)]
    proposals = [item("one", cells[:1]), item("four", cells), item("two", cells[:2]),
                 item("opts", cells[:1], {"nvfp4": {"weight.rounding": "rtz"}}),
                 item("bad", [{"block": 99, "kind": "q_proj", "method": "mxfp4"}])]
    asked = []

    def proposer(analysis: dict, number: int, count: int) -> dict:
        asked.append(count)
        return {"analysis": "llm", "stop": False, "proposals": proposals}

    predicted = {}
    for row in proposals[:3]:
        spec = json.loads(json.dumps(ta.read_proposal(tmp_path / "s", "start")["spec"]))
        for change in row["changes"]:
            spec["grid"][change["block"]][qr.KINDS.index(change["kind"])] = change["method"]
        predicted[row["name"]] = ta.evaluate(spec, tmp_path / "table.json", Fraction(9, 2))["predicted"][
            "predicted_kl_sum_minus_floor"]
    by_table = sorted(predicted, key=predicted.get)
    reply = tl.table_select(proposer, config, tmp_path)(analysis, 1, 2)
    assert asked == [6]  # factor 3 x the round's count
    assert [row["name"] for row in reply["proposals"]] == by_table[:2]
    assert "table-select 2 of 5" in reply["analysis"]
    everything = tl.table_select(proposer, config, tmp_path)(analysis, 1, 5)["proposals"]
    assert [row["name"] for row in everything] == [*by_table, "opts", "bad"]  # unpredicted last, in order
