# AI 작성 — 2026-11-09 사용자 승인 (권고안 "AC24 · AC28 골든 케이스 추가"). 수정은 사람이 승인한다.
"""SPEC AC24 (proposal checks) and AC28 (the code's checks of a parsed tune request) golden cases on the demo
data kept in the repository: tests/harness/golden_tune.yaml. No LLM call, no GPU."""

from __future__ import annotations

import copy
import json
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import yaml
from app import autotune as at
from app import export_autotune_demo as ed
from app import tune_agent as ta
from app import tune_loop as tl
from app import tune_request as tr

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "evals" / "fixtures" / "tune_inputs"
SESSION = INPUTS / "s3150"
REPLIES = ROOT / "evals" / "fixtures" / "tune"
MODEL = "Qwen/Qwen3-0.6B"
CASES = yaml.safe_load((ROOT / "tests/harness/golden_tune.yaml").read_text(encoding="utf-8"))["cases"]
CASE_KEYS = {"id", "ac", "kind", "input", "expected", "note"}
EMPTY_REPLY = {"outcome": "tune", "model": None, "budget_bits": None, "compute_budget": None,
               "keep_bf16_blocks": [], "exclude_methods": [], "only_methods": None, "questions": [],
               "reason": None, "assumptions": []}


def test_tune_golden_manifest() -> None:
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids)), ids
    assert {case["ac"] for case in CASES} == {"AC24", "AC28"}
    for case in CASES:
        assert case.keys() == CASE_KEYS, case["id"]
        assert case["kind"] in {"proposal", "session", "request"}, case["id"]
        assert case["ac"] in case["note"], case["id"]
        assert (case["ac"] == "AC28") == (case["kind"] == "request"), case["id"]


def _item(data: dict[str, Any]) -> dict[str, Any]:
    if "from_session" in data:
        record = json.loads((SESSION / "proposals" / data["from_session"] / "proposal.json").read_text())
        item = {"name": record["name"], "base": record["parent"], "options": record["spec"]["options"],
                "cites": record["cites"], "rationale": record["rationale"], "changes": []}
    else:
        source = data["from_reply"]
        kept = json.loads((REPLIES / source["run"] / f"{source['case']}.json").read_text(encoding="utf-8"))
        reply = next(call for call in kept["calls"] if call["role"] == "proposer")["stdout"]
        item = next(row for row in reply["structured_output"]["proposals"]
                    if row["name"] == source["proposal"])
    return {**copy.deepcopy(item), **copy.deepcopy(data.get("edit") or {})}


def _proposal(case: dict[str, Any], tmp_path: Path) -> None:
    data, expected = case["input"], case["expected"]
    session = tmp_path / "s"
    (session / "proposals").mkdir(parents=True)
    shutil.copyfile(SESSION / "session.json", session / "session.json")
    for name in data["session"]:
        shutil.copytree(SESSION / "proposals" / name, session / "proposals" / name)
    config = {"session": str(session), "table": str(INPUTS / "table.json"),
              "budget_bits": str(data.get("budget_bits", "403/128")), "ledger": str(INPUTS / "ledger.jsonl")}
    item = _item(data)
    if "refused" in expected:
        with pytest.raises(ta.TuneError) as refused:
            tl.materialize(config, tmp_path, item, "golden")
        assert expected["refused"] in str(refused.value), str(refused.value)
        assert not (session / "proposals" / item["name"]).exists()
        return
    assert expected["accepted"] is True
    proposal = tl.materialize(config, tmp_path, item, "golden")
    assert proposal["recipe_sha256"] == expected["recipe_sha256"]
    assert proposal["avg_bits_fraction"] == expected["avg_bits_fraction"]


def _session(case: dict[str, Any]) -> None:
    fixed = json.loads((SESSION / "session.json").read_text(encoding="utf-8"))
    table = json.loads((INPUTS / "table.json").read_text(encoding="utf-8"))
    expected = dict(case["expected"])
    assert fixed["model_id"] == table["meta"]["model_id"] == expected.pop("model_id")
    for key, value in expected.items():
        assert fixed["measure"][key] == value, key


def _solved(table: dict[str, Any], request: dict[str, Any]) -> None:
    """The knapsack (app.autotune.solve_mckp) on the constrained table keeps every request constraint."""
    limited = tr.constrained(table, request)
    rows = [[j for j, value in enumerate(layer["value"]) if value is not None] for layer in limited["layers"]]
    costs = [[at.weight_bits(limited["methods"][j], layer["in_features"]) * layer["params"] for j in row]
             for layer, row in zip(limited["layers"], rows, strict=True)]
    values = [[layer["value"][j] for j in row] for layer, row in zip(limited["layers"], rows, strict=True)]
    params = sum(layer["params"] for layer in limited["layers"])
    solution = at.solve_mckp(costs, values, Fraction(request["budget_bits"]) * params)
    chosen = [limited["methods"][row[k]] for row, k in zip(rows, solution.choice, strict=True)]
    layers = limited["layers"]
    bits = sum((at.weight_bits(method, layer["in_features"]) * layer["params"]
                for method, layer in zip(chosen, layers, strict=True)), Fraction(0)) / params
    assert bits <= Fraction(request["budget_bits"])
    kept = {block % limited["blocks"] for block in request["keep_bf16_blocks"]}
    pairs = zip(chosen, layers, strict=True)
    assert all(method == "bf16" for method, layer in pairs if layer["block"] in kept)
    assert not set(chosen) & set(request["exclude_methods"])
    if request["only_methods"]:
        assert set(chosen) <= set(request["only_methods"]) | {"bf16"}


def _request(case: dict[str, Any]) -> None:
    table = ed.table_bundle(INPUTS / "table.json")
    reply = {**EMPTY_REPLY, **copy.deepcopy(case["input"]["reply"])}
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(tr.RequestError) as refused:
            tr.check(reply, table, [MODEL])
        assert expected["error"] in str(refused.value), str(refused.value)
        return
    result = tr.check(reply, table, [MODEL])
    assert result["outcome"] == expected["outcome"], result
    if "reason_has" in expected:
        assert expected["reason_has"] in result["reason"], result["reason"]
    if expected.get("questions"):
        assert any(question.strip() for question in result["questions"]), result
    if expected.get("solved"):
        _solved(table, result)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_tune_golden(case: dict[str, Any], tmp_path: Path) -> None:
    if case["kind"] == "proposal":
        _proposal(case, tmp_path)
    elif case["kind"] == "session":
        _session(case)
    else:
        _request(case)


_EVALUATE = ta.evaluate


def _no_budget(spec: dict[str, Any], table_path: Path, budget: Fraction,
               compute_budget: Fraction | None = None) -> dict[str, Any]:
    return _EVALUATE(spec, table_path, Fraction(99), compute_budget)


# a broken check and the cases that must then fail: the goldens are evidence only if they catch these
MUTATIONS = [
    ("numbers check off", ta, "unsupported_numbers", lambda text, values: [],
     {"tune_proposal_number_not_in_record", "tune_reply_predicted_value_as_measured",
      "tune_reply_floor_subtracted_value", "tune_reply_other_proposals_parameter"}),
    ("size conflict off", tr, "cheapest_bits", lambda table: Fraction(0),
     {"tune_request_budget_below_any_assignment", "tune_request_only_int8_conflict",
      "tune_request_half_the_blocks_bf16_conflict"}),
    ("compute conflict off", tr, "cheapest_compute", lambda table: Fraction(0),
     {"tune_request_compute_below_any_assignment"}),
    ("budget check off", ta, "evaluate", _no_budget,
     {"tune_proposal_over_budget_by_cells", "tune_proposal_over_budget_by_session_budget"}),
    ("constraints ignored", tr, "constrained", lambda table, request: table,
     {"tune_request_only_int8_conflict", "tune_request_half_the_blocks_bf16_conflict",
      "tune_request_solved_within_constraints", "tune_request_solved_fraction_budget"}),
]


@pytest.mark.parametrize("module, name, broken, expected", [row[1:] for row in MUTATIONS],
                         ids=[row[0] for row in MUTATIONS])
def test_the_goldens_catch_a_broken_check(module: Any, name: str, broken: Any, expected: set[str],
                                          tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, name, broken)
    failed = set()
    for case in CASES:
        if case["id"] not in expected:
            continue
        folder = tmp_path / case["id"]
        folder.mkdir()
        try:
            test_tune_golden(case, folder)
        except (Exception, pytest.fail.Exception):
            failed.add(case["id"])
    assert failed == expected
