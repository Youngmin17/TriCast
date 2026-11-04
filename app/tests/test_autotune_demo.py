"""The Auto-tune page's data and live round: the bundle export (the solver's table form, measured numbers
attached only to the exact recipe files, the session's rounds), one live proposer round checked in a scratch
copy of the session, and the server route. The LLM and GPU are fakes; CPU only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import autotune as at
from app import autotune_live as live
from app import export_autotune_demo as ed
from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.server import create_app
from app.tests.test_tune_agent import ENV, _table
from app.tests.test_tune_loop import _session


def _kl_file(path: Path, rows: list[dict]) -> None:
    meta = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}}
    path.write_text(json.dumps({"schema": at.MEASURE_SCHEMA, "status": "complete", "meta": meta,
                                "results": rows}))


def _ppl_file(path: Path, recipe_hash: str | None, ppl: float, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "complete", "recipe_hash": recipe_hash,
                                "recipe": None if recipe_hash is None else {"name": name},
                                "metrics": {"ppl": {"ppl": ppl, "n_tokens": 1000}}, "env": ENV}))


def test_the_table_is_the_solvers_and_measurements_attach_only_to_their_exact_recipe(tmp_path: Path) -> None:
    table = _table(tmp_path, floor=0.001)
    bundle = ed.table_bundle(table)
    raw, _, layers, methods = at.load_table(table)
    assert bundle["methods"] == methods and bundle["objective"] == "kl_minus_floor"
    values = at.objective(raw, True)
    assert all(row["value"] == [values[row["name"]][m] for m in methods] for row in bundle["layers"])
    assert bundle["weight_bits"]["nvfp4"] == [4, 8, 16]
    run = tmp_path / "run"
    run.mkdir()
    out = run / "auto_4.50"
    assert at.main(["solve", "--table", str(table), "--budget-bits", "9/2", "--out", str(out)]) == 0
    spec = json.loads((out / "grid.json").read_text())
    assert ed.recipe_file_sha256(spec) == hashlib.sha256((out / "recipe.yaml").read_bytes()).hexdigest()

    def uniform(method: str) -> str:
        return ed.recipe_file_sha256({"name": f"u_{method}", "default": method, "grid": qr.uniform(method),
                                      "kv": "none"})
    _kl_file(run / "kl_seed43_auto.json", [
        {"name": "auto_4.50", "recipe_sha256": "r_auto", "file_sha256": ed.recipe_file_sha256(spec),
         "kl": 0.2},
        {"name": "u_mxfp8", "recipe_sha256": "r_u", "file_sha256": uniform("mxfp8"), "kl": 0.05},
        {"name": "u_nvfp4", "recipe_sha256": "r_n", "file_sha256": "not the generator's file", "kl": 0.3},
        {"name": "u_int8", "recipe_sha256": "r_i", "file_sha256": uniform("int8"), "kl": 0.01,
         "calibration_overrides": {"seed": 44}}])
    _ppl_file(run / "ppl_auto" / "auto_4.50.json", "r_auto", 22.5, "auto_4.50")
    _ppl_file(run / "ppl_auto" / "native.json", None, 20.9, "native")
    kl, ppl, native = ed.measurements(run)
    assert native["ppl"] == 20.9 and set(ppl) == {"r_auto"}
    rows = ed.baselines(bundle, kl, ppl, layers)
    assert [row["id"] for row in rows] == ["u_mxfp8"]  # nvfp4: other file; int8: calibration override
    assert rows[0]["kl"] == 0.05 and rows[0]["ppl"] is None and "test PPL 기록 없음" in rows[0]["rejected"][0]
    (solution,) = ed.solutions(run, kl, ppl, bundle["sha256"])
    assert solution["kl"] == 0.2 and solution["ppl"] == 22.5 and not solution["rejected"]
    assert solution["grid"] == spec["grid"] and Fraction(solution["bits"]) <= Fraction(9, 2)
    assert ed.solutions(run, kl, ppl, "another table") == []


def _round(tmp_path: Path, config: dict, items: list[dict]) -> dict:
    def runner(command, **kwargs):
        reply = {"analysis": "live", "stop": False, "proposals": items}
        out = json.dumps({"type": "result", "structured_output": reply, "total_cost_usd": 0.5,
                          "duration_ms": 1000, "modelUsage": {"claude-opus-5-5": {}}})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")
    return live.propose_round(config, tmp_path, len(items), runner)


def test_a_live_round_checks_every_proposal_in_a_scratch_copy(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    record = tl.analyze(config, tmp_path)["best"]["kl_record"]
    before = sorted(path.relative_to(tmp_path) for path in (tmp_path / "s").rglob("*"))
    good = {"name": "live1", "base": "start", "options": None, "rationale": "start KL", "cites": [record],
            "changes": [{"block": 0, "kind": "down_proj", "method": "mxfp4"}]}
    over = {**good, "name": "live2", "changes": [{"block": 0, "kind": "down_proj", "method": "bf16"}]}
    made_up = {**good, "name": "live3", "cites": ["E000000000000"],
               "changes": [{"block": 1, "kind": "down_proj", "method": "mxfp4"}]}
    result = _round(tmp_path, config, [good, over, made_up])
    assert result["mode"] == "live" and result["round"] == 1 and result["llm"]["cost_usd"] == 0.5
    status = {row["name"]: (row["status"], row["reason"]) for row in result["proposals"]}
    assert status["live1"] == ("accepted", None)
    assert "exceed the budget" in status["live2"][1] and "not in the ledger" in status["live3"][1]
    accepted = result["proposals"][0]
    assert accepted["grid"][0][qr.KINDS.index("down_proj")] == "mxfp4" and accepted["kl"] is None
    assert sorted(path.relative_to(tmp_path) for path in (tmp_path / "s").rglob("*")) == before  # untouched
    with pytest.raises(live.LiveError, match="count"):
        live.propose_round(config, tmp_path, 0)
    assert live._BUSY.acquire(blocking=False)
    try:
        with pytest.raises(live.LiveError) as busy:
            _round(tmp_path, config, [good])
        assert busy.value.status == 409
    finally:
        live._BUSY.release()


def test_the_route_needs_agent_and_answers_with_the_checked_round(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "catalog.json").write_text("{}")
    off = TestClient(create_app("demo", "none", demo)).post("/api/autotune/propose", json={"count": 2})
    assert off.status_code == 503 and off.json()["error"]["code"] == "agent_unavailable"
    assert "--agent" in off.json()["error"]["message"]
    record = tl.analyze(config, tmp_path)["best"]["kl_record"]
    item = {"name": "route1", "base": "start", "changes": [], "rationale": "start KL", "cites": [record],
            "options": {"nvfp4": {"weight.rounding": "rtz"}}}

    def runner(command, **kwargs):
        out = json.dumps({"type": "result", "structured_output": {"analysis": "a", "stop": False,
                                                                   "proposals": [item]}})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")
    loop = tmp_path / "loop.json"
    paths = {"session": tmp_path / "s", "table": tmp_path / "table.json", "ledger": tmp_path / "ledger.jsonl"}
    loop.write_text(json.dumps({**config, **{key: str(path) for key, path in paths.items()}}))
    client = TestClient(create_app("demo", "none", demo, agent=loop, agent_runner=runner))
    answer = client.post("/api/autotune/propose", json={"count": 1})
    assert answer.status_code == 200, answer.text
    (row,) = answer.json()["proposals"]
    assert row["status"] == "accepted" and row["predicted"]["unpredicted"]  # options: no table prediction
    assert client.post("/api/autotune/propose", json={"count": 9}).status_code == 400
    assert not ta.state(tmp_path / "s", tmp_path / "ledger.jsonl")["proposals"][1:]  # only the start remains
