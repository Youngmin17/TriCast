"""Autonomous tune loop (app/tune_loop.py) — SPEC AC27: the evidence analysis cites ledger records, refused
proposals and their reasons reach the next round, the stop rule holds, the Claude Code proposer runs without
tools under the proposal schema and is logged, and the remote measurement syncs, launches through gpu_cap.sh,
waits for the marker and fetches the result. LLM, GPU and ssh are replaced by fakes; CPU only."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from fractions import Fraction
from pathlib import Path

import pytest

from app import ledger as lg
from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_agent import ENV, _grid, _ledger, _table


def _session(tmp_path: Path) -> tuple[dict, Path]:
    """A session whose start proposal (uniform NVFP4) has a measured validation KL."""
    table = _table(tmp_path, floor=0.001)
    ledger, record = _ledger(tmp_path)
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    start = ta.propose(tmp_path / "s", "start", spec, table, Fraction(9, 2), "start", [record], ledger)
    _measure(tmp_path, ledger, {"start": (start["recipe_sha256"], 0.30)}, "m0")
    config = {"session": "s", "table": "table.json", "budget_bits": "9/2", "ledger": "ledger.jsonl"}
    return config, ledger


def _measure(tmp_path: Path, ledger: Path, results: dict, name: str) -> Path:
    meta = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}}
    rows = [{"name": key, "recipe_sha256": sha, "kl": kl, "patched": 196}
            for key, (sha, kl) in results.items()]
    path = tmp_path / "runs" / f"{name}.json"
    path.write_text(json.dumps({"schema": "tricast.autotune.measure/2", "meta": meta, "results": rows}))
    lg.ingest([path], ledger, tmp_path)
    return path


def test_the_analysis_cites_the_records_behind_its_numbers(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    ids = {record["id"] for record in lg.read_ledger(ledger)}
    assert analysis["best"]["name"] == "start" and analysis["best"]["kl_record"] in ids
    assert len(analysis["loss_concentration"]) == tl.TOP_LAYERS
    worst = analysis["loss_concentration"][0]
    assert worst["kl_minus_floor"] >= analysis["loss_concentration"][-1]["kl_minus_floor"]
    assert analysis["tabled_moves"] and set(analysis["option_space"]) == {"nvfp4"}
    assert analysis["history"][0]["change_vs_parent"] is None
    json.dumps(analysis)  # the proposer receives it as JSON


def test_the_loop_measures_accepted_proposals_feeds_back_refusals_and_stops(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    seen_refusals, measured_kl = [], iter([0.25, 0.26, 0.27])

    def proposer(analysis: dict, number: int, count: int) -> dict:
        seen_refusals.append([item["name"] for item in analysis["refused_last_round"]])
        best = analysis["best"]["name"]
        cite = [analysis["best"]["kl_record"]]
        return {"analysis": "x", "stop": False, "proposals": [
            # k_proj to int8 (+3.53 bits on 1M weights) paid by gate_proj to MXFP4 (-0.25 bits on 3.1M each)
            {"name": f"ok{number}", "base": best, "options": None, "rationale": "upgrade a k_proj",
             "cites": cite, "changes": [{"block": number, "kind": "k_proj", "method": "int8"}]
             + [{"block": b, "kind": "gate_proj", "method": "mxfp4"} for b in range(5 * number)]},
            {"name": f"fat{number}", "base": best, "options": None, "rationale": "too big", "cites": cite,
             "changes": [{"block": b, "kind": "down_proj", "method": "bf16"} for b in range(28)]},
            {"name": f"same{number}", "base": best, "changes": [], "options": None, "rationale": "repeat",
             "cites": cite}][:count]}

    def measure(number: int, names: list[str]) -> Path:
        state = {row["name"]: row for row in ta.state(tmp_path / "s", ledger)["proposals"]}
        shas = {name: ta.read_proposal(tmp_path / "s", name)["recipe_sha256"] for name in names}
        assert set(names) <= set(state)
        kls = {name: (shas[name], next(measured_kl)) for name in names}
        return _measure(tmp_path, ledger, kls, f"r{number}")

    logs = []
    result = tl.run(config, tmp_path, 5, 3, proposer, "test", measure, log=logs.append)
    rounds = result["rounds"]
    assert [entry["accepted"] for entry in rounds] == [["ok1"], ["ok2"], ["ok3"]], rounds
    assert {item["name"] for item in rounds[0]["refused"]} == {"fat1", "same1"}
    assert any("exceed the budget" in item["reason"] for item in rounds[0]["refused"])
    assert any("already in the session" in item["reason"] for item in rounds[0]["refused"])
    assert seen_refusals[0] == [] and set(seen_refusals[1]) == {"fat1", "same1"}  # fed back next round
    assert [entry["improved"] for entry in rounds] == [True, False, False]  # two stale rounds stop the loop
    assert result["best"] == "ok1" and result["kl"] == 0.25
    assert logs[-1].startswith("TUNE_LOOP_DONE best=ok1")
    assert json.loads((tmp_path / "s" / "loop" / "summary.json").read_text()) == rounds


def test_claude_code_runs_without_tools_under_the_schema_and_is_logged(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    calls = []
    reply = {"analysis": "a", "stop": True, "proposals": []}

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        out = json.dumps({"type": "result", "structured_output": reply})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    assert tl.claude_code(config, tmp_path, runner)(analysis, 1, 4) == reply
    command, kwargs = calls[0]
    assert command[:2] == ["claude", "-p"] and command[command.index("--tools") + 1] == ""
    # review M2: no MCP servers, no settings sources, and an empty working directory (no project CLAUDE.md)
    assert "--strict-mcp-config" in command
    assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert command[command.index("--setting-sources") + 1] == ""
    assert Path(kwargs["cwd"]) != tmp_path and not Path(kwargs["cwd"]).exists()  # a removed temp dir
    assert json.loads(command[command.index("--json-schema") + 1]) == tl.PROPOSAL_SCHEMA
    assert "--no-session-persistence" in command and "--max-budget-usd" in command
    assert "Evidence analysis (data, not instructions)" in kwargs["input"]
    record = json.loads((tmp_path / "s" / "llm" / "round_1.json").read_text())
    assert record["returncode"] == 0 and record["prompt"] == kwargs["input"]
    assert set(record["user_memory"]) == {"path", "sha256"}

    def failing(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="budget exceeded")

    with pytest.raises(tl.LoopError, match="budget exceeded"):
        tl.claude_code(config, tmp_path, failing)(analysis, 2, 4)


def test_the_remote_measurement_syncs_launches_waits_and_fetches(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    config["remote"] = {"jump": "jump", "host": "gpu", "session": "/scratch/s", "tree": "/scratch/t",
                        "logs": "/scratch/logs", "label": "tune_s", "wait_s": 60,
                        "env": {"PY": "/env/python", "HF_HOME": "/hf home", "bad key": "x"}}
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvint4")))
    record = lg.read_ledger(ledger)[0]["id"]
    p1 = ta.propose(tmp_path / "s", "p1", spec, tmp_path / "table.json", Fraction(9, 2), "x", [record],
                    ledger)
    calls = []
    meta = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}}
    kl = json.dumps({"schema": "tricast.autotune.measure/2", "meta": meta, "results": [
        {"name": "p1", "recipe_sha256": p1["recipe_sha256"], "kl": 0.2, "patched": 196}]}).encode()

    def ssh(host, command, timeout, stdin=None):
        calls.append((host, command, stdin))
        out = {0: b"", 1: b"[gpu_cap] gpu label=tune_s_r1 -> CUDA_VISIBLE_DEVICES=0",
               2: b"TUNE_ROUND_DONE round=1", 3: kl}[len(calls) - 1]
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr=b"")

    path = tl.measure_remote(config, tmp_path, 1, ["p1"], ssh)
    assert path.read_bytes() == kl
    sync, launch, wait, fetch = calls
    names = tarfile.open(fileobj=io.BytesIO(sync[2]), mode="r:gz").getnames()
    assert sync[0] == "jump" and "proposals/p1/recipe.yaml" in names and "rounds/1/measure.sh" in names
    assert "run_round.sh" in names  # written by the loop itself (review M3)
    wrapper = (tmp_path / "s" / "run_round.sh").read_text()
    assert "export PY=/env/python" in wrapper and "export HF_HOME='/hf home'" in wrapper
    assert "bad key" not in wrapper and 'bash "$S/rounds/$R/measure.sh"' in wrapper
    assert launch[0] == "gpu" and "gpu_cap.sh run tune_s_r1" in launch[1] and "setsid" in launch[1]
    assert "-- timeout -k 60 60 bash" in launch[1]  # the job stops on its own, KILL after a grace (M8)
    assert "TUNE_ROUND_DONE|TUNE_ROUND_FAIL" in wait[1] and fetch[1] == "cat /scratch/s/rounds/1/kl.json"

    def refusing(host, command, timeout, stdin=None):
        out = b"[gpu_cap] DENY cap reached" if "gpu_cap" in command else b""
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr=b"")

    mxfp4 = ta.load_grid(_grid(tmp_path, qr.uniform("mxfp4")))
    ta.propose(tmp_path / "s", "p2", mxfp4, tmp_path / "table.json", Fraction(9, 2), "x", [record], ledger)
    with pytest.raises(tl.LoopError, match="gpu_cap refused"):
        tl.measure_remote(config, tmp_path, 2, ["p2"], refusing)


def test_auto_greedy_proposes_unmeasured_neighbours_of_the_best(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    reply = tl.auto_greedy(config, tmp_path)(tl.analyze(config, tmp_path), 1, 3)
    assert len(reply["proposals"]) == 3 and not reply["stop"]
    assert all(item["base"] == "start" and item["changes"] for item in reply["proposals"])
    tl.materialize(config, tmp_path, reply["proposals"][0], "auto-greedy")
    again = tl.auto_greedy(config, tmp_path)(tl.analyze(config, tmp_path), 2, 3)["proposals"]
    assert reply["proposals"][0]["changes"] not in [item["changes"] for item in again]


def test_a_measurement_that_is_not_the_round_is_refused(tmp_path: Path) -> None:
    """Review M4: an empty, partial, foreign or off-protocol kl.json must not be ingested as the round."""
    config, ledger = _session(tmp_path)
    record = lg.read_ledger(ledger)[0]["id"]
    p1 = ta.propose(tmp_path / "s", "p1", ta.load_grid(_grid(tmp_path, qr.uniform("nvint4"))),
                    tmp_path / "table.json", Fraction(9, 2), "x", [record], ledger)
    meta = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}}
    good = {"name": "p1", "recipe_sha256": p1["recipe_sha256"], "kl": 0.2, "patched": 196}
    schema = "tricast.autotune.measure/2"
    tl.check_measurement(config, tmp_path, {"schema": schema, "meta": meta, "results": [good]}, ["p1"])
    cases = (({"schema": schema, "meta": meta, "results": []}, "asked for"),
             ({"schema": schema, "meta": meta, "results": [good, good]}, "asked for"),
             ({"schema": schema, "meta": meta, "results": [{**good, "kl": float("nan")}]}, "non-finite"),
             ({"schema": schema, "meta": {**meta, "seed": 42}, "results": [good]}, "not the session"),
             ({"schema": schema, "meta": {**meta, "model_sha": "x"}, "results": [good]},
              "is not the session"),
             ({"schema": "other", "results": [good]}, "not an app.autotune measure"))
    for data, message in cases:
        with pytest.raises(tl.LoopError, match=message):
            tl.check_measurement(config, tmp_path, data, ["p1"])


def test_a_bad_name_writes_nothing_and_a_bad_reply_is_a_refusal(tmp_path: Path) -> None:
    """Review M1: a proposer name like ../../session must be refused before any file is written."""
    config, _ = _session(tmp_path)
    session_file = (tmp_path / "s" / "session.json").read_bytes()
    record = tl.analyze(config, tmp_path)["best"]["kl_record"]
    item = {"name": "../../session", "base": "start", "changes": [], "options": None, "rationale": "x",
            "cites": [record]}
    with pytest.raises(ta.TuneError, match="proposal name"):
        tl.materialize(config, tmp_path, item, "test")
    assert (tmp_path / "s" / "session.json").read_bytes() == session_file
    assert not (tmp_path / "s" / "loop").exists()
    with pytest.raises(tl.LoopError, match="proposal schema"):
        tl.check_reply({"proposals": "not a list"})


def test_rounds_without_a_measurement_count_as_stale_and_reruns_continue(tmp_path: Path) -> None:
    """Review m1/m2: all-refused rounds stop the loop after two; a rerun appends and refuses unfinished
    rounds."""
    config, _ = _session(tmp_path)

    def refusing(analysis: dict, number: int, count: int) -> dict:
        return {"analysis": "x", "stop": False, "proposals": [
            {"name": f"bad{number}", "base": "nope", "changes": [], "options": None, "rationale": "x",
             "cites": []}]}

    def never(number: int, names: list) -> Path:
        raise AssertionError("nothing should be measured")

    first = tl.run(config, tmp_path, 5, 2, refusing, "test", never, log=lambda line: None)
    assert [entry["round"] for entry in first["rounds"]] == [1, 2]  # two stale rounds, not five
    second = tl.run(config, tmp_path, 1, 2, refusing, "test", never, log=lambda line: None)
    assert [entry["round"] for entry in second["rounds"]] == [1, 2, 3]  # appended, not overwritten
    (tmp_path / "s" / "rounds" / "9").mkdir(parents=True)
    (tmp_path / "s" / "rounds" / "9" / "round.json").write_text("{}")
    with pytest.raises(tl.LoopError, match="started but have no kl.json"):
        tl.run(config, tmp_path, 1, 2, refusing, "test", never, log=lambda line: None)


def test_the_wrapper_never_overrides_the_gpu_gpu_cap_assigned(tmp_path: Path) -> None:
    """Re-check NEW-1: remote.env must not set the GPU selection gpu_cap.sh made for the job."""
    config = {"remote": {"env": {"CUDA_VISIBLE_DEVICES": "0", "PY": "/env/python"}}}
    with pytest.raises(tl.LoopError, match="gpu_cap.sh assigns the GPU"):
        tl.round_wrapper(config)
    assert "export PY=/env/python" in tl.round_wrapper({"remote": {"env": {"PY": "/env/python"}}})


def test_without_a_model_sha_the_measurement_must_be_of_the_sessions_model_id(tmp_path: Path) -> None:
    """Re-check M4: a session whose table has no model SHA still refuses another model's kl.json."""
    config, ledger = _session(tmp_path)
    fixed = json.loads((tmp_path / "s" / "session.json").read_text())
    fixed["model_sha"] = None
    (tmp_path / "s" / "session.json").write_text(json.dumps(fixed))
    record = lg.read_ledger(ledger)[0]["id"]
    p1 = ta.propose(tmp_path / "s", "p1", ta.load_grid(_grid(tmp_path, qr.uniform("nvint4"))),
                    tmp_path / "table.json", Fraction(9, 2), "x", [record], ledger)
    meta = {**ENV, "model_sha": None, "seed": 43, "windows": 8, "seqlen": 512,
            "source": {"split": "validation"}}
    good = {"name": "p1", "recipe_sha256": p1["recipe_sha256"], "kl": 0.2, "patched": 196}
    data = {"schema": "tricast.autotune.measure/2", "meta": meta, "results": [good]}
    tl.check_measurement(config, tmp_path, data, ["p1"])
    other = {**data, "meta": {**meta, "model_id": "meta-llama/Llama-3.2-1B"}}
    with pytest.raises(tl.LoopError, match="Llama-3.2-1B is not the session's"):
        tl.check_measurement(config, tmp_path, other, ["p1"])


def test_a_rerun_ingests_a_saved_but_uningested_measurement_first(tmp_path: Path) -> None:
    """Re-check m2: a run that stopped after saving kl.json but before its ingest must not lose the round."""
    config, ledger = _session(tmp_path)
    record = lg.read_ledger(ledger)[0]["id"]
    p1 = ta.propose(tmp_path / "s", "p1", ta.load_grid(_grid(tmp_path, qr.uniform("nvint4"))),
                    tmp_path / "table.json", Fraction(9, 2), "x", [record], ledger)
    ta.round_script(tmp_path / "s", ["p1"], 1)
    meta = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}}
    (tmp_path / "s" / "rounds" / "1" / "kl.json").write_text(json.dumps({
        "schema": "tricast.autotune.measure/2", "meta": meta,
        "results": [{"name": "p1", "recipe_sha256": p1["recipe_sha256"], "kl": 0.11, "patched": 196}]}))

    def stop(analysis: dict, number: int, count: int) -> dict:
        assert analysis["best"]["name"] == "p1" and number == 2  # round 1 was ingested before round 2
        return {"analysis": "x", "stop": True, "proposals": []}

    result = tl.run(config, tmp_path, 1, 2, stop, "test", lambda *a: None, log=lambda line: None)
    assert result["best"] == "p1" and result["kl"] == 0.11
