"""A tune round whose measurement stops is resumed, and gpu_cap's refusal is seen: the launch greps the real
DENY line of gpu_cap.sh, and a round's summary entry is written as ``measuring`` before the measurement, so a
rerun measures that round again instead of losing its analysis and refusals. GPU and LLM are fakes; CPU
only."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_loop import _measure, _session

DENY_LINE = "DENY host=geneva reason=cap_reached count=2 cap=2 my_gpus=[3,0]"  # gpu_cap.sh, 2026-11-04


def test_the_launch_sees_the_real_deny_line(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    config["remote"] = {"jump": "jump", "host": "gpu", "session": "/scratch/s", "tree": "/scratch/t",
                        "logs": "/scratch/logs", "label": "tune_s", "wait_s": 60}
    launches = []

    def ssh(host, command, timeout, stdin=None):
        if "gpu_cap.sh run" in command:
            launches.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    with pytest.raises(tl.LoopError):
        tl.measure_remote(config, tmp_path, 1, ["start"], ssh)
    pattern = re.search(r"grep -m1 -E '([^']+)'", launches[0]).group(1)
    log = tmp_path / "round.log"
    log.write_text(DENY_LINE + "\n")
    found = subprocess.run(["grep", "-m1", "-E", pattern, str(log)], capture_output=True, text=True)
    assert found.stdout.strip() == DENY_LINE


def test_a_round_whose_measurement_stopped_is_measured_on_the_next_run(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    calls = []

    def proposer(analysis: dict, number: int, count: int) -> dict:
        calls.append(number)
        cite = [analysis["best"]["kl_record"]]
        return {"analysis": f"round {number}", "stop": False, "proposals": [
            {"name": f"p{number}", "base": analysis["best"]["name"], "options": None, "rationale": "x",
             "cites": cite, "changes": [{"block": number, "kind": "q_proj", "method": "mxfp4"}]}]}

    def stopped(number: int, names: list[str]) -> Path:
        raise tl.LoopError("gpu_cap refused gpu: DENY")

    with pytest.raises(tl.LoopError, match="DENY"):
        tl.run(config, tmp_path, 1, 1, proposer, "test", stopped, log=lambda _: None)
    summary = tl.json.loads((tmp_path / "s" / "loop" / "summary.json").read_text())
    assert summary == [{"round": 1, "accepted": ["p1"], "refused": [], "stop": False, "analysis": "round 1",
                        "improved": False, "status": "measuring", "before_kl": 0.3}]

    def measure(number: int, names: list[str]) -> Path:
        shas = {name: ta.read_proposal(tmp_path / "s", name)["recipe_sha256"] for name in names}
        return _measure(tmp_path, ledger, {name: (shas[name], 0.2) for name in names}, f"r{number}")

    logs = []
    result = tl.run(config, tmp_path, 0, 1, proposer, "test", measure, log=logs.append)
    (entry,) = result["rounds"]
    assert entry["best"] == "p1" and entry["best_kl"] == 0.2 and entry["improved"] is True
    assert "status" not in entry and "before_kl" not in entry and calls == [1]  # no second proposer call
    assert any("measuring again" in line for line in logs)
