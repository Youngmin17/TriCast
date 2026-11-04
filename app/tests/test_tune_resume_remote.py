"""A stopped round is measured again through the real ``measure_remote``: the round folder its first attempt
wrote is taken back (same proposals, no kl.json) and written again byte for byte, instead of being refused
as an existing round; a folder started for other proposals is refused. GPU and ssh are fakes; CPU only."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_loop import _session


def _config(tmp_path: Path) -> dict:
    config, _ = _session(tmp_path)
    config["remote"] = {"jump": "jump", "host": "gpu", "session": "/scratch/s", "tree": "/scratch/t",
                        "logs": "/scratch/logs", "label": "tune_s", "wait_s": 60}
    return config


def _sync_fails(host, command, timeout, stdin=None):
    return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"fake: no cluster")


def test_a_stopped_round_folder_is_written_again_for_the_same_proposals(tmp_path: Path) -> None:
    config = _config(tmp_path)
    folder = tmp_path / "s" / "rounds" / "1"
    ta.round_script(tmp_path / "s", ["start"], 1)  # the first attempt, whose measurement stopped
    before = {part: (folder / part).read_bytes() for part in ("measure.sh", "round.json")}
    with pytest.raises(tl.LoopError, match="sync failed"):  # past the round folder, at the (fake) cluster
        tl.measure_remote(config, tmp_path, 1, ["start"], _sync_fails)
    assert {part: (folder / part).read_bytes() for part in before} == before
    assert sorted(path.name for path in folder.iterdir()) == ["measure.sh", "round.json"]


def test_a_round_folder_of_other_proposals_or_with_kl_is_not_taken_back(tmp_path: Path) -> None:
    config = _config(tmp_path)
    ta.round_script(tmp_path / "s", ["start"], 1)
    with pytest.raises(tl.LoopError, match=r"was started for \['start'\]"):
        tl.measure_remote(config, tmp_path, 1, ["start", "start"], _sync_fails)
    (tmp_path / "s" / "rounds" / "1" / "kl.json").write_text("{}")
    with pytest.raises(ta.TuneError, match="round 1 exists"):
        tl.measure_remote(config, tmp_path, 1, ["start"], _sync_fails)


def test_a_remote_pythonpath_comes_after_the_tree() -> None:
    plain = tl.round_wrapper({"remote": {"env": {"PY": "/env/python"}}})
    assert 'export PYTHONPATH="$T/src:$T"\n' in plain  # unchanged without one
    wrapper = tl.round_wrapper({"remote": {"env": {"PY": "/env/python", "PYTHONPATH": "/deps/qm"}}})
    assert 'export PYTHONPATH="$T/src:$T":/deps/qm\n' in wrapper
    assert wrapper.count("PYTHONPATH=") == 1  # not exported twice
    found = subprocess.run(["bash", "-c", wrapper.split("echo ")[0] + 'printf %s "$PYTHONPATH"'],
                           capture_output=True, text=True, env={"T": "/t", "PATH": "/usr/bin:/bin"})
    assert found.stdout == "/t/src:/t:/deps/qm"
