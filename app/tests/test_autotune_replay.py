"""Offline demo answers (app.server --record DIR / --replay DIR, app/autotune_live.py): a live parse / propose
answer is kept as it was sent, replay serves it again marked as a recording with its time and never calls
claude, a request that was not recorded is refused with the recorded ones listed, and the flags are checked.
Live calls are fakes; CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import autotune_live as live
from app import server
from app.server import create_app

PARSED = {"mode": "live", "text": "Qwen3-0.6B 를 403/128 비트 안에서 최적화해줘",
          "request": {"outcome": "tune", "model": "Qwen/Qwen3-0.6B", "budget_bits": "403/128"},
          "llm": {"cost_usd": 0.03, "duration_s": 6.0, "models": ["claude-test"],
                  "started_utc": "2026-11-09T02:00:00+00:00"}}
ROUND = {"mode": "live", "round": 5, "analysis": "a", "stop": False, "llm": {"cost_usd": 0.3},
         "proposals": [{"name": "p1", "status": "accepted"}], "note": "검사만 했고 측정하지 않았습니다."}


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: Path) -> tuple[TestClient, list[str]]:
    demo = tmp_path / "demo"
    demo.mkdir(exist_ok=True)
    (demo / "catalog.json").write_text("{}")
    loop = tmp_path / "loop.json"
    loop.write_text(json.dumps({"session": "s", "table": "t.json", "budget_bits": "5"}))
    calls: list[str] = []

    def parse(config: dict, root: Path, text: str, **kwargs: object) -> dict:
        calls.append(f"parse {text}")
        return {**PARSED, "text": text}

    def propose(config: dict, root: Path, count: int, **kwargs: object) -> dict:
        calls.append(f"propose {count}")
        return ROUND
    monkeypatch.setattr(live, "parse_request", parse)
    monkeypatch.setattr(live, "propose_round", propose)
    return TestClient(create_app("demo", "none", demo, agent=loop, **flags)), calls


def test_a_recorded_answer_is_replayed_as_a_recording_without_calling_claude(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kept = tmp_path / "replay"
    recording, calls = _app(tmp_path, monkeypatch, record=kept)
    sent = recording.post("/api/autotune/parse", json={"text": PARSED["text"]}).json()
    assert sent["mode"] == "live"
    assert recording.post("/api/autotune/propose", json={"count": 4}).json() == ROUND
    assert calls == [f"parse {PARSED['text']}", "propose 4"] and len(list(kept.glob("*.json"))) == 2
    replaying, calls = _app(tmp_path, monkeypatch, replay=kept)
    spaced = "  Qwen3-0.6B 를  403/128 비트 안에서 최적화해줘 "
    again = replaying.post("/api/autotune/parse", json={"text": spaced})
    assert again.status_code == 200 and calls == []  # white space does not matter; claude is never called
    body = again.json()
    assert body["mode"] == "replay" and body["recorded_utc"] and body["request"] == PARSED["request"]
    assert {key: value for key, value in body.items() if key not in ("mode", "recorded_utc")} == \
        {key: value for key, value in sent.items() if key != "mode"}
    round_ = replaying.post("/api/autotune/propose", json={"count": 4}).json()
    assert round_["mode"] == "replay" and round_["proposals"] == ROUND["proposals"] and calls == []


def test_a_request_that_was_not_recorded_is_refused_with_the_recorded_ones_listed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kept = tmp_path / "replay"
    live.record_answer(kept, "parse", PARSED["text"], PARSED)
    client, calls = _app(tmp_path, monkeypatch, replay=kept)
    missing = client.post("/api/autotune/parse", json={"text": "Qwen3-0.6B 3비트"})
    assert missing.status_code == 404 and calls == []
    error = missing.json()["error"]
    assert error["code"] == "not_recorded" and PARSED["text"] in error["message"]
    assert client.post("/api/autotune/propose", json={"count": 4}).json()["error"]["code"] == "not_recorded"


def test_a_recording_of_another_request_is_not_served(tmp_path: Path) -> None:
    kept = tmp_path / "replay"
    path = live.record_answer(kept, "parse", "a", PARSED)
    entry = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**entry, "key": "b"}), encoding="utf-8")
    with pytest.raises(live.LiveError) as refused:
        live.replay_answer(kept, "parse", "a")
    assert refused.value.code == "replay_corrupt"


@pytest.mark.parametrize("argv, message", [
    (["--demo", "--replay", "x"], "go with --agent"),
    (["--demo", "--agent", "LOOP", "--record", "x", "--replay", "x"], "exclude each other"),
    (["--demo", "--agent", "LOOP", "--replay", "EMPTY"], "no recorded answers"),
])
def test_the_flags_are_checked(argv: list[str], message: str, tmp_path: Path,
                               capsys: pytest.CaptureFixture[str]) -> None:
    loop = tmp_path / "loop.json"
    loop.write_text("{}")
    (tmp_path / "EMPTY").mkdir()
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "catalog.json").write_text("{}")
    argv = [str(loop) if a == "LOOP" else str(tmp_path / "EMPTY") if a == "EMPTY" else a for a in argv]
    with pytest.raises(SystemExit):
        server.main([*argv, "--demo-dir", str(demo)])
    assert message in capsys.readouterr().err


def test_the_mode_says_what_the_buttons_do(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/autotune/mode: the presenter guide picks its disclosure by it and offers the recorded ones."""
    kept = tmp_path / "replay"
    live.record_answer(kept, "parse", PARSED["text"], PARSED)
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "catalog.json").write_text("{}")
    assert TestClient(create_app("demo", "none", demo)).get("/api/autotune/mode").json() == \
        {"agent": "off", "sentences": []}
    replaying, _ = _app(tmp_path, monkeypatch, replay=kept)
    assert replaying.get("/api/autotune/mode").json() == {"agent": "replay", "sentences": [PARSED["text"]]}
    calling, _ = _app(tmp_path, monkeypatch)
    assert calling.get("/api/autotune/mode").json()["agent"] == "live"
