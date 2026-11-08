"""The recipe file the Auto-tune page hands over (``POST /api/autotune/recipe``, app/autotune_recipe.py): for
a recorded knapsack grid it is the very file the GPU measured (SHA-256 equal to the bundle's record), for any
other grid the file ``app.autotune solve`` would write, and for a session proposal the session's own measured
file, compared with its round's record. CPU only, no LLM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import autotune_recipe as ar
from app.server import DEFAULT_DEMO_DIR, create_app

BUNDLE = json.loads((DEFAULT_DEMO_DIR / "autotune" / "bundle.json").read_text(encoding="utf-8"))


def test_every_recorded_solution_grid_gives_the_file_its_measurement_recorded() -> None:
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    assert BUNDLE["solutions"]
    for row in BUNDLE["solutions"]:
        body = {"grid": row["grid"], "budget_bits": row["budget_bits"]}
        answer = client.post("/api/autotune/recipe", json=body)
        assert answer.status_code == 200, answer.text
        out = answer.json()
        assert out["name"] == row["id"] and out["yaml"].startswith("---\n")
        assert out["sha256"] == hashlib.sha256(out["yaml"].encode()).hexdigest() == row["recipe_file_sha256"]
        assert out["measured"]["same_file"] and out["measured"]["kl"] == row["kl"]
        assert out["measured"]["ppl"] == row["ppl"]


def test_another_grid_is_named_by_its_budget_and_has_no_measurement() -> None:
    grid = [list(row) for row in BUNDLE["solutions"][0]["grid"]]
    grid[0][0] = "bf16" if grid[0][0] != "bf16" else BUNDLE["table"]["methods"][1]
    out = ar.grid_recipe(BUNDLE, grid, "403/128")
    assert out["name"] == "auto_403-128" and out["measured"] is None
    assert "--budget-bits 403/128 --name auto_403-128" in out["command"]


def test_bad_requests_are_refused_with_the_reason() -> None:
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    post = lambda body: client.post("/api/autotune/recipe", json=body)  # noqa: E731
    assert post({"grid": [["nvfp4"]]}).json()["error"]["code"] == "invalid_grid"
    grid = [list(row) for row in BUNDLE["solutions"][0]["grid"]]
    grid[1][2] = "no_such_method"
    unknown = post({"grid": grid})
    assert unknown.status_code == 422 and "no_such_method" in unknown.json()["error"]["message"]
    assert post({}).status_code == 400
    off = post({"proposal": "act_damp05_r3"})
    assert off.status_code == 503 and off.json()["error"]["code"] == "agent_unavailable"


def _session(root: Path, recorded: str) -> tuple[Path, dict]:
    session = root / "s"
    (session / "proposals" / "p1").mkdir(parents=True)
    (session / "rounds" / "1").mkdir(parents=True)
    text = "name: p1\n"
    (session / "proposals" / "p1" / "recipe.yaml").write_text(text, encoding="utf-8")
    (session / "rounds" / "1" / "kl.json").write_text(json.dumps({"results": [
        {"name": "p1", "file_sha256": recorded or hashlib.sha256(text.encode()).hexdigest(), "kl": 0.3}]}))
    proposal = {"name": "p1", "kl": 0.3, "kl_id": "E1", "ppl": None}
    bundle = {**BUNDLE, "session": {"rounds": [{"proposals": [proposal]}],
                                    "records": {"E1": {"gpu": "G", "git_sha": "abc"}}}}
    return session, bundle


def test_a_proposal_recipe_is_the_session_file_compared_with_its_round_record(tmp_path: Path) -> None:
    session, bundle = _session(tmp_path, "")
    out = ar.proposal_recipe(bundle, session, "p1")
    assert out["yaml"] == "name: p1\n" and out["measured"]["same_file"] and out["measured"]["kl"] == 0.3
    assert out["measured"]["gpu"] == "G"
    other, bundle = _session(tmp_path / "x", "0" * 64)
    assert ar.proposal_recipe(bundle, other, "p1")["measured"]["same_file"] is False
    for name, status in (("../p1", 400), ("p2", 404)):
        try:
            ar.proposal_recipe(bundle, session, name)
        except ar.RecipeError as exc:
            assert exc.status == status
        else:
            raise AssertionError(name)


def test_the_route_serves_a_proposal_file_with_agent(tmp_path: Path) -> None:
    session, bundle = _session(tmp_path, "")
    demo = tmp_path / "demo"
    (demo / "autotune").mkdir(parents=True)
    (demo / "catalog.json").write_text("{}")
    (demo / "autotune" / "bundle.json").write_text(json.dumps(bundle))
    loop = tmp_path / "loop.json"
    loop.write_text(json.dumps({"session": str(session), "table": "t.json", "budget_bits": "3"}))
    answer = TestClient(create_app("demo", "none", demo, agent=loop)).post("/api/autotune/recipe",
                                                                           json={"proposal": "p1"})
    assert answer.status_code == 200, answer.text
    assert answer.json()["kind"] == "proposal" and answer.json()["measured"]["same_file"]


def test_without_tricast_the_route_says_how_to_start_the_server(monkeypatch) -> None:
    def build(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'tricast'")
    monkeypatch.setattr(ar.qr, "build", build)
    grid = BUNDLE["solutions"][0]["grid"]
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    answer = client.post("/api/autotune/recipe", json={"grid": grid})
    assert answer.status_code == 503 and answer.json()["error"]["code"] == "tricast_missing"
    assert "PYTHONPATH=src" in answer.json()["error"]["message"]
