"""POST /api/autotune/recipe {"transfer": id}: an option transfer's recipe (a recorded knapsack grid with a
session proposal's options) is rebuilt byte for byte as the transfer script wrote it — its SHA-256 is the
one the GPU measurement recorded — and served only with --agent (demo bundle, packed session)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import autotune_recipe as ar
from app.server import DEFAULT_DEMO_DIR, create_app

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "app" / "web" / "demo" / "autotune" / "bundle.json"
LOOP = ROOT / "evals" / "fixtures" / "tune_inputs" / "s3150" / "loop.json"
SESSION = ROOT / "evals" / "fixtures" / "tune_inputs" / "s3150"
TRANSFERS = [row["id"] for row in json.loads(BUNDLE.read_text(encoding="utf-8"))["transfer"]]


@pytest.mark.parametrize("ident", TRANSFERS)
def test_every_transfer_recipe_is_the_measured_file(ident: str) -> None:
    bundle = ar.load_bundle(BUNDLE)
    answer = ar.transfer_recipe(bundle, SESSION, ident)
    row = next(item for item in bundle["transfer"] if item["id"] == ident)
    assert answer["sha256"] == row["recipe_file_sha256"] and answer["measured"]["same_file"] is True
    assert answer["measured"]["ppl"] == row["ppl"] and answer["options_from"] == row["options_from"]
    assert "damp: 0.05" in answer["yaml"] and "act_order: true" in answer["yaml"]


def test_the_route_needs_agent_and_refuses_unknown_ids() -> None:
    body = {"transfer": "xfer_auto_3.00"}
    off = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR)).post("/api/autotune/recipe", json=body)
    assert off.status_code == 503 and off.json()["error"]["code"] == "agent_unavailable"
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR, agent=LOOP))
    ok = client.post("/api/autotune/recipe", json={"transfer": "xfer_auto_3.00"})
    assert ok.status_code == 200 and ok.json()["measured"]["same_file"] is True
    assert client.post("/api/autotune/recipe", json={"transfer": "xfer_nothing"}).status_code == 404
    assert client.post("/api/autotune/recipe", json={"transfer": "../etc"}).status_code == 400
