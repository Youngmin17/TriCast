"""A uniform grid (one method on every layer) on ``POST /api/autotune/recipe`` (app/autotune_recipe.py): it is
the uniform recipe the GPU measured — the tuning app's "one method on every layer" edit hands over that very
file (SHA-256 equal to the bundle's record) with its measured values. A method with no measured uniform recipe
falls back to the ordinary grid rule and has no measurement. CPU only, no LLM."""

from __future__ import annotations

import hashlib
import json

from app import autotune_recipe as ar
from app import quantizer_recipe as qr
from app.server import DEFAULT_DEMO_DIR

BUNDLE = json.loads((DEFAULT_DEMO_DIR / "autotune" / "bundle.json").read_text(encoding="utf-8"))
BLOCKS = BUNDLE["table"]["blocks"]


def test_every_measured_uniform_recipe_is_the_file_its_measurement_recorded() -> None:
    assert BUNDLE["baselines"]
    for row in BUNDLE["baselines"]:
        out = ar.grid_recipe(BUNDLE, qr.uniform(row["method"], BLOCKS), "403/128")
        assert out["kind"] == "uniform" and out["name"] == row["id"] == f"u_{row['method']}"
        assert out["file"] == f"{row['id']}.yaml" and out["default"] == row["method"]
        assert out["sha256"] == hashlib.sha256(out["yaml"].encode()).hexdigest() == row["recipe_file_sha256"]
        assert out["measured"]["same_file"] and out["measured"]["id"] == row["id"]
        assert out["measured"]["ppl"] == row["ppl"] and out["measured"]["kl"] == row["kl"]
        assert "solve" not in out["command"]  # the knapsack did not make this grid


def test_a_uniform_grid_without_a_measured_recipe_has_no_measurement() -> None:
    assert all(row["method"] != "bf16" for row in BUNDLE["baselines"])
    out = ar.grid_recipe(BUNDLE, qr.uniform("bf16", BLOCKS), "16")
    assert out["kind"] == "knapsack" and out["measured"] is None and not out["name"].startswith("u_")


def test_a_recorded_solution_is_still_answered_as_the_solution() -> None:
    row = BUNDLE["solutions"][0]
    out = ar.grid_recipe(BUNDLE, row["grid"], row["budget_bits"])
    assert out["kind"] == "knapsack" and out["name"] == row["id"] and out["measured"]["same_file"]
