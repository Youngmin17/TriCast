"""The Auto-tune page's per-layer report errors: a method's ``tricast report`` metrics reach the page's table
only from the report of that method's measured uniform recipe (same recipe hash, same model revision), in the
method order, with where they came from. Ledger and files are fakes; CPU only."""

from __future__ import annotations

import json
from pathlib import Path

from app import autotune as at
from app import export_autotune_demo as ed
from app import ledger as lg
from app import quantizer_recipe as qr
from app.tests.test_autotune_demo import _kl_file
from app.tests.test_tune_agent import ENV, _table

LAYER = qr.layer_name(3, "up_proj")


def _report(path: Path, recipe_hash: str, name: str, model_sha: str, mse: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = {"mse": mse, "sqnr_db": 30.5, "cosine": 0.9991, "relative_frobenius": 0.03}
    path.write_text(json.dumps({
        "model": {"logits_kl": 0.1}, "recipe_hash": recipe_hash, "recipe": {"name": name}, "samples": 8,
        "seqlen": 512, "seed": 42, "env": {**ENV, "model_sha": model_sha},
        "layers": [{"name": LAYER, "weight": errors, "activation": None,
                    "output": {**errors, "mse": 1e-4}}]}))


def test_a_layer_gets_the_report_of_its_methods_measured_uniform_recipe_only(tmp_path: Path) -> None:
    table_path = _table(tmp_path, floor=0.001)
    table = ed.table_bundle(table_path)
    _, _, layers, _ = at.load_table(table_path)

    def uniform(method: str) -> str:
        return ed.recipe_file_sha256({"name": f"u_{method}", "default": method, "grid": qr.uniform(method),
                                      "kv": "none"})
    run = tmp_path / "run"
    run.mkdir()
    _kl_file(run / "kl_seed43_uniform.json", [
        {"name": "u_mxfp8", "recipe_sha256": "r_mx", "file_sha256": uniform("mxfp8"), "kl": 0.05},
        {"name": "u_int8", "recipe_sha256": "r_int8", "file_sha256": uniform("int8"), "kl": 0.01}])
    kl, ppl, _ = ed.measurements(run)
    rows = ed.baselines(table, kl, ppl, layers)
    assert {row["method"] for row in rows} == {"mxfp8", "int8"}
    reports = run / "reports"
    _report(reports / "u_mxfp8" / "report.json", "r_mx", "u_mxfp8", ENV["model_sha"], 2e-6)
    _report(reports / "u_int8" / "report.json", "r_other", "u_int8", ENV["model_sha"], 3e-6)  # other recipe
    _report(reports / "u_int8_old" / "report.json", "r_int8", "u_int8", "another revision", 4e-6)
    ledger = tmp_path / "ledger.jsonl"
    lg.ingest(sorted(reports.rglob("report.json")), ledger, tmp_path)

    sources = ed.layer_reports(table, rows, ledger)
    assert sources["metrics"] == ["weight.mse", "output.sqnr_db", "output.cosine"]
    assert list(sources["methods"]) == ["mxfp8"]
    source = sources["methods"]["mxfp8"]
    assert source["path"] == "run/reports/u_mxfp8/report.json" and source["recipe_hash"] == "r_mx"
    assert source["setting"]["samples"] == 8 and source["gpu"] == "NVIDIA A100 80GB PCIe"
    (layer,) = [row for row in table["layers"] if row["name"] == LAYER]
    expected = [None] * len(table["methods"])
    expected[table["methods"].index("mxfp8")] = [2e-6, 30.5, 0.9991]
    assert layer["report"] == expected
    assert all(row["report"] == [None] * len(table["methods"]) for row in table["layers"] if row is not layer)
