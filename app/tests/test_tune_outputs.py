"""The tuning app's other models: PPL recorded per recipe file (scripts/quantizer/tune_outputs.py), the bundle
exporter taking that PPL for a grid without a KL measurement (app/export_autotune_demo.py ``--outputs``), and
the recipe route picking a language model's bundle from the model list (``autotune/models.json``). CPU only,
no torch, no LLM, no GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import autotune as at
from app import autotune_recipe as ar
from app import export_autotune_demo as ed
from app import quantizer_recipe as qr
from app.server import DEFAULT_DEMO_DIR, create_app
from app.tests.test_tune_agent import _table

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tune_outputs",
                                              ROOT / "scripts" / "quantizer" / "tune_outputs.py")
assert SPEC is not None and SPEC.loader is not None
to = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(to)
BUNDLE = json.loads((DEFAULT_DEMO_DIR / "autotune" / "bundle.json").read_text(encoding="utf-8"))


def _record(path: Path, name: str, file_sha: str | None, ppl: float, ids: list[int], *, smoke: bool = False,
            prompts: tuple[str, ...] | None = None, model: str = "Qwen/Qwen3-8B",
            revision: str = "b968826", loaded: str | None = None) -> None:
    answers = [{"prompt": p, "ids": ids, "text": f"{name} {i}"} for i, p in enumerate(prompts or to.PROMPTS)]
    path.write_text(json.dumps({
        "model": model, "revision": revision, "smoke": smoke, "name": name,
        "file": None if file_sha is None else f"{name}.yaml", "file_sha256": file_sha, "recipe_hash": "h",
        "patched_layers": 0 if file_sha is None else 252,
        "ppl": {"ppl": ppl, "n_tokens": 1000, "n_windows": 2, "dataset_fingerprint": "f"},
        "answers": answers, "env": {"gpu_names": ["NVIDIA H200 NVL"], "git_sha": "abc", "git_dirty": False,
                                    "model_sha": loaded or revision}}))


def test_the_uniform_files_are_the_ones_the_page_hands_over() -> None:
    """``uniform`` builds through the page's own recipe route with the tables' calibration settings, so its
    files are the bundle's measured ones."""
    from app import autotune_recipe as ar

    out = {}
    for method in ("gptq_int4", "gptq_int3", "int8_w"):
        bundle = {"table": {"blocks": qr.BLOCKS, "kinds": list(qr.KINDS), "methods": ["bf16", method],
                            "path": "t", "meta": {"calibration": to.CALIBRATION, "mma": "fp32_fma"}},
                  "solutions": [], "baselines": [{"id": f"u_{method}", "method": method}]}
        out[method] = ar.grid_recipe(bundle, qr.uniform(method))["sha256"]
    recorded = {row["method"]: row["recipe_file_sha256"] for row in BUNDLE["baselines"]}
    assert out == {method: recorded[method] for method in out}


def test_uniform_writes_those_files(tmp_path: Path) -> None:
    to.uniform(qr.BLOCKS, ["gptq_int3"], tmp_path, "fp32_fma")
    recorded = next(row for row in BUNDLE["baselines"] if row["method"] == "gptq_int3")["recipe_file_sha256"]
    assert hashlib.sha256((tmp_path / "u_gptq_int3.yaml").read_bytes()).hexdigest() == recorded


def test_a_grid_without_kl_takes_the_ppl_recorded_with_its_answers(tmp_path: Path) -> None:
    table = _table(tmp_path, floor=0.001)
    run = tmp_path / "run"
    run.mkdir()
    solved = run / "auto_4.50"
    assert at.main(["solve", "--table", str(table), "--budget-bits", "9/2", "--name", "auto_4.50",
                    "--out", str(solved)]) == 0
    raw = json.loads(table.read_text())
    method = next(m for m in raw["meta"]["methods"] if m != "bf16")
    uniform = ed.recipe_file_sha256({"name": f"u_{method}", "default": method, "grid": qr.uniform(method),
                                     "kv": "none", "calibration": raw["meta"].get("calibration"),
                                     "mma": raw["meta"].get("mma")})
    records = tmp_path / "records"
    records.mkdir()
    same = {"model": raw["meta"]["model_id"], "revision": raw["meta"]["model_sha"]}
    _record(records / "native.json", "native", None, 10.0, [1], **same)
    solved_sha = hashlib.sha256((solved / "recipe.yaml").read_bytes()).hexdigest()
    _record(records / "auto_4.50.json", "auto_4.50", solved_sha, 12.5, [1], **same)
    _record(records / f"u_{method}.json", f"u_{method}", uniform, 14.0, [2], **same)
    out = tmp_path / "out"
    assert ed.main(["--table", str(table), "--measured", str(run), "--outputs", str(records),
                    "--out", str(out)]) == 0
    bundle = json.loads((out / "bundle.json").read_text())
    solution = next(row for row in bundle["solutions"] if row["id"] == "auto_4.50")
    assert solution["ppl"] == 12.5 and solution["kl"] is None and solution["rejected"] == []
    assert solution["ppl_record"]["source"] == "records/auto_4.50.json"
    assert solution["ppl_record"]["gpu"] == "NVIDIA H200 NVL"
    baseline = next(row for row in bundle["baselines"] if row["method"] == method)
    assert baseline["ppl"] == 14.0 and baseline["kl"] is None
    assert bundle["native"]["ppl"] == 10.0
    assert all(row["ppl"] is not None for row in bundle["baselines"])  # unrecorded uniforms are left out
    # the recipe route says where that PPL was measured: no KL record, so the PPL record's GPU and tree
    measured = ar.grid_recipe(bundle, solution["grid"])["measured"]
    assert measured["same_file"] is True and measured["ppl"] == 12.5
    assert measured["gpu"] == "NVIDIA H200 NVL" and measured["git_sha"] == solution["ppl_record"]["git_sha"]


@pytest.mark.parametrize("bad", ["model", "smoke", "moved_ref"])
def test_records_of_another_model_or_a_smoke_run_are_refused(tmp_path: Path, bad: str) -> None:
    """A uniform recipe's file names no model, so the same SHA-256 must not carry another model's PPL."""
    table = _table(tmp_path, floor=0.001)
    raw = json.loads(table.read_text())
    method = next(m for m in raw["meta"]["methods"] if m != "bf16")
    uniform = ed.recipe_file_sha256({"name": f"u_{method}", "default": method, "grid": qr.uniform(method),
                                     "kv": "none", "calibration": raw["meta"].get("calibration"),
                                     "mma": raw["meta"].get("mma")})
    run, records = tmp_path / "run", tmp_path / "records"
    run.mkdir()
    records.mkdir()
    _record(records / f"u_{method}.json", f"u_{method}", uniform, 14.0, [2], smoke=bad == "smoke",
            model="Qwen/Qwen3-8B" if bad == "model" else raw["meta"]["model_id"],
            revision="b968826" if bad == "model" else raw["meta"]["model_sha"],
            loaded="c0ffee0" if bad == "moved_ref" else None)
    assert ed.main(["--table", str(table), "--measured", str(run), "--outputs", str(records),
                    "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out" / "bundle.json").exists()


def test_outputs_need_the_measured_run(tmp_path: Path) -> None:
    table = _table(tmp_path, floor=0.001)
    assert ed.main(["--table", str(table), "--outputs", str(tmp_path), "--out", str(tmp_path / "o")]) == 2


def _demo_with_models(tmp_path: Path, models: list[dict] | None) -> Path:
    demo = tmp_path / "demo"
    (demo / "autotune" / "other").mkdir(parents=True)
    shutil.copy(DEFAULT_DEMO_DIR / "catalog.json", demo / "catalog.json")
    shutil.copy(DEFAULT_DEMO_DIR / "autotune" / "bundle.json", demo / "autotune" / "bundle.json")
    other = {**BUNDLE, "session": None, "transfer": []}  # a model without an AI session
    (demo / "autotune" / "other" / "bundle.json").write_text(json.dumps(other))
    if models is not None:
        (demo / "autotune" / "models.json").write_text(json.dumps({"models": models}))
    return demo


def test_the_recipe_route_takes_the_bundle_the_model_list_names(tmp_path: Path) -> None:
    demo = _demo_with_models(tmp_path, [
        {"id": "Qwen/Qwen3-0.6B", "kind": "llm", "bundle": "bundle.json"},
        {"id": "other/model", "kind": "llm", "bundle": "other/bundle.json"},
        {"id": "escape", "kind": "llm", "bundle": "../catalog.json"},
        {"id": "yolo11n", "kind": "detector", "data": "yolo11n.json"}])
    client = TestClient(create_app("demo", "none", demo))
    grid = qr.uniform("gptq_int4")
    default = client.post("/api/autotune/recipe", json={"grid": grid}).json()
    named = client.post("/api/autotune/recipe", json={"grid": grid, "model": "other/model"}).json()
    assert named["sha256"] == default["sha256"] and named["measured"]["same_file"] is True
    for model in ("nope", "yolo11n", "escape"):  # not listed, not a language model, outside the folder
        response = client.post("/api/autotune/recipe", json={"grid": grid, "model": model})
        assert response.status_code == 404 and response.json()["error"]["code"] == "unknown_model"


def test_a_session_recipe_is_only_for_the_model_whose_bundle_has_the_session(tmp_path: Path) -> None:
    demo = _demo_with_models(tmp_path, [{"id": "Qwen/Qwen3-0.6B", "kind": "llm", "bundle": "bundle.json"},
                                        {"id": "other/model", "kind": "llm", "bundle": "other/bundle.json"}])
    loop = ROOT / "evals" / "fixtures" / "tune_inputs" / "s3150" / "loop.json"
    client = TestClient(create_app("demo", "none", demo, agent=loop))
    own = client.post("/api/autotune/recipe", json={"proposal": "act_damp05_r3", "model": "Qwen/Qwen3-0.6B"})
    assert own.status_code == 200 and own.json()["measured"]["same_file"] is True
    other = client.post("/api/autotune/recipe", json={"proposal": "act_damp05_r3", "model": "other/model"})
    assert other.status_code == 404 and other.json()["error"]["code"] == "not_found"


def test_a_folder_without_a_model_list_serves_its_one_bundle_for_its_model(tmp_path: Path) -> None:
    client = TestClient(create_app("demo", "none", _demo_with_models(tmp_path, None)))
    grid = qr.uniform("gptq_int4")
    model = BUNDLE["table"]["meta"]["model_id"]
    own = client.post("/api/autotune/recipe", json={"grid": grid, "model": model})
    assert own.status_code == 200 and own.json()["measured"]["same_file"] is True
    other = client.post("/api/autotune/recipe", json={"grid": grid, "model": "other/model"})
    assert other.status_code == 404 and other.json()["error"]["code"] == "unknown_model"
