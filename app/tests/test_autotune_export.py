"""The tuning app's step 5 (app/autotune_export.py, ``POST /api/autotune/export``): the model size
under a setup is computed from the original checkpoint's tensors and the setup's Linear bits, and the
recipe bundle is a zip holding the very recipe file measured, its measurement, the size and a README.
CPU only, no torch, no LLM."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import autotune_export as ex
from app import quantizer_recipe as qr
from app.server import DEFAULT_DEMO_DIR, create_app

BUNDLE = json.loads((DEFAULT_DEMO_DIR / "autotune" / "bundle.json").read_text(encoding="utf-8"))
MODEL = BUNDLE["table"]["meta"]["model_id"]
LOOP = Path(__file__).resolve().parents[2] / "evals" / "fixtures" / "tune_inputs" / "s3150" / "loop.json"


def test_the_checkpoint_constants_match_the_table_and_the_file_size() -> None:
    model = ex.MODEL_FILES[MODEL]
    assert model["revision"] == BUNDLE["table"]["meta"]["revision"]
    assert model["linear_params"] == sum(layer["params"] for layer in BUNDLE["table"]["layers"])
    params = model["linear_params"] + model["embedding_params"] + model["norm_params"]
    assert 2 * params + model["header_bytes"] == model["file_bytes"]


def test_the_constants_match_the_cached_safetensors_header_when_it_is_here() -> None:
    import struct

    cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    path = cache / ex.MODEL_FILES[MODEL]["revision"] / "model.safetensors"
    if not path.is_file():
        pytest.skip("Qwen3-0.6B is not in the local Hugging Face cache")
    with path.open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    count = {"linear": 0, "embedding": 0, "norm": 0}
    for name, tensor in header.items():
        numel = 1
        for dim in tensor["shape"]:
            numel *= dim
        kind = ("linear" if name.split(".")[-2].endswith("_proj")
                else "embedding" if "embed_tokens" in name or "lm_head" in name else "norm")
        count[kind] += numel
        assert tensor["dtype"] == "BF16"
    model = ex.MODEL_FILES[MODEL]
    assert count == {"linear": model["linear_params"], "embedding": model["embedding_params"],
                     "norm": model["norm_params"]}
    assert path.stat().st_size == model["file_bytes"]


def test_size_shrinks_only_the_linear_weights() -> None:
    size = ex.model_size(MODEL, 3.147953869047619)
    linear, embedding, norm = size["setup"]
    assert linear["part"] == "linear" and linear["bytes"] == round(440_401_920 * 3.147953869047619 / 8)
    assert embedding["bits"] == norm["bits"] == 16 and embedding["bytes"] == size["original"][1]["bytes"]
    assert size["setup_bytes"] == linear["bytes"] + embedding["bytes"] + norm["bytes"] + 35_560
    assert 0.52 < size["ratio"] < 0.54 and abs(size["linear_ratio"] - 3.147953869047619 / 16) < 1e-12
    assert size["computed"] is True and ex.model_size("another/model", 4.0) is None


def test_linear_bits_is_the_measured_rows_or_computed_from_the_grid() -> None:
    from app import autotune_recipe as ar

    row = next(r for r in BUNDLE["solutions"] if r["id"] == "iso_3.15")
    recipe = ar.grid_recipe(BUNDLE, row["grid"], row["budget_bits"])
    assert ex.linear_bits(BUNDLE, recipe, row["grid"]) == row["bits"]
    grid = qr.uniform("gptq_int4", BUNDLE["table"]["blocks"])
    uniform = next(r for r in BUNDLE["baselines"] if r["method"] == "gptq_int4")
    assert ex.linear_bits(BUNDLE, ar.grid_recipe(BUNDLE, grid), grid) == uniform["bits"]
    edited = [list(line) for line in row["grid"]]
    edited[0][0] = "bf16"
    bits = ex.linear_bits(BUNDLE, ar.grid_recipe(BUNDLE, edited, "403/128"), edited)
    assert bits > row["bits"]  # a layer kept in BF16 costs more bits, computed from the table


def test_the_recipe_route_says_the_size() -> None:
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    row = next(r for r in BUNDLE["solutions"] if r["id"] == "iso_3.15")
    body = {"grid": row["grid"], "budget_bits": row["budget_bits"]}
    out = client.post("/api/autotune/recipe", json=body).json()
    assert out["size"]["setup"][0]["bits"] == row["bits"] and out["size"]["original_bytes"] == 1_503_300_328


def test_the_export_is_a_zip_with_the_measured_file(tmp_path: Path) -> None:
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR, agent=LOOP))
    rounds = BUNDLE["session"]["rounds"]
    proposal = next(p for r in rounds for p in r["proposals"] if p["name"] == "act_damp05_r3")
    answer = client.post("/api/autotune/export", json={"proposal": "act_damp05_r3"})
    assert answer.status_code == 200, answer.text
    assert answer.headers["content-type"] == "application/zip"
    assert 'filename="act_damp05_r3_tricast.zip"' in answer.headers["content-disposition"]
    zf = zipfile.ZipFile(io.BytesIO(answer.content))
    names = set(zf.namelist())
    assert {"act_damp05_r3.yaml", "evaluation.json", "size.json", "README.md"} <= names
    recipe = client.post("/api/autotune/recipe", json={"proposal": "act_damp05_r3"}).json()
    text = zf.read("act_damp05_r3.yaml").decode()
    assert text == recipe["yaml"] and hashlib.sha256(text.encode()).hexdigest() == recipe["sha256"]
    evaluation = json.loads(zf.read("evaluation.json"))
    assert evaluation["same_file_as_measured"] and evaluation["measured_ppl"]["ppl"] == proposal["ppl"]
    # the PPL's own record (a proposal's KL and PPL were measured on different trees), not the KL one's
    record = BUNDLE["session"]["records"][proposal["ppl_id"]]
    assert evaluation["measured_ppl"]["git_sha"] == record["git_sha"] and record["kind"] == "ppl"
    assert evaluation["this_server_run"] is None
    assert evaluation["native_ppl"] == BUNDLE["native"]["ppl"]
    assert json.loads(zf.read("size.json"))["setup"][0]["bits"] == proposal["avg_bits"]
    readme = zf.read("README.md").decode()
    assert "tricast ppl --model Qwen/Qwen3-0.6B --recipe act_damp05_r3.yaml" in readme
    assert "the very file measured" in readme and "does not write that low-bit file" in readme
    assert f"git {record['git_sha'][:7]}" in readme
    assert names == {"act_damp05_r3.yaml", "evaluation.json", "size.json", "README.md"}


def test_without_agent_the_export_of_a_session_proposal_says_why() -> None:
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    answer = client.post("/api/autotune/export", json={"proposal": "act_damp05_r3"})
    assert answer.status_code == 503 and "--agent" in answer.json()["error"]["message"]


def test_a_budget_that_is_not_short_digits_is_refused_at_once() -> None:
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR))
    grid = [list(line) for line in BUNDLE["solutions"][0]["grid"]]
    grid[0][0] = "bf16" if grid[0][0] != "bf16" else BUNDLE["table"]["methods"][1]  # not a recorded grid
    recorded = BUNDLE["solutions"][0]["grid"]  # named by its record, but its budget is checked too
    recipe, export = "/api/autotune/recipe", "/api/autotune/export"
    cases = ((recipe, grid), (export, grid), (recipe, recorded))
    for budget in ("1e1000000000", "9" * 40, "3/0"):
        for route, body in cases:
            answer = client.post(route, json={"grid": body, "budget_bits": budget})
            code = answer.json()["error"]["code"]
            assert answer.status_code == 422 and code == "invalid_budget", (route, budget)


def test_a_job_goes_into_the_bundle_only_on_a_gpu_server() -> None:
    client = TestClient(create_app("demo", "none", DEFAULT_DEMO_DIR, agent=LOOP))
    answer = client.post("/api/autotune/export", json={"proposal": "act_damp05_r3", "job": "j1"})
    assert answer.status_code == 409 and answer.json()["error"]["code"] == "live_mode_required"


def test_the_bundle_carries_a_job_of_this_recipe_text() -> None:
    from app import autotune_recipe as ar

    row = next(r for r in BUNDLE["solutions"] if r["id"] == "iso_3.15")
    recipe = ar.grid_recipe(BUNDLE, row["grid"], row["budget_bits"])
    live = {"job": "j1", "tasks": ["ppl"], "results": {"run": {"ppl": 27.1}}, "env": {"gpu": "Test GPU"}}
    zf = zipfile.ZipFile(io.BytesIO(ex.bundle_zip(BUNDLE, recipe, row["bits"], live)))
    assert json.loads(zf.read("evaluation.json"))["this_server_run"] == live
    readme = zf.read("README.md").decode()
    assert "this server's GPU (Test GPU): WikiText-2 test PPL 27.10, tasks ppl, job `j1`" in readme
    lm_only = {**live, "tasks": ["hellaswag"], "results": {"run": None, "lmeval": {"tasks": []}}}
    data = ex.bundle_zip(BUNDLE, recipe, row["bits"], lm_only)
    readme = zipfile.ZipFile(io.BytesIO(data)).read("README.md").decode()
    assert "Evaluated on this server's GPU (Test GPU): tasks hellaswag, job `j1`" in readme
    assert f"git {row['ppl_record']['git_sha'][:7]}" in readme
