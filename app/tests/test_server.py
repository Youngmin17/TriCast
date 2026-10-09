"""HTTP contract tests for app/server.py and app/jobs.py (app/README.md); no torch, no tricast."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.runners
from app import server
from app.jobs import JobQueue
from app.server import RunArgs, create_app

REPO = Path(__file__).resolve().parents[2]
RUN_KEYS = {"id", "status", "stage", "progress", "request", "server", "mma_label", "preset", "recipe", "env",
            "timing", "baseline", "emulated", "metrics", "evidence", "cached_baseline"}
SUMMARY_KEYS = {"id", "status", "task", "model", "mma", "mma_label", "format", "baseline", "input",
                "created_utc", "summary", "preview"}

HOPPER = {"algorithm": "cofda", "f_bits": 13, "chunk_size": 32, "c_mode": "fused"}
F7 = {**HOPPER, "f_bits": 7}
CATALOG = {
    "tricast": {"version": "0.1.0.dev0", "git_sha": None},
    "mode": "live",
    "device": {"kind": "none", "name": None, "backend": None},
    "tasks": [{"id": task, "kind": task.split(".")[0], "label": task, "description": ""}
              for task in ("llm.generate", "vision.detect", "vision.classify")],
    "models": [
        {"id": "org/tiny-llm", "label": "Tiny LLM", "tasks": ["llm.generate"], "revision": "abc123",
         "support": "operator", "note": ""},
        {"id": "tiny-vision", "label": "Tiny vision", "tasks": ["vision.detect", "vision.classify"],
         "revision": None, "support": "operator", "note": ""},
    ],
    "algorithms": [{"id": algorithm, "label": algorithm, "description": "", "formats": [None, "fp8_e4m3"],
                    "params": []} for algorithm in ("cofda", "fp64")],
    "presets": [{"id": "hopper", "label": "Hopper", "source": "test", "mma": HOPPER, "format": "fp8_e4m3",
                 "status": "modeled", "provenance": "test", "status_note": ""}],
    "formats": [{"id": None, "label": "no quantization", "bits": None, "note": ""},
                {"id": "fp8_e4m3", "label": "FP8 E4M3", "bits": 8, "note": ""}],
    "baselines": [{"id": "native", "label": "native", "description": ""},
                  {"id": "same_quant_fp64", "label": "same quant + FP64", "description": ""}],
}
LLM = {"task": "llm.generate", "model": "org/tiny-llm", "mma": {"algorithm": "cofda", "f_bits": 7},
       "format": "fp8_e4m3", "input": {"prompt": "hello", "max_new_tokens": 8}}
DETECT = {"task": "vision.detect", "model": "tiny-vision", "mma": {"algorithm": "cofda"},
          "format": "fp8_e4m3", "input": {"image": "street.jpg"}}
COMMON = {"timing": {"baseline_s": 0.5, "emulated_s": 1.5}, "env": {"git_sha": "abc"},
          "cached_baseline": False,
          "evidence": {"patched_linear": 7, "patched_conv2d": 0, "emulated_calls": 21,
                       "backend": "reference"}}
IMAGE = {"media": "street.jpg", "width": 4, "height": 3}
BOX = {"cls": "car", "cls_id": 2, "conf": 0.875, "xyxy": [0.0, 0.0, 2.0, 2.0]}
CAT = {"label": "tabby", "class_id": 281, "p": 0.5}
TIGER = {"label": "tiger cat", "class_id": 282, "p": 0.25}
RESULTS = {
    "llm.generate": {"baseline": {"label": "native", "text": "a", "tokens": []},
                     "emulated": {"label": "emulated", "text": "b", "tokens": []},
                     "metrics": {"first_divergence": 0, "prefix_match": 0,
                                 "teacher_forced": {"positions": 2, "top1_agreement": 0.5, "kl_mean": 0.25,
                                                    "kl_max": 0.5, "kl": [0.0, 0.5], "top1": [True, False]}},
                     **COMMON},
    "vision.detect": {"baseline": {"label": "native", "image": IMAGE, "boxes": [BOX, BOX, BOX]},
                      "emulated": {"label": "emulated", "image": IMAGE, "boxes": [BOX]},
                      "metrics": {"matched": 1, "baseline_only": 2, "emulated_only": 0, "mean_iou": 0.75,
                                  "mean_abs_conf_delta": 0.125,
                                  "pairs": [{"baseline": 0, "emulated": 0, "iou": 0.75}]},
                      **COMMON},
    "vision.classify": {"baseline": {"label": "native", "image": IMAGE, "top": [CAT, TIGER]},
                        "emulated": {"label": "emulated", "image": IMAGE, "top": [TIGER, CAT]},
                        "metrics": {"top1_same": True, "top5_overlap": 4, "kl": 0.0625,
                                    "baseline_top1_p": 0.5, "emulated_top1_p": 0.25},
                        **COMMON},
}
SUMMARIES = {
    "llm.generate": {"first_divergence": 0, "prefix_match": 0, "top1_agreement": 0.5, "kl_mean": 0.25},
    "vision.detect": {"matched": 1, "baseline_only": 2, "emulated_only": 0, "mean_iou": 0.75},
    "vision.classify": {"top1_same": True, "top5_overlap": 4, "kl": 0.0625},
}
PREVIEWS = {
    "llm.generate": {"baseline": "a", "emulated": "b"},
    "vision.detect": {"baseline_boxes": 3, "emulated_boxes": 1},
    "vision.classify": {"baseline_top1": "tabby", "emulated_top1": "tiger cat", "emulated_top1_p": 0.25},
}


def png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (200, 30, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    """What the server reads of a recorded bundle (catalog.json, media/), plus files that must never be
    served: one beside the bundle's media and one beside the live runs directory."""
    root = tmp_path / "demo"
    (root / "media").mkdir(parents=True)
    write_json(root / "catalog.json", {**CATALOG, "mode": "demo"})
    (root / "media" / "street.jpg").write_bytes(png_bytes())
    write_json(root / "secret.json", {"secret": "do-not-serve"})
    write_json(tmp_path / "secret.json", {"secret": "do-not-serve"})
    return root


@pytest.fixture
def demo(demo_dir: Path) -> TestClient:
    return TestClient(create_app("demo", "none", demo_dir))


class FakeRunner:
    """Stands in for run_live: reports progress and the baseline output early, can be held after
    that, and fails on the prompt "fail"."""

    def __init__(self) -> None:
        self.calls: list[RunArgs] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(self, args: RunArgs, progress) -> dict:
        self.calls.append(args)
        progress("load", 0.1)
        progress("baseline", 0.4, partial={"baseline": RESULTS[args.task]["baseline"]})
        self.started.set()
        self.release.wait(5)
        if args.prompt == "fail":
            raise RuntimeError("boom")
        progress("metrics", 0.9)
        return RESULTS[args.task]


def label(mma: dict) -> str:
    return "label " + json.dumps(mma, sort_keys=True)


def fake_catalog() -> types.ModuleType:
    module = types.ModuleType("app.catalog")

    def canonical_mma(mma: dict) -> dict:
        if mma.get("algorithm") == "fp64":
            return {"algorithm": "fp64"}
        if mma.get("algorithm") != "cofda":
            raise ValueError("알 수 없는 누산 알고리즘입니다.")
        return {**HOPPER, **{key: mma[key] for key in ("f_bits", "chunk_size", "c_mode") if key in mma}}

    def compose_recipe(mma: dict, format_id: str | None, task: str) -> dict:
        if mma["algorithm"] == "fp64" and task != "llm.generate":
            raise ValueError("이 알고리즘은 비전 작업에 쓸 수 없습니다.")
        return {"name": f"studio:{mma['algorithm']}:{format_id}",
                "defaults": {"weight": format_id, "activation": format_id, "mma": mma},
                "include": ["*"], "exclude": ["lm_head"], "backend": "auto"}

    def preset_for(mma: dict, format_id: str | None) -> str | None:
        return next((preset["id"] for preset in CATALOG["presets"]
                     if preset["mma"] == mma and preset["format"] == format_id), None)

    def recipe_record(recipe: dict) -> dict:
        bundled = "hopper_fp8_w8a8" if recipe["defaults"]["mma"] == HOPPER else None
        return {"name": recipe["name"], "bundled": bundled, "yaml": json.dumps(recipe)}

    module.build_catalog = lambda mode, device_info: {**CATALOG, "mode": mode, "device": device_info}
    module.canonical_mma = canonical_mma
    module.compose_recipe = compose_recipe
    module.baseline_recipe = lambda baseline, format_id: None if baseline == "native" else {"name": "fp64"}
    module.mma_label = label
    module.preset_for = preset_for
    module.recipe_record = recipe_record
    return module


@pytest.fixture
def live(demo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeRunner]:
    monkeypatch.setitem(sys.modules, "app.catalog", fake_catalog())
    runner = FakeRunner()
    app = create_app("live", "cpu", demo_dir, runner, runs_dir=tmp_path / "runs")
    return TestClient(app), runner


def wait_finished(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 5
    while True:
        record = client.get(f"/api/runs/{run_id}").json()
        if record["status"] in ("done", "error") or time.monotonic() > deadline:
            return record
        time.sleep(0.01)


def assert_error(response, status: int, code: str) -> str:
    assert response.status_code == status, response.text
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == code
    return response.json()["error"]["message"]


# --- demo mode -------------------------------------------------------------------------------------------


def test_demo_serves_health_catalog_resources_and_the_static_gui(demo: TestClient) -> None:
    assert demo.get("/api/health").json() == {"status": "ok", "mode": "demo", "device": "none"}
    assert demo.get("/api/catalog").json() == {**CATALOG, "mode": "demo"}
    assert demo.get("/api/resources").json() == {"mode": "demo", "server": None}
    root = demo.get("/")
    assert root.status_code == 200 and root.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/api/runs"), ("POST", "/api/runs"), ("GET", "/api/runs/llm-f7"), ("DELETE", "/api/runs/llm-f7"),
    ("GET", "/api/media/street.jpg"), ("GET", "/api/unknown"),
])
def test_demo_mode_answers_other_api_paths_with_not_found(demo: TestClient, method: str, path: str) -> None:
    assert_error(demo.request(method, path), 404, "not_found")


def test_demo_mode_never_imports_compute_modules(demo_dir: Path) -> None:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from fastapi.testclient import TestClient\n"
        "from app.server import create_app\n"
        f"client = TestClient(create_app('demo', 'none', Path({str(demo_dir)!r})))\n"
        "assert client.get('/api/catalog').json()['mode'] == 'demo'\n"
        "assert client.get('/api/resources').json() == {'mode': 'demo', 'server': None}\n"
        "roots = {'torch', 'tricast', 'PIL', 'yaml'}\n"
        "allowed = {'app.server', 'app.jobs', 'app.listing'}\n"
        "loaded = [name for name in sys.modules if name.split('.')[0] in roots\n"
        "          or (name.startswith('app.') and name not in allowed)]\n"
        "print(sorted(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True,
                            check=True)
    assert result.stdout.strip() == "[]"


# --- live mode -------------------------------------------------------------------------------------------


def test_live_health_and_catalog(live: tuple[TestClient, FakeRunner]) -> None:
    client, _ = live
    assert client.get("/api/health").json() == {"status": "ok", "mode": "live", "device": "cpu"}
    catalog = client.get("/api/catalog").json()
    assert catalog["mode"] == "live"
    assert catalog["device"] == {"kind": "cpu", "name": None, "backend": "reference"}


@pytest.mark.parametrize("change", [
    {"task": "llm.chat"},
    {"model": "missing"},
    {"model": "tiny-vision"},  # exists, but has no llm.generate
    {"format": "fp3"},
    {"baseline": "fp16"},
    {"mma": "cofda"},
    {"input": {"prompt": "x" * 2001}},
    {"input": {"prompt": "   "}},
    {"input": {"prompt": "hi", "max_new_tokens": 65}},
    {"input": {"prompt": "hi", "max_new_tokens": 0}},
    {"input": {"prompt": "hi", "max_new_tokens": "many"}},
])
def test_live_rejects_invalid_requests(live: tuple[TestClient, FakeRunner], change: dict) -> None:
    client, runner = live
    message = assert_error(client.post("/api/runs", json={**LLM, **change}), 400, "invalid_request")
    assert "—" not in message  # no em dash in user-facing Korean text
    assert runner.calls == []


def test_live_reports_catalog_value_errors(live: tuple[TestClient, FakeRunner]) -> None:
    client, runner = live
    without_format = {key: value for key, value in LLM.items() if key != "format"}
    assert_error(client.post("/api/runs", json=without_format), 400, "invalid_request")
    warp = {**LLM, "mma": {"algorithm": "warp"}}
    message = assert_error(client.post("/api/runs", json=warp), 400, "invalid_mma")
    assert message == "알 수 없는 누산 알고리즘입니다."
    message = assert_error(client.post("/api/runs", json={**DETECT, "mma": {"algorithm": "fp64"}}), 400,
                           "unsupported_combination")
    assert message == "이 알고리즘은 비전 작업에 쓸 수 없습니다."
    assert runner.calls == []


@pytest.mark.parametrize(("image", "status", "code"), [
    (None, 400, "invalid_request"),
    ("missing.jpg", 400, "invalid_request"),
    ("../secret.json", 400, "invalid_id"),
    ("data:image/gif;base64,R0lGODlhAQABAAAAACw=", 400, "invalid_request"),
    ("data:image/png;base64,@@@@", 400, "invalid_request"),
    (data_url(b"not an image at all"), 400, "invalid_request"),
    (data_url(b"\0" * (8 * 2**20 + 1)), 413, "image_too_large"),
])
def test_live_rejects_bad_images(live: tuple[TestClient, FakeRunner], image: str | None, status: int,
                                 code: str) -> None:
    client, runner = live
    assert_error(client.post("/api/runs", json={**DETECT, "input": {"image": image}}), status, code)
    assert runner.calls == []


@pytest.mark.parametrize("path", ["/api/runs/..%2Fsecret", "/api/runs/%2E%2E", "/api/runs/.hidden",
                                  "/api/media/..%2Fsecret.json", "/api/media/%2E%2E%2Fsecret.json"])
def test_live_rejects_path_traversal(live: tuple[TestClient, FakeRunner], path: str) -> None:
    client, _ = live
    response = client.get(path)
    assert response.status_code in (400, 404)
    assert "do-not-serve" not in response.text
    assert set(response.json()) == {"error"}


def test_live_rejects_names_outside_the_safe_alphabet(live: tuple[TestClient, FakeRunner]) -> None:
    client, _ = live
    assert_error(client.get("/api/runs/bad%24id"), 400, "invalid_id")
    assert_error(client.get("/api/media/a%20b.jpg"), 400, "invalid_id")


def test_live_upload_is_stored_by_content_and_served_by_name(live: tuple[TestClient, FakeRunner]) -> None:
    client, runner = live
    upload = {**DETECT, "input": {"image": data_url(png_bytes())}}
    response = client.post("/api/runs", json=upload)
    assert response.status_code == 202
    record = wait_finished(client, response.json()["id"])
    name = record["request"]["input"]["image"]
    assert record["status"] == "done" and name == f"upload-{hashlib.sha256(png_bytes()).hexdigest()}.png"
    assert runner.calls[0].image_path.name == name
    assert client.get(f"/api/media/{name}").content == png_bytes()
    again = client.post("/api/runs", json=upload)  # the same picture uploaded again is the same run
    assert (again.status_code, again.json()) == (200, {"id": record["id"], "cached": True})
    assert len(runner.calls) == 1


@pytest.mark.parametrize("task", ["vision.detect", "vision.classify"])
def test_live_vision_run_uses_demo_media_and_reports_preset(live: tuple[TestClient, FakeRunner],
                                                            demo_dir: Path, task: str) -> None:
    client, runner = live
    response = client.post("/api/runs", json={**DETECT, "task": task})
    assert response.status_code == 202
    record = wait_finished(client, response.json()["id"])
    assert record["request"]["input"] == {"image": "street.jpg"} and record["request"]["mma"] == HOPPER
    assert (record["preset"], record["mma_label"]) == ("hopper", label(HOPPER))
    assert record["recipe"]["bundled"] == "hopper_fp8_w8a8"
    assert {key: record[key] for key in RESULTS[task]} == RESULTS[task]
    assert runner.calls[0].image_path == demo_dir / "media" / "street.jpg" and runner.calls[0].prompt is None
    assert runner.calls[0].checkpoint is None  # no --yolo/--resnet-checkpoint flag
    assert client.get("/api/media/street.jpg").content == png_bytes()
    listed = client.get("/api/runs").json()["runs"][0]
    assert (listed["summary"], listed["preview"]) == (SUMMARIES[task], PREVIEWS[task])


def test_live_jobs_run_one_at_a_time(live: tuple[TestClient, FakeRunner], tmp_path: Path) -> None:
    client, runner = live
    runner.release.clear()
    first = client.post("/api/runs", json=LLM)
    assert first.status_code == 202 and set(first.json()) == {"id"}
    first_id = first.json()["id"]
    assert len(first_id) == 32 and int(first_id, 16) >= 0
    assert runner.started.wait(5)
    second_id = client.post("/api/runs", json={**LLM, "input": {"prompt": "second"}}).json()["id"]

    running = client.get(f"/api/runs/{first_id}").json()
    assert (running["status"], running["stage"], running["progress"]) == ("running", "baseline", 0.4)
    assert running["baseline"] == RESULTS["llm.generate"]["baseline"]  # shown before the run ends
    assert running["emulated"] is None and running["metrics"] is None
    queued = client.get(f"/api/runs/{second_id}").json()
    assert set(queued) == RUN_KEYS
    assert (queued["status"], queued["stage"], queued["progress"]) == ("queued", None, 0)
    assert queued["baseline"] is None and queued["env"] is None
    assert client.get("/api/runs").json()["runs"][0]["preview"] is None

    runner.release.set()
    done = wait_finished(client, first_id)
    assert set(done) == RUN_KEYS
    assert (done["status"], done["stage"], done["progress"]) == ("done", "metrics", 1.0)
    assert {key: done[key] for key in RESULTS["llm.generate"]} == RESULTS["llm.generate"]
    assert done["request"] == {**LLM, "mma": F7, "baseline": "native"}
    assert (done["mma_label"], done["preset"]) == (label(F7), None)
    assert done["recipe"]["name"] == "studio:cofda:fp8_e4m3" and done["recipe"]["bundled"] is None
    assert json.loads(done["recipe"]["yaml"])["defaults"]["weight"] == "fp8_e4m3"
    saved = json.loads((tmp_path / "runs" / f"{first_id}.json").read_text(encoding="utf-8"))
    assert (saved["id"], saved["status"]) == (first_id, "done")  # saved before it is shown as done
    assert wait_finished(client, second_id)["status"] == "done"
    assert [call.prompt for call in runner.calls] == ["hello", "second"]
    assert runner.calls[0].max_new_tokens == 8 and runner.calls[0].baseline_recipe is None

    runs = client.get("/api/runs").json()["runs"]
    assert [run["id"] for run in runs] == [second_id, first_id]
    assert set(runs[0]) == SUMMARY_KEYS
    assert runs[0]["input"] == {"prompt": "second", "max_new_tokens": 32} and runs[0]["mma"] == F7  # default
    assert (runs[0]["summary"], runs[0]["preview"]) == (SUMMARIES["llm.generate"], PREVIEWS["llm.generate"])


def test_live_identical_requests_share_the_unfinished_job(live: tuple[TestClient, FakeRunner]) -> None:
    client, runner = live
    runner.release.clear()
    first = client.post("/api/runs", json=LLM).json()["id"]
    assert runner.started.wait(5)
    running = client.post("/api/runs", json={**LLM, "baseline": "native"})  # the same canonical request
    assert (running.status_code, running.json()) == (202, {"id": first})
    second_request = {**LLM, "input": {"prompt": "second", "max_new_tokens": 8}}
    second = client.post("/api/runs", json=second_request).json()["id"]
    queued = client.post("/api/runs", json=second_request)
    assert (queued.status_code, queued.json()) == (202, {"id": second})
    assert [run["id"] for run in client.get("/api/runs").json()["runs"]] == [second, first]
    runner.release.set()
    assert wait_finished(client, second)["status"] == "done"
    assert [call.prompt for call in runner.calls] == ["hello", "second"]  # each request ran once


def test_live_cache_returns_the_finished_run_of_the_same_request(live: tuple[TestClient, FakeRunner]) -> None:
    client, runner = live
    first_id = client.post("/api/runs", json=LLM).json()["id"]
    assert wait_finished(client, first_id)["status"] == "done"
    same = {**LLM, "mma": {**F7, "f2_bits": 3}, "baseline": "native"}  # same canonical request
    response = client.post("/api/runs", json=same)
    assert (response.status_code, response.json()) == (200, {"id": first_id, "cached": True})
    assert len(runner.calls) == 1
    longer = client.post("/api/runs", json={**LLM, "input": {"prompt": "hello", "max_new_tokens": 9}})
    assert longer.status_code == 202 and longer.json()["id"] != first_id
    failed = client.post("/api/runs", json={**LLM, "input": {"prompt": "fail"}}).json()["id"]
    assert wait_finished(client, failed)["status"] == "error"
    retried = client.post("/api/runs", json={**LLM, "input": {"prompt": "fail"}})
    assert retried.status_code == 202 and retried.json()["id"] != failed  # errors are not cached


def test_live_restart_keeps_the_run_list_and_the_cache(demo_dir: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "app.catalog", fake_catalog())
    before = TestClient(create_app("live", "cpu", demo_dir, FakeRunner(), runs_dir=tmp_path / "runs"))
    run_id = before.post("/api/runs", json=LLM).json()["id"]
    done = wait_finished(before, run_id)
    runner = FakeRunner()
    after = TestClient(create_app("live", "cpu", demo_dir, runner, runs_dir=tmp_path / "runs"))
    listed = after.get("/api/runs").json()["runs"]
    assert [run["id"] for run in listed] == [run_id]
    assert listed[0]["summary"] == SUMMARIES["llm.generate"]
    assert listed[0]["preview"] == PREVIEWS["llm.generate"]
    assert after.get(f"/api/runs/{run_id}").json() == done
    assert after.post("/api/runs", json=LLM).json() == {"id": run_id, "cached": True}
    assert runner.calls == []


def test_live_cache_is_per_device_and_code(demo_dir: Path, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished run is reused only by a server with the same identity; another device or code runs again."""
    monkeypatch.setitem(sys.modules, "app.catalog", fake_catalog())
    code_a = {"device": "cuda", "app_sha256": "a" * 64, "git_sha": "1" * 40, "src_sha256": None}

    def client(runner: FakeRunner, identity: dict) -> TestClient:
        app = create_app("live", "cpu", demo_dir, runner, runs_dir=tmp_path / "runs", identity=identity)
        return TestClient(app)

    first = client(FakeRunner(), code_a)
    run_id = first.post("/api/runs", json=LLM).json()["id"]
    assert wait_finished(first, run_id)["server"] == code_a
    assert client(FakeRunner(), code_a).post("/api/runs", json=LLM).json() == {"id": run_id, "cached": True}
    for changed in ({**code_a, "device": "cpu"}, {**code_a, "app_sha256": "b" * 64}):
        runner = FakeRunner()
        other = client(runner, changed)
        response = other.post("/api/runs", json=LLM)
        assert response.status_code == 202 and response.json()["id"] != run_id
        wait_finished(other, response.json()["id"])
        assert len(runner.calls) == 1


def test_live_rejects_a_non_string_algorithm(live: tuple[TestClient, FakeRunner]) -> None:
    client, runner = live
    response = client.post("/api/runs", json={**LLM, "mma": {"algorithm": ["cofda"]}})
    assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"
    assert runner.calls == []


def test_live_runner_failure_becomes_an_error_record(live: tuple[TestClient, FakeRunner], tmp_path: Path,
                                                     caplog: pytest.LogCaptureFixture) -> None:
    client, _ = live
    run_id = client.post("/api/runs", json={**LLM, "baseline": "same_quant_fp64",
                                            "input": {"prompt": "fail"}}).json()["id"]
    record = wait_finished(client, run_id)
    assert set(record) == RUN_KEYS | {"error"}
    assert record["status"] == "error" and record["stage"] == "baseline"
    assert record["error"] == {"code": "run_failed", "message": "RuntimeError: boom"}
    assert record["baseline"] == RESULTS["llm.generate"]["baseline"] and record["emulated"] is None
    assert "Traceback" not in json.dumps(record)
    assert "Traceback" in caplog.text  # the stack trace stays in the server log
    saved = json.loads((tmp_path / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "error"
    later = client.post("/api/runs", json=LLM).json()["id"]
    assert wait_finished(client, later)["status"] == "done"  # the worker survived the failure
    assert_error(client.get("/api/runs/0123456789abcdef0123456789abcdef"), 404, "not_found")


# --- job queue -------------------------------------------------------------------------------------------

FIELDS = {"request": {"task": "llm.generate"}, "mma_label": "CoFDA", "preset": None, "recipe": {}}


def fields(n: int) -> dict:
    return {**FIELDS, "request": {"task": "llm.generate", "n": n}}


def wait_job(jobs: JobQueue, job_id: str) -> dict:
    deadline = time.monotonic() + 5
    while jobs.get(job_id)["status"] not in ("done", "error") and time.monotonic() < deadline:
        time.sleep(0.01)
    return jobs.get(job_id)


def test_job_queue_keeps_recent_jobs_and_serves_older_ones_from_disk(tmp_path: Path) -> None:
    jobs = JobQueue(tmp_path, keep=2)
    ids = [jobs.submit(fields(n), lambda progress: {"metrics": {}})[0] for n in range(4)]
    assert wait_job(jobs, ids[-1])["status"] == "done"
    assert [summary["id"] for summary in jobs.summaries()] == [ids[3], ids[2]]
    oldest = jobs.get(ids[0])
    assert (oldest["status"], oldest["request"]) == ("done", {"task": "llm.generate", "n": 0})


def test_job_queue_loads_the_newest_saved_jobs_at_startup(tmp_path: Path,
                                                          caplog: pytest.LogCaptureFixture) -> None:
    before = JobQueue(tmp_path)
    ids = []
    for n in range(3):
        job_id, _ = before.submit(fields(n), lambda progress, n=n: {"metrics": {}, "summary": {"n": n}})
        assert wait_job(before, job_id)["status"] == "done"
        ids.append(job_id)
        os.utime(tmp_path / f"{job_id}.json", (1_000_000_000 + n, 1_000_000_000 + n))
    notes = tmp_path / "notes.json"  # the newest file, but not a saved job
    notes.write_text("[]", encoding="utf-8")
    os.utime(notes, (1_000_000_010, 1_000_000_010))
    after = JobQueue(tmp_path, keep=3)  # reads the newest three files: notes.json, ids[2], ids[1]
    assert [summary["id"] for summary in after.summaries()] == [ids[2], ids[1]]
    assert after.summaries()[0]["summary"] == {"n": 2}
    assert "not a saved run" in caplog.text
    assert after.get(ids[0])["request"] == {"task": "llm.generate", "n": 0}  # older: read from disk
    assert after.submit(fields(1), lambda progress: {"metrics": {}}) == (ids[1], "done")
    assert after.counts() == {"queued": 0, "running": 0}


def test_job_files_are_replaced_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_replace = os.replace
    moves = []

    def replace(source, target) -> None:
        moves.append((Path(source).name, Path(target).name))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    jobs = JobQueue(tmp_path)
    job_id, status = jobs.submit(FIELDS, lambda progress: {"metrics": {}})
    assert status == "queued" and wait_job(jobs, job_id)["status"] == "done"
    assert moves == [(f"{job_id}.json.tmp", f"{job_id}.json")]
    assert [path.name for path in tmp_path.iterdir()] == [f"{job_id}.json"]


def test_job_whose_result_is_not_json_becomes_an_error(tmp_path: Path) -> None:
    jobs = JobQueue(tmp_path)
    job_id, _ = jobs.submit(FIELDS, lambda progress: {"metrics": {"kl": float("nan")}})
    record = wait_job(jobs, job_id)
    assert record["status"] == "error" and record["metrics"] is None
    assert record["error"]["code"] == "run_failed" and "JSON compliant" in record["error"]["message"]


def test_job_with_a_non_json_partial_baseline_becomes_an_error(tmp_path: Path) -> None:
    def work(progress) -> dict:
        progress("baseline", 0.5, partial={"baseline": {"logprob": float("nan")}})
        return {"metrics": {}}

    jobs = JobQueue(tmp_path)
    record = wait_job(jobs, jobs.submit(FIELDS, work)[0])
    assert record["status"] == "error" and record["baseline"] is None
    assert "JSON compliant" in record["error"]["message"]


# --- resources and the live runner -------------------------------------------------------------------------

GPU = {"index": 0, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mb": 81920, "memory_used_mb": 1024,
       "utilization_pct": 3}
VERSIONS = {"torch": "2.8.0", "triton": "3.4.0", "cuda": "12.8"}
SMI = ("0, NVIDIA A100-SXM4-80GB, 81920, 1024, 3\n"
       "1, NVIDIA A100-SXM4-80GB, 81920, [N/A], [Not Supported]\n")


@pytest.fixture
def probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake host probes: one GPU, fixed versions, torchvision but no ultralytics, no vision checkpoint
    at the runner's default path, and a Hugging Face cache holding the pinned org/tiny-llm snapshot."""
    calls = {"gpus": 0}

    def gpus() -> list[dict]:
        calls["gpus"] += 1
        return [GPU]

    hub = types.ModuleType("huggingface_hub")
    hub.constants = types.SimpleNamespace(HF_HUB_CACHE=str(tmp_path / "hub"))
    (tmp_path / "hub" / "models--org--tiny-llm" / "snapshots" / "abc123").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "app.catalog", fake_catalog())
    monkeypatch.setattr(server, "_gpus", gpus)
    monkeypatch.setattr(server, "_versions", lambda: VERSIONS)
    monkeypatch.setattr(server, "_importable", lambda module: module == "torchvision")
    monkeypatch.setattr(server, "_checkpoint_file",
                        lambda model_id, checkpoint: checkpoint or tmp_path / "absent.pt")
    return calls


def test_live_resources_report_host_queue_and_runnable_tasks(demo_dir: Path, tmp_path: Path, probes: dict,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "vision.pt"
    checkpoint.write_bytes(b"weights")
    runner = FakeRunner()
    client = TestClient(create_app("live", "cpu", demo_dir, runner, runs_dir=tmp_path / "runs",
                                   checkpoints={"tiny-vision": checkpoint}))
    assert client.get("/api/resources").json() == {"mode": "live", "server": {
        "hostname": socket.gethostname(), "device": "cpu", "gpus": [GPU], **VERSIONS,
        "queue": {"queued": 0, "running": 0}, "models_cached": ["org/tiny-llm", "tiny-vision"],
        "runnable_tasks": ["vision.classify"]}}  # llm.generate needs CUDA, vision.detect ultralytics

    runner.release.clear()
    first = client.post("/api/runs", json=LLM).json()["id"]
    assert runner.started.wait(5)
    second = client.post("/api/runs", json=DETECT).json()["id"]
    assert client.get("/api/resources").json()["server"]["queue"] == {"queued": 1, "running": 1}
    assert probes["gpus"] == 1  # the probes are reused for RESOURCE_TTL_S; the queue is counted live
    monkeypatch.setattr(server, "RESOURCE_TTL_S", 0.0)
    client.get("/api/resources")
    assert probes["gpus"] == 2
    runner.release.set()
    assert wait_finished(client, first)["status"] == wait_finished(client, second)["status"] == "done"
    assert runner.calls[1].checkpoint == checkpoint  # the flag's path reaches the vision runner


def test_live_resources_need_cuda_and_the_pinned_snapshot(demo_dir: Path, tmp_path: Path, probes: dict,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = {"kind": "cuda", "name": "A100", "backend": "triton"}
    monkeypatch.setattr(server, "_device_info", lambda device: cuda)
    monkeypatch.setattr(server, "RESOURCE_TTL_S", 0.0)
    client = TestClient(create_app("live", "cuda", demo_dir, FakeRunner(), runs_dir=tmp_path / "runs"))
    found = client.get("/api/resources").json()["server"]
    assert (found["device"], found["models_cached"], found["runnable_tasks"]) == (
        "cuda", ["org/tiny-llm"], ["llm.generate"])  # no vision checkpoint at the runner's default path
    snapshots = tmp_path / "hub" / "models--org--tiny-llm" / "snapshots"
    (snapshots / "abc123").rename(snapshots / "another-revision")
    found = client.get("/api/resources").json()["server"]
    assert (found["models_cached"], found["runnable_tasks"]) == ([], [])


def test_gpu_probe_reads_nvidia_smi_and_honours_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=SMI, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    second = {"index": 1, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mb": 81920, "memory_used_mb": None,
              "utilization_pct": None}
    assert server._gpus() == [GPU, second]
    assert calls[0][0][0] == "nvidia-smi" and calls[0][1]["timeout"] == 5
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert server._gpus() == [second]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert server._gpus() == []
    for error in (FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 5),
                  subprocess.CalledProcessError(9, "nvidia-smi")):
        def failing(args, error=error, **kwargs):
            raise error

        monkeypatch.setattr(subprocess, "run", failing)
        assert server._gpus() == []


def test_versions_report_missing_packages_as_null(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(package: str) -> str:
        if package != "triton":
            raise server.metadata.PackageNotFoundError(package)
        return "3.4.0"

    monkeypatch.setattr(server.metadata, "version", version)
    assert server._versions() == {"torch": None, "triton": "3.4.0", "cuda": None}


def test_run_live_passes_the_request_to_the_runners(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = []

    def fake(name: str):
        def run(*args, **kwargs) -> dict:
            seen.append((name, args, kwargs))
            return {"metrics": {}}

        return run

    def run_env(task: str, model_id: str, recipe: dict, baseline_recipe: dict | None,
                extra: dict | None = None, *, device: str | None = None,
                checkpoint: Path | None = None) -> dict:
        return {"task": task, "model": model_id, "recipe": recipe["name"], "extra": extra, "device": device,
                "checkpoint": checkpoint}

    common = types.ModuleType("app.runners.common")
    common.run_env = run_env
    llm = types.ModuleType("app.runners.llm")
    llm.run_generate = fake("generate")
    vision = types.ModuleType("app.runners.vision")
    vision.run_detect, vision.run_classify = fake("detect"), fake("classify")
    for module in (common, llm, vision):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(app.runners, "vision", vision, raising=False)

    def progress(stage: str, fraction: float, partial: dict | None = None) -> None:
        return None

    image, weights, recipe = tmp_path / "a.png", tmp_path / "r.pth", {"name": "studio:x"}
    result = server.run_live(RunArgs("llm.generate", "org/tiny-llm", recipe, None, "cuda", prompt="hi",
                                     max_new_tokens=4), progress)
    assert result == {"metrics": {}, "env": {"task": "llm.generate", "model": "org/tiny-llm",
                                             "recipe": "studio:x", "extra": None, "device": "cuda",
                                             "checkpoint": None}}
    server.run_live(RunArgs("vision.detect", "yolo11n", recipe, None, "cuda", image_path=image), progress)
    classify = server.run_live(RunArgs("vision.classify", "resnet18", recipe, {"name": "fp64"}, "cpu",
                                       image_path=image, checkpoint=weights), progress)
    assert (classify["env"]["device"], classify["env"]["checkpoint"]) == ("cpu", weights)
    assert seen == [
        ("generate", ("org/tiny-llm", recipe, None, "hi", 4, "cuda"), {"progress": progress}),
        ("detect", ("yolo11n", recipe, None, image, "cuda"), {"progress": progress, "checkpoint": None}),
        ("classify", ("resnet18", recipe, {"name": "fp64"}, image, "cpu"),
         {"progress": progress, "checkpoint": weights}),
    ]
