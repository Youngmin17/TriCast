"""/api/quantizer/jobs (app/server.py) with an injected runner: demo and CPU servers refuse with how to start
a GPU server; invalid recipes, tasks, limits and calibration settings are refused with what to fix; the queue,
the result shape and the calibration settings handed to the runner (as tricast's --calib-* flags would set
them).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import quantizer_recipe as qr
from app import server
from app.server import create_app
from app.tests.test_server import CATALOG, fake_catalog
from tricast.recipe import load_recipe

REPO = Path(server.__file__).resolve().parents[1]
TASKS = ["ppl", "report", "arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"]
MIXED = qr.to_yaml(qr.preset("qd_mixed"))
RESULT = {
    "metrics": {"run": {"ppl": 25.0, "n_windows": 146}, "report": None, "lmeval": None},
    "env": {"git_sha": "abc"},
    "evidence": {"out_dir": "x", "files": [], "layer_map_command": None},
}


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "catalog.json").write_text(json.dumps({**CATALOG, "mode": "demo"}), encoding="utf-8")
    return root


class FakeQuantizer:
    """Stands in for run_quantizer_live: reports stages, can be held mid-run, fails for recipe name "fail"."""

    def __init__(self) -> None:
        self.calls: list = []
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, args, progress) -> dict:
        self.calls.append(args)
        progress("모델 불러오기", 0.02)
        progress("WikiText-2 PPL", 0.5)
        self.started.set()
        self.release.wait(5)
        if args.name == "fail":
            raise RuntimeError("boom")
        return RESULT


@pytest.fixture
def live(demo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "app.catalog", fake_catalog())
    monkeypatch.setattr(
        server, "_device_info", lambda device: {"kind": "cuda", "name": "Fake GPU", "backend": "triton"}
    )
    runner = FakeQuantizer()
    app = create_app("live", "cuda", demo_dir, runs_dir=tmp_path / "runs", quantizer_runner=runner)
    return TestClient(app), runner, tmp_path / "runs"


def body(**change) -> dict:
    return {"name": "qd_mixed", "yaml": MIXED, "tasks": ["ppl"], "limit": None, **change}


def error(response) -> tuple[int, str, str]:
    data = response.json()["error"]
    return response.status_code, data["code"], data["message"]


def test_demo_mode_refuses_with_how_to_start_a_gpu_server(demo_dir: Path) -> None:
    client = TestClient(create_app("demo", "none", demo_dir))
    for response in (client.post("/api/quantizer/jobs", json=body()), client.get("/api/quantizer/jobs/abc")):
        status, code, message = error(response)
        assert (status, code) == (409, "live_mode_required")
        assert "python -m app.server --device cuda" in message and "GPU 노드" in message


def test_demo_mode_still_imports_no_torch_or_tricast(demo_dir: Path) -> None:
    code = (
        "import json, sys\nfrom pathlib import Path\nfrom fastapi.testclient import TestClient\n"
        "from app.server import create_app\n"
        "client = TestClient(create_app('demo', 'none', Path(sys.argv[1])))\n"
        "assert client.post('/api/quantizer/jobs', json=json.loads(sys.argv[2])).status_code == 409\n"
        "bad = {'torch', 'tricast'} & {name.split('.')[0] for name in sys.modules}\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code, str(demo_dir), json.dumps(body())], check=True, cwd=REPO)


def test_live_cpu_server_refuses(demo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "app.catalog", fake_catalog())
    client = TestClient(
        create_app("live", "cpu", demo_dir, runs_dir=tmp_path / "runs", quantizer_runner=FakeQuantizer())
    )
    status, code, message = error(client.post("/api/quantizer/jobs", json=body()))
    assert (status, code) == (409, "cuda_required") and "--device cuda" in message


@pytest.mark.parametrize(
    ("change", "code", "fragment"),
    [
        ({"yaml": "name: [unclosed"}, "invalid_recipe", "YAML 을 읽을 수 없습니다 (1번째 줄)"),
        ({"yaml": "- just\n- a list"}, "invalid_recipe", "매핑이어야 합니다"),
        (
            {"yaml": MIXED.replace("name: qd_mixed", "name: other")},
            "invalid_recipe",
            "요청한 이름('qd_mixed')과 다릅니다",
        ),
        (
            {"yaml": MIXED.replace("scheme: mxfp8_e4m3", "scheme: mxfp9", 1)},
            "invalid_recipe",
            "overrides[1].weight.scheme: 'mxfp9' is not one of",
        ),
        ({"yaml": MIXED.replace("preset: kivi2", "preset: kivi9")}, "invalid_recipe", "kv.preset: 'kivi9'"),
        ({"name": "bad name"}, "invalid_request", "레시피 이름은 영문·숫자로 시작"),
        ({"tasks": []}, "invalid_request", "실행할 작업을 하나 이상 고르세요"),
        ({"tasks": ["ppl", "mmlu"]}, "invalid_request", "알 수 없는 작업입니다: mmlu"),
        ({"tasks": ["piqa"], "limit": 0}, "invalid_request", "limit 은 1 이상 300 이하"),
        ({"tasks": ["piqa"], "limit": 301}, "invalid_request", "limit 은 1 이상 300 이하"),
    ],
)
def test_bad_requests_say_what_to_fix(live, change: dict, code: str, fragment: str) -> None:
    client, runner, _ = live
    status, got, message = error(client.post("/api/quantizer/jobs", json=body(**change)))
    assert status == 400 and got == code and fragment in message, message
    assert not runner.calls


def test_oversized_yaml_is_refused(live) -> None:
    client, _, _ = live
    status, code, _ = error(client.post("/api/quantizer/jobs", json=body(yaml=MIXED + "\n#" + "x" * 70000)))
    assert (status, code) == (413, "recipe_too_large")


def test_tasks_are_ordered_and_the_limit_defaults_to_the_cap(live) -> None:
    client, runner, runs = live
    runner.release.set()
    response = client.post("/api/quantizer/jobs", json=body(tasks=["hellaswag", "ppl", "arc_easy", "ppl"]))
    assert response.status_code == 202
    runner.started.wait(5)
    args = runner.calls[0]
    assert args.tasks == ("ppl", "arc_easy", "hellaswag") and args.limit == 300 and args.device == "cuda"
    assert args.recipe == qr.preset("qd_mixed") and args.name == "qd_mixed"
    assert args.out_dir.parent == runs / "quantizer" and args.out_dir.name.startswith("qd_mixed-")
    other = client.post("/api/quantizer/jobs", json=body(tasks=["report"], limit=7))
    assert other.status_code == 202
    for _ in range(100):
        if len(runner.calls) == 2:
            break
        threading.Event().wait(0.05)
    assert runner.calls[1].limit is None  # no lm-eval task, no limit


def test_queue_status_transitions_and_result_shape(live) -> None:
    client, runner, _ = live
    created = client.post("/api/quantizer/jobs", json=body(tasks=["ppl", "piqa"], limit=20))
    assert created.status_code == 202 and created.json()["status"] == "queued"
    job = created.json()["id"]
    assert runner.started.wait(5)
    running = client.get(f"/api/quantizer/jobs/{job}").json()
    assert (
        running["status"] == "running" and running["stage"] == "WikiText-2 PPL" and running["progress"] == 0.5
    )
    assert running["results"] is None and running["request"]["tasks"] == ["ppl", "piqa"]
    assert running["request"]["limit"] == 20 and len(running["request"]["recipe_sha256"]) == 64
    runner.release.set()
    for _ in range(100):
        done = client.get(f"/api/quantizer/jobs/{job}").json()
        if done["status"] == "done":
            break
        threading.Event().wait(0.05)
    assert done["status"] == "done" and done["progress"] == 1.0
    assert done["results"] == RESULT["metrics"] and done["env"] == RESULT["env"]
    assert done["evidence"] == RESULT["evidence"] and done["recipe"]["yaml"] == MIXED
    assert set(done) == {
        "id",
        "status",
        "stage",
        "progress",
        "request",
        "recipe",
        "env",
        "evidence",
        "error",
        "results",
    }
    again = client.post("/api/quantizer/jobs", json=body(tasks=["ppl", "piqa"], limit=20))
    assert again.status_code == 200 and again.json() == {"id": job, "status": "done", "cached": True}
    assert len(runner.calls) == 1
    assert client.get("/api/runs").json() == {"runs": []}  # Quantizer jobs stay out of the Studio run list


def test_a_failed_job_reports_the_error(live) -> None:
    client, runner, _ = live
    runner.release.set()
    yaml_text = MIXED.replace("name: qd_mixed", "name: fail")
    job = client.post("/api/quantizer/jobs", json=body(name="fail", yaml=yaml_text)).json()["id"]
    for _ in range(100):
        record = client.get(f"/api/quantizer/jobs/{job}").json()
        if record["status"] == "error":
            break
        threading.Event().wait(0.05)
    assert record["status"] == "error" and record["error"] == {
        "code": "run_failed",
        "message": "RuntimeError: boom",
    }
    assert record["results"] is None


def test_unknown_or_foreign_job_ids_are_404(live) -> None:
    client, _, _ = live
    assert error(client.get("/api/quantizer/jobs/nope"))[:2] == (404, "not_found")
    assert error(client.get("/api/quantizer/jobs/..%2Fsecret"))[0] in (400, 404)


# ---------- calibration settings and page recipes with options ----------
def started(runner) -> object:
    runner.release.set()
    assert runner.started.wait(5)
    return runner.calls[-1]


def test_calibration_reaches_the_runner_like_the_cli_flags(live) -> None:
    client, runner, _ = live
    settings = {"dataset": "c4", "samples": 64, "seqlen": 512, "seed": 7, "sequential": True}
    response = client.post("/api/quantizer/jobs", json=body(calibration=settings))
    assert response.status_code == 202, response.json()
    args = started(runner)
    assert args.calibration == settings and args.recipe == qr.preset(
        "qd_mixed"
    )  # the YAML itself is unchanged
    recipe = load_recipe(qr.preset("qd_mixed"))
    merged = load_recipe({**recipe.to_dict(), "calibration": {**recipe.calibration, **settings}})
    job = client.get(f"/api/quantizer/jobs/{response.json()['id']}").json()
    assert job["request"]["calibration"] == settings
    assert (
        job["request"]["recipe_sha256"] == merged.sha256 != recipe.sha256
    )  # the cache key is the recipe that runs
    assert args.out_dir.name == f"qd_mixed-{merged.sha256[:12]}"


def test_without_calibration_the_request_is_as_before(live) -> None:
    client, runner, _ = live
    response = client.post("/api/quantizer/jobs", json=body())
    args = started(runner)
    job = client.get(f"/api/quantizer/jobs/{response.json()['id']}").json()
    assert args.calibration is None and job["request"]["calibration"] is None
    assert job["request"]["recipe_sha256"] == load_recipe(qr.preset("qd_mixed")).sha256


@pytest.mark.parametrize(
    ("settings", "fragment"),
    [
        ({"samples": 513}, "calibration.samples 는 1 이상 512 이하의 정수여야 합니다 (지금 513)"),
        ({"samples": 0}, "calibration.samples 는 1 이상 512 이하의 정수여야 합니다"),
        ({"samples": True}, "calibration.samples 는 1 이상 512 이하의 정수여야 합니다"),
        ({"seqlen": 4097}, "calibration.seqlen 는 2 이상 4096 이하의 정수여야 합니다"),
        ({"seqlen": 1.5}, "calibration.seqlen 는 2 이상 4096 이하의 정수여야 합니다"),
        ({"dataset": "ptb"}, "calibration.dataset 는 wikitext2, c4, pile 중 하나여야 합니다"),
        ({"sequential": "yes"}, "calibration.sequential 는 true 또는 false여야 합니다"),
        ({"split": "test"}, "calibration 에 모르는 항목이 있습니다: split"),
    ],
)
def test_bad_calibration_settings_say_what_to_fix(live, settings: dict, fragment: str) -> None:
    client, runner, _ = live
    status, code, message = error(client.post("/api/quantizer/jobs", json=body(calibration=settings)))
    assert (status, code) == (400, "invalid_request") and fragment in message, message
    assert not runner.calls


@pytest.mark.parametrize(
    ("block", "fragment"),
    [
        (
            "calibration:\n  samples: 1024\n  seed: 42",
            "서버 보정의 samples 는 1–512 까지입니다 (이 레시피는 1024)",
        ),
        (
            "calibration:\n  seqlen: 8192\n  seed: 42",
            "서버 보정의 seqlen 는 2–4096 까지입니다 (이 레시피는 8192)",
        ),
    ],
)
def test_a_recipe_calibration_beyond_the_server_limits_is_refused(live, block: str, fragment: str) -> None:
    client, runner, _ = live
    text = MIXED.replace("calibration:\n  seed: 42", block)
    status, code, message = error(client.post("/api/quantizer/jobs", json=body(yaml=text)))
    assert (status, code) == (400, "invalid_recipe") and fragment in message and "--calib-" in message, (
        message
    )
    assert not runner.calls


def test_a_recipe_that_needs_no_calibration_is_not_held_to_the_limits(live) -> None:
    client, runner, _ = live
    data = qr.build("qz_plain", "nvfp4", qr.uniform("nvfp4"), calibration={"samples": 512})
    text = qr.to_yaml(data).replace("samples: 512", "samples: 1024")  # unused by an uncalibrated recipe
    response = client.post("/api/quantizer/jobs", json=body(name="qz_plain", yaml=text))
    assert response.status_code == 202, response.json()
    assert started(runner).recipe["calibration"] == {"samples": 1024, "seed": 42}


def test_page_recipes_with_options_are_accepted(live) -> None:
    client, runner, _ = live
    grid = qr.uniform("int8")
    grid[3] = ["gptq_int4"] * len(qr.KINDS)
    options = {
        "int8": {
            "activation.granularity": "tensor",
            "activation.observer.kind": "ema",
            "activation.observer.decay": 0.9,
            "weight.rounding": "rtz",
        },
        "gptq_int4": {"weight.group_size": 64, "weight_algo.act_order": True},
    }
    data = qr.build(
        "qz_opts",
        "int8",
        grid,
        "kivi2",
        options=options,
        calibration={"samples": 32, "seqlen": 256},
        kv_options={"residual": 64},
    )
    response = client.post("/api/quantizer/jobs", json=body(name="qz_opts", yaml=qr.to_yaml(data)))
    assert response.status_code == 202, response.json()
    args = started(runner)
    assert args.recipe == data and args.calibration is None
    recipe = load_recipe(args.recipe)
    assert (
        recipe.needs_calibration and recipe.calibration_options["samples"] == 32 and recipe.kv.residual == 64
    )
    assert (
        recipe.defaults.activation.observer.kind == "ema" and recipe.defaults.weight.rounding.value == "rtz"
    )
