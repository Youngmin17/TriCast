"""TriCast Studio HTTP server: the FastAPI app factory and ``python -m app.server``.

The API contract is app/README.md. Two modes answer it:
- demo serves the static GUI (app/web, recorded bundle in app/web/demo included) plus ``/api/health``,
  ``/api/catalog`` (the bundle's catalog.json) and ``/api/resources`` (no server GPU). The GUI reads the
  recorded runs as static files. Nothing from torch, tricast or the compute modules is imported.
- live builds the catalog with app.catalog and runs comparisons on ``--device`` through the sequential
  job queue in app/jobs.py. The compute modules are imported only in this mode.

Local tool: there is no authentication and no CORS. The server binds 127.0.0.1 by default; anyone who
can reach the port can queue GPU work and read every run, so do not expose it on a shared network.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib
import io
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException

from . import listing
from .jobs import JobQueue, Progress

log = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
DEFAULT_DEMO_DIR = WEB_DIR / "demo"
DEFAULT_RUNS_DIR = APP_DIR.parent / "runs" / "studio"
MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS = 64
DEFAULT_NEW_TOKENS = 32
MAX_IMAGE_BYTES = 8 * 1024 * 1024
RESOURCE_TTL_S = 5.0
# Run ids and media file names: [A-Za-z0-9._-], no leading dot (so never "." or ".."), at most 128.
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")
_DATA_URL = re.compile(r"data:image/(?:jpeg|png);base64,(.+)")
_NVIDIA_SMI = ("nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits")
_VISION_PACKAGES = {"vision.detect": "ultralytics", "vision.classify": "torchvision"}


class RunInput(BaseModel):
    prompt: str | None = None
    max_new_tokens: int | None = None
    image: str | None = None


class RunRequest(BaseModel):
    """``POST /api/runs`` body; ``format`` must be present and may be null."""

    task: str
    model: str
    mma: dict
    format: str | None
    baseline: str = "native"
    input: RunInput = Field(default_factory=RunInput)


@dataclass(frozen=True)
class RunArgs:
    """Everything one live comparison needs; ``checkpoint`` is the vision flag's path, if one was given."""

    task: str
    model: str
    recipe: dict
    baseline_recipe: dict | None
    device: str
    prompt: str | None = None
    max_new_tokens: int | None = None
    image_path: Path | None = None
    checkpoint: Path | None = None


Runner = Callable[[RunArgs, Progress], dict]


class ApiError(Exception):
    """Answered as ``{"error": {"code", "message"}}`` with HTTP ``status``."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _bad(message: str) -> ApiError:
    return ApiError(400, "invalid_request", message)


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def _safe_name(value: str) -> str:
    if _SAFE_NAME.fullmatch(value) is None:
        raise ApiError(400, "invalid_id", f"허용되지 않는 이름입니다 (영문·숫자·'._-' 만): {value!r}")
    return value


def _with_listing(task: str, run: Callable[[Progress], dict], progress: Progress) -> dict:
    result = run(progress)
    return {**result, "summary": listing.summary(task, result["metrics"]),
            "preview": listing.preview(task, result["baseline"], result["emulated"])}


class DemoBackend:
    """The recorded catalog and no server GPU; the GUI reads index, runs and media as static files."""

    def __init__(self, demo_dir: Path) -> None:
        self.catalog = {**json.loads((demo_dir / "catalog.json").read_text(encoding="utf-8")), "mode": "demo"}

    def resources(self) -> dict:
        return {"mode": "demo", "server": None}


class LiveBackend:
    """Validates requests against the live catalog and runs them through the job queue.

    ``checkpoints`` maps a vision model id to the checkpoint given on the command line."""

    def __init__(self, device: str, demo_dir: Path, runner: Runner, runs_dir: Path,
                 checkpoints: dict[str, Path], identity: dict | None = None) -> None:
        self._compute = importlib.import_module("app.catalog")
        # Part of every run's cache key: a result from another device or other code is never reused.
        self._identity = identity or {"device": device}
        self.catalog = self._compute.build_catalog("live", _device_info(device))
        self._device = device
        self._uploads = runs_dir / "uploads"  # kept with the runs, so a restored run still serves its image
        self._uploads.mkdir(parents=True, exist_ok=True)
        self._demo_media = demo_dir / "media"
        self._runner = runner
        self._checkpoints = checkpoints
        self._jobs = JobQueue(runs_dir)
        self._probe_lock = threading.Lock()
        self._probe: dict | None = None
        self._probed_at = 0.0

    def resources(self) -> dict:
        """``GET /api/resources``: the probes are reused for RESOURCE_TTL_S seconds, the queue is live."""
        with self._probe_lock:
            if self._probe is None or time.monotonic() - self._probed_at >= RESOURCE_TTL_S:
                self._probe, self._probed_at = self._probe_server(), time.monotonic()
            probe = self._probe
        return {"mode": "live", "server": {**probe, "queue": self._jobs.counts()}}

    def list_runs(self) -> list[dict]:
        return self._jobs.summaries()

    def get_run(self, run_id: str) -> dict:
        record = self._jobs.get(run_id)
        if record is None:
            raise ApiError(404, "not_found", f"실행을 찾을 수 없습니다: {run_id}")
        return record

    def media(self, name: str) -> Path | None:
        found = (folder / name for folder in (self._uploads, self._demo_media) if (folder / name).is_file())
        return next(found, None)

    def create_run(self, body: RunRequest) -> JSONResponse:
        """202 with a new or still unfinished run's id; 200 and ``cached`` for a finished one."""
        self._check_selection(body)
        if not isinstance(body.mma.get("algorithm"), str):
            raise _bad("mma.algorithm 은 알고리즘 id 문자열이어야 합니다.")
        try:
            mma = self._compute.canonical_mma(body.mma)
        except ValueError as exc:
            raise ApiError(400, "invalid_mma", str(exc)) from None
        try:
            recipe = self._compute.compose_recipe(mma, body.format, body.task)
            baseline = self._compute.baseline_recipe(body.baseline, body.format)
        except ValueError as exc:
            raise ApiError(400, "unsupported_combination", str(exc)) from None
        inputs, run_inputs = self._inputs(body)
        request = {"task": body.task, "model": body.model, "mma": mma, "format": body.format,
                   "baseline": body.baseline, "input": inputs}
        fields = {"request": request, "server": self._identity, "mma_label": self._compute.mma_label(mma),
                  "preset": self._compute.preset_for(mma, body.format),
                  "recipe": self._compute.recipe_record(recipe)}
        args = RunArgs(body.task, body.model, recipe, baseline, self._device, **run_inputs)
        run = partial(self._runner, args)
        job_id, status = self._jobs.submit(fields, partial(_with_listing, body.task, run))
        if status == "done":
            return JSONResponse({"id": job_id, "cached": True})
        return JSONResponse({"id": job_id}, status_code=202)

    def _check_selection(self, body: RunRequest) -> None:
        for value, items, what in ((body.task, "tasks", "작업"), (body.format, "formats", "형식"),
                                   (body.baseline, "baselines", "기준선")):
            if value not in {item["id"] for item in self.catalog[items]}:
                raise _bad(f"카탈로그에 없는 {what}입니다: {value}")
        model = next((item for item in self.catalog["models"] if item["id"] == body.model), None)
        if model is None:
            raise _bad(f"카탈로그에 없는 모델입니다: {body.model}")
        if body.task not in model["tasks"]:
            raise _bad(f"{body.model} 모델은 {body.task} 작업을 지원하지 않습니다.")

    def _inputs(self, body: RunRequest) -> tuple[dict, dict]:
        """The request's stored ``input`` and the matching :class:`RunArgs` fields."""
        if body.task == "llm.generate":
            inputs = {"prompt": _prompt(body.input.prompt),
                      "max_new_tokens": _new_tokens(body.input.max_new_tokens)}
            return inputs, inputs
        path = self._image(body.input.image)
        return {"image": path.name}, {"image_path": path, "checkpoint": self._checkpoints.get(body.model)}

    def _image(self, value: str | None) -> Path:
        if not value:
            raise _bad("이미지를 고르거나 업로드하세요.")
        if value.startswith("data:"):
            return _store_upload(value, self._uploads)
        path = self.media(_safe_name(value))
        if path is None:
            raise _bad(f"이미지를 찾을 수 없습니다: {value}")
        return path

    def _probe_server(self) -> dict:
        cached = [model["id"] for model in self.catalog["models"] if self._is_cached(model)]
        runnable = [task["id"] for task in self.catalog["tasks"] if self._can_run(task["id"], cached)]
        return {"hostname": socket.gethostname(), "device": self._device, "gpus": _gpus(), **_versions(),
                "models_cached": cached, "runnable_tasks": runnable}

    def _is_cached(self, model: dict) -> bool:
        """Vision models need the checkpoint file their runner loads, the others a Hugging Face snapshot."""
        if any(task.startswith("vision.") for task in model["tasks"]):
            return _checkpoint_file(model["id"], self._checkpoints.get(model["id"])).is_file()
        return _hf_snapshot(model["id"], model["revision"])

    def _can_run(self, task: str, cached: list[str]) -> bool:
        """A cached model for the task plus CUDA (llm) or the vision package (detect, classify)."""
        if not any(task in model["tasks"] and model["id"] in cached for model in self.catalog["models"]):
            return False
        return self._device == "cuda" if task.startswith("llm.") else _importable(_VISION_PACKAGES[task])


def _prompt(prompt: str | None) -> str:
    if not (prompt or "").strip():
        raise _bad("프롬프트를 입력하세요.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise _bad(f"프롬프트는 {MAX_PROMPT_CHARS:,}자 이하여야 합니다 (지금 {len(prompt):,}자).")
    return prompt


def _new_tokens(value: int | None) -> int:
    if value is None:
        return DEFAULT_NEW_TOKENS
    if not 1 <= value <= MAX_NEW_TOKENS:
        raise _bad(f"max_new_tokens 는 1 이상 {MAX_NEW_TOKENS} 이하여야 합니다 (지금 {value}).")
    return value


def _store_upload(data_url: str, directory: Path) -> Path:
    """Decode a JPEG/PNG data URL, check it with Pillow and save it under a name made from its content,
    so the same image uploaded again maps to the same media name (and the same cached run)."""
    from PIL import Image

    match = _DATA_URL.fullmatch(data_url)
    if match is None:
        raise _bad("업로드 이미지는 data:image/jpeg 또는 data:image/png 의 base64 data URL 이어야 합니다.")
    try:
        raw = base64.b64decode(match.group(1), validate=True)
    except binascii.Error:
        raise _bad("이미지의 base64 를 해독할 수 없습니다.") from None
    if len(raw) > MAX_IMAGE_BYTES:
        size = f"{MAX_IMAGE_BYTES // 2**20} MB 이하여야 합니다 (지금 {len(raw) / 2**20:.1f} MB)"
        raise ApiError(413, "image_too_large", f"이미지는 {size}.")
    try:
        with Image.open(io.BytesIO(raw), formats=["JPEG", "PNG"]) as image:
            kind = image.format
            image.verify()
    except Exception:  # Pillow reports malformed files with many exception types
        raise _bad("이미지를 읽을 수 없습니다. JPEG 또는 PNG 파일만 받습니다.") from None
    path = directory / f"upload-{hashlib.sha256(raw).hexdigest()}.{'jpg' if kind == 'JPEG' else 'png'}"
    if not path.is_file():  # the name is the content hash: an existing file already holds these bytes
        partial_file = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        partial_file.write_bytes(raw)
        os.replace(partial_file, path)
    return path


def _device_info(device: str) -> dict:
    """The catalog's ``device`` block. ``backend`` is what the runners request (``resolve_backend`` in
    app/runners/llm.py): Triton on CUDA, the reference on CPU."""
    if device == "cpu":
        return {"kind": "cpu", "name": None, "backend": "reference"}
    import torch

    return {"kind": "cuda", "name": torch.cuda.get_device_name(0), "backend": "triton"}


def _gpus() -> list[dict]:
    """The GPUs nvidia-smi lists (5 s timeout) that CUDA_VISIBLE_DEVICES (by index) leaves visible;
    an empty list when nvidia-smi is missing, fails or times out."""
    try:
        smi = subprocess.run(_NVIDIA_SMI, capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    keep = None if visible is None else {item.strip() for item in visible.split(",")}
    gpus = []
    for line in smi.splitlines():
        index, name, total, used, utilization = (part.strip() for part in line.split(","))
        if keep is None or index in keep:
            gpus.append({"index": int(index), "name": name, "memory_total_mb": _smi_number(total),
                         "memory_used_mb": _smi_number(used), "utilization_pct": _smi_number(utilization)})
    return gpus


def _smi_number(text: str) -> int | None:
    """An nvidia-smi count; "[N/A]" and "[Not Supported]" become null."""
    return int(text) if text.isdigit() else None


def _versions() -> dict:
    """Installed torch and triton versions and torch's CUDA build; torch is imported only if installed."""
    found: dict[str, str | None] = {}
    for package in ("torch", "triton"):
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = None
    cuda = None
    if found["torch"] is not None:
        import torch

        cuda = torch.version.cuda
    return {**found, "cuda": cuda}


def _importable(module: str) -> bool:
    """Whether ``module`` can be found, without importing it (ultralytics is slow to import)."""
    return find_spec(module) is not None


def _hf_snapshot(model_id: str, revision: str | None) -> bool:
    """Whether the Hugging Face hub cache (``HF_HUB_CACHE``, by default under ``HF_HOME``) holds the
    model's snapshot at ``revision`` (any snapshot when it is null)."""
    try:
        from huggingface_hub import constants
    except ImportError:
        return False
    snapshots = Path(constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if revision is not None:
        return (snapshots / revision).is_dir()
    return snapshots.is_dir() and any(snapshots.iterdir())


def _checkpoint_file(model_id: str, checkpoint: Path | None) -> Path:
    """The file the vision runner loads: ``checkpoint`` if given, else its env variable or default path."""
    from .runners.common import checkpoint_path

    return checkpoint_path(model_id, checkpoint)


def run_live(args: RunArgs, progress: Progress) -> dict:
    """Run one comparison with the compute modules and attach the run's ``env`` (``run_env``: pinned
    model revision without a Hub lookup, or the vision checkpoint's path and SHA256)."""
    from .runners.common import run_env

    if args.task == "llm.generate":
        from .runners.llm import run_generate

        result = run_generate(args.model, args.recipe, args.baseline_recipe, args.prompt, args.max_new_tokens,
                              args.device, progress=progress)
    else:
        from .runners import vision

        run = vision.run_detect if args.task == "vision.detect" else vision.run_classify
        result = run(args.model, args.recipe, args.baseline_recipe, args.image_path, args.device,
                     progress=progress, checkpoint=args.checkpoint)
    env = run_env(args.task, args.model, args.recipe, args.baseline_recipe, device=args.device,
                  checkpoint=args.checkpoint)
    return {**result, "env": env}


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(f"{'.'.join(str(part) for part in item['loc'][1:])}: {item['msg']}"
                             for item in exc.errors())
        return _error(400, "invalid_request", f"요청 형식이 올바르지 않습니다: {problems}")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail), exc.headers)

    @app.exception_handler(Exception)
    async def internal_error(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error", exc_info=exc)
        return _error(500, "internal_error", "서버 내부 오류입니다. 서버 로그를 확인하세요.")


def _add_run_routes(app: FastAPI, live: LiveBackend) -> None:
    @app.get("/api/runs")
    def list_runs() -> dict:
        return {"runs": live.list_runs()}

    @app.post("/api/runs")
    def create_run(body: RunRequest) -> JSONResponse:
        return live.create_run(body)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return live.get_run(_safe_name(run_id))

    @app.get("/api/media/{name}")
    def media(name: str) -> FileResponse:
        path = live.media(_safe_name(name))
        if path is None:
            raise ApiError(404, "not_found", f"이미지를 찾을 수 없습니다: {name}")
        return FileResponse(path)


def create_app(mode: str, device: str, demo_dir: Path, runner: Runner | None = None, *,
               runs_dir: Path = DEFAULT_RUNS_DIR, checkpoints: dict[str, Path] | None = None,
               identity: dict | None = None) -> FastAPI:
    """Build the app; ``mode`` is "demo" or "live", ``runner`` replaces :func:`run_live` (tests),
    ``checkpoints`` maps vision model ids to checkpoint files and ``identity`` (device and code
    hashes) joins the run cache key (live)."""
    if mode == "demo":
        backend: DemoBackend | LiveBackend = DemoBackend(demo_dir)
    else:
        backend = LiveBackend(device, demo_dir, runner or run_live, runs_dir, checkpoints or {}, identity)
    app = FastAPI(title="TriCast Studio")
    _install_error_handlers(app)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": mode, "device": device}

    @app.get("/api/resources")
    def resources() -> dict:
        return backend.resources()

    @app.get("/api/catalog")
    def catalog() -> dict:
        return backend.catalog

    if isinstance(backend, LiveBackend):
        _add_run_routes(app, backend)

    @app.api_route("/api/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"])
    def unknown_api(path: str) -> None:
        raise ApiError(404, "not_found", f"없는 API 경로입니다: /api/{path}")

    app.mount("/demo", StaticFiles(directory=demo_dir), name="demo")  # the bundle --demo-dir names
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.server", description="TriCast Studio web server")
    parser.add_argument("--demo", action="store_true", help="static GUI and catalog only (no torch/tricast)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="live-mode device")
    parser.add_argument("--host", default="127.0.0.1", help="no authentication: keep it on loopback")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR, help="recorded bundle directory")
    parser.add_argument("--yolo-checkpoint", type=Path, help="YOLO11n checkpoint, vision.detect (live)")
    parser.add_argument("--resnet-checkpoint", type=Path, help="ResNet18 checkpoint, vision.classify (live)")
    args = parser.parse_args(argv)
    if args.demo and not (args.demo_dir / "catalog.json").is_file():
        parser.error(f"no recorded bundle in {args.demo_dir} (make one with python -m app.demo)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    mode = "demo" if args.demo else "live"
    flags = {"yolo11n": args.yolo_checkpoint, "resnet18": args.resnet_checkpoint}
    checkpoints = {model: path for model, path in flags.items() if path is not None}
    identity = None
    if not args.demo:
        from tricast.eval.envinfo import source_identity

        from .runners.common import app_sha256, configure_torch

        configure_torch()  # before the first CUDA call: the recorded bundle's seed, determinism and no TF32
        source = source_identity()
        identity = {"device": args.device, "app_sha256": app_sha256(), "git_sha": source["git_sha"],
                    "src_sha256": source["src_sha256"]}
    app = create_app(mode, "none" if args.demo else args.device, args.demo_dir, checkpoints=checkpoints,
                     identity=identity)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
