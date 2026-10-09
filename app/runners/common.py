"""Helpers shared by the TriCast Studio runners and app/demo.py: the progress callback type, the explicit
backend choice, the patch context that proves emulation ran (a run's ``evidence``), the bounded baseline
cache, local checkpoint resolution, and the run's environment record (``env``) used by the live server and
the demo alike.
"""

from __future__ import annotations

import hashlib
import os
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from tricast import load_recipe, patch_model
from tricast.eval.envinfo import capture_env
from tricast.nn import iter_emuconv2d, iter_emulinear, unpatch_model

from ..catalog import MODELS

APP = Path(__file__).resolve().parents[1]
REPO = APP.parent
SEED = 42
TOP_K = 5
BASELINE_CACHE_SIZE = 4
DETECTION = {"conf": 0.25, "iou": 0.7, "imgsz": 640, "max_det": 100}
CHECKPOINT_ENV = {"yolo11n": "TRICAST_YOLO_CHECKPOINT", "resnet18": "TRICAST_RESNET_CHECKPOINT"}
DEFAULT_CHECKPOINTS = {"yolo11n": REPO / "checkpoints" / "yolo11n.pt",
                       "resnet18": REPO / "checkpoints" / "resnet18-f37072fd.pth"}
_HASH_SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".DS_Store"}


class Progress(Protocol):
    """``progress(stage, fraction, partial=None)``; ``partial`` carries the baseline Output once it exists."""

    def __call__(self, stage: str, fraction: float, partial: dict | None = None) -> None: ...


def report(progress: Progress | None, stage: str, fraction: float, partial: dict | None = None) -> None:
    if progress is not None:
        progress(stage, fraction, partial=partial)


def resolve_backend(device: str | torch.device, backend: str | None) -> str:
    """The backend to request explicitly: Triton on CUDA, else the reference. tricast's "auto" would
    fall back to the reference silently when the Triton kernels cannot be imported."""
    if backend is not None:
        return backend
    return "triton" if torch.device(device).type == "cuda" else "reference"


def llm_dtype(device: str | torch.device) -> torch.dtype:
    """Language models run in bf16 on CUDA and fp32 elsewhere."""
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


def now(device: str | torch.device) -> float:
    """Wall-clock seconds once the device has finished its queued work."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def recipe_label(recipe: Any | None) -> str:
    if recipe is None:
        return "원본 (native)"
    loaded = load_recipe(recipe)
    return loaded.description or loaded.name


def recipe_key(recipe: Any | None) -> str:
    return "native" if recipe is None else load_recipe(recipe).sha256


def require_finite(tensor: torch.Tensor, what: str) -> torch.Tensor:
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{what}: NaN/Inf 가 있습니다.")
    return tensor


@contextmanager
def emulated(model: nn.Module, recipe: Any, backend: str, *, include_conv2d: bool = False) -> Iterator[dict]:
    """Patch ``model`` with ``recipe`` and yield the run evidence; ``emulated_calls`` counts forward calls
    into patched Linear/Conv2d modules while the context is open. Always unpatches on exit."""
    handles = []
    try:
        patch_model(model, load_recipe(recipe), backend=backend, include_conv2d=include_conv2d)
        linears = [layer for _, layer in iter_emulinear(model)]
        convs = [layer for _, layer in iter_emuconv2d(model)]
        evidence = {"patched_linear": len(linears), "patched_conv2d": len(convs), "emulated_calls": 0,
                    "backend": backend}

        def count(*_: object) -> None:
            evidence["emulated_calls"] += 1

        handles = [layer.register_forward_hook(count) for layer in (*linears, *convs)]
        yield evidence
    finally:
        for handle in handles:
            handle.remove()
        unpatch_model(model)


class BaselineCache:
    """The ``size`` most recent baseline results, keyed by run inputs and bound to one model object (a weak
    reference, so a reloaded model never answers for the one it replaced). Each runner stores only what
    its comparison needs; the runner modules state the per-entry size."""

    def __init__(self, size: int = BASELINE_CACHE_SIZE) -> None:
        self._size = size
        self._items: OrderedDict[tuple, tuple[weakref.ref, Any]] = OrderedDict()

    def get(self, model: nn.Module, key: tuple) -> Any | None:
        item = self._items.get((id(model), *key))
        if item is None or item[0]() is not model:
            return None
        self._items.move_to_end((id(model), *key))
        return item[1]

    def put(self, model: nn.Module, key: tuple, value: Any) -> None:
        self._items[(id(model), *key)] = (weakref.ref(model), value)
        self._items.move_to_end((id(model), *key))
        while len(self._items) > self._size:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


def checkpoint_path(model_id: str, checkpoint: str | Path | None = None) -> Path:
    """Local checkpoint of a vision model: ``checkpoint`` when given, else ``$TRICAST_YOLO_CHECKPOINT`` /
    ``$TRICAST_RESNET_CHECKPOINT``, else ``checkpoints/`` at the repository root."""
    if checkpoint is not None:
        return Path(checkpoint)
    return Path(os.environ.get(CHECKPOINT_ENV[model_id], DEFAULT_CHECKPOINTS[model_id]))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@cache
def app_sha256() -> str:
    """Hash of the app/ tree in the format of ``tricast.eval.envinfo._tree_hash`` (length-prefixed
    repository-relative names and contents, sorted), without the recorded bundle app/web/demo, caches,
    bytecode and .DS_Store. Computed once per process, so it describes the code that was imported."""
    digest = hashlib.sha256()
    bundle = APP / "web" / "demo"
    for path in sorted(APP.rglob("*")):
        if (not path.is_file() or path.suffix in {".pyc", ".pyo"} or path.is_relative_to(bundle)
                or _HASH_SKIP.intersection(path.relative_to(APP).parts)):
            continue
        name = path.relative_to(REPO).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def configure_torch() -> None:
    """Seed 42, deterministic kernels, no TF32 (as scripts/e2e/evaluate_vision_quality.py)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _versions(*packages: str) -> dict:
    found = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def _model_fields(task: str, model_id: str, device: str | None, checkpoint: str | Path | None) -> dict:
    """The pinned model identity: the Hub revision that was requested, or the local checkpoint file."""
    model = MODELS[model_id]
    if task == "llm.generate":
        return {"model_kind": "hf", "model_sha": model.revision, "model_revision": model.revision,
                "dtype": None if device is None else str(llm_dtype(device)).removeprefix("torch."),
                "attn_implementation": "eager",
                "generation": {"do_sample": False, "num_beams": 1, "chat_template": model.chat,
                               "enable_thinking": False if model.chat else None}}
    path = checkpoint_path(model_id, checkpoint)
    digest = file_sha256(path)
    fields = {"model_kind": "local", "model_sha": digest, "model_revision": model.revision,
              "checkpoint": {"path": str(path.resolve()), "sha256": digest}, "dtype": "float32",
              "optional_versions": _versions("ultralytics", "torchvision", "opencv-python",
                                             "opencv-python-headless", "pillow")}
    if task == "vision.detect":
        fields["detection"] = {**DETECTION, "multi_label": True, "agnostic": False,
                               "nms_time_limit": "disabled",
                               "preprocessing": "Ultralytics 8.3.221 rectangular validation letterbox "
                                                "(pad 0.5, no upscaling), /255"}
    else:
        fields["preprocessing"] = "torchvision ResNet18_Weights.IMAGENET1K_V1.transforms()"
    return fields


def run_env(task: str, model_id: str, recipe: Any, baseline_recipe: Any | None, extra: dict | None = None,
            *, device: str | None = None, checkpoint: str | Path | None = None) -> dict:
    """A run's ``env``: ``tricast.eval.envinfo.capture_env`` (source identity, library versions, GPUs) with
    no Hub lookup, plus the app tree hash, the pinned model identity (Hub revision, or the vision checkpoint's
    path and SHA256), both recipe hashes, the seed and the determinism flags in effect, and the task's fixed
    settings. ``device`` adds the device and the LLM dtype; ``checkpoint`` is the file a vision runner was
    given. ``extra`` is applied last."""
    fields = {
        "app": "tricast-studio", "app_sha256": app_sha256(), "task": task, "model_id": model_id,
        "device": device, "seed": SEED,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "reduced_precision_reduction": {
            "bf16": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction},
        "recipe_sha256": recipe_key(recipe),
        "baseline_recipe_sha256": None if baseline_recipe is None else recipe_key(baseline_recipe),
        **_model_fields(task, model_id, device, checkpoint),
    }
    return capture_env(None, extra={**fields, **(extra or {})})
