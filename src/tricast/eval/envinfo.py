"""Reproducibility metadata without importing optional compute backends."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def _command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _tree_hash(directory: Path, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        name = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def source_identity(root: Path | None = None) -> dict:
    """Identify the commit and current source tree, including untracked source files.

    Only a source checkout (``<git top level>/src/tricast``) has a commit. An installed copy
    reports no commit and the hash of the installed package files, so it is never resumed as
    if it were some checkout that happens to enclose it."""
    if root is None:
        package = Path(__file__).resolve().parents[1]
        top = _command(["git", "rev-parse", "--show-toplevel"], package)
        if not top or Path(top).resolve() / "src" / "tricast" != package:
            return {"git_sha": None, "git_dirty": None, "src_sha256": _tree_hash(package, package.parent)}
        root = Path(top).resolve()
    sha = _command(["git", "rev-parse", "HEAD"], root)
    status = _command(["git", "status", "--porcelain"], root) if sha else None
    dirty = bool(status) if status is not None else None
    source_hash = _tree_hash(root / "src", root) if dirty else None
    return {"git_sha": sha, "git_dirty": dirty, "src_sha256": source_hash}


def local_model_revision(path: Path) -> str | None:
    """Hash config contents and weight names, sizes, and nanosecond modification times."""
    config = path / "config.json" if path.is_dir() else path.with_name("config.json")
    if not config.is_file():
        return None
    digest = hashlib.sha256(config.read_bytes())
    weights = sorted(path.rglob("*")) if path.is_dir() else [path]
    suffixes = {".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack"}
    found_weights = False
    for weight in weights:
        if not weight.is_file() or weight.suffix not in suffixes:
            continue
        found_weights = True
        stat = weight.stat()
        name = weight.relative_to(path).as_posix() if path.is_dir() else weight.name
        digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest() if found_weights else None


def capture_env(model_id: str | None = None, extra: dict | None = None) -> dict:
    """Capture source/version/device identity; unavailable metadata is explicitly null."""
    versions = {"python": platform.python_version()}
    for package in ("torch", "triton", "transformers", "lm_eval", "datasets"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    model_sha = None
    model_kind = None
    if model_id and Path(model_id).exists():
        model_kind = "local"
        model_sha = local_model_revision(Path(model_id))
    elif model_id:
        model_kind = "hf"
        try:
            from huggingface_hub import HfApi

            model_sha = HfApi().model_info(model_id, timeout=5).sha
        except Exception:
            # Offline/private/deleted models must not prevent saving evaluation evidence.
            model_sha = None
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    env = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        **source_identity(),
        "versions": versions,
        "gpu_names": gpu_names,
        "gpu_driver": _command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "cuda": torch.version.cuda,
        "model_id": model_id,
        "model_sha": model_sha,
        "model_kind": model_kind,
        "environment": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "CUBLAS_WORKSPACE_CONFIG", "CONDA_DEFAULT_ENV"
        )},
    }
    if extra:
        env.update(extra)
    return env
