"""Pinned official repositories: find or clone them, verify the pin, import their modules in isolation.

Nothing here edits a clone. A clone is used only at its pinned commit with an unmodified tracked tree.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class OfficialRepo:
    url: str
    commit: str  # full SHA the clone must be at
    license: str
    entry_files: tuple[str, ...]


REPOS: dict[str, OfficialRepo] = {
    "wanda": OfficialRepo(
        "https://github.com/locuslab/wanda.git", "8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6", "MIT",
        ("main.py", "lib/prune.py", "lib/data.py", "lib/layerwrapper.py", "lib/sparsegpt.py"),
    ),
    "sparsegpt": OfficialRepo(
        "https://github.com/IST-DASLab/sparsegpt.git", "147d2159dc4f3e9f73e47b32c04d7b3708f44436",
        "Apache-2.0", ("sparsegpt.py", "quant.py", "datautils.py", "llama.py"),
    ),
    "gptq": OfficialRepo(
        "https://github.com/IST-DASLab/gptq.git", "2d65066eeb06a5c9ff5184d8cebdf33662c67faf",
        "Apache-2.0", ("gptq.py", "quant.py", "datautils.py", "llama.py"),
    ),
    "awq": OfficialRepo(  # read by the fidelity test of tricast.weight_quant.clip (no runtime use; the
        # modules import the awq_inference_engine CUDA extension, so the test runs the two pinned functions)
        "https://github.com/mit-han-lab/llm-awq.git", "d6e797a42b9ef7778de8ee2352116e0f48a78d61", "MIT",
        ("awq/quantize/auto_clip.py", "awq/quantize/quantizer.py"),
    ),
    "razer": OfficialRepo(  # read by the fidelity test of tricast.weight_quant.razer (no runtime use)
        "https://github.com/abdelfattah-lab/NVFP4-RaZeR.git", "78be7fc78857b635ef789edf817cab77cc7f4a01",
        "MIT",
        ("quantize/__init__.py", "quantize/quantizer.py", "quantize/quant_config.py", "quantize/utils.py"),
    ),
}
_PACKAGE = "_tricast_official_{}"


def official_dir() -> Path:
    """Directory holding the clones: ``$TRICAST_OFFICIAL_DIR``, else ``~/.cache/tricast/official``."""
    return Path(os.environ.get("TRICAST_OFFICIAL_DIR") or "~/.cache/tricast/official").expanduser()


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} in {path} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def head(path: Path) -> str:
    """Full SHA of the commit checked out in the clone at ``path``."""
    return _git(path, "rev-parse", "HEAD")


def ensure(name: str) -> Path:
    """The clone of ``name`` at its pinned commit, cloned when absent.

    Refuses a clone at another commit or with changed tracked files; untracked files (``__pycache__``)
    are not part of the pinned tree.
    """
    if name not in REPOS:
        raise ValueError(f"unknown official repository {name!r}; known: {sorted(REPOS)}")
    repo = REPOS[name]
    path = official_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(path.parent, "clone", "--quiet", repo.url, str(path))
        _git(path, "checkout", "--quiet", "--detach", repo.commit)
    checked_out = head(path)
    if checked_out != repo.commit:
        raise RuntimeError(f"{name}: {path} is at {checked_out}, not the pinned commit {repo.commit}")
    changed = _git(path, "status", "--porcelain", "--untracked-files=no")
    if changed:
        raise RuntimeError(f"{name}: tracked files in {path} differ from {repo.commit}:\n{changed}")
    missing = [file for file in repo.entry_files if not (path / file).is_file()]
    if missing:
        raise RuntimeError(f"{name}: entry files missing in {path}: {missing}")
    return path


def _siblings(root: Path) -> set[str]:
    """Top-level names the clone provides on ``sys.path``: its ``*.py`` modules and directories."""
    names = {entry.stem if entry.is_file() else entry.name for entry in root.iterdir()
             if (entry.is_file() and entry.suffix == ".py") or entry.is_dir()}
    return {name for name in names if name.isidentifier()}


def _inside(module: object, root: Path) -> bool:
    locations = [getattr(module, "__file__", None), *getattr(module, "__path__", [])]
    locations = [location for location in locations if location]
    return bool(locations) and all(Path(location).resolve().is_relative_to(root) for location in locations)


def _check_siblings(modules: list[ModuleType], root: Path, names: set[str]) -> None:
    """Every global of these clone modules that comes from a sibling name (``quantize`` from
    ``from quant import *``, or a module ``quant``) must be defined inside the clone ``root``."""
    for module in modules:
        for key, value in vars(module).items():
            origin = value.__name__ if isinstance(value, ModuleType) else getattr(value, "__module__", None)
            if not isinstance(origin, str) or origin.split(".")[0] not in names:
                continue
            code = getattr(value, "__code__", None)
            if isinstance(value, ModuleType):
                ok = _inside(value, root)
            elif code is not None:
                ok = Path(code.co_filename).resolve().is_relative_to(root)
            else:
                ok = _inside(sys.modules.get(origin), root)
            if not ok:
                raise RuntimeError(f"{module.__name__}.{key} comes from module {origin!r} outside the pinned "
                                   f"clone {root}")


def load_module(name: str, dotted: str) -> ModuleType:
    """Import ``dotted`` (e.g. ``lib.prune``) from the pinned clone as ``_tricast_official_<name>.<dotted>``.

    The clone is on ``sys.path`` only during the import, because the 2023 scripts import their siblings
    as top-level modules (``from quant import *``). While it runs, every top-level name the clone provides
    resolves only to the clone: foreign ``sys.modules`` entries under those names are set aside, and
    clone modules loaded earlier are exposed under their bare names. Afterwards the bare names are moved
    under the private package, the foreign entries are restored exactly, and every global that came from
    a sibling name is checked to come from the clone. Bytecode is not written into the clone.
    """
    root = ensure(name)
    resolved = root.resolve()
    package = _PACKAGE.format(name)
    if package in sys.modules:
        if list(sys.modules[package].__path__) != [str(root)]:
            raise RuntimeError(f"{package} was imported from {sys.modules[package].__path__}, not {root}")
    else:
        spec = importlib.machinery.ModuleSpec(package, None, is_package=True)
        spec.submodule_search_locations = [str(root)]
        sys.modules[package] = importlib.util.module_from_spec(spec)
    names = _siblings(root)
    shadowed = {key: sys.modules.pop(key) for key in list(sys.modules) if key.split(".")[0] in names}
    prefix = package + "."
    for key in [key for key in sys.modules if key.startswith(prefix)]:
        sys.modules[key[len(prefix):]] = sys.modules[key]
    before = set(sys.modules)
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(prefix + dotted)
        loaded = [sys.modules[key] for key in sorted(set(sys.modules) - before)
                  if key.split(".")[0] in names or key.startswith(prefix)]
        _check_siblings([module, *loaded], resolved, names)
        return module
    finally:
        sys.path.remove(str(root))
        sys.dont_write_bytecode = bytecode
        for key in [key for key in sys.modules if key.split(".")[0] in names]:
            sys.modules[prefix + key] = sys.modules.pop(key)
        sys.modules.update(shadowed)
