"""Wanda, SparseGPT and magnitude pruning: the wanda repository's ``lib.prune.prune_<method>``, unchanged.

Glue only: the ``args`` namespace official ``main.py`` would build, its seeding, the model adaptation of
``compat.official_compat``, an optional data-source override, and a report (official ``check_sparsity``
plus an own zero count). Pruning zeros weights in place.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .compat import official_compat
from .repos import REPOS, head, load_module

Method = Literal["wanda", "sparsegpt", "magnitude"]
METHODS = ("wanda", "sparsegpt", "magnitude")
# official main.py: --sparsity_type choices; structured N:M requires --sparsity_ratio 0.5
SPARSITY_TYPES = {(0, 0): "unstructured", (2, 4): "2:4", (4, 8): "4:8"}
DATASETS = ("c4", "wikitext2")  # official lib/data.py get_loaders
WANDA_ROWS = 128  # official lib/prune.py prepare_calibration_input allocates 128 calibration rows


def check_args(method: str, sparsity_ratio: float, prune_n: int = 0, prune_m: int = 0,
               nsamples: int = 128, dataset: str = "c4") -> str:
    """Official ``main.py``'s ``sparsity_type`` for these settings; refuses what it or the code cannot run."""
    if method not in METHODS:
        raise ValueError(f"method: expected one of {METHODS}, got {method!r}")
    kind = SPARSITY_TYPES.get((prune_n, prune_m))
    if kind is None:
        raise ValueError("n, m: the official driver offers unstructured (0, 0), 2:4 and 4:8, "
                         f"got {prune_n}:{prune_m}")
    if not 0 < sparsity_ratio < 1:
        raise ValueError(f"ratio: expected 0 < ratio < 1, got {sparsity_ratio!r}")
    if kind != "unstructured" and sparsity_ratio != 0.5:
        raise ValueError("ratio: must be 0.5 for structured N:M sparsity (official main.py)")
    if nsamples < 1:
        raise ValueError(f"nsamples: expected >= 1, got {nsamples!r}")
    if method == "wanda" and nsamples > WANDA_ROWS:
        raise ValueError(f"nsamples: official wanda's calibration buffer has {WANDA_ROWS} rows, "
                         f"got {nsamples}")
    if dataset not in DATASETS:
        raise ValueError(f"dataset: expected one of {DATASETS} (official get_loaders), got {dataset!r}")
    return kind


@contextlib.contextmanager
def _swapped(module: ModuleType, name: str, value: Any) -> Iterator[None]:
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, saved)


def _data_source(data: ModuleType, dataset: str, loader: Sequence | None, nsamples: int, seqlen: int,
                 requests: list[str]) -> tuple[Callable[..., Any], dict]:
    """Replacement for the ``get_loaders`` name in official ``lib.prune`` (which hard-codes ``"c4"``)."""
    if loader is not None:
        batches = list(loader)
        if len(batches) != nsamples:
            raise ValueError(f"loader: expected {nsamples} batches (nsamples), got {len(batches)}")
        for batch in batches:
            if tuple(batch[0].shape) != (1, seqlen):
                raise ValueError(f"loader: batch[0] must have shape (1, {seqlen}), "
                                 f"got {tuple(batch[0].shape)}")

        def override(name: str, **kwargs: Any) -> tuple[list, None]:
            requests.append(name)
            return batches, None

        return override, {"source": "loader override", "batches": nsamples}

    def official(name: str, **kwargs: Any) -> Any:
        requests.append(name)
        return data.get_loaders(dataset, **kwargs)

    return official, {"source": f"official lib.data.get_loaders({dataset!r})", "dataset": dataset}


def _c4_config_shim(load_dataset: Callable[..., Any], deviations: list[str | dict]) -> Callable[..., Any]:
    """datasets>=2.x: the script-less ``allenai/c4`` hub repo has no config ``allenai--c4`` (builder.py
    raises "BuilderConfig not found"); the same data_files and split load under config ``en``."""

    def shim(path: str, name: str | None = None, *args: Any, **kwargs: Any) -> Any:
        if (path, name) == ("allenai/c4", "allenai--c4"):
            note = ("datasets API drift: load_dataset('allenai/c4', 'allenai--c4', ...) called with "
                    "config 'en' (same data_files and split)")
            if note not in deviations:
                deviations.append(note)
            name = "en"
        return load_dataset(path, name, *args, **kwargs)

    return shim


def prune(model: nn.Module, tokenizer: Any, method: Method, sparsity_ratio: float, prune_n: int = 0,
          prune_m: int = 0, nsamples: int = 128, seed: int = 0, device: str | torch.device | None = None,
          loader: Sequence | None = None, dataset: str = "c4", seqlen: int | None = None) -> dict:
    """Prune ``model`` in place with official ``lib.prune.prune_<method>`` and report what ran.

    ``device`` defaults to the model's. ``model.seqlen`` is ``config.max_position_embeddings`` (official
    ``main.py``) unless ``seqlen`` is given (2 <= seqlen <= max_position_embeddings), which is recorded as
    a parameter mapping in ``deviations``: for models whose official default is infeasible (Qwen3: 40960).
    Data: official ``get_loaders(dataset)`` (``c4`` is the official default), or ``loader``, ``nsamples``
    batches ``(input_ids[1, seqlen], target)``, which replaces only the data source.
    """
    sparsity_type = check_args(method, sparsity_ratio, prune_n, prune_m, nsamples, dataset)
    official_seqlen = int(model.config.max_position_embeddings)
    if seqlen is not None and not 2 <= seqlen <= official_seqlen:
        raise ValueError(f"seqlen: expected 2 <= seqlen <= max_position_embeddings ({official_seqlen}), "
                         f"got {seqlen!r}")
    official = load_module("wanda", "lib.prune")
    data = load_module("wanda", "lib.data")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    mapping = None if seqlen is None else {
        "param": "seqlen", "official": official_seqlen, "used": seqlen,
        "reason": f"official default infeasible for this model (max_position_embeddings {official_seqlen}); "
                  "paper setting",
    }
    seqlen = official_seqlen if seqlen is None else seqlen
    args = argparse.Namespace(
        model=getattr(model.config, "_name_or_path", ""), seed=seed, nsamples=nsamples,
        sparsity_ratio=sparsity_ratio, sparsity_type=sparsity_type, prune_method=method,
        cache_dir="llm_weights", use_variant=False, save=None, save_model=None, eval_zero_shot=False,
    )
    requests: list[str] = []
    deviations: list[str | dict] = [
        "args: argparse.Namespace with official main.py's fields; seeds as main.py"]
    if mapping is not None:
        deviations.append(mapping)
    source, data_report = _data_source(data, dataset, loader, nsamples, seqlen, requests)
    if method == "magnitude":
        data_report = {"source": "none (magnitude pruning reads no data)"}
    elif loader is not None:
        deviations.append("data: loader override in place of official get_loaders('c4')")
    elif dataset != "c4":
        deviations.append(f"data: official get_loaders({dataset!r}) in place of the hard-coded 'c4'")
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    started = time.perf_counter()
    with (official_compat(model, seqlen) as applied, _swapped(official, "get_loaders", source),
          _swapped(data, "load_dataset", _c4_config_shim(data.load_dataset, deviations))):
        getattr(official, f"prune_{method}")(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    wall_time = time.perf_counter() - started
    with contextlib.redirect_stdout(io.StringIO()) as log:
        overall = official.check_sparsity(model)
    per_layer = [float(value)
                 for value in re.findall(r"^layer \d+ sparsity ([0-9.]+)$", log.getvalue(), re.M)]
    linears, short_groups = [], 0
    for index, layer in enumerate(model.model.layers):
        for name, module in official.find_layers(layer).items():
            weight = module.weight.data
            zeros = int((weight == 0).sum())
            linears.append({"name": f"model.layers.{index}.{name}", "zeros": zeros, "numel": weight.numel(),
                            "density": 1 - zeros / weight.numel()})
            if prune_n:
                full = weight.shape[1] // prune_m * prune_m
                groups = weight[:, :full].reshape(weight.shape[0], -1, prune_m)
                short_groups += int(((groups == 0).sum(-1) < prune_n).sum())
    repo = REPOS["wanda"]
    return {
        "official": {"repo": "wanda", "url": repo.url, "commit": repo.commit, "license": repo.license,
                     "function": f"lib.prune.prune_{method}"},
        "official_commit": head(Path(official.__file__).resolve().parents[1]),  # the clone actually imported
        "method": method, "args": vars(args), "prune_n": prune_n, "prune_m": prune_m, "device": str(device),
        "seqlen": seqlen, "data": {**data_report, "requests": requests},
        "deviations": [*applied, *deviations],
        "check_sparsity": {"overall": overall, "layers": per_layer},
        "linears": linears,
        "nm": {"n": prune_n, "m": prune_m, "short_groups": short_groups, "ok": short_groups == 0}
        if prune_n else None,
        "wall_time_s": wall_time,
    }
