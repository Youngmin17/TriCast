"""Official GPTQ (IST-DASLab/gptq ``llama.py: llama_sequential``) called unchanged on HF decoders.

Glue only: the module-global ``args`` namespace that ``llama_sequential`` reads (the fields of official
``llama.py``'s argparse), the model adaptation of ``compat.official_compat``, the data source (official
``datautils.get_loaders`` or a ``loader`` override), moving the model back to its device, and a report.
The official code replaces each decoder Linear weight in place with its quantize-dequantize result.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .compat import official_compat
from .prune import _c4_config_shim
from .repos import REPOS, head, load_module

WBITS = (2, 3, 4, 8)  # official llama.py --wbits choices without 16 (16 = no quantization)
DATASETS = ("wikitext2", "c4")  # official datautils.get_loaders (ptb not offered)
OFFICIAL_SEQLEN = 2048  # official llama.py get_llama: model.seqlen = 2048


def check_args(wbits: int, groupsize: int = -1, static_groups: bool = False, percdamp: float = 0.01,
               nsamples: int = 128, dataset: str = "c4") -> None:
    """Refuse settings the official driver or code cannot run."""
    if wbits not in WBITS:
        raise ValueError(f"wbits: expected one of {WBITS} (official choices; 16 means no quantization), "
                         f"got {wbits!r}")
    if groupsize != -1 and groupsize < 1:
        raise ValueError(f"groupsize: expected -1 (whole row) or a positive integer, got {groupsize!r}")
    if static_groups and groupsize == -1:
        raise ValueError("static_groups: needs a groupsize (official static groups are per group)")
    if not percdamp > 0:
        raise ValueError(f"percdamp: expected > 0, got {percdamp!r}")
    if nsamples < 1:
        raise ValueError(f"nsamples: expected >= 1, got {nsamples!r}")
    if dataset not in DATASETS:
        raise ValueError(f"dataset: expected one of {DATASETS} (official get_loaders), got {dataset!r}")


@contextlib.contextmanager
def _attribute(owner: Any, name: str, value: Any) -> Iterator[None]:
    """Set ``owner.name`` for the duration, then restore it (or remove it when it was absent)."""
    missing = object()
    saved = vars(owner).get(name, missing)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if saved is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, saved)


def _counted(method: Callable[..., Any], counter: dict) -> Callable[..., Any]:
    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        counter["calls"] += 1
        return method(self, *args, **kwargs)

    return counting


def _max_distinct(weight: torch.Tensor, groupsize: int) -> int:
    """Most distinct values in one row of one contiguous column group (whole row when groupsize == -1)."""
    width = weight.shape[1] if groupsize == -1 else groupsize
    most = 0
    for start in range(0, weight.shape[1], width):
        values = weight[:, start:start + width].float().sort(dim=1).values
        distinct = (values[:, 1:] != values[:, :-1]).sum(dim=1) + 1
        most = max(most, int(distinct.max()))
    return most


def quantize(model: nn.Module, tokenizer: Any, wbits: int, groupsize: int = -1, sym: bool = False,
             act_order: bool = False, static_groups: bool = False, percdamp: float = 0.01,
             true_sequential: bool = True, nsamples: int = 128, seed: int = 0, seqlen: int | None = None,
             dataset: str = "c4", device: str | torch.device | None = None,
             loader: Sequence | None = None) -> dict:
    """Quantize every decoder Linear of ``model`` in place with official ``llama_sequential``; report it.

    ``model.seqlen`` is the official 2048 (``get_llama``) unless ``seqlen`` is given (2 <= seqlen <=
    max_position_embeddings), recorded as a parameter mapping in ``deviations``. Data: official
    ``datautils.get_loaders(dataset)``, which loads its own tokenizer by model name (``use_fast=False``),
    so ``tokenizer`` is not used; or ``loader``: ``nsamples`` batches ``(input_ids[1, seqlen], target)``.
    """
    check_args(wbits, groupsize, static_groups, percdamp, nsamples, dataset)
    limit = int(model.config.max_position_embeddings)
    if seqlen is None and OFFICIAL_SEQLEN > limit:
        raise ValueError(f"seqlen: the official {OFFICIAL_SEQLEN} exceeds max_position_embeddings ({limit}); "
                         "pass seqlen")
    if seqlen is not None and not 2 <= seqlen <= limit:
        raise ValueError(f"seqlen: expected 2 <= seqlen <= max_position_embeddings ({limit}), got {seqlen!r}")
    llama = load_module("gptq", "llama")
    datautils = load_module("gptq", "datautils")
    home = next(model.parameters()).device
    device = torch.device(device) if device is not None else home
    deviations: list[str | dict] = [
        "args: module-global llama.args = argparse.Namespace with official llama.py's fields "
        "(llama_sequential reads it)"]
    if seqlen is not None:
        feasible = "feasible" if OFFICIAL_SEQLEN <= limit else "infeasible"
        deviations.append({"param": "seqlen", "official": OFFICIAL_SEQLEN, "used": seqlen,
                           "reason": f"official default {feasible} for this model "
                                     f"(max_position_embeddings {limit}); paper setting"})
    seqlen = OFFICIAL_SEQLEN if seqlen is None else seqlen
    args = argparse.Namespace(
        model=getattr(model.config, "_name_or_path", ""), dataset=dataset, seed=seed, nsamples=nsamples,
        percdamp=percdamp, nearest=False, wbits=wbits, groupsize=groupsize, sym=sym, save="", new_eval=False,
        act_order=act_order, true_sequential=true_sequential, static_groups=static_groups,
    )
    if loader is not None:
        batches = list(loader)
        if len(batches) != nsamples:
            raise ValueError(f"loader: expected {nsamples} batches (nsamples), got {len(batches)}")
        for batch in batches:
            if tuple(batch[0].shape) != (1, seqlen):
                raise ValueError(f"loader: batch[0] must have shape (1, {seqlen}), "
                                 f"got {tuple(batch[0].shape)}")
        data = {"source": "loader override", "batches": nsamples}
        deviations.append("data: loader override in place of official datautils.get_loaders")
    else:
        import datasets

        with _attribute(datasets, "load_dataset", _c4_config_shim(datasets.load_dataset, deviations)):
            batches = datautils.get_loaders(dataset, nsamples=nsamples, seed=seed, model=args.model,
                                            seqlen=seqlen)[0]
        data = {"source": f"official datautils.get_loaders({dataset!r})", "dataset": dataset,
                "tokenizer": f"AutoTokenizer.from_pretrained({args.model!r}, use_fast=False) (official)"}
    solves, layer_calls = {"calls": 0}, {"total": 0, "with_position_ids": 0}

    def official_call(module: nn.Module, call_args: tuple, kwargs: dict) -> None:
        if "position_embeddings" not in kwargs:  # the official code's own call, not the model's
            layer_calls["total"] += 1
            layer_calls["with_position_ids"] += kwargs.get("position_ids") is not None

    hooks = [layer.register_forward_pre_hook(official_call, with_kwargs=True) for layer in model.model.layers]
    started = time.perf_counter()
    try:
        with (official_compat(model, seqlen) as applied, _attribute(llama, "args", args),
              _attribute(llama.GPTQ, "fasterquant", _counted(llama.GPTQ.fasterquant, solves))):
            quantizers = llama.llama_sequential(model, batches, device)
    finally:
        for hook in hooks:
            hook.remove()
        model.to(home)
    wall_time = time.perf_counter() - started
    deviations.append(f"device: official llama_sequential leaves decoder layers, embed_tokens and norm on "
                      f"the CPU; model moved back to {home}")
    layers = []
    for name, quantizer in quantizers.items():
        weight = model.get_submodule(name).weight.data
        layers.append({"name": name, "scale_shape": list(quantizer.scale.shape),
                       "zero_shape": list(quantizer.zero.shape), "maxq": int(quantizer.maxq),
                       "max_distinct": _max_distinct(weight, groupsize)})
    contiguous = not (act_order and groupsize != -1 and not static_groups)
    repo = REPOS["gptq"]
    return {
        "official": {"repo": "gptq", "url": repo.url, "commit": repo.commit, "license": repo.license,
                     "function": "llama.llama_sequential"},
        "official_commit": head(Path(llama.__file__).resolve().parent),
        "args": vars(args), "device": str(device), "seqlen": seqlen, "data": data,
        "deviations": [*applied, *deviations],
        "fasterquant_calls": solves["calls"], "official_layer_calls": layer_calls,
        "levels": 2 ** wbits,
        # act-order without static groups quantizes groups in Hessian order, not contiguous column groups
        "groups": "contiguous" if contiguous else "act-order order (contiguous max_distinct not bounded)",
        "layers": layers,
        "wall_time_s": wall_time,
    }
