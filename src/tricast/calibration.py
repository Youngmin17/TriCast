"""Fit transforms, activation observers, and GPTQ weights from calibration tokens."""

from __future__ import annotations

import hashlib
import random
import weakref
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .nn import EmuLinear, iter_emulinear, patch_model
from .recipe import Recipe, load_recipe
from .transforms import fit_transform_group

DEFAULT_HESSIAN_MAX_BYTES = 4 * 1024**3


def _dataset_texts(dataset: str, split: str):
    from datasets import load_dataset

    if dataset == "wikitext2":
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset == "c4":
        if split not in ("train", "validation"):
            raise ValueError("c4 calibration split must be train or validation")
        shards = "01024" if split == "train" else "00008"
        # Arrow-backed random access keeps the corpus off the Python heap.
        return load_dataset("allenai/c4", "en", split=split, keep_in_memory=False,
                            data_files={split: f"en/c4-{split}.00000-of-{shards}.json.gz"})
    elif dataset == "pile":
        data = load_dataset("NeelNanda/pile-10k", split=split)
    else:
        raise ValueError("calibration.dataset must be wikitext2, c4, or pile")
    return (row["text"] for row in data)


def _windows(
    tokenizer, texts, input_ids, samples: int, seqlen: int, seed: int, *, document_local: bool = False,
) -> list[torch.Tensor]:
    """C4 draws documents uniformly with replacement, exactly as GPTQ get_c4.

    Eligible documents have more than ``seqlen`` tokens; the final token is held
    out, matching GPTQ's inclusive randint(0, length - seqlen - 1) bounds.
    Only documents consumed before obtaining ``samples`` windows are tokenized.
    """
    if samples < 1 or seqlen < 1:
        raise ValueError("calibration samples and seqlen must be positive")
    rng = random.Random(seed)
    if input_ids is not None:
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2:
            raise ValueError("calibration input_ids must be 1-D or 2-D")
        if ids.shape[1] < seqlen or not ids.shape[0]:
            raise ValueError("calibration input_ids are shorter than seqlen")
        windows = []
        for _ in range(samples):
            row = rng.randrange(ids.shape[0])
            start = rng.randrange(ids.shape[1] - seqlen + 1)
            windows.append(ids[row:row + 1, start:start + seqlen])
        return windows
    if tokenizer is None:
        raise ValueError("a tokenizer is required for calibration texts")
    if isinstance(texts, str):
        texts = [texts]
    if document_local:
        if not len(texts):
            raise ValueError("C4 calibration dataset is empty")
        windows = []
        too_short = set()
        while len(windows) < samples:
            index = rng.randrange(len(texts))
            document = texts[index]
            text = document["text"] if isinstance(document, dict) else document
            encoded = tokenizer(text, return_tensors="pt")
            ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError("C4 tokenizer must return one document as a 2-D input_ids tensor")
            if ids.shape[1] <= seqlen:
                too_short.add(index)
                if len(too_short) == len(texts):
                    raise ValueError(f"C4 has no documents longer than seqlen={seqlen}")
                continue
            start = rng.randrange(ids.shape[1] - seqlen)
            windows.append(ids[:, start:start + seqlen].clone())
        return windows
    encoded = tokenizer("\n\n".join(texts), return_tensors="pt")
    ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    return _windows(None, None, ids, samples, seqlen, seed)


class _InputGroups:
    """Record same-input calls without keeping calibration tensors alive."""

    def __init__(self, layers: list[tuple[str, EmuLinear]]) -> None:
        self.signatures: dict[str, list[tuple[int, int]]] = {name: [] for name, _ in layers}
        self.seen: dict[tuple, tuple[weakref.ReferenceType, int]] = {}
        self.batch = 0
        self.next_id = 0
        self.handles = [layer.register_forward_pre_hook(self._hook(name), with_kwargs=True)
                        for name, layer in layers]

    def _hook(self, name: str):
        def capture(module: nn.Module, args: tuple, kwargs: dict) -> None:
            x = args[0] if args else kwargs["input"]
            try:
                version = x._version
            except RuntimeError:  # Inference tensors cannot prove absence of in-place mutations.
                version = object()
            key = (x.device, x.dtype, x.data_ptr(), tuple(x.shape), tuple(x.stride()), version)
            previous = self.seen.get(key)
            if previous is None or previous[0]() is None:
                anchor = x
                while anchor._base is not None:
                    anchor = anchor._base
                previous = (weakref.ref(anchor), self.next_id)
                self.next_id += 1
                self.seen[key] = previous
            self.signatures[name].append((self.batch, previous[1]))
        return capture

    def next_batch(self) -> None:
        self.seen.clear()
        self.batch += 1

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.seen.clear()

    def groups(self, layers: list[tuple[str, EmuLinear]]) -> list[list[tuple[str, EmuLinear]]]:
        groups: dict[tuple, list[tuple[str, EmuLinear]]] = {}
        for name, layer in layers:
            spec = layer.spec.transform
            shared = spec.share_inputs and spec.kind in ("smoothquant", "awq")
            key = (spec, tuple(self.signatures[name])) if shared and self.signatures[name] else (name,)
            groups.setdefault(key, []).append((name, layer))
        return list(groups.values())

    def hessian_groups(
        self, layers: list[tuple[str, EmuLinear]],
    ) -> list[list[tuple[str, EmuLinear]]]:
        groups: dict[tuple, list[tuple[str, EmuLinear]]] = {}
        for name, layer in layers:
            if layer.spec.weight_algo.kind != "gptq":
                continue
            spec = layer.spec.transform
            separate = spec.kind in ("smoothquant", "awq") and not spec.share_inputs
            signature = tuple(self.signatures[name])
            key = (name,) if separate or not signature else (spec, signature)
            groups.setdefault(key, []).append((name, layer))
        return list(groups.values())


def _stages(
    model: nn.Module, layers: list[tuple[str, EmuLinear]], sequential: bool,
) -> tuple[list[list[tuple[str, EmuLinear]]], str, str | None]:
    if not sequential:
        return [layers], "nonsequential", None
    decoder = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(decoder, nn.ModuleList) or not len(decoder):
        return [layers], "nonsequential", "model.model.layers is not a nonempty decoder ModuleList"
    stages = []
    remaining = dict(layers)
    for block in decoder:
        ids = {id(module) for module in block.modules()}
        stage = [(name, layer) for name, layer in layers if id(layer) in ids]
        if stage:
            stages.append(stage)
            for name, _ in stage:
                remaining.pop(name, None)
    if remaining:
        head = getattr(model, "lm_head", None)
        if any(layer is not head for layer in remaining.values()):
            return [layers], "nonsequential", "selected non-decoder linears have unknown execution order"
        stages.append(list(remaining.items()))
    return stages, "sequential", None


def _tensor_tree(value: Any, device: torch.device | str, memo: dict[int, Any]) -> Any:
    """Copy replay inputs, preserving shared masks/position embeddings within a sample."""
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = (value, value.detach().to(device=device, copy=True))
        return memo[id(value)][1]
    if isinstance(value, (tuple, list)):
        return type(value)(_tensor_tree(item, device, memo) for item in value)
    if isinstance(value, dict):
        return {key: _tensor_tree(item, device, memo) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float, torch.dtype, torch.device)):
        return value
    raise TypeError(f"unsupported replay input {type(value).__name__}")


class _DecoderInputs:
    """Capture only block-zero states plus kwargs; verify the decoder is a pure chain."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        self.blocks = blocks
        self.samples: list[tuple[torch.Tensor, list[tuple[bool, tuple, dict]]]] = []
        self.reason: str | None = None
        self.handles = []
        for index, block in enumerate(blocks):
            self.handles.append(block.register_forward_pre_hook(self._before(index), with_kwargs=True))
            self.handles.append(block.register_forward_hook(self._after(index)))

    def next_batch(self) -> None:
        self.calls: list[tuple[bool, tuple, dict]] = []
        self.hidden: torch.Tensor | None = None
        self.previous: torch.Tensor | None = None
        self.memo: dict[int, Any] = {}

    def _before(self, index: int) -> Callable:
        def capture(module: nn.Module, args: tuple, kwargs: dict) -> None:
            if self.reason is not None:
                return
            try:
                positional = bool(args) and isinstance(args[0], torch.Tensor)
                hidden = args[0] if positional else kwargs.get("hidden_states")
                if not isinstance(hidden, torch.Tensor) or index != len(self.calls):
                    raise TypeError("decoder blocks must run once in order with tensor hidden_states")
                if index and (self.previous is None or not torch.equal(hidden, self.previous)):
                    raise TypeError("hidden_states are modified outside decoder blocks")
                if not index:
                    self.hidden = hidden.detach().cpu().clone()
                rest = args[1:] if positional else args
                keywords = kwargs if positional else {k: v for k, v in kwargs.items() if k != "hidden_states"}
                rest, keywords = _tensor_tree((rest, keywords), "cpu", self.memo)
                self.calls.append((positional, rest, keywords))
            except TypeError as error:
                self.reason = str(error)
        return capture

    def _after(self, index: int) -> Callable:
        def capture(module: nn.Module, args: tuple, output: Any) -> None:
            if self.reason is not None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                self.reason = "decoder output is not a tensor or a tuple beginning with hidden_states"
            else:
                self.previous = hidden.detach().clone()
        return capture

    def finish_batch(self) -> None:
        if self.reason is None:
            if self.hidden is None or len(self.calls) != len(self.blocks):
                self.reason = "not every decoder block was invoked exactly once"
            else:
                self.samples.append((self.hidden, self.calls))
        self.hidden, self.previous = None, None
        self.memo.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def run(
        self, index: int, hidden: list[torch.Tensor], device: torch.device,
        tracker: _InputGroups | None = None,
    ) -> list[torch.Tensor]:
        outputs = []
        for state, (_, calls) in zip(hidden, self.samples, strict=True):
            if tracker is not None:
                tracker.next_batch()
            positional, args, kwargs = calls[index]
            args, kwargs = _tensor_tree((args, kwargs), device, {})
            state = state.to(device=device, copy=True)
            if positional:
                output = self.blocks[index](state, *args, **kwargs)
            else:
                output = self.blocks[index](*args, hidden_states=state, **kwargs)
            output = output[0] if isinstance(output, tuple) else output
            outputs.append(output.detach().cpu())
        return outputs


def _hessian_plan(
    stages: list[list[tuple[str, EmuLinear]]], tracker: _InputGroups, budget: int,
) -> tuple[list[list[list[tuple[str, EmuLinear]]]], int, int]:
    groups = [tracker.hessian_groups(stage) for stage in stages]
    required = max((sum(group[0][1].in_features ** 2 * 8 for group in stage) for stage in groups), default=0)
    unshared = max((sum(layer.in_features ** 2 * 8 for _, layer in stage
                        if layer.spec.weight_algo.kind == "gptq") for stage in stages), default=0)
    if required > budget:
        raise ValueError(
            f"GPTQ Hessian storage requires {required} bytes after input sharing, exceeding "
            f"hessian_max_bytes={budget} (default {DEFAULT_HESSIAN_MAX_BYTES} bytes / 4 GiB); "
            "use sequential=True or explicitly raise hessian_max_bytes. No Hessians were allocated."
        )
    return groups, required, unshared


def _fit_stage(
    stage: list[tuple[str, EmuLinear]], hessian_groups: list[list[tuple[str, EmuLinear]]],
    run: Callable[[_InputGroups | None], None],
) -> list[list[str]]:
    from .nn.linear import HessianAccumulator

    shared = {}
    for group in hessian_groups:
        accumulator = HessianAccumulator()
        for index, (name, _) in enumerate(group):
            shared[name] = (accumulator, index == 0)
    for name, layer in stage:
        if name in shared:
            accumulator, owner = shared[name]
            layer.begin_calibration(hessian_accumulator=accumulator, hessian_owner=owner)
        else:
            layer.begin_calibration()
    tracker = _InputGroups(stage)
    try:
        run(tracker)
        groups = tracker.groups(stage)
        for group in hessian_groups:
            signatures = [tracker.signatures[name] for name, _ in group]
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise ValueError(
                    "GPTQ input-sharing pattern changed after preflight; Hessians were not fitted"
                )
    finally:
        tracker.close()
    for group in groups:
        transform = None
        if len(group) > 1:
            members = [layer for _, layer in group]
            transform = fit_transform_group(
                members[0].spec.transform, [layer.weight.detach() for layer in members],
                [layer._stats.result() for layer in members],
                weight_specs=[layer.spec.weight for layer in members],
                act_specs=[layer.spec.activation for layer in members],
            )
        for _, layer in group:
            layer.prepare_calibration(transform=transform)
    if any(layer._hessian_replay for _, layer in stage):
        replay_tracker = _InputGroups(stage)
        try:
            run(replay_tracker)
            for group in hessian_groups:
                signatures = [replay_tracker.signatures[name] for name, _ in group]
                if any(signature != signatures[0] for signature in signatures[1:]):
                    raise ValueError("GPTQ input-sharing pattern changed during transformed replay")
        finally:
            replay_tracker.close()
    for _, layer in stage:
        layer.finish_calibration()
        layer._calibration_passthrough = False
    return [[name for name, _ in group] for group in groups]


@torch.no_grad()
def calibrate(
    model: nn.Module, recipe: Recipe, tokenizer=None, *, texts=None, input_ids=None,
    samples: int | None = None, seqlen: int | None = None, seed: int | None = None, device=None,
    sequential: bool | None = None, hessian_max_bytes: int = DEFAULT_HESSIAN_MAX_BYTES,
) -> dict:
    """Fit transforms/observers/GPTQ with a 4-GiB default live-Hessian storage budget.

    The budget covers shared fp64 K-by-K accumulators, not GPTQ solver workspace.
    A passthrough preflight discovers input sharing before allocating any Hessian.
    Sequential decoder calibration caches block-zero inputs on CPU and replays
    blocks; unsupported chains explicitly report why full-model replay was used.
    C4 uses memory-mapped Arrow documents and exactly GPTQ's seeded document
    draws with replacement/window bounds, tokenizing one selected document at a time.
    """
    if isinstance(hessian_max_bytes, bool) or not isinstance(hessian_max_bytes, int) or hessian_max_bytes < 0:
        raise ValueError("hessian_max_bytes must be a nonnegative integer")
    recipe = load_recipe(recipe)
    config = recipe.calibration_options
    for key, value in (("samples", samples), ("seqlen", seqlen), ("seed", seed), ("sequential", sequential)):
        if value is not None:
            config[key] = value
    samples, seqlen, seed = config["samples"], config["seqlen"], config["seed"]
    if texts is not None and input_ids is not None:
        raise ValueError("pass either texts or input_ids, not both")
    source = "input_ids" if input_ids is not None else "texts" if texts is not None else "dataset"
    document_local = source == "dataset" and config["dataset"] == "c4"
    if source == "dataset":
        texts = _dataset_texts(config["dataset"], config["split"])
    else:
        config["dataset"], config["split"] = None, None
    windows = _windows(tokenizer, texts, input_ids, samples, seqlen, seed, document_local=document_local)
    target = torch.device(device) if device is not None else next(model.parameters()).device
    devices = [target.index if target.index is not None else torch.cuda.current_device()] \
        if target.type == "cuda" else []
    full_model_forwards = 0
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if target.type == "cuda":
            with torch.cuda.device(target):
                torch.cuda.manual_seed(seed)
        if device is not None:
            model.to(device)
        device = next(model.parameters()).device
        if not list(iter_emulinear(model)):
            patch_model(model, recipe)
        layers = list(iter_emulinear(model))
        if not layers:
            raise ValueError("the recipe selects no Linear modules for calibration")
        stages, mode, reason = _stages(model, layers, config["sequential"])
        training = [(module, module.training) for module in model.modules()]
        state_fields = ("mode", "observer", "transform", "_fitted", "_weight_operand", "_outlier_operand",
                        "_weight_state", "_weight_noise", "_calibrated")
        states = [(layer, {key: getattr(layer, key) for key in state_fields}) for _, layer in layers]
        group_names = []
        model.eval()
        capture = None
        model_config = getattr(model, "config", None)
        old_use_cache = getattr(model_config, "use_cache", None)
        if old_use_cache is not None:
            model_config.use_cache = False

        def run_model(tracker: _InputGroups | None, batch_windows: list[torch.Tensor] = windows) -> None:
            nonlocal full_model_forwards
            for ids in batch_windows:
                if tracker is not None:
                    tracker.next_batch()
                full_model_forwards += 1
                model(input_ids=ids.to(device))

        try:
            for _, layer in layers:
                layer._calibration_passthrough = True
            if mode == "sequential":
                blocks = model.model.layers
                block_ids = {id(module) for block in blocks for module in block.modules()}
                supported = {
                    "transformers.models.llama.modeling_llama",
                    "transformers.models.qwen3.modeling_qwen3",
                }
                if (type(model).__module__ not in supported
                        or any(type(block).__module__ not in supported for block in blocks)):
                    reason = ("cached replay supports Llama/Qwen3 decoder forwards; "
                              "custom kwargs require full replay")
                elif any(id(layer) not in block_ids for _, layer in layers):
                    reason = "cached replay cannot capture the selected lm_head's post-decoder operations"
                elif any(layer.observer is not None and layer.observer.spec.kind == "history"
                         or layer.spec.activation is not None and layer.spec.activation.rounding.value == "sr"
                         for _, layer in layers):
                    reason = "online history or stochastic activations require full-model sequential replay"
                else:
                    capture = _DecoderInputs(blocks)
            preflight = _InputGroups(layers)
            try:
                if capture is not None:
                    for ids in windows:
                        preflight.next_batch()
                        capture.next_batch()
                        full_model_forwards += 1
                        model(input_ids=ids.to(device))
                        capture.finish_batch()
                    if capture.reason is not None:
                        reason = "cached decoder replay unavailable: " + capture.reason
                        capture.close()
                        capture = None
                elif any(layer.spec.weight_algo.kind == "gptq" for _, layer in layers):
                    run_model(preflight, windows[:1])
                hessian_groups, hessian_bytes, unshared_bytes = _hessian_plan(
                    stages, preflight, hessian_max_bytes,
                )
            finally:
                preflight.close()
                if capture is not None:
                    capture.close()
            if capture is None:
                for stage, shared in zip(stages, hessian_groups, strict=True):
                    group_names.extend(_fit_stage(stage, shared, run_model))
            else:
                hidden = [state for state, _ in capture.samples]
                # Only one layer of hidden states is retained, independent of decoder depth.
                capture.samples = [(torch.empty(0), calls) for _, calls in capture.samples]
                stages_by_layer = {id(layer): index for index, stage in enumerate(stages)
                                   for _, layer in stage}
                for index, block in enumerate(capture.blocks):
                    stage_index = next((stages_by_layer[id(module)] for module in block.modules()
                                        if id(module) in stages_by_layer), None)
                    if stage_index is not None:
                        def run_block(
                            tracker: _InputGroups | None, block_index: int = index,
                            states: list[torch.Tensor] = hidden,
                        ) -> None:
                            capture.run(block_index, states, device, tracker)
                        group_names.extend(_fit_stage(
                            stages[stage_index], hessian_groups[stage_index], run_block,
                        ))
                    if index + 1 < len(capture.blocks):
                        hidden = capture.run(index, hidden, device)
        except Exception:
            for layer, state in states:
                for key, value in state.items():
                    setattr(layer, key, value)
            raise
        finally:
            if capture is not None:
                capture.close()
            if old_use_cache is not None:
                model_config.use_cache = old_use_cache
            for _, layer in layers:
                layer._calibration_passthrough = False
                layer.clear_calibration()
            for module, was_training in training:
                module.training = was_training
    fingerprint = hashlib.sha256()
    for ids in windows:
        fingerprint.update(ids.cpu().contiguous().numpy().astype("<i8").tobytes())
    return {**config, "samples": len(windows), "source": source,
            "n_tokens": sum(ids.numel() for ids in windows), "mode": mode, "reason": reason,
            "config": config, "groups": group_names, "sequential_cached": capture is not None,
            "hessian_max_bytes": hessian_max_bytes, "hessian_bytes": hessian_bytes,
            "hessian_unshared_bytes": unshared_bytes,
            "hessian_groups": [[name for name, _ in group] for stage in hessian_groups for group in stage],
            "full_model_forwards": full_model_forwards,
            "dataset_fingerprint": fingerprint.hexdigest(), "layers": [name for name, _ in layers]}
