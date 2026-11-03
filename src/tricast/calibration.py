"""Fit transforms, activation observers, and GPTQ weights from calibration tokens."""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import random
import sys
import tempfile
import weakref
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.modules import module as torch_module

from .formats import FloatFormat, IntFormat, Pow2Format
from .mma.operand import Operand, tensor_state
from .nn import EmuLinear, iter_emuattention, iter_emuconv2d, iter_emulinear, patch_model
from .nn.attention import _InstanceForward
from .quant.observer import ObserverState
from .quant.spec import ObserverSpec
from .recipe import Recipe, _json_value, load_recipe
from .transforms import LinearTransform, fit_transform_group

DEFAULT_HESSIAN_MAX_BYTES = 4 * 1024**3
CACHE_FORMAT = 3  # bump when the stored layer state or the key layout changes
_FORMATS = {"float": FloatFormat, "int": IntFormat, "pow2": Pow2Format}


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


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy()).hexdigest()


@functools.lru_cache(maxsize=1)
def _source_sha256() -> str:
    """Every tricast .py file (relative path and content hash, sorted by path), read once per process."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()):
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{path.relative_to(root).as_posix()}\0{content}\n".encode())
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime(device: torch.device) -> dict:
    """Library and kernel settings that can change the bits of a calibration forward."""
    runtime = {"torch": torch.__version__,
               "transformers": getattr(sys.modules.get("transformers"), "__version__", None),
               "triton": _package_version("triton"), "default_dtype": str(torch.get_default_dtype()),
               "device": device.type, "threads": torch.get_num_threads(),
               "mkl_cbwr": os.environ.get("MKL_CBWR"), "precision": torch.get_float32_matmul_precision(),
               "deterministic": torch.are_deterministic_algorithms_enabled(),
               "mkldnn": torch.backends.mkldnn.enabled,
               "cudnn": [torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                         torch.backends.cudnn.deterministic]}
    if device.type == "cuda":
        backends = torch.backends.cuda
        runtime.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                       capability=list(torch.cuda.get_device_capability(device)),
                       tf32=backends.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                       reduced=[backends.matmul.allow_bf16_reduced_precision_reduction,
                                backends.matmul.allow_fp16_reduced_precision_reduction],
                       sdpa=[backends.flash_sdp_enabled(), backends.mem_efficient_sdp_enabled(),
                             backends.math_sdp_enabled()])
    return runtime


def _execution_context() -> str | None:
    """Why the caller's grad-mode context changes calibration (None: it does not). Autocast changes the
    passthrough arithmetic; inference tensors carry no version counter, so input sharing is not seen."""
    for device_type in ("cpu", "cuda"):
        if torch.is_autocast_enabled(device_type):
            return f"{device_type} autocast ({torch.get_autocast_dtype(device_type)}) is active"
    if torch.is_inference_mode_enabled():
        return "torch.inference_mode is active (inference tensors hide input sharing)"
    return None


def _plain_attributes(module: nn.Module) -> list:
    """[name, repr] of the module's public attributes that hold plain values (numbers, strings, None, and
    tuples or lists of them), sorted; ``training`` is left out (calibrate runs in eval mode)."""
    def plain(value: object) -> bool:
        if isinstance(value, (tuple, list)):
            return all(plain(item) for item in value)
        return value is None or isinstance(value, (bool, int, float, str))

    return sorted([key, repr(value)] for key, value in vars(module).items()
                  if not key.startswith("_") and key != "training" and plain(value))


def _structure(model: nn.Module) -> tuple[list, str | None]:
    """Each module's name, class, extra_repr and public plain attributes (``scaling``, ``head_dim``, ...:
    values a forward reads that config does not pin once the module exists), plus its source file's sha256
    for a class outside torch, transformers and tricast, and why the forward may still differ from that
    description (None: it cannot). A passthrough EmuLinear is described as the Linear it wraps, so patching
    alone keeps the key."""
    kv = getattr(model, "_tricast_kv_patch", None)
    own = set() if kv is None else {kv.forward_hook.id, kv.clear_hook.id}  # the KV patch's mask hooks
    reasons = []
    if torch_module._global_forward_hooks or torch_module._global_forward_pre_hooks:
        reasons.append("global module forward hooks are registered")
    if getattr(model, "config", None) is None:
        reasons.append("the model has no config")
    sources: dict[str, str] = {}
    entries = []
    for name, module in model.named_modules():
        label = name or "model"
        shown = module._original if isinstance(module, EmuLinear) else module
        cls = type(shown)
        entry = [name, f"{cls.__module__}.{cls.__qualname__}", shown.extra_repr(), _plain_attributes(shown)]
        if cls.__module__.split(".")[0] not in ("torch", "transformers", "tricast"):
            try:
                path = inspect.getsourcefile(cls)
            except TypeError:
                path = None
            if path is None or not os.path.isfile(path):
                reasons.append(f"{label}: class {cls.__qualname__} has no source file to fingerprint")
            else:
                if path not in sources:
                    sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                entry.append(sources[path])
        entries.append(entry)
        hooks = set(module._forward_hooks) | set(module._forward_pre_hooks)
        if hooks - (own if module is model else set()):
            reasons.append(f"{label} has forward hooks")
        forward = module.__dict__.get("forward")
        if forward is not None and not isinstance(forward, _InstanceForward):
            reasons.append(f"{label} has an instance-level forward")
        if isinstance(module, EmuLinear) and type(module._original) is not nn.Linear:
            reasons.append(f"{label} wraps a Linear subclass whose forward the passthrough replaces")
    return entries, reasons[0] if reasons else None


def _content_sha256(value: Any) -> str:
    """sha256 over the non-tensor values and every tensor's dtype, shape and bytes, in sorted-key order."""
    def skeleton(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return ["tensor", str(item.dtype), list(item.shape), _tensor_sha256(item)]
        if isinstance(item, dict):
            return {key: skeleton(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [skeleton(entry) for entry in item]
        return item
    return _sha256_json(skeleton(value))


def _read_inputs(path: Path) -> tuple[dict | None, str]:
    """The input-sharing record if its embedded digest matches its classes; else None and why."""
    if not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text())
        classes = data["classes"]
        valid = isinstance(classes, dict) and data["digest"] == _sha256_json(classes)
    except (ValueError, KeyError, TypeError):
        valid = False
    return (classes, "valid") if valid else (None, "invalid")


def _uncacheable(layer: EmuLinear, mode: str) -> str | None:
    """Why a layer's calibrated state depends on more than its cache key (``None``: it does not)."""
    spec = layer.spec
    activation = spec.activation
    if mode == "sequential":
        return "sequential calibration feeds each block the outputs of earlier quantized blocks"
    if spec.transform.kind in ("smoothquant", "awq") and spec.weight_algo.kind == "gptq":
        return "fitted-diagonal GPTQ accumulates its Hessian in a transformed replay (_hessian_replay)"
    if activation is not None and activation.observer is not None and activation.observer.kind == "history":
        return "a history observer keeps online delayed-scaling state"
    if activation is not None and activation.rounding.value == "sr":
        return "stochastic activation rounding"
    if spec.weight is not None and spec.weight.rounding.value == "sr":
        return "stochastic weight rounding draws noise from the seeded RNG stream in layer order"
    return None


def _input_classes(signatures: dict[str, list[tuple[int, int]]]) -> dict[str, str | None]:
    """Label each module by the first module that read the same input in every batch (None: never ran)."""
    first: dict[tuple, str] = {}
    return {name: first.setdefault(tuple(signature), name) if signature else None
            for name, signature in signatures.items()}


def _cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().cpu()


def _to(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(device)


def _format_data(fmt) -> dict | None:
    return None if fmt is None else {"kind": fmt.kind, **{f.name: getattr(fmt, f.name) for f in fields(fmt)}}


def _format(data: dict | None):
    if data is None:
        return None
    data = dict(data)
    return _FORMATS[data.pop("kind")](**data)


def _operand_data(operand: Operand | None) -> dict | None:
    if operand is None:
        return None
    return {"values": _cpu(operand.values), "fmt": _format_data(operand.fmt), "scale": _cpu(operand.scale),
            "scale_fmt": _format_data(operand.scale_fmt), "scale_kind": operand.scale_kind,
            "k_domain": operand.k_domain, "alpha": _cpu(operand.alpha)}


def _operand(data: dict | None, device: torch.device) -> Operand | None:
    from .nn.linear import _pack

    if data is None:
        return None
    return _pack(Operand(_to(data["values"], device), _format(data["fmt"]), _to(data["scale"], device),
                         _format(data["scale_fmt"]), data["scale_kind"], data["k_domain"],
                         _to(data["alpha"], device)))


def _layer_state(layer: EmuLinear) -> dict:
    """The fields calibrate() snapshots, as tensors and plain values (``torch.load(weights_only=True)``)."""
    transform = layer.transform
    return {"mode": layer.mode, "fitted": layer._fitted, "calibrated": layer._calibrated,
            "transform": {"kind": transform.kind, "block": transform.block, "diag": _cpu(transform.diag)},
            "observer": None if layer.observer is None else layer.observer.state_dict(),
            "weight_operand": _operand_data(layer._weight_operand),
            "outlier_operand": _operand_data(layer._outlier_operand),
            "weight_noise": _cpu(layer._weight_noise)}


def _restore_layer(layer: EmuLinear, state: dict) -> None:
    # Operands, diagonals and noise live with the weight; observer statistics live on the CPU.
    device = layer.weight.device
    transform = state["transform"]
    layer.mode, layer._fitted, layer._calibrated = state["mode"], state["fitted"], state["calibrated"]
    layer.transform = LinearTransform(transform["kind"], transform["block"], _to(transform["diag"], device))
    layer.observer = None
    if state["observer"] is not None:
        layer.observer = ObserverState(ObserverSpec(**state["observer"]["spec"]))
        layer.observer.load_state_dict(state["observer"])
    layer._weight_operand = _operand(state["weight_operand"], device)
    layer._outlier_operand = _operand(state["outlier_operand"], device)
    layer._weight_noise = _to(state["weight_noise"], device)
    layer._weight_state = tensor_state(layer.weight)


def _write_atomic(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except BaseException:
        os.unlink(temporary)
        raise
    os.replace(temporary, path)


class _LayerCache:
    """Nonsequential per-layer calibration results, keyed by everything they are computed from.

    Every EmuLinear is a passthrough during nonsequential calibration, so a layer's inputs are
    those of the unquantized model (all parameters and buffers, module structure, config, KV and
    attention patches, runtime) on the windows, and its result depends on that, its own weight
    and spec and, for a jointly fitted shared-input transform, the other members' specs. The MMA
    spec is left out: calibration runs no GEMM. Which layers share an input is recorded, with a
    digest of its content, from every fitting pass, so later runs know transform groups before
    any forward; each shared-input entry keeps the digest of the record it was keyed under.
    Entries carry a content digest and are verified before use: an unreadable or mismatching
    entry is refitted and overwritten (``report()["corrupt"]``).
    """

    def __init__(
        self, directory: str | os.PathLike, model: nn.Module, layers: list[tuple[str, EmuLinear]],
        calibration: dict, mode: str, device: torch.device, context: str | None,
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layers = layers
        self.linears = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, (nn.Linear, EmuLinear))]
        tensors = {}
        for prefix, module in model.named_modules():
            owned = (*module.named_parameters(recurse=False), *module.named_buffers(recurse=False))
            for name, tensor in owned:
                if isinstance(module, EmuLinear) and name == "bias":
                    tensor = tensor.to(module.weight.dtype)  # EmuLinear keeps an fp32 copy of the bias
                tensors[f"{prefix}.{name}" if prefix else name] = [str(tensor.dtype), list(tensor.shape),
                                                                   _tensor_sha256(tensor)]
        config = getattr(model, "config", None)
        config_data = config.to_dict() if hasattr(config, "to_dict") else None
        if config_data is not None:
            config_data.pop("use_cache", None)  # calibration turns it off
        kv = getattr(model, "_tricast_kv_patch", None)
        structure, structural = _structure(model)
        self.source_sha256 = _source_sha256()
        self.base = _sha256_json({
            "format": CACHE_FORMAT, "source": self.source_sha256, "runtime": _runtime(device),
            "model": {"class": f"{type(model).__module__}.{type(model).__qualname__}",
                      "commit": getattr(config, "_commit_hash", None), "config": config_data,
                      "attn_implementation": getattr(config, "_attn_implementation", None),
                      "tensors": tensors, "structure": structure},
            "kv": None if kv is None else [_json_value(kv.kv), sorted(kv.indices), kv.names],
            "attention": [[name, _json_value(module._tricast_attention.spec),
                           module._tricast_attention.backend] for name, module in iter_emuattention(model)],
            "calibration": calibration,
        })
        self.entries = {}
        for name, layer in layers:
            spec = _json_value(layer.spec)
            del spec["mma"]
            self.entries[name] = {"name": name, "weight": tensors[f"{name}.weight"],
                                  "bias": tensors.get(f"{name}.bias"), "spec": spec, "backend": layer.backend}
        conv = next((name for name, _ in iter_emuconv2d(model)), None)
        # A model-wide reason makes every layer uncacheable; its forwards record no input sharing.
        self.context = context or structural or (
            None if conv is None else f"EmuConv2d {conv} quantizes upstream activations during calibration")
        self.reasons = {}
        for name, layer in layers:
            reason = self.context or _uncacheable(layer, mode)
            if reason is not None:
                self.reasons[name] = reason
        self.shared = {name for name, layer in layers  # transform groups depend on the input record
                       if layer.spec.transform.share_inputs
                       and layer.spec.transform.kind in ("smoothquant", "awq")}
        self.inputs_path = self.dir / f"inputs-{self.base}.json"
        self.inputs, self.inputs_status = _read_inputs(self.inputs_path)
        self.inputs_digest = None if self.inputs is None else _sha256_json(self.inputs)
        self.observed: dict[str, str | None] | None = None
        self.groups = {name: self._group(name, layer) for name, layer in layers}
        intrinsic = dict(self.reasons)
        for name, group in self.groups.items():
            partner = next((member for member in group or () if member in intrinsic and member != name), None)
            if name not in self.reasons and partner is not None:
                self.reasons[name] = f"its transform is fitted jointly with uncacheable {partner}"
        self.keys = {name: self._key(name, group) for name, group in self.groups.items()
                     if group is not None and name not in self.reasons}
        self.loaded: dict[str, dict] = {}
        self.corrupt: dict[str, str] = {}
        for name, key in self.keys.items():
            if self._path(key).exists():
                state, problem = self._read_entry(name)
                if problem is None:
                    self.loaded[name] = state
                else:
                    self.corrupt[name] = problem
        # A jointly fitted transform group is loaded whole or fitted whole.
        self.hits = [name for name, _ in layers
                     if name in self.loaded and all(member in self.loaded for member in self.groups[name])]
        self.hit_set = set(self.hits)
        self.loaded = {name: self.loaded[name] for name in self.hits}

    def _path(self, key: str) -> Path:
        return self.dir / f"layer-{key}.pt"

    def _group(self, name: str, layer: EmuLinear) -> list[str] | None:
        """The layer's transform-fitting group as _InputGroups.groups forms it; None before any record."""
        spec = layer.spec.transform
        if not (spec.share_inputs and spec.kind in ("smoothquant", "awq")):
            return [name]
        candidates = [other for other, module in self.layers if module.spec.transform == spec]
        if self.inputs is None or any(other not in self.inputs for other in candidates):
            return None
        label = self.inputs[name]
        return [other for other in candidates if label is not None and self.inputs[other] == label] or [name]

    def _key(self, name: str, group: list[str]) -> str:
        members = None if len(group) == 1 else [self.entries[member] for member in group]
        return _sha256_json({"base": self.base, "layer": self.entries[name], "group": members})

    def _read_entry(self, name: str) -> tuple[dict | None, str | None]:
        """A stored state checked against its content digest, key and input-sharing evidence; else why not."""
        path = self._path(self.keys[name])
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            payload = {field: data[field] for field in ("key", "name", "inputs", "state")}
            digest = data["digest"]
        except Exception as error:  # truncated, not a torch archive, or another layout: refit it
            return None, f"unreadable ({type(error).__name__})"
        if digest != _content_sha256(payload):
            return None, "content digest mismatch"
        if payload["key"] != self.keys[name] or payload["name"] != name:
            return None, "key or name mismatch"
        if payload["inputs"] != (self.inputs_digest if name in self.shared else None):
            return None, "stored under a different input-sharing record"
        return payload["state"], None

    def load(self) -> list[list[str]]:
        """Restore every hit; return their transform groups."""
        layers = dict(self.layers)
        for name in self.hits:
            _restore_layer(layers[name], self.loaded.pop(name))
        return [self.groups[name] for name in self.hits if self.groups[name][0] == name]

    def probed(self, model: nn.Module, run: Callable[[_InputGroups | None], None]) -> Callable:
        """``run`` that records, on its first pass, which Linear modules read the same input."""
        def first_pass(tracker: _InputGroups | None) -> None:
            if self.observed is not None or self.context is not None:
                return run(tracker)
            probe = _InputGroups(self.linears)
            handle = model.register_forward_pre_hook(lambda *_: probe.next_batch())
            try:
                run(tracker)
            finally:
                handle.remove()
                probe.close()
            self.observed = _input_classes(probe.signatures)
        return first_pass

    def store(self, fitted: list[list[str]]) -> None:
        evidence = None if self.observed is None else _sha256_json(self.observed)
        if self.observed is not None and self.inputs == self.observed:
            self.inputs_status = "verified"
        elif self.observed is not None:
            if any(name in self.shared for name in self.hits):
                raise RuntimeError(f"{self.inputs_path}: this run's input sharing contradicts the record its "
                                   "cached shared-input layers were loaded under; remove the cache directory")
            record = {"digest": evidence, "classes": self.observed}
            _write_atomic(self.inputs_path, json.dumps(record).encode("utf-8"))
            self.inputs_status = "written" if self.inputs_status == "missing" else "rewritten"
            for name in self.shared:  # groups predicted from a missing, invalid or contradicted record
                self.groups[name] = None
        actual = {name: group for group in fitted for name in group}
        for name, layer in self.layers:
            if name in self.hit_set or name in self.reasons:
                continue
            group = actual[name]
            partner = next((member for member in group if member in self.reasons), None)
            if partner is not None:
                self.reasons[name] = f"its transform is fitted jointly with uncacheable {partner}"
                continue
            if self.groups[name] is None:
                self.groups[name], self.keys[name] = group, self._key(name, group)
            elif self.groups[name] != group:
                raise RuntimeError(f"{name}: cached input sharing predicted transform group "
                                   f"{self.groups[name]}, but calibration fitted {group}")
            payload = {"key": self.keys[name], "name": name,
                       "inputs": evidence if name in self.shared else None, "state": _layer_state(layer)}
            buffer = io.BytesIO()
            torch.save({**payload, "digest": _content_sha256(payload)}, buffer)
            _write_atomic(self._path(self.keys[name]), buffer.getvalue())

    def report(self) -> dict:
        """``corrupt``: entries found but refused (and refitted); ``inputs``: the input-sharing record's
        state (missing / invalid / valid when no forward ran, else verified / written / rewritten)."""
        names = [name for name, _ in self.layers]
        return {"dir": str(self.dir), "hits": list(self.hits),
                "misses": [name for name in names if name not in self.hit_set and name not in self.reasons],
                "uncacheable": {name: self.reasons[name] for name in names if name in self.reasons},
                "corrupt": dict(self.corrupt), "inputs": self.inputs_status,
                "source_sha256": self.source_sha256}


@torch.no_grad()
def calibrate(
    model: nn.Module, recipe: Recipe, tokenizer=None, *, texts=None, input_ids=None,
    samples: int | None = None, seqlen: int | None = None, seed: int | None = None, device=None,
    sequential: bool | None = None, hessian_max_bytes: int = DEFAULT_HESSIAN_MAX_BYTES,
    cache_dir: str | os.PathLike | None = None,
) -> dict:
    """Fit transforms/observers/GPTQ with a 4-GiB default live-Hessian storage budget.

    The budget covers shared fp64 K-by-K accumulators, not GPTQ solver workspace.
    A passthrough preflight discovers input sharing before allocating any Hessian.
    Sequential decoder calibration caches block-zero inputs on CPU and replays
    blocks; unsupported chains explicitly report why full-model replay was used.
    C4 uses memory-mapped Arrow documents and exactly GPTQ's seeded document
    draws with replacement/window bounds, tokenizing one selected document at a time.
    ``cache_dir`` (opt-in) reuses bit-identical nonsequential per-layer results across
    calls (see _LayerCache): only layers without a stored result are fitted, with no
    forward when all hit; ``result["cache"]`` lists hits, misses and uncacheable layers.
    """
    context = None if cache_dir is None else _execution_context()  # the caller's, before any of ours
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
    fingerprint = hashlib.sha256()
    for ids in windows:
        fingerprint.update(ids.cpu().contiguous().numpy().astype("<i8").tobytes())
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
        cache, pending, cached_groups = None, layers, []
        if cache_dir is not None:
            cache = _LayerCache(cache_dir, model, layers, {
                "dataset": config["dataset"], "split": config["split"], "samples": len(windows),
                "seqlen": seqlen, "seed": seed, "source": source, "windows": fingerprint.hexdigest(),
                "mode": mode,
            }, mode, device, context)
            pending = [(name, layer) for name, layer in layers if name not in cache.hit_set]
            if cache.hits:  # only nonsequential layers hit; the single stage keeps the layers left to fit
                stages = [pending] if pending else []
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
            if cache is not None:
                cached_groups = cache.load()
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
            preflight = _InputGroups(pending)
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
                elif any(layer.spec.weight_algo.kind == "gptq" for _, layer in pending):
                    run_model(preflight, windows[:1])
                hessian_groups, hessian_bytes, unshared_bytes = _hessian_plan(
                    stages, preflight, hessian_max_bytes,
                )
            finally:
                preflight.close()
                if capture is not None:
                    capture.close()
            if capture is None:
                run = run_model if cache is None else cache.probed(model, run_model)
                for stage, shared in zip(stages, hessian_groups, strict=True):
                    group_names.extend(_fit_stage(stage, shared, run))
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
            if cache is not None:
                cache.store(group_names)
                # Groups ordered by first member, as one fitting pass over every layer lists them.
                order = {name: index for index, (name, _) in enumerate(layers)}
                group_names = sorted(group_names + cached_groups, key=lambda group: order[group[0]])
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
    result = {**config, "samples": len(windows), "source": source,
              "n_tokens": sum(ids.numel() for ids in windows), "mode": mode, "reason": reason,
              "config": config, "groups": group_names, "sequential_cached": capture is not None,
              "hessian_max_bytes": hessian_max_bytes, "hessian_bytes": hessian_bytes,
              "hessian_unshared_bytes": unshared_bytes,
              "hessian_groups": [[name for name, _ in group] for stage in hessian_groups for group in stage],
              "full_model_forwards": full_model_forwards,
              "dataset_fingerprint": fingerprint.hexdigest(), "layers": [name for name, _ in layers]}
    if cache is not None:
        result["cache"] = cache.report()
    return result
