"""Low-bit model packages: a recipe applied to a Hugging Face causal LM, written as integer codes with their
scales (safetensors) and read back for inference.

    python -m tricast.pack build --model Qwen/Qwen3-0.6B --revision REV --recipe recipe.yaml --out DIR [--ppl]
    python -m tricast.pack run --package DIR [--ppl] [--prompt TEXT …]

A package directory holds

- ``model.safetensors``: every tensor the recipe leaves alone (token embedding, norms, an untied output
  layer) in the checkpoint's dtype, and per quantized Linear ``<name>.codes`` (codes bit-packed into
  uint8), ``<name>.scale``, ``<name>.zero`` (bit-packed like the codes), ``<name>.diag`` (its input
  transform's diagonal) or ``<name>.absmax`` (NF4). A layer whose codes cannot be recovered keeps
  ``<name>.weight``: the BF16 weight TriCast multiplies with, at full size.
- ``quant_config.json``: the recipe (text and SHA-256) the layers were quantized with, the model id and
  revision, and per layer what is stored and how many bits it takes.
- the checkpoint's ``config.json``, ``generation_config.json`` and tokenizer files.
- ``manifest.json``: every file's size and SHA-256, and the checks run when the package was written.

Decoding resolves each layer's spec from the stored recipe exactly as ``patch_model`` does and rebuilds the
weight with TriCast's own dequantization, so a decoded weight is bit for bit the one TriCast's emulation
multiplies with, the sign of a zero aside (``build`` checks every layer). A layer with an input transform
(random Hadamard, AWQ) applies it to its input at run time, as TriCast does. What does not carry over is
TriCast's MMA arithmetic: the package runs with PyTorch's matmul (BF16 inputs, fp32 accumulation; fp32 after a
transform). Weights only: a recipe with activation quantization, a KV-cache spec, sparsity, outliers, an MMA
output format other than BF16 / FP16 / FP32, or one Linear under two names is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .formats import IntFormat
from .quant.qtensor import QTensor
from .transforms import LinearTransform

SCHEMA = "tricast.pack/1"
WEIGHTS = "model.safetensors"
CONFIG = "quant_config.json"
MANIFEST = "manifest.json"
# copied from the checkpoint snapshot when present
CHECKPOINT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


# ---------- bit packing ----------
def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Unsigned ``bits``-bit codes (any shape, values < 2**bits) as a flat uint8 stream, eight codes in
    ``bits`` bytes, little-endian, the last group padded with zeros."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be 1..8, got {bits}")
    flat = codes.reshape(-1).to(torch.int64)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= 1 << bits):
        raise ValueError(f"codes do not fit in {bits} bits")
    if bits == 8:
        return flat.to(torch.uint8)
    flat = F.pad(flat, (0, -flat.numel() % 8)).reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=flat.device) * bits
    word = (flat << shifts).sum(dim=1)
    out = torch.stack([(word >> (8 * j)) & 0xFF for j in range(bits)], dim=1)
    return out.reshape(-1).to(torch.uint8)


def unpack_bits(data: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """The first ``count`` codes of a ``pack_bits`` stream, as uint8."""
    if bits == 8:
        return data[:count].clone()
    groups = data.to(torch.int64).reshape(-1, bits)
    shifts = torch.arange(bits, dtype=torch.int64, device=data.device) * 8
    word = (groups << shifts).sum(dim=1)
    codes = torch.stack([(word >> (bits * i)) & ((1 << bits) - 1) for i in range(8)], dim=1)
    return codes.reshape(-1)[:count].to(torch.uint8)


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Two fp32 tensors with the same bit pattern in every element, the sign of a zero aside: a weight that
    rounds to -0.0 decodes to +0.0 (codes carry no sign of zero), which changes no product sum that starts
    from +0.0 — TriCast's and PyTorch's both do."""
    if a.shape != b.shape or a.dtype != torch.float32 or b.dtype != torch.float32:
        return False
    return torch.equal((a + 0.0).view(torch.int32), (b + 0.0).view(torch.int32))


# ---------- what a layer stores ----------
@dataclass
class Encoded:
    """One quantized Linear: how it is stored (``kind``), its tensors and the bits they take."""

    kind: str  # "affine" (integer codes, scales, zero points), "nf4", "dense" (BF16 weight)
    shape: tuple[int, int]
    tensors: dict[str, torch.Tensor]
    meta: dict[str, Any]


def _scale_tensor(scale: torch.Tensor) -> tuple[torch.Tensor, str]:
    """A scale in the narrowest float type that holds every value exactly."""
    for dtype, name in ((torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
        narrow = scale.to(dtype)
        if torch.equal(narrow.float(), scale.float()):
            return narrow, name
    return scale.float(), "float32"


def encode_qtensor(qt: QTensor) -> Encoded | None:
    """Integer codes of an int-format QTensor with group or row scales; None for anything else."""
    spec = qt.spec
    if not isinstance(spec.format, IntFormat) or qt.global_scale is not None or len(qt.shape) != 2:
        return None
    if spec.granularity not in ("group", "row") or qt.scale is None:
        return None
    if qt.zero_point is not None and spec.zero_point != "int":
        return None
    if getattr(spec.format, "frac_bits", 0):  # fixed-point codes: stored as the dense weight instead
        return None
    qmin, bits = spec.format.qmin, spec.format.bits
    values = qt.values.to(torch.int64)
    codes = values - qmin
    tensors = {"codes": pack_bits(codes.cpu(), bits)}
    tensors["scale"], scale_dtype = _scale_tensor(qt.scale.cpu())
    meta = {
        "bits": bits,
        "qmin": qmin,
        "count": values.numel(),
        "scale_shape": list(qt.scale.shape),
        "scale_dtype": scale_dtype,
        "granularity": spec.granularity,
        "group_size": spec.group_size,
    }
    if qt.zero_point is not None:
        zero = qt.zero_point.to(torch.int64)
        if not torch.equal(zero.to(qt.zero_point.dtype), qt.zero_point):
            return None  # a fractional zero point
        meta.update(zero_shape=list(qt.zero_point.shape), zero_count=zero.numel())
        if int(zero.min()) >= qmin and int(zero.max()) <= spec.format.qmax:
            tensors["zero"] = pack_bits((zero - qmin).cpu(), bits)
            meta["zero_bits"] = bits
        else:  # outside the code range: int16 (or int32), wider than the computed size counts
            wide = -(1 << 15) <= int(zero.min()) and int(zero.max()) < 1 << 15
            tensors["zero"] = zero.to(torch.int16 if wide else torch.int32).cpu()
            meta["zero_bits"] = 16 if wide else 32
    return Encoded("affine", tuple(qt.shape), tensors, meta)


def encode_nf4(weight: torch.Tensor) -> Encoded:
    from .weight_quant.nf4 import BLOCKSIZE, nf4_quantize

    index, absmax = nf4_quantize(weight)
    return Encoded(
        "nf4",
        tuple(weight.shape),
        {"codes": pack_bits(index.cpu(), 4), "absmax": absmax.float().cpu()},
        {"bits": 4, "count": index.numel(), "blocksize": BLOCKSIZE},
    )


def encode_dense(weight: torch.Tensor) -> Encoded:
    """The weight itself: BF16 when every value is a BF16, else fp32."""
    weight = weight.detach().float().cpu()
    narrow = weight.to(torch.bfloat16)
    if torch.equal(narrow.float(), weight):
        return Encoded("dense", tuple(weight.shape), {"weight": narrow}, {"bits": 16})
    return Encoded("dense", tuple(weight.shape), {"weight": weight}, {"bits": 32})


def decode(entry: dict[str, Any], tensors: dict[str, torch.Tensor], spec: Any) -> torch.Tensor:
    """The fp32 weight a stored layer decodes to (in its transformed basis when it has a transform), the
    value TriCast's emulation multiplies with. ``spec``: the layer's LinearSpec from the recipe."""
    kind, shape = entry["kind"], tuple(entry["shape"])
    if kind == "dense":
        return tensors["weight"].float()
    if kind == "nf4":
        from .weight_quant.nf4 import nf4_dequantize

        index = unpack_bits(tensors["codes"], 4, entry["count"]).reshape(shape)
        return nf4_dequantize(index, tensors["absmax"].float(), entry["blocksize"]).to(torch.bfloat16).float()
    if kind != "affine":
        raise ValueError(f"unknown stored layer kind {kind!r}")
    qspec = spec.weight
    codes = unpack_bits(tensors["codes"], entry["bits"], entry["count"]).to(torch.int64) + entry["qmin"]
    from .formats import container_dtype

    values = codes.to(container_dtype(qspec.format)).reshape(shape)
    zero = None
    if "zero" in tensors:
        if entry["zero_bits"] in (16, 32):
            zero = tensors["zero"].to(torch.int64)
        else:
            zero = (
                unpack_bits(tensors["zero"], entry["bits"], entry["zero_count"]).to(torch.int64)
                + entry["qmin"]
            )
        zero = zero.reshape(entry["zero_shape"]).to(torch.float32)
    qt = QTensor(
        values=values,
        scale=tensors["scale"].float(),
        zero_point=zero,
        global_scale=None,
        spec=qspec,
        shape=shape,
    )
    from .nn.linear import _operand_value

    return _operand_value(qt.mma_operand())


# ---------- capture during quantization ----------
class Capture:
    """Sets ``on_quantized`` on every EmuLinear of a patched model and keeps what each layer stores."""

    def __init__(self, model: nn.Module) -> None:
        from .nn.patch import iter_emulinear

        self.layers = dict(iter_emulinear(model))
        self.names = {id(layer): name for name, layer in self.layers.items()}
        from .nn.linear import EmuLinear

        # named_modules() lists a module once; a Linear shared under a second name would load as a copy
        shared = [
            name
            for name, module in model.named_modules(remove_duplicate=False)
            if isinstance(module, EmuLinear) and self.names.get(id(module)) != name
        ]
        if shared:
            raise ValueError(f"a Linear shared under two names is not packed: {shared[0]}")
        self.encoded: dict[str, Encoded] = {}
        self.fallback: dict[str, str] = {}
        for layer in self.layers.values():
            layer.on_quantized = self._hook

    def close(self) -> None:
        for layer in self.layers.values():
            layer.on_quantized = None

    def _hook(self, layer: nn.Module, weight: torch.Tensor, result: Any) -> None:
        name = self.names[id(layer)]
        if layer.spec.sparsity.kind != "none" or layer.spec.outliers is not None:
            raise ValueError(f"{name}: sparsity and outliers are not packed")
        if layer.spec.activation is not None:
            raise ValueError(f"{name}: activation quantization is not packed (the package runs weights only)")
        out_format = getattr(getattr(layer.spec.mma, "out_format", None), "name", None)
        if out_format not in _OUT_FORMATS:
            raise ValueError(
                f"{name}: MMA output format {out_format} is not packed (bf16, fp16 or fp32 only)"
            )
        encoded = None
        if isinstance(result, QTensor):
            encoded = encode_qtensor(result)
            reason = "a format or scale layout the packer does not store as codes"
        elif layer.spec.weight_algo.kind == "nf4":
            encoded = encode_nf4(weight)
            reason = "NF4 codes did not decode to the replaced weight"
            from .weight_quant.nf4 import nf4

            if not torch.equal(nf4(weight).float(), result.float()):
                encoded = None
        else:
            reason = f"{layer.spec.weight_algo.kind} keeps no integer codes TriCast can read back"
        if encoded is None:
            self.fallback[name] = reason
            from .nn.linear import _operand_value, as_operand

            encoded = encode_dense(
                _operand_value(as_operand(result, compact=True))
            )  # what TriCast multiplies
        self.encoded[name] = encoded

    def requantize_uncalibrated(self) -> None:
        """Layers quantized at patch time (before the hooks were set) are quantized again: RTN, NF4 and HQQ
        give the same weight on the same device."""
        for name, layer in self.layers.items():
            if name not in self.encoded and not layer.needs_calibration:
                layer.requantize()


def effective_weight(layer: nn.Module) -> torch.Tensor:
    """The fp32 weight TriCast's emulation of ``layer`` multiplies with."""
    from .nn.linear import _operand_value

    return _operand_value(layer._weight_operand)


# ---------- the run-time layer ----------
_OUT_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
_OUT_FORMATS = (
    None,
    "bf16",
    "fp16",
    "fp32",
)  # an output rounding PackedLinear reproduces (None: no MMA spec)


class PackedLinear(nn.Module):
    """A Linear with a decoded low-bit weight, computed as TriCast's emulation rounds it: fp32 products and
    accumulation, the output rounded to the MMA spec's output format (BF16 for the recipes' ``fp32_fma``).
    ``mode="load"`` decodes once (the weight in BF16 when every value is a BF16, else fp32);
    ``mode="codes"`` keeps the packed codes and decodes on every forward (low memory, slower)."""

    def __init__(
        self,
        name: str,
        entry: dict[str, Any],
        tensors: dict[str, torch.Tensor],
        spec: Any,
        bias: torch.Tensor | None,
        mode: str = "load",
    ) -> None:
        super().__init__()
        if mode not in ("load", "codes"):
            raise ValueError("mode must be 'load' or 'codes'")
        self.name, self.entry, self.spec, self.mode = name, entry, spec, mode
        self.out_features, self.in_features = entry["shape"]
        transform = entry.get("transform") or {"kind": "none"}
        diag = tensors.get("diag")
        self.transform = LinearTransform(
            transform["kind"], transform.get("block", 0), None if diag is None else diag.float()
        )
        out_format = getattr(getattr(spec, "mma", None), "out_format", None)
        self.out_dtype = _OUT_DTYPES.get(getattr(out_format, "name", None))
        self.bias = None if bias is None else nn.Parameter(bias.float(), requires_grad=False)
        if mode == "load":
            weight = decode(entry, tensors, spec)
            narrow = weight.to(torch.bfloat16)
            self.register_buffer("weight", narrow if torch.equal(narrow.float(), weight) else weight)
        else:
            for key, value in tensors.items():
                if key != "diag":
                    self.register_buffer(f"stored_{key}", value)

    def _apply(self, fn, recurse=True):  # noqa: ANN001, ANN202 (torch's signature)
        # .to(dtype) / .half() / .bfloat16() move this layer but never round its weight, codes or bias: they
        # hold the values TriCast multiplies with
        device = fn(torch.zeros((), dtype=torch.float32)).device
        return super()._apply(lambda t: t.to(device), recurse)

    def _weight(self) -> torch.Tensor:
        if self.mode == "load":
            return self.weight
        tensors = {
            key[len("stored_") :]: value for key, value in self._buffers.items() if key.startswith("stored_")
        }
        return decode(self.entry, tensors, self.spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._weight()
        if self.transform.kind != "none":
            if self.transform.diag is not None and self.transform.diag.device != x.device:
                self.transform.diag = self.transform.diag.to(x.device)
            xin = self.transform.apply_activation(x)
        else:
            xin = x
        if (
            xin.dtype == torch.bfloat16
            and self.bias is None
            and self.out_dtype is torch.bfloat16
            and weight.dtype == torch.bfloat16
        ):
            return F.linear(xin, weight)  # BF16 operands, fp32 accumulation, a BF16 output: the same rule
        out = F.linear(xin.float(), weight.float(), None if self.bias is None else self.bias.float())
        if self.out_dtype is not None:
            out = out.to(self.out_dtype)
        return out.to(x.dtype)


# ---------- files ----------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_dir(model_id: str, revision: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                model_id, revision=revision, local_files_only=True, allow_patterns=list(CHECKPOINT_FILES)
            )
        )
    except Exception:  # an uncached checkpoint: the package then needs the hub for its config
        return None


def write(
    model: nn.Module,
    capture: Capture,
    recipe_text: str,
    model_id: str,
    revision: str,
    out: Path,
    checks: dict[str, Any] | None = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Write the package of a patched, calibrated ``model`` whose quantization ``capture`` recorded
    (``tokenizer``: saved when the checkpoint snapshot has no tokenizer files to copy)."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).unlink(missing_ok=True)  # a new package starts with no checks
    quantized = set(capture.layers)
    missing = sorted(quantized - set(capture.encoded))
    if missing:
        raise RuntimeError(f"no quantized weight recorded for {len(missing)} layers, e.g. {missing[0]}")
    tensors: dict[str, torch.Tensor] = {}
    seen: dict[int, str] = {}
    tied = {}
    prefixes = tuple(f"{name}." for name in quantized)
    for name, value in model.state_dict().items():
        if name.startswith(prefixes):  # a quantized layer's own tensors are stored as codes below
            continue
        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            continue
        key = value.data_ptr()
        if key in seen:  # a tied tensor (output layer = token embedding) is stored once
            tied[name] = seen[key]
            continue
        seen[key] = name
        tensors[name] = value.detach().cpu().contiguous()
    layers = {}
    for name in sorted(quantized):
        layer, encoded = capture.layers[name], capture.encoded[name]
        entry = {"kind": encoded.kind, "shape": list(encoded.shape), **encoded.meta}
        transform = layer.transform
        if transform.kind != "none":
            entry["transform"] = {"kind": transform.kind, "block": transform.block}
            if transform.diag is not None:
                diag = transform.diag.detach().cpu()
                tensors[f"{name}.diag"] = (
                    diag.to(torch.int8) if transform.kind == "random_hadamard" else diag.float()
                )
        if layer.bias is not None:
            tensors[f"{name}.bias"] = layer.bias.detach().cpu().contiguous()
            entry["bias"] = True
        if name in capture.fallback:
            entry["fallback"] = capture.fallback[name]
        for key, value in encoded.tensors.items():
            tensors[f"{name}.{key}"] = value.contiguous()
        layers[name] = entry
    save_file(tensors, str(out / WEIGHTS), metadata={"format": SCHEMA})
    config = {
        "schema": SCHEMA,
        "model": model_id,
        "revision": revision,
        "tied": tied,
        "recipe": {"sha256": hashlib.sha256(recipe_text.encode("utf-8")).hexdigest(), "yaml": recipe_text},
        "layers": layers,
    }
    (out / CONFIG).write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    snapshot = _snapshot_dir(model_id, revision)
    copied = []
    if snapshot is not None:
        for name in CHECKPOINT_FILES:
            source = snapshot / name
            if source.is_file():
                shutil.copyfile(source, out / name)
                copied.append(name)
    if "config.json" not in copied:  # no cached snapshot: the loaded model's own config
        model.config.to_json_file(str(out / "config.json"))
    if tokenizer is not None and "tokenizer_config.json" not in copied:
        tokenizer.save_pretrained(str(out))
    return finish_manifest(out, model, capture, checks or {}, copied)


def finish_manifest(
    out: Path,
    model: nn.Module | None,
    capture: Capture | None,
    checks: dict[str, Any],
    copied: list[str] | None = None,
) -> dict[str, Any]:
    """(Re)write manifest.json: every file's size and SHA-256, the bits per quantized weight, the checks."""
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != MANIFEST
    }
    config = json.loads((out / CONFIG).read_text(encoding="utf-8"))
    params = sum(e["shape"][0] * e["shape"][1] for e in config["layers"].values())
    kinds: dict[str, int] = {}
    for entry in config["layers"].values():
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    previous = json.loads((out / MANIFEST).read_text(encoding="utf-8")) if (out / MANIFEST).is_file() else {}
    manifest = {
        "schema": SCHEMA,
        "model": config["model"],
        "revision": config["revision"],
        "recipe_sha256": config["recipe"]["sha256"],
        "files": files,
        "bytes": sum(f["bytes"] for f in files.values()),
        "weights_bytes": files[WEIGHTS]["bytes"],
        "quantized_layers": len(config["layers"]),
        "quantized_params": params,
        "layer_kinds": kinds,
        "fallback_layers": {n: e["fallback"] for n, e in config["layers"].items() if "fallback" in e},
        "checks": {**previous.get("checks", {}), **checks},
        "copied_from_checkpoint": copied if copied is not None else previous.get("copied_from_checkpoint"),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


# ---------- load ----------
def load(package: str | Path, device: str = "cpu", mode: str = "load", dtype: torch.dtype = torch.bfloat16):
    """The package as a Hugging Face causal LM with PackedLinear layers, and its tokenizer (None when the
    package has no tokenizer files)."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import no_init_weights

    from .nn.patch import decoder_layers
    from .recipe import load_recipe

    package = Path(package)
    config = json.loads((package / CONFIG).read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError(f"{package}: not a {SCHEMA} package")
    source = package if (package / "config.json").is_file() else config["model"]
    hf_config = AutoConfig.from_pretrained(source, revision=None if source == package else config["revision"])
    with no_init_weights(), torch.device(device):
        model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=dtype)
    tensors = load_file(str(package / WEIGHTS), device=device)
    import yaml

    recipe = load_recipe(yaml.safe_load(config["recipe"]["yaml"]))  # the text, not a path
    if recipe.kv is not None:
        raise ValueError(f"{package}: a KV-cache spec is not packed")
    decoder = decoder_layers(model)
    for name, entry in config["layers"].items():
        spec = recipe.spec_for(name, decoder=decoder)
        if spec is None:
            raise ValueError(f"{name}: the stored recipe does not select this layer")
        stored = {
            key[len(name) + 1 :]: tensors.pop(key) for key in list(tensors) if key.startswith(name + ".")
        }
        bias = stored.pop("bias", None)
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, PackedLinear(name, entry, stored, spec, bias, mode))
    result = model.load_state_dict(tensors, strict=False)
    owned = tuple(f"{name}." for name in config["layers"])  # loaded into each PackedLinear above
    tied = set(config.get("tied") or {})
    missing = [key for key in result.missing_keys if not key.startswith(owned) and key not in tied]
    if missing or result.unexpected_keys:
        raise ValueError(
            f"{package}: tensors missing {missing[:3]} or unexpected {result.unexpected_keys[:3]}"
        )
    if config.get("tied"):
        model.tie_weights()
    model.eval()
    tokenizer = None
    if (package / "tokenizer_config.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(package)
    return model, tokenizer


# ---------- commands ----------
def build(
    model_id: str,
    revision: str,
    recipe_path: Path,
    out: Path,
    device: str = "cuda",
    ppl: bool = False,
    hessian_max_bytes: int | None = None,
) -> dict[str, Any]:
    """Patch, calibrate (no cache: the codes are recorded while they are computed), check that every decoded
    weight is the weight TriCast multiplies with, write the package, and optionally compare TriCast's PPL
    with the package's own."""
    from . import calibrate, load_recipe, patch_model
    from .eval import perplexity
    from .eval.runner import _load_model

    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty: a package is written into a new directory")
    torch.manual_seed(42)
    started = time.time()
    recipe_text = recipe_path.read_text(encoding="utf-8")
    recipe = load_recipe(str(recipe_path))
    if recipe.kv is not None:
        raise ValueError("a KV-cache spec is not packed (the package runs weights only)")
    dtype = "bfloat16" if device == "cuda" else "float32"
    model, tokenizer = _load_model(model_id, dtype, device, revision=revision)
    loaded = getattr(model.config, "_commit_hash", None)
    patch_model(model, recipe)
    capture = Capture(model)
    capture.requantize_uncalibrated()
    if recipe.needs_calibration:
        kwargs = {} if hessian_max_bytes is None else {"hessian_max_bytes": hessian_max_bytes}
        calibrate(model, recipe, tokenizer=tokenizer, device=device, **kwargs)
    capture.requantize_uncalibrated()
    model.eval()
    # every decoded layer is the weight TriCast multiplies with, bit for bit (the sign of a zero aside)
    from .nn.patch import decoder_layers

    decoder = decoder_layers(model)
    mismatched = []
    for name, layer in capture.layers.items():
        encoded = capture.encoded[name]
        entry = {"kind": encoded.kind, "shape": list(encoded.shape), **encoded.meta}
        weight = decode(entry, encoded.tensors, recipe.spec_for(name, decoder=decoder))
        if not same_bits(weight, effective_weight(layer).cpu()):
            mismatched.append(name)
    checks: dict[str, Any] = {
        "decoded_equals_tricast_weight": {"layers": len(capture.layers), "mismatched": mismatched},
        "model_sha": loaded,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} layers decode to another weight than TriCast's, e.g. {mismatched[0]}"
        )
    if ppl:
        checks["ppl_tricast"] = perplexity(model, tokenizer, device=device, seqlen=2048)["ppl"]
    capture.close()
    write(model, capture, recipe_text, model_id, revision, out, checks, tokenizer)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    if ppl:
        packed, packed_tokenizer = load(out, device=device)
        checks["ppl_package"] = perplexity(packed, packed_tokenizer or tokenizer, device=device, seqlen=2048)[
            "ppl"
        ]
        checks["ppl_relative_difference"] = checks["ppl_package"] / checks["ppl_tricast"] - 1
        del packed
    checks["seconds"] = round(time.time() - started, 1)
    return finish_manifest(out, None, None, checks)


def run(
    package: Path, device: str, ppl: bool, prompts: list[str], max_new_tokens: int, mode: str
) -> dict[str, Any]:
    from .eval import perplexity

    started = time.time()
    model, tokenizer = load(package, device=device, mode=mode)
    result: dict[str, Any] = {
        "package": str(package),
        "mode": mode,
        "load_seconds": round(time.time() - started, 1),
    }
    if device == "cuda":
        # held after loading (the build of the skeleton allocates its BF16 Linears before they are replaced,
        # so the peak of the load says nothing), then the peak of the evaluation on top of it
        torch.cuda.synchronize()
        result["gpu_memory_loaded_bytes"] = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    if ppl:
        result["ppl"] = perplexity(model, tokenizer, device=device, seqlen=2048)["ppl"]
        if device == "cuda":
            result["gpu_memory_peak_eval_bytes"] = torch.cuda.max_memory_allocated()
    answers = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            sequence = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        answers.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(sequence[0, ids.shape[1] :], skip_special_tokens=True),
            }
        )
    result["answers"] = answers
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--revision", required=True)
    b.add_argument("--recipe", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument(
        "--ppl", action="store_true", help="WikiText-2 test PPL of TriCast's model and of the package"
    )
    b.add_argument("--hessian-max-bytes", type=int)
    r = sub.add_parser("run")
    r.add_argument("--package", type=Path, required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--ppl", action="store_true")
    r.add_argument("--prompt", action="append", default=[])
    r.add_argument("--max-new-tokens", type=int, default=128)
    r.add_argument("--mode", choices=("load", "codes"), default="load")
    r.add_argument("--out", type=Path, help="also write the result as JSON to this file")
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build(
            args.model, args.revision, args.recipe, args.out, args.device, args.ppl, args.hessian_max_bytes
        )
        print(
            json.dumps(
                {
                    k: manifest[k]
                    for k in ("bytes", "weights_bytes", "layer_kinds", "fallback_layers", "checks")
                },
                ensure_ascii=False,
            )
        )
        print(f"PACK_DONE out={args.out} bytes={manifest['bytes']}", flush=True)
        return 0
    result = run(args.package, args.device, args.ppl, args.prompt, args.max_new_tokens, args.mode)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print("PACK_RUN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
