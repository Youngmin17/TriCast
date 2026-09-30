"""Layer-local arithmetic errors and end-to-end language-model quality."""

from __future__ import annotations

import copy
import hashlib
import math
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .calibration import calibrate
from .mma.api import as_operand
from .mma.operand import Operand
from .nn import EmuLinear, iter_emulinear, patch_model, unpatch_model
from .quant.api import quantize
from .recipe import Recipe, load_recipe


@dataclass
class _ErrorStats:
    count: int = 0
    signal: float = 0.0
    estimate: float = 0.0
    error: float = 0.0
    dot: float = 0.0
    maximum: float = 0.0

    def update(self, reference: torch.Tensor, emulated: torch.Tensor) -> None:
        if reference.shape != emulated.shape or not reference.numel():
            raise ValueError("error metrics require matching, nonempty tensor shapes")
        x, y = reference.detach().double(), emulated.detach().double()
        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError("error metrics encountered non-finite values")
        difference = x - y
        self.count += x.numel()
        self.signal += x.square().sum().item()
        self.estimate += y.square().sum().item()
        self.error += difference.square().sum().item()
        self.dot += (x * y).sum().item()
        self.maximum = max(self.maximum, difference.abs().max().item())

    def result(self) -> dict:
        if not self.count:
            raise ValueError("no layer activations were observed")
        sqnr = ("inf" if not self.error else "-inf" if not self.signal
                else 10.0 * math.log10(self.signal / self.error))
        relative = (math.sqrt(self.error / self.signal) if self.signal
                    else 0.0 if not self.error else "inf")
        cosine = (max(-1.0, min(1.0, self.dot / math.sqrt(self.signal * self.estimate)))
                  if self.signal and self.estimate else float(not self.signal and not self.estimate))
        return {"mse": self.error / self.count, "sqnr_db": sqnr, "max_abs_error": self.maximum,
                "relative_frobenius": relative, "cosine": cosine}


def error_metrics(reference: torch.Tensor, emulated: torch.Tensor) -> dict:
    """Global fp64 reductions; infinite ratios are strings for strict JSON output."""
    stats = _ErrorStats()
    stats.update(reference, emulated)
    return stats.result()


def _operand_value(operand: Operand) -> torch.Tensor:
    value = operand.values.float()
    scale = operand.scale_per_element()
    if scale is not None:
        value = value * scale
    if operand.alpha is not None:
        value = value * operand.alpha
    return value


def _activation_pair(layer: EmuLinear, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.reshape(-1, layer.in_features)
    activation = rows if layer.spec.transform.kind == "none" else layer.transform.apply_activation(rows)
    if layer.spec.activation is None:
        return activation, activation
    # Reconstruct exactly the operand the upcoming forward consumes, without advancing
    # its delayed observer or stochastic-rounding RNG a second time.
    amax = None
    if layer.observer is not None:
        observer = copy.deepcopy(layer.observer)
        observer.mode = layer.mode
        amax = observer.observe(activation.detach())
    devices = [activation.device.index] if activation.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        noise = layer._noise(activation, layer.spec.activation)
        operand = as_operand(quantize(activation, layer.spec.activation, backend=layer.backend,
                                     amax=amax, noise=noise))
    return activation, _operand_value(operand)


def _report_windows(
    texts: Iterable[str] | str | None, input_ids: Any, tokenizer: Any, samples: int, seqlen: int,
) -> torch.Tensor:
    if samples < 1 or seqlen < 2:
        raise ValueError("report samples must be positive and seqlen must be >= 2")
    if (texts is None) == (input_ids is None):
        raise ValueError("pass exactly one of texts or input_ids")
    if texts is not None:
        if tokenizer is None:
            raise ValueError("a tokenizer is required for report texts")
        joined = texts if isinstance(texts, str) else "\n\n".join(texts)
        input_ids = tokenizer(joined, return_tensors="pt")["input_ids"]
    ids = torch.as_tensor(input_ids)
    if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
        raise ValueError("report input_ids must contain integer token IDs")
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or not ids.shape[0] or ids.shape[1] < seqlen:
        raise ValueError("report input_ids must be 1-D or 2-D with at least one complete window")
    # Never join unrelated batch rows into the same language-model sequence.
    per_row = ids.shape[1] // seqlen
    return ids[:, :per_row * seqlen].reshape(-1, seqlen)[:samples].to(device="cpu", dtype=torch.long)


def _logits(model: nn.Module, ids: torch.Tensor, device: torch.device, streaming: bool) -> torch.Tensor:
    from .eval.ppl import _model_logits

    logits = _model_logits(model, ids.to(device), streaming=streaming).detach()
    if logits.ndim != 3 or logits.shape[:2] != ids.shape or not torch.isfinite(logits).all():
        raise ValueError("report requires finite [batch, tokens, vocabulary] logits")
    return logits.double()


def _markdown(layers: list[dict], model_metrics: dict) -> str:
    def number(value: float | str) -> str:
        return value if isinstance(value, str) else f"{value:.6g}"

    lines = ["# TriCast error report", "", "Layers sorted by output SQNR, worst first.", "",
             "| Layer | Weight SQNR (dB) | Activation SQNR (dB) | Output MSE | Output SQNR (dB) | "
             "Max abs error | Relative Frobenius | Cosine |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in layers:
        if layer["output"] is None:
            lines.append(f"| {layer['name']} (not observed) | - | - | - | - | - | - | - |")
            continue
        output = layer["output"]
        values = [layer["weight"]["sqnr_db"], layer["activation"]["sqnr_db"], output["mse"],
                  output["sqnr_db"], output["max_abs_error"], output["relative_frobenius"], output["cosine"]]
        name = layer["name"].replace("|", "\\|")
        lines.append(f"| {name} | " + " | ".join(number(value) for value in values) + " |")
    lines.extend(["", f"Logits KL(ref || emu): {number(model_metrics['logits_kl'])}; "
                  f"top-1 agreement: {number(model_metrics['top1_agreement'])}.",
                  f"PPL reference: {number(model_metrics['ppl_reference'])}; "
                  f"PPL emulated: {number(model_metrics['ppl_emulated'])}."])
    return "\n".join(lines) + "\n"


@torch.no_grad()
def layer_report(
    model: nn.Module,
    recipe: Recipe | dict | str,
    *,
    texts: Iterable[str] | str | None = None,
    input_ids: Any = None,
    tokenizer: Any = None,
    samples: int = 8,
    seqlen: int = 128,
    device: torch.device | str | None = None,
) -> dict:
    """Temporarily patch an unpatched model and measure identical evaluation windows.

    Weight/activation errors use transformed coordinates and the effective MMA
    operand, including dequant-format rounding. Output errors compare each actual
    patched-layer output with its original Linear on the identical hooked input.
    Calibration uses its recipe/default dataset, never implicit evaluation tokens.
    ``texts`` requires an explicit tokenizer; this function does not load models.
    """
    recipe = load_recipe(recipe)
    if list(iter_emulinear(model)) or getattr(model, "_tricast_kv_patch", None) is not None:
        raise ValueError("layer_report requires an unpatched model")
    windows = _report_windows(texts, input_ids, tokenizer, samples, seqlen)
    model_device = next(model.parameters()).device
    device = model_device if device is None else torch.device(device)
    if device.type == model_device.type and device.index is None:
        device = model_device
    if device != model_device:
        raise ValueError("move the model to the report device before calling layer_report")
    if recipe.needs_calibration and tokenizer is None:
        raise ValueError("a tokenizer is required for the recipe calibration dataset")
    modes = [(module, module.training) for module in model.modules()]
    handles = []
    stats: dict[str, dict[str, Any]] = {}
    calibration = None
    calibration_config = None
    nll_reference = nll_emulated = kl_sum = 0.0
    matches = n_positions = 0
    streaming = recipe.kv is not None and recipe.kv.mode == "cache"

    def before(layer: EmuLinear, args: tuple) -> None:
        x = args[0].detach()
        activation, operand = _activation_pair(layer, x)
        stats[layer.name]["activation"].update(activation, operand)

    def after(layer: EmuLinear, args: tuple, output: torch.Tensor) -> None:
        x = args[0].detach()
        reference = F.linear(x, layer.weight, None if layer.bias is None else layer.bias.to(x.dtype))
        stats[layer.name]["output"].update(reference, output)

    try:
        model.eval()
        # Logits can dominate RAM for real vocabularies; retain the baseline on disk.
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), tempfile.TemporaryFile() as baseline:
            torch.random.default_generator.manual_seed(42)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(42)
            for ids in windows.split(1):
                logits = _logits(model, ids, device, streaming)
                nll_reference += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                 ids[:, 1:].to(device).reshape(-1), reduction="sum").item()
                np.save(baseline, logits.cpu().numpy(), allow_pickle=False)
            patch = patch_model(model, recipe)
            if not patch.patched and not patch.kv:
                raise ValueError("the recipe patched no Linear modules for the report")
            if recipe.needs_calibration:
                calibration_config = recipe.calibration_options
                calibration = calibrate(model, recipe, tokenizer=tokenizer, device=device)
            for name, layer in iter_emulinear(model):
                transformed = (layer.weight.detach() if layer.spec.transform.kind == "none"
                               else layer.transform.apply_weight(layer.weight.detach()))
                stats[name] = {"weight": error_metrics(transformed, _operand_value(layer._weight_operand)),
                               "activation": _ErrorStats(), "output": _ErrorStats()}
                handles.append(layer.register_forward_pre_hook(before))
                handles.append(layer.register_forward_hook(after))
            baseline.seek(0)
            for ids in windows.split(1):
                logits = _logits(model, ids, device, streaming)
                reference = torch.from_numpy(np.load(baseline, allow_pickle=False)).to(device)
                log_p, log_q = reference.log_softmax(-1), logits.log_softmax(-1)
                kl_sum += (log_p.exp() * (log_p - log_q)).sum().item()
                matches += (reference.argmax(-1) == logits.argmax(-1)).sum().item()
                n_positions += ids.numel()
                nll_emulated += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                ids[:, 1:].to(device).reshape(-1), reduction="sum").item()
    finally:
        for handle in handles:
            handle.remove()
        unpatch_model(model)
        for module, training in modes:
            module.training = training

    layers = [{"name": name, "weight": entry["weight"],
               "status": "measured" if entry["output"].count else "not_observed",
               "activation": entry["activation"].result() if entry["activation"].count else None,
               "output": entry["output"].result() if entry["output"].count else None}
              for name, entry in stats.items()]
    layers.sort(key=lambda entry: (entry["output"] is None,
                                  float(entry["output"]["sqnr_db"]) if entry["output"] else 0, entry["name"]))
    n_tokens = len(windows) * (seqlen - 1)
    model_metrics = {"logits_kl": max(0.0, kl_sum / n_positions), "top1_agreement": matches / n_positions,
                     "ppl_reference": math.exp(nll_reference / n_tokens),
                     "ppl_emulated": math.exp(nll_emulated / n_tokens), "n_tokens": n_tokens}
    fingerprint = hashlib.sha256(windows.contiguous().numpy().astype("<i8").tobytes()).hexdigest()
    return {"layers": layers, "model": model_metrics, "markdown": _markdown(layers, model_metrics),
            "samples": len(windows), "seqlen": seqlen, "seed": 42, "dataset_fingerprint": fingerprint,
            "forward_mode": "streaming_cache" if streaming else "full_window",
            "recipe_hash": recipe.sha256, "patch_report": asdict(patch), "calibration": calibration,
            "calibration_config": calibration_config,
            "metric_notes": {"coordinate_system": "transformed weight and activation operands",
                             "output_reference": "original Linear on the patched layer's actual input",
                             "sqnr_db": "10 log10(sum(reference^2) / sum(error^2)); inf for zero error",
                             "logits_kl": "mean over all positions; negative fp64 roundoff clamped to 0",
                             "zero_norm_cosine": "1 for two zero tensors; 0 for exactly one zero tensor",
                             "infinity_encoding": "inf and -inf are JSON strings"}}
