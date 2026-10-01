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
from .formats import FP32, FloatFormat, Format, get_format
from .mma.api import as_operand, gemm
from .mma.operand import Operand
from .mma.spec import MMASpec
from .nn import EmuLinear, iter_emulinear, patch_model, unpatch_model
from .quant.api import quantize
from .recipe import Recipe, load_recipe
from .reference.cast import decode, round_to_format


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


def _ulp_format(fmt: Format | str) -> FloatFormat:
    fmt = get_format(fmt)
    if not isinstance(fmt, FloatFormat):
        raise ValueError(f"ULP distances need a float format, got {fmt}")
    return fmt


def _on_grid(x: torch.Tensor, fmt: FloatFormat, name: str) -> torch.Tensor:
    """``x`` in fp64 on the CPU, after checking that the library cast leaves every value
    unchanged. The reference decode that measures distances is exact on the CPU only: on CUDA
    it rejects grid values (seen with bf16 outputs of Qwen3-0.6B on a V100)."""
    values = x.detach().cpu().double()
    cast = round_to_format(values, fmt, "rne", saturate=False).double()
    if not bool(((cast == values) | (cast.isnan() & values.isnan())).all()):
        raise ValueError(f"{name} has values that are not on the {fmt} grid")
    return values


def _ordinal(values: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """Finite grid values as signed integers on a monotone line: the magnitude of the encoding
    (biased exponent and fraction fields, from the reference decode) carrying the value's sign.
    -0 and +0 are both 0, and neighbouring values are 1 apart, across zero too."""
    negative, exponent, significand, _ = decode(values, fmt)
    magnitude = (exponent - fmt.emin) * 2**fmt.mbits + significand
    if not fmt.subnormals:
        magnitude = magnitude - (2**fmt.mbits - 1)  # min_normal is the neighbour of zero
    magnitude = torch.where(significand == 0, 0, magnitude)
    return torch.where(negative, -magnitude, magnitude)


def ulp_distance(actual: torch.Tensor, expected: torch.Tensor, fmt: Format | str = FP32) -> torch.Tensor:
    """Exact int64 distance in units in the last place of ``fmt``, element by element.

    Both tensors must already hold values of ``fmt``; this is checked with a round trip through
    the library cast (:func:`tricast.reference.cast.round_to_format`), not through a dtype. The
    distance counts representable values: adjacent values are 1 apart, also across zero
    (``-min_subnormal`` to ``+min_subnormal`` is 2), and +0 and -0 are 0 apart. Inf and NaN
    have no distance here; :func:`ulp_error` classifies them.
    """
    fmt = _ulp_format(fmt)
    if actual.shape != expected.shape:
        raise ValueError("ULP distances require matching tensor shapes")
    a, e = _on_grid(actual, fmt, "actual"), _on_grid(expected, fmt, "expected")
    if not bool(torch.isfinite(a).all() & torch.isfinite(e).all()):
        raise ValueError("ulp_distance takes finite values; ulp_error classifies Inf and NaN")
    return (_ordinal(a, fmt) - _ordinal(e, fmt)).abs()


def _p99(distances: torch.Tensor) -> float:
    """Linear interpolation between the order statistics around rank ``(n - 1) * 0.99`` (NumPy's
    default ``linear`` method), with the rank split exactly in integers."""
    ordered = distances.flatten().sort().values
    rank, remainder = divmod((ordered.numel() - 1) * 99, 100)
    low = int(ordered[rank])
    high = int(ordered[min(rank + 1, ordered.numel() - 1)])
    return (low * 100 + (high - low) * remainder) / 100


@dataclass
class _UlpStats:
    fmt: FloatFormat
    n: int = 0
    finite: int = 0
    exact: int = 0
    total: int = 0
    maximum: int = 0
    p99: float = 0.0
    nonfinite_mismatch: int = 0

    def update(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        if actual.shape != expected.shape or not actual.numel():
            raise ValueError("ULP metrics require matching, nonempty tensor shapes")
        a, e = _on_grid(actual, self.fmt, "actual"), _on_grid(expected, self.fmt, "expected")
        finite = a.isfinite() & e.isfinite()
        special = (a.isnan() & e.isnan()) | (a.isinf() & (a == e))
        distances = (_ordinal(a[finite], self.fmt) - _ordinal(e[finite], self.fmt)).abs()
        self.n += a.numel()
        self.finite += distances.numel()
        self.exact += int((distances == 0).sum()) + int(special.sum())
        self.nonfinite_mismatch += int((~finite & ~special).sum())
        if distances.numel():
            self.total += int(distances.sum())
            self.maximum = max(self.maximum, int(distances.max()))
            self.p99 = max(self.p99, _p99(distances))

    def result(self) -> dict:
        if not self.n:
            raise ValueError("no outputs were compared")
        measured = self.finite > 0
        return {"max": self.maximum if measured else None,
                "mean": self.total / self.finite if measured else None,
                "p99": self.p99 if measured else None,
                "exact_fraction": self.exact / self.n, "n": self.n,
                "nonfinite_mismatch": self.nonfinite_mismatch}


def ulp_error(actual: torch.Tensor, expected: torch.Tensor, fmt: Format | str = FP32) -> dict:
    """ULP statistics of ``actual`` against ``expected``, both on the ``fmt`` grid.

    ``n`` counts every position. ``max``, ``mean`` and ``p99`` (:func:`ulp_distance`) cover the
    positions where both values are finite, and are ``None`` when there are none; ``p99``
    interpolates linearly between the order statistics around rank ``(count - 1) * 0.99``
    (NumPy's default). ``exact_fraction`` is the share of all ``n`` positions that match: finite
    pairs 0 ULP apart, NaN/NaN, and infinities of the same sign. ``nonfinite_mismatch`` counts
    the other positions holding an Inf or NaN: finiteness or NaN-ness differs, or the
    infinities have opposite signs.
    """
    stats = _UlpStats(_ulp_format(fmt))
    stats.update(actual, expected)
    return stats.result()


def _operand_value(operand: Operand) -> torch.Tensor:
    value = operand.values.float()
    scale = operand.scale_per_element()
    if scale is not None:
        value = value * scale
    if operand.alpha is not None:
        value = value * operand.alpha
    return value


def _activation_pair(layer: EmuLinear, x: torch.Tensor) -> tuple[torch.Tensor, Operand]:
    rows = x.reshape(-1, layer.in_features)
    activation = rows if layer.spec.transform.kind == "none" else layer.transform.apply_activation(rows)
    if layer.spec.activation is None:
        return activation, as_operand(activation)
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
    return activation, operand


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
             "Max abs error | Relative Frobenius | Cosine | MMA ULP max / mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in layers:
        if layer["output"] is None:
            lines.append(f"| {layer['name']} (not observed) | - | - | - | - | - | - | - | - |")
            continue
        output = layer["output"]
        values = [layer["weight"]["sqnr_db"], layer["activation"]["sqnr_db"], output["mse"],
                  output["sqnr_db"], output["max_abs_error"], output["relative_frobenius"], output["cosine"]]
        ulp = layer["mma_ulp"]
        ulp_cell = "-" if ulp["max"] is None else f"{ulp['max']} / {number(ulp['mean'])} ({ulp['format']})"
        name = layer["name"].replace("|", "\\|")
        lines.append(f"| {name} | " + " | ".join(number(value) for value in values) + f" | {ulp_cell} |")
    lines.extend(["", "MMA ULP: the layer's accumulation vs fp64 accumulation of the same quantized "
                  "operands, in ULPs of its MMA output format.",
                  "", f"Logits KL(ref || emu): {number(model_metrics['logits_kl'])}; "
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
    ``mma_ulp`` isolates the accumulator: the same operands and bias go through the
    layer's MMA and through fp64 accumulation with the same output format, and
    :func:`ulp_error` between the two (in ULPs of that format) is pooled over the forward calls.
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
        stats[layer.name]["activation"].update(activation, _operand_value(operand))
        # The forward's GEMM path (outlier path included) with the layer's MMA and with fp64
        # accumulation on the main path; only the accumulator differs. Recomputed here because
        # the forward output is cast to the model dtype, which can be coarser than out_format.
        exact = MMASpec("fp64", out_format=layer.spec.mma.out_format)
        stats[layer.name]["mma_ulp"].update(layer._matmul(operand, gemm_fn=gemm),
                                            layer._matmul(operand, exact, gemm_fn=gemm))

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
                effective = _operand_value(layer._weight_operand)
                if layer._outlier_operand is not None:  # disjoint supports: the sum is exact
                    effective = effective + _operand_value(layer._outlier_operand)
                stats[name] = {"weight": error_metrics(transformed, effective),
                               "activation": _ErrorStats(), "output": _ErrorStats(),
                               "mma_ulp": _UlpStats(layer.spec.mma.out_format)}
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
               "output": entry["output"].result() if entry["output"].count else None,
               "mma_ulp": ({**entry["mma_ulp"].result(), "format": str(entry["mma_ulp"].fmt)}
                           if entry["mma_ulp"].n else None)}
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
                             "mma_ulp": "the layer's GEMM vs MMASpec('fp64', out_format=its out_format) on "
                                        "identical operands and bias, in ULPs of that format; mean and max "
                                        "pooled over all forward calls, p99 = the largest p99 of a single "
                                        "call (one window, or one token when the KV cache streams; linear "
                                        "interpolation), not a pooled percentile",
                             "infinity_encoding": "inf and -inf are JSON strings"}}
