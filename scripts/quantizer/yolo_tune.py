"""YOLO11n per-layer weight bit-widths on COCO val2017: the offline demo dataset of the tuning page.

"기존 방법" = one weight quantization for every conv (uniform INT8/INT4/INT3) or the common INT8-edges /
INT4-rest heuristic; "우리 방법" = TriCast Auto-tune: a method per conv chosen by the exact multiple-choice
knapsack (app.autotune.solve_mckp) over a measured per-layer sensitivity table, at exactly the same average
bits as the setup it is compared with.

Stages (each ends with one log marker: YOLO_<STAGE>_DONE, or YOLO_FAIL stage=<stage>):
  table     D(layer, method) on 128 calibration images, one conv quantized at a time -> table.json
  solve     knapsack at u_int4 / heur_8_4 / u_int3 average bits -> solutions.json
  evaluate  per setup: COCO bbox AP (4872 evaluation images), measured D, demo detections -> eval_<setup>.json
  export    app/web/demo/autotune/yolo11n.json

Numerics: weight-only fake-quant. Each conv weight is viewed as [out, in/groups*kh*kw] (EmuConv2d's K order)
and replaced by tricast.quant.api.fake_quant (reference backend): symmetric intN (k in [-(2^(N-1)-1),
2^(N-1)-1]), per-output-channel absmax scale rounded onto fp16 (RNE), RNE elements. Activations and the conv
itself stay native FP32; this is not a low-bit kernel. ``fp16`` = the weight left unchanged (FP32 compute,
D = 0) and counted at 16 bits. The DFL conv (fixed arange(16) decoding weights, not learned) is never
quantized.

The model is app.runners.vision.load_detector (raw Ultralytics 8.3.221 model fused before anything else);
.predict()/.val()/AutoBackend are never called. COCO evaluation reuses scripts/e2e/evaluate_vision_quality.py
(rect validation, class-aware NMS, pycocotools) over all 5000 images in its batch plan, scored on the 4872
evaluation ids. Run on a CPU host (cpu_cap); CPU fp32, seed 42, deterministic algorithms.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import platform
import random
import socket
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "tricast.vision_tune/1"
TABLE_SCHEMA = "tricast.vision_tune.table/1"
CHECKPOINT = Path("/scratch/uceeeee/tricast/quality_20261005/checkpoints/yolo11n.pt")
CHECKPOINT_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
COCO = Path("/scratch/uceeeee/tricast/quality_20261005/data/coco2017")
ULTRALYTICS = "8.3.221"
IMGSZ = 640
SEED = 42
DEMO_IDS = (1000, 5001, 8021)
CALIB_COUNT = 128
CALIB_BATCH = 32
COCO_IMAGES = 5000
# element bits; the order is the knapsack's option order
METHODS = {"fp16": 16, "int8": 8, "int4": 4, "int3": 3}
QUANT_METHODS = ("int8", "int4", "int3")
SCALE_BITS = 16  # one fp16 scale per output channel
EVAL = {"conf": 0.001, "iou": 0.7, "max_det": 300, "batch": 8}  # the evaluator's defaults
# COCOeval.stats 0-5 (AP over IoU 0.5:0.95, at IoU 0.5 and 0.75, and by object area), all from one run
METRIC_KEYS = ("AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")
DRAW = {"conf": 0.25, "iou": 0.7, "max_det": 100}
UNIFORM = {"u_int8": "int8", "u_int4": "int4", "u_int3": "int3"}
HEURISTIC = "heur_8_4"
AUTO = {"auto_iso_u4": "u_int4", "auto_iso_h84": HEURISTIC, "auto_iso_u3": "u_int3"}  # auto -> size reference
SETUPS = ("native", *UNIFORM, HEURISTIC, *AUTO)
LABELS = {
    "u_int8": "INT8 균일",
    "u_int4": "INT4 균일",
    "u_int3": "INT3 균일",
    HEURISTIC: "INT8 가장자리 + INT4",
    "auto_iso_u4": "Auto-tune · INT4 균일 크기",
    "auto_iso_h84": "Auto-tune · INT8/4 경험칙 크기",
    "auto_iso_u3": "Auto-tune · INT3 균일 크기",
}
QUANTIZATION = (
    "weight-only fake-quant: symmetric per-output-channel absmax, RNE, fp16 scale; activations FP32"
)
OBJECTIVE = (
    "D = mean over the 128 calibration images of sum_a w_a*(||p_q - p||^2 + ||b_q - b||^2 / 640^2) / "
    "sum_a w_a, where y = [B, 84, A] is the fused model's raw head output (b = 4 box coordinates in "
    "letterbox pixels, p = 80 sigmoid class probabilities) of the unquantized model and y_q of the "
    "quantized one, w_a = max_c p_a (native). Table: one conv quantized at a time; the knapsack sums "
    "D over layers (additive assumption), the measured D quantizes every layer of a setup at once"
)
CALIB_RULE = (
    f"random.Random({SEED}).sample(sorted(val2017 ids - {{{', '.join(map(str, DEMO_IDS))}}}), "
    f"{CALIB_COUNT}), sorted"
)
CALIB_PREPROCESS = (
    f"fixed {IMGSZ}x{IMGSZ} letterbox: long side resized to {IMGSZ} (ceil, cv2 INTER_LINEAR, no "
    "upscaling), Ultralytics LetterBox centre pad 114, BGR->RGB, /255, batches of "
    f"{CALIB_BATCH}"
)
EVAL_RULE = (
    f"the {COCO_IMAGES - CALIB_COUNT} val2017 ids not in calibration.ids (the demo images are "
    "included); inference runs on all 5000 in the evaluator's rect batch plan (batch 8) and "
    "pycocotools scores only these ids"
)
SIZE_RULE = (
    "bytes = sum over the quantizable conv weights of params * bits / 8, bits = b + 16/K for intb (one "
    "fp16 scale per output channel, K = in_channels/groups*kh*kw) and 16 for fp16, plus every other "
    "parameter and buffer of the fused deployed model (conv biases with BatchNorm folded in, the "
    "fixed DFL conv weight) at 2 bytes; original = every parameter and buffer at 2 bytes (FP16 file). "
    "Rounded up to whole bytes. Excluded: file headers, the model graph, class names"
)


class TuneError(RuntimeError):
    """A failed protocol check: the stage stops and prints YOLO_FAIL."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TuneError(message)


# ---------- pure accounting ----------
@dataclass(frozen=True)
class ConvLayer:
    name: str
    params: int
    k: int  # in_channels / groups * kh * kw: elements per output-channel scale
    out: int


def stored_bits(method: str, k: int) -> Fraction:
    """Stored bits per weight: the element plus a share of the per-output-channel fp16 scale."""
    if method == "fp16":
        return Fraction(METHODS["fp16"])
    return METHODS[method] + Fraction(SCALE_BITS, k)


def layer_cost(layer: ConvLayer, method: str) -> Fraction:
    """Exact stored bits of one conv weight (= params * b + 16 * out for intb)."""
    return layer.params * stored_bits(method, layer.k)


def total_bits(layers: Sequence[ConvLayer], assignment: Mapping[str, str]) -> Fraction:
    require(set(assignment) == {layer.name for layer in layers}, "assignment must name every conv layer")
    return sum((layer_cost(layer, assignment[layer.name]) for layer in layers), Fraction(0))


def avg_bits(layers: Sequence[ConvLayer], assignment: Mapping[str, str]) -> Fraction:
    """Parameter-weighted average stored bits over the conv weights."""
    return total_bits(layers, assignment) / sum(layer.params for layer in layers)


def model_bytes(layers: Sequence[ConvLayer], assignment: Mapping[str, str], other_params: int) -> int:
    return math.ceil((total_bits(layers, assignment) + 16 * other_params) / 8)


def original_bytes(layers: Sequence[ConvLayer], other_params: int) -> int:
    return 2 * (sum(layer.params for layer in layers) + other_params)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def parse_fraction(text: str) -> Fraction:
    numerator, denominator = text.split("/")
    return Fraction(int(numerator), int(denominator))


def method_counts(assignment: Mapping[str, str]) -> dict[str, int]:
    counts = Counter(assignment.values())
    return {method: counts.get(method, 0) for method in METHODS}


def uniform_assignment(layers: Sequence[ConvLayer], method: str) -> dict[str, str]:
    return {layer.name: method for layer in layers}


def heuristic_assignment(layers: Sequence[ConvLayer], int8_names: Sequence[str]) -> dict[str, str]:
    """INT8 on the given layers (first conv and the Detect head's final prediction convs), INT4 elsewhere."""
    names = {layer.name for layer in layers}
    require(set(int8_names) <= names, f"unknown heuristic layers {sorted(set(int8_names) - names)}")
    return {layer.name: "int8" if layer.name in int8_names else "int4" for layer in layers}


def baseline_assignment(setup: str, layers: Sequence[ConvLayer], int8_names: Sequence[str]) -> dict[str, str]:
    if setup == "native":
        return uniform_assignment(layers, "fp16")
    if setup in UNIFORM:
        return uniform_assignment(layers, UNIFORM[setup])
    require(setup == HEURISTIC, f"{setup} is not a uniform/heuristic setup")
    return heuristic_assignment(layers, int8_names)


def calibration_ids(all_ids: Sequence[int]) -> list[int]:
    pool = sorted(set(all_ids) - set(DEMO_IDS))
    require(len(pool) >= CALIB_COUNT, "not enough images to draw the calibration set from")
    return sorted(random.Random(SEED).sample(pool, CALIB_COUNT))


def evaluation_ids(all_ids: Sequence[int], calibration: Sequence[int]) -> list[int]:
    return sorted(set(all_ids) - set(calibration))


def predicted_d(assignment: Mapping[str, str], table: Mapping[str, Mapping[str, float]]) -> float:
    """Sum of the per-layer table D (fp16 = 0): the knapsack's additive model."""
    value = 0.0
    for name, method in assignment.items():
        value += 0.0 if method == "fp16" else float(table[name][method])
    return value


def solve(
    layers: Sequence[ConvLayer], table: Mapping[str, Mapping[str, float]], budget_bits: Fraction
) -> tuple[dict[str, str], Any]:
    """Minimum predicted D with the conv weights' total stored bits <= ``budget_bits`` (exact costs)."""
    from app.autotune import solve_mckp

    methods = list(METHODS)
    costs = [[layer_cost(layer, method) for method in methods] for layer in layers]
    values = [
        [0.0 if method == "fp16" else float(table[layer.name][method]) for method in methods]
        for layer in layers
    ]
    solution = solve_mckp(costs, values, budget_bits)
    require(solution.bound == 0, f"knapsack costs were rounded (bound {solution.bound}); not exact")
    require(solution.cost <= budget_bits, "knapsack exceeded its budget")
    return {layer.name: methods[j] for layer, j in zip(layers, solution.choice, strict=True)}, solution


# ---------- numerics ----------
def weight_spec(method: str) -> Any:
    """The QuantSpec of an int method (None for fp16): symmetric intN, per-output-channel absmax scale
    rounded onto fp16 (RNE), RNE elements, saturating."""
    from tricast.quant.spec import QuantSpec, ScaleSpec

    if method == "fp16":
        return None
    require(method in QUANT_METHODS, f"unknown method {method!r}")
    return QuantSpec(
        format=f"int{METHODS[method]}",
        granularity="row",
        scale=ScaleSpec(format="fp16", method="absmax", rounding="rne"),
        rounding="rne",
    )


def fake_quant_weight(weight: Any, method: str) -> Any:
    """Conv weight [out, in/groups, kh, kw] -> its fake-quantized copy (same shape and dtype)."""
    from tricast.quant.api import fake_quant

    spec = weight_spec(method)
    if spec is None:
        return weight.detach().clone()
    rows = weight.detach().reshape(weight.shape[0], -1)
    return fake_quant(rows, spec, backend="reference").reshape(weight.shape).to(weight.dtype)


def distortion(reference: Any, output: Any, imgsz: int = IMGSZ) -> Any:
    """Per-image D of one batch of raw head outputs [B, 84, A] (float64, shape [B])."""
    ref, out = reference.double(), output.double()
    p, q = ref[:, 4:], out[:, 4:]
    weight = p.amax(dim=1)
    error = ((q - p) ** 2).sum(dim=1) + ((out[:, :4] - ref[:, :4]) ** 2).sum(dim=1) / imgsz**2
    return (weight * error).sum(dim=-1) / weight.sum(dim=-1)


# ---------- model and data ----------
def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_env() -> dict[str, Any]:
    import torch

    from tricast.eval.envinfo import source_identity

    identity = source_identity()
    return {
        "git_sha": identity["git_sha"],
        "git_dirty": identity["git_dirty"],
        "script_sha256": sha256_file(Path(__file__)),
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "ultralytics": _version("ultralytics"),
        "cpu": cpu_name(),
        "threads": torch.get_num_threads(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def setup_torch(threads: int) -> None:
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)


def load_model(checkpoint: Path) -> tuple[Any, list[str]]:
    from app.runners.vision import load_detector

    require(
        sha256_file(checkpoint) == CHECKPOINT_SHA256, f"checkpoint sha256 differs from the pin: {checkpoint}"
    )
    require(_version("ultralytics") == ULTRALYTICS, f"needs ultralytics=={ULTRALYTICS}")
    return load_detector("cpu", checkpoint)


def conv_layers(model: Any) -> tuple[list[ConvLayer], dict[str, Any], list[str], str]:
    """The quantizable convs in module order (every Conv2d except the DFL's fixed decoding conv), their
    modules, the heuristic's INT8 layers (first conv + the last conv of every Detect cv2/cv3 branch) and the
    DFL conv's name."""
    from torch import nn

    detect = model.model[-1]
    dfl = detect.dfl.conv
    names = {module: name for name, module in model.named_modules()}
    layers, modules = [], {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module is not dfl:
            k = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
            require(module.weight.numel() == module.out_channels * k, f"{name}: unexpected weight shape")
            layers.append(ConvLayer(name, module.weight.numel(), k, module.out_channels))
            modules[name] = module
    finals = [branch[-1] for head in (detect.cv2, detect.cv3) for branch in head]
    require(all(isinstance(conv, nn.Conv2d) for conv in finals), "Detect branches must end in a Conv2d")
    int8 = [layers[0].name, *(names[conv] for conv in finals)]
    return layers, modules, int8, names[dfl]


def other_params(model: Any, modules: Mapping[str, Any]) -> tuple[int, dict[str, int]]:
    """Every parameter and buffer of the fused model except the quantizable conv weights."""
    quantized = {id(module.weight) for module in modules.values()}
    detail: Counter[str] = Counter()
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if id(tensor) in quantized:
            continue
        detail[name.rsplit(".", 1)[-1] if name.endswith(".bias") else name] += tensor.numel()
    grouped = {"conv_bias": sum(v for k, v in detail.items() if k == "bias")}
    grouped.update({k: v for k, v in detail.items() if k != "bias"})
    return sum(detail.values()), grouped


def weights_digest(modules: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, module in modules.items():
        digest.update(name.encode())
        digest.update(module.weight.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def quantized(
    modules: Mapping[str, Any], assignment: Mapping[str, str], cache: dict[tuple[str, str], Any] | None = None
) -> Iterator[None]:
    """Replace the weights of the assigned convs by their fake-quantized copies, restore them on exit."""
    import torch

    saved = {}
    try:
        with torch.no_grad():
            for name, method in assignment.items():
                if method == "fp16":
                    continue
                conv = modules[name]
                saved[name] = conv.weight.detach().clone()
                key = (name, method)
                value = cache.get(key) if cache is not None else None
                if value is None:
                    value = fake_quant_weight(saved[name], method)
                    if cache is not None:
                        cache[key] = value
                conv.weight.copy_(value)
        yield
    finally:
        with torch.no_grad():
            for name, weight in saved.items():
                modules[name].weight.copy_(weight)


def annotation_ids(annotations: Path) -> list[int]:
    data = json.loads(annotations.read_text())
    ids = sorted(item["id"] for item in data["images"])
    require(len(ids) == COCO_IMAGES == len(set(ids)), "instances_val2017.json must list 5000 unique images")
    require(all(i in ids for i in DEMO_IDS), "a demo image is not in val2017")
    return ids


def image_file(coco: Path, image_id: int) -> Path:
    return coco / "val2017" / f"{image_id:012d}.jpg"


def square_input(data: bytes, stride: int) -> Any:
    """CALIB_PREPROCESS for one image: uint8 [3, 640, 640] RGB."""
    import cv2
    import numpy as np
    import torch
    from app.runners.vision import letterbox
    from ultralytics.data.augment import LetterBox

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(image is not None, "cannot decode a calibration image")
    plan = letterbox(*image.shape[:2], IMGSZ, stride)
    if plan.resized != plan.original:
        image = cv2.resize(image, plan.resized[::-1], interpolation=cv2.INTER_LINEAR)
    image = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, scaleup=False, stride=stride)(image=image)
    require(image.shape[:2] == (IMGSZ, IMGSZ), "calibration letterbox is not 640x640")
    return torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))


def calibration_batches(coco: Path, ids: Sequence[int], stride: int) -> list[Any]:
    import torch

    images = [square_input(image_file(coco, i).read_bytes(), stride) for i in ids]
    return [
        torch.stack(images[s : s + CALIB_BATCH]).float() / 255 for s in range(0, len(images), CALIB_BATCH)
    ]


def head(model: Any, batch: Any) -> Any:
    import torch

    with torch.no_grad():
        raw = model(batch)
    output = raw[0] if isinstance(raw, tuple) else raw
    require(output.ndim == 3 and output.shape[1] == 84, f"unexpected head output {tuple(output.shape)}")
    require(bool(torch.isfinite(output).all()), "nonfinite head output")
    return output


def measure_d(model: Any, batches: Sequence[Any], references: Sequence[Any]) -> float:
    import torch

    per_image = torch.cat(
        [distortion(ref, head(model, batch)) for batch, ref in zip(batches, references, strict=True)]
    )
    return float(per_image.mean().item())


def load_evaluator() -> Any:
    name = "evaluate_vision_quality"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "e2e" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


# ---------- stages ----------
def stage_table(args: argparse.Namespace) -> None:
    import torch

    setup_torch(args.threads)
    out = args.work / "table.json"
    require(not out.exists(), f"{out} exists; use a fresh --work")
    started = time.monotonic()
    model, names = load_model(args.checkpoint)
    layers, modules, int8, dfl = conv_layers(model)
    others, detail = other_params(model, modules)
    stride = int(model.stride.max().item())
    all_ids = annotation_ids(args.coco / "annotations" / "instances_val2017.json")
    calib = calibration_ids(all_ids)
    batches = calibration_batches(args.coco, calib, stride)
    references = [head(model, batch) for batch in batches]
    require(torch.equal(head(model, batches[0]), references[0]), "native forward is not deterministic")
    native = weights_digest(modules)
    print(
        f"YOLO_TABLE_START layers={len(layers)} conv_params={sum(x.params for x in layers)} other={others} "
        f"calib={len(calib)} threads={args.threads}",
        flush=True,
    )
    rows = []
    for index, layer in enumerate(layers):
        tick = time.monotonic()
        d = {}
        for method in QUANT_METHODS:
            with quantized(modules, {layer.name: method}):
                d[method] = measure_d(model, batches, references)
        require(all(math.isfinite(v) and v >= 0 for v in d.values()), f"{layer.name}: bad D {d}")
        rows.append({**asdict(layer), "d": d})
        print(
            f"YOLO_TABLE_LAYER {index + 1}/{len(layers)} {layer.name} params={layer.params} k={layer.k} "
            + " ".join(f"{m}={v:.4e}" for m, v in d.items())
            + f" t={time.monotonic() - tick:.1f}s",
            flush=True,
        )
    require(weights_digest(modules) == native, "native weights were not restored after the table")
    require(torch.equal(head(model, batches[0]), references[0]), "native output changed after the table")
    write_json(
        out,
        {
            "schema": TABLE_SCHEMA,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "imgsz": IMGSZ,
            "layers": rows,
            "other_params": others,
            "other_detail": detail,
            "dfl_conv": dfl,
            "heuristic_int8": int8,
            "names": names,
            "calibration": {
                "seed": SEED,
                "excluded": list(DEMO_IDS),
                "rule": CALIB_RULE,
                "preprocess": CALIB_PREPROCESS,
                "ids": calib,
            },
            "native_restored": True,
            "seconds": round(time.monotonic() - started, 1),
            "env": run_env(),
        },
    )
    print(
        f"YOLO_TABLE_DONE out={out} layers={len(rows)} seconds={time.monotonic() - started:.0f}", flush=True
    )


def table_layers(table: Mapping[str, Any]) -> tuple[list[ConvLayer], dict[str, dict[str, float]]]:
    layers = [ConvLayer(row["name"], row["params"], row["k"], row["out"]) for row in table["layers"]]
    return layers, {row["name"]: row["d"] for row in table["layers"]}


def stage_solve(args: argparse.Namespace) -> None:
    table_path = args.work / "table.json"
    out = args.work / "solutions.json"
    require(not out.exists(), f"{out} exists")
    table = json.loads(table_path.read_text())
    require(table["schema"] == TABLE_SCHEMA, "not a vision_tune table")
    layers, d = table_layers(table)
    params = sum(layer.params for layer in layers)
    solutions = {}
    for auto, reference in AUTO.items():
        target = avg_bits(layers, baseline_assignment(reference, layers, table["heuristic_int8"]))
        assignment, solution = solve(layers, d, target * params)
        bits = avg_bits(layers, assignment)
        require(bits <= target, f"{auto}: {bits} bits exceed the budget {target}")
        solutions[auto] = {
            "reference": reference,
            "budget_fraction": fraction_text(target),
            "bits_fraction": fraction_text(bits),
            "slack_fraction": fraction_text(target - bits),
            "slack_bits_total": fraction_text(target * params - solution.cost),
            "predicted_d": solution.value,
            "resolution": fraction_text(solution.resolution),
            "capacity_units": solution.capacity,
            "bound": fraction_text(solution.bound),
            "methods": assignment,
            "method_counts": method_counts(assignment),
        }
        print(
            f"YOLO_SOLVE {auto} budget={float(target):.6f} bits={float(bits):.6f} slack={target - bits} "
            f"predicted_d={solution.value:.4e} counts={method_counts(assignment)}",
            flush=True,
        )
    write_json(out, {"table_sha256": sha256_file(table_path), "solutions": solutions, "env": run_env()})
    print(f"YOLO_SOLVE_DONE out={out}", flush=True)


def stage_evaluate(args: argparse.Namespace) -> None:
    import torch
    from app.runners.vision import _detections, _network_input
    from torch.utils.data import DataLoader

    setup_torch(args.threads)
    started = time.monotonic()
    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    require(setups and set(setups) <= set(SETUPS), f"setups must be among {SETUPS}")
    for setup in setups:
        require(not (args.work / f"eval_{setup}.json").exists(), f"eval_{setup}.json exists")
    model, names = load_model(args.checkpoint)
    layers, modules, int8, _ = conv_layers(model)
    stride = int(model.stride.max().item())
    annotations = args.coco / "annotations" / "instances_val2017.json"
    all_ids = annotation_ids(annotations)
    calib = calibration_ids(all_ids)
    eval_ids = evaluation_ids(all_ids, calib)
    require(len(eval_ids) == COCO_IMAGES - CALIB_COUNT, "evaluation ids must be the other 4872")
    plans = {}
    if any(setup in AUTO for setup in setups):
        solutions = json.loads((args.work / "solutions.json").read_text())["solutions"]
        plans = {setup: solutions[setup]["methods"] for setup in setups if setup in AUTO}
    demo = []
    for image_id in DEMO_IDS:
        source = image_file(args.coco, image_id)
        media = args.media / f"coco_{image_id:012d}.jpg"
        digest = sha256_file(source)
        require(media.is_file() and sha256_file(media) == digest, f"{media} differs from {source}")
        batch, plan = _network_input(source.read_bytes(), IMGSZ, stride)
        demo.append(
            {
                "coco_id": image_id,
                "file": f"media/{media.name}",
                "width": plan.original[1],
                "height": plan.original[0],
                "sha256": digest,
                "batch": batch.float() / 255,
                "plan": plan,
            }
        )
    batches = calibration_batches(args.coco, calib, stride)
    references = [head(model, batch) for batch in batches]
    native = weights_digest(modules)
    evaluator = load_evaluator()
    dataset = evaluator.COCOValidation(
        args.coco / "val2017", annotations, args.work, EVAL["batch"], IMGSZ, stride, None
    )
    require(len(dataset) == COCO_IMAGES, "the evaluator dataset must hold all 5000 images")
    env = run_env()
    print(
        f"YOLO_EVAL_START setups={','.join(setups)} threads={args.threads} workers={args.workers}", flush=True
    )
    for setup in setups:
        tick = time.monotonic()
        assignment = plans[setup] if setup in AUTO else baseline_assignment(setup, layers, int8)
        require(set(assignment) == set(modules), f"{setup}: assignment does not cover the conv layers")
        predictions: list[dict] = []
        seen: list[int] = []
        with quantized(modules, assignment):
            measured = None if setup == "native" else measure_d(model, batches, references)
            loader = DataLoader(
                dataset,
                batch_size=EVAL["batch"],
                shuffle=False,
                num_workers=args.workers,
                drop_last=False,
                collate_fn=evaluator.coco_collate,
                generator=torch.Generator().manual_seed(SEED),
            )
            for index, (inputs, metadata) in enumerate(loader):
                x = inputs.float() / 255
                with torch.no_grad():
                    raw = model(x)
                predictions.extend(
                    evaluator.serialize_predictions(
                        raw,
                        metadata,
                        tuple(x.shape),
                        dataset.category_map,
                        EVAL["conf"],
                        EVAL["iou"],
                        EVAL["max_det"],
                    )
                )
                seen.extend(item["image_id"] for item in metadata)
                if (index + 1) % 125 == 0:
                    print(
                        f"YOLO_EVAL_PROGRESS {setup} images={len(seen)}/{COCO_IMAGES} "
                        f"t={time.monotonic() - tick:.0f}s",
                        flush=True,
                    )
            detections = {}
            for item in demo:
                boxes = _detections(
                    model, item["batch"], item["plan"], names, DRAW["conf"], DRAW["iou"], DRAW["max_det"]
                )
                detections[str(item["coco_id"])] = [
                    [*(round(v, 1) for v in box["xyxy"]), box["cls_id"], round(box["conf"], 3)]
                    for box in boxes
                ]
        require(weights_digest(modules) == native, f"{setup}: native weights were not restored")
        require(sorted(seen) == all_ids, f"{setup}: inference skipped or repeated images")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            metrics = evaluator.evaluate_coco(
                annotations, predictions, eval_ids, args.work / f"pred_{setup}.json"
            )
            full = (
                evaluator.evaluate_coco(annotations, predictions, all_ids, args.work / f"pred_{setup}.json")
                if setup == "native"
                else None
            )
        (args.work / f"cocoeval_{setup}.txt").write_text(log.getvalue())
        result = {
            "setup": setup,
            "methods": assignment,
            "measured_d": measured,
            "metrics": {k: metrics[k] for k in METRIC_KEYS},
            "metrics_detail": metrics,
            "metrics_5000": None if full is None else {k: full[k] for k in ("AP_50_95", "AP_50")},
            "detections": detections,
            "names": names,
            "images": [
                {k: item[k] for k in ("coco_id", "file", "width", "height", "sha256")} for item in demo
            ],
            "calibration_ids": calib,
            "evaluation_images": len(eval_ids),
            "seconds": round(time.monotonic() - tick, 1),
            "env": {**env, "stage": "evaluate"},
        }
        write_json(args.work / f"eval_{setup}.json", result)
        extra = "" if full is None else f" AP_50_95_all5000={full['AP_50_95']:.4f}"
        print(
            f"YOLO_EVAL_SETUP {setup} AP_50_95={metrics['AP_50_95']:.4f} AP_50={metrics['AP_50']:.4f}{extra} "
            f"measured_d={measured} preds={len(predictions)} t={time.monotonic() - tick:.0f}s",
            flush=True,
        )
    print(f"YOLO_EVAL_DONE setups={','.join(setups)} seconds={time.monotonic() - started:.0f}", flush=True)


def eval_metrics(setup: str, result: Mapping[str, Any]) -> dict[str, float]:
    """METRIC_KEYS of an evaluation record: an older record keeps AP50-95 / AP50 in ``metrics`` and the rest
    of the same COCOeval run in ``metrics_detail``; both must agree where they overlap."""
    detail = result.get("metrics_detail") or {}
    metrics = {k: result["metrics"].get(k, detail.get(k)) for k in METRIC_KEYS}
    require(all(isinstance(v, (int, float)) for v in metrics.values()), f"{setup}: no {METRIC_KEYS}")
    require(all(detail.get(k, v) == v for k, v in metrics.items()), f"{setup}: metrics != metrics_detail")
    return metrics


def stage_export(args: argparse.Namespace) -> None:
    table = json.loads((args.work / "table.json").read_text())
    solved = json.loads((args.work / "solutions.json").read_text())
    require(
        solved["table_sha256"] == sha256_file(args.work / "table.json"), "solutions are from another table"
    )
    evals = {}
    for setup in SETUPS:
        path = next(
            (
                d / f"eval_{setup}.json"
                for d in (args.work, *args.more)
                if (d / f"eval_{setup}.json").exists()
            ),
            None,
        )
        require(path is not None, f"eval_{setup}.json not found")
        evals[setup] = json.loads(path.read_text())
    layers, d = table_layers(table)
    others = table["other_params"]
    for setup, result in evals.items():
        require(result["calibration_ids"] == table["calibration"]["ids"], f"{setup}: other calibration ids")
        require(result["names"] == table["names"], f"{setup}: other class names")
    native = evals["native"]
    setups = []
    for setup in SETUPS[1:]:
        result = evals[setup]
        assignment = result["methods"]
        bits = avg_bits(layers, assignment)
        entry = {
            "id": setup,
            "kind": "auto" if setup in AUTO else "uniform" if setup in UNIFORM else "heuristic",
            "label": LABELS[setup],
            "bits": float(bits),
            "bits_fraction": fraction_text(bits),
            "bytes": model_bytes(layers, assignment, others),
            "methods": assignment,
            "method_counts": method_counts(assignment),
            "predicted_d": predicted_d(assignment, d),
            "measured_d": result["measured_d"],
            "metrics": eval_metrics(setup, result),
            "env": result["env"],
        }
        if setup in AUTO:
            solution = solved["solutions"][setup]
            require(solution["methods"] == assignment, f"{setup}: evaluated methods differ from the solution")
            # the exact budget the knapsack was given (the reference setup's average bits); bits <= it
            entry["budget_bits"] = solution["budget_fraction"]
        setups.append(entry)
    data = {
        "schema": SCHEMA,
        "model": {
            "id": "yolo11n",
            "label": "YOLO11n",
            "checkpoint_sha256": table["checkpoint_sha256"],
            "ultralytics": ULTRALYTICS,
            "imgsz": IMGSZ,
            "conv_layers": len(layers),
            "conv_params": sum(layer.params for layer in layers),
            "other_params": others,
            "original_bytes": original_bytes(layers, others),
            "size_rule": SIZE_RULE,
        },
        "quantization": QUANTIZATION,
        "objective": OBJECTIVE,
        "calibration": {**table["calibration"], "env": table["env"]},
        "evaluation": {
            "images": native["evaluation_images"],
            "rule": EVAL_RULE,
            "metric": "COCO bbox AP50-95 / AP50 / AP75 / AP small · medium · large (pycocotools, one run)",
            "conf": EVAL["conf"],
            "iou": EVAL["iou"],
            "max_det": EVAL["max_det"],
        },
        "methods": [{"id": method, "bits": bits} for method, bits in METHODS.items()],
        "layers": [
            {"name": layer.name, "params": layer.params, "k": layer.k, "out": layer.out, "d": d[layer.name]}
            for layer in layers
        ],
        "setups": setups,
        "native": {
            "metrics": eval_metrics("native", native),
            "metrics_5000": native["metrics_5000"],
            "bytes": original_bytes(layers, others),
            "env": native["env"],
        },
        "names": table["names"],
        "images": [
            {
                **{k: image[k] for k in ("coco_id", "file", "width", "height")},
                "detections": {setup: evals[setup]["detections"][str(image["coco_id"])] for setup in SETUPS},
            }
            for image in native["images"]
        ],
        "env": run_env(),
    }
    problems = validate(data)
    require(not problems, "exported dataset is inconsistent: " + "; ".join(problems))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, data)
    print(f"YOLO_EXPORT_DONE out={args.out} bytes={args.out.stat().st_size}", flush=True)


# ---------- validation (also used by app/tests/test_yolo_tune.py) ----------
TOP_KEYS = (
    "schema",
    "model",
    "quantization",
    "objective",
    "calibration",
    "evaluation",
    "methods",
    "layers",
    "setups",
    "native",
    "names",
    "images",
    "env",
)
SETUP_KEYS = (
    "id",
    "kind",
    "label",
    "bits",
    "bits_fraction",
    "bytes",
    "methods",
    "method_counts",
    "predicted_d",
    "measured_d",
    "metrics",
    "env",
)
ENV_KEYS = ("git_sha", "git_dirty", "script_sha256", "host", "torch", "ultralytics", "cpu", "utc")


def validate(data: Mapping[str, Any]) -> list[str]:
    """Schema and internal-consistency problems of an exported dataset (empty = consistent)."""
    problems = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    check(tuple(data) == TOP_KEYS and data["schema"] == SCHEMA, f"top-level keys {tuple(data)}")
    if problems:
        return problems
    layers = [ConvLayer(row["name"], row["params"], row["k"], row["out"]) for row in data["layers"]]
    model = data["model"]
    check(model["conv_layers"] == len(layers), "model.conv_layers != len(layers)")
    check(model["conv_params"] == sum(layer.params for layer in layers), "model.conv_params mismatch")
    check(model["original_bytes"] == original_bytes(layers, model["other_params"]), "original_bytes mismatch")
    check(all(row["params"] == row["k"] * row["out"] for row in data["layers"]), "params != k * out")
    check(
        all(
            set(row["d"]) == set(QUANT_METHODS) and all(v >= 0 for v in row["d"].values())
            for row in data["layers"]
        ),
        "layer d must hold a non-negative D per int method",
    )
    check([m["id"] for m in data["methods"]] == list(METHODS), "methods list")
    calibration, evaluation = data["calibration"], data["evaluation"]
    check(
        calibration["ids"] == sorted(set(calibration["ids"]))
        and len(calibration["ids"]) == CALIB_COUNT
        and not set(calibration["ids"]) & set(DEMO_IDS),
        "calibration ids",
    )
    check(calibration["seed"] == SEED and calibration["excluded"] == list(DEMO_IDS), "calibration rule")
    check(evaluation["images"] == COCO_IMAGES - CALIB_COUNT, "evaluation image count")
    check(len(data["names"]) == 80, "80 class names")
    for env in [
        data["env"],
        data["native"]["env"],
        calibration.get("env", {}),
        *(setup.get("env", {}) for setup in data["setups"]),
    ]:
        check(all(key in env for key in ENV_KEYS), f"env keys missing in {sorted(env)}")
    native = data["native"]
    check(
        native["bytes"] == model["original_bytes"]
        and all(math.isfinite(native["metrics"][k]) for k in METRIC_KEYS),
        "native",
    )
    by_id = {setup["id"]: setup for setup in data["setups"]}
    check(tuple(by_id) == SETUPS[1:], f"setups {tuple(by_id)}")
    names = {layer.name for layer in layers}
    for setup in data["setups"]:
        sid = setup["id"]
        keys = SETUP_KEYS + (("budget_bits",) if sid in AUTO else ())
        check(tuple(setup) == keys, f"{sid}: keys {tuple(setup)}")
        assignment = setup["methods"]
        check(set(assignment) == names and set(assignment.values()) <= set(METHODS), f"{sid}: methods")
        if set(assignment) != names:
            continue
        bits = avg_bits(layers, assignment)
        check(parse_fraction(setup["bits_fraction"]) == bits, f"{sid}: bits_fraction != recomputed bits")
        check(abs(setup["bits"] - float(bits)) < 1e-12, f"{sid}: bits")
        check(setup["bytes"] == model_bytes(layers, assignment, model["other_params"]), f"{sid}: bytes")
        check(setup["method_counts"] == method_counts(assignment), f"{sid}: method_counts")
        check(all(math.isfinite(setup["metrics"][k]) for k in METRIC_KEYS), f"{sid}: metrics")
        check(setup["measured_d"] is not None and setup["measured_d"] >= 0, f"{sid}: measured_d")
        d = {row["name"]: row["d"] for row in data["layers"]}
        check(
            abs(setup["predicted_d"] - predicted_d(assignment, d)) <= 1e-9 * max(1.0, setup["predicted_d"]),
            f"{sid}: predicted_d is not the table sum",
        )
        if sid in AUTO:
            reference = by_id.get(AUTO[sid])
            if reference is not None and "budget_bits" in setup:
                budget = parse_fraction(setup["budget_bits"])
                check(
                    budget == parse_fraction(reference["bits_fraction"]), f"{sid}: budget != reference bits"
                )
                check(bits <= budget, f"{sid}: bits exceed the budget")
    for image in data["images"]:
        check(set(image["detections"]) == set(SETUPS), f"image {image.get('coco_id')}: detection setups")
        for boxes in image["detections"].values():
            check(
                all(
                    len(box) == 6
                    and 0 <= box[4] < 80
                    and 0.25 <= box[5] <= 1
                    and box[0] <= box[2]
                    and box[1] <= box[3]
                    and box[2] <= image["width"]
                    and box[3] <= image["height"]
                    for box in boxes
                )
                and len(boxes) <= DRAW["max_det"],
                f"image {image['coco_id']}: boxes",
            )
    check([image["coco_id"] for image in data["images"]] == list(DEMO_IDS), "demo images")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("table", "solve", "evaluate", "export"):
        stage = sub.add_parser(name)
        stage.add_argument(
            "--work", type=Path, required=True, help="directory of the intermediate JSON files"
        )
        if name in ("table", "evaluate"):
            stage.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
            stage.add_argument("--coco", type=Path, default=COCO, help="holds val2017/ and annotations/")
            stage.add_argument("--threads", type=int, default=16)
        if name == "evaluate":
            stage.add_argument("--setups", required=True, help=f"comma list of {','.join(SETUPS)}")
            stage.add_argument("--workers", type=int, default=2)
            stage.add_argument("--media", type=Path, default=ROOT / "app" / "web" / "demo" / "media")
        if name == "export":
            stage.add_argument(
                "--more", type=Path, nargs="*", default=[], help="other work dirs with eval_*.json"
            )
            stage.add_argument(
                "--out", type=Path, default=ROOT / "app" / "web" / "demo" / "autotune" / "yolo11n.json"
            )
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    stages = {"table": stage_table, "solve": stage_solve, "evaluate": stage_evaluate, "export": stage_export}
    try:
        stages[args.stage](args)
    except Exception as error:
        traceback.print_exc()
        print(f"YOLO_FAIL stage={args.stage} error={type(error).__name__}: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
