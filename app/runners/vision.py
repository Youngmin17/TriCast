"""Vision runners: YOLO11n detection and ResNet18 classification, baseline vs TriCast-emulated
(every Conv2d and Linear patched, ``include_conv2d=True``).

Detection follows scripts/e2e/evaluate_vision_quality.py: the raw PyTorch model loaded from the pinned
checkpoint and fused *before* copying, Ultralytics 8.3.221 rectangular-validation letterbox (stride
padding 0.5, no upscaling in the letterbox, /255) and its class-aware multi-label NMS. The patched copy
never goes through ``predict``/``val``/AutoBackend, which can fuse away emulated layers. Checkpoints are
local files only (no downloads); ultralytics, torchvision, OpenCV and Pillow are imported on first use.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .. import metrics
from ..catalog import MODELS
from .common import (
    CHECKPOINT_ENV,
    DETECTION,
    TOP_K,
    BaselineCache,
    Progress,
    checkpoint_path,
    emulated,
    file_sha256,
    now,
    recipe_key,
    recipe_label,
    report,
    require_finite,
    resolve_backend,
)

ULTRALYTICS_VERSION = "8.3.221"
# An entry is one image's baseline result: at most 100 boxes, or 1,000 class probabilities (8 KB).
BASELINES = BaselineCache()
_MODELS: dict[tuple[str, str], tuple[nn.Module, list[str]]] = {}


@dataclass(frozen=True)
class Letterbox:
    """Ultralytics rectangular-validation geometry for one image (a batch of one)."""

    original: tuple[int, int]
    resized: tuple[int, int]
    target: tuple[int, int]
    pad: tuple[int, int]
    gain: float


def letterbox(height: int, width: int, imgsz: int, stride: int) -> Letterbox:
    """Long side resized to ``imgsz`` (ceil, only when it differs), network input padded to the stride-rounded
    rectangle ``ceil(shape * imgsz / stride + 0.5) * stride``; ``pad`` is (left, top), ``gain`` = h / h0."""
    ratio = imgsz / max(height, width)
    resized = (height, width) if ratio == 1 else (min(math.ceil(height * ratio), imgsz),
                                                  min(math.ceil(width * ratio), imgsz))
    aspect = height / width
    shape = (aspect, 1.0) if aspect < 1 else (1.0, 1 / aspect) if aspect > 1 else (1.0, 1.0)
    target = tuple(math.ceil(side * imgsz / stride + 0.5) * stride for side in shape)
    pad = (round((target[1] - resized[1]) / 2 - 0.1), round((target[0] - resized[0]) / 2 - 0.1))
    return Letterbox((height, width), resized, target, pad, resized[0] / height)


def unletterbox(boxes: torch.Tensor, plan: Letterbox) -> torch.Tensor:
    """xyxy boxes from network-input to original pixels, clipped to the image: the same arithmetic as
    ``ultralytics.utils.ops.scale_boxes(..., ratio_pad=((gain, gain), pad))``."""
    boxes = boxes.clone()
    boxes[..., [0, 2]] -= plan.pad[0]
    boxes[..., [1, 3]] -= plan.pad[1]
    boxes[..., :4] /= plan.gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, plan.original[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, plan.original[0])
    return boxes


def _verify(model_id: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{model_id} 체크포인트가 없습니다: {path}. {CHECKPOINT_ENV[model_id]} 로 "
                                "경로를 지정하세요 (자동 다운로드하지 않음).")
    digest = file_sha256(path)
    if digest != MODELS[model_id].revision:
        raise ValueError(f"{model_id} 체크포인트 SHA256 이 고정값과 다릅니다: {path} ({digest}).")


def load_detector(device: str, checkpoint: str | Path | None = None) -> tuple[nn.Module, list[str]]:
    """The fused, unpatched YOLO11n (fp32, eval) and its class names, cached per checkpoint and device."""
    path = checkpoint_path("yolo11n", checkpoint)
    key = (str(path), str(device))
    if key not in _MODELS:
        _verify("yolo11n", path)
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        if version("ultralytics") != ULTRALYTICS_VERSION:
            raise RuntimeError(f"YOLO 검출은 ultralytics=={ULTRALYTICS_VERSION} 에서만 검증되었습니다 "
                               f"(설치: {version('ultralytics')}).")
        from ultralytics.nn.tasks import DetectionModel, load_checkpoint

        model, _ = load_checkpoint(str(path), device="cpu", fuse=False)
        if (not isinstance(model, DetectionModel) or model.model[-1].nc != 80
                or getattr(model, "end2end", False)):
            raise ValueError(f"공식 80-클래스 YOLO 검출 체크포인트가 아닙니다: {path}")
        model.fuse(verbose=False)
        for module in model.modules():
            if hasattr(module, "export"):
                module.export = False
        model = model.eval().float().to(device)
        _MODELS[key] = (model, [model.names[index] for index in range(len(model.names))])
    return _MODELS[key]


def load_classifier(device: str, checkpoint: str | Path | None = None) -> tuple[nn.Module, list[str]]:
    """ResNet18 with the IMAGENET1K_V1 state dict from the local checkpoint, and the ImageNet class names."""
    path = checkpoint_path("resnet18", checkpoint)
    key = (str(path), str(device))
    if key not in _MODELS:
        _verify("resnet18", path)
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        categories = list(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
        _MODELS[key] = (model.eval().float().to(device), categories)
    return _MODELS[key]


def _network_input(data: bytes, imgsz: int, stride: int) -> tuple[torch.Tensor, Letterbox]:
    import cv2
    from ultralytics.data.augment import LetterBox

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지를 해독할 수 없습니다.")
    plan = letterbox(*image.shape[:2], imgsz, stride)
    if plan.resized != plan.original:
        image = cv2.resize(image, plan.resized[::-1], interpolation=cv2.INTER_LINEAR)
    image = LetterBox(new_shape=plan.target, scaleup=False, stride=stride)(image=image)
    return torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))[None], plan


def _detections(model: nn.Module, batch: torch.Tensor, plan: Letterbox, names: list[str], conf: float,
                iou: float, max_det: int) -> list[dict]:
    from ultralytics.utils.nms import non_max_suppression

    with torch.no_grad():
        raw = model(batch)
    prediction = require_finite(raw[0] if isinstance(raw, tuple) else raw, "YOLO 출력")
    # Ultralytics' wall-clock NMS timeout can silently drop boxes; infinity disables only that cutoff.
    kept = non_max_suppression(prediction.clone(), conf_thres=conf, iou_thres=iou, multi_label=True,
                               agnostic=False, max_det=max_det, nc=len(names), max_time_img=float("inf"))[0]
    boxes = unletterbox(kept[:, :4], plan)
    return [{"cls": names[int(cls)], "cls_id": int(cls), "conf": score, "xyxy": box}
            for box, score, cls in zip(boxes.tolist(), kept[:, 4].tolist(), kept[:, 5].tolist(), strict=True)]


@dataclass(frozen=True)
class _Baseline:
    label: str
    value: Any
    seconds: float


def _compare(native: nn.Module, recipe: Any, baseline_recipe: Any | None, key: tuple, device: str,
             backend: str | None, progress: Progress | None, compute: Callable[[nn.Module], Any],
             output: Callable[[str, Any], dict]) -> tuple[_Baseline, Any, float, dict, bool]:
    """Baseline (cached per model, input and baseline recipe) and emulated results of ``compute``; each
    patched model is a deep copy of the fused native one."""
    backend = resolve_backend(device, backend)
    baseline = BASELINES.get(native, key)
    cached = baseline is not None
    if not cached:
        report(progress, "baseline", 0.05)
        start = now(device)
        if baseline_recipe is None:
            value = compute(native)
        else:
            model = copy.deepcopy(native)
            with emulated(model, baseline_recipe, backend, include_conv2d=True):
                value = compute(model)
        baseline = _Baseline(recipe_label(baseline_recipe), value, now(device) - start)
        BASELINES.put(native, key, baseline)
    report(progress, "baseline", 0.4, partial={"baseline": output(baseline.label, baseline.value)})
    report(progress, "patch", 0.42)
    start = now(device)
    model = copy.deepcopy(native)
    with emulated(model, recipe, backend, include_conv2d=True) as evidence:
        report(progress, "emulated", 0.45)
        value = compute(model)
    seconds = now(device) - start
    report(progress, "metrics", 0.95)
    return baseline, value, seconds, evidence, cached


def run_detect(model_id: str, recipe: Any, baseline_recipe: Any | None, image_path: str | Path, device: str,
               backend: str | None = None, progress: Progress | None = None,
               conf: float = DETECTION["conf"], iou: float = DETECTION["iou"],
               imgsz: int = DETECTION["imgsz"], max_det: int = DETECTION["max_det"],
               checkpoint: str | Path | None = None) -> dict:
    """Detect objects in one image with the baseline (``None`` = native) and the emulated model; boxes are
    in original image pixels and ``image.media`` is the image's file name. ``checkpoint`` overrides
    :func:`checkpoint_path` (its SHA256 must still match the catalog revision)."""
    if model_id != "yolo11n":
        raise ValueError(f"검출 모델이 아닙니다: {model_id!r}")
    report(progress, "load", 0.0)
    native, names = load_detector(device, checkpoint)
    data = Path(image_path).read_bytes()
    batch, plan = _network_input(data, imgsz, int(native.stride.max().item()))
    batch = batch.to(device).float() / 255
    picture = {"media": Path(image_path).name, "width": plan.original[1], "height": plan.original[0]}

    def compute(model: nn.Module) -> list[dict]:
        return _detections(model, batch, plan, names, conf, iou, max_det)

    def output(label: str, boxes: list[dict]) -> dict:
        return {"label": label, "image": picture, "boxes": boxes}

    key = ("detect", hashlib.sha256(data).hexdigest(), conf, iou, imgsz, max_det, recipe_key(baseline_recipe))
    baseline, boxes, seconds, evidence, cached = _compare(native, recipe, baseline_recipe, key, device,
                                                          backend, progress, compute, output)
    return {"baseline": output(baseline.label, baseline.value),
            "emulated": output(recipe_label(recipe), boxes),
            "metrics": metrics.match_boxes(baseline.value, boxes),
            "timing": {"baseline_s": baseline.seconds, "emulated_s": seconds}, "evidence": evidence,
            "cached_baseline": cached}


def run_classify(model_id: str, recipe: Any, baseline_recipe: Any | None, image_path: str | Path, device: str,
                 backend: str | None = None, progress: Progress | None = None,
                 checkpoint: str | Path | None = None) -> dict:
    """Classify one image (torchvision IMAGENET1K_V1 preprocessing: resize 256, center crop 224, normalize)
    with the baseline (``None`` = native) and the emulated model; top-5 softmax probabilities.
    ``checkpoint`` overrides :func:`checkpoint_path` (its SHA256 must still match the catalog revision)."""
    if model_id != "resnet18":
        raise ValueError(f"분류 모델이 아닙니다: {model_id!r}")
    report(progress, "load", 0.0)
    native, categories = load_classifier(device, checkpoint)
    from PIL import Image
    from torchvision.models import ResNet18_Weights

    data = Path(image_path).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
    batch = ResNet18_Weights.IMAGENET1K_V1.transforms()(rgb).unsqueeze(0).to(device)
    picture = {"media": Path(image_path).name, "width": rgb.width, "height": rgb.height}

    def compute(model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            logits = require_finite(model(batch)[0], "ResNet18 logits")
        return torch.softmax(logits.double(), dim=-1).cpu()

    def output(label: str, probs: torch.Tensor) -> dict:
        order = torch.argsort(probs, descending=True, stable=True)[:TOP_K].tolist()
        return {"label": label, "image": picture,
                "top": [{"label": categories[index], "class_id": index, "p": probs[index].item()}
                        for index in order]}

    key = ("classify", hashlib.sha256(data).hexdigest(), recipe_key(baseline_recipe))
    baseline, probs, seconds, evidence, cached = _compare(native, recipe, baseline_recipe, key, device,
                                                          backend, progress, compute, output)
    return {"baseline": output(baseline.label, baseline.value),
            "emulated": output(recipe_label(recipe), probs),
            "metrics": metrics.classify_metrics(baseline.value, probs),
            "timing": {"baseline_s": baseline.seconds, "emulated_s": seconds}, "evidence": evidence,
            "cached_baseline": cached}
