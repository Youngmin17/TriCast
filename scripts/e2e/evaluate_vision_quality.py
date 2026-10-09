"""Full, paired pretrained vision quality evaluation on explicitly local assets.

No model/dataset downloads. Run on an isolated UCL sm_80+ GPU. ResNet18 uses
TorchVision ImageNet-1k V1 preprocessing and WNID-organized validation labels.
YOLO11n uses the pinned Ultralytics 8.3.221 raw PyTorch model, validation image
preprocessing/NMS and the official COCO bbox evaluator. Never call .val() or
AutoBackend on the patched model: those paths may fuse away its emulated layers.

Default: the entire 50,000-image ImageNet or 5,000-image COCO2017 validation
split. --limit is diagnostic only; its result cannot satisfy the full gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from typing import Any, TextIO
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tricast import load_recipe, patch_model
from tricast.eval.envinfo import capture_env
from tricast.nn.linear import EmuLinear
from tricast.nn.patch import iter_emuconv2d, iter_emulinear, unpatch_model

ULTRALYTICS_VERSION = "8.3.221"
FULL_COUNTS = {"resnet": 50_000, "yolo": 5_000}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((list(value.shape), str(value.dtype))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def output_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in output_tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in output_tensors(item)]
    return []


def finite(value: Any, label: str) -> None:
    values = output_tensors(value)
    require(bool(values), f"{label}: no tensors")
    require(all(bool(torch.isfinite(tensor).all()) for tensor in values), f"{label}: NaN/Inf output")


def assert_bits(left: Any, right: Any, label: str) -> None:
    a, b = output_tensors(left), output_tensors(right)
    require(len(a) == len(b) and bool(a), f"{label}: output structure changed")
    for x, y in zip(a, b, strict=True):
        require(x.shape == y.shape and x.dtype == y.dtype, f"{label}: shape/dtype changed")
        require(tensor_hash(x) == tensor_hash(y), f"{label}: output bits changed")


class Tee:
    """Keep normal progress output and per-run logs without shell redirection."""

    def __init__(self, stream: TextIO, path: Path) -> None:
        self.stream = stream
        self.file = path.open("a", buffering=1)

    def write(self, message: str) -> int:
        self.file.write(message)
        return self.stream.write(message)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def isatty(self) -> bool:
        return False


class DispatchWitness:
    """Require every selected layer to reach its selected Triton MMA each batch."""

    def __init__(self, model: nn.Module) -> None:
        self.expected = {layer.name: layer.spec.mma.algorithm for _, layer in iter_emulinear(model)}
        self.allowed = {name: {algorithm} for name, algorithm in self.expected.items()}
        for _, conv in iter_emuconv2d(model):
            for group in range(conv.groups):
                name = f"{conv.name}.group{group}"
                self.expected[name] = conv.spec.mma.algorithm
                self.allowed[name] = {conv.spec.mma.algorithm}
                if conv.spec.outliers is not None:
                    self.allowed[name].add("fp32_fma")
        for _, layer in iter_emulinear(model):
            if layer.spec.outliers is not None:
                self.allowed[layer.name].add("fp32_fma")
        require(bool(self.expected), "recipe selected no emulated arithmetic")
        self.layers: Counter[str] = Counter()
        self.gemms: dict[str, Counter[str]] = {name: Counter() for name in self.expected}
        self.kernel_calls: Counter[str] = Counter()
        self.active = ""
        self.stack = ExitStack()
        self.checked_batches = 0

    def __enter__(self) -> DispatchWitness:
        from tricast.kernels import mma as kernels
        from tricast.nn import linear

        original_matmul, original_gemm, original_kernel = EmuLinear._matmul, linear.gemm, kernels.gemm_triton

        def matmul(layer: EmuLinear, *args: Any, **kwargs: Any) -> torch.Tensor:
            require(layer.name in self.expected, f"untracked emulated layer {layer.name}")
            self.layers[layer.name] += 1
            previous, self.active = self.active, layer.name
            try:
                return original_matmul(layer, *args, **kwargs)
            finally:
                self.active = previous

        def gemm(a: Any, b: Any, spec: Any, **kwargs: Any) -> torch.Tensor:
            require(self.active in self.expected, "GEMM outside a tracked emulated layer")
            require(kwargs.get("backend") == "triton", f"{self.active}: non-Triton backend")
            self.gemms[self.active][spec.algorithm] += 1
            return original_gemm(a, b, spec, **kwargs)

        def kernel(*args: Any, **kwargs: Any) -> torch.Tensor:
            self.kernel_calls[self.active] += 1
            return original_kernel(*args, **kwargs)

        self.stack.enter_context(patch.object(EmuLinear, "_matmul", matmul))
        self.stack.enter_context(patch.object(linear, "gemm", gemm))
        self.stack.enter_context(patch.object(kernels, "gemm_triton", kernel))
        return self

    def __exit__(self, *args: Any) -> None:
        self.stack.__exit__(*args)

    def snapshot(self) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, int]]:
        return (dict(self.layers), {name: dict(counts) for name, counts in self.gemms.items()},
                dict(self.kernel_calls))

    def check_batch(self, before: tuple[dict[str, int], dict[str, dict[str, int]], dict[str, int]]) -> None:
        before_layers, before_gemms, before_kernels = before
        for name, algorithm in self.expected.items():
            layer_calls = self.layers[name] - before_layers.get(name, 0)
            gemm_calls = {selected: count - before_gemms.get(name, {}).get(selected, 0)
                          for selected, count in self.gemms[name].items()}
            kernel_calls = self.kernel_calls[name] - before_kernels.get(name, 0)
            require(layer_calls > 0, f"bypass: {name} not called in current batch")
            require(gemm_calls.get(algorithm, 0) >= layer_calls,
                    f"bypass: {name} lacked selected {algorithm} MMA dispatches in current batch")
            require(self.gemms[name].keys() <= self.allowed[name], f"{name}: unexpected MMA algorithm")
            require(kernel_calls == sum(gemm_calls.values()),
                    f"bypass: {name} GEMM did not execute Triton kernel dispatch in current batch")
        self.checked_batches += 1

    def summary(self) -> dict:
        require(self.checked_batches > 0, "no batches validated")
        return {"bypass_detected": False, "checked_batches": self.checked_batches,
                "selected_layer_algorithms": self.expected, "layer_calls": dict(self.layers),
                "gemm_calls": {name: dict(counts) for name, counts in self.gemms.items()},
                "triton_kernel_dispatch_calls": dict(self.kernel_calls)}


def manifest(records: list[dict], root: Path, out: Path) -> str:
    """Hash file contents, assigned labels/IDs and ordered paths, not just timestamps."""
    digest = hashlib.sha256()
    with out.open("w") as stream:
        for index, record in enumerate(records):
            path = root / record["file"]
            require(path.is_file(), f"missing dataset image: {path}")
            record.update(bytes=path.stat().st_size, sha256=sha256_file(path))
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode())
            stream.write(line)
            if (index + 1) % 5_000 == 0:
                print(f"MANIFEST_PROGRESS images={index + 1}", flush=True)
    return digest.hexdigest()


class ImageNetValidation(Dataset):
    def __init__(self, root: Path, out: Path, limit: int | None) -> None:
        from torchvision.datasets import ImageFolder
        from torchvision.models import ResNet18_Weights

        folder = ImageFolder(root)
        require(len(folder.classes) == 1_000 and all(re.fullmatch(r"n\d{8}", cls) for cls in folder.classes),
                "ImageNet val root must contain 1,000 official WNID class folders")
        records = [{"file": Path(path).relative_to(root).as_posix(), "target": target,
                    "wnid": folder.classes[target]} for path, target in folder.samples]
        counts = Counter(record["target"] for record in records)
        require(len(records) == FULL_COUNTS["resnet"] and set(counts.values()) == {50},
                "full ImageNet-1k val must contain 50,000 images, 50 per class (no partial source split)")
        filenames = [Path(record["file"]).name for record in records]
        require(len(set(filenames)) == len(filenames) and
                all(re.fullmatch(r"ILSVRC2012_val_\d{8}\.JPEG", name) for name in filenames),
                "ImageNet validation filenames must be unique official ILSVRC2012_val names")
        self.root, self.records = root, records if limit is None else records[:limit]
        self.manifest_sha256 = manifest(self.records, root, out / "dataset_manifest.jsonl")
        self.class_index_sha256 = hashlib.sha256(
            json.dumps(folder.class_to_idx, sort_keys=True).encode(),
        ).hexdigest()
        (out / "class_index.json").write_text(json.dumps(folder.class_to_idx, indent=2) + "\n")
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        from PIL import Image

        record = self.records[index]
        data = (self.root / record["file"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == record["sha256"], "ImageNet bytes changed after manifest")
        with Image.open(io.BytesIO(data)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record["target"], index


class COCOValidation(Dataset):
    def __init__(self, root: Path, annotations: Path, out: Path, batch: int, imgsz: int,
                 stride: int, limit: int | None) -> None:
        from ultralytics.data.converter import coco80_to_coco91_class

        annotation = json.loads(annotations.read_text())
        images = sorted(annotation["images"], key=lambda item: item["file_name"])
        require(len(images) == FULL_COUNTS["yolo"] and len({item["id"] for item in images}) == len(images),
                "COCO2017 val annotation must contain 5,000 unique images")
        require(sorted(item["id"] for item in annotation["categories"]) == sorted(coco80_to_coco91_class()),
                "COCO annotation category IDs do not match the official 80-class mapping")
        if limit is not None:
            images = images[:limit]
        self.category_map = coco80_to_coco91_class()
        self.root, self.imgsz, self.stride = root, imgsz, stride
        # Match Ultralytics rectangular validation batch planning: filename order,
        # NumPy's default aspect-ratio sort, stride rounding and validation pad=0.5.
        aspect = np.array([item["height"] / item["width"] for item in images])
        order = aspect.argsort()
        images, aspect = [images[index] for index in order], aspect[order]
        self.records = [{"file": item["file_name"], "image_id": item["id"],
                         "height": item["height"], "width": item["width"]} for item in images]
        self.shapes = []
        for start in range(0, len(images), batch):
            ratios = aspect[start:start + batch]
            shape = ([float(ratios.max()), 1.0] if ratios.max() < 1 else
                     [1.0, float(1 / ratios.min())] if ratios.min() > 1 else [1.0, 1.0])
            target = (np.ceil(np.array(shape) * imgsz / stride + 0.5).astype(int) * stride).tolist()
            self.shapes.extend([target] * len(ratios))
        self.manifest_sha256 = manifest(self.records, root, out / "dataset_manifest.jsonl")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        import cv2
        from ultralytics.data.augment import LetterBox

        record = self.records[index]
        data = (self.root / record["file"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == record["sha256"], "COCO bytes changed after manifest")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(image is not None, f"cannot decode {record['file']}")
        h0, w0 = image.shape[:2]
        require((h0, w0) == (record["height"], record["width"]), "COCO image/annotation dimensions differ")
        ratio = self.imgsz / max(h0, w0)
        if ratio != 1:
            w, h = min(math.ceil(w0 * ratio), self.imgsz), min(math.ceil(h0 * ratio), self.imgsz)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]
        target_h, target_w = self.shapes[index]
        require(h <= target_h and w <= target_w, "rectangular shape would downscale the pre-resized image")
        left, top = round((target_w - w) / 2 - 0.1), round((target_h - h) / 2 - 0.1)
        image = LetterBox(new_shape=(target_h, target_w), scaleup=False, stride=self.stride)(image=image)
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
        return tensor, {"image_id": record["image_id"], "file": record["file"],
                        "original_shape": (h0, w0), "ratio_pad": ((h / h0, w / w0), (left, top))}


def coco_collate(items: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    return torch.stack([item[0] for item in items]), [item[1] for item in items]


def serialize_predictions(raw: Any, metadata: list[dict], input_shape: tuple[int, ...],
                          category_map: list[int], conf: float, iou: float, max_det: int) -> list[dict]:
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes, xyxy2xywh

    prediction = raw[0] if isinstance(raw, tuple) else raw
    finite(prediction, "decoded YOLO outputs")
    # Upstream NMS has a wall-clock timeout that can silently drop remaining
    # images. Infinity disables only that time cutoff, not confidence/IoU gates.
    outputs = non_max_suppression(prediction.clone(), conf_thres=conf, iou_thres=iou, multi_label=True,
                                  agnostic=False, max_det=max_det, nc=80, max_time_img=float("inf"))
    require(len(outputs) == len(metadata), "NMS returned incorrect batch length")
    results = []
    for boxes, item in zip(outputs, metadata, strict=True):
        finite(boxes, "post-NMS boxes")
        if not boxes.numel():
            continue
        boxes[:, :4] = scale_boxes(input_shape[-2:], boxes[:, :4], item["original_shape"],
                                   ratio_pad=item["ratio_pad"])
        # Match the pinned validator's FP32 arithmetic before Python rounding;
        # subtracting .tolist() coordinates would instead use float64.
        bbox = xyxy2xywh(boxes[:, :4])
        bbox[:, :2] -= bbox[:, 2:] / 2
        for box, score, cls in zip(bbox.tolist(), boxes[:, 4].tolist(), boxes[:, 5].tolist(), strict=True):
            category = int(cls)
            require(0 <= category < len(category_map), "NMS produced invalid category")
            results.append({"image_id": item["image_id"], "category_id": category_map[category],
                            "bbox": [round(value, 3) for value in box],
                            "score": round(score, 5)})
    return results


def evaluate_coco(annotations: Path, predictions: list[dict], image_ids: list[int], out: Path) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    out.write_text(json.dumps(predictions) + "\n")
    truth = COCO(str(annotations))
    if predictions:
        detected = truth.loadRes(str(out))
    else:
        # Empty predictions are a valid zero-quality model, not a skip/fallback.
        detected = COCO()
        detected.dataset = {"images": copy.deepcopy(truth.dataset["images"]),
                            "categories": copy.deepcopy(truth.dataset["categories"]), "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(truth, detected, iouType="bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {"AP_50_95": float(evaluator.stats[0]), "AP_50": float(evaluator.stats[1]),
               "AP_75": float(evaluator.stats[2]), "AP_small": float(evaluator.stats[3]),
               "AP_medium": float(evaluator.stats[4]), "AP_large": float(evaluator.stats[5])}
    require(all(math.isfinite(value) for value in metrics.values()),
            "COCO evaluator returned nonfinite metrics")
    return {**metrics, "prediction_count": len(predictions), "prediction_sha256": sha256_file(out),
            "evaluator": "pycocotools.COCOeval bbox", "image_count": len(image_ids),
            "iou_thresholds": evaluator.params.iouThrs.tolist(), "max_dets": evaluator.params.maxDets}


def load_model(args: argparse.Namespace) -> tuple[nn.Module, dict]:
    if args.family == "resnet":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        identity = {"architecture": "torchvision.ResNet18", "weights": "IMAGENET1K_V1",
                    "source": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                    "loaded_state_sha256": state_hash(model), "fused_before_patch": False}
    else:
        require(version("ultralytics") == ULTRALYTICS_VERSION,
                f"YOLO adapter requires ultralytics=={ULTRALYTICS_VERSION}")
        from ultralytics.nn.tasks import DetectionModel, load_checkpoint

        model, _ = load_checkpoint(str(args.checkpoint), device="cpu", fuse=False)
        require(isinstance(model, DetectionModel) and model.model[-1].nc == 80,
                "checkpoint must be an official 80-class YOLO detection model")
        require(model.yaml.get("scale") == "n" and
                Path(model.yaml.get("yaml_file", "")).stem in {"yolo11", "yolo11n"},
                "checkpoint configuration must identify YOLO11 nano (not another YOLO family/size)")
        require(not getattr(model, "end2end", False), "end-to-end YOLO heads are unsupported")
        identity = {"architecture": "ultralytics.YOLO11n", "weights": "official yolo11n.pt local pin",
                    "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                    "loaded_state_sha256": state_hash(model), "fused_before_patch": True}
        model.fuse(verbose=False)
        for module in model.modules():
            if hasattr(module, "export"):
                module.export = False
    model = model.eval().float()
    identity["effective_native_state_sha256"] = state_hash(model)
    return model, identity


def paired_quality(args: argparse.Namespace, result: dict, env: dict) -> None:
    model, model_identity = load_model(args)
    native = model.to("cuda").eval()
    emulated = copy.deepcopy(native)
    original_layers = {name: layer for name, layer in emulated.named_modules()
                       if isinstance(layer, (nn.Conv2d, nn.Linear))}
    recipe = load_recipe(args.recipe)
    report = patch_model(emulated, recipe, backend="triton", include_conv2d=True)
    require(any(name in original_layers and isinstance(original_layers[name], nn.Conv2d)
                for name, _ in report.patched), "recipe did not select any Conv2d")
    require(state_hash(emulated) == model_identity["effective_native_state_sha256"],
            "patching changed original state_dict weight/bias bytes")
    if args.family == "resnet":
        dataset = ImageNetValidation(args.images, args.out, args.limit)
        preprocessing = {"recipe": "ResNet18_Weights.IMAGENET1K_V1.transforms",
                         "resize_short_edge": 256, "crop": 224, "interpolation": "PIL bilinear",
                         "antialias": True, "rgb": True, "label_order": "lexicographically sorted WNID",
                         "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        collate = None
    else:
        stride = int(native.stride.max().item())
        dataset = COCOValidation(args.images, args.annotations, args.out, args.batch, args.imgsz,
                                 stride, args.limit)
        preprocessing = {"recipe": "Ultralytics v8.3.221 rectangular validation", "imgsz": args.imgsz,
                         "rect": True, "rect_pad": 0.5, "stride": stride, "scaleup": False,
                         "resize": "long side imgsz, ceil dimensions, cv2 INTER_LINEAR",
                         "padding": 114, "rgb": True, "normalization": "float32 / 255"}
        collate = coco_collate
    expected = FULL_COUNTS[args.family] if args.limit is None else args.limit
    require(len(dataset) == expected, f"expected {expected} images, found {len(dataset)}")
    env.update(model=model_identity, dataset_manifest_sha256=dataset.manifest_sha256,
               preprocessing=preprocessing, recipe_sha256=recipe.sha256)
    if args.family == "resnet":
        env["class_index_sha256"] = dataset.class_index_sha256
    else:
        env["annotation_sha256"] = sha256_file(args.annotations)
        env["postprocessing"] = {"conf": args.conf, "iou": args.iou, "max_det": args.max_det,
                                 "multi_label": True, "agnostic": False, "max_nms": 30_000,
                                 "nms": "torchvision.ops.nms via ultralytics.utils.nms",
                                 "nms_timeout": "disabled, no partial-image truncation",
                                 "json_bbox_decimals": 3, "json_score_decimals": 5,
                                 "augmentation": False, "half": False, "export": False,
                                 "coco_category_ids": dataset.category_map}
    (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False, collate_fn=collate,
                        generator=torch.Generator().manual_seed(42))
    correct = {name: [0, 0] for name in ("native", "emulated")}
    predictions: dict[str, list[dict]] = {"native": [], "emulated": []}
    images_seen: list[int] = []
    input_digest = hashlib.sha256()
    first_input, first_native = None, None
    logits_max_diff = 0.0
    started = time.monotonic()
    with torch.no_grad(), DispatchWitness(emulated) as witness:
        # No AutoBackend, Model.val, export, compile or fusion may touch the
        # patched graph. Native fusion (YOLO) was completed before copying.
        guard = patch.object(emulated, "fuse", side_effect=RuntimeError("patched model refusion forbidden"))
        with guard if args.family == "yolo" else ExitStack():
            for batch_index, batch in enumerate(loader):
                inputs = batch[0].to("cuda", non_blocking=True).float()
                if args.family == "yolo":
                    inputs = inputs / 255
                finite(inputs, "preprocessed inputs")
                input_digest.update(tensor_hash(inputs).encode())
                before = witness.snapshot()
                native_output = native(inputs)
                emulated_output = emulated(inputs)
                finite(native_output, "native raw model")
                finite(emulated_output, "emulated raw model")
                witness.check_batch(before)
                if first_input is None:
                    first_input = inputs.clone()
                    first_native = [value.detach().clone() for value in output_tensors(native_output)]
                if args.family == "resnet":
                    target = batch[1].to("cuda")
                    images_seen.extend(batch[2].tolist())
                    require(native_output.shape == emulated_output.shape == (inputs.shape[0], 1_000),
                            "ResNet logits shape changed")
                    logits_max_diff = max(logits_max_diff,
                                          float((native_output - emulated_output).abs().max().item()))
                    for name, output in (("native", native_output), ("emulated", emulated_output)):
                        indices = output.argsort(dim=-1, descending=True, stable=True)[:, :5]
                        correct[name][0] += int((indices[:, 0] == target).sum().item())
                        correct[name][1] += int((indices == target[:, None]).any(dim=1).sum().item())
                else:
                    metadata = batch[1]
                    images_seen.extend(item["image_id"] for item in metadata)
                    for name, output in (("native", native_output), ("emulated", emulated_output)):
                        predictions[name].extend(serialize_predictions(output, metadata, tuple(inputs.shape),
                                                                       dataset.category_map, args.conf,
                                                                       args.iou, args.max_det))
                count = len(images_seen)
                if (batch_index + 1) % 25 == 0 or count == expected:
                    print(f"VISION_QUALITY_PROGRESS images={count}/{expected} batches={batch_index + 1}",
                          flush=True)
                    result.update(processed_images=count, last_completed_batch=batch_index)
                    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        require(len(images_seen) == expected and len(set(images_seen)) == expected,
                "evaluation omitted or duplicated images")
        result["dispatch"] = witness.summary()
    unpatch_model(emulated)
    require(all(emulated.get_submodule(name) is layer for name, layer in original_layers.items()),
            "unpatch did not restore original Conv2d/Linear module identities")
    require(state_hash(emulated) == model_identity["effective_native_state_sha256"],
            "unpatch changed native weight/bias bytes")
    with torch.no_grad():
        restored = emulated(first_input)
    assert_bits(restored, first_native, "unpatched native first-batch restoration")
    result.update(processed_images=expected, unique_images=expected, all_outputs_finite=True,
                  full_dataset_completed=args.limit is None, diagnostic_only=args.limit is not None,
                  restoration={"sample": "first real preprocessed batch", "bit_equal": True,
                               "native_state_identity_preserved": True},
                  transformed_input_stream_sha256=input_digest.hexdigest(),
                  elapsed_seconds_not_benchmark=time.monotonic() - started,
                  patch_report={"patched": report.patched, "skipped": report.skipped,
                                "layers": report.layers})
    if args.family == "resnet":
        result["metrics"] = {name: {"top1": values[0] / expected, "top5": values[1] / expected,
                                    "top1_correct": values[0], "top5_correct": values[1]}
                             for name, values in correct.items()}
        result["logits_vs_native_max_diff"] = logits_max_diff
        result["delta"] = {metric: result["metrics"]["emulated"][metric] - result["metrics"]["native"][metric]
                           for metric in ("top1", "top5")}
    else:
        require(sha256_file(args.annotations) == env["annotation_sha256"],
                "COCO annotations changed while evaluation was running")
        result["metrics"] = {name: evaluate_coco(args.annotations, items, images_seen,
                                                 args.out / f"predictions_{name}.json")
                             for name, items in predictions.items()}
        result["delta"] = {metric: result["metrics"]["emulated"][metric] - result["metrics"]["native"][metric]
                           for metric in ("AP_50_95", "AP_50")}


def full_hash(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("expected a full 64-character SHA256")
    return value.lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=("resnet", "yolo"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", type=full_hash, required=True)
    parser.add_argument("--images", type=Path, required=True,
                        help="ImageNet WNID val root or COCO val2017 directory")
    parser.add_argument("--annotations", type=Path, help="official instances_val2017.json (YOLO only)")
    parser.add_argument("--recipe", default="fp8_f7_lowacc")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO validation long side")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, help="diagnostic subset ONLY; never a full quality run")
    parser.add_argument("--base-git-sha", help="base commit of an uploaded dirty source archive")
    parser.add_argument("--source-archive-sha256", type=full_hash)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("positive batch, nonnegative workers and imgsz multiple of 32 required")
    if args.limit is not None and not 0 < args.limit < FULL_COUNTS[args.family]:
        parser.error("--limit must be a strict smaller positive diagnostic subset")
    if not 0 < args.conf < 1 or not 0 < args.iou < 1 or args.max_det < 1:
        parser.error("confidence/IoU must be in (0,1) and max-det positive")
    if args.base_git_sha and not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", args.base_git_sha):
        parser.error("--base-git-sha must be a full 40/64-character hex commit")
    if args.family == "yolo" and (args.annotations is None or not args.annotations.is_file()):
        parser.error("YOLO requires an existing local --annotations instances_val2017.json")
    if not args.checkpoint.is_file() or not args.images.is_dir():
        parser.error("checkpoint and images must exist locally; this harness never downloads assets")
    if any((args.out / name).exists() for name in ("env.json", "result.json")):
        parser.error("--out already contains evaluation evidence; use a fresh directory")
    args.out.mkdir(parents=True, exist_ok=True)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(stdout, args.out / "stdout.log"), Tee(stderr, args.out / "stderr.log")
    env = capture_env(extra={"command": [sys.executable, *sys.argv], "family": args.family,
                             "seed": 42, "dtype": "float32", "batch": args.batch, "workers": args.workers,
                             "base_git_sha": args.base_git_sha,
                             "source_archive_sha256": args.source_archive_sha256,
                             "harness_sha256": sha256_file(Path(__file__)),
                             "expected_checkpoint_sha256": args.checkpoint_sha256,
                             "checkpoint_path": str(args.checkpoint.resolve()),
                             "image_root": str(args.images.resolve()),
                             "deterministic_algorithms": True, "performance_claim": False})
    result: dict[str, Any] = {
        "status": "running", "family": args.family, "recipe": args.recipe,
        "diagnostic_only": args.limit is not None, "full_dataset_completed": False,
        "expected_images": FULL_COUNTS[args.family] if args.limit is None else args.limit,
        "processed_images": 0,
    }
    (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    try:
        for package in ("torchvision", "ultralytics", "pycocotools", "opencv-python", "pillow"):
            try:
                env.setdefault("optional_versions", {})[package] = version(package)
            except Exception:
                env.setdefault("optional_versions", {})[package] = None
        actual_hash = sha256_file(args.checkpoint)
        require(actual_hash == args.checkpoint_sha256,
                "checkpoint bytes do not match the explicit SHA256 pin")
        if args.family == "resnet":
            require(actual_hash.startswith("f37072fd"),
                    "checkpoint is not the torchvision ImageNet V1 artifact")
        env["actual_checkpoint_sha256"] = actual_hash
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "isolate exactly one UCL GPU using CUDA_VISIBLE_DEVICES")
        require(torch.cuda.get_device_capability()[0] >= 8, "Triton quality execution needs sm_80+")
        require(tuple(int(part) for part in version("triton").split(".")[:2]) >= (3, 4),
                "TriCast Triton IEEE helper requires Triton>=3.4")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        np.random.seed(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        paired_quality(args, result, env)
        result["status"] = "passed_diagnostic" if args.limit is not None else "passed_full_quality"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"VISION_QUALITY_DONE status={result['status']} out={args.out}", flush=True)
        sys.stdout.file.close()
        sys.stderr.file.close()
        sys.stdout, sys.stderr = stdout, stderr
    return 0 if result["status"] in ("passed_diagnostic", "passed_full_quality") else 1


if __name__ == "__main__":
    raise SystemExit(main())
