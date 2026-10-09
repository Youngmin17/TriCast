"""Record the TriCast Studio demo bundle: a design-point sweep (app/README.md, "데모 묶음").

Every run compares a baseline (native, or the same format with FP64 accumulation) with one virtual
accumulation design on a fixed input; native outputs are computed once per (model, input) and reused.
Detection first screens COCO val2017 images with CoFDA fused F7, keeps the two most different ones and
the median (rule and all scores in ``demo_scores.json``), then sweeps designs on them; classification
reuses those images. Runs are written as they finish and ``--resume`` keeps finished ones. The last line
is the terminal marker ``DEMO_BUNDLE_DONE runs=<n> failed=<k>``.

UCL cluster (NFS paths, see the ``--help`` defaults): run with the tricast conda env, PYTHONPATH holding the
repository, ``src/`` and /scratch/uceeeee/tricast/validation_20261005/deps (ultralytics, torchvision), and
``HF_HOME=/scratch/uceeeee/.cache/huggingface``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from . import listing
from .catalog import (
    FORMATS,
    MODELS,
    PRESETS,
    baseline_recipe,
    build_catalog,
    canonical_mma,
    compose_recipe,
    mma_label,
    mma_slug,
    preset_for,
    preset_mma,
    recipe_record,
)
from .metrics import detection_score, pick_examples
from .runners import common, llm, vision

REPO = Path(__file__).resolve().parents[1]
CLUSTER = Path("/scratch/uceeeee/tricast/quality_20261005")
MAX_NEW_TOKENS = 48
FORMAT = "fp8_tensor"
LICENSES = (4, 5, 7, 8)
MAX_MEDIA_SIDE = 1024
QWEN, LLAMA, YOLO, RESNET = "Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B", "yolo11n", "resnet18"
QWEN_PROMPTS = (
    "Explain in two sentences why an accumulator's precision matters in a dot product.",
    "What is 17 × 23? Answer with the number first, then one short sentence.",
    "내적에서 누산기 정밀도가 왜 중요한지 두 문장으로 설명해 줘.",
)
LLAMA_PROMPTS = (
    "The precision of the accumulator in a dot product matters because",
    "In one sentence, a tensor core computes",
)
ROLES = {"top1": "차이 점수 1위", "top2": "차이 점수 2위", "median": "차이 점수 중앙값", "quick": "빠른 점검"}


@dataclass(frozen=True)
class Point:
    """One design point: canonical mma, input format, baseline and the sweep it belongs to."""

    mma: dict
    format: str | None
    baseline: str
    sweep: str


@dataclass(frozen=True)
class RunSpec:
    """A planned run; ``role`` says why its image was chosen (``None`` for LLM runs)."""

    id: str
    task: str
    model: str
    point: Point
    input: dict
    title: str
    selection: str
    role: str | None = None


@dataclass
class Tally:
    failed: int = 0


def cofda(f_bits: int, c_mode: str = "fused") -> dict:
    return canonical_mma({"algorithm": "cofda", "f_bits": f_bits, "chunk_size": 32, "c_mode": c_mode})


def _values(values: tuple[int, ...]) -> str:
    return "{" + ",".join(str(value) for value in values) + "}"


def sweep_points(fused: tuple[int, ...], decoupled: tuple[int, ...], presets: tuple[str, ...],
                 sweep: str) -> list[Point]:
    points = [Point(cofda(f_bits), FORMAT, "native", sweep) for f_bits in fused]
    points += [Point(cofda(f_bits, "decoupled"), FORMAT, "native", sweep) for f_bits in decoupled]
    return points + [Point(preset_mma(preset), PRESETS[preset].format, "native", sweep) for preset in presets]


def qwen_points(prompt: int, quick: bool) -> list[Point]:
    if quick:
        return sweep_points((7, 13), (), (), "빠른 점검: CoFDA fused F ∈ {7,13}")
    if prompt > 1:
        return sweep_points((7, 13, 25), (7,), ("fp64",),
                            "축소 스윕: CoFDA fused F ∈ {7,13,25}, decoupled F7, FP64 · FP8 텐서 스케일")
    fused, decoupled = (5, 7, 9, 11, 13, 16, 20, 25), (5, 7, 9, 11, 13)
    sweep = (f"설계점 스윕: CoFDA CS32 fused F ∈ {_values(fused)}, "
             f"decoupled (F2 23) F ∈ {_values(decoupled)} · FP8 텐서 스케일")
    points = sweep_points(fused, decoupled, (), sweep)
    points += sweep_points((), (), ("ada", "deepseek_promote", "blackwell_fp4", "fp64"),
                           "preset 점: 각 preset 이 전제하는 형식")
    isolation = "누산만 비교: 같은 FP8 형식 + FP64 누산 기준선"
    return points + [Point(cofda(f_bits), FORMAT, "same_quant_fp64", isolation) for f_bits in (7, 13)]


def _title(subject: str, point: Point) -> str:
    parts = [subject, mma_label(point.mma), FORMATS[point.format].label]
    preset = preset_for(point.mma, point.format)
    if preset is not None:
        parts.append(f"preset {PRESETS[preset].label}")
    if point.baseline == "same_quant_fp64":
        parts.append("기준선 FP64 누산")
    return " · ".join(parts)


def _spec(prefix: str, subject: str, task: str, model: str, point: Point, inputs: dict, label: str,
          selection: str, role: str | None = None) -> RunSpec:
    run_id = f"{prefix}-{subject}-{mma_slug(point.mma)}-{point.format or 'none'}-{point.baseline}"
    return RunSpec(run_id, task, model, point, inputs, _title(label, point), selection, role)


def llm_specs(quick: bool) -> list[RunSpec]:
    generation = f"greedy, max_new_tokens {MAX_NEW_TOKENS}, seed {common.SEED}"
    specs = []
    for number, prompt in enumerate(QWEN_PROMPTS[:1] if quick else QWEN_PROMPTS, 1):
        specs += [_spec("qwen3", f"p{number}", "llm.generate", QWEN, point, {"prompt": prompt},
                        f"Qwen3-0.6B · 프롬프트 {number}", f"{point.sweep} · {generation}")
                  for point in qwen_points(number, quick)]
    if not quick:
        points = sweep_points((7, 13, 25), (7,), ("fp64",),
                              "베이스 모델 이어 쓰기: CoFDA fused F ∈ {7,13,25}, decoupled F7, FP64 · "
                              "FP8 텐서 스케일")
        for number, prompt in enumerate(LLAMA_PROMPTS, 1):
            specs += [_spec("llama3.2-1b", f"p{number}", "llm.generate", LLAMA, point, {"prompt": prompt},
                            f"Llama-3.2-1B · 프롬프트 {number}", f"{point.sweep} · {generation}")
                      for point in points]
    return specs


def vision_specs(selection: dict | None, quick: bool) -> list[RunSpec]:
    if selection is None:
        return []
    if quick:
        detect = classify = sweep_points((7, 13), (), (), "빠른 점검: CoFDA fused F ∈ {7,13}")
    else:
        fused, decoupled = (5, 7, 9, 11, 13, 16, 20, 25), (5, 7, 9, 13)
        detect = sweep_points(fused, decoupled, ("fp64",),
                              f"설계점 스윕: CoFDA CS32 fused F ∈ {_values(fused)}, "
                              f"decoupled F ∈ {_values(decoupled)}, FP64")
        classify = sweep_points((5, 7, 9, 13, 25), (), ("fp64",), "설계점 스윕: CoFDA CS32 fused "
                                                                  "F ∈ {5,7,9,13,25}, FP64")
    specs = []
    for task, prefix, model, label, points in (("vision.detect", "det", YOLO, "YOLO11n", detect),
                                               ("vision.classify", "cls", RESNET, "ResNet18", classify)):
        for image in selection["selected"]:
            subject = Path(image["media"]).stem
            why = f"{selection['rule']} · 이 이미지: {ROLES[image['role']]} (점수 {image['score']:.3f})"
            role = f"{ROLES[image['role']]} ({image['score']:.1f})"
            specs += [_spec(prefix, subject, task, model, point, {"image": image["media"]},
                            f"{label} · COCO {image['image_id']}", f"{why} · {point.sweep}", role)
                      for point in points]
    return specs


def write_json(path: Path, value: object) -> None:
    """Write strict JSON atomically, so an interrupted run never leaves a file that --resume would keep."""
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1)
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_record(spec: RunSpec, request_input: dict, recipe: dict, result: dict, env: dict) -> dict:
    """``GET /api/runs/{id}`` for a finished run (app/README.md)."""
    point = spec.point
    request = {"task": spec.task, "model": spec.model, "mma": point.mma, "format": point.format,
               "baseline": point.baseline, "input": request_input}
    return {"id": spec.id, "status": "done", "stage": None, "progress": 1.0, "request": request,
            "mma_label": mma_label(point.mma), "preset": preset_for(point.mma, point.format),
            "recipe": recipe_record(recipe), "env": env, **result}


def index_entry(spec: RunSpec, run: dict) -> dict:
    point = spec.point
    return {"id": spec.id, "task": spec.task, "model": spec.model, "mma": point.mma, "format": point.format,
            "baseline": point.baseline, "input": spec.input, "title": spec.title, "selection": spec.selection,
            "role": spec.role, "summary": listing.summary(spec.task, run["metrics"]),
            "preview": listing.preview(spec.task, run["baseline"], run["emulated"])}


class Clock:
    """Progress lines for one case: elapsed time and a remaining-time estimate from the mean so far."""

    def __init__(self, case: str, todo: int, kept: int) -> None:
        self.case, self.todo, self.done, self.start = case, todo, 0, time.monotonic()
        print(f"== {case}: 실행 {todo}개, 기존 {kept}개 유지", flush=True)

    def tick(self, run_id: str, seconds: float, ok: bool) -> None:
        self.done += 1
        elapsed = time.monotonic() - self.start
        remaining = elapsed / self.done * (self.todo - self.done)
        print(f"[{self.case} {self.done}/{self.todo}] {'ok' if ok else 'FAILED'} {run_id} {seconds:.1f}s · "
              f"경과 {elapsed:.0f}s · 남은 예상 {remaining:.0f}s", flush=True)

    def finish(self) -> None:
        print(f"== {self.case}: 끝, {time.monotonic() - self.start:.0f}s", flush=True)


def attempt(spec: RunSpec, work: Callable[[], dict], target: Path, clock: Clock, tally: Tally) -> dict | None:
    """Run and write one record; a failure is counted and logged, and the sweep goes on."""
    start = time.monotonic()
    try:
        record = work()
        write_json(target, record)
    except Exception:
        tally.failed += 1
        traceback.print_exc()
        clock.tick(spec.id, time.monotonic() - start, ok=False)
        return None
    clock.tick(spec.id, time.monotonic() - start, ok=True)
    return record


def record_llm(spec: RunSpec, device: str) -> dict:
    point = spec.point
    recipe = compose_recipe(point.mma, point.format, spec.task)
    baseline = baseline_recipe(point.baseline, point.format)
    result = llm.run_generate(spec.model, recipe, baseline, spec.input["prompt"], MAX_NEW_TOKENS, device)
    model, _ = llm.load_llm(spec.model, device, MODELS[spec.model].revision)
    env = common.run_env(spec.task, spec.model, recipe, baseline,
                         {"model_commit": getattr(model.config, "_commit_hash", None)}, device=device)
    request_input = {"prompt": spec.input["prompt"], "max_new_tokens": MAX_NEW_TOKENS}
    return run_record(spec, request_input, recipe, result, env)


def record_vision(spec: RunSpec, image: Path, device: str) -> dict:
    point = spec.point
    recipe = compose_recipe(point.mma, point.format, spec.task)
    baseline = baseline_recipe(point.baseline, point.format)
    run = vision.run_detect if spec.task == "vision.detect" else vision.run_classify
    result = run(spec.model, recipe, baseline, image, device)
    env = common.run_env(spec.task, spec.model, recipe, baseline, device=device)
    return run_record(spec, {"image": image.name}, recipe, result, env)


def llm_case(args: argparse.Namespace, out: Path, tally: Tally) -> None:
    specs = llm_specs(args.quick)
    todo = [spec for spec in specs if not (args.resume and (out / "runs" / f"{spec.id}.json").is_file())]
    clock = Clock("llm", len(todo), len(specs) - len(todo))
    for spec in todo:
        attempt(spec, partial(record_llm, spec, args.device), out / "runs" / f"{spec.id}.json", clock, tally)
    clock.finish()


def coco_images(annotations: Path, count: int) -> tuple[list[dict], dict[int, dict]]:
    """The first ``count`` val2017 images by id whose license allows redistribution (ids 4, 5, 7, 8)."""
    data = read_json(annotations)
    images = sorted((image for image in data["images"] if image["license"] in LICENSES),
                    key=lambda image: image["id"])
    return images[:count], {item["id"]: item for item in data["licenses"]}


def prepare_media(source: Path, target: Path) -> None:
    """Copy the image when its long side is at most 1024 px, else save a downscaled JPEG."""
    from PIL import Image

    with Image.open(source) as image:
        width, height = image.size
        if max(width, height) <= MAX_MEDIA_SIDE:
            shutil.copyfile(source, target)
            return
        scale = MAX_MEDIA_SIDE / max(width, height)
        resized = image.convert("RGB").resize((round(width * scale), round(height * scale)),
                                              Image.Resampling.LANCZOS)
    resized.save(target, quality=92)


def _screen(args: argparse.Namespace, tally: Tally) -> tuple[list[dict], Point]:
    """Detection with CoFDA fused F7 on every screened image; returns one score row per image."""
    images, licenses = coco_images(args.coco_ann, 1 if args.quick else args.images)
    point = Point(cofda(7), FORMAT, "native", "선별")
    (args.cache / "media").mkdir(parents=True, exist_ok=True)
    (args.cache / "runs").mkdir(parents=True, exist_ok=True)
    rows, todo = [], []
    for image in images:
        media = f"coco_{image['file_name']}"
        path = args.cache / "media" / media
        if not (args.resume and path.is_file()):
            prepare_media(args.coco_root / image["file_name"], path)
        spec = _spec("det", Path(media).stem, "vision.detect", YOLO, point, {"image": media}, "", "")
        target = args.cache / "runs" / f"{spec.id}.json"
        license_ = licenses[image["license"]]
        rows.append({"image_id": image["id"], "file_name": image["file_name"], "media": media,
                     "flickr_url": image.get("flickr_url"), "coco_url": image.get("coco_url"),
                     "license": {"id": license_["id"], "name": license_["name"], "url": license_["url"]},
                     "resized": max(image["width"], image["height"]) > MAX_MEDIA_SIDE, "run": spec.id})
        if not (args.resume and target.is_file()):
            todo.append((spec, path, target))
    clock = Clock("detect 선별 (CoFDA fused F7)", len(todo), len(images) - len(todo))
    for spec, path, target in todo:
        attempt(spec, partial(record_vision, spec, path, args.device), target, clock, tally)
    clock.finish()
    for row in rows:
        target = args.cache / "runs" / f"{row['run']}.json"
        metrics = read_json(target)["metrics"] if target.is_file() else None
        row.update(score=None if metrics is None else detection_score(metrics),
                   **{key: None if metrics is None else metrics[key]
                      for key in ("matched", "baseline_only", "emulated_only", "mean_iou")})
    return rows, point


def detect_case(args: argparse.Namespace, out: Path, tally: Tally) -> None:
    rows, point = _screen(args, tally)
    scores = {row["image_id"]: row["score"] for row in rows if row["score"] is not None}
    picks = [(rows[0]["image_id"], "quick")] if args.quick and scores else pick_examples(scores)
    roles = dict(picks)
    rule = (f"COCO val2017 에서 license id ∈ {{4,5,7,8}} 인 이미지를 image id 순으로 앞 {len(rows)}장 골라 "
            f"YOLO11n 을 {mma_label(point.mma)} · {FORMATS[FORMAT].label} 로 돌리고, "
            "차이 점수 = baseline_only + emulated_only + (1 − mean_iou) "
            "(짝이 없으면 1, 양쪽 모두 상자가 없으면 0) 가 가장 큰 2장과 하위 중앙값 1장을 골랐다 "
            "(동점은 image id 가 작은 쪽). 전체 점수: demo_scores.json")
    for row in rows:
        row["role"] = roles.get(row["image_id"])
    selected = [{key: row[key] for key in ("image_id", "media", "role", "score")}
                for image_id, _ in picks for row in rows if row["image_id"] == image_id]
    document = {"rule": rule, "quick": args.quick, "design": point.mma, "format": FORMAT,
                "detection": common.DETECTION, "images": rows, "selected": selected}
    write_json(out / "demo_scores.json", document)
    chosen = [row for row in rows if row["role"] is not None]
    for row in chosen:
        shutil.copyfile(args.cache / "media" / row["media"], out / "media" / row["media"])
        record = read_json(args.cache / "runs" / f"{row['run']}.json")
        write_json(out / "runs" / f"{row['run']}.json", record)
        if args.render:
            render_compare(out / "media" / row["media"], record, out / "media" / f"{row['run']}_compare.png")
    write_sources(out / "media" / "SOURCES.md", chosen)
    specs = [spec for spec in vision_specs(document, args.quick) if spec.task == "vision.detect"]
    _sweep("detect", specs, args, out, tally, kept={row["run"] for row in chosen})


def classify_case(args: argparse.Namespace, out: Path, tally: Tally) -> None:
    scores = out / "demo_scores.json"
    if not scores.is_file():
        print("SKIP classify: demo_scores.json 이 없습니다. 검출 선별(--cases detect)을 먼저 실행하세요",
              flush=True)
        return
    try:
        vision.load_classifier(args.device)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"SKIP classify: {exc}", flush=True)
        return
    specs = [spec for spec in vision_specs(read_json(scores), args.quick) if spec.task == "vision.classify"]
    _sweep("classify", specs, args, out, tally)


def _sweep(case: str, specs: list[RunSpec], args: argparse.Namespace, out: Path, tally: Tally,
           kept: frozenset[str] | set[str] = frozenset()) -> None:
    """Run ``specs`` on their bundle media; ``kept`` ids (screening runs copied in) are not run again."""
    todo = [spec for spec in specs if spec.id not in kept
            and not (args.resume and (out / "runs" / f"{spec.id}.json").is_file())]
    clock = Clock(case, len(todo), len(specs) - len(todo))
    for spec in todo:
        image = out / "media" / spec.input["image"]
        target = out / "runs" / f"{spec.id}.json"
        record = attempt(spec, partial(record_vision, spec, image, args.device), target, clock, tally)
        if record is not None and args.render and spec.task == "vision.detect":
            render_compare(image, record, out / "media" / f"{spec.id}_compare.png")
    clock.finish()


def write_sources(path: Path, rows: list[dict]) -> None:
    lines = ["# Demo media sources", "",
             "COCO 2017 validation images (https://cocodataset.org), redistributed under the Flickr "
             "license of each image as recorded in instances_val2017.json. Only license ids 4, 5, 7 and 8 "
             "were eligible.", "",
             "| media | COCO image id | source | license | file |", "|---|---|---|---|---|"]
    for row in rows:
        license_ = row["license"]
        stored = "downscaled to long side 1024 px" if row["resized"] else "original bytes"
        lines.append(f"| {row['media']} | {row['image_id']} | {row['flickr_url'] or row['coco_url']} | "
                     f"[{license_['name']}]({license_['url']}) | {stored} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_compare(image_path: Path, run: dict, target: Path) -> None:
    """Side-by-side baseline | emulated with boxes: green = paired, red = only on that side."""
    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    header, gap = 22, 8
    canvas = Image.new("RGB", (image.width * 2 + gap, image.height + header), "white")
    draw = ImageDraw.Draw(canvas)
    request = run["request"]
    titles = ("native" if request["baseline"] == "native" else "same format, fp64 accumulation",
              f"{mma_slug(request['mma'])} / {request['format'] or 'unquantized'}")
    pairs = run["metrics"]["pairs"]
    paired = ({pair["baseline"] for pair in pairs}, {pair["emulated"] for pair in pairs})
    for panel, (side, title) in enumerate(zip(("baseline", "emulated"), titles, strict=True)):
        left = panel * (image.width + gap)
        canvas.paste(image, (left, header))
        draw.text((left + 4, 4), title, fill="black")
        for index, box in enumerate(run[side]["boxes"]):
            color = (0, 150, 0) if index in paired[panel] else (220, 30, 30)
            x1, y1, x2, y2 = box["xyxy"]
            draw.rectangle((left + x1, header + y1, left + x2, header + y2), outline=color, width=2)
            draw.text((left + x1 + 2, header + y1 + 1), f"{box['cls']} {box['conf']:.2f}", fill=color)
    canvas.save(target)


def write_index(out: Path, quick: bool) -> int:
    """index.json from the planned runs that exist; returns their number."""
    scores = out / "demo_scores.json"
    specs = llm_specs(quick) + vision_specs(read_json(scores) if scores.is_file() else None, quick)
    entries = [index_entry(spec, read_json(out / "runs" / f"{spec.id}.json")) for spec in specs
               if (out / "runs" / f"{spec.id}.json").is_file()]
    write_json(out / "index.json", {"runs": entries})
    stale = sorted({path.stem for path in (out / "runs").glob("*.json")} - {entry["id"] for entry in entries})
    if stale:
        print(f"index 에 없는 run 파일 {len(stale)}개 (이전 선택의 잔여물): {stale}", flush=True)
    return len(entries)


CASES = {"llm": llm_case, "detect": detect_case, "classify": classify_case}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.demo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "studio_demo",
                        help="bundle directory; keep it outside tracked paths (runs/ is git-ignored), or "
                             "every run after the first records git_dirty, then copy it into app/web/demo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", default="llm,detect,classify",
                        help="comma-separated: llm, detect, classify")
    parser.add_argument("--coco-root", type=Path, default=CLUSTER / "data" / "coco2017" / "val2017")
    parser.add_argument("--coco-ann", type=Path,
                        default=CLUSTER / "data" / "coco2017" / "annotations" / "instances_val2017.json")
    parser.add_argument("--yolo-checkpoint", type=Path, default=CLUSTER / "checkpoints" / "yolo11n.pt")
    parser.add_argument("--resnet-checkpoint", type=Path,
                        default=CLUSTER / "checkpoints" / "resnet18-f37072fd.pth")
    parser.add_argument("--images", type=int, default=24, help="COCO images screened for detection")
    parser.add_argument("--cache", type=Path, default=REPO / "runs" / "studio_demo_cache",
                        help="screening runs and media (outside the bundle)")
    parser.add_argument("--resume", action="store_true", help="keep finished runs")
    parser.add_argument("--render", action="store_true",
                        help="also write media/<run id>_compare.png (detection)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: Qwen prompt 1 and one image with CoFDA fused F7/F13 "
                             "(use its own --out)")
    args = parser.parse_args(argv)
    args.cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    if not args.cases or set(args.cases) - set(CASES):
        parser.error(f"--cases must select from {', '.join(CASES)}")
    if args.images < 1:
        parser.error("--images must be positive")
    if "detect" in args.cases and not (args.coco_root.is_dir() and args.coco_ann.is_file()):
        parser.error("detect needs an existing --coco-root directory and --coco-ann file")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ[common.CHECKPOINT_ENV[YOLO]] = str(args.yolo_checkpoint)
    os.environ[common.CHECKPOINT_ENV[RESNET]] = str(args.resnet_checkpoint)
    common.configure_torch()
    for directory in (args.out / "runs", args.out / "media"):
        directory.mkdir(parents=True, exist_ok=True)
    tally, started = Tally(), time.monotonic()
    for case in args.cases:
        try:
            CASES[case](args, args.out, tally)
        except Exception:
            tally.failed += 1
            traceback.print_exc()
            print(f"== {case}: 중단 (위 traceback)", flush=True)
    catalog = build_catalog("demo", {"kind": "none", "name": None, "backend": None})
    write_json(args.out / "catalog.json", catalog)
    runs = write_index(args.out, args.quick)
    print(f"== 전체 {time.monotonic() - started:.0f}s", flush=True)
    print(f"DEMO_BUNDLE_DONE runs={runs} failed={tally.failed}", flush=True)
    return 0 if tally.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
