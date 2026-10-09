"""TriCast Studio catalog: parameter spaces of virtual accumulation algorithms, labelled presets, input
formats, models and tasks, and the composition of a design and a format into a validated TriCast recipe
(app/README.md, "가상 알고리즘과 형식 → 레시피" and ``GET /api/catalog``).

Preset numbers come from ``tricast.mma.spec.PRESETS`` or a bundled recipe, never from this file; their
verification status mirrors ``support_matrix.yaml`` (checked by app/tests/test_catalog.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files

import yaml

import tricast
from tricast import load_recipe
from tricast.eval.envinfo import source_identity
from tricast.mma.spec import MAX_F_BITS, MMASpec
from tricast.mma.spec import PRESETS as MMA_PRESETS
from tricast.quant.spec import get_scheme
from tricast.recipe import Recipe, list_recipes

TASKS = (
    {"id": "llm.generate", "kind": "llm", "label": "텍스트 생성",
     "description": "같은 프롬프트를 greedy 로 생성해 처음 갈라지는 위치를 찾고, baseline 생성열을 "
                    "두 모델에 똑같이 넣어 위치별 다음 토큰 분포(KL·top-1)를 비교한다"},
    {"id": "vision.detect", "kind": "vision", "label": "객체 검출",
     "description": "YOLO11n 검출 상자를 같은 클래스·IoU ≥ 0.5 로 짝지어 비교한다"},
    {"id": "vision.classify", "kind": "vision", "label": "이미지 분류",
     "description": "ResNet18 상위 5 클래스와 1,000 클래스 확률 분포의 KL 을 비교한다"},
)
BASELINES = (
    {"id": "native", "label": "원본 (native)",
     "description": "패치하지 않은 모델. 차이는 양자화와 누산을 함께 바꾼 결과다"},
    {"id": "same_quant_fp64", "label": "같은 형식 + FP64 누산",
     "description": "같은 입력 형식으로 양자화하고 FP64 로 누산한다. 차이는 누산만의 효과다"},
)


@dataclass(frozen=True)
class Param:
    """One field of an algorithm's parameter schema (catalog ``algorithms[].params``)."""

    key: str
    label: str
    kind: str
    default: int | str
    help: str
    min: int | None = None
    max: int | None = None
    ui_min: int | None = None
    ui_max: int | None = None
    options: tuple[int | str, ...] = ()
    when: dict[str, int | str] = field(default_factory=dict)
    advanced: bool = False
    unit: str | None = None

    def applies(self, values: dict) -> bool:
        return all(values.get(key) == value for key, value in self.when.items())

    def coerce(self, value: object) -> int | str:
        if self.kind == "int" or isinstance(self.default, int):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                    isinstance(value, float) and not value.is_integer()):
                raise ValueError(f"{self.label}: 정수여야 합니다 (받은 값 {value!r}).")
            value = int(value)
        if self.kind == "int" and not self.min <= value <= self.max:
            raise ValueError(f"{self.label}: {self.min}~{self.max} 범위여야 합니다 (받은 값 {value}).")
        if self.kind == "choice" and value not in self.options:
            choices = ", ".join(str(option) for option in self.options)
            raise ValueError(f"{self.label}: {choices} 중 하나여야 합니다 (받은 값 {value!r}).")
        return value

    def to_json(self) -> dict:
        optional = {"min": self.min, "max": self.max, "ui_min": self.ui_min, "ui_max": self.ui_max,
                    "options": list(self.options) or None, "when": dict(self.when) or None,
                    "advanced": self.advanced or None, "unit": self.unit}
        return {"key": self.key, "label": self.label, "kind": self.kind, "default": self.default,
                "help": self.help, **{key: value for key, value in optional.items() if value is not None}}


@dataclass(frozen=True)
class Algorithm:
    id: str
    label: str
    description: str
    params: tuple[Param, ...] = ()


_NORM = Param("norm_rounding", "정규화 반올림", "choice", "rtz",
              "F 비트로 줄일 때 rtz 는 버림 (NADPE), rne 는 최근접 짝수 (TriCast 확장)",
              options=("rtz", "rne"), advanced=True)
_COFDA = (
    Param("f_bits", "F", "int", 13,
          "청크 합과 누산값에 남기는 소수 비트 수. NADPE 기준 Hopper FP8 13, Blackwell FP8 25",
          min=1, max=MAX_F_BITS, ui_min=3, ui_max=28, unit="bit"),
    Param("chunk_size", "CS", "choice", 32, "한 번에 같은 지수로 정렬해 더하는 곱의 개수",
          options=(4, 8, 16, 32, 64, 128)),
    Param("c_mode", "C 결합", "choice", "fused",
          "fused: 누산기가 청크와 함께 정렬된다. "
          "decoupled: 청크 합을 따로 구해 F2 비트로 누산기에 더한다",
          options=("fused", "decoupled")),
    Param("f2_bits", "F2", "int", 23,
          "decoupled 결합에서 청크 합을 누산기에 더할 때의 소수 비트 수 (NADPE 는 23)",
          min=1, max=MAX_F_BITS, ui_min=8, ui_max=28, when={"c_mode": "decoupled"}, unit="bit"),
    Param("promote_interval", "FP32 승격 주기", "choice", 0,
          "0 이 아니면 이 개수의 곱마다 부분합을 CUDA core FP32 FMA 로 옮긴다 (DeepSeek-V3). "
          "CS 의 배수여야 한다", options=(0, 64, 128, 256)),
    _NORM,
)
_GDFS = (
    Param("f_bits", "F", "int", 35, "타일 합에 남기는 소수 비트 수. NADPE 기준 Blackwell FP4 35",
          min=1, max=MAX_F_BITS, ui_min=8, ui_max=40, unit="bit"),
    Param("g_bits", "G", "int", 6,
          "그룹 안에서 곱을 정렬해 더할 때의 소수 비트 수. NADPE 기준 Blackwell FP4 6",
          min=1, max=MAX_F_BITS, ui_min=2, ui_max=16, unit="bit"),
    Param("group_size", "GS", "choice", 16, "한 그룹의 곱 개수", options=(8, 16, 32)),
    Param("k_tile", "KT", "choice", 64, "그룹 합들을 한 번에 합치는 K 구간. KT / GS 는 1~8",
          options=(16, 32, 64, 128, 256)),
    _NORM,
)
ALGORITHMS = {algorithm.id: algorithm for algorithm in (
    Algorithm("cofda", "CoFDA",
              "CS 개 곱마다 최대 지수에 맞춰 F 비트로 자른 뒤 정확히 더하고 FP32 누산기에 합친다 "
              "(NADPE, src/tricast/reference/mma.py)", _COFDA),
    Algorithm("gdfs", "GDFS",
              "GS 개 곱을 G 비트로 더한 그룹 합들을 K 타일마다 F 비트로 한 번에 합친다 "
              "(NADPE, src/tricast/reference/mma.py)", _GDFS),
    Algorithm("fp32_fma", "FP32 FMA", "IEEE fp32 FMA 를 K 순서로 이어 누산한다 (CUDA core SGEMM 의 정의)"),
    Algorithm("fp64", "FP64", "fp64 FMA 로 누산하고 마지막에 fp32 로 한 번 반올림한다 (수치 기준)"),
)}


@dataclass(frozen=True)
class InputFormat:
    """An input format: QuantSpec schemes for weights and activations, plus an optional transform."""

    id: str | None
    label: str
    note: str
    weight: str | None = None
    activation: str | None = None
    transform: str | dict = "none"

    @property
    def bits(self) -> int | None:
        return None if self.weight is None else get_scheme(self.weight).format.bits


_MX = "32개마다 E8M0 공유 스케일 (OCP MX v1.0)"
FORMATS = {fmt.id: fmt for fmt in (
    InputFormat(None, "양자화 없음", "가중치·활성을 모델 dtype 그대로 쓴다 (LLM bf16, 비전 fp32)"),
    InputFormat("fp8_tensor", "FP8 E4M3 · 텐서 스케일", "텐서마다 fp32 스케일 하나 (W8A8)",
                "fp8_tensor", "fp8_tensor"),
    InputFormat("fp8_block128", "FP8 E4M3 · 블록 스케일",
                "가중치 128×128 블록, 활성 1×128 그룹마다 fp32 스케일 (DeepSeek-V3 §3.3.2)",
                "fp8_block128", "fp8_group128"),
    InputFormat("mxfp8_e4m3", "MXFP8 (E4M3)", _MX, "mxfp8_e4m3", "mxfp8_e4m3"),
    InputFormat("mxfp6_e3m2", "MXFP6 (E3M2)", _MX, "mxfp6_e3m2", "mxfp6_e3m2"),
    InputFormat("mxfp4", "MXFP4 (E2M1)", _MX, "mxfp4", "mxfp4"),
    InputFormat("nvfp4", "NVFP4 (E2M1)", "16개마다 UE4M3 스케일과 텐서마다 fp32 스케일 (NVIDIA NVFP4)",
                "nvfp4", "nvfp4"),
    # The transform is the one of the bundled mxfp4_rht recipe.
    InputFormat("mxfp4_rht", "MXFP4 + RHT",
                "seed 42 무작위 Hadamard 변환 뒤 MXFP4 (QuaRot·RHT). "
                "LLM 전용: Conv2d 는 변환을 지원하지 않는다",
                "mxfp4", "mxfp4", {"kind": "random_hadamard", "seed": 42}),
)}


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    tasks: tuple[str, ...]
    revision: str | None
    support: str
    note: str
    chat: bool = False


MODELS = {model.id: model for model in (
    Model("Qwen/Qwen3-0.6B", "Qwen3-0.6B", ("llm.generate",),
          "c1899de289a04d12100db370d81485cdf75e47ca", "full_eval",
          "채팅 모델. 채팅 템플릿(생각 모드 끔)으로 생성한다. 전체 PPL·lm-eval 기록: "
          "support_matrix.yaml model_families.qwen.pretrained_quality", chat=True),
    Model("meta-llama/Llama-3.2-1B", "Llama-3.2-1B", ("llm.generate",),
          "4e20de362430cd3b72f300e6b0f18e50e7166e08", "full_eval",
          "베이스 모델. 채팅 템플릿 없이 프롬프트를 그대로 이어 쓴다. "
          "WikiText-2·Winogrande 전체 평가 기록: "
          "support_matrix.yaml model_families.llama.pretrained_quality"),
    Model("yolo11n", "YOLO11n", ("vision.detect",),
          "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1", "full_eval",
          "Ultralytics 8.3.221 공식 yolo11n.pt (revision 은 체크포인트 SHA256). "
          "COCO val2017 5,000장 전체 평가 기록 (fp8_f7_lowacc): "
          "support_matrix.yaml model_families.yolo.pretrained_quality"),
    Model("resnet18", "ResNet18", ("vision.classify",),
          "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec", "operator",
          "torchvision IMAGENET1K_V1 (revision 은 체크포인트 SHA256). "
          "연산자 통합만 검증했고 ImageNet 정확도는 측정하지 않았다 (미검증)"),
)}


@dataclass(frozen=True)
class Preset:
    """A labelled starting point: the MMA of a tricast preset or of a bundled recipe, and its format."""

    id: str
    label: str
    source: str
    format: str | None
    status: str
    status_note: str
    mma_preset: str | None = None
    recipe: str | None = None


_NADPE = "NVIDIA {} 모델링 (NADPE)"
_FIG6 = "NADPE MICRO'26 Fig. 6(a) 설계 공간"
_BLACKWELL_NOTE = "Blackwell 실리콘에서 대조하지 않았다 (미검증)"
_F7_NOTE = "실제 칩이 아닌 누산기 설계 공간의 한 점"
PRESETS = {preset.id: preset for preset in (
    Preset("hopper", "Hopper FP8", _NADPE.format("Hopper"), "fp8_tensor", "partial_mismatch",
           "H200 WGMMA 실리콘 대조 140/141 통과, FP32-C 직접 입력 1건 불일치. "
           "native 비트 일치는 검증되지 않았다 (support_matrix.yaml integration.silicon_probe.wgmma)",
           mma_preset="nvidia_hopper_fp8"),
    Preset("ada", "Ada FP8", _NADPE.format("Ada"), "fp8_tensor", "modeled",
           "Ada 실리콘에서 대조하지 않았다 (미검증)", mma_preset="nvidia_ada_fp8"),
    Preset("blackwell_fp8", "Blackwell FP8", _NADPE.format("Blackwell"), "fp8_tensor", "modeled",
           _BLACKWELL_NOTE, mma_preset="nvidia_blackwell_fp8"),
    Preset("blackwell_fp4", "Blackwell FP4", _NADPE.format("Blackwell"), "nvfp4", "modeled",
           _BLACKWELL_NOTE, mma_preset="nvidia_blackwell_fp4"),
    Preset("deepseek_promote", "DeepSeek-V3 FP8 승격", "DeepSeek-V3 보고서 §3.3.2", "fp8_block128",
           "modeled", "실리콘 대조 안 함. 가중치 128×128 블록·활성 1×128 그룹 스케일 "
           "(번들 레시피 deepseek_fp8_block 과 같은 조합)", mma_preset="deepseek_fp8_promote128"),
    Preset("f7_fused", "F7 fused 설계점", _FIG6, "fp8_tensor", "design_point", _F7_NOTE,
           recipe="fp8_f7_lowacc"),
    Preset("f7_decoupled", "F7 decoupled 설계점", _FIG6, "fp8_tensor", "design_point", _F7_NOTE,
           recipe="fp8_f7_decoupled"),
    Preset("fp64", "FP64 정확 누산", "TriCast 수치 기준 (정의)", "fp8_tensor", "reference",
           "하드웨어가 아닌 정의. 같은 형식의 누산 오차를 재는 기준점", mma_preset="fp64"),
)}


def canonical_mma(mma: dict) -> dict:
    """The canonical form of a design: its schema keys whose ``when`` holds, missing ones at ``default``.

    Raises ValueError (Korean) for an unknown algorithm or key, or a value outside the schema."""
    algorithm = ALGORITHMS.get(mma.get("algorithm"))
    if algorithm is None:
        raise ValueError(f"알 수 없는 누산 알고리즘입니다: {mma.get('algorithm')!r} "
                         f"(가능: {', '.join(ALGORITHMS)}).")
    unknown = set(mma) - {"algorithm"} - {param.key for param in algorithm.params}
    if unknown:
        raise ValueError(f"{algorithm.label} 에 없는 파라미터입니다: {', '.join(sorted(unknown))}.")
    result: dict = {"algorithm": algorithm.id}
    for param in algorithm.params:
        if param.applies(result):
            result[param.key] = param.coerce(mma.get(param.key, param.default))
    return result


def mma_label(mma: dict) -> str:
    """Human-readable design, e.g. ``CoFDA · F7 · CS32 · fused``."""
    m = canonical_mma(mma)
    if m["algorithm"] == "cofda":
        parts = ["CoFDA", f"F{m['f_bits']}", f"CS{m['chunk_size']}", m["c_mode"]]
        if m["c_mode"] == "decoupled":
            parts.append(f"F2 {m['f2_bits']}")
        if m["promote_interval"]:
            parts.append(f"FP32 승격 {m['promote_interval']}")
    elif m["algorithm"] == "gdfs":
        parts = ["GDFS", f"F{m['f_bits']}", f"G{m['g_bits']}", f"GS{m['group_size']}", f"KT{m['k_tile']}"]
    else:
        return {"fp32_fma": "FP32 FMA 순차 누산", "fp64": "FP64 정확 누산"}[m["algorithm"]]
    if m["norm_rounding"] == "rne":
        parts.append("RNE")
    return " · ".join(parts)


def mma_slug(mma: dict) -> str:
    """File-name-safe design id, e.g. ``cofda-f7-cs32-fused``."""
    m = canonical_mma(mma)
    if m["algorithm"] == "cofda":
        slug = f"cofda-f{m['f_bits']}-cs{m['chunk_size']}-{m['c_mode']}"
        if m["c_mode"] == "decoupled":
            slug += f"-f2_{m['f2_bits']}"
        if m["promote_interval"]:
            slug += f"-p{m['promote_interval']}"
    elif m["algorithm"] == "gdfs":
        slug = f"gdfs-f{m['f_bits']}-g{m['g_bits']}-gs{m['group_size']}-kt{m['k_tile']}"
    else:
        return m["algorithm"].replace("_", "")
    return slug + ("-rne" if m["norm_rounding"] == "rne" else "")


def _format(format_id: str | None) -> InputFormat:
    if format_id not in FORMATS:
        raise ValueError(f"알 수 없는 입력 형식입니다: {format_id!r}.")
    return FORMATS[format_id]


def _k_domains(fmt: InputFormat) -> set[int]:
    """Sizes of the K-varying scale domains of the format's operands (group size, block columns)."""
    specs = [get_scheme(scheme) for scheme in (fmt.weight, fmt.activation) if scheme is not None]
    return ({spec.group_size for spec in specs if spec.granularity == "group"}
            | {spec.block[1] for spec in specs if spec.granularity == "block"})


def _check_design(m: dict, fmt: InputFormat) -> None:
    """MMASpec and scale-domain rules of TriCast (src/tricast/reference/mma.py), in Korean."""
    if m["algorithm"] == "cofda":
        chunk, span, what = m["chunk_size"], m["promote_interval"], "FP32 승격 주기"
        if span % chunk:
            raise ValueError(f"FP32 승격 주기 {span} 는 청크 크기 CS {chunk} 의 배수여야 합니다.")
        if span and m["c_mode"] == "decoupled":
            raise ValueError("FP32 승격을 쓰면 승격 구간 안은 fused 로 누산합니다. "
                             "decoupled 와 함께 쓸 수 없습니다.")
    elif m["algorithm"] == "gdfs":
        tile, span, what = m["k_tile"], m["group_size"], "GDFS 그룹 크기"
        if tile % span or not 1 <= tile // span <= 8:
            raise ValueError(f"GDFS 의 K 타일 {tile} 은 그룹 크기 GS {span} 의 1~8 배여야 합니다.")
    else:
        return
    for domain in sorted(_k_domains(fmt)):
        if span and domain % span:
            raise ValueError(f"{what} {span} 이 {fmt.label} 의 스케일 구간 {domain} 을 넘습니다. "
                             f"{domain} 의 약수만 쓸 수 있습니다.")


def _recipe(m: dict, fmt: InputFormat) -> dict:
    recipe = {
        "name": f"studio:{mma_slug(m)}:{fmt.id or 'none'}",
        "description": f"{mma_label(m)} · {fmt.label}",
        "defaults": {"weight": fmt.weight, "activation": fmt.activation, "mma": dict(m),
                     "transform": json.loads(json.dumps(fmt.transform)), "weight_algo": "rtn"},
        "include": ["*"], "exclude": ["lm_head"], "backend": "auto",
    }
    try:
        load_recipe(recipe)
    except ValueError as exc:
        raise ValueError(f"TriCast 레시피 검증에 실패했습니다: {exc}") from exc
    return recipe


def compose_recipe(mma: dict, format_id: str | None, task: str) -> dict:
    """A load_recipe-valid recipe dict for a design (any mma dict, canonicalized) and an input format.

    Raises ValueError with a Korean message when the task, format or design is not supported."""
    if task not in {item["id"] for item in TASKS}:
        raise ValueError(f"알 수 없는 작업입니다: {task!r}.")
    m, fmt = canonical_mma(mma), _format(format_id)
    _check_design(m, fmt)
    if task != "llm.generate" and fmt.transform != "none":
        raise ValueError(f"{fmt.label} 은 변환(transform)을 쓰므로 비전 작업에 쓸 수 없습니다. "
                         "TriCast 의 Conv2d 는 inference·동적 양자화·RTN 만 지원합니다.")
    return _recipe(m, fmt)


def baseline_recipe(baseline: str, format_id: str | None) -> dict | None:
    """``None`` for the native baseline, else the same input format with FP64 accumulation."""
    if baseline == "native":
        return None
    if baseline != "same_quant_fp64":
        raise ValueError(f"알 수 없는 기준선입니다: {baseline!r}.")
    return _recipe(canonical_mma({"algorithm": "fp64"}), _format(format_id))


def preset_mma(preset_id: str) -> dict:
    """Canonical mma of a preset, read from its tricast preset or bundled recipe."""
    preset = PRESETS[preset_id]
    spec: MMASpec = (MMA_PRESETS[preset.mma_preset] if preset.mma_preset
                     else load_recipe(preset.recipe).defaults.mma)
    keys = [param.key for param in ALGORITHMS[spec.algorithm].params]
    return canonical_mma({"algorithm": spec.algorithm, **{key: getattr(spec, key) for key in keys}})


def preset_provenance(preset_id: str) -> str:
    """The tricast preset's provenance, or the ``# Source:`` line of the bundled recipe."""
    preset = PRESETS[preset_id]
    if preset.mma_preset:
        return MMA_PRESETS[preset.mma_preset].provenance
    text = files("tricast").joinpath("recipes", f"{preset.recipe}.yaml").read_text(encoding="utf-8")
    source = next(line.removeprefix("# Source:").strip() for line in text.splitlines()
                  if line.startswith("# Source:"))
    return f"{source} (TriCast 번들 레시피 {preset.recipe})"


def preset_for(mma: dict, format_id: str | None) -> str | None:
    """The id of the preset with this canonical mma and format, if any."""
    m = canonical_mma(mma)
    return next((preset.id for preset in PRESETS.values()
                 if preset.format == format_id and preset_mma(preset.id) == m), None)


def _semantics(recipe: Recipe) -> str:
    data = recipe.to_dict()
    for key in ("name", "description"):
        del data[key]
    for key in ("name", "provenance"):
        del data["defaults"]["mma"][key]
    return json.dumps(data, sort_keys=True)


@cache
def _bundled() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in list_recipes():
        result.setdefault(_semantics(load_recipe(name)), name)
    return result


def bundled_recipe(recipe: dict) -> str | None:
    """Name of the bundled recipe with the same arithmetic (names, descriptions and labels ignored)."""
    return _bundled().get(_semantics(load_recipe(recipe)))


def recipe_record(recipe: dict) -> dict:
    """The run result's ``recipe`` field."""
    return {"name": recipe["name"], "bundled": bundled_recipe(recipe),
            "yaml": yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True)}


def _model_json(model: Model) -> dict:
    return {"id": model.id, "label": model.label, "tasks": list(model.tasks), "revision": model.revision,
            "support": model.support, "note": model.note}


def _algorithm_json(algorithm: Algorithm) -> dict:
    return {"id": algorithm.id, "label": algorithm.label, "description": algorithm.description,
            "formats": list(FORMATS), "params": [param.to_json() for param in algorithm.params]}


def _preset_json(preset: Preset) -> dict:
    return {"id": preset.id, "label": preset.label, "source": preset.source, "mma": preset_mma(preset.id),
            "format": preset.format, "status": preset.status, "provenance": preset_provenance(preset.id),
            "status_note": preset.status_note}


def build_catalog(mode: str, device_info: dict) -> dict:
    """``GET /api/catalog`` (app/README.md)."""
    return {
        "tricast": {"version": tricast.__version__, "git_sha": source_identity()["git_sha"]},
        "mode": mode,
        "device": dict(device_info),
        "tasks": [dict(task) for task in TASKS],
        "models": [_model_json(model) for model in MODELS.values()],
        "algorithms": [_algorithm_json(algorithm) for algorithm in ALGORITHMS.values()],
        "presets": [_preset_json(preset) for preset in PRESETS.values()],
        "formats": [{"id": fmt.id, "label": fmt.label, "bits": fmt.bits, "note": fmt.note,
                     "k_domains": sorted(_k_domains(fmt))} for fmt in FORMATS.values()],
        "baselines": [dict(baseline) for baseline in BASELINES],
    }
