"""Conservative, deterministic extraction of explicitly named arithmetic choices."""

from __future__ import annotations

import copy
import re
from typing import Any, get_args

from ..formats import ALIASES, REGISTRY, get_format
from ..mma.spec import PRESETS, Algorithm
from ..quant.spec import KV_PRESETS, SCHEMES, TransformKind, WeightAlgo

TASKS = ("wikitext2_ppl", "hellaswag", "coqa", "arc_easy", "arc_challenge", "piqa",
         "winogrande", "lambada_openai")
_QUANT_FIELDS = ("scheme", "format", "granularity", "group_size", "scale_format", "scale_method",
                 "scale_rounding", "rounding", "zero_point", "observer")
_MMA_FIELDS = ("preset", "algorithm", "f_bits", "chunk_size", "c_mode", "g_bits", "group_size",
               "promote_interval")
_AXIS_NAMES = {
    "f_bits": r"f_bits|f",
    "chunk_size": r"chunk_size|chunk|cs",
    "g_bits": r"g_bits|g",
    "group_size": r"group_size|gs",
    "c_mode": r"c_mode",
}
_ROLE = re.compile(r"(?<![a-z_])(?:weights?|activation(?:s)?)(?![a-z_])|가중치|활성(?:화)?", re.I)


def recipe_names() -> list[str]:
    from ..recipe import list_recipes

    return sorted(list_recipes())


def _pattern(name: str) -> str:
    return r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])"


def _names(text: str, names: Any) -> list[str]:
    return sorted((name for name in names if re.search(_pattern(name), text, re.I)),
                  key=lambda name: (re.search(_pattern(name), text, re.I).start(), name))


def _unique(values: list[Any], field: str, questions: list[str]) -> Any:
    values = list(dict.fromkeys(values))
    if len(values) > 1:
        questions.append(f"{field}: 여러 값 {values} 중 어느 값을 사용할까요?")
        return None
    return values[0] if values else None


def _numeric(
    text: str, names: str, field: str, questions: list[str], *, integer: bool = True,
) -> int | float | None:
    prefix = rf"(?<![a-z0-9_])(?:{names})(?![a-z0-9_])\s*(?:=|:)?\s*"
    tokens = re.findall(prefix + r"([^\s,;\[\]()]+)?", text, re.I)
    values = []
    for token in tokens:
        token = re.sub(r"(?:으로|로|개|까지)$", "", token.rstrip("."))
        if not re.fullmatch(r"-?\d+" if integer else r"-?\d+(?:\.\d+)?", token):
            questions.append(f"{field}: 명시한 값 {token!r}을 유효한 숫자로 지정해 주세요.")
        else:
            values.append(int(token) if integer or "." not in token else float(token))
    return _unique(values, field, questions)


def _number(text: str, names: str, field: str, questions: list[str]) -> int | None:
    return _numeric(text, names, field, questions)


def _format_tokens(text: str, questions: list[str]) -> list[str]:
    values = []
    for token in re.findall(r"(?<![a-z0-9_])[a-z][a-z0-9_]*(?::[a-z0-9_=+-]+)*", text, re.I):
        token = token.lower()
        if token in SCHEMES or token in ("fp8", "fp4", "float"):
            continue
        if token not in REGISTRY and token not in ALIASES and not re.match(
            r"(?:u?int\d|u?e\d+m|(?:fp|bf|tf)\d)", token,
        ):
            continue
        try:
            get_format(token)
        except (ValueError, TypeError):
            questions.append(f"format: 알 수 없는 형식 {token!r}입니다. 정확한 형식을 지정해 주세요.")
        else:
            values.append(token)
    return values


def _scoped_quant(text: str, questions: list[str]) -> tuple[dict[str, Any], str]:
    fields = {
        "scheme": tuple(SCHEMES), "granularity": ("tensor", "row", "group", "block", "channel", "token"),
        "rounding": ("rne", "rna", "rtz", "rup", "rdn", "sr"),
        "scale_rounding": ("rne", "rna", "rtz", "rup", "rdn", "sr"),
        "scale_method": ("absmax", "pow2_floor", "pow2_ceil", "mse", "percentile"),
        "observer": ("minmax", "ema", "history", "percentile", "mse"),
        "zero_point": ("none", "int", "float"), "format": None, "scale_format": None,
        "group_size": None,
    }
    scoped: dict[str, Any] = {}
    for field, choices in fields.items():
        spelling = field.replace("scale_", r"scale[._]")
        prefix = rf"(?<![a-z0-9_.]){spelling}(?![a-z0-9_])\s*(?:=|:)\s*"
        pattern = prefix + r"([^\s,;\[\]()]+)?"
        values = []
        for match in re.finditer(pattern, text, re.I):
            value = (match[1] or "").strip("\"'").rstrip(".").lower()
            value = re.sub(r"(?:으로|로)$", "", value)
            valid = bool(value)
            if choices is not None:
                valid = value in choices
            elif field == "group_size":
                valid = bool(re.fullmatch(r"-?\d+", value))
                value = int(value) if valid else value
            else:
                try:
                    get_format(value)
                except (ValueError, TypeError):
                    valid = False
            if valid:
                if field == "granularity" and value in ("channel", "token"):
                    value = "row"
                values.append(value)
            else:
                questions.append(f"{field}: 알 수 없거나 잘못된 값 {value!r}입니다.")
        if re.search(prefix, text, re.I):
            scoped[field] = _unique(values, field, questions)
            text = re.sub(pattern, " ", text, flags=re.I)
    return scoped, text


def _quant(text: str, questions: list[str]) -> dict[str, Any] | None:
    result = dict.fromkeys(_QUANT_FIELDS)
    for name in _names(text, PRESETS):
        text = re.sub(_pattern(name), " ", text, flags=re.I)
    scoped, text = _scoped_quant(text, questions)
    scale_match = re.search(r"scale_format\s+([a-z0-9_:=+-]+)", text, re.I)
    if scale_match:
        try:
            get_format(scale_match[1].lower())
        except (ValueError, TypeError):
            questions.append(f"scale_format: 알 수 없는 형식 {scale_match[1]!r}입니다.")
        else:
            result["scale_format"] = scale_match[1].lower()
        text = text[:scale_match.start()] + " " + text[scale_match.end():]
    for field, names in {"zero_point": ("none", "int", "float"),
                         "observer": ("minmax", "ema", "history", "percentile", "mse")}.items():
        pattern = rf"(?<![a-z0-9_.]){field}\s+({'|'.join(names)})(?![a-z_])"
        values = re.findall(pattern, text, re.I)
        if values:
            result[field] = _unique([value.lower() for value in values], field, questions)
            text = re.sub(pattern, " ", text, flags=re.I)
    result["scheme"] = _unique(_names(text, SCHEMES), "scheme", questions)
    result["format"] = _unique(_format_tokens(text, questions), "format", questions)
    for field, names in {
        "granularity": ("tensor", "row", "group", "block"),
        "rounding": ("rne", "rna", "rtz", "rup", "rdn", "sr"),
        "scale_method": ("absmax", "pow2_floor", "pow2_ceil", "mse", "percentile"),
    }.items():
        result[field] = _unique(_names(text, names), field, questions)
    result["group_size"] = _number(text, "group_size", "QuantChoice.group_size", questions)
    for field, value in scoped.items():
        if result[field] is not None and value is not None and result[field] != value:
            questions.append(f"{field}: 한 필드에 서로 다른 값을 지정했습니다.")
        result[field] = value
    return result if any(value is not None for value in result.values()) else None


def _recipe_options(text: str, questions: list[str]) -> tuple[dict[str, Any], str]:
    options: dict[str, Any] = dict.fromkeys(("kv", "layers", "modules", "skip", "calibration"))
    kv = dict.fromkeys(("preset", "mode", "residual"))
    kv_pattern = r"(?<![a-z0-9_])kv(?:\.preset)?\s*(?:=|:)\s*([^\s,;]+)?"
    explicit = [value.lower().rstrip(".") for value in re.findall(kv_pattern, text, re.I)]
    for value in explicit:
        if value not in KV_PRESETS:
            questions.append(f"kv.preset: 알 수 없는 프리셋 {value!r}입니다.")
    text = re.sub(kv_pattern, " ", text, flags=re.I)
    bare = []
    for name in _names(text, KV_PRESETS):
        matches = list(re.finditer(_pattern(name), text, re.I))
        for match in reversed(matches):
            prefix = text[:match.start()]
            if re.search(r"(?:weights?|activations?|가중치|활성화?|scheme)\s*(?:=|:)?\s*$", prefix, re.I):
                continue
            bare.append(name)
            text = text[:match.start()] + " " + text[match.end():]
    kv["preset"] = _unique([value for value in explicit if value in KV_PRESETS] + bare,
                           "kv.preset", questions)
    mode_pattern = r"(?<![a-z0-9_])(?:(?:kv\.)?mode|모드)(?![a-z0-9_])\s*(?:=|:)?\s*([^\s,;]+)"
    matches = re.findall(mode_pattern, text, re.I)
    if matches:
        values = [value.rstrip(".").lower() for value in matches]
        if any(value not in ("cache", "fakequant") for value in values):
            questions.append("kv.mode: cache 또는 fakequant를 지정해 주세요.")
        else:
            kv["mode"] = _unique(values, "kv.mode", questions)
        text = re.sub(mode_pattern, " ", text, flags=re.I)
    elif kv["preset"]:
        kv["mode"] = _unique(_names(text, ("cache", "fakequant")), "kv.mode", questions)
        text = re.sub(r"\b(?:cache|fakequant)\b", " ", text, flags=re.I)
    kv["residual"] = _number(text, r"(?:kv\.)?residual|잔여", "kv.residual", questions)
    text = re.sub(r"(?<![a-z0-9_])(?:(?:kv\.)?residual|잔여)\s*(?:=|:)?\s*[^\s,;]+",
                  " ", text, flags=re.I)
    if any(value is not None for value in kv.values()):
        options["kv"] = kv
        if kv["preset"] is None:
            questions.append("kv.preset: KV 양자화 프리셋을 지정해 주세요.")
    for field, names in {"layers": r"layers|레이어", "modules": r"modules|모듈"}.items():
        pattern = (rf"(?<![a-z0-9_])(?:{names})(?:는|를|은|을)?(?![a-z0-9_가-힣])\s*(?:=|:)?\s*"
                   r"(\"[^\"]*\"|'[^']*'|\[[^\]]*\]|[^\s,;]+(?:\s*,\s*[^\s,;]+)*)")
        values = []
        for match in re.finditer(pattern, text, re.I):
            value = match[1].strip("\"'[]").rstrip(".")
            values.append(value)
        value = _unique(values, field, questions)
        if value is not None:
            options[field] = ([part.strip(" \"'") for part in value.split(",")]
                              if field == "modules" else value)
        text = re.sub(pattern, " ", text, flags=re.I)
    skip_pattern = (r"(?<![a-z0-9_])(?:skip|제외)(?![a-z0-9_])"
                    r"(?:\s*(?:=|:)\s*([^\s,;]+)|\s+(true|false)(?![a-z]))?")
    matches = list(re.finditer(skip_pattern, text, re.I))
    if matches:
        values = []
        for match in matches:
            value = (match[1] or match[2] or "true").rstrip(".").lower()
            if value not in ("true", "false"):
                questions.append("skip: true 또는 false를 지정해 주세요.")
            else:
                values.append(value == "true")
        options["skip"] = _unique(values, "skip", questions)
        text = re.sub(skip_pattern, " ", text, flags=re.I)
    calibration = dict.fromkeys(("dataset", "samples", "seqlen", "seed", "sequential"))
    dataset_pattern = (r"(?<![a-z0-9_])(?:(?:calibration\.)?dataset|데이터셋)\s*(?:=|:)?\s*"
                       r"([^\s,;]+)")
    datasets = [value.strip("\"'").rstrip(".") for value in re.findall(dataset_pattern, text, re.I)]
    calibration["dataset"] = _unique(datasets, "calibration.dataset", questions)
    text = re.sub(dataset_pattern, " ", text, flags=re.I)
    calibration_context = bool(re.search(
        r"(?<![a-z0-9_.])calibration(?![a-z0-9_.])|교정|보정", text, re.I,
    ))
    for field, names in {"samples": "samples|샘플", "seqlen": "seqlen|시퀀스_길이",
                         "seed": "seed|시드"}.items():
        prefix = r"(?:calibration\.)?" if field == "samples" or calibration_context else r"calibration\."
        names = rf"(?<!\.){prefix}(?:{names})"
        calibration[field] = _number(text, names, f"calibration.{field}", questions)
        text = re.sub(rf"(?<![a-z0-9_.])(?:{names})\s*(?:=|:)?\s*[^\s,;]+", " ", text, flags=re.I)
    sequential_pattern = (r"(?<![a-z0-9_])(?:(?:calibration\.)?sequential|순차)(?![a-z0-9_])"
                          r"(?:\s*(?:=|:)\s*([^\s,;]+)|\s+(true|false)(?![a-z]))?")
    matches = list(re.finditer(sequential_pattern, text, re.I))
    if matches:
        values = []
        for match in matches:
            value = (match[1] or match[2] or "true").rstrip(".").lower()
            if value not in ("true", "false"):
                questions.append("calibration.sequential: true 또는 false를 지정해 주세요.")
            else:
                values.append(value == "true")
        calibration["sequential"] = _unique(values, "calibration.sequential", questions)
        text = re.sub(sequential_pattern, " ", text, flags=re.I)
    if any(value is not None for value in calibration.values()):
        options["calibration"] = calibration
    return options, text


def _sweep(text: str, questions: list[str]) -> tuple[dict[str, Any] | None, str]:
    candidates = []
    for axis, names in _AXIS_NAMES.items():
        prefix = rf"(?<![a-z0-9_])(?:{names})(?![a-z0-9_])(?:를|을)?\s*(?:=|:)?\s*"
        span = re.search(prefix + r"(?:from\s*)?(-?\d+)\s*(?:에서|부터|to|~|\.\.)\s*(-?\d+)"
                         r"(?![\d.a-z])\s*(?:까지)?", text, re.I)
        listed = re.search(prefix + r"\[([^\]]+)\]", text, re.I)
        comma_list = re.search(prefix + r"(?:over\s+)?(-?\d+(?:\s*,\s*-?\d+)+)"
                               r"(?![\d.a-z]|\s*,)(?:으로|로)?", text, re.I)
        match = span or listed or comma_list
        if match is None:
            continue
        values: list[Any] = []
        if span and axis != "c_mode":
            low, high = int(span[1]), int(span[2])
            if 0 <= high - low < 128:
                values = list(range(low, high + 1))
        elif listed or comma_list:
            parts = [value.strip(" '\"") for value in (listed or comma_list)[1].split(",")]
            if axis == "c_mode" and all(value in ("fused", "decoupled") for value in parts):
                values = parts
            elif axis != "c_mode" and all(re.fullmatch(r"-?\d+", value) for value in parts):
                values = [int(value) for value in parts]
        if not values or len(values) > 128:
            questions.append("sweep.values: 유효한 값 목록 또는 오름차순 범위를 128개 이하로 명시해 주세요.")
            return None, text
        candidates.append((axis, values, match))
    if len(candidates) != 1:
        questions.append("sweep: MMA 축 하나와 명시적 값 목록 또는 시작과 끝을 알려 주세요.")
        return None, text
    axis, values, match = candidates[0]
    return {"axis": axis, "values": values}, text[:match.start()] + text[match.end():]


def parse_offline(text: str) -> dict[str, Any]:
    """Return schema-shaped data; unresolved choices stay null and block execution."""
    questions: list[str] = []
    assumptions: list[str] = []
    options, working = _recipe_options(text, questions)
    if (re.search(r"(?<![a-z0-9_.])calibration(?![a-z0-9_.])|교정|보정", text, re.I)
            and re.search(r"(?<![a-z0-9_.])(?:seqlen|seed|시퀀스_길이|시드)\s*(?:=|:)?", text, re.I)
            and (_names(text, TASKS) or re.search(r"\bppl\b|평가|evaluat", text, re.I))):
        questions.append("evaluation/calibration: seqlen/seed의 평가용/보정용 접두어를 지정해 주세요.")
    evaluation = {}
    for field, names in {"seqlen": "seqlen|시퀀스_길이", "seed": "seed|시드"}.items():
        names = rf"(?<!\.)(?:evaluation\.)?(?:{names})"
        evaluation[field] = _number(working, names, f"evaluation.{field}", questions)
        working = re.sub(rf"(?<![a-z0-9_.])(?:{names})\s*(?:=|:)?\s*[^\s,;]+",
                         " ", working, flags=re.I)
    intent = "evaluate"
    for candidate, pattern in (("report", r"\breport\b|리포트|오차\s*분석|레이어별.*(?:오차|MSE|SQNR)"),
                               ("explain", r"설명|explain"), ("inspect", r"조회|inspect|list options"),
                               ("sweep", r"스윕|sweep"), ("compare", r"비교|compare|\bvs\.?\b")):
        if re.search(pattern, text, re.I):
            intent = candidate
            break
    if intent == "report":
        working = re.sub(r"(?<![a-z0-9_.=])(?:MSE|SQNR|cos(?:ine)?|logits\s+KL)(?![a-z0-9_])",
                         " ", working, flags=re.I)
    working = re.sub(r"\b(?:MSE|SQNR|cos)(?:/(?:MSE|SQNR|cos))+\b", " ", working, flags=re.I)
    models = re.findall(r"(?<![\w/])([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)", working)
    models = [value.rstrip(".") for value in models]
    model = _unique(models, "model", questions)
    defer_model = not models and bool(re.search(
        r"모델[^.!?]*(?:나중|미정|정하지|물어|질문)|(?:ask|choose|decide)[^.!?]*\bmodel\b|"
        r"\bmodel\b[^.!?]*(?:later|undecided|ask)", text, re.I,
    ))
    if defer_model:
        questions.append("model: 평가할 Hugging Face 모델을 지정해 주세요.")
    for value in models:
        working = working.replace(value, " ")
    unresolved_models: list[str] = []
    if not models and not defer_model:
        token = r"([A-Za-z][A-Za-z0-9_.-]*)"
        candidates = re.findall(r"\bmodel\s*(?:=|:)?\s*" + token, working, re.I)
        candidates += re.findall(token + r"\s*모델", working, re.I)
        candidates += re.findall(r"\bon\s+" + token, working, re.I)
        candidates += [value for value in re.findall(
            r"(?<![\w./])([A-Za-z][A-Za-z0-9_.]*(?:-[A-Za-z0-9_.]+)+)", working,
        ) if re.search(r"\d", value)]
        known = set(REGISTRY) | set(ALIASES) | set(SCHEMES) | set(TASKS) | set(PRESETS)
        known.update(recipe_names())
        known.update(("ppl", "wikitext2", "a", "the"))
        unresolved_models = list(dict.fromkeys(
            value.rstrip(".") for value in candidates if value.rstrip(".").lower() not in known
        ))
        if unresolved_models:
            questions.append(f"model: {unresolved_models!r}의 정확한 Hugging Face 모델 ID를 지정해 주세요.")
            for value in unresolved_models:
                working = re.sub(_pattern(value), " ", working, flags=re.I)
        else:
            model = "Qwen/Qwen3-0.6B"
            assumptions.append("model 미지정: Qwen/Qwen3-0.6B 기본값을 사용합니다.")
    tasks = _names(working, TASKS)
    has_ppl = re.search(_pattern("ppl") + "|" + _pattern("wikitext2"), working, re.I)
    if has_ppl and "wikitext2_ppl" not in tasks:
        tasks.insert(0, "wikitext2_ppl")
    if tasks and intent in ("report", "explain"):
        if (re.search(r"\blayer\b|레이어별|오차\s*분석|\bMSE\b|\bSQNR\b|logits\s+KL", text, re.I)
                or not re.search(r"평가|evaluat|\brun\b|\bmeasure\b", text, re.I)):
            questions.append("intent: 명시한 과제 평가와 설명/레이어 리포트 중 실행할 작업을 지정해 주세요.")
        else:
            intent = "evaluate"
            for candidate, pattern in (("sweep", r"스윕|sweep"), ("compare", r"비교|compare|\bvs\.?\b")):
                if re.search(pattern, text, re.I):
                    intent = candidate
                    break
    if not tasks and intent not in ("explain", "inspect", "report"):
        tasks = ["wikitext2_ppl"]
        assumptions.append("tasks 미지정: wikitext2_ppl 기본값을 사용합니다.")
    limits: dict[str, Any] = {"max_windows": _number(working, "max_windows", "limits.max_windows", questions),
                              "limit": None}
    limits["limit"] = _numeric(working, "limit", "limits.limit", questions, integer=False)
    if any(value is None for value in limits.values()) and intent not in ("explain", "inspect", "report"):
        assumptions.append("미지정 limits는 null입니다. runner 기본값은 평가량을 제한하지 않습니다.")
    sweep = None
    if intent == "sweep":
        sweep, working = _sweep(working, questions)
    request = {"intent": intent, "model": model, "recipes": [], "sweep": sweep, "tasks": tasks,
               "limits": limits, "assumptions": assumptions, "questions": questions,
               "evaluation": evaluation if any(value is not None for value in evaluation.values()) else None,
               "report_inputs": None,
               "topic": text.strip() if intent in ("explain", "inspect") else None}
    if intent in ("explain", "inspect"):
        return request

    mma = dict.fromkeys(_MMA_FIELDS)
    presets = _names(working, PRESETS)
    preset_compare = intent == "compare" and len(presets) > 1
    mma["preset"] = None if preset_compare else _unique(presets, "mma.preset", questions)
    algorithms = [name for name in _names(working, get_args(Algorithm)) if name not in presets]
    mma["algorithm"] = _unique(algorithms, "mma.algorithm", questions)
    mma["c_mode"] = _unique(_names(working, ("fused", "decoupled")), "mma.c_mode", questions)
    for field, names in {"f_bits": "f_bits|f", "chunk_size": "chunk_size|chunk|cs", "g_bits": "g_bits|g",
                         "group_size": "gs", "promote_interval": "promote_interval|pi"}.items():
        mma[field] = _number(working, names, f"mma.{field}", questions)
    if mma["algorithm"] == "gdfs":
        mma["group_size"] = _number(working, "group_size|gs", "mma.group_size", questions)
    needs_gdfs = mma["g_bits"] is not None or (
        sweep is not None and sweep["axis"] in ("g_bits", "group_size")
    )
    algorithm = PRESETS[mma["preset"]].algorithm if mma["preset"] else mma["algorithm"]
    if needs_gdfs and algorithm != "gdfs":
        questions.append("mma.algorithm: G/GS는 GDFS 그룹 설정입니다. gdfs를 사용할까요?")
    if mma["preset"]:
        assumptions.append(f"mma 미지정 필드는 preset {mma['preset']} 정의를 상속합니다.")
    mma_choice = mma if any(value is not None for value in mma.values()) else None
    transform = _unique(_names(working, [name for name in get_args(TransformKind) if name != "none"]),
                        "transform", questions)
    weight_algo = _unique(_names(working, get_args(WeightAlgo)), "weight_algo", questions)
    bases = _names(working, recipe_names())
    role_matches = list(_ROLE.finditer(working))
    quant_text = working
    for base in bases:
        quant_text = re.sub(_pattern(base), " ", quant_text, flags=re.I)
    if mma["algorithm"] == "gdfs":
        quant_text = re.sub(r"group_size\s*(?:=|:)?\s*\d+", " ", quant_text, flags=re.I)
    for name in _names(quant_text, PRESETS):
        quant_text = re.sub(_pattern(name), " ", quant_text, flags=re.I)

    alternatives: list[tuple[str | None, Any, Any]] = []
    if role_matches:
        operands: dict[str, Any] = {"weight": None, "activation": None}
        before_first = working[:role_matches[0].start()]
        prefix_errors: list[str] = []
        prefix_formats = _format_tokens(before_first, prefix_errors)
        bridge = working[role_matches[0].end():role_matches[1].start()] if len(role_matches) > 1 else ""
        postfix_bridge = bool(re.match(r"\s*(?:and\b|와|과|및|&|,)", bridge, re.I)) and bool(
            _names(bridge, SCHEMES) or _format_tokens(bridge, [])
        )
        postfix = bool(re.search(r"[a-z][a-z0-9_]*(?::[a-z0-9_=+-]+)*\s*$", before_first, re.I)) and bool(
            _names(before_first, SCHEMES) or prefix_formats or prefix_errors or postfix_bridge
        )
        for index, match in enumerate(role_matches):
            key = "weight" if match[0].lower().startswith("weight") or match[0] == "가중치" else "activation"
            if postfix:
                start = role_matches[index - 1].end() if index else 0
                segment = working[start:match.start()]
            else:
                end = role_matches[index + 1].start() if index + 1 < len(role_matches) else len(working)
                segment = working[match.end():end]
            choice = _quant(segment, questions)
            if operands[key] is not None:
                questions.append(f"{key}: 중복된 피연산자 설정을 하나로 명시해 주세요.")
            operands[key] = choice
        if postfix:
            trailing = _quant(working[role_matches[-1].end():], questions)
            if trailing is not None:
                explicit = {field: value for field, value in trailing.items() if value is not None}
                questions.append(f"weight/activation: 후행 설정 {explicit}을 어느 피연산자에 적용할까요?")
        if len(role_matches) == 2:
            bridge = working[role_matches[0].end():role_matches[1].start()].strip()
            shared = re.fullmatch(r"(?:와|과|및|and|&|,|\s)*", bridge, re.I)
            if shared and sum(value is not None for value in operands.values()) == 1:
                choice = next(value for value in operands.values() if value is not None)
                operands = {"weight": copy.deepcopy(choice), "activation": copy.deepcopy(choice)}
        for key, choice in operands.items():
            if choice is None and any(
                (match[0].lower().startswith("weight") or match[0] == "가중치") == (key == "weight")
                for match in role_matches
            ):
                questions.append(f"{key}: 명시한 형식 또는 스킴을 해석할 수 없습니다. 이름을 지정해 주세요.")
        alternatives = [(bases[0] if len(bases) == 1 else None, operands["weight"], operands["activation"])]
        if len(bases) > 1:
            questions.append("base: 피연산자 설정을 어느 레시피에 적용할까요?")
    elif preset_compare:
        choice = _quant(quant_text, questions)
        base = bases[0] if len(bases) == 1 else None
        alternatives = [(base, copy.deepcopy(choice), copy.deepcopy(choice)) for _ in presets]
        if len(bases) > 1:
            questions.append("base: 프리셋 비교에 사용할 기본 레시피 하나를 지정해 주세요.")
        if choice is None and base is None:
            questions.append("weight/activation: 프리셋 비교에 사용할 입력 형식 또는 스킴을 지정해 주세요.")
        else:
            assumptions.append("피연산자 미지정: 각 프리셋의 weight와 activation에 같은 스킴을 적용합니다.")
    elif bases:
        choice = _quant(quant_text, questions)
        alternatives = [(name, copy.deepcopy(choice), copy.deepcopy(choice)) for name in bases]
        if choice is not None:
            assumptions.append("피연산자 미지정: 추가 형식/스킴을 base의 weight와 activation에 적용합니다.")
    elif intent == "compare":
        choices = _names(quant_text, set(SCHEMES) | set(_format_tokens(quant_text, questions)))
        modifiers = quant_text
        for name in choices:
            modifiers = re.sub(_pattern(name), " ", modifiers, flags=re.I)
        alternatives = [(None, _quant(name + " " + modifiers, questions),
                         _quant(name + " " + modifiers, questions)) for name in choices]
        if alternatives:
            assumptions.append("피연산자 미지정: 각 비교 형식/스킴을 weight와 activation 모두에 적용합니다.")
    else:
        choice = _quant(quant_text, questions)
        if choice is not None:
            alternatives = [(None, choice, copy.deepcopy(choice))]
            assumptions.append("피연산자 미지정: 명시한 형식/스킴을 weight와 activation 모두에 적용합니다.")
        elif options["kv"] is not None:
            alternatives = [(None, None, None)]
        elif mma_choice:
            alternatives = [(None, None, None)]
            questions.append("weight/activation: 입력 형식 또는 기존 레시피를 지정해 주세요.")

    if preset_compare and role_matches:
        alternatives *= len(presets)
    for index, (base, weight, activation) in enumerate(alternatives):
        name = base or f"request_{index + 1}"
        request["recipes"].append({"name": name, "base": base, "weight": weight, "activation": activation,
                                   "mma": copy.deepcopy(mma_choice), "transform": transform,
                                   "weight_algo": weight_algo, **copy.deepcopy(options)})
        if preset_compare:
            request["recipes"][-1]["mma"] = {**mma, "preset": presets[index]}
            assumptions.append(f"mma 미지정 필드는 preset {presets[index]} 정의를 상속합니다.")
        if base:
            assumptions.append(f"미지정 레시피 필드는 base {base} 정의를 상속합니다.")
        for role, choice in (("weight", weight), ("activation", activation)):
            if choice is not None and not base and not choice["scheme"] and not choice["format"]:
                questions.append(f"recipes[{index}].{role}: format 또는 scheme을 지정해 주세요.")
    if not alternatives:
        questions.append("recipes: 평가할 형식, 스킴 또는 기존 레시피를 지정해 주세요.")
    if intent == "compare" and len(alternatives) < 2:
        questions.append("compare: 비교할 두 개 이상의 레시피/스킴을 지정해 주세요.")
    return request
