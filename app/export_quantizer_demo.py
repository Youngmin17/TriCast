"""Export the data the Studio Quantizer demo (``app/web/quantizer.html``) reads: the recipe generator's
tables, the presets, the method catalog and the recorded GPU runs that belong to a preset.

Pure stdlib (json, hashlib, pathlib) plus ``app.quantizer_recipe``: no torch, no tricast.

    python -m app.export_quantizer_demo --runs runs/20261013_quantizer_w2 \\
        --native runs/20261013_quantizer_w1/ppl_base/native.json \\
        [--lmeval runs/20261013_quantizer_w3] --out app/web/demo/quantizer

Writes ``<out>/presets.json`` (indent=1, sorted keys, UTF-8; byte-identical for the same inputs).
Mappings in ``generator`` (``app.quantizer_recipe``'s METHODS, KV, PARENT and the option tables OPTIONS,
METHOD_OPTIONS, KV_OPTIONS, CALIBRATION) are written as ``{"items": [[key, value], ...]}`` so the page keeps
their key order, which is the YAML key order. The page's words for the options are ``OPTION_INFO``.

Recorded results are looked up by preset id: WikiText-2 PPL at ``<runs>/ppl_*/<id>.json``, per-layer
reports at ``<runs>/report/<id>/report.json`` and lm-eval runs at ``<lmeval>/lmeval_*/<id>.json`` (BF16:
``<lmeval>/lmeval_*/native.json``). A record is attached only if it is finished, its ``recipe.name`` is the
preset id and its recipe is the preset's current recipe; otherwise the field stays null with status
"기록 없음", ``<field>_rejected`` says why and the export prints it. tricast records recipes in normalised
form (``Recipe.to_dict``), so "the same recipe" means: the SHA-256 of the recorded dict's canonical JSON
equals the file's ``recipe_hash`` and the preset's tricast SHA-256 in ``PINS``. A pin counts only while the
preset's YAML still has the digest it was derived from, so a changed generator never vouches for old runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app import quantizer_recipe as qr

SCHEMA = "tricast.quantizer_demo/1"
MODEL_ID = "Qwen/Qwen3-0.6B"
REPO = Path(__file__).resolve().parents[1]
MISSING = "기록 없음"
RECORDED = "기록됨"

# preset id -> (SHA-256 of to_yaml(preset(id)), tricast load_recipe(that YAML).sha256), tricast at ab489f1.
# Re-derive with runs/20261013_quantizer_w1/layers_tests/make_qz_fixtures.py (repo venv) when a preset
# changes.
PINS = {
    "qd_mxfp4": ("b90963f4b9cacf26b0980beff357ccecc108d5bb3b32de82fb209d9bee5d3b0e",
                 "e6f0c97f769b840eef7d2bd2e8d7df7dc04fc1dd7455c16e22598702c315bb2d"),
    "qd_mxfp4_ends_bf16": ("4b7702f0c7bef641e868c32a2856a803b78f189d2d9969849a4e67a77b59101d",
                           "483ec568f4a06eb819cb40f3c544b5106f38e7607793d060fedd40e2631e0bad"),
    "qd_nvfp4": ("140a734d1fcd47b9dc1855ca5d49c3f4fe81ca88a1e1f971f3214e45c9d5976f",
                 "9f2612422bcc5f145620219bdf23aaf768198f816b4dbe12454a55fbb30e9c48"),
    "qd_nvint4": ("1af60a8968c38b9236d329183686f1333974c8cbeb8cfd440d7a43f091fb5b89",
                  "ba9c8928b96f286d1f83935af0ccd8335b283c626f4f3aadf21c38d74d1f0dc9"),
    "qd_mixed": ("ed27d580cbcd3e5e94a484e68129879b4a95941804356a66502df42222b58f33",
                 "11b6ac48769499219789f04e3012f2b13cb163f36dd72d849acba12ff65634e4"),
}

# Qwen3-0.6B Linear shapes (in_features, out_features) per kind.
SHAPES = {"q_proj": (1024, 2048), "k_proj": (1024, 1024), "v_proj": (1024, 1024), "o_proj": (2048, 1024),
          "gate_proj": (1024, 3072), "up_proj": (1024, 3072), "down_proj": (3072, 1024)}

# Stored weight bits per element: (element bits, scale bits, elements per scale); None = one scale per output
# row, i.e. per in_features elements. NVFP4's single fp32 tensor scale is left out (one per tensor).
WEIGHT_BITS: dict[str, tuple[int, int, int | None]] = {
    "nvfp4": (4, 8, 16), "awq_nvfp4": (4, 8, 16), "smoothquant_nvfp4": (4, 8, 16), "nvint4": (4, 8, 16),
    "bfp12": (4, 8, 16), "mxfp4": (4, 8, 32), "mxfp8": (8, 8, 32),
    "gptq_int4": (4, 20, 128),  # uint4, per 128: fp16 scale + 4-bit zero point
    "int8": (8, 32, None),      # fp32 scale per output row
    "bf16": (16, 0, 1),
}

# Page order (families together): id, label, code, family, W/A bits, one-line description.
METHOD_INFO = (
    ("nvfp4", "NVFP4", "NV4", "fp4", "W4 · A4",
     "FP4(E2M1) 원소 16개마다 UE4M3 스케일, 텐서마다 FP32 스케일"),
    ("awq_nvfp4", "AWQ + NVFP4", "AWQ", "fp4", "W4 · A4",
     "보정 데이터로 찾은 AWQ 채널 스케일을 적용한 뒤 NVFP4"),
    ("smoothquant_nvfp4", "SmoothQuant + NVFP4", "SQ", "fp4", "W4 · A4",
     "활성 이상치를 가중치 쪽으로 옮긴 뒤(α 0.5) NVFP4"),
    ("mxfp4", "MXFP4", "MX4", "fp4", "W4 · A4",
     "FP4(E2M1) 원소 32개마다 E8M0(2의 거듭제곱) 스케일"),
    ("gptq_int4", "GPTQ INT4", "GQ", "int4", "W4 · A16",
     "가중치만 GPTQ 로 UINT4(128개 그룹, FP16 스케일 + 영점), 활성은 BF16"),
    ("nvint4", "NVINT4", "NI4", "int4", "W4 · A4",
     "NVFP4 의 블록 구조(16개마다 UE4M3 스케일)에 대칭 INT4 원소"),
    ("mxfp8", "MXFP8", "MX8", "8bit", "W8 · A8",
     "FP8(E4M3) 원소 32개마다 E8M0 스케일"),
    ("int8", "INT8", "I8", "8bit", "W8 · A8",
     "가중치는 출력 행마다, 활성은 토큰마다 FP32 스케일 하나인 INT8"),
    ("bfp12", "BFP (MSFP12)", "B12", "bfp", "W4 · A4",
     "4비트 원소 16개가 8비트 지수 하나를 공유하는 블록 부동소수점"),
    ("bf16", "BF16 유지", "BF", "bf16", "W16 · A16",
     "양자화하지 않고 원래 BF16 으로 둡니다"),
)
NOTES = {"nvint4": "새 형식 · NVFP4 구조 + INT4"}
# What a method's options panel says is fixed, and why (the format's definition, not a page choice).
_NV16 = "블록 16 · UE4M3 블록 스케일 + FP32 텐서 스케일 (NVFP4 정의)"
FIXED = {
    "nvfp4": _NV16, "awq_nvfp4": _NV16, "smoothquant_nvfp4": _NV16,
    "nvint4": "블록 16 · UE4M3 + FP32 스케일 (NVFP4 구조, 원소만 INT4)",
    "mxfp4": "블록 32 · E8M0 스케일 pow2_floor (OCP MX 정의)",
    "mxfp8": "블록 32 · E8M0 스케일 pow2_floor (OCP MX 정의)",
    "bfp12": "블록 16 · 8비트 공유 지수 (MSFP12 정의)",
    "gptq_int4": "UINT4 · FP16 스케일 + 영점 — 범위를 [min, max] 로 정해 스케일 고르기가 없음 · 활성 BF16",
    "int8": "가중치는 출력 행마다 FP32 스케일 하나",
}
_ROUNDING_WORDS = {"rne": "rne (최근접, 동률 짝수)", "rtz": "rtz (0 쪽 버림)", "sr": "sr (확률적)"}
_SCALE_WORDS = {"absmax": "absmax (최댓값)", "mse": "mse (오차 최소 비율)",
                "percentile": "percentile (분위수)"}
# option id -> (label, choice words or None, one-line reason shown while the option does not apply or None).
# Without a reason, an option that does not apply is a sub-field of another choice and is hidden instead.
OPTION_INFO: dict[str, tuple[str, dict[str, str] | None, str | None]] = {
    "weight.group_size": ("가중치 그룹 크기 (원소 수)", None, None),
    "weight.scale.method": ("가중치 스케일 고르기", _SCALE_WORDS, None),
    "weight.scale.percentile": ("가중치 분위수 (%)", None, None),
    "weight.rounding": ("가중치 원소 반올림", _ROUNDING_WORDS, None),
    "activation.granularity": ("활성 스케일 단위", {"row": "토큰마다 (row)", "tensor": "텐서 하나 (tensor)"},
                               None),
    "activation.scale.method": ("활성 스케일 고르기", _SCALE_WORDS,
                                "정적 observer 가 amax 를 정하므로 활성 스케일은 absmax 로만 계산됩니다"),
    "activation.scale.percentile": ("활성 분위수 (%)", None, None),
    "activation.rounding": ("활성 원소 반올림", _ROUNDING_WORDS, None),
    "activation.observer.kind": ("활성 observer (스케일 출처)",
                                 {"none": "없음 (동적, 호출마다)", "minmax": "minmax (보정 최댓값)",
                                  "ema": "ema (이동 평균)", "history": "history (지난 호출, 보정 없음)",
                                  "percentile": "percentile (보정 분위수)"},
                                 "정적 observer 는 텐서 하나의 스케일에서만 됩니다 — "
                                 "활성 스케일 단위를 '텐서 하나'로 바꾸세요"),
    "activation.observer.decay": ("EMA decay (0 이상 1 미만)", None, None),
    "activation.observer.history_len": ("history 길이 (호출 수)", None, None),
    "activation.observer.percentile": ("observer 분위수 (%)", None, None),
    "transform.alpha": ("SmoothQuant α (0–1, 가중치로 옮기는 비율)", None, None),
    "transform.grid": ("AWQ 탐색 점 수 (α 0–1)", None, None),
    "transform.share_inputs": ("같은 입력을 받는 Linear 끼리 스케일 공유 (q·k·v, gate·up)", None, None),
    "weight_algo.damp": ("GPTQ damp (헤시안 대각 평균의 비율)", None, None),
    "weight_algo.act_order": ("GPTQ act_order (헤시안 대각이 큰 열부터)", None, None),
    "weight_algo.block_size": ("GPTQ 블록 크기 (한 번에 갱신하는 열 수)", None, None),
}
CALIBRATION_INFO = {"dataset": ("데이터셋", {"wikitext2": "WikiText-2", "c4": "C4", "pile": "Pile"}),
                    "samples": ("샘플 수 (창 개수)", None), "seqlen": ("창 길이 (토큰)", None),
                    "seed": ("seed", None),
                    "sequential": ("순차 보정 (앞 블록을 양자화한 상태로 다음 블록 통계)", None)}
KV_OPTION_INFO = {"residual": "residual (원정밀로 두는 최근 토큰 수)"}
KV_INFO = (  # id, label, short, description
    ("none", "없음", "BF16", "KV 캐시는 BF16 그대로"),
    ("kivi2", "KIVI-2", "UINT2 · 그룹 32", "K·V 를 UINT2 로, 32개 그룹마다 FP16 스케일과 영점"),
    ("kivi4", "KIVI-4", "UINT4 · 그룹 32", "K·V 를 UINT4 로, 32개 그룹마다 FP16 스케일과 영점"),
    ("kv_fp8", "FP8", "E4M3 · 텐서", "K·V 를 FP8(E4M3)로, 텐서마다 스케일 하나"),
)
# lm-eval tasks the page shows: (task, reported metric, label); acc_norm where lm-eval has it, else acc.
BENCHMARKS = (("arc_easy", "acc_norm", "ARC-Easy"), ("arc_challenge", "acc_norm", "ARC-Challenge"),
              ("piqa", "acc_norm", "PIQA"), ("winogrande", "acc", "WinoGrande"),
              ("hellaswag", "acc_norm", "HellaSwag"))
ENV_KEYS = ("git_sha", "git_dirty", "gpu_names", "model_id", "model_sha", "utc", "cuda", "dtype", "seed")
MODEL_KEYS = ("logits_kl", "top1_agreement", "ppl_reference", "ppl_emulated", "n_tokens")
NONFINITE = ("inf", "-inf", "nan")


class ExportError(ValueError):
    """An input that cannot be used; the message names the file and what to do."""


def _number(value: Any, where: str) -> float | int | str | None:
    """A finite number, a non-finite token ("inf", "-inf", "nan") or None."""
    if value is None or (isinstance(value, str) and value in NONFINITE):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{where}: expected a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
    return value


def _finite(value: Any, where: str) -> float | int:
    number = _number(value, where)
    if not isinstance(number, (int, float)):
        raise ExportError(f"{where} is {number!r}, expected a finite number")
    return number


def _source(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO)) if resolved.is_relative_to(REPO) else str(resolved)


def read_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"{path}: cannot read JSON ({error})") from error
    if not isinstance(record, dict):
        raise ExportError(f"{path}: not a JSON object")
    if record.get("status") != "complete":
        raise ExportError(f"{path}: status is {record.get('status')!r}, expected 'complete' "
                          "(only finished runs are shown)")
    return record


def env_view(path: Path | str, record: dict[str, Any], model_id: str = MODEL_ID) -> dict[str, Any]:
    env = record.get("env")
    if not isinstance(env, dict) or env.get("model_id") != model_id:
        raise ExportError(f"{path}: env.model_id is not {model_id!r}; the demo grid is for that model")
    missing = [key for key in ("git_sha", "git_dirty", "model_sha", "gpu_names") if key not in env]
    if missing:
        raise ExportError(f"{path}: env is missing {', '.join(missing)} (results need their environment)")
    return {key: env[key] for key in ENV_KEYS if key in env}


def recipe_sha256(recipe: dict[str, Any]) -> str:
    """tricast's ``Recipe.sha256`` of a recorded (normalised) recipe dict: SHA-256 of its canonical JSON."""
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def preset_sha256(preset_id: str) -> str | None:
    """The preset's pinned tricast SHA-256, or None when the pin was derived from a different preset YAML."""
    digest = hashlib.sha256(qr.to_yaml(qr.preset(preset_id)).encode("utf-8")).hexdigest()
    yaml_pin, recipe_pin = PINS.get(preset_id, (None, None))
    return recipe_pin if digest == yaml_pin else None


def check_recipe(path: Path, record: dict[str, Any], preset_id: str) -> str:
    """The honesty check: the record names the preset and holds exactly its recipe. Returns the SHA-256."""
    recipe = record.get("recipe")
    name = recipe.get("name") if isinstance(recipe, dict) else None
    if name != preset_id:
        raise ExportError(f"{path}: recipe.name is {name!r}, expected {preset_id!r}; a run is shown only for "
                          "the preset it was made from")
    digest = recipe_sha256(recipe)
    if record.get("recipe_hash") != digest:
        raise ExportError(f"{path}: recipe_hash {str(record.get('recipe_hash'))[:12]} does not match the "
                          f"recorded recipe (SHA-256 {digest[:12]}); the file was edited after the run")
    expected = preset_sha256(preset_id)
    if expected is None:
        raise ExportError(f"{path}: preset {preset_id} changed in app/quantizer_recipe.py since its tricast "
                          "SHA-256 was pinned; re-derive PINS (make_qz_fixtures.py) before showing its runs")
    if digest != expected:
        raise ExportError(f"{path}: the recorded recipe is not preset {preset_id} (SHA-256 {digest[:12]}, "
                          f"preset {expected[:12]}); run it again with the YAML from "
                          f"python -m app.quantizer_recipe --preset {preset_id}")
    return digest


# ---------- record views: the shapes the page renders (also for live jobs, app/runners/quantizer.py) ----
def ppl_view(path: Path | str, record: dict[str, Any], model_id: str = MODEL_ID) -> dict[str, Any]:
    metrics = record.get("metrics")
    ppl = metrics.get("ppl") if isinstance(metrics, dict) else None
    if not isinstance(ppl, dict) or not all(key in ppl for key in ("ppl", "n_windows", "n_tokens",
                                                                    "dataset_fingerprint")):
        raise ExportError(f"{path}: metrics.ppl needs ppl, n_windows, n_tokens and dataset_fingerprint")
    return {"ppl": _finite(ppl["ppl"], f"{path}: metrics.ppl.ppl"), "n_windows": ppl["n_windows"],
            "n_tokens": ppl["n_tokens"], "dataset_fingerprint": ppl["dataset_fingerprint"],
            "forward_mode": ppl.get("forward_mode"), "recipe_hash": record.get("recipe_hash"),
            "env": env_view(path, record, model_id), "source": _source(Path(path)),
            "wall_time_s": _number(record.get("wall_time_s"), f"{path}: wall_time_s"), "ppl_delta_pct": None}


def report_view(path: Path | str, record: dict[str, Any], model_id: str = MODEL_ID) -> dict[str, Any]:
    """Output SQNR of the 196 decoder Linear layers in block × kind order (null = not measured, e.g. BF16)."""
    model = record.get("model")
    layers = record.get("layers")
    if not isinstance(model, dict) or not isinstance(layers, list):
        raise ExportError(f"{path}: needs model metrics and a layers list (a tricast report.json)")
    by_name: dict[str, dict[str, Any]] = {}
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or not isinstance(layer.get("name"), str) or layer["name"] in by_name:
            raise ExportError(f"{path}: layers[{index}] needs a unique string name")
        by_name[layer["name"]] = layer
    sqnr = []
    for b in range(qr.BLOCKS):
        for kind in qr.KINDS:
            output = by_name.get(qr.layer_name(b, kind), {}).get("output")
            value = output.get("sqnr_db") if isinstance(output, dict) else None
            sqnr.append(_number(value, f"{path}: {qr.layer_name(b, kind)}.output.sqnr_db"))
    return {"sqnr_db": sqnr, "measured": sorted(by_name), "recipe_hash": record.get("recipe_hash"),
            "env": env_view(path, record, model_id), "source": _source(Path(path)),
            **{key: _number(model.get(key), f"{path}: model.{key}") for key in MODEL_KEYS},
            **{key: record.get(key) for key in ("samples", "seqlen", "dataset_fingerprint")}}


def lmeval_view(path: Path | str, record: dict[str, Any], tasks: Sequence[str] | None = None,
                model_id: str = MODEL_ID) -> dict[str, Any]:
    """One score per BENCHMARKS task (all of them unless ``tasks`` names some) with its stderr and evaluated
    document count, plus the macro average over those tasks."""
    metrics = record.get("metrics")
    lm_eval = metrics.get("lm_eval") if isinstance(metrics, dict) else None
    if not isinstance(lm_eval, dict) or not isinstance(lm_eval.get("results"), dict):
        raise ExportError(f"{path}: metrics.lm_eval.results is missing (not an lm-eval run)")
    samples = lm_eval.get("n-samples") if isinstance(lm_eval.get("n-samples"), dict) else {}
    shots = lm_eval.get("n-shot") if isinstance(lm_eval.get("n-shot"), dict) else {}
    rows = []
    for task, metric, label in BENCHMARKS:
        if tasks is not None and task not in tasks:
            continue
        scores = lm_eval["results"].get(task)
        if not isinstance(scores, dict) or f"{metric},none" not in scores:
            raise ExportError(f"{path}: lm-eval has no {metric} for {task}; the page needs all of "
                              f"{', '.join(item[0] for item in BENCHMARKS)}")
        stderr = scores.get(f"{metric}_stderr,none")
        stderr = stderr if isinstance(stderr, (int, float)) and math.isfinite(stderr) else None
        count = samples.get(task, {})
        rows.append({"task": task, "label": label, "metric": metric,
                     "value": _finite(scores[f"{metric},none"], f"{path}: {task}.{metric}"),
                     "stderr": stderr,
                     "n": count.get("effective") if isinstance(count, dict) else count,
                     "num_fewshot": shots.get(task), "delta_pts": None})
    config = lm_eval.get("config") if isinstance(lm_eval.get("config"), dict) else {}
    batch = lm_eval.get("batch_size")
    env = record.get("env") if isinstance(record.get("env"), dict) else {}
    versions = env.get("versions") if isinstance(env.get("versions"), dict) else {}
    return {"tasks": rows, "macro": sum(item["value"] for item in rows) / len(rows) if rows else None,
            "macro_delta_pts": None, "limit": config.get("limit"),
            "batch_size": batch.get("used") if isinstance(batch, dict) else batch,
            "lm_eval_version": versions.get("lm_eval"), "recipe_hash": record.get("recipe_hash"),
            "env": env_view(path, record, model_id), "source": _source(Path(path)),
            "wall_time_s": _number(record.get("wall_time_s"), f"{path}: wall_time_s")}


def compare_ppl(run: dict[str, Any], native: dict[str, Any] | None) -> dict[str, Any]:
    if native is not None:
        pairs = {key: (run[key], native[key]) for key in ("dataset_fingerprint", "n_windows", "n_tokens")}
        pairs["model_sha"] = (run["env"]["model_sha"], native["env"]["model_sha"])
        differ = [f"{key} {a!r} vs {b!r}" for key, (a, b) in pairs.items() if a != b]
        if differ:
            raise ExportError(f"{run['source']}: not measured on the same windows as the BF16 run "
                              f"{native['source']} ({'; '.join(differ)}); a change against BF16 would "
                              "mix data")
        run["ppl_delta_pct"] = (run["ppl"] / native["ppl"] - 1.0) * 100.0
    return run


def compare_lmeval(run: dict[str, Any], native: dict[str, Any] | None) -> dict[str, Any]:
    if native is not None:
        pairs = zip(run["tasks"], native["tasks"], strict=True)
        differ = [f"{a['task']} n {a['n']!r} vs {b['n']!r}" for a, b in pairs
                  if (a["n"], a["num_fewshot"]) != (b["n"], b["num_fewshot"])]
        if run["limit"] != native["limit"] or run["env"]["model_sha"] != native["env"]["model_sha"]:
            differ.append(f"limit {run['limit']!r} vs {native['limit']!r}")
        if differ:
            raise ExportError(f"{run['source']}: not the same documents as the BF16 lm-eval run "
                              f"{native['source']} ({'; '.join(differ)})")
        for item, base in zip(run["tasks"], native["tasks"], strict=True):
            item["delta_pts"] = (item["value"] - base["value"]) * 100.0
        run["macro_delta_pts"] = (run["macro"] - native["macro"]) * 100.0
    return run


# ---------- the bundle ----------
def _ordered(value: Any) -> Any:
    """Mappings as {"items": [[key, value], ...]}, recursively: JSON with sorted keys keeps their order."""
    if isinstance(value, dict):
        return {"items": [[key, _ordered(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return [_ordered(item) for item in value]
    return value


def generator() -> dict[str, Any]:
    return {"methods": _ordered(qr.METHODS), "calibrated": sorted(qr.CALIBRATED), "kv": _ordered(qr.KV),
            "description": qr.DESCRIPTION, "kinds": list(qr.KINDS), "parent": _ordered(qr.PARENT),
            "blocks": qr.BLOCKS, "options": _ordered(qr.OPTIONS),
            "method_options": _ordered(qr.METHOD_OPTIONS), "kv_options": _ordered(qr.KV_OPTIONS),
            "calibration": _ordered(qr.CALIBRATION),
            "complete": _ordered({"transform": qr.TRANSFORM_FIELDS, "weight_algo": qr.WEIGHT_ALGO_FIELDS}),
            "calibrating_observers": list(qr.CALIBRATING_OBSERVERS), "float_min": qr.FLOAT_MIN}


def _bits(method: str, kind: str) -> float:
    element, scale, per = WEIGHT_BITS[method]
    return element + scale / (per or SHAPES[kind][0])


def catalog() -> dict[str, Any]:
    methods_now = sorted(qr.METHODS)
    if sorted(row[0] for row in METHOD_INFO) != methods_now or sorted(WEIGHT_BITS) != methods_now:
        raise ExportError("METHOD_INFO / WEIGHT_BITS do not list exactly the methods of app.quantizer_recipe."
                          "METHODS; add the new method's label, code and bits here")
    if sorted(OPTION_INFO) != sorted(qr.OPTIONS) or sorted(CALIBRATION_INFO) != sorted(qr.CALIBRATION) \
            or {key for spec in qr.KV_OPTIONS.values() for key in spec} - set(KV_OPTION_INFO):
        raise ExportError("OPTION_INFO / CALIBRATION_INFO / KV_OPTION_INFO do not name exactly the options "
                          "of app.quantizer_recipe; add the new option's words here")
    methods = []
    for method, label, code, family, wa, description in METHOD_INFO:
        element, scale, per = WEIGHT_BITS[method]
        methods.append({"id": method, "label": label, "code": code, "family": family, "wa_bits": wa,
                        "description": description, "note": NOTES.get(method),
                        "calibration": method in qr.CALIBRATED, "skip": qr.METHODS[method] is None,
                        "bits_note": "16" if method == "bf16" else
                        f"{element} + {scale}/{per if per else 'in_features'}",
                        "weight_bits": {kind: _bits(method, kind) for kind in qr.KINDS},
                        "bits": [element, scale, per], "fixed": FIXED.get(method)})
    kinds = [{"id": kind, "parent": qr.PARENT[kind], "in_features": n_in, "out_features": n_out,
              "params": n_in * n_out} for kind, (n_in, n_out) in SHAPES.items()]
    return {"methods": methods, "kinds": kinds,
            "kv": [{"id": kv, "label": label, "short": short, "description": text}
                   for kv, label, short, text in KV_INFO],
            "benchmarks": [{"task": task, "metric": metric, "label": label}
                           for task, metric, label in BENCHMARKS],
            "option_info": {option: {"label": label, "choices": choices, "unavailable": reason}
                            for option, (label, choices, reason) in OPTION_INFO.items()},
            "calibration_info": {key: {"label": label, "choices": choices}
                                 for key, (label, choices) in CALIBRATION_INFO.items()},
            "kv_option_info": {key: {"label": label} for key, label in KV_OPTION_INFO.items()}}


def _attach(paths: list[Path], preset_id: str, view, compare=None, native=None,
            grid=None) -> tuple[Any, str | None]:
    """(view, None) for an attachable record, (None, reason) for a rejected one, (None, None) when absent."""
    if not paths:
        return None, None
    try:
        if len(paths) > 1:
            raise ExportError(f"{preset_id} is recorded more than once: {', '.join(map(str, paths))}; "
                              "keep one")
        record = read_record(paths[0])
        check_recipe(paths[0], record, preset_id)
        result = view(paths[0], record)
        if grid is not None:
            quantized = sorted(qr.layer_name(b, kind) for b in range(qr.BLOCKS)
                               for k, kind in enumerate(qr.KINDS) if grid[b][k] != "bf16")
            if result["measured"] != quantized:
                extra = sorted(set(result["measured"]) - set(quantized))
                lost = sorted(set(quantized) - set(result["measured"]))
                raise ExportError(f"{paths[0]}: measured layers differ from the preset's quantized layers "
                                  f"(unexpected {extra[:3]}, missing {lost[:3]}); the patch did not follow "
                                  "the grid")
        return (compare(result, native) if compare else result), None
    except ExportError as error:
        return None, str(error)


def _native(paths: list[Path], view, what: str) -> dict[str, Any] | None:
    if not paths:
        return None
    if len(paths) > 1:
        raise ExportError(f"{what}: more than one BF16 run ({', '.join(map(str, paths))}); keep one")
    record = read_record(paths[0])
    if record.get("recipe_hash") != "native":
        raise ExportError(f"{paths[0]}: recipe_hash is {record.get('recipe_hash')!r}, expected 'native' "
                          "(the unquantized BF16 run)")
    return view(paths[0], record)


def build(runs: Path, native_path: Path, lmeval: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """The bundle and the reasons records were not attached."""
    native = _native([native_path] if native_path.is_file() else [], ppl_view, "--native")
    lm_paths = sorted(lmeval.glob("lmeval_*/native.json")) if lmeval else []
    lm_native = _native(lm_paths, lmeval_view, "--lmeval")
    rejected: list[str] = []
    presets = []
    for preset_id, (default, grid, kv, label) in qr.PRESETS.items():
        found = {"run": _attach(sorted(runs.glob(f"ppl_*/{preset_id}.json")), preset_id, ppl_view,
                                compare_ppl, native),
                 "report": _attach([path for path in [runs / "report" / preset_id / "report.json"]
                                    if path.is_file()], preset_id, report_view, grid=grid),
                 "lmeval": _attach(sorted(lmeval.glob(f"lmeval_*/{preset_id}.json")) if lmeval else [],
                                   preset_id, lmeval_view, compare_lmeval, lm_native)}
        entry = {"id": preset_id, "label": label, "default": default, "grid": grid, "kv": kv,
                 "yaml": qr.to_yaml(qr.preset(preset_id)), "recipe_sha256": preset_sha256(preset_id)}
        for what, (value, reason) in found.items():
            entry.update({what: value, f"{what}_status": RECORDED if value else MISSING,
                          f"{what}_rejected": reason})
            if reason:
                rejected.append(reason)
        presets.append(entry)
    bundle = {"schema": SCHEMA, "model": {"id": MODEL_ID, "blocks": qr.BLOCKS}, "generator": generator(),
              **catalog(), "native": native, "native_status": RECORDED if native else MISSING,
              "lmeval_native": lm_native, "lmeval_native_status": RECORDED if lm_native else MISSING,
              "presets": presets}
    return bundle, rejected


def dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.export_quantizer_demo",
                                     description="Export presets and recorded runs for the Quantizer demo.")
    parser.add_argument("--runs", type=Path, required=True, help="run directory with ppl_*/ and report/")
    parser.add_argument("--native", type=Path, required=True, help="the BF16 PPL run on the same windows")
    parser.add_argument("--lmeval", type=Path,
                        help="run directory with lmeval_*/<id>.json and one native.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory (Studio reads app/web/demo/quantizer)")
    args = parser.parse_args(argv)
    try:
        payload, rejected = build(args.runs, args.native, args.lmeval)
        text = dump(payload)
    except ExportError as error:
        print(f"export_quantizer_demo: {error}", file=sys.stderr)
        return 1
    for reason in rejected:
        print(f"export_quantizer_demo: not shown: {reason}", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "presets.json").write_text(text, encoding="utf-8")
    print(f"BF16 PPL {payload['native_status']} · BF16 lm-eval {payload['lmeval_native_status']}")
    for preset in payload["presets"]:
        found = [f"{what} {preset[what + '_status']}" + (" (거부됨)" if preset[what + "_rejected"] else "")
                 for what in ("run", "report", "lmeval")]
        print(f"{preset['id']:<20} {' · '.join(found)}")
    print(f"{args.out / 'presets.json'}  {len(text.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
