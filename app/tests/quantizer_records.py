"""Records for the Quantizer and layer-map tests.

``data/quantizer`` holds the five preset YAMLs exactly as the recorded GPU runs used them, their
tricast-normalised recipes (``Recipe.to_dict``, what a runner records as ``recipe``; re-derived by
scripts/quantizer/make_qz_fixtures.py) and the BF16 WikiText-2 PPL record the preset runs are compared with
(an A100 run, trimmed to the fields the exporter reads). Everything else here is SYNTHETIC: made-up PPL, SQNR
and lm-eval numbers in the shapes tricast writes, so the exporters' checks can be exercised without GPU runs.
They are test inputs only (scripts/quantizer/make_qz_synth.py builds the browser checks' synthetic bundle from
them) and never page data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr

DATA = Path(__file__).resolve().parent / "data" / "quantizer"
NATIVE = DATA / "native_ppl.json"


def preset_yaml(preset_id: str) -> str:
    return (DATA / "recipes" / f"{preset_id}.yaml").read_text(encoding="utf-8")


def fixture(preset_id: str) -> dict[str, Any]:
    return json.loads((DATA / "recipes" / f"{preset_id}.recipe.json").read_text(encoding="utf-8"))


def native_record() -> dict[str, Any]:
    return json.loads(NATIVE.read_text(encoding="utf-8"))


def write(path: Path, record: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def ppl_record(preset_id: str, ppl: float = 30.0) -> dict[str, Any]:
    recipe = fixture(preset_id)
    native = native_record()
    return {
        "status": "complete",
        "recipe": recipe,
        "recipe_hash": ex.recipe_sha256(recipe),
        "metrics": {"ppl": {**native["metrics"]["ppl"], "ppl": ppl}},
        "env": native["env"],
        "wall_time_s": 120.0,
    }


def report_record(preset_id: str) -> dict[str, Any]:
    recipe = fixture(preset_id)
    _, grid, _, _ = qr.PRESETS[preset_id]
    layers = [
        {"name": qr.layer_name(b, kind), "status": "measured", "output": {"sqnr_db": 10.0 + b + k / 10}}
        for b in range(qr.BLOCKS)
        for k, kind in enumerate(qr.KINDS)
        if grid[b][k] != "bf16"
    ]
    return {
        "status": "complete",
        "recipe": recipe,
        "recipe_hash": ex.recipe_sha256(recipe),
        "layers": layers,
        "model": {
            "logits_kl": 0.25,
            "top1_agreement": 0.75,
            "ppl_reference": 40.0,
            "ppl_emulated": 50.0,
            "n_tokens": 4088,
        },
        "env": native_record()["env"],
        "samples": 8,
        "seqlen": 512,
        "dataset_fingerprint": "synthetic",
    }


def lmeval_record(preset_id: str | None, shift: float = 0.0, n: int = 300) -> dict[str, Any]:
    """A tricast runner record with an lm-eval result (made-up scores; shapes as lm-eval writes them)."""
    recipe = fixture(preset_id) if preset_id else {"name": "native"}
    base = {"arc_easy": 0.62, "arc_challenge": 0.36, "piqa": 0.68, "winogrande": 0.57, "hellaswag": 0.47}
    results = {}
    for task, metric, _ in ex.BENCHMARKS:
        value = base[task] - shift
        results[task] = {"alias": task, f"{metric},none": value, f"{metric}_stderr,none": 0.028}
        if metric == "acc_norm":
            results[task].update({"acc,none": value - 0.01, "acc_stderr,none": 0.027})
    env = {**native_record()["env"], "versions": {**native_record()["env"]["versions"], "lm_eval": "0.4.13"}}
    return {
        "status": "complete",
        "recipe": recipe,
        "recipe_hash": ex.recipe_sha256(recipe) if preset_id else "native",
        "metrics": {
            "lm_eval": {
                "results": results,
                "n-shot": {task: 0 for task in base},
                "n-samples": {task: {"original": 1000, "effective": n} for task in base},
                "config": {"limit": 300.0, "batch_size": 16},
                "batch_size": {"requested": 16, "used": 16, "reason": None},
            }
        },
        "env": env,
        "wall_time_s": 900.0,
    }


def synthetic_runs(runs: Path) -> Path:
    """The recorded-run layout the Quantizer exporter reads, with SYNTHETIC records: PPL for qd_mixed and
    qd_nvint4, a report for qd_mixed, lm-eval for BF16 and qd_mixed (lm-eval under ``runs/lm``)."""
    write(runs / "ppl_demo" / "qd_mixed.json", ppl_record("qd_mixed", ppl=26.5))
    write(runs / "report" / "qd_mixed" / "report.json", report_record("qd_mixed"))
    write(runs / "ppl_fmt" / "qd_nvint4.json", ppl_record("qd_nvint4", ppl=31.25))
    write(runs / "lm" / "lmeval_A" / "native.json", lmeval_record(None))
    write(runs / "lm" / "lmeval_A" / "qd_mixed.json", lmeval_record("qd_mixed", shift=0.025))
    return runs


# ---------- SYNTHETIC tricast report.json files for app/export_layers.py ----------
# kind -> (weight format, granularity, group size, activation format or None, transform, weight algorithm)
LAYER_KINDS = {
    "int8": ("int8", "row", 0, "int8", "none", "rtn"),
    "w4a16": ("uint4", "group", 128, None, "none", "gptq"),
    "nvint4": ("int4", "group", 16, "int4", "none", "rtn"),
}


def _errors(rng: random.Random, exact: bool = False) -> dict[str, Any]:
    if exact:
        return {"mse": 0.0, "sqnr_db": "inf", "max_abs_error": 0.0, "relative_frobenius": 0.0, "cosine": 1.0}
    mse = rng.uniform(1e-6, 1e-3)
    return {
        "mse": mse,
        "sqnr_db": rng.uniform(8.0, 40.0),
        "max_abs_error": rng.uniform(0.01, 0.9),
        "relative_frobenius": rng.uniform(0.01, 0.4),
        "cosine": rng.uniform(0.9, 1.0),
    }


def layer_report(recipe_id: str, kind: str) -> dict[str, Any]:
    """A SYNTHETIC ``tricast report`` record for the 196 decoder Linears of Qwen3-0.6B (seeded by the recipe
    id). ``w4a16`` has exact (``"inf"`` SQNR) activations like a weight-only recipe; ``nvint4`` carries one
    ``"nan"`` and one ``"-inf"`` metric so every non-finite token takes the flag path."""
    weight_format, granularity, group, activation_format, transform, algo = LAYER_KINDS[kind]
    rng = random.Random(f"{recipe_id}/{kind}")
    layers, entries = [], []
    for b in range(qr.BLOCKS):
        for kind_name in qr.KINDS:
            name = qr.layer_name(b, kind_name)
            ulp = {
                "max": rng.randrange(0, 600),
                "mean": rng.uniform(0, 0.01),
                "p99": float(rng.randrange(0, 3)),
                "exact_fraction": rng.uniform(0.9, 1.0),
                "n": 4194304,
                "nonfinite_mismatch": 0,
                "format": "bf16",
            }
            layer = {
                "name": name,
                "status": "measured",
                "weight": _errors(rng),
                "activation": _errors(rng, exact=activation_format is None),
                "output": _errors(rng),
                "mma_ulp": ulp,
            }
            layers.append(layer)
            operand = {
                "format": weight_format,
                "granularity": granularity,
                "group_size": group,
                "rounding": "rne",
                "mma_input": "dequant",
            }
            entries.append(
                {
                    "name": name,
                    "weight": operand,
                    "activation": None
                    if activation_format is None
                    else {**operand, "format": activation_format},
                    "mma": {"algorithm": "cofda", "f_bits": 23, "g_bits": 32, "name": ""},
                    "transform": transform,
                    "weight_algo": algo,
                    "sparsity": "none",
                    "outliers": None,
                    "skipped": None,
                }
            )
    if kind == "nvint4":
        layers[5]["output"]["sqnr_db"] = "nan"
        layers[40]["weight"]["sqnr_db"] = "-inf"
    rng.shuffle(layers)  # the exporter orders rows itself
    env = {**native_record()["env"], "gpu_names": ["SYNTHETIC"]}
    return {
        "status": "complete",
        "layers": layers,
        "patch_report": {
            "decoder_layers": "model.layers",
            "layers": entries,
            "patched": [],
            "skipped": ["lm_head"],
            "kv": [],
            "unused_overrides": [],
        },
        "model": {
            "logits_kl": rng.uniform(0, 0.3),
            "top1_agreement": rng.uniform(0.7, 0.95),
            "ppl_reference": 43.7,
            "ppl_emulated": rng.uniform(44.0, 60.0),
            "n_tokens": 4088,
        },
        "recipe": {"name": recipe_id, "description": f"SYNTHETIC {kind} report"},
        "env": env,
        "samples": 8,
        "seqlen": 512,
        "dataset_fingerprint": "synthetic",
        "forward_mode": "full_window",
        "recipe_hash": "0" * 64,
    }
