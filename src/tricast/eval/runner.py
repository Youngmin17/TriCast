"""Recipe sweeps, resumable evaluation records, and a compact result table."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
import re
import time
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..recipe import Recipe, load_recipe
from .envinfo import capture_env
from .ppl import perplexity

NATIVE = "native"  # run name of the unpatched model when a config sets native_baseline: true


def _mapping(value: dict | str | Path) -> dict:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    with Path(value).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError("evaluation configuration must be a mapping")
    return result


def expand_sweep(cfg: dict | str | Path) -> list[Recipe]:
    """Expand dotted paths under defaults into the Cartesian product of axis values."""
    cfg = _mapping(cfg)
    base = load_recipe(cfg["base_recipe"])
    axes = cfg.get("axes", {})
    if not isinstance(axes, dict) or any(not isinstance(v, list) or not v for v in axes.values()):
        raise ValueError("axes must map field paths to nonempty lists")
    result = []
    for values in itertools.product(*axes.values()):
        data = base.to_dict()
        suffix = []
        for path, value in zip(axes, values, strict=True):
            parts = path.removeprefix("defaults.").split(".")
            target = data["defaults"]
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    raise ValueError(f"unknown sweep axis {path!r}")
                target = target[part]
            if parts[-1] not in target:
                raise ValueError(f"unknown sweep axis {path!r}")
            target[parts[-1]] = value
            suffix.append(f"{path.replace('.', '_')}_{value}")
        if any(path.removeprefix("defaults.").startswith("mma.") for path in axes):
            data["defaults"]["mma"].update(name="", provenance="")
        data["name"] = base.name + ("__" + "__".join(suffix) if suffix else "")
        result.append(load_recipe(data))
    return result


def _load_model(model_id: str, dtype: str, device: str, revision: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16,
              "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if dtype != "auto" and dtype not in dtypes:
        raise ValueError(f"unknown dtype {dtype!r}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtypes.get(dtype, "auto"), revision=revision,
    )
    return model.to(device).eval(), tokenizer


def _json_default(value: Any) -> Any:
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8",
    )
    temporary.replace(path)



def _finite_number(value) -> bool:
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _valid_metrics(metrics, tasks: dict) -> bool:
    if not isinstance(metrics, dict) or metrics.keys() != tasks.keys():
        return False
    if "ppl" in metrics:
        ppl = metrics["ppl"]
        if not isinstance(ppl, dict):
            return False
        if not all(_finite_number(ppl.get(key)) for key in ("ppl", "nll")):
            return False
        if ppl["ppl"] <= 0 or ppl["nll"] < 0:
            return False
        if any(type(ppl.get(key)) is not int or ppl[key] <= 0 for key in ("n_tokens", "n_windows")):
            return False
        if not isinstance(ppl.get("dataset_fingerprint"), str) or not ppl["dataset_fingerprint"]:
            return False
    if "lm_eval" in metrics:
        lm_eval = metrics["lm_eval"]
        if not isinstance(lm_eval, dict) or not isinstance(lm_eval.get("results"), dict):
            return False
        if not lm_eval["results"] or not isinstance(lm_eval.get("n-samples"), dict):
            return False
        group_subtasks = lm_eval.get("group_subtasks", {})
        groups = lm_eval.get("groups", {})
        if not isinstance(group_subtasks, dict) or not isinstance(groups, dict):
            return False
        group_names = set(groups) | {name for name, children in group_subtasks.items() if children}
        leaf_count = 0
        for task, scores in lm_eval["results"].items():
            if not isinstance(task, str) or not isinstance(scores, dict) or not scores:
                return False
            numeric = [value for value in scores.values() if isinstance(value, Real)]
            if not numeric or not all(_finite_number(value) for value in numeric):
                return False
            if any(not isinstance(value, (Real, str)) for value in scores.values()):
                return False
            if task in group_names:
                continue
            leaf_count += 1
            samples = lm_eval["n-samples"].get(task)
            if not isinstance(samples, dict):
                return False
            counts = (samples.get(key) for key in ("original", "effective"))
            if any(type(value) is not int or value <= 0 for value in counts):
                return False
        if not leaf_count:
            return False
    return True


def _valid_env(env: Any) -> bool:
    required = {"utc", "hostname", "git_sha", "git_dirty", "src_sha256", "versions", "gpu_names",
                "gpu_driver", "cuda", "model_id", "model_sha", "environment", "seed", "dtype", "device"}
    return (
        isinstance(env, dict)
        and required <= env.keys()
        and isinstance(env["versions"], dict)
        and {"python", "torch", "triton", "transformers", "lm_eval", "datasets"} <= env["versions"].keys()
        and isinstance(env["environment"], dict)
        and isinstance(env["gpu_names"], list)
        and all(isinstance(env[key], str) and env[key] for key in ("utc", "hostname", "model_id"))
    )


def _resumable(env: dict) -> bool:
    return (
        _valid_env(env)
        and isinstance(env["git_sha"], str) and bool(env["git_sha"])
        and isinstance(env["git_dirty"], bool)
        and (not env["git_dirty"] or bool(env["src_sha256"]))
        and isinstance(env["model_sha"], str) and bool(env["model_sha"])
    )


def _valid_record(record, run_hash: str, recipe_hash: str, recipe: dict, tasks: dict) -> bool:
    report = record.get("patch_report") if isinstance(record, dict) else None
    return (
        isinstance(record, dict)
        and record.get("status") == "complete"
        and record.get("run_hash") == run_hash
        and record.get("recipe_hash") == recipe_hash
        and record.get("recipe") == recipe
        and _valid_env(record.get("env"))
        and _resumable(record["env"])
        and isinstance(report, dict)
        and (report == {"native": True} if recipe_hash == NATIVE
             else bool(report.get("patched") or report.get("kv")))
        and _finite_number(record.get("wall_time_s"))
        and record["wall_time_s"] >= 0
        and _valid_metrics(record.get("metrics"), tasks)
    )


def _recipes(cfg: dict, base_dir: Path) -> list[Recipe]:
    entries = cfg.get("recipes")
    if entries is None:
        entries = [{"base_recipe": cfg["base_recipe"], "axes": cfg.get("axes", {})}]
    if not isinstance(entries, list) or not entries:
        raise ValueError("recipes must be a nonempty list")
    result = []
    for entry in entries:
        if isinstance(entry, str) and (base_dir / entry).is_file():
            entry = _mapping(base_dir / entry)
        if isinstance(entry, dict) and "base_recipe" in entry:
            base_recipe = entry["base_recipe"]
            if isinstance(base_recipe, str) and (base_dir / base_recipe).is_file():
                entry = {**entry, "base_recipe": _mapping(base_dir / base_recipe)}
            result.extend(expand_sweep(entry))
        else:
            result.append(load_recipe(entry))
    names = [recipe.name for recipe in result]
    if len(set(names)) != len(names):
        raise ValueError("recipe names must be unique within a run")
    return result


def _summary(output_dir: Path, results: list[dict]) -> None:
    lines = ["| Recipe | PPL | lm-eval | Seconds |", "| --- | ---: | --- | ---: |"]
    for result in results:
        metrics = result["metrics"]
        ppl = metrics.get("ppl", {}).get("ppl")
        scores = []
        for task, values in metrics.get("lm_eval", {}).get("results", {}).items():
            for key, value in values.items():
                if isinstance(value, Real) and "stderr" not in key:
                    scores.append(f"{task}/{key}: {value:.6g}")
        name = result["recipe"]["name"].replace("|", "\\|")
        ppl_text = f"{ppl:.6g}" if ppl is not None else "—"
        score_text = "; ".join(scores).replace("|", "\\|") or "—"
        lines.append(f"| {name} | {ppl_text} | {score_text} | {result['wall_time_s']:.3f} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=_json_default).encode()).hexdigest()


def run_config(cfg: dict | str | Path, *, calibration: dict | None = None) -> list[dict]:
    """Run each recipe from an unchanged model; resume only with verified source/model identity."""
    from ..calibration import calibrate
    from ..nn.patch import patch_model, unpatch_model

    base_dir = Path.cwd() if isinstance(cfg, dict) else Path(cfg).resolve().parent
    cfg = _mapping(cfg)
    model_id = str(cfg["model"])
    dtype = cfg.get("dtype", "auto")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg.get("seed", 42))
    task_config = cfg.get("tasks", {})
    if not task_config or set(task_config) - {"ppl", "lm_eval"}:
        raise ValueError("tasks must contain ppl and/or lm_eval")
    overrides = {**(cfg.get("calibration") or {}), **(calibration or {})}
    recipes = _recipes(cfg, base_dir)
    for recipe in recipes:
        if not recipe.name or re.search(r"[/\\]", recipe.name) or recipe.name in {".", ".."}:
            raise ValueError(f"recipe name is not a safe output filename: {recipe.name!r}")
        if recipe.name.casefold() in {"env", NATIVE}:
            raise ValueError(f"recipe name {recipe.name!r} is reserved (env.json, native baseline)")
    output_dir = Path(cfg.get("output_dir", "runs/tricast"))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    model = tokenizer = None
    env = capture_env(model_id, {"seed": seed, "dtype": dtype, "device": device})
    # None is the unpatched model, evaluated first under the same identity and resume rules.
    runs: list[Recipe | None] = ([None] if cfg.get("native_baseline") else []) + recipes
    for recipe in runs:
        if recipe is not None and overrides:
            recipe = load_recipe({
                **recipe.to_dict(), "calibration": {**(recipe.calibration or {}), **overrides},
            })
        name = recipe.name if recipe else NATIVE
        recipe_hash = recipe.sha256 if recipe else NATIVE
        recipe_dict = recipe.to_dict() if recipe else {"name": NATIVE}
        calibration_config = recipe.calibration_options if recipe else None
        identity = {"model": model_id, "model_sha": env.get("model_sha"), "dtype": dtype,
                    "device": device, "seed": seed, "tasks": task_config, "recipe_hash": recipe_hash,
                    "git_sha": env.get("git_sha"), "git_dirty": env.get("git_dirty"),
                    "src_sha256": env.get("src_sha256"), "calibration": calibration_config,
                    "versions": env.get("versions"), "gpu_names": env.get("gpu_names"),
                    "gpu_driver": env.get("gpu_driver"), "cuda": env.get("cuda")}
        run_hash = _run_hash(identity)
        path = output_dir / f"{name}.json"
        if path.exists() and _resumable(env):
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if _valid_record(previous, run_hash, recipe_hash, recipe_dict, task_config):
                previous["result_path"] = str(path.resolve())
                results.append(previous)
                continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        started = time.perf_counter()
        if model is None:
            if env.get("model_kind") == "hf" and env.get("model_sha"):
                model, tokenizer = _load_model(model_id, dtype, device, revision=env["model_sha"])
            else:
                model, tokenizer = _load_model(model_id, dtype, device)
            actual_sha = getattr(model.config, "_commit_hash", None)
            if actual_sha and env.get("model_kind") != "local":
                env["model_sha"] = actual_sha
                identity["model_sha"] = actual_sha
                run_hash = _run_hash(identity)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        run_env = {**copy.deepcopy(env), "recipe_hash": recipe_hash,
                   "calibration_config": calibration_config, "dataset_fingerprints": {}}
        record = {"status": "failed", "run_hash": run_hash, "recipe_hash": recipe_hash,
                  "recipe": recipe_dict, "metrics": {}, "env": run_env, "calibration": None,
                  "calibration_config": calibration_config, "patch_report": {},
                  "result_path": str(path.resolve())}
        try:
            if recipe is None:
                record["patch_report"] = {"native": True}
            else:
                report = patch_model(model, recipe)
                record["patch_report"] = asdict(report)
                if not report.patched and not report.kv:
                    record["error"] = "recipe selected no modules: zero patches were applied"
            if "error" not in record:
                if recipe is not None and recipe.needs_calibration:
                    record["calibration"] = calibrate(model, recipe, tokenizer=tokenizer, device=device)
                    run_env["dataset_fingerprints"]["calibration"] = record["calibration"].get(
                        "dataset_fingerprint",
                    )
                model.eval()
                metrics = record["metrics"]
                if "ppl" in task_config:
                    metrics["ppl"] = perplexity(model, tokenizer, device=device, **task_config["ppl"])
                    run_env["dataset_fingerprints"]["ppl"] = metrics["ppl"]["dataset_fingerprint"]
                if "lm_eval" in task_config:
                    from .lmeval import evaluate

                    metrics["lm_eval"] = evaluate(model, tokenizer, **task_config["lm_eval"])
                    fingerprints = metrics["lm_eval"].get("dataset_fingerprints")
                    run_env["dataset_fingerprints"]["lm_eval"] = fingerprints
                if not _valid_metrics(metrics, task_config):
                    raise ValueError("evaluation returned incomplete or non-finite metrics")
                record["status"] = "complete"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["metrics"] = {}
            raise
        finally:
            try:
                unpatch_model(model)
            finally:
                record["wall_time_s"] = time.perf_counter() - started
                _write_json(output_dir / "env.json", run_env)
                _write_json(path, record)
        results.append(json.loads(path.read_text(encoding="utf-8")))
    _summary(output_dir, results)
    return results
