"""Plan evaluations and inspect completed records without implicit execution."""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..recipe import load_recipe

COMPARE_RUNS_SCHEMA = {
    "type": "object",
    "properties": {"paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64}},
    "required": ["paths"], "additionalProperties": False,
}


def _plan(config: dict) -> dict:
    import torch

    from ..eval.runner import expand_sweep

    if not isinstance(config.get("model"), str) or not config["model"].strip():
        raise ValueError("model: expected a nonempty model identifier")
    tasks = config.get("tasks")
    if not isinstance(tasks, dict) or not tasks or set(tasks) - {"ppl", "lm_eval"}:
        raise ValueError("tasks: expected ppl and/or lm_eval")
    if any(not isinstance(value, dict) for value in tasks.values()):
        raise ValueError("tasks: each task configuration must be an object")
    for name, task in tasks.items():
        allowed = ({"dataset", "split", "seqlen", "max_windows", "batch_size", "texts"} if name == "ppl"
                   else {"tasks", "num_fewshot", "limit", "batch_size", "log_samples"})
        if set(task) - allowed:
            raise ValueError(f"tasks.{name}: unsupported options {sorted(set(task) - allowed)}")
        for key, minimum in (("seqlen", 2), ("max_windows", 1), ("batch_size", 1), ("num_fewshot", 0)):
            value = task.get(key)
            if key in task and value is None and key in ("seqlen", "batch_size"):
                raise ValueError(f"tasks.{name}.{key}: null is not an integer")
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"tasks.{name}.{key}: expected an integer >= {minimum}")
        limit = task.get("limit")
        if limit is not None and (type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0):
            raise ValueError(f"tasks.{name}.limit: expected a finite positive number")
        if name == "lm_eval" and (not isinstance(task.get("tasks"), list) or not task["tasks"]
                                  or any(not isinstance(t, str) or not t.strip() for t in task["tasks"])):
            raise ValueError("tasks.lm_eval.tasks: expected nonempty task names")
    dtype = config.get("dtype", "auto")
    if dtype not in ("auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"):
        raise ValueError(f"dtype: unsupported dtype {dtype!r}")
    device = str(torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    seed = config.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed: expected an integer in [0, 2**32)")
    entries = config.get("recipes")
    if entries is None and "base_recipe" in config:
        entries = [{"base_recipe": config["base_recipe"], "axes": config.get("axes", {})}]
    if not isinstance(entries, list) or not entries:
        raise ValueError("recipes: expected a nonempty list")
    recipes = []
    for entry in entries:
        if isinstance(entry, dict) and "base_recipe" in entry:
            recipes.extend(expand_sweep(entry))
        else:
            recipes.append(load_recipe(entry))
    if len({recipe.name for recipe in recipes}) != len(recipes):
        raise ValueError("recipes: names must be unique")
    for recipe in recipes:
        if (not recipe.name or re.search(r"[/\\]", recipe.name) or recipe.name in (".", "..")
                or recipe.name.casefold() == "env"):
            raise ValueError(f"recipes.name: unsafe or reserved output name {recipe.name!r}")
    output_dir = config.get("output_dir")
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = f"runs/{timestamp}_{uuid4().hex}"
    return {"model": config["model"], "recipes": [recipe.to_dict() for recipe in recipes],
            "dtype": dtype, "device": device, "seed": seed,
            "tasks": copy.deepcopy(tasks), "output_dir": str(output_dir)}


def run_eval(run_config: dict, *, execute: bool = False) -> dict:
    """Return a validated plan; only execute=True can load models or run evaluation."""
    from ..agent.request import estimate_cost

    if not isinstance(run_config, dict) or type(execute) is not bool:
        return {"is_error": True, "error": "run_config must be an object and execute a boolean"}
    config = copy.deepcopy(run_config)
    try:
        json.dumps(config, allow_nan=False)
        plan = _plan(config)
        cost = estimate_cost(plan)
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
        return {"is_error": True, "error": str(exc)}
    result = {"is_error": False, "execute": execute, "plan": plan, "cost_estimate": cost}
    if not execute:
        return result
    from ..eval.runner import run_config as evaluate

    try:
        records = evaluate(copy.deepcopy(plan))
    except Exception as exc:
        return {**result, "is_error": True, "error": f"evaluation failed: {type(exc).__name__}"}
    output_dir = Path(plan["output_dir"]).resolve()
    return {**result, "paths": [str(output_dir / f"{record['recipe']['name']}.json") for record in records],
            "results": records}


def _finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return True


def _is_record(record: object) -> bool:
    if not isinstance(record, dict) or record.get("status") != "complete":
        return False
    if not all(isinstance(record.get(key), str) and record[key] for key in ("recipe_hash", "run_hash")):
        return False
    if not all(isinstance(record.get(key), dict) for key in ("recipe", "metrics", "env")):
        return False
    if not isinstance(record["recipe"].get("name"), str) or not record["metrics"] or not record["env"]:
        return False
    if set(record["metrics"]) - {"ppl", "lm_eval"}:
        return False
    if "ppl" in record["metrics"]:
        ppl = record["metrics"]["ppl"]
        if not isinstance(ppl, dict) or type(ppl.get("ppl")) not in (int, float) or ppl["ppl"] <= 0:
            return False
    if "lm_eval" in record["metrics"]:
        evaluation = record["metrics"]["lm_eval"]
        if not isinstance(evaluation, dict) or not isinstance(evaluation.get("results"), dict):
            return False
        if not evaluation["results"] or any(
            not isinstance(scores, dict) or not scores for scores in evaluation["results"].values()
        ):
            return False
    duration = record.get("wall_time_s")
    return type(duration) in (int, float) and duration >= 0 and _finite_json(record)


def compare_runs(paths: list[str]) -> dict:
    """Read evaluation records with their environments; do not assert benchmark parity."""
    if not isinstance(paths, list) or not 1 <= len(paths) <= 64 or any(not isinstance(p, str) for p in paths):
        return {"is_error": True, "error": "paths must contain between 1 and 64 JSON record paths"}
    records = []
    for value in paths:
        path = Path(value).expanduser()
        try:
            if path.suffix != ".json" or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                return {"is_error": True, "error": f"not a readable evaluation record: {path}"}
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"is_error": True, "error": f"could not read evaluation record: {path}"}
        if not _is_record(record):
            return {"is_error": True, "error": f"invalid or incomplete evaluation record: {path}"}
        records.append({"path": str(path), "recipe": record["recipe"]["name"],
                        "recipe_hash": record["recipe_hash"], "run_hash": record["run_hash"],
                        "metrics": record["metrics"], "env": record["env"],
                        "wall_time_s": record["wall_time_s"]})
    return {"is_error": False, "runs": records, "comparable": False,
            "warnings": ["Descriptive comparison only: model/dataset revisions, tasks, seeds, dtype, "
                         "device, and execution settings must be matched before claiming benchmark parity."]}
