"""Plan layer-error reports without loading models until execution is requested."""

from __future__ import annotations

import copy
import hashlib
import json

from jsonschema import Draft202012Validator

from ..recipe import load_recipe

_INPUT_PROPERTIES = {
    "texts": {"type": ["array", "null"], "minItems": 1, "maxItems": 64,
              "items": {"type": "string", "minLength": 1, "maxLength": 65536}},
    "input_ids": {"type": ["array", "null"], "minItems": 1, "maxItems": 64,
                  "items": {"type": "array", "minItems": 2, "maxItems": 65536,
                            "items": {"type": "integer", "minimum": 0}}},
}
LAYER_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": "string", "minLength": 1},
        "recipe": {"type": "string", "minLength": 1},
        **copy.deepcopy(_INPUT_PROPERTIES),
    },
    "required": ["model", "recipe", "texts", "input_ids"],
    "additionalProperties": False,
}


def _plan(config: dict) -> dict:
    import torch

    model = config.get("model")
    if not isinstance(model, str) or not model.strip() or any(char.isspace() for char in model):
        raise ValueError("model: expected a nonempty model identifier without whitespace")
    tasks = config.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != {"report"} or not isinstance(tasks["report"], dict):
        raise ValueError("tasks: expected a report task object")
    task = copy.deepcopy(tasks["report"])
    schema = {"type": "object", "properties": {
        **_INPUT_PROPERTIES, "samples": {"type": "integer", "minimum": 1},
        "seqlen": {"type": "integer", "minimum": 2},
    }, "additionalProperties": False}
    errors = list(Draft202012Validator(schema).iter_errors(task))
    if errors:
        path = ".".join(str(part) for part in errors[0].absolute_path)
        raise ValueError(f"tasks.report{'.' + path if path else ''}: {errors[0].message}")
    texts, input_ids = task.get("texts"), task.get("input_ids")
    if texts is not None and input_ids is not None:
        raise ValueError("tasks.report: choose texts or input_ids, not both")
    if texts is not None and any(not text.strip() for text in texts):
        raise ValueError("tasks.report.texts: texts must be nonempty")
    if input_ids is not None and (
        any(type(token) is not int for row in input_ids for token in row)
        or len({len(row) for row in input_ids}) != 1
    ):
        raise ValueError("tasks.report.input_ids: expected rectangular rows of integer token ids")
    for name in ("samples", "seqlen"):
        if name in task and type(task[name]) is not int:
            raise ValueError(f"tasks.report.{name}: expected an integer")
    task.setdefault("seqlen", len(input_ids[0]) if input_ids is not None else 128)
    if input_ids is not None and task["seqlen"] > len(input_ids[0]):
        raise ValueError("tasks.report.seqlen: input_ids must contain at least one complete window")
    task.setdefault("samples", len(input_ids) * (len(input_ids[0]) // task["seqlen"])
                    if input_ids is not None else 8)
    entries = config.get("recipes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("recipes: expected a nonempty list")
    recipes = [load_recipe(entry) for entry in entries]
    if len({recipe.name for recipe in recipes}) != len(recipes):
        raise ValueError("recipes: names must be unique")
    dtype = config.get("dtype", "auto")
    if dtype not in ("auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"):
        raise ValueError(f"dtype: unsupported dtype {dtype!r}")
    device = str(torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    seed = config.get("seed", 42)
    if type(seed) is not int or seed != 42:
        raise ValueError("seed: layer_report currently fixes analysis RNG to 42; other seeds are unsupported")
    return {"model": model, "recipes": [recipe.to_dict() for recipe in recipes],
            "tasks": {"report": copy.deepcopy(task)}, "dtype": dtype, "device": device, "seed": seed}


def layer_report(run_config: dict, *, execute: bool = False) -> dict:
    """Validate a report plan; only explicit execution loads models and analyzes supplied inputs."""
    if not isinstance(run_config, dict) or type(execute) is not bool:
        return {"is_error": True, "error": "run_config must be an object and execute a boolean"}
    try:
        json.dumps(run_config, allow_nan=False)
        plan = _plan(copy.deepcopy(run_config))
    except (KeyError, ValueError, TypeError, OSError, RuntimeError) as exc:
        return {"is_error": True, "error": str(exc)}
    task = plan["tasks"]["report"]
    needs_inputs = task.get("texts") is None and task.get("input_ids") is None
    result = {"is_error": False, "execute": execute, "plan": plan, "needs_inputs": needs_inputs}
    if not execute:
        return result
    if needs_inputs:
        return {**result, "is_error": True,
                "error": "tasks.report: supply texts or input_ids before executing a layer report"}
    try:
        import torch

        from ..analysis import layer_report as analyze
        from ..eval.envinfo import capture_env
        from ..eval.runner import _load_model

        torch.manual_seed(plan["seed"])
        model, tokenizer = _load_model(plan["model"], plan["dtype"], plan["device"])
        inputs = {"texts": task["texts"]} if task.get("texts") is not None else {
            "input_ids": torch.tensor(task["input_ids"], dtype=torch.long, device=plan["device"]),
        }
        windows = {"samples": task["samples"], "seqlen": task["seqlen"]}
        fingerprint = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        records = []
        for data in plan["recipes"]:
            recipe = load_recipe(data)
            torch.manual_seed(plan["seed"])
            analysis = analyze(model, recipe, tokenizer=tokenizer, **inputs, **windows)
            if not isinstance(analysis, dict) or analysis.get("is_error"):
                raise ValueError("layer analysis did not return a successful report")
            metrics = copy.deepcopy(analysis)
            env = metrics.pop("env", None)
            if not metrics or set(metrics) <= {"is_error"}:
                raise ValueError("layer analysis returned no metrics")
            if not isinstance(env, dict) or not env:
                env = capture_env(extra={
                    "model_id": plan["model"],
                    "model_sha": getattr(getattr(model, "config", None), "_commit_hash", None),
                    "seed": plan["seed"], "dtype": plan["dtype"], "device": plan["device"],
                    "input_fingerprint": fingerprint, "recipe_hash": recipe.sha256,
                })
            records.append({"recipe": recipe.to_dict(), "recipe_hash": recipe.sha256,
                            "metrics": {"report": metrics}, "env": env})
        json.dumps(records, allow_nan=False)
    except Exception as exc:
        return {**result, "is_error": True, "error": f"layer report failed: {type(exc).__name__}"}
    return {**result, "results": records}
