"""Strict natural-language requests and validated evaluation plans."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ..formats import get_format
from ..recipe import Recipe, load_recipe

REQUEST_SCHEMA: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas" / "emulation_request.schema.json").read_text("utf-8")
)


def api_schema(schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project a nonrecursive schema for strict APIs; validate the full schema locally."""
    root = REQUEST_SCHEMA if schema is None else schema
    allowed = {
        "$ref", "$defs", "type", "properties", "required", "additionalProperties",
        "items", "enum", "const", "anyOf", "allOf", "format", "title", "description",
    }
    formats = {"date-time", "time", "date", "email", "hostname", "ipv4", "ipv6", "uri", "uuid"}

    def check(value: Any, ancestors: frozenset[int] = frozenset()) -> None:
        if isinstance(value, dict):
            if id(value) in ancestors:
                raise ValueError("recursive schemas are not supported by the API")
            ancestors = ancestors | {id(value)}
            if "$ref" in value:
                ref = value["$ref"]
                if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")):
                    raise ValueError("API schema references must be local JSON pointers")
                target: Any = root
                try:
                    for part in ref[2:].split("/") if ref != "#" else ():
                        target = target[part.replace("~1", "/").replace("~0", "~")]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"unresolved schema reference {ref!r}") from exc
                check(target, ancestors)
            for key, child in value.items():
                if key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
                    for item in child.values():
                        check(item, ancestors)
                elif key not in ("const", "enum", "default", "examples"):
                    check(child, ancestors)
        elif isinstance(value, list):
            for child in value:
                check(child, ancestors)

    def project(value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key not in allowed:
                continue
            if key in ("properties", "$defs"):
                result[key] = {name: project(item) for name, item in child.items()}
            elif key in ("anyOf", "allOf"):
                result[key] = [project(item) for item in child]
            elif key == "items":
                if isinstance(child, dict):
                    result[key] = project(child)
            elif key == "format":
                if child in formats:
                    result[key] = child
            else:
                result[key] = copy.deepcopy(child)
        kind = result.get("type", [])
        if kind == "object" or "object" in kind or "properties" in result or "additionalProperties" in result:
            result["additionalProperties"] = False
        return result

    check(root)
    return project(root)


class RequestValidationError(ValueError):
    """One or more field-qualified request errors."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def _path(parts: Any) -> str:
    result = ""
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else ("." if result else "") + part
    return result or "request"


def _integer_errors(value: Any, schema: dict, path: tuple = ()) -> list[str]:
    if "$ref" in schema:
        schema = REQUEST_SCHEMA["$defs"][schema["$ref"].split("/")[-1]]
    if isinstance(value, float) and "integer" in schema.get("type", []):
        return [f"{_path(path)}: expected an integer value, not a floating-point number"]
    if isinstance(value, dict):
        return [
            error
            for key, child in value.items()
            for error in _integer_errors(child, schema["properties"][key], (*path, key))
        ]
    if isinstance(value, list):
        return [
            error
            for index, child in enumerate(value)
            for error in _integer_errors(child, schema["items"], (*path, index))
        ]
    return []


def _merge(base: dict, changes: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _quant_choice(choice: dict) -> dict:
    result = {
        key: value
        for key, value in choice.items()
        if value is not None and key not in ("scale_format", "scale_method", "scale_rounding", "observer")
    }
    scale = {
        key.removeprefix("scale_"): choice[key]
        for key in ("scale_format", "scale_method", "scale_rounding")
        if choice[key] is not None
    }
    if scale:
        result["scale"] = scale
    if choice["observer"] is not None:
        result["observer"] = {"kind": choice["observer"]}
    return result


@dataclass(frozen=True)
class EmulationRequest:
    """Explicit choices stay unchanged; resolution records defaults in assumptions."""

    intent: str
    model: str | None
    recipes: list[dict]
    sweep: dict | None
    tasks: list[str]
    limits: dict[str, int | float | None]
    assumptions: list[str]
    questions: list[str]
    topic: str | None
    evaluation: dict[str, int | None] | None = None
    report_inputs: dict | None = None
    _omitted_options: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict) -> EmulationRequest:
        """Validate structured JSON without inferring missing arithmetic."""
        data = copy.deepcopy(data)
        omitted = tuple(key for key in ("evaluation", "report_inputs")
                        if isinstance(data, dict) and key not in data)
        for key in omitted:
            data[key] = None
        if isinstance(data, dict) and isinstance(data.get("recipes"), list):
            for entry in data["recipes"]:
                if not isinstance(entry, dict):
                    continue
                for key in ("kv", "calibration", "layers", "modules", "skip"):
                    entry.setdefault(key, None)
                for key in ("weight", "activation"):
                    if isinstance(entry.get(key), dict):
                        entry[key].setdefault("scale_rounding", None)
        errors = [
            f"{_path(error.absolute_path)}: {error.message}"
            for error in Draft202012Validator(REQUEST_SCHEMA).iter_errors(data)
        ]
        if errors:
            raise RequestValidationError(errors)
        errors.extend(_integer_errors(data, REQUEST_SCHEMA))
        if data["intent"] in ("explain", "inspect"):
            if any(value is not None for value in (data["evaluation"] or {}).values()):
                errors.append("evaluation: explain/inspect cannot execute evaluation options")
            if data["tasks"] and data["recipes"]:
                errors.append("tasks: choose evaluation or explanation/inspection of these recipes")
            elif data["tasks"]:
                message = "tasks: explain/inspect describes tasks only; no evaluation will be executed."
                if message not in data["assumptions"]:
                    data["assumptions"].append(message)
        try:
            json.dumps(data, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise RequestValidationError([f"request: {exc}"]) from exc
        model = data["model"]
        if model is not None and (not model or re.search(r"\s", model)):
            errors.append("model: expected a nonempty Hugging Face repository id without whitespace")
        for index, entry in enumerate(data["recipes"]):
            for operand in ("weight", "activation"):
                choice = entry[operand]
                if choice is None:
                    continue
                for key in ("format", "scale_format"):
                    if choice[key] is not None:
                        try:
                            resolved = get_format(choice[key])
                            if choice[key].strip().lower() in ("fp8", "fp4", "float", "half"):
                                message = (
                                    f"recipes[{index}].{operand}.{key}: {choice[key]} -> {resolved.name} "
                                    "(format alias convention)."
                                )
                                if message not in data["assumptions"]:
                                    data["assumptions"].append(message)
                        except (ValueError, TypeError) as exc:
                            errors.append(f"recipes[{index}].{operand}.{key}: {exc}")
        inputs = data["report_inputs"]
        if inputs is not None:
            texts, ids = inputs["texts"], inputs["input_ids"]
            if texts is not None and ids is not None:
                errors.append("report_inputs: choose texts or input_ids, not both")
            if texts is not None and any(not text.strip() for text in texts):
                errors.append("report_inputs.texts: texts must be nonempty")
            if ids is not None and len({len(row) for row in ids}) != 1:
                errors.append("report_inputs.input_ids: expected rectangular token rows")
            if data["intent"] != "report":
                errors.append("report_inputs: inputs require report intent")
        sweep = data["sweep"]
        if sweep is not None:
            for index, value in enumerate(sweep["values"]):
                valid = value in ("fused", "decoupled") if sweep["axis"] == "c_mode" else type(value) is int
                if not valid:
                    errors.append(f"sweep.values[{index}]: invalid value for {sweep['axis']}: {value!r}")
            if len(set(sweep["values"])) != len(sweep["values"]):
                errors.append("sweep.values: values must be unique")
        if errors:
            raise RequestValidationError(errors)
        return cls(**copy.deepcopy(data), _omitted_options=omitted)

    def to_dict(self) -> dict:
        """Return JSON-compatible data without exposing mutable request internals."""
        data = asdict(self)
        # Keep legacy request round-trips exact; new parser output includes both options.
        for key in data.pop("_omitted_options"):
            data.pop(key)
        return data

    def _assume(self, message: str) -> None:
        if message not in self.assumptions:
            self.assumptions.append(message)

    def _recipe_data(self, entry: dict, index: int) -> dict:
        path = f"recipes[{index}]"
        base_name = entry["base"]
        if base_name is not None:
            if not base_name or re.search(r"[/\\]", base_name) or Path(base_name).suffix:
                raise ValueError(f"{path}.base: expected an existing recipe name, not a path")
            try:
                data = load_recipe(base_name).to_dict()
            except (ValueError, OSError, TypeError) as exc:
                raise ValueError(f"{path}.base: {exc}") from exc
        else:
            data = {"name": entry["name"], "defaults": {}}
            if not any(entry[key] is not None for key in ("weight", "activation", "mma", "kv")):
                raise ValueError(f"{path}: choose a base recipe or explicit operand/MMA configuration")
        name = entry["name"]
        if re.search(r"[/\\]", name) or name in (".", "..") or name.casefold() == "env":
            raise ValueError(f"{path}.name: expected a safe, nonreserved output name")
        data["name"] = name
        defaults = data["defaults"]
        for key in ("weight", "activation", "mma"):
            choice = entry[key]
            if choice is None:
                continue
            changes = (
                _quant_choice(choice)
                if key != "mma"
                else {field: value for field, value in choice.items() if value is not None}
            )
            selector = "preset" if key == "mma" else "scheme"
            if key != "mma" and selector not in changes and "format" not in changes and not defaults.get(key):
                raise ValueError(f"{path}.{key}.format: specify format or scheme, or inherit a base recipe")
            previous = defaults.get(key) or {}
            defaults[key] = changes if selector in changes else _merge(previous, changes)
            if key == "mma" and selector not in changes and changes:
                defaults[key].update(name="", provenance="")
        for key in ("transform", "weight_algo"):
            if entry[key] is not None:
                previous = defaults.get(key, {})
                defaults[key] = (
                    _merge(previous, {"kind": entry[key]})
                    if previous.get("kind") == entry[key]
                    else {"kind": entry[key]}
                )
        for key in ("kv", "calibration"):
            if entry[key] is not None:
                changes = {field: value for field, value in entry[key].items() if value is not None}
                previous = data.get(key) or {}
                data[key] = changes if "preset" in changes else _merge(previous, changes)
        selectors = {key: entry[key] for key in ("layers", "modules") if entry[key] is not None}
        kv_only = data.get("kv") is not None and base_name is None and not any(
            entry[key] is not None for key in ("weight", "activation", "mma", "transform", "weight_algo")
        )
        if data.get("kv") is not None:
            if entry["skip"]:
                raise ValueError(f"{path}.skip: KV selection inversion is not supported")
            if kv_only and entry["modules"] is not None:
                raise ValueError(
                    f"{path}.modules: KV-only requests select attention layers, not linear modules"
                )
            if entry["layers"] is not None:
                data["kv"]["layers"] = entry["layers"]
                self._assume(f"{path}.layers: select KV attention layer indices as well as linear layers.")
            if entry["modules"] is not None:
                self._assume(f"{path}.modules: select linear modules only; KV selects attention layers.")
        if entry["skip"] is not None and not selectors:
            raise ValueError(f"{path}.skip: requires layers or modules")
        if kv_only:
            data["overrides"] = [{"match": "*", "skip": True}]
            self._assume(f"{path}: KV-only request; leave all linear modules unpatched.")
        elif selectors:
            if entry["skip"]:
                data["overrides"] = [{**selectors, "skip": True}, *data.get("overrides", [])]
                self._assume(f"{path}: skip selected layers/modules; apply the recipe to other modules.")
            else:
                if data.get("overrides"):
                    raise ValueError(
                        f"{path}: selection-only requests need a base without existing overrides"
                    )
                data["overrides"] = [selectors, {"match": "*", "skip": True}]
                self._assume(f"{path}: apply the recipe only to selected layers/modules; skip all others.")
        return data

    def _record_defaults(self, recipe: Recipe, entry: dict, index: int) -> None:
        resolved = recipe.to_dict()["defaults"]
        for key in ("weight", "activation", "mma", "transform", "weight_algo"):
            choice = entry[key]
            value = resolved[key]
            path = f"recipes[{index}].{key}"
            if choice is None and entry["base"] is not None:
                self._assume(f"{path}: inherited from base recipe {entry['base']}.")
                continue
            if value is None:
                self._assume(f"{path}: unspecified; use the unquantized input dtype.")
                continue
            if key in ("transform", "weight_algo"):
                missing = dict(value)
                if choice is not None:
                    missing.pop("kind", None)
            else:
                explicit = (
                    {}
                    if choice is None
                    else _quant_choice(choice)
                    if key != "mma"
                    else {field: val for field, val in choice.items() if val is not None}
                )
                missing = {
                    field: val
                    for field, val in value.items()
                    if field not in explicit and field not in ("name", "provenance")
                }
                for nested in ("scale", "observer"):
                    if isinstance(explicit.get(nested), dict) and isinstance(value.get(nested), dict):
                        missing[nested] = {
                            field: val
                            for field, val in value[nested].items()
                            if field not in explicit[nested]
                        }
            if missing:
                source = f"base recipe {entry['base']}" if entry["base"] is not None else "spec/scheme/preset"
                self._assume(
                    f"{path}: omitted options use {source} defaults: "
                    + json.dumps(missing, sort_keys=True, ensure_ascii=False)
                    + "."
                )
        for key in ("kv", "calibration"):
            value = recipe.to_dict().get(key)
            if key == "calibration" and (value is not None or recipe.needs_calibration):
                value = recipe.calibration_options
            choice = entry[key]
            if value is None:
                continue
            if choice is None:
                source = f"base recipe {entry['base']}" if entry["base"] is not None else "engine defaults"
                self._assume(
                    f"recipes[{index}].{key}: inherited from {source}: "
                    + json.dumps(value, sort_keys=True, ensure_ascii=False) + "."
                )
            else:
                missing = {field: val for field, val in value.items() if choice.get(field) is None}
                if missing:
                    self._assume(
                        f"recipes[{index}].{key}: omitted options use spec/preset defaults: "
                        + json.dumps(missing, sort_keys=True, ensure_ascii=False) + "."
                    )

    def to_recipes(self) -> list[Recipe]:
        """Resolve base choices and collect recipe-validation errors with request paths."""
        if not self.recipes and self.intent not in ("explain", "inspect"):
            raise RequestValidationError(["recipes: evaluation needs at least one recipe"])
        alternatives = len(self.recipes) * (len(self.sweep["values"]) if self.sweep is not None else 1)
        if self.intent == "compare" and alternatives < 2:
            raise RequestValidationError(["recipes: compare needs at least two recipe alternatives"])
        recipes: list[Recipe] = []
        errors: list[str] = []
        for index, entry in enumerate(self.recipes):
            try:
                recipe = load_recipe(self._recipe_data(entry, index))
                if recipe.defaults.mma.algorithm != "gdfs":
                    for key in ("g_bits", "group_size"):
                        if (entry["mma"] or {}).get(key) is not None:
                            errors.append(f"recipes[{index}].mma.{key}: requires algorithm gdfs")
                    if self.sweep is not None and self.sweep["axis"] in ("g_bits", "group_size"):
                        errors.append(f"sweep.axis: {self.sweep['axis']} requires algorithm gdfs "
                                      f"for recipes[{index}]")
                recipes.append(recipe)
            except (ValueError, TypeError, OSError) as exc:
                message = str(exc)
                if message.startswith("defaults."):
                    message = f"recipes[{index}]." + message.removeprefix("defaults.")
                elif not message.startswith(f"recipes[{index}]"):
                    message = f"recipes[{index}]: {message}"
                errors.append(message)
        names = [recipe.name for recipe in recipes]
        if len(set(names)) != len(names):
            errors.append("recipes: recipe names must be unique")
        if errors:
            raise RequestValidationError(errors)
        for index, (recipe, entry) in enumerate(zip(recipes, self.recipes, strict=True)):
            self._record_defaults(recipe, entry, index)
        return recipes

    def to_run_config(self) -> dict:
        """Create runner tasks and fully expanded sweep recipes; never execute them."""
        if self.questions:
            raise RequestValidationError(
                ["questions: resolve open questions before creating an execution plan"]
            )
        if self.intent in ("explain", "inspect"):
            return {
                "model": self.model,
                "recipes": [recipe.to_dict() for recipe in self.to_recipes()],
                "tasks": {},
            }
        if not self.model:
            raise RequestValidationError(["model: evaluation requires a Hugging Face repository id"])
        if not self.tasks and self.intent != "report":
            raise RequestValidationError(["tasks: choose at least one evaluation task"])
        if self.intent == "sweep" and self.sweep is None:
            raise RequestValidationError(["sweep: sweep intent needs an axis and values"])
        if self.intent == "report" and self.sweep is not None:
            raise RequestValidationError(["sweep: report does not support sweep expansion"])
        recipes = self.to_recipes()
        evaluation = self.evaluation or {}
        seed = evaluation.get("seed")
        if seed is None:
            seed = 42
            self._assume("evaluation.seed: unspecified; use runner default seed=42.")
        seqlen = evaluation.get("seqlen")
        if self.intent == "report":
            if self.tasks:
                raise RequestValidationError(["tasks: report cannot discard evaluation tasks; "
                                              "request evaluation and layer analysis separately"])
            if seed != 42:
                raise RequestValidationError(["evaluation.seed: layer reports only support seed=42"])
            task = {key: value for key, value in (self.report_inputs or {}).items() if value is not None}
            if seqlen is not None:
                task["seqlen"] = seqlen
            else:
                self._assume("tasks.report.seqlen: use the supplied token row length, or 128 for texts.")
            self._assume("tasks.report.samples: use all complete supplied token windows, or 8 for texts.")
            return {
                "model": self.model,
                "recipes": [recipe.to_dict() for recipe in recipes],
                "tasks": {"report": task},
                "seed": seed,
            }
        if seqlen is not None and "wikitext2_ppl" not in self.tasks:
            raise RequestValidationError(["evaluation.seqlen: only PPL and layer reports support seqlen"])
        if self.sweep is not None:
            expanded = []
            errors = []
            axis = self.sweep["axis"]
            for recipe in recipes:
                for index, value in enumerate(self.sweep["values"]):
                    data = recipe.to_dict()
                    data["name"] += f"__{axis}_{value}"
                    data["defaults"]["mma"].update({axis: value, "name": "", "provenance": ""})
                    try:
                        expanded.append(load_recipe(data))
                    except (ValueError, TypeError) as exc:
                        errors.append(f"sweep.values[{index}] ({axis}={value!r}): {exc}")
            if errors:
                raise RequestValidationError(errors)
            recipes = expanded
        tasks = {}
        if "wikitext2_ppl" in self.tasks:
            if seqlen is None:
                seqlen = 2048
                self._assume("evaluation.seqlen: unspecified; use runner default seqlen=2048.")
            tasks["ppl"] = {"dataset": "wikitext2", "seqlen": seqlen}
            self._assume("tasks.wikitext2_ppl: use runner defaults split=test, batch_size=1.")
            if self.limits["max_windows"] is not None:
                tasks["ppl"]["max_windows"] = self.limits["max_windows"]
            else:
                self._assume("limits.max_windows: null means all complete dataset windows (no window limit).")
        lm_tasks = list(dict.fromkeys(task for task in self.tasks if task != "wikitext2_ppl"))
        if lm_tasks:
            tasks["lm_eval"] = {"tasks": lm_tasks}
            if self.limits["limit"] is not None:
                tasks["lm_eval"]["limit"] = self.limits["limit"]
            else:
                self._assume("limits.limit: null means all task examples (no sample limit).")
            self._assume(
                "tasks.lm_eval: use runner defaults num_fewshot=null, batch_size=8, log_samples=false."
            )
        self._assume(
            "run: use dtype=auto, backend-selected device, and a unique output directory under runs/."
        )
        return {
            "model": self.model,
            "recipes": [recipe.to_dict() for recipe in recipes],
            "tasks": tasks,
            "seed": seed,
        }

    def estimate_cost(
        self,
        *,
        model_parameters: int | None = None,
        tokens: int | None = None,
        throughput_macs_per_second: float = 1e9,
    ) -> dict:
        """Estimate arithmetic volume without downloading or loading the model."""
        return estimate_cost(
            self.to_run_config(),
            model_parameters=model_parameters,
            tokens=tokens,
            throughput_macs_per_second=throughput_macs_per_second,
        )


def estimate_cost(
    run_config: dict,
    *,
    model_parameters: int | None = None,
    tokens: int | None = None,
    throughput_macs_per_second: float = 1e9,
) -> dict:
    """Parameter × token × recipe proxy, not a measured latency prediction."""
    for name, value in (("model_parameters", model_parameters), ("tokens", tokens)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")
    if (
        isinstance(throughput_macs_per_second, bool)
        or not isinstance(throughput_macs_per_second, (int, float))
        or not math.isfinite(throughput_macs_per_second)
        or throughput_macs_per_second <= 0
    ):
        raise ValueError("throughput_macs_per_second must be finite and positive")
    assumptions = [
        "Estimate only: parameters × tokens × recipes approximates emulated linear MACs; "
        "excludes attention, calibration, model loading and tokenization.",
        f"Assumed throughput={throughput_macs_per_second:g} MAC/s; not a measured device rate.",
    ]
    recipes = run_config.get("recipes", [])
    count = len(recipes)
    if tokens is None:
        tasks = run_config.get("tasks", {})
        ppl = tasks.get("ppl", {})
        if set(tasks) == {"ppl"} and ppl.get("max_windows") is not None:
            tokens = ppl["max_windows"] * ppl.get("seqlen", 2048)
            assumptions.append(
                "tokens_per_recipe is a max_windows × seqlen upper bound, not observed tokens."
            )
        else:
            assumptions.append("Token count is unknown; provide tokens for a numerical MAC estimate.")
    if model_parameters is None:
        assumptions.append(
            "Model parameter count is unknown; repository names are not parameter-count evidence."
        )
    macs = None if model_parameters is None or tokens is None else model_parameters * tokens * count
    return {
        "label": "추정",
        "is_estimate": True,
        "model_parameters": model_parameters,
        "tokens_per_recipe": tokens,
        "recipe_count": count,
        "emulated_macs": macs,
        "estimated_seconds": None if macs is None else macs / throughput_macs_per_second,
        "throughput_macs_per_second": throughput_macs_per_second,
        "assumptions": assumptions,
    }
