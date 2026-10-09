"""Validated, reproducible quantization and accumulation recipes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from fnmatch import fnmatchcase
from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python 3.10 keeps this protocol in importlib.abc.
    from importlib.abc import Traversable

import yaml
from jsonschema import Draft202012Validator

from .formats import FloatFormat, IntFormat, Pow2Format, get_format
from .mma.spec import MMASpec
from .quant.spec import _GRANULARITY_SYNONYMS, KVSpec, QuantSpec, TransformSpec, WeightAlgoSpec, get_kv_spec
from .quant.structure import OutlierSpec, SparsitySpec

CALIBRATION_DEFAULTS = {
    "dataset": "wikitext2", "split": "train", "samples": 128, "seqlen": 2048,
    "seed": 0, "sequential": False,
}
_SELECTORS = {"match", "layers", "modules"}
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass
class LinearSpec:
    """The two operand formats, accumulator, and preprocessing for one linear."""

    weight: QuantSpec | None = None
    activation: QuantSpec | None = None
    mma: MMASpec = field(default_factory=MMASpec)
    transform: TransformSpec = field(default_factory=TransformSpec)
    weight_algo: WeightAlgoSpec = field(default_factory=WeightAlgoSpec)
    sparsity: SparsitySpec = field(default_factory=SparsitySpec)
    outliers: OutlierSpec | None = None

    def __post_init__(self) -> None:
        # Errors name the field first; load_recipe prefixes the recipe path.
        if self.outliers is not None and self.weight is None:
            raise ValueError("outliers: requires a weight QuantSpec")
        if self.weight_algo.kind == "gptq":
            if self.sparsity.kind != "none":
                raise ValueError("sparsity: not supported with gptq")
            if self.outliers is not None:
                raise ValueError("outliers: not supported with gptq")


def _json_value(value: Any) -> Any:
    if isinstance(value, (FloatFormat, IntFormat, Pow2Format)):
        return {"kind": value.kind, **{f.name: _json_value(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, SparsitySpec):
        # The kind and the fields it uses; unused fields are 0.
        return {f.name: getattr(value, f.name) for f in fields(value) if getattr(value, f.name)}
    if isinstance(value, LinearSpec):
        # Unset structure options are left out, so recipes without them keep their JSON and hash.
        data = {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
        if value.sparsity.kind == "none":
            del data["sparsity"]
        if value.outliers is None:
            del data["outliers"]
        return data
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _merge(base: dict, changes: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            # A new named scheme/preset supplies its own defaults, not the old one's; a sparsity
            # mapping is complete (its required kind decides which fields exist), and so is a
            # format mapping (inheriting e.g. subnormals or bias would define another format).
            if "scheme" in value or "preset" in value or key in ("sparsity", "format", "dequant_format"):
                result[key] = copy.deepcopy(value)
            else:
                result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
        if key == "mma" and isinstance(value, dict) and value and "preset" not in value:
            result[key]["name"] = ""
            result[key]["provenance"] = ""
    return result


def _parse_layers(selector: str, path: str) -> set[int]:
    indices = set()
    for part in selector.split(","):
        part = part.strip()
        if part == "-1" or re.fullmatch(r"\d+", part):
            indices.add(int(part))
        elif re.fullmatch(r"\d+-\d+", part):
            start, end = map(int, part.split("-"))
            if start > end:
                raise ValueError(f"{path}: layer range must be increasing")
            indices.update(range(start, end + 1))
        else:
            raise ValueError(f"{path}: use comma-separated layer indices or ranges (last layer: -1)")
    return indices


def _layer_index(module_name: str, decoder: tuple[str, int] | None) -> int | None:
    """Decoder-block index of a module: its position under ``decoder = (list name, length)``,
    or, without a resolved decoder list, the ``layers.<i>`` component of its name."""
    if decoder is None:
        match = _LAYER_INDEX.search(module_name)
        return int(match[1]) if match else None
    prefix = decoder[0] + "." if decoder[0] else ""
    if not module_name.startswith(prefix):
        return None
    head = module_name[len(prefix):].split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _matches(entry: dict, module_name: str, num_layers: int | None,
             decoder: tuple[str, int] | None = None) -> bool:
    if "match" in entry and not fnmatchcase(module_name, entry["match"]):
        return False
    if "modules" in entry and module_name.rsplit(".", 1)[-1] not in entry["modules"]:
        return False
    if "layers" in entry:
        index = _layer_index(module_name, decoder)
        if index is None:
            return False
        if decoder is not None:
            num_layers = decoder[1]
        indices = _parse_layers(entry["layers"], "layers")
        if -1 in indices:
            if num_layers is None:
                raise ValueError("layers: -1 needs num_layers to identify the last decoder layer")
            indices = (indices - {-1}) | {num_layers - 1}
        if index not in indices:
            return False
    return True


def _override_changes(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key not in _SELECTORS | {"skip"}}


@dataclass
class Recipe:
    """A default linear specification with ordered, conjunctive layer selectors."""

    name: str
    description: str = ""
    defaults: LinearSpec = field(default_factory=LinearSpec)
    include: list[str] = field(default_factory=lambda: ["*"])
    exclude: list[str] = field(default_factory=lambda: ["lm_head"])
    overrides: list[dict] = field(default_factory=list)
    calibration: dict | None = None
    backend: str = "auto"
    kv: KVSpec | None = None
    kv_layers: str | None = None

    def _selection(self, module_name: str, num_layers: int | None,
                   decoder: tuple[str, int] | None = None) -> tuple[int | None, str | None]:
        if not any(fnmatchcase(module_name, pattern) for pattern in self.include):
            return None, "not included"
        if any(fnmatchcase(module_name, pattern) for pattern in self.exclude):
            return None, "excluded"
        for index, entry in enumerate(self.overrides):
            if _matches(entry, module_name, num_layers, decoder):
                return index, f"overrides[{index}].skip" if entry.get("skip", False) else None
        return None, None

    def spec_for(self, module_name: str, *, num_layers: int | None = None,
                 decoder: tuple[str, int] | None = None) -> LinearSpec | None:
        """Resolve selectors. ``layers:`` indices refer to ``decoder = (name of the decoder-block
        ModuleList, its length)``; without it, to the ``layers.<i>`` component of the module name,
        and ``num_layers`` is then required for the last-layer index ``-1``."""
        index, reason = self._selection(module_name, num_layers, decoder)
        if reason is not None:
            return None
        if index is None:
            return self.defaults
        changes = _override_changes(self.overrides[index])
        return _linear_spec(_merge(_json_value(self.defaults), changes), f"overrides[{index}]")

    def skip_reason(self, module_name: str, *, num_layers: int | None = None,
                    decoder: tuple[str, int] | None = None) -> str | None:
        """Explain why a module is left in full precision, or return ``None``."""
        return self._selection(module_name, num_layers, decoder)[1]

    def override_index(self, module_name: str, *, decoder: tuple[str, int] | None = None) -> int | None:
        """Index of the first override that selects this (included, not excluded) module."""
        return self._selection(module_name, None if decoder is None else decoder[1], decoder)[0]

    @property
    def calibration_options(self) -> dict:
        """Effective settings shared by every calibration entry point."""
        return {**CALIBRATION_DEFAULTS, **(self.calibration or {})}

    @property
    def needs_calibration(self) -> bool:
        """Whether a configured path needs offline statistics (history is online)."""
        specs = [self.defaults]
        specs.extend(
            _linear_spec(_merge(_json_value(self.defaults), _override_changes(entry)), f"overrides[{i}]")
            for i, entry in enumerate(self.overrides) if not entry.get("skip", False)
        )
        return any(
            spec.transform.kind in ("smoothquant", "awq")
            or spec.weight_algo.kind == "gptq"
            or any(q is not None and q.observer is not None and q.observer.kind != "history"
                   for q in (spec.weight, spec.activation))
            for spec in specs
        )

    def to_dict(self) -> dict:
        """Return a JSON-compatible mapping that can be loaded again."""
        kv = _json_value(self.kv)
        if kv is not None and self.kv_layers is not None:
            kv["layers"] = self.kv_layers
        return {
            "name": self.name,
            "description": self.description,
            "defaults": _json_value(self.defaults),
            "include": list(self.include),
            "exclude": list(self.exclude),
            "overrides": copy.deepcopy(self.overrides),
            "calibration": copy.deepcopy(self.calibration),
            "backend": self.backend,
            "kv": kv,
        }

    @property
    def sha256(self) -> str:
        """Hash the canonical JSON, independent of mapping insertion order."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _path(parts) -> str:
    result = ""
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else ("." if result else "") + part
    return result or "recipe"


def _expand_shorthand(data: dict) -> dict:
    data = copy.deepcopy(data)
    containers = [data.get("defaults", {})]
    overrides = data.get("overrides", [])
    if isinstance(overrides, list):
        containers.extend(overrides)
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key, selector in (("weight", "scheme"), ("activation", "scheme"), ("mma", "preset"),
                              ("transform", "kind"), ("weight_algo", "kind")):
            if isinstance(container.get(key), str):
                container[key] = {selector: container[key]}
    return data


def _canonical_granularity(value):
    """Replace granularity synonyms (QuantSpec accepts them) with canonical names, recursively."""
    if isinstance(value, dict):
        return {key: (_GRANULARITY_SYNONYMS.get(item, item) if key == "granularity" and isinstance(item, str)
                      else _canonical_granularity(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_granularity(item) for item in value]
    return value


def _validate(data: dict) -> None:
    schema_path = Path(__file__).with_name("schemas") / "recipe.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(data))
    if errors:
        error = errors[0]
        while error.context:
            error = max(error.context, key=lambda e: len(e.absolute_path))
        raise ValueError(f"{_path(error.absolute_path)}: {error.message}")


def _quant_spec(data: dict | None, path: str) -> QuantSpec | None:
    if data is None:
        return None
    for key in ("format", "dequant_format"):
        if key in data:
            try:
                get_format(data[key])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}.{key}: {exc}") from exc
    scale = data.get("scale")
    if isinstance(scale, dict) and "format" in scale:
        try:
            get_format(scale["format"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}.scale.format: {exc}") from exc
    try:
        return QuantSpec.from_dict(data)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _linear_spec(data: dict, path: str) -> LinearSpec:
    weight = _quant_spec(data.get("weight"), f"{path}.weight")
    activation = _quant_spec(data.get("activation"), f"{path}.activation")
    result = {}
    for key, constructor in (("mma", MMASpec.from_dict), ("transform", lambda d: TransformSpec(**d)),
                             ("weight_algo", lambda d: WeightAlgoSpec(**d))):
        try:
            result[key] = constructor(data.get(key, {}))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}.{key}: {exc}") from exc
    # These specs name the failing field first ("n: ..."), so the path continues with it.
    for key, constructor in (("sparsity", SparsitySpec), ("outliers", OutlierSpec)):
        if data.get(key) is not None:
            try:
                result[key] = constructor(**data[key])
            except ValueError as exc:
                raise ValueError(f"{path}.{key}.{exc}") from exc
    try:
        return LinearSpec(weight=weight, activation=activation, **result)
    except ValueError as exc:
        raise ValueError(f"{path}.{exc}") from exc


def list_recipes() -> list[str]:
    """List bundled recipe names, independent of the current working directory."""
    return sorted({path.name.rsplit(".", 1)[0] for path in files("tricast").joinpath("recipes").iterdir()
                   if path.is_file() and path.name.endswith((".yaml", ".yml", ".json"))})


def _recipe_path(value: str | Path) -> Path | Traversable:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.suffix or len(candidate.parts) != 1:
        raise FileNotFoundError(f"recipe path does not exist: {candidate}")
    for suffix in (".yaml", ".yml", ".json"):
        path = files("tricast").joinpath("recipes", f"{value}{suffix}")
        if path.is_file():
            return path
    raise FileNotFoundError(f"unknown recipe {str(value)!r}; no bundled recipe was found")


def load_recipe(source: str | Path | dict | Recipe) -> Recipe:
    """Load a mapping, YAML/JSON path, or bundled recipe name."""
    if isinstance(source, Recipe):
        return source
    if isinstance(source, dict):
        data = copy.deepcopy(source)
    else:
        path = _recipe_path(source)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("recipe: expected an object")
    data = _canonical_granularity(_expand_shorthand(data))
    _validate(data)
    defaults = _linear_spec(data.get("defaults", {}), "defaults")
    overrides = data.get("overrides", [])
    for index, entry in enumerate(overrides):
        if "layers" in entry:
            _parse_layers(entry["layers"], f"overrides[{index}].layers")
        _linear_spec(_merge(_json_value(defaults), _override_changes(entry)), f"overrides[{index}]")
    kv_data = data.get("kv")
    kv_layers = None
    if isinstance(kv_data, dict):
        kv_data = dict(kv_data)
        kv_layers = kv_data.pop("layers", None)
        if kv_layers is not None:
            _parse_layers(kv_layers, "kv.layers")
    try:
        kv = get_kv_spec(kv_data) if kv_data is not None else None
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"kv: {exc}") from exc
    return Recipe(
        name=data["name"], description=data.get("description", ""), defaults=defaults,
        include=data.get("include", ["*"]), exclude=data.get("exclude", ["lm_head"]),
        overrides=overrides, calibration=data.get("calibration"), backend=data.get("backend", "auto"),
        kv=kv, kv_layers=kv_layers,
    )
