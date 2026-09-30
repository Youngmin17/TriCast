"""Discover registered specifications without constructing a model."""

from __future__ import annotations

from dataclasses import asdict

from ..formats import REGISTRY
from ..mma.spec import PRESETS
from ..quant.spec import SCHEMES
from ..rag import search
from ..recipe import list_recipes, load_recipe

DESCRIBE_SCHEMA = {
    "type": "object", "properties": {"name": {"type": "string"}},
    "required": ["name"], "additionalProperties": False,
}
LIST_OPTIONS_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["format", "scheme", "preset", "recipe"]}},
    "required": ["kind"], "additionalProperties": False,
}


def list_options(kind: str) -> dict:
    """List bundled names in deterministic order, accepting singular or plural kinds."""
    kind = kind.removesuffix("s") if isinstance(kind, str) else ""
    options = {"format": REGISTRY, "scheme": SCHEMES, "preset": PRESETS}
    if kind == "recipe":
        names = list_recipes()
    elif kind in options:
        names = sorted(options[kind])
    else:
        return {"is_error": True, "error": "kind must be format, scheme, preset, or recipe"}
    return {"is_error": False, "kind": kind, "options": names}


def describe(name: str) -> dict:
    """Describe a bundled name and attach source citations, including preset provenance."""
    if not isinstance(name, str):
        return {"is_error": True, "error": "name must be a string"}
    name = name.strip().lower()
    if name in REGISTRY:
        fmt = REGISTRY[name]
        bits = fmt.ebits if fmt.kind == "pow2" else fmt.bits
        kind, details = "format", {**asdict(fmt), "kind": fmt.kind, "bits": bits,
                                   "max_normal": fmt.max_normal}
    elif name in SCHEMES:
        kind, details = "scheme", asdict(SCHEMES[name])
    elif name in PRESETS:
        kind, details = "preset", asdict(PRESETS[name])
    elif name in list_recipes():
        try:
            kind, details = "recipe", load_recipe(name).to_dict()
        except (ValueError, OSError, TypeError) as exc:
            return {"is_error": True, "error": str(exc)}
    else:
        return {"is_error": True, "error": f"unknown catalog name {name!r}"}
    sources = search(name, k=3)
    return {"is_error": False, "kind": kind, "name": name, "details": details, "sources": sources}
