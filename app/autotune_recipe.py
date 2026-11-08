"""The recipe file behind a grid on the Auto-tune page (``POST /api/autotune/recipe``): what the engineer
takes away from the page.

- A knapsack grid: the YAML ``app.autotune solve`` writes for it on the bundle's table (the same default
  method, calibration, MMA and name rule, ``write_grid``), so the page hands over that exact file. When the
  grid is a recorded solution's, its name is that solution's and the file's SHA-256 is compared with the one
  the GPU measurement recorded (``export_autotune_demo.recipe_file_sha256``).
- A tune-session proposal (served with ``--agent``): the session's own ``proposals/<name>/recipe.yaml``,
  compared with the ``file_sha256`` its round's measurement (``rounds/<n>/kl.json``) recorded, with the
  bundle's measured values for that proposal.

Pure stdlib plus ``app.autotune`` / ``app.quantizer_recipe``: no torch, no tricast, no LLM call.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from app import autotune as at
from app import quantizer_recipe as qr

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class RecipeError(ValueError):
    """A grid or proposal the page cannot get a recipe for; ``status`` is the HTTP status."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RecipeError(404, "bundle_missing", f"번들이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _measured(row: dict[str, Any], sha: str, file_sha: str | None) -> dict[str, Any]:
    record = row.get("kl_record") or {}
    return {"id": row.get("id") or row.get("name"), "same_file": file_sha is not None and sha == file_sha,
            "recorded_sha256": file_sha, "kl": row.get("kl"), "ppl": row.get("ppl"),
            "gpu": record.get("gpu"), "git_sha": record.get("git_sha")}


def grid_recipe(bundle: dict[str, Any], grid: Any, budget_bits: str | None = None) -> dict[str, Any]:
    """The recipe ``solve`` writes for ``grid`` (rows of method ids, blocks × kinds of the bundle's table)."""
    table = bundle["table"]
    blocks, kinds, methods = table["blocks"], table["kinds"], table["methods"]
    if (not isinstance(grid, list) or len(grid) != blocks
            or not all(isinstance(row, list) and len(row) == len(kinds) for row in grid)):
        raise RecipeError(422, "invalid_grid", f"격자는 {blocks} 블록 × {len(kinds)} 종류여야 합니다")
    unknown = sorted({method for row in grid for method in row if method not in methods})
    if unknown:
        raise RecipeError(422, "invalid_grid", f"표에 없는 방법입니다: {unknown}")
    recorded = next((row for row in bundle.get("solutions") or []
                     if row["grid"] == grid and (row.get("kv") or "none") == "none"), None)
    if recorded is not None:
        name = recorded["id"]
    else:
        try:
            name = at.default_name(Fraction(budget_bits), "none") if budget_bits else "auto_grid"
        except (ValueError, ZeroDivisionError) as exc:
            raise RecipeError(422, "invalid_budget", f"크기 예산을 읽지 못했습니다: {budget_bits!r}") from exc
    meta = table["meta"]
    calibration = meta.get("calibration") if any(method in qr.CALIBRATED_ALL for method in methods) else None
    default = at.default_method([method for row in grid for method in row], methods)
    try:
        recipe = qr.build(name, default, grid, "none", calibration=calibration, mma=meta.get("mma"))
    except (ValueError, AssertionError) as exc:
        raise RecipeError(422, "invalid_grid", f"레시피를 만들 수 없는 격자입니다: {exc}") from exc
    text = f"---\n{qr.to_yaml(recipe)}\n"  # as app.autotune.write_grid writes recipe.yaml
    sha = _sha256(text)
    budget = f" --budget-bits {budget_bits}" if budget_bits else ""
    return {"kind": "knapsack", "name": name, "file": f"{name}.yaml", "yaml": text, "sha256": sha,
            "default": default,
            "measured": None if recorded is None
            else _measured(recorded, sha, recorded.get("recipe_file_sha256")),
            "command": f"python -m app.autotune solve --table {table['path']}{budget} --name {name} "
                       f"--out runs/{name}"}


def proposal_recipe(bundle: dict[str, Any], session: Path, name: str) -> dict[str, Any]:
    """The recipe file the tune session wrote (and measured) for proposal ``name``."""
    if not SAFE_NAME.fullmatch(name):
        raise RecipeError(400, "invalid_request", f"제안 이름이 올바르지 않습니다: {name!r}")
    path = session / "proposals" / name / "recipe.yaml"
    if not path.is_file():
        raise RecipeError(404, "not_found", f"세션에 그 제안이 없습니다: {name}")
    text = path.read_text(encoding="utf-8")
    row = next((proposal for round_ in (bundle.get("session") or {}).get("rounds") or []
                for proposal in round_["proposals"] if proposal["name"] == name), None)
    sha, measured = _sha256(text), None
    if row is not None and row.get("kl") is not None:
        record = ((bundle.get("session") or {}).get("records") or {}).get(row.get("kl_id"), {})
        recorded = next((result.get("file_sha256") for path in sorted((session / "rounds").glob("*/kl.json"))
                         for result in json.loads(path.read_text(encoding="utf-8")).get("results", [])
                         if result.get("name") == name), None)
        measured = {**_measured({**row, "kl_record": record}, sha, recorded), "id": name}
    return {"kind": "proposal", "name": name, "file": f"{name}.yaml", "yaml": text, "sha256": sha,
            "default": None, "measured": measured,
            "command": f"tricast recipe-check {session.name}/proposals/{name}/recipe.yaml"}
