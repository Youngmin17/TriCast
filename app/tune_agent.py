"""Tools for an agent-driven auto-tune loop over the evidence ledger (docs/SPEC.md AC24,
docs/prompts/tune_agent.md). The agent (Claude Code in this repository) reads the evidence, proposes Quantizer
grids with a rationale that cites ledger records, has the proposals measured on a GPU, ingests the results and
repeats; the tools keep every proposal checkable and every number traceable.

    python -m app.tune_agent search "block13 oproj nvfp4"           # ledger documents (BM25)
    python -m app.tune_agent layer "model.layers.0.*"                # every record of matching layers
    python -m app.tune_agent propose --session S --name p1 --table T --budget-bits 9/2 --grid grid.json \
        --rationale "..." --cite E1a2b3c4d5e6,E...                   # validated proposal (+ recipe.yaml)
    python -m app.tune_agent neighbors --session S --from p1 --table T --budget-bits 9/2 --top 8
    python -m app.tune_agent round --session S --names p1,p2 --round 1   # GPU measure script for them
    python -m app.tune_agent state --session S                       # proposals with their measurements

A proposal is refused when its grid does not build (the Quantizer's own ``build``), when it exceeds the
budget, when a method has no table entry, when a cited id or an id named in the rationale is not in the
ledger, or when its rationale states a measured number (a decimal) that neither a cited record nor the
proposal's own computed values support at the precision it is written with. Parameters are written
``name=value`` (``damp=0.005``) and must be options of the proposal or of a cited session recipe; they never
support a measured number. Integers (block numbers, counts) are not checked. Predictions come from
``app.autotune.predict`` — the table's single-layer KL sum, raw and with ``app.autotune.FLOOR_RULE`` when
the table has floors (neighbours are ranked on the latter), plus the table's KV entries for the grid's KV
preset; they rank candidates, they are never reported as measurements. Selection measures whole-recipe KL on
validation windows (WikiText-2 train, seed 43 by default) — never on the test set the final PPL uses.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import re
import shlex
import sys
from collections import Counter
from collections.abc import Sequence
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from app import autotune as at
from app import ledger as lg
from app import quantizer_recipe as qr

PROPOSAL_SCHEMA = "tricast.tune.proposal/1"
ROUND_SCHEMA = "tricast.tune.round/1"
SESSION_SCHEMA = "tricast.tune.session/1"
# Whole-recipe KL windows of a session: WikiText-2 validation (calibration draws from train, so the two never
# share tokens), a seed other than the table's 42 (docs/spikes/2026-11-03_tune_agent.md §3).
MEASURE_DEFAULTS = {"windows": 8, "seqlen": 512, "seed": 43, "split": "validation"}
# A decimal written in a rationale, with its sign: 0.0026, -0.25, −0.63, .25, 26.16, 1e-3 (not inside an
# identifier such as E1a2b, and a hyphen right after a word or number is a dash, not a sign: 0.1-0.2).
_DECIMAL = re.compile(r"(?<![\w.])(?:(?<![\w)\].])(?P<sign>[-+\u2212]))?"
                      r"(?P<number>(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?!\w|\.\d)")


class TuneError(ValueError):
    """A refused input; the CLI prints it and exits with status 2."""


# ---------- numbers ----------
def unsupported_numbers(text: str, values: Sequence[float]) -> list[str]:
    """Decimals in ``text`` (signed) that no value rounds to at the precision they are written with."""
    missing = []
    for match in _DECIMAL.finditer(text):
        sign = "-" if match.group("sign") in ("-", "\u2212") else ""
        written = Decimal(sign + match.group("number"))
        step = Decimal(1).scaleb(written.as_tuple().exponent)
        supported = False
        for value in values:
            try:
                supported = Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_EVEN) == written
            except InvalidOperation:  # quantize beyond the context precision
                supported = False
            if supported:
                break
        if not supported:
            missing.append(match.group(0))
    return list(dict.fromkeys(missing))


_PARAMETER = re.compile(r"\b([a-z][a-z_]*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _parameters(spec: dict[str, Any]) -> dict[str, set[float]]:
    """{short name: values} of a grid spec's options (``weight_algo.damp`` -> ``damp``) and calibration."""
    out: dict[str, set[float]] = {}
    for options in (spec.get("options") or {}).values():
        for option, value in (options or {}).items():
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                out.setdefault(option.rsplit(".", 1)[-1], set()).add(float(value))
    for key, value in (spec.get("calibration") or {}).items():
        if not isinstance(value, bool) and isinstance(value, (int, float)):
            out.setdefault(key, set()).add(float(value))
    return out


def _strip_parameters(text: str, specs: Sequence[dict[str, Any]]) -> str:
    """``text`` without the ``name=value`` parameter statements that one of ``specs`` holds; any other
    ``name=value`` stays and is checked as a measured number. Parameters are never evidence for measurements:
    "damp=0.005" is accepted, "KL 0.005" is not, whatever the options hold."""
    known: dict[str, set[float]] = {}
    for spec in specs:
        for key, values in _parameters(spec).items():
            known.setdefault(key, set()).update(values)

    def keep(match: re.Match[str]) -> str:
        try:
            value = float(match.group(2))
        except ValueError:
            return match.group(0)
        return " " if value in known.get(match.group(1), set()) else match.group(0)

    return _PARAMETER.sub(keep, text)


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    return []


# ---------- grids ----------
def load_grid(path: Path) -> dict[str, Any]:
    """A Quantizer ``--from-json`` spec ({name, default, grid, kv, options, calibration, kv_options}) or a
    bare grid (list of rows); returns the spec with defaults filled in."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(spec, list):
        spec = {"grid": spec}
    if not isinstance(spec, dict) or "grid" not in spec:
        raise TuneError(f"{path}: expected a grid or {{name, default, grid, ...}}")
    grid = spec["grid"]
    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
        raise TuneError(f"{path}: grid must be a list of rows")
    used = [method for row in grid for method in row if method != "bf16"]
    default = spec.get("default") or (at.default_method(used, list(qr.METHODS)) if used else "nvfp4")
    return {"default": default, "grid": grid, "kv": spec.get("kv", "none"), "options": spec.get("options"),
            "calibration": spec.get("calibration"), "kv_options": spec.get("kv_options")}


def _recipe(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        return qr.build(name, spec["default"], spec["grid"], spec["kv"], options=spec["options"],
                        calibration=spec["calibration"], kv_options=spec["kv_options"])
    except (ValueError, AssertionError) as exc:
        raise TuneError(f"the grid does not build: {exc}") from exc


def _recipe_sha256(recipe: dict[str, Any]) -> str:
    from tricast.recipe import load_recipe  # the hash tricast run / measure record as recipe_hash

    return load_recipe(recipe).sha256


def load_full_table(table_path: Path) -> tuple[dict[str, Any], bytes, list[at.Layer], list[str]]:
    """``app.autotune.load_table``, refusing a table that does not cover every decoder block of its model:
    sizes and budgets are model-wide, so a layer without a row would not be counted."""
    table, raw, layers, methods = at.load_table(table_path)
    meta = table["meta"]
    blocks = len({layer.block for layer in layers})
    if blocks * len(qr.KINDS) != len(layers) or blocks < meta.get("model_blocks", blocks):
        raise TuneError(f"{table_path}: covers {blocks} of the model's {meta.get('model_blocks')} blocks; "
                        "the agent needs a table of every Linear layer")
    return table, raw, layers, methods


def method_bits(method: str, in_features: int, options: dict[str, Any] | None) -> Fraction:
    """Stored bits per weight of ``method`` with its options: ``app.autotune.weight_bits``, except that a
    ``weight.group_size`` option replaces the group the scale bits are spread over (gptq_int4)."""
    element, scale, per = at.ex.WEIGHT_BITS[method]
    group = ((options or {}).get(method) or {}).get("weight.group_size")
    if group is not None and per is not None:
        return element + Fraction(scale, group)
    return at.weight_bits(method, in_features)


def _unpredicted(spec: dict[str, Any], table: dict[str, Any]) -> str | None:
    """Why the table cannot predict ``spec`` (its entries are each method's defaults), or None."""
    if any(spec.get("options") or {}) or spec.get("kv_options"):
        return "method or KV options: the table has entries for the default options only"
    used = {method for row in spec["grid"] for method in row}
    if used & set(qr.CALIBRATED_ALL) or spec.get("calibration"):
        windows = table["meta"].get("calibration_windows")
        settings = qr.check_calibration(spec.get("calibration"))
        keys = ("dataset", "samples", "seqlen", "seed")
        differs = any(settings[key] != windows.get(key) for key in keys) if windows else True
        if differs or settings["sequential"]:
            return "calibration settings differ from the table's calibrated entries"
    return None


def evaluate(spec: dict[str, Any], table_path: Path, budget: Fraction) -> dict[str, Any]:
    """Average bits (options included) and, when the table can predict it, the table's prediction for
    ``spec``; refuses an over-budget grid, a partial table, or a method the table has no entry for."""
    table, raw, layers, methods = load_full_table(table_path)
    chosen = [spec["grid"][layer.block][qr.KINDS.index(layer.kind)] for layer in layers]
    total = sum((layer.params * method_bits(method, layer.in_features, spec.get("options"))
                 for layer, method in zip(layers, chosen, strict=True)), Fraction(0))
    bits = total / sum(layer.params for layer in layers)
    if bits > budget:
        raise TuneError(f"average bits {float(bits):.6g} exceed the budget {float(budget):.6g}")
    try:
        raw_sum = at.predict(table, layers, methods, spec["grid"])[0]
        corrected = at.predict(table, layers, methods, spec["grid"], subtract_floor=True)[0]
    except at.AutotuneError as exc:
        raise TuneError(str(exc)) from exc
    reason = _unpredicted(spec, table)
    predicted = {"predicted_kl_sum": None if reason else raw_sum,
                 "predicted_kl_sum_minus_floor": None if reason or at.floors(table) is None else corrected,
                 "predicted_kv_kl_sum": None if reason else at.predict_kv(table, spec["kv"]),
                 "unpredicted": reason, "floor_rule": at.FLOOR_RULE}
    return {"avg_bits": float(bits), "avg_bits_fraction": str(bits), "budget_bits": float(budget),
            "budget_bits_fraction": str(budget), "method_counts": dict(sorted(Counter(chosen).items())),
            "predicted": predicted,
            "table": {"path": str(table_path), "sha256": hashlib.sha256(raw).hexdigest()}}


# ---------- session ----------
def _proposal_dir(session: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise TuneError(f"proposal name {name!r}: use letters, digits, '_', '.', '-'")
    return session / "proposals" / name


def read_proposal(session: Path, name: str) -> dict[str, Any]:
    path = _proposal_dir(session, name) / "proposal.json"
    if not path.is_file():
        raise TuneError(f"no proposal {name!r} in {session}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_session(session: Path) -> dict[str, Any]:
    path = session / "session.json"
    if not path.is_file():
        raise TuneError(f"{session} has no session.json (it is written with the first proposal)")
    return json.loads(path.read_text(encoding="utf-8"))


def _session(session: Path, table_path: Path, write: bool) -> dict[str, Any]:
    """The session's fixed protocol — model, table and measurement windows — written (``write``) by the first
    proposal that passes every check; later proposals must use the same table."""
    table, raw, _, _ = load_full_table(table_path)
    digest = hashlib.sha256(raw).hexdigest()
    path = session / "session.json"
    if path.is_file():
        fixed = read_session(session)
        if fixed["table"]["sha256"] != digest:
            raise TuneError(f"{table_path} is not this session's table ({fixed['table']['path']})")
        return fixed
    meta = table["meta"]
    fixed = {"schema": SESSION_SCHEMA, "model_id": meta["model_id"], "model_sha": meta.get("model_sha"),
             "table": {"path": str(table_path), "sha256": digest}, "measure": dict(MEASURE_DEFAULTS)}
    if write:
        session.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fixed, indent=1) + "\n", encoding="utf-8")
    return fixed


def propose(session: Path, name: str, spec: dict[str, Any], table_path: Path, budget: Fraction,
            rationale: str, cites: Sequence[str], ledger: Path, *, author: str = "claude-code",
            parent: str | None = None, checks: bool = True) -> dict[str, Any]:
    """Validate and record one proposal: session/proposals/<name>/{grid.json, recipe.yaml, proposal.json}.
    ``checks=False`` (the no-check ablation) keeps the build and the budget but drops the evidence checks:
    rationale, cited ids, ids named in the rationale and the supported-numbers guard."""
    target = _proposal_dir(session, name)
    if target.exists():
        raise TuneError(f"proposal {name!r} exists; proposals are never overwritten")
    index = {record["id"]: record for record in lg.read_ledger(ledger)}
    if checks:
        if not rationale.strip():
            raise TuneError("a proposal needs a rationale")
        if not cites:
            raise TuneError("a proposal cites at least one ledger record")
        unknown = [item for item in cites if item not in index]
        if unknown:
            raise TuneError(f"cited ids not in the ledger {ledger}: {unknown}")
        mentioned = sorted(set(re.findall(r"\bE[0-9a-f]{6,}\b", rationale)) - set(index))
        if mentioned:
            raise TuneError(f"the rationale names ids that are not ledger records: {mentioned}")
    if parent is not None:
        read_proposal(session, parent)
    _session(session, table_path, write=False)
    recipe = _recipe(name, spec)
    computed = evaluate(spec, table_path, budget)
    cites = [item for item in cites if item in index]  # unchecked proposals keep only real records
    evidence = [number for item in cites for number in _numbers(index[item]["value"])]
    evidence += [number for item in cites for number in _numbers(index[item].get("stderr"))]
    # parameters ("damp=0.005") may come from the proposal's own options or those of cited session recipes
    specs = {json.loads(path.read_text(encoding="utf-8"))["recipe_sha256"]: json.loads(path.read_text(
        encoding="utf-8"))["spec"] for path in (session / "proposals").glob("*/proposal.json")}
    cited_specs = [specs[index[item]["recipe_hash"]] for item in cites
                   if index[item].get("recipe_hash") in specs]
    measured_text = _strip_parameters(rationale, [spec, *cited_specs])
    unsupported = unsupported_numbers(measured_text, evidence + _numbers(computed)) if checks else []
    if unsupported:
        raise TuneError(f"rationale numbers not supported by the cited records or the proposal: "
                        f"{unsupported}")
    _session(session, table_path, write=True)  # only a proposal that passed every check fixes the session
    proposal = {"schema": PROPOSAL_SCHEMA, "name": name, "author": author, "parent": parent,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "rationale": rationale, "cites": list(cites), "checks": checks, **computed,
                "recipe_sha256": _recipe_sha256(recipe), "spec": spec}
    target.mkdir(parents=True)
    (target / "grid.json").write_text(json.dumps({"name": name, **spec}, indent=1) + "\n", encoding="utf-8")
    (target / "recipe.yaml").write_text(qr.to_yaml(recipe) + "\n", encoding="utf-8")
    (target / "proposal.json").write_text(json.dumps(proposal, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    return proposal


def neighbors(spec: dict[str, Any], table_path: Path, budget: Fraction, top: int) -> list[dict[str, Any]]:
    """Grids one move away from ``spec`` that fit the budget, by predicted change (smallest first; ties by
    layer then method order): a layer switched to another tabled method, paid for when needed by switching
    the other layers with the least predicted loss per freed bit down to cheaper methods."""
    table, _, layers, methods = load_full_table(table_path)
    reason = _unpredicted(spec, table)
    if reason:
        raise TuneError(f"neighbours are ranked by the table, which cannot rank this grid: {reason}")
    objective = at.objective(table, subtract_floor=True)  # the table's kl when it has no floors
    value = [[objective[layer.name][method] for method in methods] for layer in layers]
    current = [methods.index(spec["grid"][layer.block][qr.KINDS.index(layer.kind)]) for layer in layers]
    options = spec.get("options")
    costs = [[layer.params * method_bits(method, layer.in_features, options) for method in methods]
             for layer in layers]
    spent = sum(costs[i][j] for i, j in enumerate(current))
    capacity = budget * sum(layer.params for layer in layers)
    moves = []
    for i, layer in enumerate(layers):
        for j, method in enumerate(methods):
            if j == current[i] or value[i][j] is None:
                continue
            choice = list(current)
            choice[i] = j
            delta = value[i][j] - value[i][current[i]]
            over = spent - costs[i][current[i]] + costs[i][j] - capacity
            # fund an overrun from the cheapest downgrades elsewhere: (loss per freed bit, layer, method)
            downgrades = sorted(
                ((value[k][m] - value[k][current[k]]) / (costs[k][current[k]] - costs[k][m]), k, m)
                for k in range(len(layers)) if k != i for m in range(len(methods))
                if value[k][m] is not None and costs[k][m] < costs[k][current[k]])
            used: set[int] = set()
            for _, k, m in downgrades:
                if over <= 0:
                    break
                if k in used:
                    continue
                used.add(k)
                over -= costs[k][current[k]] - costs[k][m]
                delta += value[k][m] - value[k][current[k]]
                choice[k] = m
            if over > 0:
                continue
            grid = [list(row) for row in spec["grid"]]
            for k, item in enumerate(layers):
                grid[item.block][qr.KINDS.index(item.kind)] = methods[choice[k]]
            moves.append({"predicted_delta": delta, "layer": layer.name, "to": method,
                          "funded_by": sorted(layers[k].name for k in used), "grid": grid})
    moves.sort(key=lambda move: (move["predicted_delta"], move["layer"], methods.index(move["to"])))
    return moves[:top]


def round_script(session: Path, names: Sequence[str], number: int) -> Path:
    """session/rounds/<n>/measure.sh: whole-recipe KL of the named proposals on the session's validation
    windows in one model load (``app.autotune measure --calibrate``: a recipe that needs calibration is
    calibrated on its own settings); TUNE_ROUND_DONE only when measure succeeded; plus round.json."""
    fixed = read_session(session)
    for name in names:
        read_proposal(session, name)
    folder = session / "rounds" / str(number)
    if folder.exists():
        raise TuneError(f"round {number} exists in {session}")
    folder.mkdir(parents=True)
    measure = fixed["measure"]
    recipes = " ".join(f'"$S/proposals/{name}/recipe.yaml"' for name in names)  # names: [A-Za-z0-9_.-]+
    revision = ["--revision", fixed["model_sha"]] if fixed.get("model_sha") else []
    command = shlex.join(["--model", fixed["model_id"], *revision, "--device", "cuda", "--split",
                          measure["split"], "--windows", str(measure["windows"]), "--seqlen",
                          str(measure["seqlen"]), "--seed", str(measure["seed"]), "--calibrate"])
    script = "\n".join([
        f"# tune round {number}: whole-recipe KL of {', '.join(names)} on the session's validation windows.",
        "# S = this session directory on the cluster; T = a clean TriCast tree; PY = its python.",
        'S=${S:?set S to the session directory}; T=${T:?set T to the TriCast tree}; PY=${PY:-python}',
        'cd "$T" || exit 3',
        # CALIB_CACHE (remote.env): tricast calibrate's per-layer result cache, shared by sessions of a model
        f'"$PY" -m app.autotune measure {command} ${{CALIB_CACHE:+--calib-cache "$CALIB_CACHE"}} '
        f'--out "$S/rounds/{number}/kl.json" {recipes} '
        f'|| {{ rc=$?; echo "TUNE_ROUND_FAIL round={number} rc=$rc"; exit "$rc"; }}',
        f'echo "TUNE_ROUND_DONE round={number}"', ""])
    (folder / "measure.sh").write_text(script, encoding="utf-8")
    record = {"schema": ROUND_SCHEMA, "round": number, "names": list(names), "measure": measure}
    (folder / "round.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return folder / "measure.sh"


def _matches(record: dict[str, Any], fixed: dict[str, Any]) -> bool:
    """A ledger measurement of the session's model, and for whole-recipe KL on the session's windows."""
    if (record.get("model_sha") != fixed["model_sha"]) if fixed.get("model_sha") \
            else (record.get("model_id") != fixed["model_id"]):
        return False
    if record["kind"] != "recipe_kl":
        return True
    setting, measure = record.get("setting") or {}, fixed["measure"]
    return all(setting.get(key) == measure[key] for key in ("windows", "seqlen", "seed", "split"))


def state(session: Path, ledger: Path) -> dict[str, Any]:
    """Every proposal of the session with what the ledger measured for its recipe hash on the session's model
    (whole-recipe KL only on the session's validation windows)."""
    fixed = read_session(session)
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in lg.read_ledger(ledger):
        if record.get("recipe_hash") and record["kind"] in ("recipe_kl", "ppl", "lmeval") \
                and _matches(record, fixed):
            by_hash.setdefault(record["recipe_hash"], []).append(record)
    rows = []
    for path in sorted((session / "proposals").glob("*/proposal.json")):
        proposal = json.loads(path.read_text(encoding="utf-8"))
        measured = by_hash.get(proposal["recipe_sha256"], [])
        kl = sorted((r["value"], r["id"], (r.get("setting") or {}).get("seed")) for r in measured
                    if r["kind"] == "recipe_kl")
        ppl = sorted((r["value"], r["id"]) for r in measured if r["kind"] == "ppl")
        rows.append({"name": proposal["name"], "author": proposal["author"], "parent": proposal["parent"],
                     "avg_bits": proposal["avg_bits"], "predicted": proposal["predicted"],
                     "measured_kl": [{"kl": v, "id": i, "seed": s} for v, i, s in kl],
                     "ppl": [{"ppl": v, "id": i} for v, i in ppl]})
    rounds = sorted(int(path.parent.name) for path in (session / "rounds").glob("*/round.json"))
    return {"session": str(session), "protocol": fixed, "proposals": rows, "rounds": rounds,
            "measured_proposals": sum(bool(row["measured_kl"]) for row in rows)}


# ---------- CLI ----------
def _fraction(text: str) -> Fraction:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"not a number or fraction: {text!r}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.tune_agent", description=__doc__.split("\n\n")[0])
    parser.add_argument("--ledger", type=Path, default=Path("runs/ledger.jsonl"))
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="rank ledger documents for a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    layer = sub.add_parser("layer", help="records of the layers matching a glob")
    layer.add_argument("pattern")
    layer.add_argument("--kind", choices=lg.KINDS)
    for name in ("propose", "neighbors", "round", "state"):
        item = sub.add_parser(name)
        item.add_argument("--session", type=Path, required=True)
        if name in ("propose", "neighbors"):
            item.add_argument("--table", type=Path, required=True)
            item.add_argument("--budget-bits", type=_fraction, required=True)
    proposer = sub.choices["propose"]
    proposer.add_argument("--name", required=True)
    proposer.add_argument("--grid", type=Path, required=True)
    proposer.add_argument("--rationale", required=True)
    proposer.add_argument("--cite", required=True, help="comma-separated ledger ids (at least one)")
    proposer.add_argument("--parent")
    proposer.add_argument("--author", default="claude-code")
    near = sub.choices["neighbors"]
    near.add_argument("--from", dest="source", required=True, help="a proposal name or a grid json path")
    near.add_argument("--top", type=int, default=8)
    near.add_argument("--write", type=Path, help="write each neighbour grid as <dir>/n<i>.json")
    rounder = sub.choices["round"]
    rounder.add_argument("--names", required=True)
    rounder.add_argument("--round", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "search":
            from tricast.rag import search_evidence

            output: Any = search_evidence(args.query, args.k, ledgers=[str(lg.docs_path(args.ledger))])
        elif args.command == "layer":
            output = [record for record in lg.read_ledger(args.ledger)
                      if fnmatch.fnmatchcase(record.get("layer", ""), args.pattern)
                      and (args.kind is None or record["kind"] == args.kind)]
        elif args.command == "propose":
            cites = [item for item in args.cite.split(",") if item]
            output = propose(args.session, args.name, load_grid(args.grid), args.table, args.budget_bits,
                             args.rationale, cites, args.ledger, author=args.author, parent=args.parent)
        elif args.command == "neighbors":
            source = Path(args.source)
            spec = load_grid(source) if source.suffix == ".json" else \
                read_proposal(args.session, args.source)["spec"]
            output = neighbors(spec, args.table, args.budget_bits, args.top)
            if args.write:
                args.write.mkdir(parents=True, exist_ok=True)
                for index, move in enumerate(output):
                    grid = json.dumps({**spec, "grid": move["grid"]})
                    (args.write / f"n{index}.json").write_text(grid + "\n", encoding="utf-8")
            output = [{key: value for key, value in move.items() if key != "grid"} for move in output]
        elif args.command == "round":
            output = {"script": str(round_script(args.session, args.names.split(","), args.round))}
        else:
            output = state(args.session, args.ledger)
    except (TuneError, at.AutotuneError, lg.LedgerError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
