"""Export what the Auto-tune page (``app/web/autotune.html``) reads: one per-layer KL table in the page's
form, the uniform baselines, the knapsack solutions and a tune-agent session, every measured number with its
record.

Pure stdlib plus ``app.autotune`` / ``app.quantizer_recipe`` / ``app.export_quantizer_demo`` (and
``app.tune_agent`` / ``app.ledger`` for a session): no torch, no tricast.

    python -m app.export_autotune_demo --table <run>/table.json --out app/web/demo/autotune \\
        [--measured <run> [--transfer <dir>] [--outputs <records>]] \\
        [--session <session> --ledger runs/ledger.jsonl]

Writes ``<out>/bundle.json`` (indent=1, sorted keys, UTF-8; byte-identical for the same inputs).

The table is written as the solver sees it (``app.autotune.objective``: the floor-subtracted KL when the
table has floors), with each method's ``WEIGHT_BITS`` so the page's solver (``app/web/js/mckp.js``) prices
layers exactly as ``app.autotune.solve`` does. A measured number is attached to a grid only when the measured
recipe file is the YAML the generator writes for that grid (its SHA-256 equals the record's ``file_sha256``)
and a test PPL only when the PPL record's ``recipe_hash`` is that measurement's ``recipe_sha256``; otherwise
the field stays null with the reason. With ``--outputs`` (``scripts/quantizer/tune_outputs.py record``
records), a grid with no KL measurement takes the test PPL recorded with its answers, matched by the same
recipe-file SHA-256 (the KL stays null). With ``--measured`` and a ledger, each layer also gets the per-layer
``tricast report`` errors of every method's measured uniform recipe (``layer_reports``, same recipe hash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app import autotune as at
from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr

SCHEMA = "tricast.autotune_demo/1"
TABLE_META = ("model_id", "revision", "model_sha", "dtype", "device", "gpu_names", "windows", "seqlen",
              "seed", "source", "dataset_fingerprint", "git_sha", "git_dirty", "mma", "calibration",
              "methods")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def table_bundle(path: Path, profile: Path | None = None) -> dict[str, Any]:
    """The table in the page's form: per layer, the raw KL, the floor and the solver's value of every method
    in the table's order (null where the method is not tabled for that layer), and with ``profile`` (an
    ``app.layer_profile`` file of the same model revision) the layer's distribution and sparsity."""
    table, raw, layers, methods = at.load_table(path)
    stats: dict[str, dict[str, Any]] = {}
    if profile is not None:
        data = json.loads(profile.read_text(encoding="utf-8"))
        if data.get("meta", {}).get("model_sha") != table["meta"].get("model_sha"):
            raise ValueError(f"{profile}: a profile of another model revision than the table's")
        stats = {row["name"]: {part: row.get(part) for part in ("weight", "activation", "sparsity")}
                 for row in data["layers"]}
    floor = at.floors(table)
    values = at.objective(table, floor is not None)
    kl = at.objective(table, False)
    meta = table["meta"]
    return {
        "sha256": _sha256(raw), "schema": table["schema"], "methods": methods, "path": str(path),
        "objective": "kl_minus_floor" if floor is not None else "kl", "floor_rule": at.FLOOR_RULE,
        "weight_bits": {method: list(ex.WEIGHT_BITS[method]) for method in methods},
        "kinds": list(qr.KINDS), "blocks": at.meta_blocks(table),
        "meta": {key: meta.get(key) for key in TABLE_META},
        "layers": [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
                    "in_features": layer.in_features, "kl": [kl[layer.name][method] for method in methods],
                    "value": [values[layer.name][method] for method in methods],
                    "floor": None if floor is None else floor[layer.name],
                    **({"profile": stats[layer.name]} if layer.name in stats else {})} for layer in layers],
    }


def recipe_file_sha256(spec: dict[str, Any]) -> str:
    """SHA-256 of the recipe file ``app.autotune.write_grid`` and ``--from-json`` write for a grid spec."""
    recipe = qr.build(spec["name"], spec["default"], spec["grid"], spec.get("kv", "none"),
                      calibration=spec.get("calibration"), mma=spec.get("mma"))
    return _sha256(f"---\n{qr.to_yaml(recipe)}\n".encode())


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


Records = dict[str, dict[str, Any]]


def measurements(root: Path) -> tuple[Records, Records, dict[str, Any] | None]:
    """The run directory's whole-recipe KL results by recipe-file SHA-256 (top-level ``kl_seed*.json``;
    results measured with calibration overrides are left out), its finished PPL records by recipe hash
    (``ppl*/*.json``) and the native (unpatched BF16) PPL record, if any."""
    kl: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("kl_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        for row in data.get("results", []):
            if row.get("calibration_overrides"):
                continue
            kl.setdefault(row["file_sha256"], {
                "kl": row["kl"], "recipe_sha256": row["recipe_sha256"], "name": row["name"],
                "source": _rel(path, root), "gpu": (meta.get("gpu_names") or [None])[0],
                "git_sha": meta.get("git_sha"), "git_dirty": meta.get("git_dirty"), "seed": meta.get("seed"),
                "windows": meta.get("windows"), "seqlen": meta.get("seqlen"),
                "split": (meta.get("source") or {}).get("split")})
    ppl: dict[str, dict[str, Any]] = {}
    native = None
    for path in sorted(root.glob("ppl*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        value = ((record.get("metrics") or {}).get("ppl") or {})
        env = record.get("env") or {}
        view = {"ppl": value.get("ppl"), "n_tokens": value.get("n_tokens"), "source": _rel(path, root),
                "gpu": (env.get("gpu_names") or [env.get("gpu")])[0], "git_sha": env.get("git_sha"),
                "git_dirty": env.get("git_dirty")}
        if record.get("recipe") is None or path.stem == "native":
            native = native or view
        elif record.get("recipe_hash"):
            ppl.setdefault(record["recipe_hash"], view)
    return kl, ppl, native


def output_records(root: Path, meta: dict[str, Any]) -> tuple[Records, dict[str, Any] | None]:
    """``scripts/quantizer/tune_outputs.py record`` results (one JSON per recipe file): the test PPL measured
    with the answers, by recipe-file SHA-256, and the unquantized model's (``native.json``). Every record
    must be of the table's model and revision (a uniform recipe's file names neither, so its SHA-256 is the
    same for every model) and a full measurement (not ``--smoke``)."""
    by_file: Records = {}
    native = None
    for path in sorted(root.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("smoke"):
            raise ValueError(f"{path}: a --smoke record (2 PPL windows), not a measurement")
        loaded = (row.get("env") or {}).get("model_sha")  # the commit the record loaded, not the asked ref
        if row.get("model") != meta.get("model_id") or loaded is None or loaded != meta.get("model_sha"):
            raise ValueError(f"{path}: a record of {row.get('model')}@{loaded}, not of the table's "
                             f"{meta.get('model_id')}@{meta.get('model_sha')}")
        env = row.get("env") or {}
        view = {"ppl": row["ppl"]["ppl"], "n_tokens": row["ppl"]["n_tokens"],
                "source": f"{root.name}/{path.name}", "gpu": (env.get("gpu_names") or [None])[0],
                "git_sha": env.get("git_sha"), "git_dirty": env.get("git_dirty")}
        if row.get("name") == "native":
            native = view
        else:
            by_file[row["file_sha256"]] = view
    return by_file, native


def attach(row: dict[str, Any], spec: dict[str, Any], kl: Records, ppl: Records,
           digest: str | None = None, outputs: Records | None = None) -> dict[str, Any]:
    """``row`` with the measured KL of the grid's exact recipe file (``digest``, by default the file
    ``recipe_file_sha256`` names) and that recipe's PPL, or without a KL measurement the PPL recorded with
    the file's answers (``outputs``), or null + why."""
    digest = digest or recipe_file_sha256(spec)
    measured = kl.get(digest)
    row["recipe_file_sha256"] = digest
    row["kl"] = None if measured is None else measured["kl"]
    row["kl_record"] = measured
    row["ppl"] = None
    row["ppl_record"] = None
    row["rejected"] = []
    if measured is None:
        recorded = (outputs or {}).get(digest)
        if recorded is not None:
            row["ppl"], row["ppl_record"] = recorded["ppl"], recorded
            return row
        row["rejected"].append("측정 기록 없음: 이 격자의 레시피 파일 해시와 같은 KL 기록이 없습니다")
        return row
    record = ppl.get(measured["recipe_sha256"])
    if record is None:
        row["rejected"].append("test PPL 기록 없음: 같은 레시피 해시의 완료된 PPL 기록이 없습니다")
    else:
        row["ppl"], row["ppl_record"] = record["ppl"], record
    return row


def baselines(table: dict[str, Any], kl: Records, ppl: Records,
              layers: list[at.Layer], outputs: Records | None = None) -> list[dict[str, Any]]:
    """Every uniform recipe (one method on every Linear, named u_<method> as the uniform runs name them) whose
    measured KL is in the run, smallest bits first."""
    meta = table["meta"]
    calibration = meta.get("calibration")
    out = []
    for method in table["methods"]:
        if method == "bf16":
            continue
        spec = {"name": f"u_{method}", "default": method, "grid": qr.uniform(method, table["blocks"]),
                "kv": "none", "calibration": calibration if method in qr.CALIBRATED_ALL else None,
                "mma": meta.get("mma")}
        bits = at.avg_bits(layers, [method] * len(layers))
        row = attach({"id": spec["name"], "method": method, "bits": float(bits), "bits_fraction": str(bits)},
                     spec, kl, ppl, outputs=outputs)
        if row["kl"] is not None or row["ppl"] is not None:
            out.append(row)
    return sorted(out, key=lambda row: (row["bits"], row["id"]))


REPORT_METRICS = (("weight", "mse"), ("output", "sqnr_db"), ("output", "cosine"))


def layer_reports(table: dict[str, Any], rows: list[dict[str, Any]], ledger: Path) -> dict[str, Any]:
    """Write into every layer of the page's ``table`` a ``report`` list in the method order: the
    ``REPORT_METRICS`` values of the ``tricast report`` of that method's uniform recipe (one of ``rows``), or
    null. A report counts only when its ``recipe_hash`` is the uniform KL record's ``recipe_sha256`` (the
    same recipe) and its model revision is the table's; reports of one recipe from several files take the
    first path. Returns, per method, the report file, recipe hash, setting and environment."""
    from app import ledger as lg

    method_of = {row["kl_record"]["recipe_sha256"]: row["method"] for row in rows if row["kl_record"]}
    wanted = {f"{tensor}.{metric}" for tensor, metric in REPORT_METRICS}
    sources: dict[str, dict[str, Any]] = {}
    found: dict[tuple[str, str], dict[str, float]] = {}
    for record in sorted(lg.read_ledger(ledger), key=lambda record: record["source"]["path"]):
        method = method_of.get(record.get("recipe_hash"))
        key = f"{record.get('tensor')}.{record.get('metric')}"
        if (record["kind"] != "layer_report" or method is None or key not in wanted
                or record.get("model_sha") != table["meta"].get("model_sha")):
            continue
        env = record.get("env") or {}
        source = sources.setdefault(method, {
            "path": record["source"]["path"], "sha256": record["source"].get("sha256"),
            "recipe_hash": record["recipe_hash"], "recipe_name": record.get("recipe_name"),
            "setting": record.get("setting"), "gpu": (env.get("gpu_names") or [None])[0],
            "git_sha": env.get("git_sha"), "git_dirty": env.get("git_dirty")})
        if record["source"]["path"] == source["path"]:
            found.setdefault((record["layer"], method), {})[key] = record["value"]
    order = [f"{tensor}.{metric}" for tensor, metric in REPORT_METRICS]
    for layer in table["layers"]:
        cells = [found.get((layer["name"], method)) for method in table["methods"]]
        layer["report"] = [None if cell is None else [cell.get(key) for key in order] for cell in cells]
    return {"metrics": order, "methods": dict(sorted(sources.items()))}


def solutions(root: Path, kl: Records, ppl: Records, table_sha: str,
              outputs: Records | None = None) -> list[dict[str, Any]]:
    """The knapsack solutions of the run (``*/solution.json`` + ``grid.json`` solved on this table)."""
    out = []
    for path in sorted(root.glob("*/solution.json")):
        solution = json.loads(path.read_text(encoding="utf-8"))
        if solution.get("table", {}).get("sha256") != table_sha:
            continue
        spec = json.loads((path.parent / "grid.json").read_text(encoding="utf-8"))
        row = {"id": solution["name"], "dir": _rel(path.parent, root), "objective": solution["objective"],
               "budget_bits": solution["budget"]["given"].get("budget_bits"),
               "bits": solution["achieved"]["avg_bits"],
               "predicted_kl_sum": solution["predicted_kl_sum"],
               "predicted_kl_sum_minus_floor": solution["predicted_kl_sum_minus_floor"],
               "method_counts": solution["method_counts"], "grid": spec["grid"], "kv": spec.get("kv", "none"),
               "compute": solution.get("compute")}
        out.append(attach(row, spec, kl, ppl, outputs=outputs))
    return sorted(out, key=lambda row: (row["bits"], row["id"]))


def transfers(root: Path, solved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Knapsack solutions with a session's method options applied and their cells unchanged
    (``scripts/quantizer/autotune_xfer.py``: ``<root>/<name>/grid.json`` with ``base`` and ``options_from``),
    each with the measured KL of its exact recipe file — the tune loop's proposal file — and that recipe's PPL
    (``kl_seed*.json`` / ``ppl*/`` in ``root`` or a directory under it). A grid whose cells are not its base
    solution's is rejected: it would not be the same size."""
    from app import tune_agent as ta

    kl: Records = {}
    ppl: Records = {}
    for part in [root, *sorted(path for path in root.iterdir() if path.is_dir())]:
        found_kl, found_ppl, _ = measurements(part)
        for key, value in found_kl.items():
            kl.setdefault(key, value)
        for key, value in found_ppl.items():
            ppl.setdefault(key, value)
    by_id = {row["id"]: row for row in solved}
    out = []
    for path in sorted(root.glob("*/grid.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "base" not in raw:
            continue
        spec, base = ta.load_grid(path), by_id.get(raw["base"])
        text = qr.to_yaml(ta._recipe(raw["name"], spec)) + "\n"  # as tune_agent.propose writes recipe.yaml
        row = {"id": raw["name"], "base": raw["base"], "options_from": raw.get("options_from"),
               "dir": _rel(path.parent, root), "options_changed": changed_options(spec["options"]),
               "bits": None if base is None else base["bits"], "grid": spec["grid"]}
        attach(row, spec, kl, ppl, _sha256(text.encode()))
        if base is None or base["grid"] != spec["grid"]:
            row["rejected"].insert(0, f"기준 해 {raw['base']} 와 칸이 다릅니다: 같은 크기의 옵션 적용이 아님")
            row.update(kl=None, ppl=None)
        out.append(row)
    return sorted(out, key=lambda row: (row["bits"] is None, row["bits"], row["id"]))


def changed_options(options: Any) -> dict[str, dict[str, Any]] | None:
    """Per method, the options a proposal sets away from their defaults
    (``app.quantizer_recipe.method_options``), methods with none left out; null when the options do not check
    (a refused reply can carry such)."""
    if not isinstance(options, dict):
        return None
    try:
        changed = {method: qr.method_options(method, values) for method, values in options.items()}
    except (KeyError, TypeError, ValueError):
        return None
    return {method: values for method, values in changed.items() if values}


def grid_changes(session: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    """The cells a recorded proposal sets away from its parent's grid, in block then kind order: the changes
    of a proposer with no LLM reply on file (``tune_loop.replan``)."""
    from app import tune_agent as ta

    if record.get("parent") is None:
        return []
    base = ta.read_proposal(session, record["parent"])["spec"]["grid"]
    return [{"block": block, "kind": kind, "method": record["spec"]["grid"][block][k]}
            for block, row in enumerate(base) for k, kind in enumerate(qr.KINDS)
            if record["spec"]["grid"][block][k] != row[k]]


def session_bundle(session: Path, ledger: Path) -> dict[str, Any]:
    """A tune-loop session: per round the proposer's analysis and spend, every proposal it made (accepted with
    its checks' computed values and measurements, or refused with the reason) and the ledger records cited."""
    from app import ledger as lg
    from app import tune_agent as ta

    state = ta.state(session, ledger)
    rows = {row["name"]: row for row in state["proposals"]}
    loop = session / "loop.json"
    config = json.loads(loop.read_text(encoding="utf-8")) if loop.is_file() else {}
    summary = json.loads((session / "loop" / "summary.json").read_text(encoding="utf-8"))
    records = {record["id"]: record for record in lg.read_ledger(ledger)}
    cited: set[str] = set()
    rounds = []
    for entry in summary:
        number = entry["round"]
        raw = session / "llm" / f"round_{number}.json"
        llm, replied = None, {}
        if raw.is_file():
            call = json.loads(raw.read_text(encoding="utf-8"))
            out = json.loads(call["stdout"]) if call.get("stdout") else {}
            reply = out.get("structured_output") or {}
            replied = {item["name"]: item for item in reply.get("proposals", [])}
            llm = {"models": sorted(out.get("modelUsage") or {}), "cost_usd": out.get("total_cost_usd"),
                   "duration_s": None if out.get("duration_ms") is None else out["duration_ms"] / 1000,
                   "returncode": call.get("returncode"), "started_utc": call.get("started_utc"),
                   "aliases": call.get("aliases") is not None}
        refused = {item["name"]: item["reason"] for item in entry.get("refused", [])}
        proposals = []
        for name in [*entry.get("accepted", []), *refused]:
            item = replied.get(name, {})
            row = rows.get(name)
            record = ta.read_proposal(session, name) if row is not None else None
            kept = record or {}
            changes = item.get("changes", grid_changes(session, record) if record and not item else [])
            proposal = {"name": name, "base": item.get("base", kept.get("parent")),
                        "changes": changes, "options": item.get("options"),
                        "options_changed": changed_options(item.get("options")),
                        "rationale": item.get("rationale", kept.get("rationale")),
                        "cites": item.get("cites", kept.get("cites", [])),
                        "status": "refused" if name in refused else "accepted", "reason": refused.get(name),
                        "author": kept.get("author")}
            if row is not None:
                proposal.update({"avg_bits": row["avg_bits"], "predicted": row["predicted"],
                                 "kl": row["measured_kl"][0]["kl"] if row["measured_kl"] else None,
                                 "kl_id": row["measured_kl"][0]["id"] if row["measured_kl"] else None,
                                 "ppl": row["ppl"][0]["ppl"] if row.get("ppl") else None,
                                 "ppl_id": row["ppl"][0]["id"] if row.get("ppl") else None,
                                 "grid": record["spec"]["grid"]})
            cited.update(proposal["cites"])
            cited.update(key for key in (proposal.get("kl_id"), proposal.get("ppl_id")) if key)
            proposals.append(proposal)
        rounds.append({"round": number, "analysis": entry.get("analysis"), "stop": entry.get("stop"),
                       "llm": llm, "proposals": proposals})
    measured = [row for row in state["proposals"] if row["measured_kl"]]
    best = min(measured, default=None,
               key=lambda row: (row["measured_kl"][0]["kl"], row["avg_bits"], row["name"]))
    start = next((row["name"] for row in state["proposals"] if row["parent"] is None), None)

    def view(record: dict[str, Any]) -> dict[str, Any]:
        env = record.get("env") or {}
        return {"kind": record.get("kind"), "layer": record.get("layer"), "method": record.get("method"),
                "metric": record.get("metric"), "value": record.get("value"),
                "recipe_hash": record.get("recipe_hash"), "source": (record.get("source") or {}).get("path"),
                "gpu": (env.get("gpu_names") or [None])[0], "git_sha": env.get("git_sha")}

    return {"name": session.name, "budget_bits": config.get("budget_bits"), "start": start,
            "proposer": "claude-code", "rounds": rounds,
            "best": None if best is None else {"name": best["name"], "kl": best["measured_kl"][0]["kl"],
                                               "ppl": best["ppl"][0]["ppl"] if best.get("ppl") else None,
                                               "avg_bits": best["avg_bits"]},
            "records": {key: view(records[key]) for key in sorted(cited) if key in records},
            "missing_records": sorted(key for key in cited if key not in records)}


def dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.export_autotune_demo",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--measured", type=Path,
                        help="run directory with kl_seed*.json, ppl*/ and solution directories")
    parser.add_argument("--session", type=Path, help="a tune-loop session directory (with loop/summary.json)")
    parser.add_argument("--ledger", type=Path, default=Path("runs/ledger.jsonl"))
    parser.add_argument("--profile", type=Path, help="an app.layer_profile file of the table's model")
    parser.add_argument("--transfer", type=Path,
                        help="option-transfer grids and their measurements "
                             "(scripts/quantizer/autotune_xfer.py); needs --measured")
    parser.add_argument("--outputs", type=Path,
                        help="scripts/quantizer/tune_outputs.py records (PPL with answers); needs --measured")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        table = table_bundle(args.table, args.profile)
        payload: dict[str, Any] = {"schema": SCHEMA, "table": table, "native": None, "baselines": [],
                                   "solutions": [], "transfer": [], "session": None}
        if args.transfer is not None and args.measured is None:
            raise ValueError("--transfer needs --measured (the solutions it applies options to)")
        if args.outputs is not None and args.measured is None:
            raise ValueError("--outputs needs --measured (the solutions they are matched to)")
        if args.measured is not None:
            kl, ppl, native = measurements(args.measured)
            outputs, outputs_native = (output_records(args.outputs, table["meta"]) if args.outputs is not None
                                       else (None, None))
            _, _, layers, _ = at.load_table(args.table)
            payload.update(native=native or outputs_native,
                           baselines=baselines(table, kl, ppl, layers, outputs),
                           solutions=solutions(args.measured, kl, ppl, table["sha256"], outputs))
            if args.ledger.is_file():
                table["reports"] = layer_reports(table, payload["baselines"], args.ledger)
            if args.transfer is not None:
                payload["transfer"] = transfers(args.transfer, payload["solutions"])
        if args.session is not None:
            payload["session"] = session_bundle(args.session, args.ledger)
    except (at.AutotuneError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bundle.json").write_text(dump(payload), encoding="utf-8")
    for row in [*payload["baselines"], *payload["solutions"], *payload["transfer"]]:
        for reason in row["rejected"]:
            print(f"{row['id']}: {reason}")
    print(f"AUTOTUNE_DEMO_EXPORT_DONE out={args.out / 'bundle.json'} layers={len(table['layers'])} "
          f"baselines={len(payload['baselines'])} solutions={len(payload['solutions'])} "
          f"transfer={len(payload['transfer'])} "
          f"session_rounds={0 if payload['session'] is None else len(payload['session']['rounds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
