"""Layer cards: one JSON object per decoder Linear with what TriCast measured or knows about it, so the
Auto-tune page, the heuristics (the knapsack prices and values layers from the same table) and the agent
(which retrieves the ledger's layer documents built from the same records) work from one set of facts.

    python -m app.layer_cards --table <run>/table.json [--profile profile.json] [--ledger runs/ledger.jsonl]
        --out cards.json

Per layer: shape and size; for every tabled method its stored bits (``app.autotune.weight_bits``), size in
bytes, single-layer KL and the floor-subtracted value the solver uses, and the method's option space
(``app.quantizer_recipe.METHOD_OPTIONS`` with defaults); the per-layer report metrics (MSE, SQNR, cosine of
weight, activation and output) of every recipe the ledger has a ``tricast report`` for on this model; and
the unquantized layer profile (``app.layer_profile``: weight / activation distribution, sparsity
sensitivity). Every number taken from a recorded file carries its ledger record id when the ledger has it
(``ids``), so a citation can be checked. Pure stdlib plus app modules; no torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app import autotune as at
from app import ledger as lg
from app import quantizer_recipe as qr

SCHEMA = "tricast.layer_cards/1"


def _option_space(method: str) -> dict[str, Any]:
    try:  # indexing, not .get: a Python-only method's options come from METHOD_OPTIONS.__missing__
        names = qr.METHOD_OPTIONS[method]
    except KeyError:
        names = ()
    return {option: qr.OPTIONS[option].get("default") for option in names}


def cards(table_path: Path, profile_path: Path | None = None, ledger_path: Path | None = None,
          root: Path | None = None) -> dict[str, Any]:
    table, raw, layers, methods = at.load_table(table_path)
    meta = table["meta"]
    floor = at.floors(table)
    values = at.objective(table, floor is not None)
    kl = at.objective(table, False)
    records = lg.read_ledger(ledger_path) if ledger_path is not None and ledger_path.is_file() else []
    root = root or Path.cwd()
    source = lg._relative(table_path, root)
    kl_ids = {(record["layer"], record["method"]): record["id"] for record in records
              if record["kind"] == "layer_kl" and record["source"]["path"] == source}
    reports: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["kind"] == "layer_report" and record.get("model_sha") == meta.get("model_sha"):
            label = record.get("recipe_name") or str(record.get("recipe_hash"))[:12]
            entry = reports[record["layer"]].setdefault(label, {"recipe_hash": record.get("recipe_hash"),
                                                               "metrics": {}, "ids": {}})
            key = f"{record['tensor']}.{record['metric']}"
            entry["metrics"][key] = record["value"]
            entry["ids"][key] = record["id"]
    profile = {}
    profile_meta = None
    if profile_path is not None:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        if data.get("meta", {}).get("model_sha") != meta.get("model_sha"):
            raise ValueError(f"{profile_path}: a profile of another model revision than the table's")
        profile = {row["name"]: row for row in data["layers"]}
        profile_meta = {key: data["meta"].get(key) for key in ("windows", "seqlen", "seed", "source",
                                                                "gpu_names", "git_sha", "git_dirty")}
        psource = lg._relative(profile_path, root)
        stat_ids = {(record["layer"], record["tensor"], record["metric"]): record["id"] for record in records
                    if record["kind"] == "layer_stat" and record["source"]["path"] == psource}
    rows = []
    for layer in layers:
        options = {}
        for method in methods:
            if kl[layer.name][method] is None:
                continue
            bits = at.weight_bits(method, layer.in_features)
            options[method] = {"bits": float(bits), "bytes": float(bits * layer.params / 8),
                               "kl": kl[layer.name][method], "value": values[layer.name][method],
                               "options": _option_space(method), "id": kl_ids.get((layer.name, method))}
        card: dict[str, Any] = {"name": layer.name, "block": layer.block, "kind": layer.kind,
                                "params": layer.params, "in_features": layer.in_features,
                                "out_features": layer.params // layer.in_features,
                                "bf16_bytes": layer.params * 2,
                                "floor": None if floor is None else floor[layer.name],
                                "floor_id": kl_ids.get((layer.name, "floor")), "methods": options,
                                "reports": dict(sorted(reports.get(layer.name, {}).items()))}
        if layer.name in profile:
            row = profile[layer.name]
            card["profile"] = {part: row.get(part) for part in ("weight", "activation", "sparsity")}
            card["profile_ids"] = {f"{part}.{stat}": stat_ids.get((layer.name, part, stat))
                                   for part, stats in lg.PROFILE_STATS.items() for stat in stats}
        rows.append(card)
    return {"schema": SCHEMA, "table": {"path": source, "sha256": hashlib.sha256(raw).hexdigest(),
                                        "objective": "kl_minus_floor" if floor is not None else "kl",
                                        **{key: meta.get(key) for key in ("model_id", "model_sha", "mma",
                                                                          "windows", "seqlen", "seed")}},
            "profile": profile_meta, "layers": rows}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.layer_cards", description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--ledger", type=Path, default=Path("runs/ledger.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = cards(args.table, args.profile, args.ledger)
    except (at.AutotuneError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
    with_ids = sum(1 for card in payload["layers"] for row in card["methods"].values() if row["id"])
    print(f"LAYER_CARDS_DONE layers={len(payload['layers'])} method_entries_with_ids={with_ids} "
          f"out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
