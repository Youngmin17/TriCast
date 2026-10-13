"""Export ``tricast report`` results as the bundle the Studio layer map reads (``app/web/layers.html``).

Pure stdlib (json, pathlib): no torch, no tricast, so it runs wherever the report files are.

    python -m app.export_layers runs/<run>/report --out app/web/demo/layers \\
        --label int8_row_w8a8="INT8 W8A8" --label mxfp8_w_a=MXFP8 ...

Each input is a ``report.json`` or a directory searched for ``report.json`` files. Only reports with
``status == "complete"`` and the keys listed in ``REQUIRED`` are accepted. Output (``indent=1``, sorted keys,
UTF-8, byte-identical for the same inputs and labels):

- ``index.json``: ``{"schema", "columns", "reports"}`` with one entry per report (id = recipe name): label,
  model id/sha, an env subset, samples/seqlen/dataset fingerprint, model-level metrics, layer count and the
  data file. Order: the ``--label`` order, then unlabeled reports by id.
- ``<id>.json``: ``rows`` of ``[name, block, kind, status, metrics…]`` (names in ``columns``) in block order
  (decoder block, then q, k, v, o, gate, up, down, then declaration order; layers outside numbered blocks
  last, in declaration order), ``spec[i]`` indexing ``specs`` (per-layer formats, algorithm, transform, MMA),
  ``kinds`` (module paths inside a block, in that order). Non-finite metrics, which the report writes as the
  strings "inf" / "-inf", become null and are listed in ``nonfinite`` as ``[row, column, token]``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "tricast.layer_map/1"
GROUPS = ("weight", "activation", "output")
ERROR_KEYS = ("mse", "sqnr_db", "max_abs_error", "relative_frobenius", "cosine")
ULP_KEYS = ("mean", "max", "p99", "exact_fraction")
COLUMNS = ("name", "block", "kind", "status", *(f"{group}.{key}" for group in GROUPS for key in ERROR_KEYS),
           *(f"mma_ulp.{key}" for key in ULP_KEYS))
REQUIRED = ("status", "layers", "patch_report", "model", "recipe", "env", "samples", "seqlen",
            "dataset_fingerprint")
MODEL_KEYS = ("logits_kl", "top1_agreement", "ppl_reference", "ppl_emulated", "n_tokens")
ENV_KEYS = ("git_sha", "git_dirty", "model_id", "model_sha", "gpu_names")
ENV_COPY = ("git_sha", "git_dirty", "src_sha256", "gpu_names", "cuda", "dtype", "seed", "utc", "versions")
KIND_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NONFINITE = ("inf", "-inf", "nan")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Used only when the report does not name its decoder layer list (patch_report.decoder_layers).
_BLOCK = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)\.(.+)$")


class ExportError(ValueError):
    """A report that cannot be exported; the message names the file and the field."""


def _require(where: str, name: str, value: Any, keys: Sequence[str]) -> None:
    if not isinstance(value, dict):
        raise ExportError(f"{where}: {name} must be an object")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ExportError(f"{where}: {name} is missing {', '.join(missing)}")


def load_report(path: Path) -> dict[str, Any]:
    """Read one report.json and check what the bundle needs; raises ExportError otherwise."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"{path}: cannot read JSON ({error})") from error
    if not isinstance(report, dict):
        raise ExportError(f"{path}: not a JSON object")
    if report.get("status") != "complete":
        raise ExportError(f"{path}: status is {report.get('status')!r}, expected 'complete' "
                          "(only finished tricast report runs are exported)")
    missing = [key for key in REQUIRED if key not in report]
    if missing:
        raise ExportError(f"{path}: missing keys {', '.join(missing)}")
    _require(str(path), "model", report["model"], MODEL_KEYS)
    _require(str(path), "env", report["env"], ENV_KEYS)
    if not isinstance(report["recipe"], dict) or not isinstance(report["recipe"].get("name"), str):
        raise ExportError(f"{path}: recipe.name must be a string")
    patch = report["patch_report"]
    if not isinstance(patch, dict) or not isinstance(patch.get("layers"), list):
        raise ExportError(f"{path}: patch_report.layers must be a list")
    layers = report["layers"]
    if not isinstance(layers, list) or not layers:
        raise ExportError(f"{path}: layers must be a nonempty list")
    names: set[str] = set()
    for index, layer in enumerate(layers):
        where = f"{path}: layers[{index}]"
        if not isinstance(layer, dict) or not isinstance(layer.get("name"), str) \
                or not isinstance(layer.get("status"), str):
            raise ExportError(f"{where}: needs a string name and status")
        if layer["name"] in names:
            raise ExportError(f"{where}: duplicate layer {layer['name']!r}")
        names.add(layer["name"])
        for group in (*GROUPS, "mma_ulp"):
            if group not in layer:
                raise ExportError(f"{where}: missing {group}")
            if layer[group] is not None:
                _require(where, group, layer[group], ULP_KEYS if group == "mma_ulp" else ERROR_KEYS)
    return report


def _number(value: Any, where: str) -> tuple[int | float | None, str | None]:
    """(number, None) for a finite number, (None, token) for a non-finite one, (None, None) for null."""
    if value is None:
        return None, None
    if isinstance(value, str) and value in NONFINITE:
        return None, value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{where}: expected a number, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        return None, "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
    return value, None


def block_and_kind(name: str, decoder: str | None) -> tuple[int | None, str]:
    """Decoder block number and the module path inside the block; (None, name) outside numbered blocks."""
    if decoder:
        if name.startswith(f"{decoder}."):
            index, _, rest = name[len(decoder) + 1:].partition(".")
            if index.isdigit() and rest:
                return int(index), rest
        return None, name
    match = _BLOCK.search(name)
    return (int(match.group(1)), match.group(2)) if match else (None, name)


def _kind_rank(kind: str) -> int:
    leaf = kind.rsplit(".", 1)[-1]
    return KIND_ORDER.index(leaf) if leaf in KIND_ORDER else len(KIND_ORDER)


def _order_key(name: str, decoder: str | None, declared: dict[str, int]) -> tuple[Any, ...]:
    block, kind = block_and_kind(name, decoder)
    position = declared.get(name, len(declared))
    if block is None:
        return (1, 0, 0, position, name)
    return (0, block, _kind_rank(kind), position, name)


def _quant(spec: Any) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return None
    return {key: spec.get(key) for key in ("format", "granularity", "group_size")}


def _layer_spec(entry: dict[str, Any] | None, ulp_format: Any) -> dict[str, Any] | None:
    if entry is None:
        return None
    mma = entry.get("mma")
    return {"weight": _quant(entry.get("weight")), "activation": _quant(entry.get("activation")),
            "weight_algo": entry.get("weight_algo"), "transform": entry.get("transform"),
            "sparsity": entry.get("sparsity"), "skipped": entry.get("skipped"),
            "mma": ({key: mma.get(key) for key in ("algorithm", "name", "f_bits", "g_bits")}
                    if isinstance(mma, dict) else None),
            "ulp_format": ulp_format}


def layer_file(report_id: str, label: str, report: dict[str, Any], path: Path) -> dict[str, Any]:
    """The per-report data file: compact rows in block order plus deduplicated per-layer specs."""
    patch = report["patch_report"]
    decoder = patch.get("decoder_layers") if isinstance(patch.get("decoder_layers"), str) else None
    entries = {entry["name"]: entry for entry in patch["layers"]
               if isinstance(entry, dict) and isinstance(entry.get("name"), str)}
    declared = {name: index for index, name in enumerate(entries)}
    layers = sorted(report["layers"], key=lambda layer: _order_key(layer["name"], decoder, declared))
    rows: list[list[Any]] = []
    nonfinite: list[list[Any]] = []
    spec_index: list[int] = []
    specs: list[dict[str, Any] | None] = []
    seen_specs: dict[str, int] = {}
    kinds: list[str] = []
    for row_index, layer in enumerate(layers):
        name = layer["name"]
        block, kind = block_and_kind(name, decoder)
        if kind not in kinds:
            kinds.append(kind)
        row: list[Any] = [name, block, kind, layer["status"]]
        for column in COLUMNS[4:]:
            group, key = column.split(".")
            values = layer[group]
            value, token = _number(values[key] if values is not None else None,
                                   f"{path}: {name}.{column}")
            row.append(value)
            if token is not None:
                nonfinite.append([row_index, column, token])
        rows.append(row)
        ulp = layer["mma_ulp"]
        spec = _layer_spec(entries.get(name), ulp.get("format") if ulp is not None else None)
        spec_key = json.dumps(spec, sort_keys=True)
        if spec_key not in seen_specs:
            seen_specs[spec_key] = len(specs)
            specs.append(spec)
        spec_index.append(seen_specs[spec_key])
    first = {kind: index for index, kind in enumerate(kinds)}
    kinds = sorted(kinds, key=lambda kind: (_kind_rank(kind), first[kind]))
    return {"schema": SCHEMA, "id": report_id, "label": label, "columns": list(COLUMNS), "rows": rows,
            "nonfinite": nonfinite, "spec": spec_index, "specs": specs, "kinds": kinds}


def index_entry(report_id: str, label: str, report: dict[str, Any], path: Path, file: str) -> dict[str, Any]:
    """One index.json entry: who measured what, where, and the model-level result."""
    env = report["env"]
    metrics: dict[str, Any] = {}
    nonfinite: dict[str, str] = {}
    for key in MODEL_KEYS:
        value, token = _number(report["model"][key], f"{path}: model.{key}")
        metrics[key] = value
        if token is not None:
            nonfinite[key] = token
    skipped = report["patch_report"].get("skipped") or []
    return {"id": report_id, "label": label, "file": file,
            "description": str(report["recipe"].get("description") or ""),
            "model": {"id": env["model_id"], "sha": env["model_sha"]},
            "env": {key: env[key] for key in ENV_COPY if key in env},
            "samples": report["samples"], "seqlen": report["seqlen"],
            "dataset_fingerprint": report["dataset_fingerprint"],
            "forward_mode": report.get("forward_mode"), "recipe_hash": report.get("recipe_hash"),
            "metrics": metrics, "nonfinite": nonfinite, "layers": len(report["layers"]),
            "skipped": sorted(str(name) for name in skipped)}


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def build_bundle(reports: Sequence[tuple[Path, dict[str, Any]]], labels: dict[str, str]) -> dict[str, str]:
    """File name -> text for the whole bundle. ``labels`` (recipe -> label) also sets the order."""
    by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, report in reports:
        report_id = report["recipe"]["name"]
        if not _ID.fullmatch(report_id):
            raise ExportError(f"{path}: recipe name {report_id!r} is not a safe file name")
        if report_id in by_id:
            raise ExportError(f"{path}: recipe {report_id!r} also comes from {by_id[report_id][0]}")
        by_id[report_id] = (path, report)
    unknown = [report_id for report_id in labels if report_id not in by_id]
    if unknown:
        raise ExportError(f"--label names recipes that are not among the reports: {', '.join(unknown)}")
    files: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for report_id in [*labels, *sorted(report_id for report_id in by_id if report_id not in labels)]:
        path, report = by_id[report_id]
        label = labels.get(report_id, report_id)
        file = f"{report_id}.json"
        files[file] = _dump(layer_file(report_id, label, report, path))
        entries.append(index_entry(report_id, label, report, path, file))
    files["index.json"] = _dump({"schema": SCHEMA, "columns": list(COLUMNS), "reports": entries})
    return files


def find_reports(inputs: Iterable[Path]) -> list[Path]:
    """report.json files named directly or found under directories, each once, in a stable order."""
    found: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            direct = item / "report.json"
            paths = [direct] if direct.is_file() else sorted(item.rglob("report.json"))
            if not paths:
                raise ExportError(f"{item}: no report.json inside")
        elif item.is_file():
            paths = [item]
        else:
            raise ExportError(f"{item}: no such file or directory")
        for path in paths:
            if path.resolve() not in seen:
                seen.add(path.resolve())
                found.append(path)
    return found


def write_bundle(out: Path, files: dict[str, str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(files):
        (out / name).write_text(files[name], encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.export_layers",
                                     description="Export tricast report results for the Studio layer map.")
    parser.add_argument("reports", nargs="+", type=Path, help="report.json files or directories holding them")
    parser.add_argument("--out", type=Path, required=True,
                        help="bundle directory (Studio reads app/web/demo/layers)")
    parser.add_argument("--label", action="append", default=[], metavar="RECIPE=LABEL",
                        help="display name for a recipe; repeat it, the order of --label is the page order")
    args = parser.parse_args(argv)
    labels: dict[str, str] = {}
    for item in args.label:
        report_id, sep, text = item.partition("=")
        if not sep or not report_id or not text:
            parser.error(f"--label expects RECIPE=LABEL, got {item!r}")
        if report_id in labels:
            parser.error(f"--label {report_id} is given twice")
        labels[report_id] = text
    try:
        reports = [(path, load_report(path)) for path in find_reports(args.reports)]
        files = build_bundle(reports, labels)
    except ExportError as error:
        print(f"export_layers: {error}", file=sys.stderr)
        return 1
    write_bundle(args.out, files)
    for name in sorted(files):
        print(f"{args.out / name}  {len(files[name].encode('utf-8'))} bytes")
    total = sum(len(text.encode("utf-8")) for text in files.values())
    print(f"{len(reports)} reports, {total} bytes in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
