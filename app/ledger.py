"""Evidence ledger: every number TriCast measured for a layer or a recipe, one record per number, plus the
retrieval documents ``tricast.rag.search_evidence`` ranks (docs/SPEC.md AC23).

    python -m app.ledger ingest runs/20261013_quantizer_w2 runs/20261014_autotune --ledger runs/ledger.jsonl
    python -m app.ledger stats --ledger runs/ledger.jsonl
    python -m app.ledger verify --ledger runs/ledger.jsonl
    python -m app.ledger get E1a2b3c4d5e6 --ledger runs/ledger.jsonl

ingest walks the given files and directories for the records TriCast writes and keeps only the ones it
recognises by shape:

- auto-tune table (``tricast.autotune.table/*`` or a ``table.entries.jsonl`` with its config header):
  ``layer_kl`` — KL of the model with only that layer on that method (``method: floor`` = patched, not
  quantized);
- auto-tune measure (``tricast.autotune.measure/*``): ``recipe_kl`` — KL of a whole recipe;
- auto-tune solution (``tricast.autotune.solution/*``): ``solution`` — the budget, achieved bits and the
  predicted KL sum;
- ``tricast run`` record (``run_hash`` + ``metrics`` + ``env``): ``ppl`` and ``lmeval`` (one per task and
  metric, with stderr and sample count);
- ``tricast report`` record (``layers`` + ``model`` + ``recipe_hash``): ``layer_report`` (one per layer,
  tensor and metric) and ``model_report``.

Every record carries the environment it was measured in (git SHA, dirty flag, GPU, versions) and its source:
the file's sha256 and a JSON pointer (``#L<n>/<key>`` for a jsonl line) to the exact value, so ``verify``
reads the number back. A value without an environment is refused (AGENTS.md §3.3). The record id is a hash
of (kind, source path, pointer). Re-ingesting a changed file: a value still at its pointer keeps its id and
gets the new file hash (a resumed run appends lines); a different value at the same pointer, or a pointer that
is gone, is refused (a result was rewritten in place), and so is a record whose value stayed but whose other
fields (model, layer, setting, environment) changed. Records are never deleted, so a cited id always resolves:
a finished ``<stem>.json`` table marks the records of its ``<stem>.entries.jsonl`` ``superseded_by`` its own
record of the same (layer, method) and value, and documents skip superseded records.

The documents group the records for retrieval — one per layer (every method's KL, floors and report metrics
across all sources), one per recipe hash (every metric measured with it) and one per (table, method) (its
most and least sensitive layers). Each document lists the ids it was built from; numbers in documents are
rounded to 6 significant digits, the records keep the exact values. Stdlib only.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence

try:
    import fcntl
except ImportError:  # not POSIX: ingests are not serialised
    fcntl = None
from pathlib import Path
from typing import Any

SCHEMA = "tricast.ledger/1"
DOCS_SCHEMA = "tricast.ledger.docs/1"
KINDS = ("layer_kl", "recipe_kl", "solution", "ppl", "lmeval", "layer_report", "model_report")
LAYER_METRICS = ("sqnr_db", "mse", "cosine", "relative_frobenius")  # per tensor of a tricast report layer
LAYER_TENSORS = ("weight", "activation", "output")
MODEL_METRICS = ("logits_kl", "top1_agreement", "ppl_reference", "ppl_emulated")
LMEVAL_METRICS = ("acc", "acc_norm")
ENV_KEYS = ("git_sha", "git_dirty", "hostname", "utc", "gpu_names", "torch", "transformers", "triton")
SETTING_KEYS = ("windows", "seqlen", "seed", "dataset_fingerprint", "dtype")


def _window_setting(meta: dict[str, Any]) -> dict[str, Any]:
    """The KL windows of an auto-tune table or measure: count, length, seed, split, fingerprint, dtype."""
    return {**{key: meta.get(key) for key in SETTING_KEYS}, "split": (meta.get("source") or {}).get("split")}
TOP = 5  # layers named per side in a method document


class LedgerError(ValueError):
    """A refused input; the CLI prints it and exits with status 2."""


# ---------- records ----------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _env(env: dict[str, Any] | None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """The comparable part of a TriCast environment capture (``env`` of a run, or an auto-tune ``meta``)."""
    merged: dict[str, Any] = {}
    for source in (meta or {}, env or {}):
        versions = source.get("versions") or {}
        for key in ENV_KEYS:
            value = source.get(key, versions.get(key))
            if value is not None:
                merged[key] = value
    return merged


def _record(kind: str, value: Any, source: dict[str, Any], env: dict[str, Any],
            **fields: Any) -> dict[str, Any]:
    where = f"{source['path']}{source['pointer']}"
    if kind not in KINDS:
        raise LedgerError(f"unknown record kind {kind!r}")
    if not _is_number(value):
        raise LedgerError(f"{where}: value is not a number ({value!r})")
    if not env.get("git_sha") or "git_dirty" not in env:
        raise LedgerError(f"{where}: no environment capture (AGENTS.md §3.3)")
    key = json.dumps([kind, source["path"], source["pointer"]])
    return {"id": "E" + _sha256(key.encode())[:12], "schema": SCHEMA, "kind": kind, "value": value,
            **{name: item for name, item in fields.items() if item is not None}, "env": env, "source": source}


def _pointer(*parts: Any) -> str:
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _layer_fields(name: str) -> dict[str, Any]:
    """block and kind of a decoder Linear ``model.layers.<b>.<self_attn|mlp>.<kind>``; name only otherwise."""
    parts = name.split(".")
    if len(parts) == 5 and parts[:2] == ["model", "layers"] and parts[2].isdigit():
        return {"layer": name, "block": int(parts[2]), "layer_kind": parts[4]}
    return {"layer": name}


def _model(env: dict[str, Any] | None) -> dict[str, Any]:
    return {"model_id": (env or {}).get("model_id"), "model_sha": (env or {}).get("model_sha")}


def _from_table(data: dict[str, Any], source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """One ``layer_kl`` per (layer, method); a not-applicable method (None) has no record. The patch floor is
    method ``floor``, whether the table keeps it among the methods or in the layer's ``floor`` field."""
    meta = data.get("meta") or {}
    env = _env(meta.get("env"), meta)
    setting = _window_setting(meta)
    recipes = meta.get("method_recipe_sha256") or {}
    for index, layer in enumerate(data.get("layers") or []):
        fields = {"model_id": meta.get("model_id"), "model_sha": meta.get("model_sha"),
                  **_layer_fields(layer["name"]), "metric": "kl", "setting": setting}
        for method, kl in sorted((layer.get("kl") or {}).items()):
            if kl is not None:
                pointer = _pointer("layers", index, "kl", method)
                yield _record("layer_kl", kl, {**source, "pointer": pointer}, env, method=method,
                              recipe_hash=recipes.get(method), **fields)
        if _is_number(layer.get("floor")):
            pointer = _pointer("layers", index, "floor")
            yield _record("layer_kl", layer["floor"], {**source, "pointer": pointer}, env, method="floor",
                          **fields)
    for index, row in enumerate(data.get("kv") or []):  # v2: one block's KV cache only, no Linear
        for preset, kl in sorted((row.get("kl") or {}).items()):
            if kl is not None:
                pointer = _pointer("kv", index, "kl", preset)
                yield _record("layer_kl", kl, {**source, "pointer": pointer}, env, method=f"kv:{preset}",
                              model_id=meta.get("model_id"), model_sha=meta.get("model_sha"),
                              layer=row["name"], block=row["block"], layer_kind="kv", metric="kl",
                              recipe_hash=(meta.get("kv_recipe_sha256") or {}).get(preset), setting=setting)


def _from_entries(lines: list[str], source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    meta = json.loads(lines[0])["config"]
    env = _env(None, meta)
    setting = _window_setting(meta)
    for number, line in enumerate(lines[1:], start=2):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:  # the cut-off last line of a killed run (app.autotune drops it too)
            continue
        if entry.get("kl") is None:
            continue
        yield _record("layer_kl", entry["kl"], {**source, "pointer": f"#L{number}/kl"}, env,
                      model_id=meta.get("model_id"), model_sha=meta.get("model_sha"),
                      **_layer_fields(entry["layer"]), method=entry["method"], metric="kl",
                      recipe_hash=entry.get("recipe_sha256"), setting=setting)


def _from_measure(data: dict[str, Any], source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    meta = data.get("meta") or {}
    env = _env(meta.get("env"), meta)
    setting = _window_setting(meta)
    for index, result in enumerate(data.get("results") or []):
        yield _record("recipe_kl", result["kl"], {**source, "pointer": _pointer("results", index, "kl")}, env,
                      model_id=meta.get("model_id"), model_sha=meta.get("model_sha"),
                      recipe_name=result.get("name"), recipe_hash=result.get("recipe_sha256"), metric="kl",
                      patched=result.get("patched"), kv=result.get("kv") or None, setting=setting)


def _from_solution(data: dict[str, Any], source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    table = data.get("table") or {}
    env = {"git_sha": table.get("git_sha"), "git_dirty": table.get("git_dirty")}
    achieved = data.get("achieved") or {}
    yield _record("solution", data.get("predicted_kl_sum"), {**source, "pointer": "/predicted_kl_sum"}, env,
                  model_id=table.get("model_id"), model_sha=table.get("model_sha"),
                  recipe_name=data.get("name"), metric="predicted_kl_sum", avg_bits=achieved.get("avg_bits"),
                  budget_bits=(data.get("budget") or {}).get("avg_bits"),
                  method_counts=data.get("method_counts"), kv=data.get("kv"),
                  table_sha256=table.get("sha256"))


def _recipe_fields(data: dict[str, Any]) -> dict[str, Any]:
    recipe = data.get("recipe") or {}
    return {"recipe_name": recipe.get("name"), "recipe_hash": data.get("recipe_hash"),
            "kv": (recipe.get("kv") or {}).get("preset"),
            "calibration": data.get("calibration_config") if data.get("calibration") else None}


def _from_run(data: dict[str, Any], source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    env, model, recipe = _env(data.get("env")), _model(data.get("env")), _recipe_fields(data)
    metrics = data.get("metrics") or {}
    ppl = metrics.get("ppl")
    if isinstance(ppl, dict) and "ppl" in ppl:
        setting = {key: ppl.get(key) for key in ("n_windows", "dataset_fingerprint", "forward_mode")}
        yield _record("ppl", ppl["ppl"], {**source, "pointer": "/metrics/ppl/ppl"}, env, **model, **recipe,
                      metric="ppl", n=ppl.get("n_tokens"), setting=setting)
    results = (metrics.get("lm_eval") or {}).get("results") or {}
    for task, values in sorted(results.items()):
        for metric in LMEVAL_METRICS:
            key = f"{metric},none"
            if key in values:
                pointer = _pointer("metrics", "lm_eval", "results", task, key)
                yield _record("lmeval", values[key], {**source, "pointer": pointer}, env, **model, **recipe,
                              task=task, metric=metric, stderr=values.get(f"{metric}_stderr,none"),
                              n=values.get("sample_len"))


def _from_report(data: dict[str, Any], source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    env, model = _env(data.get("env")), _model(data.get("env"))
    recipe = {"recipe_name": (data.get("recipe") or {}).get("name"), "recipe_hash": data.get("recipe_hash")}
    setting = {key: data.get(key) for key in ("samples", "seqlen", "seed", "dataset_fingerprint")}
    for index, layer in enumerate(data.get("layers") or []):
        for tensor in LAYER_TENSORS:
            values = layer.get(tensor)
            for metric in LAYER_METRICS if isinstance(values, dict) else ():
                if _is_number(values.get(metric)):
                    pointer = _pointer("layers", index, tensor, metric)
                    yield _record("layer_report", values[metric], {**source, "pointer": pointer}, env,
                                  **model, **recipe, **_layer_fields(layer["name"]), tensor=tensor,
                                  metric=metric, setting=setting)
    for metric in MODEL_METRICS:
        value = (data.get("model") or {}).get(metric)
        if _is_number(value):
            yield _record("model_report", value, {**source, "pointer": _pointer("model", metric)}, env,
                          **model, **recipe, metric=metric, setting=setting)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def records_of(path: Path, root: Path) -> tuple[list[dict[str, Any]], str | None]:
    """(records, None) for a recognised file, ([], reason) otherwise."""
    raw = path.read_bytes()
    source = {"path": _relative(path, root), "sha256": _sha256(raw)}
    text = raw.decode("utf-8")
    if path.name.endswith(".entries.jsonl"):
        lines = text.splitlines()
        if lines and "config" in json.loads(lines[0]):
            return list(_from_entries(lines, source)), None
        return [], "jsonl without an auto-tune config header"
    if path.suffix != ".json":
        return [], "not json"
    data = json.loads(text)
    if not isinstance(data, dict):
        return [], "not an object"
    schema = str(data.get("schema", ""))
    if schema.startswith("tricast.autotune.table/"):
        return list(_from_table(data, source)), None
    if schema.startswith("tricast.autotune.measure/"):
        return list(_from_measure(data, source)), None
    if schema.startswith("tricast.autotune.solution/"):
        return list(_from_solution(data, source)), None
    if {"run_hash", "metrics", "env"} <= set(data):
        return list(_from_run(data, source)), None
    if {"layers", "model", "recipe_hash", "env"} <= set(data):
        return list(_from_report(data, source)), None
    return [], "unrecognised shape"


def _duplicate(path: Path) -> bool:
    """A table's ``<stem>.entries.jsonl`` beside its finished ``<stem>.json`` holds the same measurements."""
    return path.name.endswith(".entries.jsonl") and path.with_name(
        path.name.removesuffix(".entries.jsonl") + ".json").is_file()


def _files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(item for item in path.rglob("*") if item.is_file() and not _duplicate(item)
                              and (item.suffix == ".json" or item.name.endswith(".entries.jsonl")))
        elif path.is_file():
            yield path
        else:
            raise LedgerError(f"{path}: no such file or directory")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        if record.get("schema") != SCHEMA:
            raise LedgerError(f"{path}: a line is not a {SCHEMA} record")
    return records


@contextlib.contextmanager
def _exclusive(ledger: Path) -> Iterator[None]:
    """Serialise ingests into one ledger (a tune loop and a person may ingest at the same time): the whole
    read-modify-write runs under an exclusive lock on ``<ledger>.lock``."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger.with_name(ledger.name + ".lock"), "w", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def ingest(paths: Sequence[Path], ledger: Path, root: Path) -> dict[str, Any]:
    """Add the records of ``paths`` to ``ledger`` under the re-ingest rules of the module docstring; returns
    counts. Nothing is written when a file is refused. Concurrent ingests are serialised (``_exclusive``)."""
    with _exclusive(ledger):
        return _ingest(paths, ledger, root)


def _ingest(paths: Sequence[Path], ledger: Path, root: Path) -> dict[str, Any]:
    known = {record["id"]: record for record in read_ledger(ledger)}
    added: dict[str, int] = defaultdict(int)
    skipped = refreshed = superseded = 0
    for path in _files(paths):
        records, reason = records_of(path, root)
        skipped += reason is not None
        source = _relative(path, root)
        gone = {key for key, record in known.items() if record["source"]["path"] == source} - \
            {record["id"] for record in records}
        if gone:
            raise LedgerError(f"{source}: {len(gone)} recorded values are no longer in the file (rewritten?)")
        for record in records:
            old = known.get(record["id"])
            if old is None:
                known[record["id"]] = record
                added[record["kind"]] += 1
            elif _content(old) != _content(record):
                changed = sorted(key for key in set(_content(old)) | set(_content(record))
                                 if _content(old).get(key) != _content(record).get(key))
                raise LedgerError(f"{source}{record['source']['pointer']}: changed in place ({changed})")
            elif old["source"]["sha256"] != record["source"]["sha256"]:
                known[record["id"]] = {**record, **({"superseded_by": old["superseded_by"]}
                                                    if "superseded_by" in old else {})}
                refreshed += 1
        if records and records[0]["kind"] == "layer_kl" and path.suffix == ".json":
            superseded += _supersede(known, source.removesuffix(".json") + ".entries.jsonl", records)
    ordered = sorted(known.values(), key=lambda record: record["id"])
    _write_atomic(ledger, "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                                  for record in ordered))
    return {"ledger": str(ledger), "records": len(ordered), "added": dict(sorted(added.items())),
            "refreshed": refreshed, "superseded": superseded, "skipped_files": skipped}


def _content(record: dict[str, Any]) -> dict[str, Any]:
    """A record without its file hash and supersession: what must not change under the same id."""
    return {**{key: value for key, value in record.items() if key not in ("source", "superseded_by")},
            "path": record["source"]["path"], "pointer": record["source"]["pointer"]}


def _supersede(known: dict[str, dict[str, Any]], log: str, records: Sequence[dict[str, Any]]) -> int:
    """Mark the records of a finished table's resume log ``superseded_by`` the table's record of the same
    (layer, method); refuse a log value the table does not hold."""
    table = {(record["layer"], record["method"]): record for record in records
             if record["kind"] == "layer_kl"}
    count = 0
    for key, record in list(known.items()):
        if record["source"]["path"] != log or "superseded_by" in record:
            continue
        target = table.get((record["layer"], record["method"]))
        if target is None or target["value"] != record["value"]:
            raise LedgerError(f"{log}{record['source']['pointer']}: not this value in the finished table")
        known[key] = {**record, "superseded_by": target["id"]}
        count += 1
    return count


def verify(record: dict[str, Any], root: Path) -> bool:
    """True when the record's value is the value at its source pointer, in a file with the same sha256."""
    source = record["source"]
    path = Path(source["path"])
    raw = (path if path.is_absolute() else root / path).read_bytes()
    if _sha256(raw) != source["sha256"]:
        return False
    pointer = source["pointer"]
    if pointer.startswith("#L"):
        number, _, key = pointer[2:].partition("/")
        value: Any = json.loads(raw.decode("utf-8").splitlines()[int(number) - 1])[key]
    else:
        value = json.loads(raw)
        for part in pointer.split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[int(part)] if isinstance(value, list) else value[part]
    return value == record["value"]


# ---------- retrieval documents ----------
def _g(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _where(record: dict[str, Any]) -> str:
    env = record["env"]
    gpu = ",".join(env.get("gpu_names") or []) or env.get("hostname") or "?"
    dirty = " dirty" if env.get("git_dirty") else ""
    return f"{gpu} git {str(env.get('git_sha'))[:7]}{dirty}"


def _setting(record: dict[str, Any]) -> str:
    setting = record.get("setting") or {}
    for count in ("windows", "samples"):
        if count in setting:
            return f"{setting.get(count)}x{setting.get('seqlen')} seed {setting.get('seed')}"
    return ""


def _layer_doc(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    head = name
    if "block" in first:  # "block 13" and the BM25 tokens block13, oproj
        head += (f" (block {first['block']} block{first['block']} {first['layer_kind'].replace('_', '')} "
                 f"{first['layer_kind']})")
    lines = [f"layer {head} · model {first.get('model_id')}@{str(first.get('model_sha'))[:7]}"]
    by_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[(record["kind"], record["source"]["path"])].append(record)
    for (kind, path), group in sorted(by_source.items()):
        if kind == "layer_kl":
            ordered = sorted(group, key=lambda item: (item["method"] != "floor", item["value"]))
            parts = [f"{record['method']} {_g(record['value'])} [{record['id']}]" for record in ordered]
            lines.append(f"single-layer KL, table {path} ({_setting(group[0])}, {_where(group[0])}): "
                         + "; ".join(parts))
        else:
            ordered = sorted(group, key=lambda item: (LAYER_TENSORS.index(item["tensor"]), item["metric"]))
            parts = [f"{record['tensor']} {record['metric']} {_g(record['value'])} [{record['id']}]"
                     for record in ordered]
            label = group[0].get("recipe_name") or str(group[0].get("recipe_hash"))[:8]
            lines.append(f"layer report, recipe {label} (hash {str(group[0].get('recipe_hash'))[:8]}, "
                         f"{_setting(group[0])}, {_where(group[0])}): " + "; ".join(parts))
    return {"source": f"ledger:layer/{first.get('model_sha')}", "section": f"layer {name}",
            "text": "\n".join(lines), "ids": sorted(record["id"] for record in records)}


def _recipe_line(record: dict[str, Any]) -> str:
    kind = record["kind"]
    if kind == "lmeval":
        what = (f"lm-eval {record['task']} {record['metric']} {_g(record['value'])} "
                f"± {_g(record.get('stderr'))} (n {record.get('n')})")
    elif kind == "recipe_kl":
        what = f"whole-recipe KL {_g(record['value'])} ({_setting(record)}, patched {record.get('patched')})"
    elif kind == "ppl":
        what = f"WikiText-2 test PPL {_g(record['value'])} (tokens {record.get('n')})"
    elif kind == "solution":
        what = (f"auto-tune solution: predicted KL sum {_g(record['value'])}, avg bits "
                f"{_g(record.get('avg_bits'))} for budget {_g(record.get('budget_bits'))}, "
                f"methods {json.dumps(record.get('method_counts'), sort_keys=True)}")
    else:
        what = f"{record['metric']} {_g(record['value'])}"
    extra = f", KV {record['kv']}" if record.get("kv") else ""
    extra += ", calibrated" if record.get("calibration") else ""
    return f"{what}{extra} — {_where(record)} [{record['id']}]"


def _recipe_doc(key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({record["recipe_name"] for record in records if record.get("recipe_name")})
    ordered = sorted(records, key=lambda item: (KINDS.index(item["kind"]), item.get("task") or "",
                                                 item["metric"], item["id"]))
    lines = [f"recipe {' / '.join(names) or '?'} · hash {key}"] + [_recipe_line(record) for record in ordered]
    return {"source": "ledger:recipe", "section": f"recipe {' / '.join(names) or key[:8]}",
            "text": "\n".join(lines), "ids": sorted(record["id"] for record in records)}


def _method_doc(path: str, method: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: (item["value"], item["layer"]))
    worst, best = ordered[::-1][:TOP], ordered[:TOP]
    mean = sum(record["value"] for record in ordered) / len(ordered)

    def names(group: list[dict[str, Any]]) -> str:
        return "; ".join(f"{record['layer']} {_g(record['value'])} [{record['id']}]" for record in group)

    lines = [f"method {method} · single-layer KL over {len(ordered)} layers of table {path} "
             f"({_setting(ordered[0])}, {_where(ordered[0])}): mean {_g(mean)}",
             f"most sensitive: {names(worst)}", f"least sensitive: {names(best)}"]
    return {"source": "ledger:method", "section": f"method {method} table {path}", "text": "\n".join(lines),
            "ids": sorted(record["id"] for record in worst + best)}


def documents(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrieval documents {source, section, text, ids}, ordered by (source, section)."""
    layers: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    recipes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    methods: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "superseded_by" in record:  # the same number is in the finished table's record
            continue
        if record["kind"] in ("layer_kl", "layer_report"):
            layers[(record.get("model_sha"), record["layer"])].append(record)
        if record["kind"] == "layer_kl" and record["method"] != "floor":
            methods[(record["source"]["path"], record["method"])].append(record)
        if record["kind"] in ("recipe_kl", "ppl", "lmeval", "model_report", "solution"):
            recipes[record.get("recipe_hash") or f"name:{record.get('recipe_name')}"].append(record)
    docs = [_layer_doc(name, group) for (_, name), group in layers.items()]
    docs += [_recipe_doc(key, group) for key, group in recipes.items()]
    docs += [_method_doc(path, method, group) for (path, method), group in methods.items()]
    return sorted(docs, key=lambda doc: (doc["source"], doc["section"]))


def write_documents(records: Sequence[dict[str, Any]], path: Path) -> int:
    docs = documents(records)
    _write_atomic(path, "".join(json.dumps({"schema": DOCS_SCHEMA, **doc}, sort_keys=True, allow_nan=False)
                                + "\n" for doc in docs))
    return len(docs)


def docs_path(ledger: Path) -> Path:
    """``runs/ledger.jsonl`` -> ``runs/ledger.docs.jsonl`` (what config/rag.yaml's ``ledgers`` names)."""
    return ledger.with_name(ledger.name.removesuffix(".jsonl") + ".docs.jsonl")


# ---------- CLI ----------
def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ledger", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="add the records of run files or directories")
    ingest_parser.add_argument("paths", nargs="+", type=Path)
    get_parser = sub.add_parser("get", help="print records by id")
    get_parser.add_argument("ids", nargs="+")
    for item in (ingest_parser, sub.add_parser("stats"), sub.add_parser("verify"), get_parser):
        item.add_argument("--ledger", type=Path, default=Path("runs/ledger.jsonl"))
        item.add_argument("--root", type=Path, default=_repo_root(), help="base of relative source paths")
    args = parser.parse_args(argv)
    try:
        records = read_ledger(args.ledger)
        if args.command == "ingest":
            summary = ingest(args.paths, args.ledger, args.root)
            summary["documents"] = write_documents(read_ledger(args.ledger), docs_path(args.ledger))
            print(json.dumps(summary, indent=1))
        elif args.command == "stats":
            counts: dict[str, int] = defaultdict(int)
            for record in records:
                counts[record["kind"]] += 1
            print(json.dumps({"records": len(records), "kinds": dict(sorted(counts.items()))}, indent=1))
        elif args.command == "verify":
            bad = [record["id"] for record in records if not verify(record, args.root)]
            print(json.dumps({"records": len(records), "mismatches": len(bad), "first": bad[:20]}, indent=1))
            return 1 if bad else 0
        else:
            index = {record["id"]: record for record in records}
            missing = [item for item in args.ids if item not in index]
            if missing:
                raise LedgerError(f"no record {missing}")
            print(json.dumps([index[item] for item in args.ids], indent=1, ensure_ascii=False))
    except (LedgerError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
