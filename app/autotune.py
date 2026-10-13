"""Auto-tune v1 for the Studio Quantizer: a per-layer KL table and a multiple-choice knapsack over it
(docs/spikes/2026-10-13_autotune_mckp.md §4).

    python -m app.autotune table   --model Qwen/Qwen3-0.6B --revision <sha> --device cuda --out table.json
    python -m app.autotune solve   --table table.json --budget-bits 4.5 [--kv kivi2] --out <dir>
    python -m app.autotune predict --table table.json <grid.json | preset id> ...
    python -m app.autotune measure --model Qwen/Qwen3-0.6B --device cuda --seed 43 --out kl.json <recipe> ...
    python -m app.autotune random  --seeds 1,2,3,4 --out <dir>       # seeded random grids for the C2 check

table: native logits once on tricast's calibration windows (WikiText-2 train, joined with blank lines,
8 × 512 tokens drawn with the seed — tricast.calibration), then for every decoder Linear and every
quantization method, only that layer with that method: the method's recipe from
``app.quantizer_recipe.build`` with tricast's ``Recipe.include`` narrowed to the layer, so the spec is the
one any Quantizer grid gives that layer. The value is KL(native || quantized) of the next-token
distributions (OBJECTIVE). BF16 is 0 by definition and not run. Each entry is appended to
``<out stem>.entries.jsonl`` when it finishes; a rerun with the same configuration (code hashes included)
skips the entries already there, and ``<out stem>.lock`` keeps a second run off the same --out.

solve: minimise sum kl(layer, method) subject to sum params x bits(layer, method) <= budget, bits from
``app.export_quantizer_demo.WEIGHT_BITS`` (the deck's Linear-weighted average bits). Exact DP on costs rounded
up to multiples of the resolution (default: the exact GCD of all costs, so nothing is rounded). Writes the
Quantizer's --from-json grid (grid.json), the YAML that ``--from-json grid.json`` prints (recipe.yaml) and
solution.json. The grid default is the most frequent non-BF16 choice (``default_method``); the recipe
name defaults to ``default_name`` (exact budget and KV preset).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import socket
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr

TABLE_SCHEMA = "tricast.autotune.table/1"
MEASURE_SCHEMA = "tricast.autotune.measure/1"
SOLUTION_SCHEMA = "tricast.autotune.solution/1"
FREE = tuple(m for m in qr.METHODS if m not in qr.CALIBRATED)  # bf16 nvfp4 nvint4 mxfp4 mxfp8 int8 bfp12
OBJECTIVE = ("KL(native || quantized) of the next-token distributions, mean over every position of every "
             "window, fp64, negative roundoff clamped to 0 (tricast report's logits_kl)")
MAX_DP_CELLS = 1 << 28  # layers x budget units of the DP's backtracking table (one byte each)
# The table configuration a resumed run must share with the entries already written (code included: the
# app modules that define methods, grids and bits, and the tricast package actually imported).
RESUME_KEYS = ("model_id", "revision", "model_sha", "dtype", "device", "gpu_names", "windows", "seqlen",
               "seed", "source", "dataset_fingerprint", "git_sha", "git_dirty", "src_sha256", "torch",
               "transformers", "triton", "app_sha256", "tricast_path")

# (model_id, dtype, device, revision) -> (model, tokenizer), as app/runners/quantizer.py; tests inject one.
ModelFactory = Callable[[str, str, str, str | None], tuple[Any, Any]]


class AutotuneError(ValueError):
    """A refused input; the CLI prints it and exits with status 2."""


def provenance() -> dict[str, Any]:
    """sha256 of this module and the app modules it takes methods, grids and bits from, and the directory of
    the tricast package actually imported."""
    import tricast

    files = {"app/autotune.py": Path(__file__), "app/quantizer_recipe.py": Path(qr.__file__),
             "app/export_quantizer_demo.py": Path(ex.__file__)}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    return {"app_sha256": hashes, "tricast_path": str(Path(tricast.__file__).resolve().parent)}


def run_env() -> dict[str, Any]:
    """This run's own environment (tricast's capture_env without a model: git sha / dirty, host, utc,
    versions) and provenance."""
    from tricast.eval.envinfo import capture_env

    return {**capture_env(None), **provenance()}


def default_name(bits: Fraction, kv: str) -> str:
    """auto_<numerator>[-<denominator>][_<kv>]: the exact budget in bits (9/2 -> auto_9-2) and the KV preset,
    so different budgets or KV presets never share a name."""
    budget = str(bits.numerator) if bits.denominator == 1 else f"{bits.numerator}-{bits.denominator}"
    return f"auto_{budget}" + ("" if kv == "none" else f"_{kv}")


@dataclass(frozen=True)
class Layer:
    name: str
    block: int
    kind: str
    params: int
    in_features: int


# ---------- sizes ----------
def weight_bits(method: str, in_features: int) -> Fraction:
    """Stored bits per weight: element + scale bits / elements per scale (WEIGHT_BITS; None = one per row)."""
    element, scale, per = ex.WEIGHT_BITS[method]
    return element + Fraction(scale, per or in_features)


def cost(layer: Layer, method: str) -> Fraction:
    return layer.params * weight_bits(method, layer.in_features)


def avg_bits(layers: Sequence[Layer], methods: Sequence[str]) -> Fraction:
    """Linear-weighted average stored bits of ``layers`` using ``methods`` (one per layer)."""
    total = sum((cost(layer, method) for layer, method in zip(layers, methods, strict=True)), Fraction(0))
    return total / sum(layer.params for layer in layers)


# ---------- multiple-choice knapsack ----------
@dataclass(frozen=True)
class Solution:
    choice: list[int]  # option index per row
    value: float  # sum of the chosen values, added row by row
    cost: Fraction  # sum of the chosen exact costs
    resolution: Fraction
    capacity: int  # floor(budget / resolution) units
    bound: Fraction  # sum over rows of the largest rounding: the DP is optimal for (budget - bound)


def _gcd(values: Sequence[Fraction]) -> Fraction:
    common = math.lcm(*(value.denominator for value in values))
    return Fraction(math.gcd(*(int(value * common) for value in values)), common)


def solve_mckp(costs: Sequence[Sequence[Fraction | int | float]], values: Sequence[Sequence[float]],
               budget: Fraction | int | float, resolution: Fraction | int | float | None = None) -> Solution:
    """One option per row, minimising the sum of values with the sum of costs <= ``budget``.

    Costs are rounded up to multiples of ``resolution`` (default: the exact GCD of all costs, so nothing
    rounds) and an exact integer DP runs on them. The choice therefore never exceeds ``budget``, and its
    value is at most the true optimum for ``budget - bound``, where ``bound`` = sum over rows of the largest
    rounding (0 when every cost is a multiple of the resolution). Ties: the smallest rounded total cost,
    then, deciding rows from the last one, the earliest option of the row."""
    import numpy as np

    rows = [[Fraction(c) for c in row] for row in costs]
    if not rows or len(rows) != len(values) or any(len(row) != len(v) or not row for row, v in
                                                   zip(rows, values, strict=True)):
        raise AutotuneError("costs and values need the same non-empty rows")
    if any(len(row) > 127 for row in rows) or any(c < 0 for row in rows for c in row):
        raise AutotuneError("each row needs at most 127 options with non-negative costs")
    budget = Fraction(budget)
    cheapest = sum((min(row) for row in rows), Fraction(0))
    if cheapest > budget:
        raise AutotuneError(f"infeasible budget {float(budget):.10g}: the cheapest choice costs "
                            f"{float(cheapest):.10g}")
    if resolution is None:
        nonzero = [c for row in rows for c in row if c]
        step = _gcd(nonzero) if nonzero else Fraction(1)
    else:
        step = Fraction(resolution)
    if step <= 0:
        raise AutotuneError(f"resolution must be positive, got {resolution}")
    units = [[math.ceil(c / step) for c in row] for row in rows]
    capacity = math.floor(budget / step)
    if sum(min(row) for row in units) > capacity:
        raise AutotuneError(f"the budget fits only before rounding costs up to multiples of {float(step):g}; "
                            "use a finer resolution")
    capacity = min(capacity, sum(max(row) for row in units))  # budget beyond the dearest choice adds nothing
    if len(rows) * (capacity + 1) > MAX_DP_CELLS:
        raise AutotuneError(f"{len(rows)} rows x {capacity + 1} budget units is too large; use a coarser "
                            "resolution")
    bound = sum((max(u * step - c for u, c in zip(urow, row, strict=True)) for urow, row in
                 zip(units, rows, strict=True)), Fraction(0))
    dp = np.full(capacity + 1, np.inf)
    dp[0] = 0.0
    picks = np.full((len(rows), capacity + 1), -1, dtype=np.int8)
    for i, (urow, vrow) in enumerate(zip(units, values, strict=True)):
        new = np.full(capacity + 1, np.inf)
        for j, (u, v) in enumerate(zip(urow, vrow, strict=True)):
            if u > capacity:
                continue
            candidate = dp[:capacity + 1 - u] + float(v)
            better = candidate < new[u:]
            new[u:][better] = candidate[better]
            picks[i, u:][better] = j
        dp = new
    left = int(np.argmin(dp))  # the first minimum: the smallest rounded total cost
    if not math.isfinite(dp[left]):
        raise AutotuneError("no choice fits the budget")
    choice = [0] * len(rows)
    for i in range(len(rows) - 1, -1, -1):
        choice[i] = int(picks[i, left])
        left -= units[i][choice[i]]
    value = 0.0
    for i, j in enumerate(choice):
        value += float(values[i][j])
    total = sum((rows[i][j] for i, j in enumerate(choice)), Fraction(0))
    return Solution(choice, value, total, step, capacity, bound)


# ---------- grids ----------
def default_method(used: Sequence[str], allowed: Sequence[str]) -> str:
    """The grid default: the most frequent non-BF16 method in ``used``, ties to the earlier method in
    app.quantizer_recipe.METHODS; with none, the first non-BF16 method of ``allowed`` (build needs one)."""
    counts = Counter(method for method in used if method != "bf16")
    if counts:
        return min(counts, key=lambda method: (-counts[method], list(qr.METHODS).index(method)))
    return next(method for method in qr.METHODS if method in allowed and method != "bf16")


def write_grid(out_dir: Path, name: str, default: str, grid: qr.Grid, kv: str) -> dict[str, Any]:
    """grid.json (the Quantizer's --from-json spec) and recipe.yaml (what ``--from-json grid.json``
    prints)."""
    recipe = qr.build(name, default, grid, kv)
    _dump(out_dir / "grid.json", {"name": name, "default": default, "grid": grid, "kv": kv})
    _write_atomic(out_dir / "recipe.yaml", f"---\n{qr.to_yaml(recipe)}\n")
    return recipe


def parse_methods(text: str) -> list[str]:
    """Comma-separated method ids -> the calibration-free methods, in app.quantizer_recipe.METHODS order."""
    given = {method.strip() for method in text.split(",") if method.strip()}
    unknown = sorted(given - set(qr.METHODS))
    if unknown:
        raise AutotuneError(f"unknown methods {unknown}; known: {list(qr.METHODS)}")
    calibrated = sorted(given & qr.CALIBRATED)
    if calibrated:
        raise AutotuneError(f"{calibrated} need calibration data; auto-tune v1 takes only the "
                            f"calibration-free methods {list(FREE)}")
    methods = [method for method in qr.METHODS if method in given]
    if all(method == "bf16" for method in methods):
        raise AutotuneError("give at least one quantization method besides bf16")
    return methods


# ---------- tables ----------
def load_table(path: Path) -> tuple[dict[str, Any], bytes, list[Layer], list[str]]:
    raw = path.read_bytes()
    table = json.loads(raw)
    if table.get("schema") != TABLE_SCHEMA or table.get("status") != "complete":
        raise AutotuneError(f"{path}: not a complete {TABLE_SCHEMA} table")
    layers = [Layer(entry["name"], entry["block"], entry["kind"], entry["params"], entry["in_features"])
              for entry in table["layers"]]
    return table, raw, layers, list(table["meta"]["methods"])


def predict(table: dict[str, Any], layers: Sequence[Layer], methods: Sequence[str],
            grid: qr.Grid) -> tuple[float, Fraction]:
    """(sum of the table's kl over its layers for the methods ``grid`` gives them, their average bits)."""
    if len(grid) != qr.BLOCKS or any(len(row) != len(qr.KINDS) for row in grid):
        raise AutotuneError(f"grid must be {qr.BLOCKS} blocks x {len(qr.KINDS)} kinds")
    missing = sorted({method for row in grid for method in row} - set(methods))
    if missing:
        raise AutotuneError(f"the grid uses {missing}, which the table has no KL for "
                            f"(table: {list(methods)})")
    kl = {entry["name"]: entry["kl"] for entry in table["layers"]}
    chosen = [grid[layer.block][qr.KINDS.index(layer.kind)] for layer in layers]
    total = 0.0
    for layer, method in zip(layers, chosen, strict=True):
        total += kl[layer.name][method]
    return total, avg_bits(layers, chosen)


# ---------- model side (torch and tricast imported here only) ----------
@dataclass
class Session:
    model: Any
    windows: list[Any]  # [1, seqlen] token tensors
    native: list[Any]  # their native logits, on the model's device
    meta: dict[str, Any]


def _load(model_id: str, dtype: str, device: str, revision: str | None) -> tuple[Any, Any]:
    from tricast.eval.runner import _load_model  # what tricast run / tricast report load with

    return _load_model(model_id, dtype, device, revision=revision)


def _seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def logits(model: Any, ids: Any) -> Any:
    import torch

    from tricast.eval.ppl import _model_logits

    with torch.inference_mode():
        out = _model_logits(model, ids.to(next(model.parameters()).device))
    if not torch.isfinite(out).all():
        raise RuntimeError("the model produced non-finite logits")
    return out


def mean_kl(model: Any, windows: Sequence[Any], native: Sequence[Any]) -> float:
    """OBJECTIVE for ``model`` against the native logits of the same windows."""
    import torch

    total, positions = 0.0, 0
    with torch.inference_mode():
        for ids, reference in zip(windows, native, strict=True):
            log_p = reference.double().log_softmax(-1)
            log_q = logits(model, ids).double().log_softmax(-1)
            total += (log_p.exp() * (log_p - log_q)).sum().item()
            positions += ids.numel()
    return max(0.0, total / positions)


def open_session(args: argparse.Namespace, model_factory: ModelFactory | None = None) -> Session:
    """Model, windows and native logits, with the environment the results are reported with (AGENTS.md
    §3.3)."""
    import torch

    from tricast.calibration import _dataset_texts, _windows
    from tricast.eval.envinfo import capture_env

    if args.device == "cuda" and not torch.cuda.is_available():
        raise AutotuneError("--device cuda: no CUDA device is visible")
    dtype = "bfloat16" if args.device == "cuda" else "float32"  # app/runners/quantizer.py's choice
    _seed(args.seed)
    env = capture_env(args.model, {"seed": args.seed, "dtype": dtype, "device": args.device})
    revision = args.revision or (env.get("model_sha") if env.get("model_kind") == "hf" else None)
    model, tokenizer = (model_factory or _load)(args.model, dtype, args.device, revision)
    actual = getattr(model.config, "_commit_hash", None)
    if actual and env.get("model_kind") != "local":
        env["model_sha"] = actual
    if args.text:
        text = Path(args.text).read_text(encoding="utf-8")
        texts: Any = [text]
        source = {"text": str(Path(args.text).resolve()), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    else:
        texts, source = _dataset_texts("wikitext2", "train"), {"dataset": "wikitext2", "split": "train"}
    windows = _windows(tokenizer, texts, None, args.windows, args.seqlen, args.seed)
    digest = hashlib.sha256()  # tricast.calibrate's dataset_fingerprint of the same windows
    for ids in windows:
        digest.update(ids.cpu().contiguous().numpy().astype("<i8").tobytes())
    started = time.perf_counter()
    native = [logits(model, ids) for ids in windows]
    meta = {"model_id": args.model, "revision": args.revision, "model_sha": env.get("model_sha"),
            "model_kind": env.get("model_kind"), "dtype": dtype, "device": args.device,
            "gpu_names": env.get("gpu_names"), "windows": args.windows, "seqlen": args.seqlen,
            "seed": args.seed, "source": source, "dataset_fingerprint": digest.hexdigest(),
            "git_sha": env.get("git_sha"), "git_dirty": env.get("git_dirty"),
            "src_sha256": env.get("src_sha256"),
            **{name: env["versions"].get(name) for name in ("torch", "transformers", "triton")},
            **provenance(), "native_wall_s": time.perf_counter() - started, "env": env}
    return Session(model, windows, native, meta)


def model_layers(model: Any, blocks: int | None = None) -> tuple[list[Layer], int]:
    """The Quantizer's Linear layers (block x kind) of the first ``blocks`` decoder blocks, and the model's
    block count."""
    from torch import nn

    from tricast.nn.patch import decoder_layers

    decoder = decoder_layers(model)
    if decoder is None or decoder[0] != "model.layers":
        raise AutotuneError("auto-tune v1 needs the Quantizer's layer names: "
                            "model.layers.<b>.<self_attn|mlp>.<kind>")
    if decoder[1] > qr.BLOCKS:
        raise AutotuneError(f"the Quantizer grid has {qr.BLOCKS} blocks; this model has {decoder[1]}")
    layers = []
    for block in range(decoder[1] if blocks is None else min(blocks, decoder[1])):
        for kind in qr.KINDS:
            name = qr.layer_name(block, kind)
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise AutotuneError(f"{name} is not a Linear")
            layers.append(Layer(name, block, kind, module.weight.numel(), module.in_features))
    return layers, decoder[1]


def method_recipe(method: str) -> Any:
    """The tricast Recipe of the Quantizer grid with every layer on ``method``."""
    from tricast.recipe import load_recipe

    return load_recipe(qr.build(f"autotune_{method}", method, qr.uniform(method)))


def _entry_kl(session: Session, recipe: Any, layer: Layer, seed: int) -> float:
    """OBJECTIVE with only ``layer`` patched with ``recipe``'s spec."""
    import torch

    from tricast.nn.patch import patch_model, unpatch_model

    torch.manual_seed(seed)  # every entry starts from the same RNG state, whatever ran before it
    report = patch_model(session.model, replace(recipe, include=[layer.name]))
    try:
        patched = [name for name, _ in report.patched]
        if patched != [layer.name]:
            raise RuntimeError(f"the one-layer recipe for {layer.name} patched {patched}")
        return mean_kl(session.model, session.windows, session.native)
    finally:
        unpatch_model(session.model)


def _resume(path: Path, config: dict[str, Any],
            shas: dict[str, str]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    """Entries already in ``path`` ({(layer, method): entry}) and the runs that wrote them; starts the file
    (header = configuration) if it is missing, refuses a different configuration or method recipe, and
    drops a last line cut off by a kill."""
    if not path.exists():
        _write_atomic(path, json.dumps({"config": config}) + "\n")
        return {}, []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header = json.loads(lines[0])["config"]
    if header != config:
        changed = sorted(key for key in set(header) | set(config) if header.get(key) != config.get(key))
        raise AutotuneError(f"{path} holds entries of another configuration (differs in {changed}); "
                            "use another --out")
    complete = [line for line in lines if line.endswith("\n")]
    if len(complete) != len(lines):
        _write_atomic(path, "".join(complete))
    done, runs = {}, []
    for line in complete[1:]:
        entry = json.loads(line)
        if "run" in entry:
            runs.append(entry["run"])
            continue
        if entry["method"] in shas and entry["recipe_sha256"] != shas[entry["method"]]:
            raise AutotuneError(f"{path}: the {entry['method']} recipe changed since its entries were "
                                "written; use another --out")
        done[(entry["layer"], entry["method"])] = entry
    return done, runs


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, TypeError, ValueError, OverflowError):
        return True  # someone else's process, or a lock we cannot read: treat it as held
    return True


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold ``path`` (created exclusively) for one table run. A lock left by a dead process of this host is
    taken over; any other existing lock refuses the run."""
    me = {"hostname": socket.gethostname(), "pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                     time.gmtime())}
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                holder = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                holder = None
            stale = (isinstance(holder, dict) and holder.get("hostname") == me["hostname"]
                     and not _alive(holder.get("pid")))
            if attempt or not stale:
                raise AutotuneError(f"{path} is held by {holder}: another table run is writing this --out "
                                    "(delete the lock if that run is gone)") from None
            path.unlink(missing_ok=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(me, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def cmd_table(args: argparse.Namespace, model_factory: ModelFactory | None = None) -> int:
    methods = parse_methods(args.methods)
    out = Path(args.out)
    with _locked(out.with_name(out.stem + ".lock")):
        return _table(args, model_factory, methods, out)


def _table(args: argparse.Namespace, model_factory: ModelFactory | None, methods: list[str],
           out: Path) -> int:
    started = time.perf_counter()
    session = open_session(args, model_factory)
    layers, model_blocks = model_layers(session.model, args.blocks)
    recipes = {method: method_recipe(method) for method in methods if method != "bf16"}
    shas = {method: recipe.sha256 for method, recipe in recipes.items()}
    entries_path = out.with_name(out.stem + ".entries.jsonl")
    done, runs = _resume(entries_path, {key: session.meta[key] for key in RESUME_KEYS}, shas)
    env = session.meta["env"]
    runs.append({"utc": env.get("utc"), "hostname": env.get("hostname"), "device": args.device,
                 "gpu_names": env.get("gpu_names"),
                 "cuda_visible_devices": env.get("environment", {}).get("CUDA_VISIBLE_DEVICES"),
                 "pid": os.getpid()})
    todo = [(layer, method) for layer in layers for method in recipes]
    print(f"AUTOTUNE_TABLE_START layers={len(layers)} methods={len(methods)} entries={len(todo)} "
          f"done={sum(1 for layer, method in todo if (layer.name, method) in done)} out={out}", flush=True)
    computed = 0
    with entries_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"run": runs[-1]}) + "\n")
        log.flush()
        for index, layer in enumerate(layers):
            layer_started = time.perf_counter()
            for method in recipes:
                if (layer.name, method) in done:
                    continue
                entry_started = time.perf_counter()
                kl = _entry_kl(session, recipes[method], layer, args.seed)
                entry = {"layer": layer.name, "method": method, "kl": kl, "recipe_sha256": shas[method],
                         "wall_s": time.perf_counter() - entry_started}
                log.write(json.dumps(entry, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                done[(layer.name, method)] = entry
                computed += 1
            values = " ".join(f"{method}={done[(layer.name, method)]['kl']:.4g}" for method in recipes)
            seconds = time.perf_counter() - layer_started
            print(f"[{index + 1}/{len(layers)}] {layer.name} {values} ({seconds:.1f}s)", flush=True)
    wall = time.perf_counter() - started
    rows = [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
             "in_features": layer.in_features,
             "kl": {method: 0.0 if method == "bf16" else done[(layer.name, method)]["kl"]
                    for method in methods}}
            for layer in layers]
    meta = {**session.meta, "methods": methods, "blocks": len(layers) // len(qr.KINDS),
            "model_blocks": model_blocks, "method_recipe_sha256": shas, "entries": len(todo),
            "entries_computed_this_run": computed,
            "entries_wall_s": sum(done[(layer.name, method)]["wall_s"] for layer, method in todo),
            "entries_path": str(entries_path), "runs": runs, "wall_s": wall}
    _dump(out, {"schema": TABLE_SCHEMA, "status": "complete", "objective": OBJECTIVE, "meta": meta,
                "layers": rows})
    print(f"AUTOTUNE_TABLE_DONE layers={len(layers)} methods={len(methods)} wall_s={wall:.1f}", flush=True)
    return 0


def cmd_measure(args: argparse.Namespace, model_factory: ModelFactory | None = None) -> int:
    import torch

    from tricast.nn.patch import patch_model, unpatch_model
    from tricast.recipe import load_recipe

    started = time.perf_counter()
    recipes = [(Path(path), load_recipe(Path(path))) for path in args.recipes]
    for path, recipe in recipes:
        if recipe.needs_calibration:
            raise AutotuneError(f"{path}: needs calibration; auto-tune v1 measures calibration-free "
                                "recipes only")
        if recipe.kv is not None and recipe.kv.mode == "cache":
            raise AutotuneError(f"{path}: kv mode 'cache' streams token by token; use mode fakequant")
    session = open_session(args, model_factory)
    results = []
    for path, recipe in recipes:
        recipe_started = time.perf_counter()
        torch.manual_seed(args.seed)
        report = patch_model(session.model, recipe)
        try:
            if not report.patched and not report.kv:
                raise RuntimeError(f"{path}: the recipe patched nothing")
            kl = mean_kl(session.model, session.windows, session.native)
        finally:
            unpatch_model(session.model)
        results.append({"recipe": str(path), "name": recipe.name, "recipe_sha256": recipe.sha256,
                        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "kl": kl,
                        "patched": len(report.patched), "kv": report.kv,
                        "wall_s": time.perf_counter() - recipe_started})
        print(f"MEASURE {recipe.name} kl={kl:.6g} patched={len(report.patched)} kv={len(report.kv)} "
              f"({results[-1]['wall_s']:.1f}s)", flush=True)
    meta = {**session.meta, "wall_s": time.perf_counter() - started}
    _dump(Path(args.out), {"schema": MEASURE_SCHEMA, "status": "complete", "objective": OBJECTIVE,
                           "meta": meta, "results": results})
    print(f"AUTOTUNE_MEASURE_DONE n={len(results)}", flush=True)
    return 0


# ---------- commands without a model ----------
def cmd_solve(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    table_path = Path(args.table)
    table, raw, layers, methods = load_table(table_path)
    params = sum(layer.params for layer in layers)
    bits = args.budget_bits if args.budget_bits is not None else args.budget_fraction * 16
    if bits <= 0:
        raise AutotuneError("the budget must be positive")
    budget = bits * params
    cheapest = sum((min(cost(layer, method) for method in methods) for layer in layers), Fraction(0))
    if cheapest > budget:
        raise AutotuneError(f"infeasible budget: {float(bits):.6g} bits is below the cheapest assignment of "
                            f"these methods, {float(cheapest / params):.6g} bits")
    kl = {entry["name"]: entry["kl"] for entry in table["layers"]}
    solution = solve_mckp([[cost(layer, method) for method in methods] for layer in layers],
                          [[kl[layer.name][method] for method in methods] for layer in layers], budget,
                          args.resolution)
    chosen = [methods[j] for j in solution.choice]
    default = default_method(chosen, methods)
    grid = qr.uniform(default)
    for layer, method in zip(layers, chosen, strict=True):
        grid[layer.block][qr.KINDS.index(layer.kind)] = method
    name = args.name or default_name(bits, args.kv)
    out = Path(args.out)
    write_grid(out, name, default, grid, args.kv)
    achieved = solution.cost / params
    tabled = sorted({layer.block for layer in layers})
    counts = Counter(chosen)
    meta = table["meta"]
    result = {
        "schema": SOLUTION_SCHEMA, "name": name, "default": default, "kv": args.kv,
        "kv_note": "the table has no KV term: predicted_kl_sum covers the Linear layers only",
        "budget": {"avg_bits": float(bits), "bits_params": float(budget),
                   "given": ({"budget_bits": str(args.budget_bits)} if args.budget_bits is not None
                             else {"budget_fraction": str(args.budget_fraction)})},
        "achieved": {"avg_bits": float(achieved), "bits_params": float(solution.cost),
                     "size_pct_of_bf16": float(achieved / 16 * 100)},
        "predicted_kl_sum": solution.value,
        "resolution": {"bits_params": float(solution.resolution), "capacity_units": solution.capacity,
                       "bound_bits_params": float(solution.bound),
                       "bound_avg_bits": float(solution.bound / params),
                       "meaning": "the choice never exceeds the budget, and its kl sum is at most the "
                                  "optimum for a budget smaller by the bound (0 = exact)"},
        "method_counts": {method: counts[method] for method in methods if counts[method]},
        "tie_rules": {"solver": "smallest rounded total cost, then the earliest method (METHODS order) of "
                                "each layer, deciding layers from the last",
                      "default": "most frequent non-BF16 choice, ties to the earlier method in METHODS"},
        "table": {"path": str(table_path), "sha256": hashlib.sha256(raw).hexdigest(), "methods": methods,
                  **{key: meta.get(key) for key in ("model_id", "model_sha", "dataset_fingerprint", "windows",
                                                    "seqlen", "seed", "git_sha", "git_dirty")}},
        "table_blocks": tabled, "model_blocks": meta.get("model_blocks"),
        "untabled_blocks_in_grid": [block for block in range(qr.BLOCKS) if block not in tabled],
        "layers": [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
                    "method": method, "kl": kl[layer.name][method],
                    "bits": float(weight_bits(method, layer.in_features))}
                   for layer, method in zip(layers, chosen, strict=True)],
        "solver_wall_s": time.perf_counter() - started,
        "env": run_env(),
    }
    _dump(out / "solution.json", result)
    if meta.get("model_blocks") is not None and meta["model_blocks"] > len(tabled):
        print(f"WARNING: the table covers {len(tabled)} of {meta['model_blocks']} blocks; the others get the "
              f"default {default} without being counted", flush=True)
    counted = " ".join(f"{method}={count}" for method, count in result["method_counts"].items())
    print(f"SOLVE {name}: {counted} default={default}")
    print(f"AUTOTUNE_SOLVE_DONE name={name} avg_bits={float(achieved):.6f} budget_bits={float(bits):.6f} "
          f"kl_sum={solution.value:.6g} out={out}", flush=True)
    return 0


def _grid_specs(item: str) -> list[dict[str, Any]]:
    if item in qr.PRESETS:
        default, grid, kv, _ = qr.PRESETS[item]
        return [{"name": item, "default": default, "grid": grid, "kv": kv}]
    specs = json.loads(Path(item).read_text(encoding="utf-8"))
    return specs if isinstance(specs, list) else [specs]


def cmd_predict(args: argparse.Namespace) -> int:
    table, raw, layers, methods = load_table(Path(args.table))
    results = []
    for item in args.grids:
        for spec in _grid_specs(item):
            carried = [key for key in ("options", "kv_options", "calibration") if spec.get(key)]
            if carried:
                raise AutotuneError(f"{item}: {spec['name']} sets {carried}; the v1 table has entries only "
                                    "for the methods without options")
            total, bits = predict(table, layers, methods, spec["grid"])
            kv = spec.get("kv", "none")
            results.append({"source": item, "name": spec["name"], "kl_sum": total, "avg_bits": float(bits),
                            "kv": kv})
            note = "" if kv == "none" else " (KV not in the table)"
            print(f"PREDICT {spec['name']} kl_sum={total:.6g} avg_bits={float(bits):.6f} kv={kv}{note}")
    if args.out:
        _dump(Path(args.out), {"table": args.table, "table_sha256": hashlib.sha256(raw).hexdigest(),
                               "env": run_env(), "results": results})
    print(f"AUTOTUNE_PREDICT_DONE n={len(results)}", flush=True)
    return 0


def cmd_random(args: argparse.Namespace) -> int:
    """One grid per seed: every layer an independent ``random.Random(seed).choice`` of the methods."""
    methods = parse_methods(args.methods)
    for seed in args.seeds:
        rng = random.Random(seed)
        grid = [[rng.choice(methods) for _ in qr.KINDS] for _ in range(qr.BLOCKS)]
        name = f"rand_s{seed}"
        default = default_method([method for row in grid for method in row], methods)
        write_grid(Path(args.out) / name, name, default, grid, args.kv)
        print(f"RANDOM {name} default={default} out={Path(args.out) / name}")
    print(f"AUTOTUNE_RANDOM_DONE n={len(args.seeds)}", flush=True)
    return 0


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file of a unique name next to ``path``, then rename it over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _dump(path: Path, value: Any) -> None:
    _write_atomic(path, json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def _seeds(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None, model_factory: ModelFactory | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.autotune", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    table = commands.add_parser("table", help="per-layer x per-method KL table")
    measure = commands.add_parser("measure", help="whole-recipe KL on (held-out) windows")
    for command, seed in ((table, 42), (measure, 43)):
        command.add_argument("--model", default=ex.MODEL_ID)
        command.add_argument("--revision", help="model revision (a full commit sha when the Hub is offline)")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--windows", type=int, default=8)
        command.add_argument("--seqlen", type=int, default=512)
        command.add_argument("--seed", type=int, default=seed)
        command.add_argument("--text", metavar="FILE", help="UTF-8 text in place of WikiText-2 train")
        command.add_argument("--out", required=True)
    table.add_argument("--methods", default=",".join(FREE))
    table.add_argument("--blocks", type=int, help="only the first N decoder blocks")
    measure.add_argument("recipes", nargs="+", metavar="RECIPE")
    solve = commands.add_parser("solve", help="multiple-choice knapsack over a table")
    solve.add_argument("--table", required=True)
    budget = solve.add_mutually_exclusive_group(required=True)
    budget.add_argument("--budget-bits", type=Fraction, help="Linear-weighted average bits")
    budget.add_argument("--budget-fraction", type=Fraction, help="fraction of the BF16 Linear size")
    solve.add_argument("--resolution", type=Fraction, help="cost unit in bits x params (default: exact GCD)")
    solve.add_argument("--kv", choices=sorted(qr.KV), default="none")
    solve.add_argument("--name")
    solve.add_argument("--out", required=True)
    pred = commands.add_parser("predict", help="sum of the table's kl for grids or Quantizer presets")
    pred.add_argument("--table", required=True)
    pred.add_argument("grids", nargs="+", metavar="GRID",
                      help="grid.json (--from-json format) or a preset id")
    pred.add_argument("--out")
    rand = commands.add_parser("random", help="seeded random grids (grid.json + recipe.yaml each)")
    rand.add_argument("--seeds", type=_seeds, required=True)
    rand.add_argument("--methods", default=",".join(FREE))
    rand.add_argument("--kv", choices=sorted(qr.KV), default="none")
    rand.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "table":
            return cmd_table(args, model_factory)
        if args.command == "measure":
            return cmd_measure(args, model_factory)
        return {"solve": cmd_solve, "predict": cmd_predict, "random": cmd_random}[args.command](args)
    except AutotuneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
