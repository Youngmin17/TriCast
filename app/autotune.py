"""Auto-tune v2 for the Studio Quantizer: a per-layer KL table and a multiple-choice knapsack over it
(docs/spikes/2026-10-13_autotune_mckp.md §4, v2 conditions in §6).

    python -m app.autotune table   --model Qwen/Qwen3-0.6B --revision <sha> --device cuda --out table.json
                                   [--calibrate --methods ...,gptq_int4,awq_nvfp4,smoothquant_nvfp4
                                    --calib-dataset wikitext2 --calib-samples 128 --calib-seqlen 2048
                                    --calib-seed 42] [--kv-presets kivi2,kivi4,kv_fp8]
    python -m app.autotune solve   --table table.json --budget-bits 4.5 [--kv kivi2] [--no-subtract-floor]
                                   --out <dir>
    python -m app.autotune predict --table table.json <grid.json | preset id> ...
    python -m app.autotune measure --model Qwen/Qwen3-0.6B --device cuda --seed 43 --out kl.json
                                   [--calibrate [--calib-samples N ...]] <recipe> ...
    python -m app.autotune random  --seeds 1,2,3,4 --out <dir>       # seeded random grids for the C2 check

measure: whole-recipe KL on (held-out) windows. A recipe that needs calibration is refused unless --calibrate
is given; then it is patched, calibrated by tricast's calibrate with its own calibration block (--calib-*
override it, as for tricast run) on the calibration dataset — never the measured windows — and measured.
Each result records whether it was calibrated, the calibration settings, windows fingerprint and forwards.

table: native logits once on tricast's calibration windows (WikiText-2 train, joined with blank lines,
8 × 512 tokens drawn with the seed — tricast.calibration), then for every decoder Linear and every
quantization method, only that layer with that method: the method's recipe from
``app.quantizer_recipe.build`` with tricast's ``Recipe.include`` narrowed to the layer, so the spec is the
one any Quantizer grid gives that layer. The value is KL(native || quantized) of the next-token
distributions (OBJECTIVE). BF16 is 0 by definition and not run. After the method entries come
- the patch floor of every layer (row field ``floor``): only that layer patched with no quantization on the
  default MMA path (FLOOR_FIELDS, include = the layer, exclude lm_head), i.e. what the emulated GEMM alone
  costs. BF16 is a skip (no patch), so it has no floor.
- with --kv-presets, one KV entry per decoder block and preset (top-level ``kv``): only that block's KV cache
  quantized (``kv: {preset, mode: fakequant, layers: "<block>"}``), no Linear patched.
Calibrated methods (app.quantizer_recipe.CALIBRATED; only with --calibrate) patch the one layer, run
``tricast.calibration.calibrate`` on that model state with the recipe's own calibration block (the --calib-*
settings, as ``tricast run`` calibrates the solved recipe), then measure. Input sharing: AWQ and SmoothQuant
fit one scale for all Linears that read the same input (q/k/v; gate/up) in a full recipe, so a one-layer
calibration of those kinds is not the spec the full recipe gives them. awq_nvfp4 and smoothquant_nvfp4 are
therefore tabled only for the kinds with a unique input (UNIQUE_INPUT: o_proj, down_proj); on q/k/v/gate/up
their value is null and solve never chooses them there. gptq_int4 is tabled for all seven kinds (GPTQ shares
only the Hessian, which is the same for every reader of an input).
Calibration and KL windows: --split validation draws the KL windows (table and measure) from WikiText-2
validation instead of train. Whenever something is calibrated, ``calibration_overlap`` (table meta, measure
result) records the tokens the KL windows share with the calibration windows: from one dataset split, both
start-offset sets are replayed as tricast.calibration._windows draws them (checked against both windows'
fingerprints) and the positions covered by both are counted; different splits share 0 (disjoint_by_split).
Identical KL and calibration windows (same dataset, split, count, seqlen and seed) are refused before the
model loads, since a calibrated method would then be scored on its own calibration data.
Each entry is appended to ``<out stem>.entries.jsonl`` when it finishes (methods, then floors, then KV); a
rerun with the same configuration (code hashes and calibration settings included) skips the entries already
there, and ``<out stem>.lock`` keeps a second run off the same --out. load_table reads v1 and v2 tables.

solve: minimise sum value(layer, method) subject to sum params x bits(layer, method) <= budget, bits from
``app.export_quantizer_demo.WEIGHT_BITS`` (the deck's Linear-weighted average bits). value = kl, or, when the
table has floors (v2) and --no-subtract-floor is not given, max(0, kl - floor) for every non-BF16 method
(BF16 stays 0). Exact DP on costs rounded up to multiples of the resolution (default: the exact GCD of all
costs, so nothing is rounded). KV is not chosen: --kv fixes it and its bits are not in the weight budget.
Writes the Quantizer's --from-json grid (grid.json, with the table's calibration settings when it tabled
calibrated methods), the YAML that ``--from-json grid.json`` prints (recipe.yaml) and solution.json. The
grid default is the most frequent non-BF16 choice (``default_method``); grid rows of blocks the table does
not cover (or the model does not have, e.g. Llama-3.2-1B's 16 of 28) get that default and are listed. The
recipe name defaults to ``default_name`` (exact budget and KV preset).
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

TABLE_SCHEMA = "tricast.autotune.table/2"
TABLE_SCHEMAS = ("tricast.autotune.table/1", TABLE_SCHEMA)  # what load_table reads
MEASURE_SCHEMA = "tricast.autotune.measure/2"  # /2: results say whether and how they were calibrated
SOLUTION_SCHEMA = "tricast.autotune.solution/2"
FREE = tuple(m for m in qr.METHODS if m not in qr.CALIBRATED)  # bf16 nvfp4 nvint4 mxfp4 mxfp8 int8 bfp12
# The Python-only methods (app.quantizer_recipe.EXTENDED: nf4 hqq_int4 razer_nvfp4) are calibration-free too;
# they are tabled when --methods names them (the default stays FREE, the page's alphabet).
SHARED_INPUT = ("awq_nvfp4", "smoothquant_nvfp4")  # one scale per input in a full recipe (module docstring)
UNIQUE_INPUT = ("o_proj", "down_proj")  # the kinds no other Linear of the block reads the input of
# The patch floor: the layer patched with no quantization on the default MMA path (the control
# runs/20261014_autotune/*/floor/ctrl_f23_b13_o.yaml of the v1 spike).
FLOOR_FIELDS = {"weight": None, "activation": None, "transform": "none", "weight_algo": "rtn"}
KV_PRESETS = tuple(preset for preset in qr.KV if preset != "none")
OBJECTIVE = ("KL(native || quantized) of the next-token distributions, mean over every position of every "
             "window, fp64, negative roundoff clamped to 0 (tricast report's logits_kl)")
FLOOR_RULE = "max(0, kl - floor) for every non-BF16 method; BF16 is a skip (no patch): 0, no floor"
SHARED_INPUT_RULE = (f"{' / '.join(SHARED_INPUT)} are tabled only for {' / '.join(UNIQUE_INPUT)}, null "
                     "elsewhere: a full recipe fits one scale for all Linears that read the same input "
                     "(q/k/v; gate/up)")
MAX_DP_CELLS = 1 << 28  # layers x budget units of the DP's backtracking table (one byte each)
# The table configuration a resumed run must share with the entries already written (code included: the
# app modules that define methods, grids and bits, and the tricast package actually imported).
RESUME_KEYS = ("model_id", "revision", "model_sha", "dtype", "device", "gpu_names", "windows", "seqlen",
               "seed", "source", "dataset_fingerprint", "git_sha", "git_dirty", "src_sha256", "torch",
               "transformers", "triton", "app_sha256", "tricast_path", "calibration")
# What a calibrated entry keeps of tricast.calibration.calibrate's result.
CALIBRATION_SUMMARY = ("dataset", "split", "samples", "seqlen", "seed", "sequential", "mode", "n_tokens",
                       "dataset_fingerprint", "full_model_forwards", "layers")
WINDOW_KEYS = ("dataset", "split", "samples", "seqlen", "seed")  # what decides a set of drawn windows

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
    """Stored bits per weight: element + scale bits / elements per scale (WEIGHT_BITS; None = one per row).
    Exact when ``in_features`` is a multiple of the group (every Linear ``in_features`` of the evaluated
    models is a multiple of 128); a partial last group is counted as a fraction of a scale (undercounted)."""
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
    app.quantizer_recipe.ALL_METHODS; with none, the first non-BF16 method of ``allowed`` (build needs
    one)."""
    counts = Counter(method for method in used if method != "bf16")
    if counts:
        return min(counts, key=lambda method: (-counts[method], list(qr.ALL_METHODS).index(method)))
    return next(method for method in qr.ALL_METHODS if method in allowed and method != "bf16")


def write_grid(out_dir: Path, name: str, default: str, grid: qr.Grid, kv: str,
               calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """grid.json (the Quantizer's --from-json spec) and recipe.yaml (what ``--from-json grid.json``
    prints); ``calibration`` (app.quantizer_recipe.check_calibration settings) goes into both."""
    recipe = qr.build(name, default, grid, kv, calibration=calibration)
    spec = {"name": name, "default": default, "grid": grid, "kv": kv}
    _dump(out_dir / "grid.json", spec if calibration is None else {**spec, "calibration": calibration})
    _write_atomic(out_dir / "recipe.yaml", f"---\n{qr.to_yaml(recipe)}\n")
    return recipe


def applicable(method: str, kind: str) -> bool:
    """Whether a table has an entry for ``method`` on a ``kind`` layer (module docstring: input sharing)."""
    return method not in SHARED_INPUT or kind in UNIQUE_INPUT


def parse_methods(text: str, calibrate: bool = False) -> list[str]:
    """Comma-separated method ids -> methods in app.quantizer_recipe.ALL_METHODS order; the calibrated ones
    only with ``calibrate`` (table --calibrate)."""
    given = {method.strip() for method in text.split(",") if method.strip()}
    unknown = sorted(given - set(qr.ALL_METHODS))
    if unknown:
        raise AutotuneError(f"unknown methods {unknown}; known: {list(qr.ALL_METHODS)}")
    calibrated = sorted(given & qr.CALIBRATED)
    if calibrated and not calibrate:
        raise AutotuneError(f"{calibrated} need calibration data; add --calibrate to calibrate each "
                            f"one-layer entry on the --calib-* windows (without it, only the "
                            f"calibration-free methods {[*FREE, *qr.EXTENDED]})")
    methods = [method for method in qr.ALL_METHODS if method in given]
    if all(method == "bf16" for method in methods):
        raise AutotuneError("give at least one quantization method besides bf16")
    return methods


def parse_kv_presets(text: str) -> list[str]:
    """Comma-separated KV preset ids (app.quantizer_recipe.KV without none), in KV order."""
    given = {preset.strip() for preset in text.split(",") if preset.strip()}
    unknown = sorted(given - set(KV_PRESETS))
    if unknown:
        raise AutotuneError(f"unknown KV presets {unknown}; known: {list(KV_PRESETS)}")
    return [preset for preset in KV_PRESETS if preset in given]


def calibration_settings(args: argparse.Namespace) -> dict[str, Any]:
    """The table's --calib-* flags as app.quantizer_recipe.check_calibration settings (every key)."""
    given = {key: getattr(args, f"calib_{key}") for key in ("dataset", "samples", "seqlen", "seed")}
    try:
        return qr.check_calibration(given)
    except ValueError as exc:
        raise AutotuneError(str(exc)) from None


# ---------- tables ----------
def load_table(path: Path) -> tuple[dict[str, Any], bytes, list[Layer], list[str]]:
    """A complete v1 or v2 table: (table, its bytes, layers, methods). In v2, ``floor`` is on every layer row
    or on none, and a kl is null exactly where ``applicable`` says the method is not tabled."""
    raw = path.read_bytes()
    table = json.loads(raw)
    if table.get("schema") not in TABLE_SCHEMAS or table.get("status") != "complete":
        raise AutotuneError(f"{path}: not a complete table of {list(TABLE_SCHEMAS)}")
    layers = [Layer(entry["name"], entry["block"], entry["kind"], entry["params"], entry["in_features"])
              for entry in table["layers"]]
    methods = list(table["meta"]["methods"])
    if len({"floor" in entry for entry in table["layers"]}) > 1:
        raise AutotuneError(f"{path}: only some layers have a floor")
    for entry in table["layers"]:
        wrong = [method for method in methods
                 if (entry["kl"][method] is None) == applicable(method, entry["kind"])]
        if wrong:
            raise AutotuneError(f"{path}: {entry['name']} has null where a value belongs or a value where "
                                f"null belongs for {wrong} ({SHARED_INPUT_RULE})")
    return table, raw, layers, methods


def floors(table: dict[str, Any]) -> dict[str, float] | None:
    """{layer name: patch floor} when the table measured floors (v2), else None."""
    rows = table["layers"]
    return {entry["name"]: entry["floor"] for entry in rows} if rows and "floor" in rows[0] else None


def objective(table: dict[str, Any], subtract_floor: bool) -> dict[str, dict[str, float | None]]:
    """{layer: {method: value}}: the table's kl, or, with ``subtract_floor`` and floors in the table,
    FLOOR_RULE. None = the method is not tabled for that layer."""
    floor = floors(table) if subtract_floor else None
    values: dict[str, dict[str, float | None]] = {}
    for entry in table["layers"]:
        values[entry["name"]] = {
            method: kl if kl is None or floor is None or method == "bf16"
            else max(0.0, kl - floor[entry["name"]]) for method, kl in entry["kl"].items()}
    return values


def predict(table: dict[str, Any], layers: Sequence[Layer], methods: Sequence[str],
            grid: qr.Grid, subtract_floor: bool = False) -> tuple[float, Fraction]:
    """(sum of the table's kl — FLOOR_RULE with ``subtract_floor`` — over its layers for the methods
    ``grid`` gives them, their average bits)."""
    if len(grid) < qr.BLOCKS or any(len(row) != len(qr.KINDS) for row in grid):
        raise AutotuneError(f"grid must be >= {qr.BLOCKS} blocks x {len(qr.KINDS)} kinds")
    beyond = sorted({layer.block for layer in layers if layer.block >= len(grid)})
    if beyond:
        raise AutotuneError(f"the grid has {len(grid)} blocks; the table has layers in blocks {beyond}")
    missing = sorted({method for row in grid for method in row} - set(methods))
    if missing:
        raise AutotuneError(f"the grid uses {missing}, which the table has no KL for "
                            f"(table: {list(methods)})")
    values = objective(table, subtract_floor)
    chosen = [grid[layer.block][qr.KINDS.index(layer.kind)] for layer in layers]
    untabled = [f"{layer.name}={method}" for layer, method in zip(layers, chosen, strict=True)
                if values[layer.name][method] is None]
    if untabled:
        raise AutotuneError(f"the grid uses {untabled}, which the table has no KL for ({SHARED_INPUT_RULE})")
    total = 0.0
    for layer, method in zip(layers, chosen, strict=True):
        total += values[layer.name][method]
    return total, avg_bits(layers, chosen)


def predict_kv(table: dict[str, Any], kv: str) -> float | None:
    """Sum of the table's KV entries of preset ``kv`` over its blocks (a Quantizer KV preset covers every
    block): 0 for none, None when the table has no entries for that preset."""
    if kv == "none":
        return 0.0
    rows = table.get("kv") or []
    if not rows or kv not in rows[0]["kl"]:
        return None
    total = 0.0
    for row in rows:
        total += row["kl"][kv]
    return total


# ---------- model side (torch and tricast imported here only) ----------
@dataclass
class Session:
    model: Any
    windows: list[Any]  # [1, seqlen] token tensors
    native: list[Any]  # their native logits, on the model's device
    meta: dict[str, Any]
    tokenizer: Any = None  # what calibrated entries draw their calibration windows with


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
        texts, source = _dataset_texts("wikitext2", args.split), {"dataset": "wikitext2", "split": args.split}
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
    return Session(model, windows, native, meta, tokenizer)


def kl_windows(args: argparse.Namespace) -> dict[str, Any]:
    """Where the KL windows come from (dataset None = --text), refusing --split with --text."""
    if args.text and args.split != "train":
        raise AutotuneError("--split chooses the WikiText-2 split; --text replaces WikiText-2")
    return {"dataset": None if args.text else "wikitext2", "split": None if args.text else args.split,
            "samples": args.windows, "seqlen": args.seqlen, "seed": args.seed}


def calibration_windows(recipe: Any) -> dict[str, Any]:
    """Where tricast's calibrate draws ``recipe``'s calibration windows from."""
    options = recipe.calibration_options
    return {key: options[key] for key in WINDOW_KEYS}


def refuse_identical_windows(kl: dict[str, Any], calibration: dict[str, Any], what: str) -> None:
    """Calibrating and measuring on the same windows would score a calibrated method on its own data."""
    if kl == calibration:
        raise AutotuneError(f"{what}: the KL windows {kl} are the calibration windows; change --seed, "
                            "--windows, --seqlen or --split (or the calibration settings)")


def window_starts(tokens: int, samples: int, seqlen: int, seed: int) -> list[int]:
    """The start offsets tricast.calibration._windows draws from a one-row stream of ``tokens`` tokens."""
    rng = random.Random(seed)
    starts = []
    for _ in range(samples):
        rng.randrange(1)  # the row (one joined stream)
        starts.append(rng.randrange(tokens - seqlen + 1))
    return starts


def window_overlap(tokenizer: Any, kl: dict[str, Any], kl_fingerprint: str,
                   calibration: dict[str, Any]) -> dict[str, Any]:
    """Tokens shared by the KL windows and the calibration windows. From one dataset split, both offset sets
    are replayed on the joined, tokenized split (as tricast.calibration._windows draws them; the KL replay
    must reproduce ``kl_fingerprint``) and the positions inside a window of both sets are counted. Different
    splits or datasets share none; --text windows are not compared (None)."""
    same_dataset = kl["dataset"] == calibration["dataset"]
    out: dict[str, Any] = {"kl_windows": dict(kl), "calibration_windows": dict(calibration),
                           "disjoint_by_dataset": kl["dataset"] is not None and not same_dataset,
                           "disjoint_by_split": same_dataset and kl["split"] != calibration["split"]}
    if kl["dataset"] is None:
        return {**out, "calibration_overlap_tokens": None,
                "note": "the KL windows come from --text; their overlap with the calibration dataset is not "
                        "counted"}
    if out["disjoint_by_dataset"] or out["disjoint_by_split"]:
        return {**out, "calibration_overlap_tokens": 0}
    import numpy as np
    import torch

    from tricast.calibration import _dataset_texts

    encoded = tokenizer("\n\n".join(_dataset_texts(kl["dataset"], kl["split"])), return_tensors="pt")
    ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)[0]
    covered = []
    for key, spec in (("kl_windows", kl), ("calibration_windows", calibration)):
        starts = window_starts(ids.numel(), spec["samples"], spec["seqlen"], spec["seed"])
        mask = np.zeros(ids.numel(), dtype=bool)
        digest = hashlib.sha256()  # tricast's dataset_fingerprint of the windows at these offsets
        for start in starts:
            mask[start:start + spec["seqlen"]] = True
            digest.update(ids[start:start + spec["seqlen"]].numpy().astype("<i8").tobytes())
        covered.append(mask)
        out[key] = {**spec, "offsets_sha256": hashlib.sha256(json.dumps(starts).encode()).hexdigest(),
                    "dataset_fingerprint": digest.hexdigest()}
    if out["kl_windows"]["dataset_fingerprint"] != kl_fingerprint:
        raise RuntimeError("the replayed KL window offsets do not reproduce the measured windows")
    return {**out, "stream_tokens": ids.numel(),
            "calibration_overlap_tokens": int((covered[0] & covered[1]).sum())}


def model_layers(model: Any, blocks: int | None = None) -> tuple[list[Layer], int]:
    """The Quantizer's Linear layers (block x kind) of the first ``blocks`` decoder blocks, and the model's
    block count."""
    from torch import nn

    from tricast.nn.patch import decoder_layers

    decoder = decoder_layers(model)
    if decoder is None or decoder[0] != "model.layers":
        raise AutotuneError("auto-tune v1 needs the Quantizer's layer names: "
                            "model.layers.<b>.<self_attn|mlp>.<kind>")
    layers = []
    for block in range(decoder[1] if blocks is None else min(blocks, decoder[1])):
        for kind in qr.KINDS:
            name = qr.layer_name(block, kind)
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise AutotuneError(f"{name} is not a Linear")
            layers.append(Layer(name, block, kind, module.weight.numel(), module.in_features))
    return layers, decoder[1]


def meta_blocks(table: dict[str, Any]) -> int:
    """The model's decoder block count the table records (0 if it does not)."""
    return int(table["meta"].get("model_blocks") or 0)


def method_recipe(method: str, calibration: dict[str, Any] | None = None) -> Any:
    """The tricast Recipe of the Quantizer grid with every layer on ``method`` (and ``calibration``
    settings, for the calibrated methods)."""
    from tricast.recipe import load_recipe

    return load_recipe(qr.build(f"autotune_{method}", method, qr.uniform(method), calibration=calibration))


def floor_recipe() -> Any:
    """FLOOR_FIELDS on every Linear but lm_head; an entry narrows ``include`` to one layer."""
    from tricast.recipe import load_recipe

    return load_recipe({"name": "autotune_floor", "description": "no quantization, default MMA (patch floor)",
                        "defaults": dict(FLOOR_FIELDS), "exclude": ["lm_head"]})


def kv_recipe(preset: str) -> Any:
    """The Quantizer's KV preset (fakequant) on every block and no Linear; an entry sets ``kv_layers`` to one
    block."""
    from tricast.recipe import load_recipe

    return load_recipe({"name": f"autotune_kv_{preset}", "description": "KV cache only, no Linear",
                        "defaults": dict(FLOOR_FIELDS), "include": [], "exclude": ["lm_head"],
                        "kv": dict(qr.KV[preset])})


def attention_name(block: int) -> str:
    """The attention module of a Quantizer block (what tricast's KV patch reports)."""
    return qr.layer_name(block, "q_proj").rpartition(".")[0]


def _patched_kl(session: Session, recipe: Any, seed: int, layers: Sequence[str], kv: Sequence[str] = (),
                calibrate: bool = False) -> tuple[float, dict[str, Any] | None]:
    """OBJECTIVE with ``recipe`` patched, refused unless it patched exactly the Linears ``layers`` and the
    attention modules ``kv``. With ``calibrate``, tricast's calibrate runs on that model state first with
    the recipe's own calibration settings (as tricast run does) and its CALIBRATION_SUMMARY comes back."""
    import torch

    from tricast.nn.patch import patch_model, unpatch_model

    torch.manual_seed(seed)  # every entry starts from the same RNG state, whatever ran before it
    report = patch_model(session.model, recipe)
    try:
        patched = [name for name, _ in report.patched]
        if patched != list(layers) or report.kv != list(kv):
            raise RuntimeError(f"{recipe.name}: patched Linears {patched} and KV {report.kv}, expected "
                               f"{list(layers)} and {list(kv)}")
        summary = None
        if calibrate:
            from tricast.calibration import calibrate as fit

            result = fit(session.model, recipe, tokenizer=session.tokenizer)
            summary = {key: result[key] for key in CALIBRATION_SUMMARY}
            if summary["layers"] != list(layers):
                raise RuntimeError(f"{recipe.name}: calibrated {summary['layers']}, expected {list(layers)}")
        return mean_kl(session.model, session.windows, session.native), summary
    finally:
        unpatch_model(session.model)


def _entry_kl(session: Session, recipe: Any, layer: Layer, seed: int) -> float:
    """OBJECTIVE with only ``layer`` patched with ``recipe``'s spec."""
    return _patched_kl(session, replace(recipe, include=[layer.name]), seed, [layer.name])[0]


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
    methods = parse_methods(args.methods, args.calibrate)
    kv_presets = parse_kv_presets(args.kv_presets)
    calibration = calibration_settings(args)
    kl = kl_windows(args)
    calibrated = [method for method in methods if method in qr.CALIBRATED]
    if calibrated:
        refuse_identical_windows(kl, calibration_windows(method_recipe(calibrated[0], calibration)), "table")
    out = Path(args.out)
    with _locked(out.with_name(out.stem + ".lock")):
        return _table(args, model_factory, methods, kv_presets, calibration, out)


def _table(args: argparse.Namespace, model_factory: ModelFactory | None, methods: list[str],
           kv_presets: list[str], calibration: dict[str, Any], out: Path) -> int:
    started = time.perf_counter()
    session = open_session(args, model_factory)
    session.meta["calibration"] = calibration
    layers, model_blocks = model_layers(session.model, args.blocks)
    blocks = sorted({layer.block for layer in layers})
    recipes = {method: method_recipe(method, calibration if method in qr.CALIBRATED else None)
               for method in methods if method != "bf16"}
    shas = {method: recipe.sha256 for method, recipe in recipes.items()}
    floor = floor_recipe()
    kv = {preset: kv_recipe(preset) for preset in kv_presets}
    kv_shas = {preset: recipe.sha256 for preset, recipe in kv.items()}
    entries_path = out.with_name(out.stem + ".entries.jsonl")
    entry_shas = {**shas, "floor": floor.sha256, **{f"kv:{preset}": sha for preset, sha in kv_shas.items()}}
    done, runs = _resume(entries_path, {key: session.meta[key] for key in RESUME_KEYS}, entry_shas)
    calibrated = [method for method in recipes if method in qr.CALIBRATED]
    overlap = None if not calibrated else window_overlap(session.tokenizer, kl_windows(args),
                                                         session.meta["dataset_fingerprint"],
                                                         calibration_windows(recipes[calibrated[0]]))
    fingerprints = sorted({entry["calibration"]["dataset_fingerprint"] for entry in done.values()
                           if "calibration" in entry})
    if len(fingerprints) > 1:
        raise AutotuneError(f"{entries_path}: calibrated entries of different calibration windows "
                            f"{fingerprints}; use another --out")
    env = session.meta["env"]
    runs.append({"utc": env.get("utc"), "hostname": env.get("hostname"), "device": args.device,
                 "gpu_names": env.get("gpu_names"),
                 "cuda_visible_devices": env.get("environment", {}).get("CUDA_VISIBLE_DEVICES"),
                 "pid": os.getpid()})
    todo = [(layer, method) for layer in layers for method in recipes if applicable(method, layer.kind)]
    kv_todo = [(attention_name(block), f"kv:{preset}") for block in blocks for preset in kv_presets]
    print(f"AUTOTUNE_TABLE_START layers={len(layers)} methods={len(methods)} entries={len(todo)} "
          f"done={sum(1 for layer, method in todo if (layer.name, method) in done)} "
          f"floors={len(layers)} floors_done={sum(1 for layer in layers if (layer.name, 'floor') in done)} "
          f"kv_entries={len(kv_todo)} kv_done={sum(1 for key in kv_todo if key in done)} out={out}",
          flush=True)
    computed: Counter[str] = Counter()
    with entries_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"run": runs[-1]}) + "\n")
        log.flush()

        def record(entry: dict[str, Any], kind: str) -> None:
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            done[(entry["layer"], entry["method"])] = entry
            computed[kind] += 1

        for index, layer in enumerate(layers):  # 1. methods, calibrated ones included
            layer_started = time.perf_counter()
            for method in recipes:
                if not applicable(method, layer.kind) or (layer.name, method) in done:
                    continue
                entry_started = time.perf_counter()
                entry: dict[str, Any] = {"layer": layer.name, "method": method}
                if method in qr.CALIBRATED:
                    kl, summary = _patched_kl(session, replace(recipes[method], include=[layer.name]),
                                              args.seed, [layer.name], calibrate=True)
                    assert summary is not None and overlap is not None
                    replayed = overlap["calibration_windows"].get("dataset_fingerprint")
                    if replayed is not None and summary["dataset_fingerprint"] != replayed:
                        raise RuntimeError("the replayed calibration window offsets do not reproduce the "
                                           "windows tricast calibrated on")
                    if fingerprints and summary["dataset_fingerprint"] != fingerprints[0]:
                        raise AutotuneError(f"{layer.name} {method}: calibration windows "
                                            f"{summary['dataset_fingerprint']} differ from the other "
                                            f"calibrated entries' {fingerprints[0]} (the calibration data "
                                            "changed); use another --out")
                    fingerprints = [summary["dataset_fingerprint"]]
                    entry["calibration"] = summary
                else:
                    kl = _entry_kl(session, recipes[method], layer, args.seed)
                record({**entry, "kl": kl, "recipe_sha256": shas[method],
                        "wall_s": time.perf_counter() - entry_started}, "methods")
            values = " ".join(f"{method}={done[(layer.name, method)]['kl']:.4g}" for method in recipes
                              if applicable(method, layer.kind))
            seconds = time.perf_counter() - layer_started
            print(f"[{index + 1}/{len(layers)}] {layer.name} {values} ({seconds:.1f}s)", flush=True)
        for index, layer in enumerate(layers):  # 2. patch floors
            if (layer.name, "floor") not in done:
                entry_started = time.perf_counter()
                kl = _patched_kl(session, replace(floor, include=[layer.name]), args.seed, [layer.name])[0]
                record({"layer": layer.name, "method": "floor", "kl": kl, "recipe_sha256": floor.sha256,
                        "wall_s": time.perf_counter() - entry_started}, "floor")
            value = done[(layer.name, "floor")]["kl"]
            print(f"FLOOR [{index + 1}/{len(layers)}] {layer.name} kl={value:.4g}", flush=True)
        for index, block in enumerate(blocks if kv_presets else []):  # 3. one block's KV cache
            name = attention_name(block)
            for preset in kv_presets:
                if (name, f"kv:{preset}") in done:
                    continue
                entry_started = time.perf_counter()
                kl = _patched_kl(session, replace(kv[preset], kv_layers=str(block)), args.seed, [], [name])[0]
                record({"layer": name, "method": f"kv:{preset}", "kl": kl, "recipe_sha256": kv_shas[preset],
                        "wall_s": time.perf_counter() - entry_started}, "kv")
            values = " ".join(f"{preset}={done[(name, f'kv:{preset}')]['kl']:.4g}" for preset in kv_presets)
            print(f"KV [{index + 1}/{len(blocks)}] {name} {values}", flush=True)
    wall = time.perf_counter() - started
    rows = [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
             "in_features": layer.in_features,
             "kl": {method: 0.0 if method == "bf16" else
                    done[(layer.name, method)]["kl"] if applicable(method, layer.kind) else None
                    for method in methods},
             "floor": done[(layer.name, "floor")]["kl"]}
            for layer in layers]
    kv_rows = [{"block": block, "name": attention_name(block),
                "kl": {preset: done[(attention_name(block), f"kv:{preset}")]["kl"] for preset in kv_presets}}
               for block in blocks] if kv_presets else []
    summaries = [done[(layer.name, method)]["calibration"] for layer, method in todo
                 if method in qr.CALIBRATED]
    meta = {**session.meta, "methods": methods, "blocks": len(layers) // len(qr.KINDS),
            "model_blocks": model_blocks, "method_recipe_sha256": shas, "entries": len(todo),
            "entries_computed_this_run": computed["methods"],
            "entries_wall_s": sum(done[(layer.name, method)]["wall_s"] for layer, method in todo),
            "shared_input_rule": SHARED_INPUT_RULE,
            "calibration_windows": {key: summaries[0][key] for key in
                                    ("dataset", "split", "samples", "seqlen", "seed", "n_tokens",
                                     "dataset_fingerprint")} if summaries else None,
            "calibration_overlap": overlap,
            "floor_recipe_sha256": floor.sha256, "floor_entries": len(layers),
            "floor_entries_computed_this_run": computed["floor"],
            "floor_wall_s": sum(done[(layer.name, "floor")]["wall_s"] for layer in layers),
            "kv_presets": kv_presets, "kv_recipe_sha256": kv_shas, "kv_entries": len(kv_todo),
            "kv_entries_computed_this_run": computed["kv"],
            "kv_wall_s": sum(done[key]["wall_s"] for key in kv_todo),
            "entries_path": str(entries_path), "runs": runs, "wall_s": wall}
    _dump(out, {"schema": TABLE_SCHEMA, "status": "complete", "objective": OBJECTIVE, "meta": meta,
                "layers": rows, "kv": kv_rows})
    print(f"AUTOTUNE_TABLE_DONE layers={len(layers)} methods={len(methods)} wall_s={wall:.1f} "
          f"floors={len(layers)} kv_entries={len(kv_todo)}", flush=True)
    return 0


def cmd_measure(args: argparse.Namespace, model_factory: ModelFactory | None = None) -> int:
    import torch

    from tricast.nn.patch import patch_model, unpatch_model
    from tricast.recipe import load_recipe

    started = time.perf_counter()
    # tricast run's --calib-* overrides, applied to the recipes that calibrate
    overrides = {key: value for key in ("dataset", "samples", "seqlen", "seed")
                 if (value := getattr(args, f"calib_{key}")) is not None}
    measured = kl_windows(args)
    recipes = []
    for path in map(Path, args.recipes):
        recipe = load_recipe(path)
        if recipe.needs_calibration:
            if not args.calibrate:
                raise AutotuneError(f"{path}: needs calibration; add --calibrate to calibrate it with its "
                                    "own calibration block (--calib-* override it) before the KL")
            if overrides:
                data = recipe.to_dict()
                data["calibration"] = {**(recipe.calibration or {}), **overrides}
                recipe = load_recipe(data)
            refuse_identical_windows(measured, calibration_windows(recipe), str(path))
        if recipe.kv is not None and recipe.kv.mode == "cache":
            raise AutotuneError(f"{path}: kv mode 'cache' streams token by token; use mode fakequant")
        recipes.append((path, recipe))
    session = open_session(args, model_factory)
    results = []
    overlaps: dict[tuple[Any, ...], dict[str, Any]] = {}  # by calibration windows
    for path, recipe in recipes:
        recipe_started = time.perf_counter()
        overlap = None
        if recipe.needs_calibration:
            windows = calibration_windows(recipe)
            key = tuple(windows.values())
            if key not in overlaps:
                overlaps[key] = window_overlap(session.tokenizer, measured,
                                               session.meta["dataset_fingerprint"], windows)
            overlap = overlaps[key]
        torch.manual_seed(args.seed)
        report = patch_model(session.model, recipe)
        calibration = None
        try:
            if not report.patched and not report.kv:
                raise RuntimeError(f"{path}: the recipe patched nothing")
            if recipe.needs_calibration:  # on the recipe's calibration windows, never the measured ones
                from tricast.calibration import calibrate

                fitted = calibrate(session.model, recipe, tokenizer=session.tokenizer,
                                   cache_dir=args.calib_cache)
                calibration = {key: fitted[key] for key in (*CALIBRATION_SUMMARY, "groups")}
                if args.calib_cache is not None:  # what was loaded, fitted, or could not be cached
                    cache = fitted["cache"]
                    calibration["cache"] = {"dir": cache["dir"], "hits": len(cache["hits"]),
                                            "misses": cache["misses"], "uncacheable": cache["uncacheable"],
                                            "source_sha256": cache["source_sha256"]}
                assert overlap is not None
                replayed = overlap["calibration_windows"].get("dataset_fingerprint")
                if replayed is not None and calibration["dataset_fingerprint"] != replayed:
                    raise RuntimeError("the replayed calibration window offsets do not reproduce the windows "
                                       "tricast calibrated on")
            kl = mean_kl(session.model, session.windows, session.native)
        finally:
            unpatch_model(session.model)
        results.append({"recipe": str(path), "name": recipe.name, "recipe_sha256": recipe.sha256,
                        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "kl": kl,
                        "patched": len(report.patched), "kv": report.kv,
                        "calibrated": calibration is not None, "calibration": calibration,
                        "calibration_overrides": overrides if calibration is not None else None,
                        "calibration_overlap": overlap, "wall_s": time.perf_counter() - recipe_started})
        fitted_text = "" if calibration is None else f" calibrated={calibration['dataset_fingerprint'][:12]}"
        print(f"MEASURE {recipe.name} kl={kl:.6g} patched={len(report.patched)} kv={len(report.kv)}"
              f"{fitted_text} ({results[-1]['wall_s']:.1f}s)", flush=True)
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
    kl = objective(table, False)
    options = [[method for method in methods if kl[layer.name][method] is not None] for layer in layers]
    rows = list(zip(layers, options, strict=True))
    cheapest = sum((min(cost(layer, method) for method in row) for layer, row in rows), Fraction(0))
    if cheapest > budget:
        raise AutotuneError(f"infeasible budget: {float(bits):.6g} bits is below the cheapest assignment of "
                            f"these methods, {float(cheapest / params):.6g} bits")
    floor = floors(table)
    subtract = floor is not None and not args.no_subtract_floor
    values = objective(table, subtract)
    solution = solve_mckp([[cost(layer, method) for method in row] for layer, row in rows],
                          [[values[layer.name][method] for method in row] for layer, row in rows], budget,
                          args.resolution)
    chosen = [row[j] for row, j in zip(options, solution.choice, strict=True)]
    default = default_method(chosen, methods)
    # the page's BLOCKS rows (byte-identical solutions for models up to that size), or one row per block
    grid = qr.uniform(default, max(qr.BLOCKS, meta_blocks(table), 1 + max(layer.block for layer in layers)))
    for layer, method in zip(layers, chosen, strict=True):
        grid[layer.block][qr.KINDS.index(layer.kind)] = method
    name = args.name or default_name(bits, args.kv)
    out = Path(args.out)
    meta = table["meta"]
    calibration = meta.get("calibration") if any(method in qr.CALIBRATED for method in methods) else None
    write_grid(out, name, default, grid, args.kv, calibration)
    achieved = solution.cost / params
    tabled = sorted({layer.block for layer in layers})
    counts = Counter(chosen)
    raw_sum = 0.0
    for layer, method in zip(layers, chosen, strict=True):
        raw_sum += kl[layer.name][method]
    floor_sum = None
    if floor is not None:
        corrected = values if subtract else objective(table, True)
        floor_sum = 0.0
        for layer, method in zip(layers, chosen, strict=True):
            floor_sum += corrected[layer.name][method]
    model_blocks = meta.get("model_blocks")
    result = {
        "schema": SOLUTION_SCHEMA, "name": name, "default": default, "kv": args.kv,
        "kv_note": "the solver keeps KV fixed at --kv: KV bits are not in the weight budget and KV is not "
                   "chosen; predicted_kv_kl_sum = the table's KV entries of that preset summed over its "
                   "blocks (null: the table has none), apart from predicted_kl_sum (Linear layers only)",
        "budget": {"avg_bits": float(bits), "bits_params": float(budget),
                   "given": ({"budget_bits": str(args.budget_bits)} if args.budget_bits is not None
                             else {"budget_fraction": str(args.budget_fraction)})},
        "achieved": {"avg_bits": float(achieved), "bits_params": float(solution.cost),
                     "size_pct_of_bf16": float(achieved / 16 * 100)},
        "objective": "kl_minus_floor" if subtract else "kl",
        "floor": {"table_has_floors": floor is not None, "subtracted": subtract, "rule": FLOOR_RULE},
        "predicted_kl_sum": raw_sum,
        "predicted_kl_sum_minus_floor": floor_sum,
        "predicted_kv_kl_sum": predict_kv(table, args.kv),
        "calibration": None if calibration is None else {"settings": calibration,
                                                         "windows": meta.get("calibration_windows")},
        "resolution": {"bits_params": float(solution.resolution), "capacity_units": solution.capacity,
                       "bound_bits_params": float(solution.bound),
                       "bound_avg_bits": float(solution.bound / params),
                       "meaning": "the choice never exceeds the budget, and its kl sum is at most the "
                                  "optimum for a budget smaller by the bound (0 = exact)"},
        "method_counts": {method: counts[method] for method in methods if counts[method]},
        "tie_rules": {"solver": "smallest rounded total cost, then the earliest method (METHODS order) of "
                                "each layer, deciding layers from the last",
                      "default": "most frequent non-BF16 choice, ties to the earlier method in METHODS"},
        "table": {"path": str(table_path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "schema": table["schema"], "methods": methods,
                  **{key: meta.get(key) for key in ("model_id", "model_sha", "dataset_fingerprint", "windows",
                                                    "seqlen", "seed", "git_sha", "git_dirty")}},
        "table_blocks": tabled, "model_blocks": model_blocks,
        "untabled_blocks_in_grid": [block for block in range(len(grid)) if block not in tabled],
        "grid_blocks_beyond_model": [] if model_blocks is None else list(range(model_blocks, len(grid))),
        "layers": [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
                    "method": method, "kl": kl[layer.name][method],
                    "floor": None if floor is None else floor[layer.name],
                    "value": values[layer.name][method],
                    "bits": float(weight_bits(method, layer.in_features))}
                   for layer, method in zip(layers, chosen, strict=True)],
        "solver_wall_s": time.perf_counter() - started,
        "env": run_env(),
    }
    _dump(out / "solution.json", result)
    if model_blocks is not None and model_blocks > len(tabled):
        print(f"WARNING: the table covers {len(tabled)} of {model_blocks} blocks; the others get the "
              f"default {default} without being counted", flush=True)
    counted = " ".join(f"{method}={count}" for method, count in result["method_counts"].items())
    print(f"SOLVE {name}: {counted} default={default}")
    print(f"AUTOTUNE_SOLVE_DONE name={name} avg_bits={float(achieved):.6f} budget_bits={float(bits):.6f} "
          f"objective={result['objective']} value={solution.value:.6g} kl_sum={raw_sum:.6g} out={out}",
          flush=True)
    return 0


def _grid_specs(item: str) -> list[dict[str, Any]]:
    if item in qr.PRESETS:
        default, grid, kv, _ = qr.PRESETS[item]
        return [{"name": item, "default": default, "grid": grid, "kv": kv}]
    specs = json.loads(Path(item).read_text(encoding="utf-8"))
    return specs if isinstance(specs, list) else [specs]


def cmd_predict(args: argparse.Namespace) -> int:
    table, raw, layers, methods = load_table(Path(args.table))
    floor = floors(table)
    calibration = (table["meta"].get("calibration") if any(method in qr.CALIBRATED for method in methods)
                   else None)
    results = []
    for item in args.grids:
        for spec in _grid_specs(item):
            carried = [key for key in ("options", "kv_options", "calibration") if spec.get(key)
                       and (key != "calibration" or calibration is None)]
            if carried:
                raise AutotuneError(f"{item}: {spec['name']} sets {carried}; the v1 table has entries only "
                                    "for the methods without options")
            uses_calibrated = any(method in qr.CALIBRATED for row in spec["grid"] for method in row)
            if calibration is not None and (spec.get("calibration") or uses_calibrated):
                try:
                    settings = qr.check_calibration(spec.get("calibration"))
                except ValueError as exc:
                    raise AutotuneError(f"{item}: {exc}") from None
                if settings != calibration:
                    raise AutotuneError(f"{item}: {spec['name']} calibrates with {settings}; the table's "
                                        f"calibrated entries used {calibration}")
            total, bits = predict(table, layers, methods, spec["grid"])
            corrected = None if floor is None else predict(table, layers, methods, spec["grid"], True)[0]
            kv = spec.get("kv", "none")
            kv_sum = predict_kv(table, kv)
            results.append({"source": item, "name": spec["name"], "kl_sum": total,
                            "kl_sum_minus_floor": corrected, "avg_bits": float(bits), "kv": kv,
                            "predicted_kv_kl_sum": kv_sum})
            minus = "" if corrected is None else f" kl_sum_minus_floor={corrected:.6g}"
            kv_text = " (KV not in the table)" if kv_sum is None else f" kv_kl_sum={kv_sum:.6g}"
            print(f"PREDICT {spec['name']} kl_sum={total:.6g}{minus} avg_bits={float(bits):.6f} "
                  f"kv={kv}{kv_text}")
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
        command.add_argument("--text", metavar="FILE", help="UTF-8 text in place of WikiText-2")
        command.add_argument("--split", choices=("train", "validation"), default="train",
                             help="the WikiText-2 split of the KL windows (default train, as v1)")
        command.add_argument("--out", required=True)
    table.add_argument("--methods", default=",".join(FREE))
    table.add_argument("--blocks", type=int, help="only the first N decoder blocks")
    table.add_argument("--calibrate", action="store_true",
                       help=f"accept the calibrated methods {sorted(qr.CALIBRATED)}: each entry calibrates "
                            "its one layer (tricast calibrate) with the --calib-* settings")
    for key in ("dataset", "samples", "seqlen", "seed"):
        spec = qr.CALIBRATION[key]
        table.add_argument(f"--calib-{key}", default=spec["default"], choices=spec.get("choices"),
                           type=str if spec["type"] == "choice" else int,
                           help=f"calibration {key} (default {spec['default']}), also in the solved recipe")
    table.add_argument("--kv-presets", default="", metavar="PRESETS",
                       help=f"comma-separated KV presets {list(KV_PRESETS)}: one entry per block and preset")
    measure.add_argument("--calibrate", action="store_true",
                         help="calibrate the recipes that need it (tricast calibrate with their own "
                              "calibration block) after patching, before the KL")
    for key in ("dataset", "samples", "seqlen", "seed"):
        spec = qr.CALIBRATION[key]
        measure.add_argument(f"--calib-{key}", choices=spec.get("choices"),
                             type=str if spec["type"] == "choice" else int,
                             help=f"override the recipe's calibration {key} (as tricast run --calib-{key})")
    measure.add_argument("--calib-cache", metavar="DIR",
                         help="tricast calibrate's per-layer result cache (bit-identical, reused by runs)")
    measure.add_argument("recipes", nargs="+", metavar="RECIPE")
    solve = commands.add_parser("solve", help="multiple-choice knapsack over a table")
    solve.add_argument("--table", required=True)
    budget = solve.add_mutually_exclusive_group(required=True)
    budget.add_argument("--budget-bits", type=Fraction, help="Linear-weighted average bits")
    budget.add_argument("--budget-fraction", type=Fraction, help="fraction of the BF16 Linear size")
    solve.add_argument("--resolution", type=Fraction, help="cost unit in bits x params (default: exact GCD)")
    solve.add_argument("--kv", choices=sorted(qr.KV), default="none")
    solve.add_argument("--no-subtract-floor", action="store_true",
                       help="minimise the raw kl even when the table has patch floors (v1 behaviour)")
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
