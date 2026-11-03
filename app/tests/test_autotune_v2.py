"""app/autotune.py, Auto-tune v2 (docs/spikes/2026-10-13_autotune_mckp.md §6, the four v2 conditions): patch
floors and the floor-subtracting solver against brute force, calibrated one-layer entries (GPTQ fires; AWQ /
SmoothQuant only on the unique-input kinds), measure --calibrate, per-block KV entries, v1 tables still read,
and a 2-block tiny Llama through table -> solve -> predict -> measure on CPU. No GPU, no download: the tiny
Llama, an inline text and the calibration dataset (tricast.calibration._dataset_texts) are injected.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

import tricast.calibration
import tricast.nn.patch
from app import autotune as at
from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr
from app.tests.test_autotune import ByteTokenizer, from_json
from tricast.nn.linear import EmuLinear, _operand_value
from tricast.recipe import load_recipe

QWEN = [at.Layer(qr.layer_name(b, k), b, k, ex.SHAPES[k][0] * ex.SHAPES[k][1], ex.SHAPES[k][0])
        for b in range(qr.BLOCKS) for k in qr.KINDS]  # Qwen3-0.6B's 196 Linear layers
ALL = list(qr.METHODS)  # bf16 gptq_int4 awq_nvfp4 smoothquant_nvfp4 nvfp4 nvint4 mxfp4 mxfp8 int8 bfp12
TEXT = " ".join(f"token{i % 37} word{i % 11}." for i in range(120))
CALIB_TEXT = " ".join(f"calib{i % 23} sample{i % 7}," for i in range(200))
LLAMA_ARGS = ["--device", "cpu", "--windows", "2", "--seqlen", "192"]  # > KIVI's 128-token residual + 32
# runs/20261014_autotune/a100/floor/ctrl_f23_b13_o.yaml, the v1 spike's floor control (runs/ is not in git)
CONTROL = """name: ctrl_f23_b13_o
description: no quantization, default MMA, only model.layers.13.self_attn.o_proj
defaults: {weight: null, activation: null, transform: none, weight_algo: rtn}
include: ['model.layers.13.self_attn.o_proj']
exclude: [lm_head]
"""
# Block 0 of runs/20261014_autotune/a100/table.json (tricast.autotune.table/1, Qwen3-0.6B c1899de, A100)
V1_ROWS = [
    ("q_proj", 2097152, 1024, [0.0025494483369260844, 0.002611082103779303, 0.0031399737828578097,
                               0.0013037900125582348, 0.0012025719722771647, 0.0039109799539570715]),
    ("k_proj", 1048576, 1024, [0.002428966991423563, 0.002519940170869726, 0.003555773021781219,
                               0.0013123446718108654, 0.0012092044854663119, 0.00372305140834462]),
    ("v_proj", 1048576, 1024, [0.010688632254114627, 0.011139231430683016, 0.01701972636708983,
                               0.0019805328839665663, 0.001648978445719442, 0.01914068264156113]),
    ("o_proj", 2097152, 2048, [0.008329430171548056, 0.00864873634139967, 0.01660412595468446,
                               0.0020474448403832654, 0.0013628161203268112, 0.012196823089490001]),
    ("gate_proj", 3145728, 1024, [0.0034320406990477393, 0.0036245439488464893, 0.006674401760596874,
                                  0.0014828285252793197, 0.00133244028765851, 0.005422393158233409]),
    ("up_proj", 3145728, 1024, [0.003911449191628977, 0.004407704330644052, 0.0060323202107367205,
                                0.0014333155198623247, 0.0013929519732583915, 0.0057412049335977345]),
    ("down_proj", 3145728, 3072, [0.004068459232528198, 0.005746677422270783, 0.008438889232820662,
                                  0.0016320324525125275, 0.0017728685566199684, 0.007603322981578805]),
]


def run(argv: list[str], **kwargs: Any) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        rc = at.main(argv, **kwargs)
    return rc, stdout.getvalue(), stderr.getvalue()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- floors in the solver (no model) ----------
def random_table(rng: random.Random, path: Path, calibrated: bool) -> tuple[list[at.Layer], list[str]]:
    """A v2 table over 1-5 Qwen3-0.6B layers of blocks 0-1, seeded kl and floors with ties and kl < floor,
    null for AWQ / SmoothQuant on the shared-input kinds."""
    layers = sorted(rng.sample(QWEN[:14], rng.randint(1, 5)), key=QWEN.index)
    methods = [m for m in ALL if calibrated or m in at.FREE]
    rows = []
    for layer in layers:
        kl = {m: 0.0 if m == "bf16" else round(rng.random() * 2, 2) * 1e-3 if at.applicable(m, layer.kind)
              else None for m in methods}
        rows.append({"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
                     "in_features": layer.in_features, "kl": kl, "floor": round(rng.random(), 2) * 1e-3})
    meta = {"methods": methods, "model_blocks": qr.BLOCKS,
            "calibration": qr.check_calibration(None) if calibrated else None}
    path.write_text(json.dumps({"schema": at.TABLE_SCHEMA, "status": "complete", "objective": at.OBJECTIVE,
                                "meta": meta, "layers": rows, "kv": []}), encoding="utf-8")
    return layers, methods


def brute_force(costs: list[list[Fraction]], values: list[list[float]], budget: Fraction) -> float:
    """The least sum of values (added row by row, as the DP does) over every choice within the budget."""
    scale = math.lcm(*(c.denominator for row in costs for c in row), budget.denominator)
    total_cost, total_value = np.zeros(1, dtype=np.int64), np.zeros(1)
    for cost_row, value_row in zip(costs, values, strict=True):
        ints = np.asarray([int(c * scale) for c in cost_row], dtype=np.int64)
        total_cost = (total_cost[:, None] + ints[None, :]).ravel()
        total_value = (total_value[:, None] + np.asarray(value_row)[None, :]).ravel()
    fits = total_cost <= int(budget * scale)
    return float(total_value[fits].min())


def test_floor_subtracting_solver_equals_brute_force_and_no_subtract_is_v1(tmp_path: Path) -> None:
    """40 random instances (half with the calibrated methods and their nulls): the solved objective equals the
    brute-force optimum of max(0, kl - floor) (BF16 0), and with --no-subtract-floor that of the raw kl, never
    over the budget and never a null option. Free-method tables: --no-subtract-floor gives the same grid as
    the same table written as v1 (no floors). The floor must change the choice somewhere (not vacuous)."""
    rng = random.Random(7)
    changed = 0
    for t in range(40):
        calibrated = t % 2 == 1
        path = tmp_path / f"t{t}.json"
        layers, methods = random_table(rng, path, calibrated)
        table = load(path)
        params = sum(layer.params for layer in layers)
        bits = Fraction(rng.uniform(4.3, 16.5)).limit_denominator(1000)
        options = [[m for m in methods if row["kl"][m] is not None] for row in table["layers"]]
        costs = [[at.cost(layer, m) for m in row] for layer, row in zip(layers, options, strict=True)]
        raw = [[row["kl"][m] for m in opts] for row, opts in zip(table["layers"], options, strict=True)]
        minus = [[0.0 if m == "bf16" else max(0.0, row["kl"][m] - row["floor"]) for m in opts]
                 for row, opts in zip(table["layers"], options, strict=True)]
        chosen = {}
        for flag, values, key in (([], minus, "predicted_kl_sum_minus_floor"),
                                  (["--no-subtract-floor"], raw, "predicted_kl_sum")):
            out = tmp_path / f"s{t}{'_raw' if flag else ''}"
            rc, _, err = run(["solve", "--table", str(path), "--budget-bits", str(bits), *flag,
                              "--out", str(out)])
            assert rc == 0, err
            solution = load(out / "solution.json")
            assert solution["objective"] == ("kl" if flag else "kl_minus_floor")
            assert solution["floor"] == {"table_has_floors": True, "subtracted": not flag,
                                         "rule": at.FLOOR_RULE}
            picks = [entry["method"] for entry in solution["layers"]]
            assert all(table["layers"][i]["kl"][m] is not None for i, m in enumerate(picks)), t
            assert at.avg_bits(layers, picks) <= bits
            total = 0.0
            for i, m in enumerate(picks):
                total += values[i][options[i].index(m)]
            assert solution[key] == total == brute_force(costs, values, bits * params), (t, flag)
            chosen[bool(flag)] = (picks, (out / "grid.json").read_bytes())
        changed += chosen[False][0] != chosen[True][0]
        if not calibrated:  # the same table as v1: no floors, schema 1
            v1 = {**table, "schema": "tricast.autotune.table/1", "kv": None}
            v1["layers"] = [{key: value for key, value in row.items() if key != "floor"}
                            for row in table["layers"]]
            del v1["kv"]
            path.write_text(json.dumps(v1), encoding="utf-8")
            rc, _, err = run(["solve", "--table", str(path), "--budget-bits", str(bits),
                              "--out", str(tmp_path / "v1")])
            assert rc == 0, err
            assert (tmp_path / "v1" / "grid.json").read_bytes() == chosen[True][1], t
            assert load(tmp_path / "v1" / "solution.json")["objective"] == "kl"
    assert changed >= 3


def test_v1_tables_still_load_solve_and_predict(tmp_path: Path) -> None:
    """A real v1 table (block 0 of the A100 run) loads; without floors the objective is the raw kl, the
    flag changes nothing, and predict reports no floor-corrected sum and no KV entries."""
    rows = [{"name": qr.layer_name(0, kind), "block": 0, "kind": kind, "params": params, "in_features": k,
             "kl": dict(zip(at.FREE, [0.0, *values], strict=True))} for kind, params, k, values in V1_ROWS]
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"schema": "tricast.autotune.table/1", "status": "complete",
                                "objective": at.OBJECTIVE,
                                "meta": {"model_id": "Qwen/Qwen3-0.6B", "methods": list(at.FREE), "blocks": 1,
                                         "model_blocks": 28}, "layers": rows}), encoding="utf-8")
    table, _, layers, methods = at.load_table(path)
    assert methods == list(at.FREE) and len(layers) == 7 and at.floors(table) is None
    grids = []
    for flag in ([], ["--no-subtract-floor"]):
        rc, stdout, err = run(["solve", "--table", str(path), "--budget-bits", "5", *flag,
                               "--out", str(tmp_path / f"s{len(grids)}")])
        assert rc == 0, err
        assert "WARNING: the table covers 1 of 28 blocks" in stdout
        solution = load(tmp_path / f"s{len(grids)}" / "solution.json")
        assert solution["schema"] == at.SOLUTION_SCHEMA and solution["table"]["schema"].endswith("/1")
        assert solution["objective"] == "kl" and solution["predicted_kl_sum_minus_floor"] is None
        assert solution["floor"]["table_has_floors"] is False and solution["calibration"] is None
        assert solution["predicted_kv_kl_sum"] == 0.0  # --kv none
        grids.append((tmp_path / f"s{len(grids)}" / "grid.json").read_bytes())
    assert grids[0] == grids[1]
    rc, stdout, err = run(["predict", "--table", str(path), "qd_nvfp4", "--out", str(tmp_path / "p.json")])
    assert rc == 0, err
    result = load(tmp_path / "p.json")["results"][0]
    total = 0.0
    for row in rows:
        total += row["kl"]["nvfp4"]
    assert result["kl_sum"] == total and result["kl_sum_minus_floor"] is None
    assert result["predicted_kv_kl_sum"] == 0.0
    spec = tmp_path / "kv.json"
    spec.write_text(json.dumps({"name": "kv", "default": "nvfp4", "grid": qr.uniform("nvfp4"),
                                "kv": "kivi2"}))
    rc, stdout, _ = run(["predict", "--table", str(path), str(spec), "--out", str(tmp_path / "p2.json")])
    assert rc == 0 and "(KV not in the table)" in stdout
    assert load(tmp_path / "p2.json")["results"][0]["predicted_kv_kl_sum"] is None


def test_inconsistent_tables_are_refused(tmp_path: Path) -> None:
    rng = random.Random(3)
    path = tmp_path / "t.json"
    while True:  # a calibrated table with a q_proj and an o_proj row
        random_table(rng, path, True)
        table = load(path)
        kinds = [row["kind"] for row in table["layers"]]
        if "q_proj" in kinds and "o_proj" in kinds:
            break
    q, o = kinds.index("q_proj"), kinds.index("o_proj")
    for edit, words in ((lambda t: t["layers"][q]["kl"].update(awq_nvfp4=0.001),
                         "a value where null belongs"),
                        (lambda t: t["layers"][o]["kl"].update(smoothquant_nvfp4=None), "null where a value"),
                        (lambda t: t["layers"][o].pop("floor"), "only some layers have a floor")):
        broken = json.loads(json.dumps(table))
        edit(broken)
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(at.AutotuneError, match=words):
            at.load_table(path)


def test_floor_recipe_is_the_v1_control_spec() -> None:
    control = load_recipe(yaml.safe_load(CONTROL))
    layer = "model.layers.13.self_attn.o_proj"
    floor = dataclasses.replace(at.floor_recipe(), include=[layer])
    assert (floor.defaults, floor.include, floor.exclude) == (control.defaults, control.include,
                                                              control.exclude)
    assert floor.overrides == control.overrides == [] and floor.kv is None and not floor.needs_calibration
    assert control.defaults.weight is None and control.defaults.activation is None
    for name in (layer, "model.layers.13.self_attn.q_proj", "lm_head"):
        assert floor.spec_for(name, decoder=("model.layers", qr.BLOCKS)) == control.spec_for(
            name, decoder=("model.layers", qr.BLOCKS))


def test_bad_flags_are_refused_before_any_model_loads(tmp_path: Path) -> None:
    def no_model(*args: object) -> tuple[object, object]:
        raise AssertionError("the model must not load")

    for extra, words in ((["--kv-presets", "kivi2,kivi3"], "unknown KV presets ['kivi3']"),
                         (["--calibrate", "--methods", "bf16,gptq_int4", "--calib-samples", "0"],
                          "calibration.samples: 0 is outside [1, 512]"),
                         (["--methods", "bf16,awq_nvfp4"],
                          "['awq_nvfp4'] need calibration data; add --calibrate")):
        rc, _, err = run(["table", "--out", str(tmp_path / "t.json"), *LLAMA_ARGS, *extra],
                         model_factory=no_model)
        assert rc == 2 and words in err, err


# ---------- tiny Llama (2 decoder blocks, fewer than the grid's 28) ----------
MODELS: list[Any] = []


def tiny_llama(model_id: str, dtype: str, device: str, revision: str | None) -> tuple[object, object]:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    cfg = transformers.LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=32, max_position_embeddings=256)
    model = transformers.LlamaForCausalLM(cfg).eval()
    MODELS.append(model)
    return model, ByteTokenizer()


def make_root(root: Path) -> Path:
    (root / "model").mkdir()  # an existing path: capture_env records a local model, no Hub lookup
    (root / "text.txt").write_text(TEXT, encoding="utf-8")
    return root


@contextlib.contextmanager
def spies(calib_text: str = CALIB_TEXT, splits: dict[str, str] | None = None) -> Any:
    """Record every patch report and, at every KL measurement, the patched Linears (with their effective
    weights) and KV attention modules; serve ``calib_text`` as every dataset, or ``splits[split]``."""
    reports: list[tuple[list[str], list[str]]] = []
    states: list[dict[str, Any]] = []
    datasets: list[tuple[str, str]] = []
    real_patch, real_kl = tricast.nn.patch.patch_model, at.mean_kl

    def patch(model: Any, recipe: Any, **kwargs: Any) -> Any:
        report = real_patch(model, recipe, **kwargs)
        reports.append(([name for name, _ in report.patched], list(report.kv)))
        return report

    def kl(model: Any, *args: Any) -> float:
        states.append({"linears": {name: (m.spec, _operand_value(m._weight_operand).clone())
                                   for name, m in model.named_modules() if isinstance(m, EmuLinear)},
                       "kv": [name for name, m in model.named_modules()
                              if hasattr(m, "_tricast_kv_attention")]})
        return real_kl(model, *args)

    def texts(dataset: str, split: str) -> Any:
        datasets.append((dataset, split))
        return iter([calib_text if splits is None else splits[split]])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tricast.nn.patch, "patch_model", patch)
        mp.setattr(at, "mean_kl", kl)
        mp.setattr(tricast.calibration, "_dataset_texts", texts)
        yield {"reports": reports, "states": states, "datasets": datasets}


def table_run(root: Path, out: Path, *extra: str, calib_text: str = CALIB_TEXT) -> tuple[int, str, str, dict]:
    with spies(calib_text) as spy:
        rc, stdout, stderr = run(["table", "--model", str(root / "model"), "--text", str(root / "text.txt"),
                                  *LLAMA_ARGS, "--out", str(out), *extra], model_factory=tiny_llama)
    return rc, stdout, stderr, spy


FREE_ARGS = ("--methods", "bf16,nvfp4,int8", "--kv-presets", "kv_fp8,kivi2")


@pytest.fixture(scope="module")
def llama(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    pytest.importorskip("transformers")
    root = make_root(tmp_path_factory.mktemp("llama"))
    rc, stdout, stderr, spy = table_run(root, root / "table.json", *FREE_ARGS)
    assert rc == 0, stderr
    lines = stdout.splitlines()
    assert lines[0].startswith("AUTOTUNE_TABLE_START layers=14 methods=3 entries=28 done=0 floors=14 "
                               "floors_done=0 kv_entries=4 kv_done=0")
    assert sum(line.startswith("[") for line in lines) == 14
    assert sum(line.startswith("FLOOR [") for line in lines) == 14
    assert sum(line.startswith("KV [") for line in lines) == 2
    assert lines[-1].startswith("AUTOTUNE_TABLE_DONE layers=14 methods=3 wall_s=")
    return root, spy


def test_tiny_llama_floors_and_kv_entries(llama: tuple[Path, dict], tmp_path: Path) -> None:
    root, spy = llama
    table = load(root / "table.json")
    meta = table["meta"]
    assert table["schema"] == "tricast.autotune.table/2" and meta["model_blocks"] == meta["blocks"] == 2
    names = [row["name"] for row in table["layers"]]
    assert names == [qr.layer_name(b, k) for b in range(2) for k in qr.KINDS]
    for row in table["layers"]:  # a floor for every layer, > 0: the emulated GEMM alone moves the logits
        assert math.isfinite(row["floor"]) and row["floor"] > 0 and row["kl"]["bf16"] == 0.0
    assert meta["floor_entries"] == meta["floor_entries_computed_this_run"] == 14
    assert meta["kv_presets"] == ["kivi2", "kv_fp8"] and meta["kv_entries"] == 4
    assert meta["calibration"] == qr.check_calibration(None) and meta["calibration_windows"] is None
    assert [row["block"] for row in table["kv"]] == [0, 1]
    for row in table["kv"]:
        assert row["name"] == f"model.layers.{row['block']}.self_attn"
        assert all(math.isfinite(v) and v > 0 for v in row["kl"].values())
    # what was patched at each measurement, in table order: 28 method entries, 14 floors, 4 KV entries
    expected = ([([name], []) for name in names for _ in range(2)] + [([name], []) for name in names]
                + [([], [f"model.layers.{b}.self_attn"]) for b in range(2) for _ in range(2)])
    assert spy["reports"] == expected  # tricast's own patch reports: Linear list and KV list
    assert [(list(s["linears"]), s["kv"]) for s in spy["states"]] == expected  # the live model at measurement
    for state in spy["states"][28:42]:  # the floor: the layer patched with no quantization
        (spec, weight), = state["linears"].values()
        assert spec.weight is None and spec.activation is None and spec.transform.kind == "none"
    model = MODELS[-1]
    assert not any(isinstance(m, EmuLinear) or hasattr(m, "_tricast_kv_attention") for m in model.modules())

    # the same measurement through `measure` on the table's windows: the v1 control spec and a KV-only recipe
    control = tmp_path / "ctrl.yaml"
    control.write_text(CONTROL.replace("13", "1"), encoding="utf-8")
    kv_only = tmp_path / "kv_only.yaml"
    kv_only.write_text(yaml.safe_dump({"name": "kv_only_b1", "defaults": dict(at.FLOOR_FIELDS), "include": [],
                                       "exclude": ["lm_head"],
                                       "kv": {"preset": "kivi2", "mode": "fakequant", "layers": "1"}}))
    rc, _, err = run(["measure", "--model", str(root / "model"), "--text", str(root / "text.txt"),
                      *LLAMA_ARGS, "--seed", "42", "--out", str(tmp_path / "kl.json"), str(control),
                      str(kv_only)], model_factory=tiny_llama)
    assert rc == 0, err
    measured = load(tmp_path / "kl.json")
    assert measured["meta"]["dataset_fingerprint"] == meta["dataset_fingerprint"]
    floor = next(row["floor"] for row in table["layers"] if row["name"] == "model.layers.1.self_attn.o_proj")
    assert [r["kl"] for r in measured["results"]] == [floor, table["kv"][1]["kl"]["kivi2"]]
    assert [(r["patched"], r["kv"]) for r in measured["results"]] == [(1, []),
                                                                      (0, ["model.layers.1.self_attn"])]


def test_tiny_llama_table_solve_predict_measure(llama: tuple[Path, dict], tmp_path: Path) -> None:
    root, _ = llama
    table = load(root / "table.json")
    out = tmp_path / "auto"
    rc, stdout, err = run(["solve", "--table", str(root / "table.json"), "--budget-bits", "6",
                           "--kv", "kivi2", "--out", str(out)])
    assert rc == 0, err
    assert stdout.splitlines()[-1].startswith("AUTOTUNE_SOLVE_DONE name=auto_6_kivi2 ")
    solution = load(out / "solution.json")
    spec = load(out / "grid.json")
    assert from_json(out / "grid.json") == (out / "recipe.yaml").read_text(encoding="utf-8")
    assert "calibration" not in spec and solution["calibration"] is None
    assert solution["objective"] == "kl_minus_floor" and solution["model_blocks"] == 2
    assert solution["grid_blocks_beyond_model"] == solution["untabled_blocks_in_grid"] == list(range(2, 28))
    assert all(row == [spec["default"]] * 7 for row in spec["grid"][2:])  # rows beyond the model: the default
    assert "KV bits are not in the weight budget" in solution["kv_note"]
    assert solution["predicted_kv_kl_sum"] == table["kv"][0]["kl"]["kivi2"] + table["kv"][1]["kl"]["kivi2"]
    raw = minus = 0.0
    for row, chosen in zip(table["layers"], solution["layers"], strict=True):
        assert chosen["kl"] == row["kl"][chosen["method"]] and chosen["floor"] == row["floor"]
        value = 0.0 if chosen["method"] == "bf16" else max(0.0, chosen["kl"] - row["floor"])
        assert chosen["value"] == value
        raw += chosen["kl"]
        minus += value
    assert (solution["predicted_kl_sum"], solution["predicted_kl_sum_minus_floor"]) == (raw, minus)
    assert solution["achieved"]["avg_bits"] <= 6

    rc, stdout, err = run(["predict", "--table", str(root / "table.json"), str(out / "grid.json"),
                           "--out", str(tmp_path / "predict.json")])
    assert rc == 0, err
    result = load(tmp_path / "predict.json")["results"][0]
    assert (result["kl_sum"], result["kl_sum_minus_floor"]) == (raw, minus)
    assert result["predicted_kv_kl_sum"] == solution["predicted_kv_kl_sum"] and result["kv"] == "kivi2"

    rc, stdout, err = run(["measure", "--model", str(root / "model"), "--text", str(root / "text.txt"),
                           *LLAMA_ARGS, "--seed", "43", "--out", str(tmp_path / "kl.json"),
                           str(out / "recipe.yaml")], model_factory=tiny_llama)
    assert rc == 0, err
    measured = load(tmp_path / "kl.json")["results"][0]
    assert measured["patched"] == sum(entry["method"] != "bf16" for entry in solution["layers"])
    assert measured["kv"] == ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    assert math.isfinite(measured["kl"]) and measured["kl"] > 0


# ---------- calibrated methods ----------
CALIB_ARGS = ("--calibrate", "--methods", "bf16,nvfp4,gptq_int4,awq_nvfp4,smoothquant_nvfp4", "--blocks", "1",
              "--calib-samples", "4", "--calib-seqlen", "64")
SETTINGS = qr.check_calibration({"samples": 4, "seqlen": 64})


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, Any]:
    pytest.importorskip("transformers")
    root = make_root(tmp_path_factory.mktemp("calibrated"))
    rc, stdout, stderr, spy = table_run(root, root / "table.json", *CALIB_ARGS)
    assert rc == 0, stderr
    assert stdout.splitlines()[0].startswith("AUTOTUNE_TABLE_START layers=7 methods=5 entries=18 done=0 "
                                             "floors=7")
    return root, spy, MODELS[-1]


def test_calibrated_entries_fire_and_shared_input_kinds_are_null(calibrated: tuple[Path, dict, Any]) -> None:
    root, spy, model = calibrated
    table = load(root / "table.json")
    meta = table["meta"]
    for row in table["layers"]:
        unique = row["kind"] in ("o_proj", "down_proj")
        for method in ("awq_nvfp4", "smoothquant_nvfp4"):
            assert (row["kl"][method] is None) == (not unique), (row["name"], method)
        for method in ("nvfp4", "gptq_int4", *(("awq_nvfp4", "smoothquant_nvfp4") if unique else ())):
            assert math.isfinite(row["kl"][method]) and row["kl"][method] > 0
    # the calibration settings are in the resume configuration and the windows in the table
    assert "calibration" in at.RESUME_KEYS and meta["calibration"] == SETTINGS
    windows = meta["calibration_windows"]
    assert (windows["dataset"], windows["split"], windows["samples"], windows["seqlen"], windows["seed"],
            windows["n_tokens"]) == ("wikitext2", "train", 4, 64, 42, 256)
    assert len(windows["dataset_fingerprint"]) == 64
    entries = [json.loads(line) for line in (root / "table.entries.jsonl").read_text().splitlines()[2:]]
    calibrated_entries = [e for e in entries if e["method"] in qr.CALIBRATED]
    assert len(calibrated_entries) == 7 + 2 + 2 and spy["datasets"] == [("wikitext2", "train")] * 11
    for entry in calibrated_entries:  # each one calibrated exactly its layer, on the same windows
        assert entry["calibration"]["layers"] == [entry["layer"]]
        assert entry["calibration"]["mode"] == "nonsequential"
        assert entry["calibration"]["dataset_fingerprint"] == windows["dataset_fingerprint"]
    assert all("calibration" not in e for e in entries if e["method"] not in qr.CALIBRATED)

    # firing: the GPTQ o_proj entry measured a weight other than plain RTN INT4 of the same spec
    order = [(e["layer"], e["method"]) for e in entries]
    state = spy["states"][order.index(("model.layers.0.self_attn.o_proj", "gptq_int4"))]
    (spec, gptq), = state["linears"].values()
    assert spec.weight_algo.kind == "gptq"
    original = model.get_submodule("model.layers.0.self_attn.o_proj")
    fresh, _ = tiny_llama("", "float32", "cpu", None)
    assert isinstance(original, torch.nn.Linear)  # unpatched, and its weight untouched by the calibration
    assert torch.equal(original.weight, fresh.get_submodule("model.layers.0.self_attn.o_proj").weight)
    rtn_spec = dataclasses.replace(spec, weight_algo=dataclasses.replace(spec.weight_algo, kind="rtn"))
    rtn = _operand_value(EmuLinear.from_linear(original, rtn_spec)._weight_operand)
    assert gptq.shape == rtn.shape == original.weight.shape and not torch.equal(gptq, rtn)
    assert (gptq - rtn).abs().max() > 0 and (gptq - original.weight).norm() < 0.5 * original.weight.norm()

    # solve: never AWQ / SmoothQuant on q/k/v/gate/up; grid and recipe carry the table's calibration
    for bits in ("4.4", "5", "6", "9"):
        out = root / f"auto_{bits}"
        rc, _, err = run(["solve", "--table", str(root / "table.json"), "--budget-bits", bits,
                          "--out", str(out)])
        assert rc == 0, err
        solution = load(out / "solution.json")
        for entry in solution["layers"]:
            assert at.applicable(entry["method"], entry["kind"]) and entry["kl"] is not None
        spec_json = load(out / "grid.json")
        assert spec_json["calibration"] == SETTINGS and solution["calibration"]["settings"] == SETTINGS
        assert solution["calibration"]["windows"] == windows
        text = (out / "recipe.yaml").read_text(encoding="utf-8")
        assert from_json(out / "grid.json") == text
        options = load_recipe(yaml.safe_load(text)).calibration_options
        assert {key: options[key] for key in SETTINGS} == SETTINGS
        rc, _, err = run(["predict", "--table", str(root / "table.json"), str(out / "grid.json")])
        assert rc == 0, err
    assert any(e["method"] in qr.CALIBRATED for e in load(root / "auto_9" / "solution.json")["layers"])

    # predict refuses other calibration settings, and AWQ where the table has no entry
    grid = qr.uniform("nvfp4")
    grid[0][qr.KINDS.index("o_proj")] = "gptq_int4"
    for extra, words in (({"calibration": {**SETTINGS, "samples": 8}}, "calibrates with"),
                         ({}, "calibrates with"),  # default settings (128 x 2048) are not the table's
                         ({"calibration": SETTINGS, "q": True}, "model.layers.0.self_attn.q_proj=awq_nvfp4")):
        g = [row[:] for row in grid]
        if extra.pop("q", False):
            g[0][0] = "awq_nvfp4"
        path = root / "g.json"
        path.write_text(json.dumps({"name": "g", "default": "nvfp4", "grid": g, "kv": "none", **extra}))
        rc, _, err = run(["predict", "--table", str(root / "table.json"), str(path)])
        assert rc == 2 and words in err, err


def test_measure_calibrates_recipes_that_need_it(calibrated: tuple[Path, dict, Any], tmp_path: Path) -> None:
    """A grid with GPTQ INT4 on one layer is calibrated by measure --calibrate on the calibration windows (not
    the measured ones) and, on the table's windows, measures exactly the table's calibrated entry; the same
    recipe with that layer on RTN INT4 is not calibrated and measures something else."""
    root, _, _ = calibrated
    table = load(root / "table.json")
    layer = "model.layers.0.self_attn.o_proj"
    grid = qr.uniform("bf16")
    grid[0][qr.KINDS.index("o_proj")] = "gptq_int4"
    gptq = at.write_grid(tmp_path / "gptq", "one_gptq", "gptq_int4", grid, "none", SETTINGS)
    rtn = {**gptq, "name": "one_rtn", "defaults": {**gptq["defaults"], "weight_algo": "rtn"}}
    del rtn["calibration"]
    (tmp_path / "rtn.yaml").write_text(yaml.safe_dump(rtn), encoding="utf-8")
    decoder = ("model.layers", 2)
    specs = [load_recipe(data).spec_for(layer, decoder=decoder) for data in (gptq, rtn)]
    assert specs[1] == dataclasses.replace(specs[0], weight_algo=dataclasses.replace(specs[0].weight_algo,
                                                                                     kind="rtn"))
    assert load_recipe(gptq).needs_calibration and not load_recipe(rtn).needs_calibration
    recipes = [str(tmp_path / "gptq" / "recipe.yaml"), str(tmp_path / "rtn.yaml")]
    common = ["measure", "--model", str(root / "model"), "--text", str(root / "text.txt"), *LLAMA_ARGS,
              "--seed", "42"]

    def no_model(*args: object) -> tuple[object, object]:
        raise AssertionError("the model must not load")

    rc, _, err = run([*common, "--out", str(tmp_path / "no.json"), *recipes], model_factory=no_model)
    assert rc == 2 and "needs calibration; add --calibrate" in err
    with spies() as spy:
        rc, stdout, err = run([*common, "--calibrate", "--out", str(tmp_path / "kl.json"), *recipes],
                              model_factory=tiny_llama)
    assert rc == 0, err
    measured = load(tmp_path / "kl.json")
    assert measured["schema"] == "tricast.autotune.measure/2"
    g, r = measured["results"]
    windows = table["meta"]["calibration_windows"]
    assert g["calibrated"] is True and g["calibration_overrides"] == {}
    assert g["calibration"]["layers"] == [layer] and g["calibration"]["mode"] == "nonsequential"
    assert g["calibration"]["dataset_fingerprint"] == windows["dataset_fingerprint"]
    assert g["calibration"]["dataset_fingerprint"] != measured["meta"]["dataset_fingerprint"]
    assert g["calibration"]["full_model_forwards"] > 0 and g["calibration"]["n_tokens"] == 4 * 64
    assert spy["datasets"] == [("wikitext2", "train")]  # one calibration: the GPTQ recipe only
    assert r["calibrated"] is False and r["calibration"] is None and r["calibration_overrides"] is None
    assert g["kl"] != r["kl"] and g["patched"] == r["patched"] == 1
    entry = next(row for row in table["layers"] if row["name"] == layer)["kl"]["gptq_int4"]
    assert g["kl"] == entry  # the full-recipe path = the table's one-layer entry, exactly

    with spies():  # --calib-* override the recipe's block, recorded per result
        rc, _, err = run([*common, "--calibrate", "--calib-samples", "3", "--out", str(tmp_path / "kl3.json"),
                          recipes[0]], model_factory=tiny_llama)
    assert rc == 0, err
    g3 = load(tmp_path / "kl3.json")["results"][0]
    assert g3["calibration_overrides"] == {"samples": 3} and g3["calibration"]["samples"] == 3
    assert g3["calibration"]["dataset_fingerprint"] != g["calibration"]["dataset_fingerprint"]
    assert g3["recipe_sha256"] != g["recipe_sha256"]


def test_calibrated_floor_and_kv_entries_resume(calibrated: tuple[Path, dict, Any], tmp_path: Path) -> None:
    root, _, _ = calibrated
    out = tmp_path / "t.json"
    entries = tmp_path / "t.entries.jsonl"
    rc, _, err, _ = table_run(root, out, *CALIB_ARGS, "--kv-presets", "kv_fp8")
    assert rc == 0, err
    first = load(out)
    lines = entries.read_text(encoding="utf-8").splitlines(keepends=True)
    gptq = next(i for i, line in enumerate(lines)
                if '"model.layers.0.mlp.down_proj", "method": "gptq_int4"' in line)
    kept = [line for i, line in enumerate(lines) if i != gptq][:-3]  # drop a calibrated entry, 2 floors, 1 KV
    assert [json.loads(line)["method"] for line in lines[-3:]] == ["floor", "floor", "kv:kv_fp8"]
    entries.write_text("".join(kept), encoding="utf-8")
    out.unlink()
    rc, stdout, err, spy = table_run(root, out, *CALIB_ARGS, "--kv-presets", "kv_fp8")
    assert rc == 0, err
    assert stdout.splitlines()[0].startswith("AUTOTUNE_TABLE_START layers=7 methods=5 entries=18 done=17 "
                                             "floors=7 floors_done=5 kv_entries=1 kv_done=0")
    meta = load(out)["meta"]
    assert (meta["entries_computed_this_run"], meta["floor_entries_computed_this_run"],
            meta["kv_entries_computed_this_run"]) == (1, 2, 1)
    assert spy["datasets"] == [("wikitext2", "train")]  # only the dropped entry calibrated again
    again = load(out)
    assert again["layers"] == first["layers"] and again["kv"] == first["kv"]  # deterministic, exactly
    assert len(meta["runs"]) == 2

    # refused: other calibration settings (resume configuration) and other calibration data (windows)
    rc, _, err, _ = table_run(root, out, *CALIB_ARGS, "--calib-seed", "7")
    assert rc == 2 and "differs in ['calibration']" in err, err
    lines = entries.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if '"gptq_int4"' not in line or "down_proj" not in line]
    entries.write_text("".join(kept), encoding="utf-8")
    before = entries.read_bytes()
    rc, _, err, _ = table_run(root, out, *CALIB_ARGS, calib_text=CALIB_TEXT.upper())
    assert rc == 2 and "differ from the other calibrated entries'" in err, err
    after = entries.read_bytes()
    assert after.startswith(before) and after[len(before):].startswith(b'{"run": ')  # the run line only
    assert after[len(before):].count(b"\n") == 1
    assert not (tmp_path / "t.lock").exists()


# ---------- KL windows vs calibration windows (--split, overlap) ----------
def random_text(seed: int, length: int) -> str:
    """Non-periodic text: every 64-character window occurs once, so a window's position can be found back."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,"
    return "".join(rng.choice(alphabet) for _ in range(length))


SPLITS = {"train": random_text(1, 420), "validation": random_text(2, 2400)}
SMALL = ["--device", "cpu", "--windows", "4", "--seqlen", "64"]


def drawn(split: str, samples: int, seqlen: int, seed: int) -> list[torch.Tensor]:
    """The windows tricast.calibration._windows draws from that injected split."""
    return tricast.calibration._windows(ByteTokenizer(), [SPLITS[split]], None, samples, seqlen, seed)


def fingerprint(windows: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for ids in windows:
        digest.update(ids.cpu().contiguous().numpy().astype("<i8").tobytes())
    return digest.hexdigest()


def one_layer_gptq(folder: Path, calibration: dict[str, Any]) -> Path:
    grid = qr.uniform("bf16")
    grid[0][qr.KINDS.index("o_proj")] = "gptq_int4"
    at.write_grid(folder, "one_gptq", "gptq_int4", grid, "none", qr.check_calibration(calibration))
    return folder / "recipe.yaml"


def test_identical_kl_and_calibration_windows_are_refused_before_any_model_loads(tmp_path: Path) -> None:
    def no_model(*args: object) -> tuple[object, object]:
        raise AssertionError("the model must not load")

    recipe = one_layer_gptq(tmp_path / "g", {"samples": 4, "seqlen": 64, "seed": 42})
    text = tmp_path / "text.txt"
    text.write_text(TEXT, encoding="utf-8")
    cases = [(["table", "--calibrate", "--methods", "bf16,gptq_int4", *SMALL, "--calib-samples", "4",
               "--calib-seqlen", "64"], "the KL windows"),  # table seed 42 = calibration seed 42
             (["measure", "--calibrate", *SMALL, "--seed", "42", str(recipe)], "the KL windows"),
             (["measure", "--calibrate", *SMALL, "--seed", "43", "--calib-seed", "43", str(recipe)],
              "the KL windows"),  # the override makes them identical
             (["table", *SMALL, "--split", "validation", "--text", str(text)], "--split chooses")]
    for argv, words in cases:
        rc, _, err = run([*argv, "--out", str(tmp_path / "o.json")], model_factory=no_model)
        assert rc == 2 and words in err, (argv, err)
        if words == "the KL windows":
            assert "are the calibration windows" in err and "'split': 'train'" in err


def test_calibration_overlap_matches_a_brute_force_recount(tmp_path: Path) -> None:
    """Same split (train), different seeds: table and measure count the tokens the KL windows (4 x 64, seed
    43) share with the calibration windows (6 x 64, seed 7) exactly as a recount of the windows tricast draws,
    found back in the token stream (offsets included); the default split stays v1's source."""
    pytest.importorskip("transformers")
    root = make_root(tmp_path)
    stream = ByteTokenizer()(SPLITS["train"])["input_ids"][0].tolist()

    def found(samples: int, seed: int) -> tuple[str, set[int]]:
        """sha256 of the windows' start offsets found back in the stream, and the positions they cover."""
        starts = []
        for window in drawn("train", samples, 64, seed):
            hits = [s for s in range(len(stream) - 63) if stream[s:s + 64] == window[0].tolist()]
            assert len(hits) == 1
            starts.append(hits[0])
        covered = {position for start in starts for position in range(start, start + 64)}
        return hashlib.sha256(json.dumps(starts).encode()).hexdigest(), covered

    (kl_offsets, kl_covered), (cal_offsets, cal_covered) = found(4, 43), found(6, 7)
    expected = len(kl_covered & cal_covered)
    assert 0 < expected < 4 * 64
    kl_fp, cal_fp = fingerprint(drawn("train", 4, 64, 43)), fingerprint(drawn("train", 6, 64, 7))
    with spies(splits=SPLITS):
        rc, _, err = run(["table", "--model", str(root / "model"), *SMALL, "--seed", "43", "--calibrate",
                          "--methods", "bf16,gptq_int4", "--blocks", "1", "--calib-samples", "6",
                          "--calib-seqlen", "64", "--calib-seed", "7", "--out", str(tmp_path / "t.json")],
                         model_factory=tiny_llama)
        assert rc == 0, err
        recipe = one_layer_gptq(tmp_path / "g", {"samples": 6, "seqlen": 64, "seed": 7})
        rc, _, err = run(["measure", "--model", str(root / "model"), *SMALL, "--calibrate",
                          "--out", str(tmp_path / "kl.json"), str(recipe)], model_factory=tiny_llama)
        assert rc == 0, err
    meta = load(tmp_path / "t.json")["meta"]
    measured = load(tmp_path / "kl.json")
    assert meta["source"] == measured["meta"]["source"] == {"dataset": "wikitext2", "split": "train"}  # as v1
    assert meta["dataset_fingerprint"] == measured["meta"]["dataset_fingerprint"] == kl_fp
    for overlap in (meta["calibration_overlap"], measured["results"][0]["calibration_overlap"]):
        assert overlap["calibration_overlap_tokens"] == expected
        assert overlap["disjoint_by_split"] is False and overlap["disjoint_by_dataset"] is False
        assert overlap["stream_tokens"] == len(stream)
        kl, cal = overlap["kl_windows"], overlap["calibration_windows"]
        assert (kl["dataset_fingerprint"], kl["offsets_sha256"]) == (kl_fp, kl_offsets)
        assert (cal["dataset_fingerprint"], cal["offsets_sha256"]) == (cal_fp, cal_offsets)
    assert meta["calibration_overlap"] == measured["results"][0]["calibration_overlap"]
    assert meta["calibration_windows"]["dataset_fingerprint"] == cal_fp
    assert measured["results"][0]["calibration"]["dataset_fingerprint"] == cal_fp


def test_validation_split_runs_end_to_end(tmp_path: Path) -> None:
    """--split validation: table (with GPTQ calibrated on train) -> solve -> predict -> measure --calibrate;
    the KL windows are validation's, the calibration windows train's, and the overlap is 0 by split."""
    pytest.importorskip("transformers")
    root = make_root(tmp_path)
    common = ["--model", str(root / "model"), *LLAMA_ARGS, "--split", "validation"]
    with spies(splits=SPLITS) as spy:
        rc, _, err = run(["table", *common, "--calibrate", "--methods", "bf16,nvfp4,gptq_int4",
                          "--blocks", "1", "--calib-samples", "4", "--calib-seqlen", "64",
                          "--out", str(tmp_path / "table.json")], model_factory=tiny_llama)
        assert rc == 0, err
    assert spy["datasets"] == [("wikitext2", "validation")] + [("wikitext2", "train")] * 7  # KL, then 7 GPTQ
    table = load(tmp_path / "table.json")
    meta = table["meta"]
    assert meta["source"] == {"dataset": "wikitext2", "split": "validation"}
    assert meta["dataset_fingerprint"] == fingerprint(drawn("validation", 2, 192, 42))
    overlap = meta["calibration_overlap"]
    assert overlap["calibration_overlap_tokens"] == 0 and overlap["disjoint_by_split"] is True
    assert (overlap["kl_windows"]["split"], overlap["calibration_windows"]["split"]) == ("validation",
                                                                                        "train")
    assert all(math.isfinite(row["kl"]["gptq_int4"]) and row["floor"] > 0 for row in table["layers"])

    out = tmp_path / "auto"
    rc, _, err = run(["solve", "--table", str(tmp_path / "table.json"), "--budget-bits", "6",
                      "--out", str(out)])
    assert rc == 0, err
    rc, _, err = run(["predict", "--table", str(tmp_path / "table.json"), str(out / "grid.json")])
    assert rc == 0, err
    recipe = one_layer_gptq(tmp_path / "g", {"samples": 4, "seqlen": 64})
    with spies(splits=SPLITS):
        rc, _, err = run(["measure", *common, "--seed", "43", "--calibrate",
                          "--out", str(tmp_path / "kl.json"), str(out / "recipe.yaml"), str(recipe)],
                         model_factory=tiny_llama)
        assert rc == 0, err
    measured = load(tmp_path / "kl.json")
    assert measured["meta"]["source"] == {"dataset": "wikitext2", "split": "validation"}
    assert measured["meta"]["dataset_fingerprint"] == fingerprint(drawn("validation", 2, 192, 43))
    for result in measured["results"]:
        assert math.isfinite(result["kl"]) and result["kl"] > 0
        if result["calibrated"]:
            assert result["calibration_overlap"]["calibration_overlap_tokens"] == 0
            assert result["calibration_overlap"]["disjoint_by_split"] is True
        else:
            assert result["calibration_overlap"] is None
    assert measured["results"][1]["calibrated"] is True
