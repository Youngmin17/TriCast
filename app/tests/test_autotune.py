"""app/autotune.py, Auto-tune v1 (docs/spikes/2026-10-13_autotune_mckp.md): the knapsack solver against brute
force (C1), the deck's Linear average bits, solved grids through the Quantizer generator and tricast's own
override resolution, and a tiny random Qwen3 end to end on CPU (table determinism and resume, solve, predict,
measure). No GPU, no Qwen3-0.6B, no dataset download: the tiny model and an inline text are injected the way
test_quantizer_runner.py does.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import random
import socket
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import tricast
from app import autotune as at
from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr
from tricast.recipe import load_recipe

REPO = Path(qr.__file__).resolve().parents[1]
DECODER = ("model.layers", qr.BLOCKS)
QWEN = [at.Layer(qr.layer_name(b, k), b, k, ex.SHAPES[k][0] * ex.SHAPES[k][1], ex.SHAPES[k][0])
        for b in range(qr.BLOCKS) for k in qr.KINDS]  # Qwen3-0.6B's 196 Linear layers
ALONE = {m: load_recipe({"name": m, "defaults": dict(f)}).defaults for m, f in qr.METHODS.items() if f}
TEXT = " ".join(f"token{i % 37} word{i % 11}." for i in range(120))
TINY_ARGS = ["--device", "cpu", "--windows", "2", "--seqlen", "32"]


# ---------- C1: the solver against brute force ----------
def brute_force(costs: list[list[int]], values: list[list[float]], budget: int) -> float:
    """The least sum of values over every choice whose integer cost sum is <= budget (inf if none), adding
    values layer by layer as the DP does."""
    total_cost, total_value = np.zeros(1, dtype=np.int64), np.zeros(1)
    for cost_row, value_row in zip(costs, values, strict=True):
        total_cost = (total_cost[:, None] + np.asarray(cost_row, dtype=np.int64)[None, :]).ravel()
        total_value = (total_value[:, None] + np.asarray(value_row)[None, :]).ravel()
    fits = total_cost <= budget
    return float(total_value[fits].min()) if fits.any() else math.inf


def check_choice(solution: at.Solution, costs: list[list[Fraction]], values: list[list[float]],
                 budget: Fraction) -> None:
    assert solution.cost == sum(costs[i][j] for i, j in enumerate(solution.choice)) <= budget
    value = 0.0
    for i, j in enumerate(solution.choice):
        value += values[i][j]
    assert solution.value == value


def test_solver_equals_brute_force_on_random_instances() -> None:
    """Exact instances (real WEIGHT_BITS costs, default resolution = their GCD, bound 0): the DP's value is
    the brute-force optimum. Coarse instances (integer costs, resolution 3-60): it lies between the optimum
    for the budget and the optimum for the budget minus the reported bound. Never over the budget."""
    rng = random.Random(42)
    exact = coarse = 0
    for t in range(64):
        n = rng.randint(1, 8)
        values = [[0.0 if m == "bf16" else round(rng.random(), 2) * 1e-3 for m in at.FREE] for _ in range(n)]
        if t % 2 == 0:  # params a multiple of 3072, so every cost is a whole multiple of 32 bits
            layers = [at.Layer(f"l{i}", 0, "q_proj", 3072 * rng.randint(1, 64), rng.choice([64, 1024, 3072]))
                      for i in range(n)]
            costs = [[at.cost(layer, m) for m in at.FREE] for layer in layers]
            params = sum(layer.params for layer in layers)
            budget = Fraction(rng.uniform(4.25, 16.5)) * params
            solution = at.solve_mckp(costs, values, budget)
            scale = math.lcm(*(c.denominator for row in costs for c in row))
            assert solution.bound == 0
            assert solution.value == brute_force([[int(c * scale) for c in row] for row in costs], values,
                                                 math.floor(budget * scale)), t
            exact += 1
        else:
            ints = [[rng.randint(1, 1000) for _ in at.FREE] for _ in range(n)]
            costs = [[Fraction(c) for c in row] for row in ints]
            step = rng.choice([3, 7, 25, 60])
            # at least n steps above the cheapest choice, so the rounded cheapest choice still fits
            low_budget = sum(min(row) for row in ints) + n * step
            budget = Fraction(rng.randint(low_budget, max(low_budget, sum(max(row) for row in ints))))
            solution = at.solve_mckp(costs, values, budget, resolution=step)
            assert solution.bound >= 0
            low = brute_force(ints, values, int(budget))
            high = brute_force(ints, values, math.floor(budget - solution.bound))
            assert low <= solution.value <= high, t
            coarse += 1
        check_choice(solution, costs, values, budget)
    assert exact >= 25 and coarse >= 25


def test_infeasible_budgets_are_refused(tmp_path: Path) -> None:
    costs = [[Fraction(3), Fraction(5)], [Fraction(2), Fraction(9)]]
    with pytest.raises(ValueError, match="infeasible budget 4: the cheapest choice costs 5"):
        at.solve_mckp(costs, [[1.0, 0.0], [1.0, 0.0]], 4)
    with pytest.raises(ValueError, match="fits only before rounding costs up to multiples of 4"):
        at.solve_mckp([[Fraction(5)]], [[0.0]], 6, resolution=4)  # 5 rounds up to 8 > 6
    table = write_qwen_table(tmp_path / "table.json")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = at.main(["solve", "--table", str(table), "--budget-bits", "4.2", "--out", str(tmp_path / "s")])
    assert rc == 2
    assert "4.2 bits is below the cheapest assignment of these methods, 4.25 bits" in err.getvalue()
    assert not (tmp_path / "s").exists()


def test_tie_rules_pick_the_smaller_cost_then_the_earlier_option() -> None:
    assert at.solve_mckp([[2, 1]], [[0.0, 0.0]], 5).choice == [1]  # same value: the smaller total cost
    assert at.solve_mckp([[1, 1, 1]], [[0.5, 0.5, 0.5]], 5).choice == [0]  # same value and cost: the earliest
    # [0, 0] and [1, 1] both cost 3 with value 0.1 (budget 3): the last row decides first, earliest option
    assert at.solve_mckp([[1, 2], [2, 1]], [[0.1, 0.0], [0.0, 0.1]], 3).choice == [0, 0]
    # the grid default: most frequent non-BF16 choice, ties to the earlier method in METHODS
    assert at.default_method(["mxfp8", "nvfp4", "mxfp8", "nvfp4", "bf16", "bf16", "bf16"], at.FREE) == "nvfp4"
    assert at.default_method(["bf16"], at.FREE) == "nvfp4"
    assert at.default_method(["bf16"], ["bf16", "int8"]) == "int8"


# ---------- sizes ----------
def test_linear_average_bits_are_the_decks() -> None:
    def preset_bits(preset_id: str) -> Fraction:
        grid = qr.PRESETS[preset_id][1]
        return at.avg_bits(QWEN, [grid[layer.block][qr.KINDS.index(layer.kind)] for layer in QWEN])

    assert abs(float(preset_bits("qd_mixed")) - 5.722) <= 0.001
    assert preset_bits("qd_nvfp4") == Fraction(9, 2) and preset_bits("qd_mxfp4") == Fraction(17, 4)
    for method in qr.METHODS:  # the shared WEIGHT_BITS function, per Qwen3-0.6B kind
        for kind in qr.KINDS:
            assert abs(float(at.weight_bits(method, ex.SHAPES[kind][0])) - ex._bits(method, kind)) <= 1e-12


# ---------- solved grids -> Quantizer recipe -> tricast ----------
def write_qwen_table(path: Path, seed: int = 0) -> Path:
    """A table shaped like the Qwen3-0.6B one, with seeded random KL (BF16 0)."""
    rng = random.Random(seed)
    layers = [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
               "in_features": layer.in_features,
               "kl": {m: 0.0 if m == "bf16" else rng.random() * 1e-3 for m in at.FREE}} for layer in QWEN]
    path.write_text(json.dumps({"schema": at.TABLE_SCHEMA, "status": "complete", "objective": at.OBJECTIVE,
                                "meta": {"methods": list(at.FREE), "model_blocks": qr.BLOCKS},
                                "layers": layers}), encoding="utf-8")
    return path


def from_json(path: Path) -> str:
    return subprocess.run([sys.executable, "-m", "app.quantizer_recipe", "--from-json", str(path)], cwd=REPO,
                          capture_output=True, text=True, check=True, encoding="utf-8").stdout


@pytest.mark.parametrize("budget", [["--budget-bits", "4.25"], ["--budget-bits", "4.50"],
                                    ["--budget-bits", "5.09"], ["--budget-bits", "5.72", "--kv", "kivi2"],
                                    ["--budget-fraction", "0.3"], ["--budget-bits", "16"]])
def test_solved_grid_resolves_to_each_chosen_methods_own_spec(tmp_path: Path, budget: list[str]) -> None:
    table_path = write_qwen_table(tmp_path / "table.json")
    out = tmp_path / "auto"
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        assert at.main(["solve", "--table", str(table_path), *budget, "--out", str(out)]) == 0
    assert stdout.getvalue().splitlines()[-1].startswith("AUTOTUNE_SOLVE_DONE name=auto_")
    spec = json.loads((out / "grid.json").read_text(encoding="utf-8"))
    solution = json.loads((out / "solution.json").read_text(encoding="utf-8"))
    text = (out / "recipe.yaml").read_text(encoding="utf-8")
    assert from_json(out / "grid.json") == text  # byte for byte
    data = qr.build(spec["name"], spec["default"], spec["grid"], spec["kv"])
    assert yaml.safe_load(text) == data
    assert qr.resolve(spec["default"], data.get("overrides", [])) == spec["grid"]
    recipe = load_recipe(data)
    chosen = [entry["method"] for entry in solution["layers"]]
    assert [entry["name"] for entry in solution["layers"]] == [layer.name for layer in QWEN]
    for layer, method in zip(QWEN, chosen, strict=True):
        assert spec["grid"][layer.block][qr.KINDS.index(layer.kind)] == method
        assert recipe.spec_for(layer.name, decoder=DECODER) == (None if method == "bf16" else ALONE[method])
    for method in at.FREE[1:]:  # what the table measured each method with is that same spec
        assert at.method_recipe(method).defaults == ALONE[method]
    bits = Fraction(budget[1]) * (16 if budget[0] == "--budget-fraction" else 1)
    achieved = at.avg_bits(QWEN, chosen)
    assert achieved <= bits and solution["achieved"]["avg_bits"] == float(achieved)
    assert solution["achieved"]["size_pct_of_bf16"] == float(achieved / 16 * 100)
    assert solution["budget"]["avg_bits"] == float(bits) and solution["resolution"]["bound_bits_params"] == 0
    assert solution["table"]["sha256"] == hashlib.sha256(table_path.read_bytes()).hexdigest()
    table = json.loads(table_path.read_text(encoding="utf-8"))
    assert solution["predicted_kl_sum"] == at.predict(table, QWEN, list(at.FREE), spec["grid"])[0]
    counts = {m: chosen.count(m) for m in at.FREE if m != "bf16" and chosen.count(m)}
    assert spec["default"] == (max(counts, key=lambda m: (counts[m], -list(qr.METHODS).index(m))) if counts
                               else "nvfp4")
    assert spec["kv"] == (budget[3] if "--kv" in budget else "none")
    if budget[1] == "4.25":  # only MXFP4 fits: the uniform grid, i.e. the qd_mxfp4 layers
        assert spec["grid"] == qr.uniform("mxfp4") and "overrides" not in data
    if budget[1] == "16":  # everything fits: all BF16 (KL 0)
        assert set(chosen) == {"bf16"} and solution["predicted_kl_sum"] == 0


def test_predict_sums_the_table_and_refuses_methods_it_lacks(tmp_path: Path) -> None:
    table_path = write_qwen_table(tmp_path / "table.json")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    out = tmp_path / "predict.json"
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        assert at.main(["predict", "--table", str(table_path), "qd_nvfp4", "qd_mxfp4_ends_bf16",
                        "--out", str(out)]) == 0
    assert stdout.getvalue().splitlines()[-1] == "AUTOTUNE_PREDICT_DONE n=2"
    results = json.loads(out.read_text(encoding="utf-8"))["results"]
    nvfp4 = 0.0
    for entry in table["layers"]:
        nvfp4 += entry["kl"]["nvfp4"]
    ends = 0.0
    for entry in table["layers"]:
        ends += entry["kl"]["bf16" if entry["block"] in (0, qr.BLOCKS - 1) else "mxfp4"]
    assert [r["kl_sum"] for r in results] == [nvfp4, ends]
    assert [r["avg_bits"] for r in results] == [4.5, float(Fraction(285, 56))]  # (26 x 4.25 + 2 x 16) / 28
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        assert at.main(["predict", "--table", str(table_path), "qd_mixed"]) == 2
    assert "['awq_nvfp4', 'gptq_int4'], which the table has no KL for" in err.getvalue()
    env = json.loads(out.read_text(encoding="utf-8"))["env"]  # predict's own run, next to the table's sha
    assert env["git_sha"] and env["git_dirty"] is not None and env["hostname"] and env["utc"]
    assert env["versions"]["torch"] and len(env["app_sha256"]) == 3 and env["tricast_path"]


@pytest.mark.parametrize("extra", [{"options": {"nvfp4": {"weight.rounding": "rtz"}}},
                                   {"kv": "kivi2", "kv_options": {"residual": 256}},
                                   {"calibration": {"samples": 16}}])
def test_predict_refuses_grids_with_options_the_table_has_no_entries_for(tmp_path: Path,
                                                                         extra: dict) -> None:
    table_path = write_qwen_table(tmp_path / "table.json")
    path = tmp_path / "grid.json"
    spec = {"name": "opt", "default": "nvfp4", "grid": qr.uniform("nvfp4"), "kv": "none", **extra}
    path.write_text(json.dumps(spec), encoding="utf-8")
    from_json(path)  # a valid Quantizer spec
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        assert at.main(["predict", "--table", str(table_path), str(path)]) == 2
    key = next(key for key in ("options", "kv_options", "calibration") if key in extra)
    words = f"opt sets ['{key}']; the v1 table has entries only for the methods without options"
    assert words in err.getvalue()
    path.write_text(json.dumps({**spec, key: {}}), encoding="utf-8")  # empty: nothing ignored, accepted
    with contextlib.redirect_stdout(io.StringIO()):
        assert at.main(["predict", "--table", str(table_path), str(path)]) == 0


def test_default_names_encode_the_exact_budget_and_kv(tmp_path: Path) -> None:
    table_path = write_qwen_table(tmp_path / "table.json")
    cases = [(["--budget-bits", "285/56"], "auto_285-56"),  # qd_mxfp4_ends_bf16's exact size
             (["--budget-bits", "5.0892857"], "auto_50892857-10000000"),  # both were auto_5.08929
             (["--budget-bits", "5.0892857", "--kv", "kivi2"], "auto_50892857-10000000_kivi2"),
             (["--budget-bits", "4.50"], "auto_9-2"), (["--budget-fraction", "0.28125"], "auto_9-2"),
             (["--budget-bits", "16", "--kv", "kv_fp8"], "auto_16_kv_fp8")]
    for index, (budget, name) in enumerate(cases):
        out = tmp_path / f"s{index}"
        with contextlib.redirect_stdout(io.StringIO()):
            assert at.main(["solve", "--table", str(table_path), *budget, "--out", str(out)]) == 0
        assert json.loads((out / "grid.json").read_text(encoding="utf-8"))["name"] == name
        assert json.loads((out / "solution.json").read_text(encoding="utf-8"))["name"] == name
        assert f"\nname: {name}\n" in (out / "recipe.yaml").read_text(encoding="utf-8")
    assert len({name for _, name in cases}) == len(cases) - 1  # only the same exact budget (9/2) shares one


def test_measure_refuses_a_kv_cache_mode_recipe_before_the_model_loads(tmp_path: Path) -> None:
    data = qr.build("cache", "nvfp4", qr.uniform("nvfp4"), "kivi2")
    data["kv"]["mode"] = "cache"
    path = tmp_path / "cache.yaml"
    path.write_text(qr.to_yaml(data) + "\n", encoding="utf-8")
    load_recipe(path)  # a valid tricast recipe

    def no_model(*args: object) -> tuple[object, object]:
        raise AssertionError("the model must not load")

    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert at.main(["measure", "--out", str(tmp_path / "kl.json"), str(path), *TINY_ARGS],
                       model_factory=no_model) == 2
    assert "kv mode 'cache' streams token by token; use mode fakequant" in err.getvalue()


def test_calibrated_methods_and_recipes_are_refused_before_any_model_loads(tmp_path: Path) -> None:
    mixed = tmp_path / "qd_mixed.yaml"
    mixed.write_text(qr.to_yaml(qr.preset("qd_mixed")) + "\n", encoding="utf-8")

    def no_model(*args: object) -> tuple[object, object]:
        raise AssertionError("the model must not load")

    for argv, words in ((["table", "--methods", "bf16,gptq_int4", "--out", str(tmp_path / "t.json")],
                         "['gptq_int4'] need calibration data"),
                        (["measure", "--out", str(tmp_path / "kl.json"), str(mixed)], "needs calibration")):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert at.main([*argv, *TINY_ARGS], model_factory=no_model) == 2
        assert words in err.getvalue()


def test_random_grids_are_seeded_quantizer_grids(tmp_path: Path) -> None:
    for run in ("a", "b"):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            assert at.main(["random", "--seeds", "1,2", "--out", str(tmp_path / run)]) == 0
        assert stdout.getvalue().splitlines()[-1] == "AUTOTUNE_RANDOM_DONE n=2"
    grids = []
    for seed in (1, 2):
        folder = tmp_path / "a" / f"rand_s{seed}"
        twin = tmp_path / "b" / f"rand_s{seed}"
        assert (folder / "grid.json").read_bytes() == (twin / "grid.json").read_bytes()
        spec = json.loads((folder / "grid.json").read_text(encoding="utf-8"))
        rng = random.Random(seed)
        assert spec["grid"] == [[rng.choice(list(at.FREE)) for _ in qr.KINDS] for _ in range(qr.BLOCKS)]
        assert from_json(folder / "grid.json") == (folder / "recipe.yaml").read_text(encoding="utf-8")
        grids.append(spec["grid"])
    assert grids[0] != grids[1]


# ---------- tiny random Qwen3 end to end (CPU) ----------
class ByteTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": torch.tensor([[ord(c) % 128 for c in text]])}


def tiny_qwen3(model_id: str, dtype: str, device: str, revision: str | None) -> tuple[object, object]:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(42)
    cfg = transformers.Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=64)
    return transformers.Qwen3ForCausalLM(cfg).eval(), ByteTokenizer()


def table_main(root: Path, out: Path, *extra: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        rc = at.main(["table", "--model", str(root / "model"), "--text", str(root / "text.txt"), *TINY_ARGS,
                      "--out", str(out), *extra], model_factory=tiny_qwen3)
    return rc, stdout.getvalue(), stderr.getvalue()


def run_table(root: Path, out: Path) -> list[str]:
    rc, stdout, stderr = table_main(root, out)
    assert rc == 0, stderr
    return stdout.splitlines()


def kl_values(path: Path) -> dict[str, dict[str, float]]:
    return {entry["name"]: entry["kl"] for entry in json.loads(path.read_text(encoding="utf-8"))["layers"]}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("autotune")
    (root / "model").mkdir()  # an existing path: capture_env records a local model, no Hub lookup
    (root / "text.txt").write_text(TEXT, encoding="utf-8")
    lines = run_table(root, root / "table.json")
    assert lines[0].startswith("AUTOTUNE_TABLE_START layers=14 methods=7 entries=84 done=0")
    assert sum(line.startswith("[") for line in lines) == 14  # a progress line per layer
    assert lines[-1].startswith("AUTOTUNE_TABLE_DONE layers=14 methods=7 wall_s=")
    return root


def test_tiny_table_is_complete_deterministic_and_resumable(tiny: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    table = json.loads((tiny / "table.json").read_text(encoding="utf-8"))
    first = kl_values(tiny / "table.json")
    assert list(first) == [qr.layer_name(b, k) for b in range(2) for k in qr.KINDS]
    for kl in first.values():
        assert list(kl) == list(at.FREE) and kl["bf16"] == 0.0
        # every quantized layer moves the logits: 0 would mean the one-layer patch did not fire
        assert all(math.isfinite(v) and v > 0 for m, v in kl.items() if m != "bf16")
    meta = table["meta"]
    assert meta["device"] == "cpu" and (meta["windows"], meta["seqlen"], meta["seed"]) == (2, 32, 42)
    assert meta["git_sha"] and meta["git_dirty"] is not None and meta["torch"] and meta["transformers"]
    assert len(meta["dataset_fingerprint"]) == 64 and meta["model_kind"] == "local" and meta["wall_s"] > 0
    assert meta["entries"] == meta["entries_computed_this_run"] == 84
    assert set(meta["method_recipe_sha256"]) == set(at.FREE[1:])
    modules = ("autotune.py", "quantizer_recipe.py", "export_quantizer_demo.py")
    assert meta["app_sha256"] == {f"app/{name}": hashlib.sha256((REPO / "app" / name).read_bytes())
                                  .hexdigest() for name in modules}
    assert meta["tricast_path"] == str(Path(tricast.__file__).resolve().parent)
    assert {"app_sha256", "tricast_path"} <= set(at.RESUME_KEYS)  # a resume after code edits is refused
    assert [run["device"] for run in meta["runs"]] == ["cpu"] and meta["runs"][0]["hostname"]
    assert not list(tiny.glob("*.lock")) and not list(tiny.glob(".*.tmp")) and not list(tiny.glob("*.tmp"))

    run_table(tiny, tmp_path / "again.json")
    assert kl_values(tmp_path / "again.json") == first  # same seed: the same values, exactly

    class Killed(Exception):
        pass

    real, calls = at._entry_kl, []

    def dies_after_ten(*args: object) -> float:
        if len(calls) == 10:
            raise Killed
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(at, "_entry_kl", dies_after_ten)
    with pytest.raises(Killed), contextlib.redirect_stdout(io.StringIO()):
        run_table(tiny, tmp_path / "resumed.json")
    entries = tmp_path / "resumed.entries.jsonl"
    lines = entries.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 1 + 10  # header, the run's record, the finished entries
    assert not (tmp_path / "resumed.json").exists() and not (tmp_path / "resumed.lock").exists()
    with entries.open("a", encoding="utf-8") as log:
        log.write('{"layer": "model.layers.1.mlp.up')  # a line cut off by the kill
    calls.clear()
    monkeypatch.setattr(at, "_entry_kl", lambda *args: calls.append(args) or real(*args))
    lines = run_table(tiny, tmp_path / "resumed.json")
    assert lines[0].startswith("AUTOTUNE_TABLE_START layers=14 methods=7 entries=84 done=10")
    assert len(calls) == 84 - 10  # nothing recomputed
    assert kl_values(tmp_path / "resumed.json") == first
    resumed = json.loads((tmp_path / "resumed.json").read_text(encoding="utf-8"))["meta"]
    assert resumed["entries_computed_this_run"] == 74
    assert len(resumed["runs"]) == 2 and all(run["utc"] and run["hostname"] for run in resumed["runs"])


def test_resume_is_refused_for_another_configuration_code_or_method_recipe(tiny: Path,
                                                                           tmp_path: Path) -> None:
    source = (tiny / "table.entries.jsonl").read_text(encoding="utf-8")

    def tampered(edit: object, *extra: str) -> str:
        entries = tmp_path / "copy.entries.jsonl"
        entries.write_text(edit(source) if callable(edit) else source, encoding="utf-8")
        before = entries.read_bytes()
        rc, _, stderr = table_main(tiny, tmp_path / "copy.json", *extra)
        assert rc == 2 and entries.read_bytes() == before  # refused before anything is written
        assert not (tmp_path / "copy.json").exists() and not (tmp_path / "copy.lock").exists()
        return stderr

    assert "differs in ['dataset_fingerprint', 'seed']" in tampered(None, "--seed", "7")

    def code_edit(text: str) -> str:
        lines = text.splitlines(keepends=True)
        header = json.loads(lines[0])
        header["config"]["app_sha256"]["app/autotune.py"] = "0" * 64  # entries written by other code
        return json.dumps(header) + "\n" + "".join(lines[1:])

    assert "differs in ['app_sha256']" in tampered(code_edit)

    def recipe_edit(text: str) -> str:
        lines = text.splitlines(keepends=True)
        index = next(i for i, line in enumerate(lines) if '"method": "nvfp4"' in line)
        entry = json.loads(lines[index])
        entry["recipe_sha256"] = "0" * 64
        lines[index] = json.dumps(entry) + "\n"
        return "".join(lines)

    assert "the nvfp4 recipe changed since its entries were written" in tampered(recipe_edit)


def test_a_held_lock_refuses_a_table_run_and_a_dead_runs_lock_is_taken_over(tiny: Path,
                                                                            tmp_path: Path) -> None:
    out, lock = tmp_path / "t.json", tmp_path / "t.lock"
    small = ("--methods", "bf16,int8", "--blocks", "1")
    here = socket.gethostname()
    for holder in ({"hostname": "elsewhere", "pid": 1}, {"hostname": here, "pid": os.getpid()}):
        lock.write_text(json.dumps(holder), encoding="utf-8")
        rc, _, stderr = table_main(tiny, out, *small)
        assert rc == 2 and "is held by" in stderr and json.loads(lock.read_text(encoding="utf-8")) == holder
        assert not (tmp_path / "t.entries.jsonl").exists()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock.write_text(json.dumps({"hostname": here, "pid": dead.pid}), encoding="utf-8")
    rc, stdout, stderr = table_main(tiny, out, *small)
    assert rc == 0, stderr
    assert stdout.splitlines()[-1].startswith("AUTOTUNE_TABLE_DONE layers=7 methods=2")
    assert not lock.exists() and out.exists()


def test_tiny_solve_predict_and_measure(tiny: Path, tmp_path: Path) -> None:
    table = json.loads((tiny / "table.json").read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        assert at.main(["solve", "--table", str(tiny / "table.json"), "--budget-bits", "6",
                        "--out", str(tmp_path / "auto")]) == 0
    solution = json.loads((tmp_path / "auto" / "solution.json").read_text(encoding="utf-8"))
    assert from_json(tmp_path / "auto" / "grid.json") == (tmp_path / "auto" / "recipe.yaml").read_text()
    assert solution["achieved"]["avg_bits"] <= 6 and solution["resolution"]["bound_bits_params"] == 0
    assert solution["table_blocks"] == [0, 1] and solution["model_blocks"] == 2
    assert solution["name"] == "auto_6" and solution["env"]["git_sha"] and solution["env"]["hostname"]
    assert solution["env"]["app_sha256"] == table["meta"]["app_sha256"]  # solved by the code that measured
    expected = 0.0
    for entry, chosen in zip(table["layers"], solution["layers"], strict=True):
        assert chosen["kl"] == entry["kl"][chosen["method"]]
        expected += entry["kl"][chosen["method"]]
    assert solution["predicted_kl_sum"] == expected
    with contextlib.redirect_stdout(io.StringIO()):
        assert at.main(["predict", "--table", str(tiny / "table.json"), str(tmp_path / "auto" / "grid.json"),
                        "--out", str(tmp_path / "predict.json")]) == 0
    assert json.loads((tmp_path / "predict.json").read_text("utf-8"))["results"][0]["kl_sum"] == expected

    nvfp4 = tmp_path / "qd_nvfp4.yaml"
    nvfp4.write_text(qr.to_yaml(qr.preset("qd_nvfp4")) + "\n", encoding="utf-8")
    one = qr.uniform("bf16")
    one[1][qr.KINDS.index("v_proj")] = "nvfp4"
    single = tmp_path / "one" / "recipe.yaml"
    at.write_grid(single.parent, "one_v1", "nvfp4", one, "none")
    for seed, recipes in ((43, [tmp_path / "auto" / "recipe.yaml", nvfp4]), (42, [single])):
        out = tmp_path / f"kl_{seed}.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            assert at.main(["measure", "--model", str(tiny / "model"), "--text", str(tiny / "text.txt"),
                            *TINY_ARGS, "--seed", str(seed), "--out", str(out), *map(str, recipes)],
                           model_factory=tiny_qwen3) == 0
        assert stdout.getvalue().splitlines()[-1] == f"AUTOTUNE_MEASURE_DONE n={len(recipes)}"
        measured = json.loads(out.read_text(encoding="utf-8"))
        assert measured["status"] == "complete" and measured["meta"]["seed"] == seed
        assert measured["meta"]["git_sha"] and measured["meta"]["dataset_fingerprint"]
        assert all(math.isfinite(r["kl"]) and r["kl"] >= 0 for r in measured["results"])
        if seed == 43:  # held-out windows; the solved recipe patches its non-BF16 layers, NVFP4 all 14
            assert measured["meta"]["dataset_fingerprint"] != table["meta"]["dataset_fingerprint"]
            quantized = sum(entry["method"] != "bf16" for entry in solution["layers"])
            assert [r["patched"] for r in measured["results"]] == [quantized, 14]
        else:  # the table's windows: a grid with only that layer quantized measures exactly its entry
            assert measured["meta"]["dataset_fingerprint"] == table["meta"]["dataset_fingerprint"]
            assert measured["results"][0]["patched"] == 1
            entry = kl_values(tiny / "table.json")[qr.layer_name(1, "v_proj")]["nvfp4"]
            assert measured["results"][0]["kl"] == entry
