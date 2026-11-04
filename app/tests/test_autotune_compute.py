"""Track C's compute budget (protocol §3): the B200 GEMM class of every method, the two-constraint knapsack
against brute force, and ``solve --compute-budget`` on a two-block Qwen-shaped table. CPU only."""

from __future__ import annotations

import contextlib
import io
import json
import math
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from app import autotune as at
from app import quantizer_recipe as qr
from app.tests.test_autotune import QWEN, from_json

FP4 = ("nvfp4", "mxfp4", "awq_nvfp4", "smoothquant_nvfp4", "had_nvfp4", "had_mxfp4")
UNRATED = ("nvint4", "bfp12", "razer_nvfp4")


def brute_force2(costs: list[list[int]], compute: list[list[int]], values: list[list[float]], budget: int,
                 compute_budget: int) -> float:
    """The least sum of values over every choice within both integer budgets (inf if none), added layer by
    layer as the DP does."""
    total_cost, total_compute = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    total_value = np.zeros(1)
    for cost_row, compute_row, value_row in zip(costs, compute, values, strict=True):
        total_cost = (total_cost[:, None] + np.asarray(cost_row, dtype=np.int64)[None, :]).ravel()
        total_compute = (total_compute[:, None] + np.asarray(compute_row, dtype=np.int64)[None, :]).ravel()
        total_value = (total_value[:, None] + np.asarray(value_row)[None, :]).ravel()
    fits = (total_cost <= budget) & (total_compute <= compute_budget)
    return float(total_value[fits].min()) if fits.any() else math.inf


def test_every_method_has_its_b200_gemm_class() -> None:
    for method, spec in qr.PYTHON_METHODS.items():
        expected = ("bf16" if spec is None or spec["activation"] is None else "fp4" if method in FP4
                    else "fp8" if method == "mxfp8" else "int8" if method == "int8" else None)
        assert at.compute_class(method) == expected, method
    assert {method for method in qr.PYTHON_METHODS if at.compute_class(method) is None} == set(UNRATED)
    assert at.compute_class("int8_w") == at.compute_class("gptq_int2") == "bf16"  # A16: a BF16 GEMM
    weights = at.COMPUTE_WEIGHT
    assert weights["bf16"] == 2 * weights["fp8"] == 2 * weights["int8"] == 4 * weights["fp4"] == 1
    assert "nvidia.com/en-us/data-center/hgx" in at.COMPUTE_SOURCE and "Dense is 1/2" in at.COMPUTE_SOURCE
    layer = QWEN[0]
    assert at.compute_cost(layer, "nvfp4") == Fraction(layer.params, 4)
    with pytest.raises(at.AutotuneError, match="nvint4 has no sourced B200 Tensor Core rate"):
        at.compute_cost(layer, "nvint4")


def test_two_constraint_solver_equals_brute_force_on_random_instances() -> None:
    """Exact instances (default resolutions, both bounds 0): the brute-force optimum. Coarse instances:
    between the optimum for both budgets and the optimum for both budgets minus their bounds. Never over
    either."""
    rng = random.Random(7)
    exact = coarse = 0
    for t in range(80):
        n, k = rng.randint(1, 6), rng.randint(1, 5)
        values = [[round(rng.random(), 3) for _ in range(k)] for _ in range(n)]
        ints = [[rng.randint(1, 60) for _ in range(k)] for _ in range(n)]
        cints = [[rng.randint(0, 40) for _ in range(k)] for _ in range(n)]
        costs = [[Fraction(c) for c in row] for row in ints]
        compute = [[Fraction(c) for c in row] for row in cints]
        step, cstep = (None, None) if t % 2 == 0 else (rng.choice([2, 3, 5]), rng.choice([2, 4]))
        slack = 0 if step is None else n * max(step, cstep)  # the rounded cheapest choice still fits
        budget = rng.randint(sum(min(row) for row in ints) + slack, sum(max(row) for row in ints) + slack)
        compute_budget = rng.randint(sum(min(row) for row in cints) + slack,
                                     sum(max(row) for row in cints) + slack)
        best = brute_force2(ints, cints, values, budget, compute_budget)
        if not math.isfinite(best) and step is None:
            with pytest.raises(at.AutotuneError, match="no choice fits both budgets"):
                at.solve_mckp2(costs, compute, values, budget, compute_budget)
            continue
        try:
            solution = at.solve_mckp2(costs, compute, values, budget, compute_budget, step, cstep)
        except at.AutotuneError as error:  # rounding left no joint choice; only the coarse case may do that
            assert step is not None and "no choice fits both budgets" in str(error), t
            continue
        assert solution.cost == sum(costs[i][j] for i, j in enumerate(solution.choice)) <= budget
        assert solution.compute == sum(compute[i][j] for i, j in enumerate(solution.choice)) <= compute_budget
        if step is None:
            assert solution.bound == solution.compute_bound == 0
            assert solution.value == best, t
            exact += 1
        else:
            high = brute_force2(ints, cints, values, math.floor(budget - solution.bound),
                                math.floor(compute_budget - solution.compute_bound))
            assert best <= solution.value <= high, t
            coarse += 1
    assert exact >= 25 and coarse >= 15


def test_a_slack_compute_budget_gives_the_one_constraint_optimum() -> None:
    rng = random.Random(3)
    for _ in range(30):
        n, k = rng.randint(1, 6), rng.randint(2, 5)
        values = [[round(rng.random(), 3) for _ in range(k)] for _ in range(n)]
        costs = [[Fraction(rng.randint(1, 50)) for _ in range(k)] for _ in range(n)]
        compute = [[Fraction(rng.randint(1, 9)) for _ in range(k)] for _ in range(n)]
        budget = rng.randint(int(sum(min(row) for row in costs)), int(sum(max(row) for row in costs)))
        loose = sum(max(row) for row in compute)
        assert at.solve_mckp2(costs, compute, values, budget, loose).value == at.solve_mckp(costs, values,
                                                                                           budget).value


def test_two_constraint_ties_and_refusals() -> None:
    assert at.solve_mckp2([[1, 1]], [[2, 1]], [[0.0, 0.0]], 5, 5).choice == [1]  # same cost: less compute
    assert at.solve_mckp2([[2, 1]], [[1, 2]], [[0.0, 0.0]], 5, 5).choice == [1]  # the smaller cost first
    assert at.solve_mckp2([[1, 1, 1]], [[1, 1, 1]], [[0.5, 0.5, 0.5]], 5, 5).choice == [0]  # the earliest
    with pytest.raises(at.AutotuneError, match="infeasible compute budget 1: the cheapest choice costs 2"):
        at.solve_mckp2([[1]], [[2]], [[0.0]], 5, 1)
    with pytest.raises(at.AutotuneError, match="no choice fits both budgets"):  # each fits alone, not both
        at.solve_mckp2([[1, 5]], [[5, 1]], [[0.0, 0.0]], 2, 2)
    rows = [[Fraction(0), Fraction(1 << 20)]] * 300  # the DP spans the costs above each row's cheapest
    with pytest.raises(at.AutotuneError, match="is too large; use a resolution of at least"):
        at.solve_mckp2(rows, [[1, 2]] * 300, [[0.0, 0.0]] * 300, 300 << 20, 600, resolution=1 << 10)


def two_block_table(path: Path) -> tuple[Path, list[at.Layer]]:
    rng = random.Random(1)
    layers = [layer for layer in QWEN if layer.block < 2]
    rows = [{"name": layer.name, "block": layer.block, "kind": layer.kind, "params": layer.params,
             "in_features": layer.in_features,
             "kl": {m: 0.0 if m == "bf16" else rng.random() * 1e-3 for m in at.FREE}} for layer in layers]
    path.write_text(json.dumps({"schema": at.TABLE_SCHEMA, "status": "complete", "objective": at.OBJECTIVE,
                                "meta": {"methods": list(at.FREE), "model_blocks": 2}, "layers": rows}),
                    encoding="utf-8")
    return path, layers


def solve(table: Path, out: Path, *extra: str) -> tuple[int, str, str]:
    stdout, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(err):
        rc = at.main(["solve", "--table", str(table), "--out", str(out), *extra])
    return rc, stdout.getvalue(), err.getvalue()


def test_solve_with_a_compute_budget_keeps_both_budgets_and_records_them(tmp_path: Path) -> None:
    table, layers = two_block_table(tmp_path / "table.json")
    args = ("--budget-bits", "6", "--resolution", str(1 << 18), "--compute-budget", "3/8")
    rc, _, err = solve(table, tmp_path / "refused", *args)
    assert rc == 2 and "['nvint4', 'bfp12'] have no sourced B200 Tensor Core rate" in err
    assert not (tmp_path / "refused").exists()
    rc, stdout, _ = solve(table, tmp_path / "c", *args, "--drop-unrated")
    assert rc == 0 and stdout.splitlines()[-1].startswith("AUTOTUNE_SOLVE_DONE name=auto_6_c3-8 ")
    solution = json.loads((tmp_path / "c" / "solution.json").read_text())
    chosen = [row["method"] for row in solution["layers"]]
    assert not set(chosen) & set(UNRATED)
    params = sum(layer.params for layer in layers)
    spent = sum((at.compute_cost(layer, method) for layer, method in zip(layers, chosen, strict=True)),
                Fraction(0))
    compute = solution["compute"]
    assert compute["achieved"] == float(spent / params) <= 3 / 8
    assert compute["dropped_unrated"] == ["nvint4", "bfp12"] and compute["source"] == at.COMPUTE_SOURCE
    assert compute["classes"]["mxfp8"] == "fp8"
    assert compute["achieved_x_uniform_fp4"] == compute["achieved"] * 4
    assert solution["achieved"]["avg_bits"] <= 6
    assert "then the smallest rounded compute" in solution["tie_rules"]["solver"]
    recipe = (tmp_path / "c" / "recipe.yaml").read_text(encoding="utf-8")
    assert from_json(tmp_path / "c" / "grid.json") == recipe
    # the default resolutions are exact (both bounds 0) and no worse than the coarse bits resolution
    assert solve(table, tmp_path / "exact", "--budget-bits", "6", "--compute-budget", "3/8",
                 "--drop-unrated")[0] == 0
    exact = json.loads((tmp_path / "exact" / "solution.json").read_text())
    assert exact["resolution"]["bound_bits_params"] == 0
    assert exact["compute"]["resolution"]["bound_compute_params"] == 0
    assert solution["resolution"]["bound_bits_params"] > 0
    assert exact["predicted_kl_sum"] <= solution["predicted_kl_sum"]
    # uniform-FP4 compute leaves only FP4 GEMMs; without a compute budget nothing changes
    assert solve(table, tmp_path / "fp4", *args[:4], "--compute-budget", "1/4", "--drop-unrated")[0] == 0
    fp4 = json.loads((tmp_path / "fp4" / "solution.json").read_text())
    assert {row["method"] for row in fp4["layers"]} <= {"nvfp4", "mxfp4"}
    assert solve(table, tmp_path / "plain", "--budget-bits", "6")[0] == 0
    plain = json.loads((tmp_path / "plain" / "solution.json").read_text())
    assert plain["compute"] is None and plain["name"] == "auto_6"
    assert "compute" not in plain["tie_rules"]["solver"]
