"""Agent-driven auto-tune tools (app/tune_agent.py) — SPEC AC24: proposals are refused unless they build,
fit the budget, cite ledger records and state only supported numbers; neighbours fit the budget; rounds
measure on validation windows; the state joins proposals to measurements by recipe hash. Synthetic table,
CPU only."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from app import autotune as at
from app import export_quantizer_demo as ex
from app import ledger as lg
from app import quantizer_recipe as qr
from app import tune_agent as ta

METHODS = ["bf16", "nvfp4", "nvint4", "mxfp4", "mxfp8", "int8", "bfp12"]
ENV = {"git_sha": "73f7893", "git_dirty": False, "gpu_names": ["NVIDIA A100 80GB PCIe"],
       "model_id": "Qwen/Qwen3-0.6B", "model_sha": "c1899de"}


def _table(tmp_path: Path, floor: float | None = None) -> Path:
    """An auto-tune table over Qwen3-0.6B's 196 Linear shapes with seeded KL values (4-bit > 8-bit > 0): v1,
    or v2 with every layer's patch floor = ``floor`` and kivi2 KV rows."""
    rng = random.Random(7)
    layers = []
    for block in range(qr.BLOCKS):
        for kind in qr.KINDS:
            out_features, in_features = ex.SHAPES[kind]
            base = rng.uniform(0.001, 0.01)
            kl = {"bf16": 0.0, "nvfp4": base, "nvint4": base * rng.uniform(0.9, 1.2),
                  "mxfp4": base * rng.uniform(1.3, 2.0), "mxfp8": base * 0.1, "int8": base * 0.08,
                  "bfp12": base * 1.1}
            layers.append({"name": qr.layer_name(block, kind), "block": block, "kind": kind,
                           "params": out_features * in_features, "in_features": in_features, "kl": kl,
                           **({} if floor is None else {"floor": floor})})
    meta = {"model_id": "Qwen/Qwen3-0.6B", "model_sha": "c1899de", "windows": 8, "seqlen": 512, "seed": 42,
            "methods": METHODS, "blocks": qr.BLOCKS, "model_blocks": qr.BLOCKS, **ENV, "env": ENV}
    table = {"schema": "tricast.autotune.table/1", "status": "complete", "meta": meta, "layers": layers}
    if floor is not None:
        table["schema"] = at.TABLE_SCHEMA
        table["kv"] = [{"block": block, "name": f"model.layers.{block}.self_attn", "kl": {"kivi2": 0.1}}
                       for block in range(qr.BLOCKS)]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    return path


def _ledger(tmp_path: Path, recipe_hash: str = "h0") -> tuple[Path, str]:
    run = {"run_hash": "r", "recipe_hash": recipe_hash, "recipe": {"name": "x"}, "env": ENV,
           "metrics": {"ppl": {"ppl": 26.15988887580222, "n_tokens": 298862}}}
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "runs" / "run.json").write_text(json.dumps(run))
    path = tmp_path / "ledger.jsonl"
    lg.ingest([tmp_path / "runs" / "run.json"], path, tmp_path)
    return path, lg.read_ledger(path)[0]["id"]


def _grid(tmp_path: Path, grid: list[list[str]], name: str = "g.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"default": "nvfp4", "grid": grid}))
    return path


def test_numbers_are_supported_at_their_written_precision() -> None:
    values = [26.15988887580222, 0.0026136211788828816, 4.5]
    assert ta.unsupported_numbers("PPL 26.16 vs 26.1599, KL 0.00261 at 4.50 bits", values) == []
    assert ta.unsupported_numbers("blocks 0 and 27, 196 layers, id E1a2b3c4d5e6, budget 9/2", values) == []
    wrong = ta.unsupported_numbers("PPL 26.17 and KL 0.0027 and 2.5e-4", values)
    assert wrong == ["26.17", "0.0027", "2.5e-4"]
    # signs and leading-dot decimals are numbers too (review: -0.25 passed as 0.25, .25 was not checked)
    assert ta.unsupported_numbers("delta -0.25 and \u22120.25 and .25", [0.25]) == ["-0.25", "\u22120.25"]
    assert ta.unsupported_numbers("SQNR -3.5 dB", [-3.5]) == []
    assert ta.unsupported_numbers("range 0.1-0.25 and (x)-0.25", [0.1, 0.25]) == []  # dashes, not signs
    assert ta.unsupported_numbers("KL -0.25. Then .99.", [0.25]) == ["-0.25", ".99"]  # sentence-final period


def test_a_valid_proposal_is_recorded_with_its_recipe_and_prediction(tmp_path: Path) -> None:
    table = _table(tmp_path)
    ledger, record = _ledger(tmp_path)
    session = tmp_path / "s"
    grid = qr.uniform("nvfp4")
    grid[0] = ["int8"] * len(qr.KINDS)
    grid[1] = ["mxfp4"] * len(qr.KINDS)
    spec = ta.load_grid(_grid(tmp_path, grid))
    # block 0 on int8 (+3.53 bits on 1/28 of the weights) is not paid for by block 1 on MXFP4: 4.617 bits
    proposal = ta.propose(session, "p1", spec, table, Fraction(19, 4),
                          f"protect block 0 like the 26.16 PPL record [{record}]", [record], ledger)
    folder = session / "proposals" / "p1"
    assert json.loads((folder / "proposal.json").read_text()) == proposal
    assert (folder / "recipe.yaml").read_text() == qr.to_yaml(qr.build("p1", "nvfp4", grid)) + "\n"
    layers = at.load_table(table)[2]
    chosen = [grid[layer.block][qr.KINDS.index(layer.kind)] for layer in layers]
    assert proposal["avg_bits_fraction"] == str(at.avg_bits(layers, chosen))
    assert 4.5 < proposal["avg_bits"] <= 4.75
    table_data, _, layers, methods = at.load_table(table)
    assert proposal["predicted"]["predicted_kl_sum"] == at.predict(table_data, layers, methods, grid)[0]
    assert proposal["predicted"]["predicted_kl_sum_minus_floor"] is None  # a v1 table has no floors
    assert proposal["predicted"]["predicted_kv_kl_sum"] == 0.0
    assert proposal["method_counts"] == {"int8": 7, "mxfp4": 7, "nvfp4": 182}
    with pytest.raises(ta.TuneError, match="never overwritten"):
        ta.propose(session, "p1", spec, table, Fraction(19, 4), "again", [record], ledger)


@pytest.mark.parametrize("change, message", [
    ({"budget": Fraction(4)}, "exceed the budget"),
    ({"cites": ["Edeadbeef0000"]}, "not in the ledger"),
    ({"rationale": "measured PPL 25.01 here"}, "not supported"),
    ({"rationale": " "}, "needs a rationale"),
    ({"cites": []}, "at least one ledger record"),
    ({"grid": [["nvfp4"] * 7] * 3}, "does not build"),
])
def test_bad_proposals_are_refused(tmp_path: Path, change: dict, message: str) -> None:
    table = _table(tmp_path)
    ledger, record = _ledger(tmp_path)
    spec = ta.load_grid(_grid(tmp_path, change.get("grid", qr.uniform("nvfp4"))))
    with pytest.raises(ta.TuneError, match=message):
        ta.propose(tmp_path / "s", "p", spec, table, change.get("budget", Fraction(9, 2)),
                   change.get("rationale", "uniform"), change.get("cites", [record]), ledger)
    assert not (tmp_path / "s" / "proposals" / "p").exists()


def test_neighbours_fit_the_budget_and_come_sorted(tmp_path: Path) -> None:
    table = _table(tmp_path)
    budget = Fraction(9, 2)
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    moves = ta.neighbors(spec, table, budget, 12)
    assert len(moves) == 12
    deltas = [move["predicted_delta"] for move in moves]
    assert deltas == sorted(deltas)
    table_data, _, layers, methods = at.load_table(table)
    base = at.predict(table_data, layers, methods, spec["grid"], subtract_floor=True)[0]
    for move in moves:
        chosen = [move["grid"][layer.block][qr.KINDS.index(layer.kind)] for layer in layers]
        assert at.avg_bits(layers, chosen) <= budget
        after = at.predict(table_data, layers, methods, move["grid"], subtract_floor=True)[0]
        assert after - base == pytest.approx(move["predicted_delta"], rel=1e-9, abs=1e-12)
    funded = [move for move in ta.neighbors(spec, table, budget, 2000) if move["to"] == "int8"]
    assert funded and all(move["funded_by"] for move in funded)  # int8 costs more: paid by downgrades


def test_round_measures_on_validation_windows_and_state_joins_by_recipe_hash(tmp_path: Path) -> None:
    table = _table(tmp_path)
    session = tmp_path / "s"
    ledger, record = _ledger(tmp_path)
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    proposal = ta.propose(session, "p1", spec, table, Fraction(9, 2), "uniform", [record], ledger)
    assert json.loads((session / "session.json").read_text())["measure"] == ta.MEASURE_DEFAULTS
    script = ta.round_script(session, ["p1"], 1).read_text()
    assert "--split validation --windows 8 --seqlen 512 --seed 43 --calibrate" in script
    assert '"$S/proposals/p1/recipe.yaml"' in script
    assert "--model Qwen/Qwen3-0.6B --revision c1899de" in script
    # the done marker only after measure succeeded (review: it was printed after a refused measure)
    assert 'TUNE_ROUND_FAIL round=1 rc=$rc"; exit "$rc"; }' in script
    assert script.index("TUNE_ROUND_FAIL") < script.index('echo "TUNE_ROUND_DONE round=1"')
    with pytest.raises(ta.TuneError, match="round 1 exists"):
        ta.round_script(session, ["p1"], 1)
    assert ta.state(session, ledger)["proposals"][0]["measured_kl"] == []

    def measured(name: str, kl: float, **meta: object) -> None:
        settings = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}, **meta}
        result = {"name": "p1", "recipe_sha256": proposal["recipe_sha256"], "kl": kl, "patched": 196}
        path = tmp_path / "runs" / f"{name}.json"
        record = {"schema": "tricast.autotune.measure/2", "meta": settings, "results": [result]}
        path.write_text(json.dumps(record))
        lg.ingest([path], ledger, tmp_path)

    measured("other_model", 0.11, model_sha="0000000")  # review: another model's result was joined
    measured("table_windows", 0.12, seed=42)
    measured("train_split", 0.13, source={"split": "train"})
    measured("session", 0.27)
    row = ta.state(session, ledger)["proposals"][0]
    assert [(item["kl"], item["seed"]) for item in row["measured_kl"]] == [(0.27, 43)]


def test_shell_metacharacters_in_the_model_id_stay_one_argument(tmp_path: Path) -> None:
    table = _table(tmp_path)
    data = json.loads(table.read_text())
    data["meta"]["model_id"] = "m; id #"
    table.write_text(json.dumps(data))
    ledger, record = _ledger(tmp_path)
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    ta.propose(tmp_path / "s", "p1", spec, table, Fraction(9, 2), "uniform", [record], ledger)
    script = ta.round_script(tmp_path / "s", ["p1"], 1).read_text()
    assert "--model 'm; id #'" in script


def test_option_sizes_count_and_option_variants_are_not_predicted(tmp_path: Path) -> None:
    """Review: GPTQ with group size 32 was costed at the default 128 (4.15625 instead of 4.625 bits)."""
    table = _table(tmp_path)
    data = json.loads(table.read_text())
    data["meta"]["methods"] = [*METHODS, "gptq_int4"]
    for row in data["layers"]:
        row["kl"]["gptq_int4"] = row["kl"]["nvfp4"] * 0.9
    data["meta"]["calibration_windows"] = {"dataset": "wikitext2", "samples": 128, "seqlen": 2048, "seed": 42}
    table.write_text(json.dumps(data))
    ledger, record = _ledger(tmp_path)
    grid = qr.uniform("gptq_int4")
    spec = {**ta.load_grid(_grid(tmp_path, grid)), "options": {"gptq_int4": {"weight.group_size": 32}}}
    assert ta.method_bits("gptq_int4", 1024, spec["options"]) == Fraction(4625, 1000)
    with pytest.raises(ta.TuneError, match="exceed the budget"):
        ta.propose(tmp_path / "s", "g32", spec, table, Fraction(42, 10), "gptq", [record], ledger)
    proposal = ta.propose(tmp_path / "s", "g32", spec, table, Fraction(47, 10), "gptq", [record], ledger)
    assert proposal["avg_bits"] == 4.625 and proposal["predicted"]["predicted_kl_sum"] is None
    assert "options" in proposal["predicted"]["unpredicted"]
    default = ta.load_grid(_grid(tmp_path, grid))
    plain = ta.propose(tmp_path / "s", "g128", default, table, Fraction(47, 10), "gptq", [record], ledger)
    assert plain["avg_bits"] == 4.15625 and plain["predicted"]["predicted_kl_sum"] is not None
    other = {**ta.load_grid(_grid(tmp_path, grid)), "calibration": {"seed": 7}}
    seeded = ta.propose(tmp_path / "s", "seed7", other, table, Fraction(47, 10), "gptq", [record], ledger)
    assert seeded["predicted"]["unpredicted"].startswith("calibration settings differ")


def test_a_table_of_part_of_the_model_is_refused(tmp_path: Path) -> None:
    """Review: a 1-block table priced only that block; BF16 elsewhere passed a 4.5-bit budget (15.59 bits)."""
    table = _table(tmp_path)
    data = json.loads(table.read_text())
    data["layers"] = [row for row in data["layers"] if row["block"] == 0]
    data["meta"]["blocks"] = 1
    table.write_text(json.dumps(data))
    ledger, record = _ledger(tmp_path)
    grid = [["nvfp4"] * len(qr.KINDS)] + [["bf16"] * len(qr.KINDS)] * (qr.BLOCKS - 1)
    spec = ta.load_grid(_grid(tmp_path, grid))
    with pytest.raises(ta.TuneError, match="covers 1 of the model's 28 blocks"):
        ta.propose(tmp_path / "s", "p", spec, table, Fraction(9, 2), "x", [record], ledger)
    with pytest.raises(ta.TuneError, match="covers 1 of the model's 28 blocks"):
        ta.neighbors(spec, table, Fraction(9, 2), 3)


def test_v2_tables_rank_neighbours_on_the_floor_subtracted_kl_and_add_the_kv_entries(tmp_path: Path) -> None:
    """With floors, an 8-bit method's KL is mostly floor (docs/spikes/2026-10-13_autotune_mckp.md §6): a floor
    above every 8-bit value makes upgrading any layer to mxfp8/int8 predict exactly -kl(4-bit method)."""
    table = _table(tmp_path, floor=0.002)
    ledger, record = _ledger(tmp_path)
    table_data, _, layers, methods = at.load_table(table)
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    objective = at.objective(table_data, subtract_floor=True)
    for move in ta.neighbors(spec, table, Fraction(6), 50):
        if move["to"] in ("mxfp8", "int8") and not move["funded_by"]:
            assert move["predicted_delta"] == -objective[move["layer"]]["nvfp4"]
    spec["kv"] = "kivi2"
    proposal = ta.propose(tmp_path / "s", "kv", spec, table, Fraction(9, 2), "kivi2 cache", [record], ledger)
    predicted = proposal["predicted"]
    assert predicted["predicted_kv_kl_sum"] == pytest.approx(0.1 * qr.BLOCKS)
    assert predicted["predicted_kl_sum_minus_floor"] < predicted["predicted_kl_sum"]
    records = lg.records_of(table, tmp_path)[0]
    kv = [r for r in records if r["method"] == "kv:kivi2"]
    assert len(kv) == qr.BLOCKS and {r["layer_kind"] for r in kv} == {"kv"} and kv[13]["block"] == 13


def test_a_refused_first_proposal_fixes_no_session_and_unrankable_grids_get_no_neighbours(
        tmp_path: Path) -> None:
    """Re-check NEW-4 and M1: a refused proposal must not pin the session's table; neighbours are ranked by
    the table, so a grid with untabled options gets none."""
    table_a = _table(tmp_path)
    ledger, record = _ledger(tmp_path)
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    with pytest.raises(ta.TuneError, match="not supported"):
        ta.propose(tmp_path / "s", "p", spec, table_a, Fraction(9, 2), "PPL 25.01", [record], ledger)
    assert not (tmp_path / "s" / "session.json").exists()
    other = tmp_path / "b"
    other.mkdir()
    table_b = _table(other, floor=0.001)
    ta.propose(tmp_path / "s", "p", spec, table_b, Fraction(9, 2), "uniform", [record], ledger)
    assert json.loads((tmp_path / "s" / "session.json").read_text())["table"]["path"] == str(table_b)
    with pytest.raises(ta.TuneError, match="cannot rank this grid"):
        ta.neighbors({**spec, "options": {"nvfp4": {"weight.rounding": "rtz"}}}, table_b, Fraction(9, 2), 3)


def test_a_session_without_a_model_sha_joins_by_model_id(tmp_path: Path) -> None:
    """Re-check M7: with no model SHA in the table, another model's measurement must not join."""
    table = _table(tmp_path)
    data = json.loads(table.read_text())
    data["meta"]["model_sha"] = None
    table.write_text(json.dumps(data))
    ledger, record = _ledger(tmp_path)
    spec = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    proposal = ta.propose(tmp_path / "s", "p1", spec, table, Fraction(9, 2), "uniform", [record], ledger)
    for name, model_id, kl in (("other", "meta-llama/Llama-3.2-1B", 0.5), ("same", "Qwen/Qwen3-0.6B", 0.3)):
        meta = {**ENV, "model_id": model_id, "model_sha": None, "seed": 43, "windows": 8, "seqlen": 512,
                "source": {"split": "validation"}}
        result = {"name": "p1", "recipe_sha256": proposal["recipe_sha256"], "kl": kl, "patched": 196}
        path = tmp_path / "runs" / f"{name}.json"
        measured = {"schema": "tricast.autotune.measure/2", "meta": meta, "results": [result]}
        path.write_text(json.dumps(measured))
        lg.ingest([path], ledger, tmp_path)
    assert [item["kl"] for item in ta.state(tmp_path / "s", ledger)["proposals"][0]["measured_kl"]] == [0.3]


def test_a_proposal_may_state_its_own_option_values(tmp_path: Path) -> None:
    """A damp of 0.005 is the proposal's parameter, not a measurement: it needs no cited record."""
    table = _table(tmp_path)
    ledger, record = _ledger(tmp_path)
    grid = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    spec = {**grid, "options": {"nvfp4": {"weight.rounding": "rtz"}}}
    ta.propose(tmp_path / "s", "own", spec, table, Fraction(9, 2), "rtz rounding", [record], ledger)
    gptq = {**ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4"))),
            "options": {"nvfp4": {"weight.scale.method": "percentile", "weight.scale.percentile": 99.5}}}
    ta.propose(tmp_path / "s", "pct", gptq, table, Fraction(9, 2), "percentile 99.5 scale", [record], ledger)
    with pytest.raises(ta.TuneError, match="not supported"):
        ta.propose(tmp_path / "s", "bad", gptq, table, Fraction(9, 2), "percentile 99.4", [record], ledger)


def test_a_cited_recipe_supports_its_own_option_values(tmp_path: Path) -> None:
    table = _table(tmp_path)
    ledger, record = _ledger(tmp_path)
    grid = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    first = ta.propose(tmp_path / "s", "pct", {**grid, "options": {"nvfp4": {
        "weight.scale.method": "percentile", "weight.scale.percentile": 99.5}}}, table, Fraction(9, 2), "pct",
        [record], ledger)
    meta = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}}
    result = {"name": "pct", "recipe_sha256": first["recipe_sha256"], "kl": 0.31, "patched": 196}
    (tmp_path / "runs" / "pct.json").write_text(json.dumps({"schema": "tricast.autotune.measure/2",
                                                             "meta": meta, "results": [result]}))
    lg.ingest([tmp_path / "runs" / "pct.json"], ledger, tmp_path)
    measured = next(r["id"] for r in lg.read_ledger(ledger) if r["kind"] == "recipe_kl")
    ta.propose(tmp_path / "s", "next", grid, table, Fraction(9, 2), "99.5 gave 0.31", [measured], ledger)
    with pytest.raises(ta.TuneError, match="not supported"):
        ta.propose(tmp_path / "s", "next2", grid, table, Fraction(9, 2), "percentile 99.5", [record], ledger)


def test_ids_named_in_the_rationale_must_be_ledger_records(tmp_path: Path) -> None:
    """Loop spike finding: a rationale named "E3993cc21a10", which is no record (only cites were checked)."""
    table = _table(tmp_path)
    ledger, record = _ledger(tmp_path)
    grid = ta.load_grid(_grid(tmp_path, qr.uniform("nvfp4")))
    ta.propose(tmp_path / "s", "ok", grid, table, Fraction(9, 2), f"as {record} shows", [record], ledger)
    with pytest.raises(ta.TuneError, match="not ledger records: \\['E3993cc21a10'\\]"):
        ta.propose(tmp_path / "s", "bad", grid, table, Fraction(9, 2), "see E3993cc21a10", [record], ledger)
