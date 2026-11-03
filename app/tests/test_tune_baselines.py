"""Black-box controls of the tune loop (app/tune_loop.py evolution / random_search) and the fixed-budget stop
rule: proposals fit the budget, build, are new and cite their parent's record; the random stream is seeded;
random search never selects, evolution starts from the best; with stale_rounds 0 every round is measured.
Tables, ledgers and measurements are fakes; CPU only."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_agent import ENV
from app.tests.test_tune_loop import _measure, _session


def _register(tmp_path: Path, ledger: Path, proposals: list[dict], kls: list[float], name: str) -> None:
    config = {"session": "s", "table": "table.json", "budget_bits": "9/2", "ledger": "ledger.jsonl"}
    shas = {}
    for item in proposals:
        shas[item["name"]] = tl.materialize(config, tmp_path, item, "test")["recipe_sha256"]
    rows = {key: (sha, kl) for (key, sha), kl in zip(shas.items(), kls, strict=True)}
    _measure(tmp_path, ledger, rows, name)


def test_evolution_proposals_fit_build_are_new_and_cite_their_parent(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    reply = tl.check_reply(tl.evolution(config, tmp_path)(analysis, 1, 6))
    assert len(reply["proposals"]) == 6 and not reply["stop"]
    grids = []
    for item in reply["proposals"]:
        assert item["base"] == "start" and item["cites"] == [analysis["best"]["kl_record"]]
        assert 1 <= len(item["changes"]) or item["options"]
        grids.append(tl.materialize(config, tmp_path, item, "evolution"))  # every check of propose passes
    assert all(Fraction(row["avg_bits_fraction"]) <= Fraction(9, 2) for row in grids)


def test_the_random_stream_is_seeded_by_label_seed_and_round(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    first = tl.evolution(config, tmp_path, seed=3)(analysis, 1, 4)["proposals"]
    again = tl.evolution(config, tmp_path, seed=3)(analysis, 1, 4)["proposals"]
    other = tl.evolution(config, tmp_path, seed=4)(analysis, 1, 4)["proposals"]
    assert first == again and first != other


def test_random_search_never_selects_and_evolution_starts_from_the_best(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    child = tl.evolution(config, tmp_path)(analysis, 1, 1)["proposals"]
    _register(tmp_path, ledger, child, [0.20], "m1")  # lower than the start's 0.30
    analysis = tl.analyze(config, tmp_path)
    assert analysis["best"]["name"] == child[0]["name"]
    def bases(proposer: tl.Proposer, count: int) -> set:
        return {item["base"] for item in proposer(analysis, 2, count)["proposals"]}

    assert bases(tl.random_search(config, tmp_path), 4) == {"start"}
    assert bases(tl.evolution(config, tmp_path), 4) == {child[0]["name"]}
    assert bases(tl.evolution(config, tmp_path, parents=2, seed=1), 8) == {"start", child[0]["name"]}


def test_an_option_mutation_respects_when_and_drops_defaults() -> None:
    grid = qr.uniform("nvfp4")
    for seed in range(40):
        options: dict = {}
        changed = tl._mutate_option(random.Random(seed), grid, options)
        if changed is None:
            continue
        method, option = changed.split(".", 1)
        assert method == "nvfp4" and option in qr.METHOD_OPTIONS["nvfp4"]
        rule = qr.OPTIONS[option].get("when")
        assert rule is None or qr.OPTIONS[rule[0]]["default"] in rule[1]  # active with every other at default
        default = qr.OPTIONS[option]["default"]
        assert options.get("nvfp4", {}).get(option, default) != default
    back = {"gptq_int4": {"weight_algo.act_order": True}}
    grid = qr.uniform("gptq_int4")
    for seed in range(200):
        trial = json.loads(json.dumps(back))
        if tl._mutate_option(random.Random(seed), grid, trial) == "gptq_int4.weight_algo.act_order":
            assert trial == {}  # flipped back to the default: no option left
            break
    else:
        raise AssertionError("act_order was never drawn")


def test_with_stale_rounds_zero_every_round_is_measured(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    proposer = tl.evolution(config, tmp_path)

    def measure(number: int, names: list) -> Path:
        shas = {name: ta.read_proposal(tmp_path / "s", name)["recipe_sha256"] for name in names}
        return _measure(tmp_path, ledger, {name: (sha, 0.9) for name, sha in shas.items()}, f"r{number}")

    result = tl.run(config, tmp_path, 4, 2, proposer, "evolution", measure, log=lambda line: None,
                    stale_rounds=0)
    assert [entry["round"] for entry in result["rounds"]] == [1, 2, 3, 4] and result["best"] == "start"
    assert sum(len(entry["accepted"]) for entry in result["rounds"]) == 8


def test_the_proposal_schema_names_every_selectable_method() -> None:
    """The Claude proposer can choose the Python-only methods (nf4, hqq_int4, razer_nvfp4) by name."""
    assert tl._CHANGE["properties"]["method"]["enum"] == list(qr.ALL_METHODS)
    assert set(qr.EXTENDED) <= set(tl._CHANGE["properties"]["method"]["enum"])


def test_a_recipe_already_measured_is_refused_however_its_options_are_written(tmp_path: Path) -> None:
    """Review F3: explicit defaults, empty or unused option entries build the same recipe as the base."""
    config, _ = _session(tmp_path)
    record = tl.analyze(config, tmp_path)["best"]["kl_record"]
    default = qr.OPTIONS["weight.rounding"]["default"]
    unused = {"gptq_int4": {"weight_algo.damp": 0.1}}  # gptq_int4 is not in the uniform NVFP4 grid
    for options in ({"bf16": {}}, {"nvfp4": {"weight.rounding": default}}, unused):
        item = {"name": "again", "base": "start", "changes": [], "options": options, "rationale": "x",
                "cites": [record]}
        with pytest.raises(ta.TuneError, match="already in the session"):
            tl.materialize(config, tmp_path, item, "test")


def test_inactive_options_are_dropped_to_a_fixed_point() -> None:
    """Review F5: a percentile left behind when its scale method went back to absmax is dropped."""
    options = {"nvfp4": {"weight.scale.percentile": 95.0}, "int8": {}}
    assert tl._drop_inactive(options) == {}
    kept = {"nvfp4": {"weight.scale.method": "percentile", "weight.scale.percentile": 95.0}}
    assert tl._drop_inactive(kept) == kept


def test_a_fixed_budget_makes_up_refused_proposals_and_records_a_shortfall(tmp_path: Path) -> None:
    """Review F4: with measurements=N the loop runs until N proposals are measured, refusals included."""
    config, ledger = _session(tmp_path)
    inner = tl.evolution(config, tmp_path)
    asked = []

    def half_refused(analysis: dict, number: int, count: int) -> dict:
        asked.append(count)
        reply = inner(analysis, number, count)
        bad = {**reply["proposals"][0], "name": f"bad{number}", "cites": ["Enope"]}  # refused: no such record
        return {**reply, "proposals": ([bad] if number % 2 else []) + reply["proposals"]}

    def measure(number: int, names: list) -> Path:
        shas = {name: ta.read_proposal(tmp_path / "s", name)["recipe_sha256"] for name in names}
        return _measure(tmp_path, ledger, {name: (sha, 0.9) for name, sha in shas.items()}, f"r{number}")

    result = tl.run(config, tmp_path, 20, 2, half_refused, "evolution", measure, log=lambda line: None,
                    measurements=5)
    assert result["budget"] == {"measurements": 5, "measured": 5, "shortfall": 0}
    # 1 + 2 + 1 accepted in rounds 1-3 (two refusals), then the last round asks for the 1 still missing
    assert asked == [2, 2, 2, 1] and [len(entry["accepted"]) for entry in result["rounds"]] == [1, 2, 1, 1]

    def stop(analysis: dict, number: int, count: int) -> dict:
        return {"analysis": "x", "stop": True, "proposals": []}

    short = tl.run(config, tmp_path, 20, 2, stop, "evolution", measure, log=lambda line: None, measurements=7)
    assert short["budget"] == {"measurements": 7, "measured": 5, "shortfall": 2}


@pytest.mark.parametrize("flags", [["--parents", "0"], ["--stale-rounds", "-1"], ["--measurements", "0"],
                                   ["--per-round", "0"]])
def test_the_cli_refuses_out_of_range_flags(flags: list, tmp_path: Path) -> None:
    """Review F7."""
    with pytest.raises(SystemExit) as exit_info:
        tl.main(["run", "--config", str(tmp_path / "loop.json"), *flags])
    assert exit_info.value.code == 2


def test_a_resumed_fixed_budget_counts_rounds_measured_before_the_summary_was_written(tmp_path: Path) -> None:
    """Re-check F4: kl.json saved, summary.json not yet written, then a rerun: those measurements count."""
    config, ledger = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    first = tl.evolution(config, tmp_path)(analysis, 1, 2)["proposals"]
    for item in first:
        tl.materialize(config, tmp_path, item, "evolution")
    names = [item["name"] for item in first]
    ta.round_script(tmp_path / "s", names, 1)
    shas = {name: ta.read_proposal(tmp_path / "s", name)["recipe_sha256"] for name in names}
    meta = {**ENV, "seed": 43, "windows": 8, "seqlen": 512, "source": {"split": "validation"}}
    (tmp_path / "s" / "rounds" / "1" / "kl.json").write_text(json.dumps({
        "schema": "tricast.autotune.measure/2", "meta": meta,
        "results": [{"name": name, "recipe_sha256": sha, "kl": 0.9, "patched": 196}
                    for name, sha in shas.items()]}))
    asked = []

    def proposer(analysis: dict, number: int, count: int) -> dict:
        asked.append(count)
        return tl.evolution(config, tmp_path, seed=5)(analysis, number, count)

    def measure(number: int, names: list) -> Path:
        rows = {name: (ta.read_proposal(tmp_path / "s", name)["recipe_sha256"], 0.9) for name in names}
        return _measure(tmp_path, ledger, rows, f"r{number}")

    result = tl.run(config, tmp_path, 5, 2, proposer, "evolution", measure, log=lambda line: None,
                    measurements=3)
    assert asked == [1] and result["budget"] == {"measurements": 3, "measured": 3, "shortfall": 0}
