"""Black-box controls of the tune loop (app/tune_loop.py evolution / random_search) and the fixed-budget stop
rule: proposals fit the budget, build, are new and cite their parent's record; the random stream is seeded;
random search never selects, evolution starts from the best; with stale_rounds 0 every round is measured.
Tables, ledgers and measurements are fakes; CPU only."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
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
