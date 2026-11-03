"""Comparison arms of the tune loop (protocol docs/research/agentic_mixed_quant/protocol.md §6): Optuna TPE,
option-only coordinate search, LLM-then-TPE, the Claude proposer's analysis masks, and the no-check
ablation. Same checks and measurement path as every proposer; LLM, GPU and ssh are fakes. CPU only."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from app import quantizer_recipe as qr
from app import tune_agent as ta
from app import tune_loop as tl
from app.tests.test_tune_loop import _measure, _session

pytest.importorskip("optuna")


def test_tpe_proposals_fit_build_are_new_and_are_seeded(tmp_path: Path) -> None:
    config, ledger = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    first = tl.tpe_search(config, tmp_path, seed=1)(analysis, 1, 4)
    again = tl.tpe_search(config, tmp_path, seed=1)(analysis, 1, 4)
    assert first == again and len(first["proposals"]) == 4
    for item in first["proposals"]:
        proposal = tl.materialize(config, tmp_path, item, "tpe")  # every check of propose passes
        assert Fraction(proposal["avg_bits_fraction"]) <= Fraction(9, 2)
    names = [item["name"] for item in first["proposals"]]
    rows = {name: (ta.read_proposal(tmp_path / "s", name)["recipe_sha256"], 0.2 + 0.01 * i)
            for i, name in enumerate(names)}
    _measure(tmp_path, ledger, rows, "m1")
    # round 2 rebuilds the study from all five measured proposals and proposes new grids only
    second = tl.tpe_search(config, tmp_path, seed=1)(tl.analyze(config, tmp_path), 2, 3)["proposals"]
    assert len(second) == 3 and {item["base"] for item in second} == {names[0]}
    assert not {item["name"] for item in second} & set(names)


def test_option_greedy_changes_one_option_and_no_cell(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    reply = tl.option_greedy(config, tmp_path)(tl.analyze(config, tmp_path), 1, 5)
    assert len(reply["proposals"]) == 5
    for item in reply["proposals"]:
        assert item["changes"] == [] and len(item["options"]) == 1
        (method, chosen), = item["options"].items()
        assert method == "nvfp4" and len(chosen) == 1
        tl.materialize(config, tmp_path, item, "option-greedy")


def test_the_hybrid_hands_over_after_its_seed_rounds() -> None:
    def first(analysis: dict, number: int, count: int) -> dict:
        return {"analysis": "first", "stop": False, "proposals": []}

    def then(analysis: dict, number: int, count: int) -> dict:
        return {"analysis": "then", "stop": False, "proposals": []}

    proposer = tl.hybrid(first, then, 2)
    assert [proposer({}, n, 1)["analysis"] for n in (1, 2, 3, 4)] == ["first", "first", "then", "then"]


def test_a_mask_withholds_its_fields_from_the_prompt(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    prompts = []

    def runner(command, **kwargs):
        prompts.append(kwargs["input"])
        reply = {"analysis": "a", "stop": True, "proposals": []}
        out = json.dumps({"type": "result", "structured_output": reply})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    masked = {**config, "llm": {"mask": ["no-history", "no-evidence"]}}
    tl.claude_code(masked, tmp_path, runner)(analysis, 1, 4)
    shown = json.loads(prompts[0].split("\n", 1)[1])
    assert not {"history", "predicted_vs_measured", "catalog"} & set(shown)
    assert {"best", "loss_concentration", "option_space"} <= set(shown)
    record = json.loads((tmp_path / "s" / "llm" / "round_1.json").read_text())
    assert record["mask"] == ["no-history", "no-evidence"]
    with pytest.raises(tl.LoopError, match="unknown ablations"):
        tl.claude_code({**config, "llm": {"mask": ["no-such"]}}, tmp_path, runner)


def test_the_no_check_ablation_drops_evidence_checks_but_keeps_the_budget(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    loose = {**config, "checks": False}
    unsupported = {"name": "loose", "base": "start", "options": None, "cites": [],
                   "rationale": "KL drops to 0.0123 (no record says so), see E0123456789ab",
                   "changes": [{"block": 0, "kind": "k_proj", "method": "mxfp4"}]}
    with pytest.raises(ta.TuneError):
        tl.materialize(config, tmp_path, unsupported, "test")
    proposal = tl.materialize(loose, tmp_path, unsupported, "test")
    assert proposal["checks"] is False and proposal["cites"] == []
    fat = {**unsupported, "name": "fat", "changes": [{"block": b, "kind": "down_proj", "method": "bf16"}
                                                      for b in range(qr.BLOCKS)]}
    with pytest.raises(ta.TuneError, match="exceed the budget"):
        tl.materialize(loose, tmp_path, fat, "test")


def test_the_shared_reply_check_accepts_lowbit_changes_from_the_black_box_arms() -> None:
    """Review 2026-11-04 MAJOR 2: check_reply applies to every proposer, so it must accept every Python-side
    method; the schema sent to Claude Code is a separate, narrower one."""
    def reply(method: str) -> dict:
        change = {"block": 0, "kind": "q_proj", "method": method}
        return {"analysis": "", "stop": False, "proposals": [
            {"name": "x", "base": "start", "options": None, "rationale": "x", "cites": ["E0"],
             "changes": [change]}]}

    for method in qr.LOWBIT:
        assert tl.check_reply(reply(method))
    with pytest.raises(tl.LoopError, match="proposal schema"):
        tl.check_reply(reply("no_such_method"))
    assert set(tl.REPLY_SCHEMA["properties"]["proposals"]["items"]["properties"]["changes"]["items"]
               ["properties"]["method"]["enum"]) == set(qr.PYTHON_METHODS)
