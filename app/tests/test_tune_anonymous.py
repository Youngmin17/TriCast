"""The anonymous-methods ablation of the Claude proposer (llm.mask anonymous): every method id becomes a
seeded alias in the prompt and the schema, the catalog is withheld, and a reply written with the aliases is
decoded to method ids and goes through the usual checks. LLM and GPU are fakes; CPU only."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from app import quantizer_recipe as qr
from app import tune_loop as tl
from app.tests.test_tune_loop import _session


def test_aliases_are_a_seeded_bijection_without_digits() -> None:
    aliases = tl.method_aliases(0)
    assert set(aliases) == set(qr.PYTHON_METHODS) and len(set(aliases.values())) == len(aliases)
    assert all(re.fullmatch(r"method_[a-z]+", alias) for alias in aliases.values())
    assert tl.method_aliases(0) == aliases and tl.method_aliases(1) != aliases
    assert not set(aliases.values()) & set(qr.PYTHON_METHODS)


def test_the_anonymous_prompt_names_no_method_and_its_reply_is_decoded(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    aliases = tl.method_aliases(3)
    seen = {}

    def runner(command, **kwargs):
        seen["prompt"] = kwargs["input"]
        seen["schema"] = json.loads(command[command.index("--json-schema") + 1])
        cites = [analysis["best"]["kl_record"]]
        reply = {"analysis": "a", "stop": False, "proposals": [
            {"name": "anon1", "base": "start", "rationale": "start KL", "cites": cites,
             "options": {aliases["nvint4"]: {"weight.rounding": "rtz"}},
             "changes": [{"block": 0, "kind": "down_proj", "method": aliases["mxfp4"]},
                         {"block": 1, "kind": "q_proj", "method": aliases["nvint4"]}]}]}
        out = json.dumps({"type": "result", "structured_output": reply})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    anonymous = {**config, "llm": {"mask": ["anonymous"], "alias_seed": 3}}
    reply = tl.check_reply(tl.claude_code(anonymous, tmp_path, runner)(analysis, 1, 2))
    shown = json.loads(seen["prompt"].split("\n", 1)[1])
    assert "catalog" not in shown and "option_space" in shown
    text = json.dumps(shown)
    assert not [method for method in qr.PYTHON_METHODS if f'"{method}"' in text]  # no method id, value or key
    assert set(shown["option_space"]) == {aliases["nvfp4"]}  # the start grid is uniform NVFP4
    enum = seen["schema"]["properties"]["proposals"]["items"]["properties"]["changes"]["items"]["properties"][
        "method"]["enum"]
    assert enum == list(aliases.values())
    (item,) = reply["proposals"]
    assert [change["method"] for change in item["changes"]] == ["mxfp4", "nvint4"]
    assert item["options"] == {"nvint4": {"weight.rounding": "rtz"}}
    tl.materialize(config, tmp_path, item, "claude-code")  # the decoded proposal passes every check
    record = json.loads((tmp_path / "s" / "llm" / "round_1.json").read_text())
    assert record["mask"] == ["anonymous"] and record["aliases"] == aliases


def test_without_the_mask_nothing_is_renamed(tmp_path: Path) -> None:
    config, _ = _session(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    seen = {}

    def runner(command, **kwargs):
        seen["prompt"], seen["command"] = kwargs["input"], command
        reply = {"analysis": "a", "stop": True, "proposals": []}
        out = json.dumps({"type": "result", "structured_output": reply})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")

    tl.claude_code(config, tmp_path, runner)(analysis, 1, 1)
    assert '"nvfp4"' in seen["prompt"]
    assert json.loads(seen["command"][seen["command"].index("--json-schema") + 1]) == tl.PROPOSAL_SCHEMA
