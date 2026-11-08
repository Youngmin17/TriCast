"""evals/judge_agreement.py: the labeling sheet carries the judge's whole input and none of its verdicts, and
agreement counts only labels that meet a true/false judge verdict on the same unit of the same run, with the
direction of every disagreement."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("judge_agreement",
                                              Path(__file__).parents[1] / "evals" / "judge_agreement.py")
JA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = JA
SPEC.loader.exec_module(JA)

REASON = "JUDGE-REASON-MARKER"


def _verdict(**passes: bool | None) -> dict:
    return {name: None if value is None else {"reason": REASON, "pass": value}
            for name, value in passes.items()}


RESULT = {"run": {"run_id": "R1"}, "cases": [
    {"id": "r_q1", "category": "clarify", "judge": [
        {"unit": "request", "required": ["q1", "q2"], "input": {"query": "알아서 해줘"},
         "verdict": _verdict(q1=True, q2=True)}]},
    {"id": "a_c1", "category": "constraint", "judge": [
        {"unit": "p1", "required": ["c1", "c2", "c3"], "input": {"retrieved": [{"text": "RETRIEVED-DOC"}]},
         "verdict": _verdict(c1=True, c2=False, c3=None)}]},
    {"id": "r_o1", "category": "out_of_scope", "judge": None}]}


def test_the_sheet_shows_the_judge_input_and_no_verdict() -> None:
    text = JA.sheet(RESULT)
    assert "RETRIEVED-DOC" in text and "알아서 해줘" in text
    assert REASON not in text and '"pass"' not in text
    assert "## r_q1 · unit `request`" in text and "## a_c1 · unit `p1`" in text and "r_o1" not in text
    assert text.count("| q1 | | |") == 1 and text.count("| c3 | | |") == 1


def test_agreement_counts_matched_true_false_pairs_and_their_direction() -> None:
    labels = [
        {"run_id": "R1", "id": "r_q1", "unit": "request", "q1": True, "q2": False},  # q1 agree, q2 lenient
        {"run_id": "R1", "id": "a_c1", "unit": "p1", "c1": True, "c2": True, "c3": True},  # =, strict, null
        {"run_id": "R0", "id": "r_q1", "unit": "request", "q1": False},  # another run
        {"run_id": "R1", "id": "a_c1", "unit": "p9", "c1": False},  # no such unit
    ]
    report = JA.agree(RESULT, labels)
    assert report["judgments"] == 4 and report["overall"] == 0.5 and report["met"] is False
    assert report["criteria"]["q2"] == {"n": 1, "agree": 0, "lenient": 1, "strict": 0, "rate": 0.0}
    assert report["criteria"]["c2"]["strict"] == 1 and "c3" not in report["criteria"]
    assert len(report["unmatched"]) == 3  # the other run, the missing unit, c3 judged null


def test_agree_without_labels_says_so(tmp_path: Path) -> None:
    path = tmp_path / "results" / "R1.json"
    path.parent.mkdir()
    path.write_text(json.dumps(RESULT))
    assert JA.main(["agree", str(path), str(tmp_path / "none.jsonl")]) == 1
    assert JA.main(["sheet", str(path)]) == 0 and (path.parent / "R1_labeling_full.md").is_file()
