"""Fault injection on a real run's stored replies (docs/prompts/delegation_evals.md §1-2): each row puts one
wrong answer into a copy of ``evals/fixtures/tune/<run_id>``, re-grades it with the runner in stored mode
(no LLM call) and holds the verdict against the expected one and against a stored re-grade of the untouched
replies. The runner and its criteria are not changed; this only drives it.

    python evals/inject_faults.py <run_id> --table <table.json> --ledger <ledger.jsonl> --fixture s3150=<dir>

Prints one INJECT line per row (PASS when the runner gave the expected verdict and moved only that case) and
INJECT_DONE <passed>/<rows>. The copies and the re-graded results go to a temporary directory and are removed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evals" / "fixtures" / "tune"
CASES = "r_c2,r_q1,r_i1,r_h2,r_k2,a_c1,a_i1"
Calls = list[dict[str, Any]]


def _reply(calls: Calls, role: str) -> dict[str, Any]:
    return next(call for call in calls if call["role"] == role)["stdout"]["structured_output"]


def _set(role: str, **fields: Any) -> Callable[[Calls], None]:
    return lambda calls: _reply(calls, role).update(fields)


def _cite_planted(calls: Calls) -> None:
    _reply(calls, "proposer")["proposals"][0]["cites"].append("E000000000000")


def _first_judge_c2_false(calls: Calls) -> None:
    _reply(calls, "judge")["c2"]["pass"] = False


def _judge_sentence(calls: Calls) -> None:
    next(call for call in calls if call["role"] == "judge")["stdout"] = "The questions ask for both values."


def _metric(result: dict[str, Any], name: str) -> dict[str, Any]:
    return result["metrics"][name]


def _case(result: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(case for case in result["cases"] if case["id"] == case_id)


def _judge_unit(result: dict[str, Any]) -> str:
    return _case(result, "a_c1")["judge"][0]["unit"]


ROWS: list[tuple[str, str, Callable[[Calls], None], Callable[[dict, dict], tuple[bool, str]]]] = [
    ("r_c2 keep_bf16_blocks [0]", "r_c2", _set("request", keep_bf16_blocks=[0]),
     lambda base, out: (_case(out, "r_c2")["status"] == "fail"
                        and not _case(out, "r_c2")["checks"]["keep_bf16_blocks"]["pass"]
                        and _metric(out, "A1")["numerator"] == _metric(base, "A1")["numerator"] - 1,
                        f"A1 {_metric(base, 'A1')['numerator']} -> {_metric(out, 'A1')['numerator']}")),
    ("r_q1 model filled, no assumption", "r_q1", _set("request", model="Qwen/Qwen3-0.6B", assumptions=[]),
     lambda base, out: (_case(out, "r_q1")["invented"] == ["model"]
                        and _metric(out, "A2")["value"] == _metric(base, "A2")["value"] + 1,
                        f"A2 {_metric(base, 'A2')['value']} -> {_metric(out, 'A2')['value']}")),
    ("r_i1 budget_bits 16", "r_i1", _set("request", budget_bits="16"),
     lambda base, out: (not _case(out, "r_i1")["checks"]["budget_bits"]["pass"]
                        and _metric(out, "A6")["value"] == _metric(base, "A6")["value"] + 1,
                        f"A6 {_metric(base, 'A6')['value']} -> {_metric(out, 'A6')['value']}")),
    ("r_h2 only_methods quip_e8p", "r_h2", _set("request", only_methods=["quip_e8p"]),
     lambda base, out: ("RequestError" in _case(out, "r_h2")["checks"]["format"]["detail"]
                        and _metric(out, "A7")["numerator"] == _metric(base, "A7")["numerator"] - 1,
                        f"A7 {_metric(base, 'A7')['numerator']} -> {_metric(out, 'A7')['numerator']}")),
    ("r_k2 outcome tune (model misses the conflict)", "r_k2", _set("request", outcome="tune"),
     lambda base, out: (_case(out, "r_k2")["output"]["result"]["outcome"] == "conflict"
                        and _case(out, "r_k2")["status"] == _case(base, "r_k2")["status"]
                        and _metric(out, "A5") == _metric(base, "A5"), "code turns it into a conflict")),
    ("a_i1 a proposal cites E000000000000", "a_i1", _cite_planted,
     lambda base, out: (_case(out, "a_i1")["output"]["proposals"][0]["status"] == "refused"
                        and not _case(out, "a_i1")["checks"]["not_cited"]["pass"]
                        and _metric(out, "B5")["value"] == _metric(base, "B5")["value"] + 1
                        and _metric(out, "B2")["value"] == _metric(base, "B2")["value"],
                        f"B5 {_metric(base, 'B5')['value']} -> {_metric(out, 'B5')['value']}, "
                        f"B2 {_metric(out, 'B2')['value']}")),
    ("a_c1 judge c2 false", "a_c1", _first_judge_c2_false,
     lambda base, out: (_case(out, "a_c1")["judge"][0]["pass"] is False
                        and _metric(out, "B8")["denominator"] == _metric(base, "B8")["denominator"],
                        f"B8 {_metric(base, 'B8')['numerator']}/{_metric(base, 'B8')['denominator']} -> "
                        f"{_metric(out, 'B8')['numerator']}/{_metric(out, 'B8')['denominator']}")),
    ("judge reply a sentence", "r_q1", _judge_sentence,
     lambda base, out: (_case(out, "r_q1")["status"] == "error"
                        and _metric(out, "A8")["denominator"] == _metric(base, "A8")["denominator"]
                        and out["run"]["cases"]["total"] == base["run"]["cases"]["total"],
                        f"A8 errors {_metric(out, 'A8').get('error_cases')}, still in the denominator")),
]


def regrade(source: str, out: Path, args: list[str], change: str) -> dict[str, Any]:
    before = set(out.glob("*.json"))
    command = [sys.executable, str(ROOT / "evals" / "run_tune_evals.py"), *args, "--mode", "stored",
               "--from-run", source, "--only", CASES, "--out", str(out), "--change", change,
               "--golden", "n/a"]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    (made,) = set(out.glob("*.json")) - before
    result = json.loads(made.read_text(encoding="utf-8"))
    shutil.rmtree(FIXTURES / result["run"]["run_id"], ignore_errors=True)  # the stored run's own copy
    made.unlink()
    return result


def main(argv: list[str]) -> int:
    run_id, args = argv[0], argv[1:]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="inject_") as scratch:
        out = Path(scratch)
        base = regrade(run_id, out, args, "fault injection: untouched replies")
        for number, (row, case_id, edit, check) in enumerate(ROWS, 1):
            name = f"inject{number}_{run_id}"
            target = FIXTURES / name
            shutil.copytree(FIXTURES / run_id, target)
            try:
                path = target / f"{case_id}.json"
                kept = json.loads(path.read_text(encoding="utf-8"))
                edit(kept["calls"])
                path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
                result = regrade(name, out, args, f"fault injection: {row}")
            finally:
                shutil.rmtree(target, ignore_errors=True)
            ok, detail = check(base, result)
            moved = [case["id"] for case in result["cases"]
                     if case["id"] != case_id and case["status"] != _case(base, case["id"])["status"]]
            ok = ok and not moved
            passed += ok
            extra = f"; also moved {moved}" if moved else ""
            print(f"INJECT {'PASS' if ok else 'FAIL'} {number} {row}: {detail}{extra}")
    print(f"INJECT_DONE {passed}/{len(ROWS)} run={run_id}")
    return 0 if passed == len(ROWS) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
