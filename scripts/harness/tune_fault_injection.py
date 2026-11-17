"""Fault injection for the AC24/AC28 tune golden harness (docs/prompts/delegation_tune.md).

Runs ALL cases of tests/harness/golden_tune.yaml once clean, then once per fault, and prints which cases fail:
  A. the five MUTATIONS of tests/test_tune_golden.py (one function of app/ replaced), and
  B. eight checks of app/tune_agent.py · app/tune_request.py turned off in memory (condition -> `if False:`).
Files on disk are never written; each module's namespace is restored from a snapshot after its fault, and the
clean run is repeated at the end. Needs the app dependencies (pyyaml, jsonschema), no GPU, no LLM call.

    PYTHONPATH=src python scripts/harness/tune_fault_injection.py [out.json]

Prints one FAULT line per fault and TUNE_FAULTS_DONE <caught exactly>/<faults>.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
import pytest  # noqa: E402

import test_tune_golden as golden  # noqa: E402

ta, tr = golden.ta, golden.tr
# (name, module, exact text whose first line becomes `if False:`, the cases that must then fail)
SOURCE_FAULTS: list[tuple[str, ModuleType, str, set[str]]] = [
    ("no-citation check off", ta, "        if not cites:\n", {"tune_proposal_no_citation"}),
    ("ledger citation check off", ta, '        if unknown:\n            raise TuneError(f"cited ids',
     {"tune_proposal_cite_outside_ledger"}),
    ("rationale id check off", ta, "        if mentioned:\n", {"tune_proposal_rationale_names_unknown_id"}),
    ("schema check off", tr, "    if errors:\n", {"tune_request_budget_not_a_number"}),
    ("unknown model allowed", tr, '    if result["model"] is not None and result["model"] not in models:\n',
     {"tune_request_model_without_table"}),
    ("unknown method allowed", tr, '        if unknown:\n            raise RequestError(f"{field}',
     {"tune_request_unknown_excluded_method", "tune_request_unknown_allowed_method"}),
    ("block range check off", tr, "    if outside:\n", {"tune_request_block_outside_model"}),
    ("missing model/budget not asked", tr,
     '    if result["outcome"] == "tune" and (result["model"] is None or result["budget_bits"] is None):\n',
     {"tune_request_no_model_asks", "tune_request_no_size_budget_asks"}),
]


def run_all(tag: str) -> dict[str, str]:
    """Every golden case once; case id -> first line of its failure."""
    failed: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case in golden.CASES:
            folder = Path(tmp) / tag.replace(" ", "_").replace("/", "_") / case["id"]
            folder.mkdir(parents=True)
            try:
                golden.test_tune_golden(case, folder)
            except (Exception, pytest.fail.Exception) as exc:
                first = (str(exc).strip().splitlines() or [""])[0]
                failed[case["id"]] = f"{type(exc).__name__}: {first[:160]}"
    return failed


def report(kind: str, name: str, file: str, change: str, expected: set[str],
           failed: dict[str, str]) -> dict[str, Any]:
    row = {"kind": kind, "name": name, "file": file, "change": change, "expected": sorted(expected),
           "failed": failed, "caught_exactly": set(failed) == expected}
    print(f"FAULT [{kind}] {name}: {file} failed={sorted(failed)} caught_exactly={row['caught_exactly']}")
    return row


def main(argv: list[str]) -> int:
    out: dict[str, Any] = {"cases": len(golden.CASES), "clean": run_all("clean"), "faults": []}
    print(f"CLEAN failed={len(out['clean'])} of {len(golden.CASES)}")
    for name, module, attr, broken, expected in golden.MUTATIONS:
        original = getattr(module, attr)
        setattr(module, attr, broken)
        try:
            failed = run_all(name)
        finally:
            setattr(module, attr, original)
        file = f"{module.__name__.replace('.', '/')}.py"
        change = f"{attr} replaced (tests/test_tune_golden.py MUTATIONS)"
        out["faults"].append(report("A", name, file, change, expected, failed))
    for name, module, old, expected in SOURCE_FAULTS:
        path = Path(module.__file__)
        source = path.read_text(encoding="utf-8")
        assert source.count(old) == 1, (name, source.count(old))
        first = old.splitlines()[0]
        indent = first[: len(first) - len(first.lstrip())]
        mutated = source.replace(old, f"{indent}if False:\n" + old[len(first) + 1:], 1)
        snapshot = dict(module.__dict__)
        exec(compile(mutated, str(path), "exec"), module.__dict__)
        for key, value in snapshot.items():  # keep the original exception classes
            if isinstance(value, type):
                module.__dict__[key] = value
        try:
            failed = run_all(name)
        finally:
            module.__dict__.clear()
            module.__dict__.update(snapshot)
        change = f"`{first.strip()}` -> `if False:`"
        out["faults"].append(report("B", name, str(path.relative_to(ROOT)), change, expected, failed))
    out["clean_after"] = run_all("clean after")
    print(f"CLEAN_AFTER failed={len(out['clean_after'])} of {len(golden.CASES)}")
    if argv:
        Path(argv[0]).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    caught = sum(row["caught_exactly"] for row in out["faults"])
    print(f"TUNE_FAULTS_DONE {caught}/{len(out['faults'])}")
    ok = caught == len(out["faults"]) and not out["clean"] and not out["clean_after"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
