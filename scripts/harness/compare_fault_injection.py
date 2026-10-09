"""Fault injection for the AC13/AC14 golden harness (docs/prompts/delegation_compare.md §3-3, §8-5).

Copies src/ and tests/ to a temporary directory, replaces one rule of src/tricast/eval/compare.py at a time,
runs tests/test_compare_golden.py there and prints which golden cases fail. The working tree is not touched.

    python scripts/harness/compare_fault_injection.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path("src/tricast/eval/compare.py")

# name -> (original text, injected text); each original must occur exactly once in compare.py
FAULTS = {
    "verdict logic removed": (
        '    return {"relative": relative, "pass": relative <= bound, "limit": bound}',
        '    return {"relative": 0.0, "pass": True, "limit": bound}'),
    "same-condition check removed": (
        "    for path, valid, kind in SAME_CONDITIONS:", "    for path, valid, kind in ():"),
    "boundary excluded (<)": ('"pass": relative <= bound', '"pass": relative < bound'),
    "limit argument ignored": ('"pass": relative <= bound', '"pass": relative <= 1e-3'),
    "infinite emulated PPL is an error": (
        "    relative = abs(emulated - reference)",
        "    if math.isinf(emulated):\n        raise ValueError('emu.metrics.ppl.ppl: inf')\n"
        "    relative = abs(emulated - reference)"),
    "invalid baseline accepted": (
        "    if not (_finite_number(value) and value > 0):",
        "    if not isinstance(value, Real) or value == 0:"),
    "NaN limit accepted": (
        "    if not (_finite_number(limit) and limit >= 0):",
        "    if not (isinstance(limit, Real) and not limit < 0):"),
    "bool accepted as PPL": (
        "    if not isinstance(value, Real) or isinstance(value, bool):",
        "    if not isinstance(value, Real):"),
    "baseline metric check skipped": (
        '    for name, record in (("base", base), ("emu", emu)):',
        '    for name, record in (("emu", emu),):'),
    "limit key dropped": (
        '"pass": relative <= bound, "limit": bound}', '"pass": relative <= bound}'),
}


def _run(repo: Path) -> tuple[str, list[str]]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_compare_golden.py", "-q", "-rf",
         "-p", "no:cacheprovider"],
        cwd=repo, capture_output=True, text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    lines = result.stdout.strip().splitlines()
    failed = [line.split("[", 1)[1].split("]", 1)[0] for line in lines if line.startswith("FAILED")]
    return lines[-1] if lines else result.stderr.strip(), failed


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        for name in ("src", "tests"):
            shutil.copytree(ROOT / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
        original = (repo / TARGET).read_text(encoding="utf-8")
        summary, _ = _run(repo)
        print(f"baseline: {summary}")
        undetected = 0
        for name, (old, new) in FAULTS.items():
            assert original.count(old) == 1, name
            (repo / TARGET).write_text(original.replace(old, new), encoding="utf-8")
            summary, failed = _run(repo)
            undetected += not failed
            print(f"{name}: {summary} {failed}")
        (repo / TARGET).write_text(original, encoding="utf-8")
        print(f"restored: {_run(repo)[0]}")
    return 1 if undetected else 0


if __name__ == "__main__":
    sys.exit(main())
