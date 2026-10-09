"""AGENTS.md rules -> docs/SPEC.md acceptance criteria -> judging tests stay linked to things that exist."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / "docs/SPEC.md").read_text(encoding="utf-8")
AGENTS = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
GOLDEN = yaml.safe_load((ROOT / "tests/harness/golden_cases.yaml").read_text(encoding="utf-8"))
REPO_PATH = re.compile(r"(?:tests|app/tests|scripts|configs|evals)/")
PENDING = {"AC12"}  # ACs judged by no test yet ("(테스트 예정)"); this set may only shrink


def _section(text: str, heading: str) -> str:
    assert f"\n{heading}" in text, f"heading {heading!r} not found"
    return text.split(f"\n{heading}", 1)[1].split("\n## ", 1)[0]


def _table(section: str, column: str) -> list[tuple[str, str]]:
    """(first cell, cell under the header starting with `column`) per body row of the section's table."""
    lines = [line.strip().strip("|") for line in section.splitlines() if line.startswith("|")]
    header, _, *rows = [[cell.strip() for cell in line.split("|")] for line in lines]
    index = next(i for i, name in enumerate(header) if name.startswith(column))
    for row in rows:
        assert len(row) == len(header), f"{row[0]}: {len(row)} cells, header has {len(header)}"
    return [(row[0], row[index]) for row in rows]


AC_ROWS = _table(_section(SPEC, "## 6. 수용 기준"), "판정 방법")
AC_NAMES = {first.strip("*") for first, _ in AC_ROWS}
TRACE_ROWS = _table(_section(SPEC, "## 9. 추적성"), "테스트")
RULES = re.findall(r"^(\d+)\. (.*?)(?=^\d+\. |\Z)", _section(AGENTS, "## 3. 절대 규칙"), re.M | re.S)


def _references(cell: str) -> list[tuple[str, str]]:
    """(path, test name or "") per backticked repo path; bare `test_x.py` / `test_x` reuse the last one."""
    refs: list[tuple[str, str]] = []
    for token in re.findall(r"`([^`]+)`", cell):
        if REPO_PATH.match(token):
            path, _, name = token.partition("::")
        elif refs and re.fullmatch(r"test_\w+\.py", token):
            path, name = str(PurePosixPath(refs[-1][0]).parent / token), ""
        elif refs and re.fullmatch(r"test_\w+", token):
            path, name = refs[-1][0], token
        else:
            continue
        refs.append((path, name))
    return refs


def _exists(path: str, name: str, where: str) -> None:
    files = sorted(ROOT.glob(path)) if "*" in path else [ROOT / path]
    assert files and all(file.exists() for file in files), f"{where}: {path} does not exist"
    if name:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert re.search(rf"^\s*def {name}\(", source, re.M), f"{where}: {path} has no def {name}"


def test_ac_rows_are_numbered_from_one() -> None:
    numbers = []
    for first, _ in AC_ROWS:
        match = re.fullmatch(r"\*\*AC(\d+)\*\*", first)
        assert match, f"docs/SPEC.md §6: row {first!r} is not '**ACn**'"
        numbers.append(int(match.group(1)))
    assert numbers and numbers == list(range(1, len(numbers) + 1)), f"docs/SPEC.md §6: AC numbers {numbers}"


def test_ac_judgment_paths_exist() -> None:
    for first, cell in AC_ROWS:
        for path, name in _references(cell):
            _exists(path, name, f"docs/SPEC.md §6 {first}")


def test_only_known_acs_lack_a_test() -> None:
    pending = set()
    for first, cell in AC_ROWS:
        if not _references(cell):
            assert "(테스트 예정)" in cell, f"docs/SPEC.md §6 {first}: no test path and not marked pending"
            pending.add(first.strip("*"))
    assert pending <= PENDING, f"docs/SPEC.md §6: new ACs without a test {sorted(pending - PENDING)}"


def test_golden_cases_cite_spec_acs() -> None:
    assert GOLDEN
    for case in GOLDEN:
        where = f"tests/harness/golden_cases.yaml {case.get('id')}"
        assert case.get("ac") in AC_NAMES, f"{where}: ac {case.get('ac')!r} is not an AC of docs/SPEC.md §6"


def test_absolute_rules_point_to_spec_acs() -> None:
    assert RULES
    for number, text in RULES:
        cited = re.findall(r"\(↔ (AC\d+)\)", text)
        assert cited, f"AGENTS.md §3 rule {number}: no '(↔ ACn)'"
        for ac in cited:
            assert ac in AC_NAMES, f"AGENTS.md §3 rule {number}: {ac} is not an AC row of docs/SPEC.md §6"


def test_traceability_table_paths_exist() -> None:
    refs = [(first, ref) for first, cell in TRACE_ROWS for ref in _references(cell)]
    assert refs
    for first, (path, name) in refs:
        _exists(path, name, f"docs/SPEC.md §9 {first}")
