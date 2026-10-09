"""docs/ontology.yaml stays traceable: every class, attribute and relation cites interview logs that exist."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = yaml.safe_load((ROOT / "docs/ontology.yaml").read_text(encoding="utf-8"))
INTERVIEWS = (ROOT / "docs/research/interviews.md").read_text(encoding="utf-8")
SPEC = (ROOT / "docs/SPEC.md").read_text(encoding="utf-8")
LOG_ROWS = re.findall(r"^\| (\d{2}) \|", INTERVIEWS, re.M)
EVIDENCE = re.compile(r"^로그 (\d{2})$")


def _relations(body: dict) -> list[tuple[str, str, list[str]]]:
    out = []
    for relation in body.get("relations") or []:
        (verb, target), = [(key, value) for key, value in relation.items() if key != "evidence"]
        out.append((verb, target, relation.get("evidence") or []))
    return out


def _cited(evidence: list[str], where: str) -> None:
    assert evidence, f"{where}: no evidence"
    for item in evidence:
        match = EVIDENCE.match(item)
        assert match, f"{where}: {item!r} is not '로그 NN'"
        assert match.group(1) in LOG_ROWS, f"{where}: {item} is not a row of docs/research/interviews.md"


def test_interview_log_table_is_numbered_once() -> None:
    assert LOG_ROWS and len(LOG_ROWS) == len(set(LOG_ROWS))


def test_every_item_cites_an_existing_log() -> None:
    for name, body in ONTOLOGY["classes"].items():
        _cited(body.get("evidence") or [], name)
        for attribute, spec in (body.get("attributes") or {}).items():
            if not spec.get("key"):
                _cited(spec.get("evidence") or [], f"{name}.{attribute}")
        for verb, target, evidence in _relations(body):
            _cited(evidence, f"{name} {verb} {target}")


def test_relations_target_defined_classes() -> None:
    names = set(ONTOLOGY["classes"])
    for name, body in ONTOLOGY["classes"].items():
        for verb, target, _ in _relations(body):
            assert target in names, f"{name} {verb} {target}: undefined class"


def test_scope_partitions_classes_and_cites_logs() -> None:
    scope = ONTOLOGY["scope"]
    managed = scope["managed"]
    external = scope["external_reference"]["classes"]
    assert sorted(managed + external) == sorted(ONTOLOGY["classes"]), "each class sits in one scope tier"
    for item in scope["out_of_scope"]:
        assert item["reason"], item["item"]
        _cited(item["evidence"], f"out_of_scope {item['item']}")


def test_synonyms_are_nonempty_strings() -> None:
    for name, body in ONTOLOGY["classes"].items():
        attributes = (body.get("attributes") or {}).values()
        lists = [body.get("synonyms")] + [spec.get("synonyms") for spec in attributes]
        for synonyms in filter(None, lists):
            assert all(isinstance(word, str) and word.strip() for word in synonyms), name


def test_out_of_scope_matches_spec_exclusions() -> None:
    """Ontology out-of-scope items and SPEC §4's excluded column name the same things."""
    section = SPEC[SPEC.index("## 4. 범위"):SPEC.index("## 5.")]
    excluded = [row.split("|")[2].strip() for row in section.splitlines()
                if row.startswith("| ") and not row.startswith(("| ✅", "|---"))]
    items = [entry["item"] for entry in ONTOLOGY["scope"]["out_of_scope"]]
    for item in items:
        assert any(item in cell for cell in excluded), f"{item}: not in SPEC §4 excluded column"
    for cell in excluded:
        assert any(item in cell for item in items), f"SPEC §4 excluded {cell!r}: no ontology item"



ROWS = {match.group(1): match.group(0)
        for match in re.finditer(r"^\| (\d{2}) \|.*$", INTERVIEWS, re.M)}
QUOTE = re.compile(r'"([^"]+)"')
TAG = re.compile(r"`([^`]+)`")


def _table(text: str, heading: str) -> list[list[str]]:
    section = text[text.index(heading):]
    section = section[:section.index("\n#", 1)]
    lines = section.splitlines()
    rows = [line for line, after in zip(lines, lines[1:] + [""], strict=True)
            if line.startswith("| ") and not after.startswith(("|---", "|:-"))]  # drop header rows
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]


def _cited_in_rows(cell: str, logs: list[str], allowed: set[str], where: str) -> None:
    text = " ".join(ROWS[log] for log in logs)
    for quote in QUOTE.findall(cell):
        assert quote in text, f"{where}: quote not in interview logs {logs}: {quote}"
    for tag in TAG.findall(cell):
        assert tag in allowed or f"`{tag}`" in text, f"{where}: `{tag}` not in interview logs {logs}"


def test_class_trace_table_quotes_interview_rows() -> None:
    """ontology.md §4-4: every quote and tag comes from a cited log, and the log is the class's evidence."""
    vocabulary = set(ONTOLOGY["classes"])
    vocabulary |= {name for body in ONTOLOGY["classes"].values() for name in (body.get("attributes") or {})}
    rows = _table((ROOT / "docs/ontology.md").read_text(encoding="utf-8"), "### 4-4.")
    assert {row[0].strip("`") for row in rows} == set(ONTOLOGY["classes"])
    for name, logs, cell in rows:
        logs = re.findall(r"\d{2}", logs)
        evidence = ONTOLOGY["classes"][name.strip("`")]["evidence"]
        assert all(f"로그 {log}" in evidence for log in logs), (name, logs)
        _cited_in_rows(cell, logs, vocabulary | {"granularity", "rounding"}, f"ontology.md §4-4 {name}")


def test_problem_evidence_tables_quote_interview_rows() -> None:
    """PROBLEM.md §3-0: behaviour evidence and the unaffected group quote the interview log table verbatim."""
    rows = _table((ROOT / "docs/PROBLEM.md").read_text(encoding="utf-8"), "### 3-0.")
    assert len(rows) >= 10
    for row in rows:
        _cited_in_rows(" ".join(row[1:]), [row[0]], set(), f"PROBLEM.md §3-0 로그 {row[0]}")
