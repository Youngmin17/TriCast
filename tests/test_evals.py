"""AI 작성 초안 — 2026-10-01 팀 승인

AC6 evaluation criteria are reviewable annotations, not parser-derived labels.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_evals", ROOT / "evals/run_evals.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


@pytest.fixture
def schema() -> dict[str, Any]:
    return {
        "type": "object", "required": ["intent", "model"],
        "properties": {
            "intent": {"enum": ["evaluate", "compare", "explain", "sweep"]},
            "model": {"type": ["string", "null"]},
            "recipes": {"type": "array", "items": {"$ref": "#/$defs/recipe"}},
            "tasks": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array"}, "questions": {"type": "array"},
        },
        "$defs": {"recipe": {"type": "object", "properties": {
            "mma": {"type": "object", "properties": {"f_bits": {"type": "integer"},
                                                      "g_bits": {"type": "integer"}}},
        }}},
    }


def fake_result(document: Any, *, errors: list[str] | None = None) -> SimpleNamespace:
    request = None if document is None else SimpleNamespace(to_dict=lambda: copy.deepcopy(document))
    return SimpleNamespace(request=request, errors=errors or [], source="offline", raw=None)


def schema_has_path(schema: dict[str, Any], path: str) -> bool:
    """Follow declared fields and local schema references, not arbitrary object keys."""
    def walk(node: Any, parts: list[str | int]) -> bool:
        if not isinstance(node, dict):
            return False
        if not parts:
            return True
        if "$ref" in node:
            assert node["$ref"].startswith("#/"), "eval schema references must be local"
            target = schema
            for key in node["$ref"][2:].split("/"):
                target = target[key.replace("~1", "/").replace("~0", "~")]
            if walk(target, parts):
                return True
        if any(walk(branch, parts) for op in ("anyOf", "oneOf", "allOf") for branch in node.get(op, [])):
            return True
        part, *tail = parts
        if isinstance(part, int):
            prefix = node.get("prefixItems", [])
            return walk(prefix[part] if part < len(prefix) else node.get("items"), tail)
        return part in node.get("properties", {}) and walk(node["properties"][part], tail)

    return walk(schema, RUN.path_parts(path))


def test_evalset_format() -> None:
    items = RUN.load_evalset()
    assert len(items) == 30
    assert len({item["id"] for item in items}) == 30
    assert Counter(item["language"] for item in items) == {"ko": 18, "en": 12}
    assert Counter(item["difficulty"] for item in items) == {"easy": 10, "medium": 12, "hard": 8}
    required = {"id", "text", "expected", "must_be_null", "ambiguous", "note"}
    for item in items:
        assert required <= item.keys()
        assert item["_draft"] == RUN.DRAFT
        assert isinstance(item["id"], str) and item["id"].strip()
        assert isinstance(item["text"], str) and item["text"].strip()
        assert isinstance(item["expected"], dict) and item["expected"]
        assert isinstance(item["ambiguous"], bool)
        assert isinstance(item["note"], str) and "AC6" in item["note"]
        assert isinstance(item["must_be_null"], list)
        assert len(set(item["must_be_null"])) == len(item["must_be_null"])
        assert not set(item["expected"]) & set(item["must_be_null"])
        for path in [*item["expected"], *item["must_be_null"]]:
            RUN.path_parts(path)


def test_evalset_fields_exist_in_request_schema() -> None:
    pytest.importorskip("tricast.agent.parser")
    schema = json.loads(RUN.SCHEMA.read_text(encoding="utf-8"))
    for item in RUN.load_evalset():
        for path in [*item["expected"], *item["must_be_null"]]:
            assert schema_has_path(schema, path), f"{item['id']}: unknown schema field {path}"


@pytest.mark.parametrize("path,expected", [
    ("recipes[0].mma.f_bits", 13), ("recipes[1].mma", None),
    ("recipes[0].mma.g_bits", None), ("model.revision", None),
])
def test_get_path(path: str, expected: Any) -> None:
    assert RUN.get_path({"recipes": [{"mma": {"f_bits": 13}}], "model": None}, path) == expected


@pytest.mark.parametrize("path", ["", "recipes[-1]", "recipes..mma", "recipes[0]junk", "recipes[*].mma"])
def test_invalid_path_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="invalid field path"):
        RUN.get_path({}, path)


@pytest.mark.parametrize("assumptions,covered", [
    (["Defaults applied"], False), (["f_bits=13"], False), (["recipes[0].mma.f_bits=13"], True),
    (["recipes[0].mma.f_bits_extra=13"], False), (["recipes[1].mma.f_bits=13"], False),
    ([{"path": "recipes[0].mma.f_bits", "value": 13, "reason": "default"}], True),
    ([{"path": "recipes[0].mma", "reason": "defaults"}], False),
])
def test_assumption_requires_exact_path(assumptions: list[Any], covered: bool) -> None:
    assert RUN.assumption_covers(assumptions, "recipes[0].mma.f_bits") is covered


def test_metric_counts_hand_derived(schema: dict[str, Any]) -> None:
    first = {"id": "one", "expected": {"intent": "compare", "model": "org/model", "tasks": ["hellaswag"]},
             "must_be_null": ["recipes[0].mma.f_bits", "recipes[0].mma.g_bits"], "ambiguous": True}
    document = {"intent": "compare", "model": "org/model", "tasks": ["hellaswag"],
                "recipes": [{"mma": {"f_bits": 13, "g_bits": 6}}],
                "assumptions": ["recipes[0].mma.g_bits=6 (default)"], "questions": ["Which precision?"]}
    second = {"id": "two", "expected": {"intent": "evaluate", "model": "org/model"},
              "must_be_null": ["recipes[0].mma.f_bits"], "ambiguous": True}
    rows = [RUN.score_item(first, fake_result(document), schema),
            RUN.score_item(second, fake_result({"intent": "explain", "questions": []}), schema)]
    summary = RUN.summarize(rows)
    # AC6: 1/2 schema valid; 3/5 fields correct; 1/3 checked paths invented; 1/2 questions asked.
    assert summary["schema_valid_rate"] == 1 / 2
    assert summary["field_accuracy"] == 3 / 5
    assert summary["invention_rate"] == 1 / 3
    assert summary["ambiguity_recall"] == 1 / 2
    assert rows[0]["inventions"] == ["recipes[0].mma.f_bits"]
    assert rows[1]["fields"]["model"]["missing"]


def test_empty_denominators_are_not_perfect_scores() -> None:
    result = RUN.summarize([])
    assert all(result[key] is None for key in (
        "schema_valid_rate", "field_accuracy", "invention_rate", "ambiguity_recall",
    ))


def test_parse_failure_counts_as_failed_item(schema: dict[str, Any]) -> None:
    item = {"id": "failed", "text": "parse this", "expected": {"intent": "evaluate"},
            "must_be_null": [], "ambiguous": True}

    def broken_parse(text: str, *, llm: str) -> None:
        raise RuntimeError("sensitive provider headers must not be saved")

    result = RUN.evaluate_items([item], broken_parse, schema)
    assert result["schema_valid_rate"] == 0
    assert result["field_accuracy"] == 0
    assert result["ambiguity_recall"] == 0
    assert result["invention_rate"] is None
    assert result["per_item"][0]["parse_errors"] == ["parser raised RuntimeError"]


def test_result_errors_do_not_replace_schema_validation(schema: dict[str, Any]) -> None:
    item = {"id": "invalid", "expected": {"model": 1}, "must_be_null": [], "ambiguous": True}
    row = RUN.score_item(item, fake_result({"intent": "evaluate", "model": True, "questions": [""]}), schema)
    assert not row["schema_valid"]
    assert row["field_correct"] == 0  # JSON true is not the expected numeric 1.
    assert not row["asked_question"]


def test_schema_path_resolver_checks_refs(schema: dict[str, Any]) -> None:
    assert schema_has_path(schema, "recipes[0].mma.f_bits")
    assert not schema_has_path(schema, "recipes[0].mma.f_bitss")
    assert not schema_has_path(schema, "model.revision")


def test_anthropic_without_key_is_friendly(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        RUN.main(["--llm", "anthropic"])
    assert exc.value.code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_limit_must_be_positive() -> None:
    with pytest.raises(SystemExit) as exc:
        RUN.main(["--limit", "0"])
    assert exc.value.code == 2


def test_offline_end_to_end(tmp_path: Path, capsys: Any) -> None:
    pytest.importorskip("tricast.agent.parser")
    destination = tmp_path / "offline.json"
    assert RUN.main(["--llm", "offline", "--out", str(destination)]) == 0
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["counts"]["items"] == 30
    assert len(report["per_item"]) == 30
    assert report["schema_valid_rate"] == 1.0
    assert report["model"] == "offline"
    assert len(report["evalset_sha256"]) == 64
    assert len(report["schema_sha256"]) == 64
    assert report["git_sha"]
    assert report["started_at"] <= report["finished_at"]
    assert "invention_rate" in capsys.readouterr().out
    # A smoke test checks wiring, not an invented 100% semantic quality threshold.
    for row in report["per_item"]:
        assert row["request"] is not None
        assert not row["parse_errors"]


def test_returned_anthropic_error_details_are_not_saved(schema: dict[str, Any]) -> None:
    item = {"id": "auth-failure", "expected": {"intent": "evaluate"},
            "must_be_null": [], "ambiguous": False}
    result = fake_result(None, errors=["anthropic: AuthenticationError: x-api-key=SYNTHETIC_SECRET"])
    result.source = "anthropic"
    row = RUN.score_item(item, result, schema)
    assert "SYNTHETIC_SECRET" not in json.dumps(row)
    assert row["parse_errors"] == ["anthropic parse/validation error (details omitted)"]
    assert not row["schema_valid"]


def test_existing_output_is_rejected_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "existing.json"
    destination.write_text("previous evidence", encoding="utf-8")

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> None:
        pytest.fail("an existing output must be rejected before evaluating")

    monkeypatch.setattr(RUN, "evaluate_items", unexpected_evaluation)
    with pytest.raises(SystemExit) as exc:
        RUN.main(["--llm", "offline", "--out", str(destination)])
    assert exc.value.code == 2
    assert destination.read_text(encoding="utf-8") == "previous evidence"


def test_cli_snapshots_inputs_and_records_requested_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: dict[str, Any],
) -> None:
    import hashlib

    parser = pytest.importorskip("tricast.agent.parser")
    item = {"id": "one", "text": "test request", "expected": {"intent": "evaluate"},
            "must_be_null": [], "ambiguous": False}
    evalset_path, schema_path = tmp_path / "evalset.jsonl", tmp_path / "schema.json"
    evalset_bytes = (json.dumps(item) + "\n").encode()
    schema_bytes = json.dumps(schema).encode()
    evalset_path.write_bytes(evalset_bytes)
    schema_path.write_bytes(schema_bytes)
    monkeypatch.setattr(RUN, "EVALSET", evalset_path)
    monkeypatch.setattr(RUN, "SCHEMA", schema_path)
    monkeypatch.setattr(
        RUN, "_git", lambda *args: "initial-sha" if args == ("rev-parse", "HEAD") else " M file",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-key-not-a-credential")
    monkeypatch.setattr(parser, "request_options", lambda: {"model": "fixture-model"})
    calls = []

    def fake_parse(text: str, *, llm: str) -> SimpleNamespace:
        calls.append((text, llm))
        # Simulate another lane changing files while a provider call is in flight.
        evalset_path.write_text("changed", encoding="utf-8")
        schema_path.write_text("changed", encoding="utf-8")
        monkeypatch.setattr(RUN, "_git", lambda *args: "changed-sha")
        result = fake_result({"intent": "evaluate", "model": None})
        result.source = "anthropic"
        return result

    monkeypatch.setattr(parser, "parse_request", fake_parse)
    output = tmp_path / "result.json"
    assert RUN.main(["--llm", "anthropic", "--limit", "1", "--out", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [("test request", "anthropic")]
    assert report["git_sha"] == "initial-sha" and report["git_dirty"] is True
    assert report["evalset_sha256"] == hashlib.sha256(evalset_bytes).hexdigest()
    assert report["schema_sha256"] == hashlib.sha256(schema_bytes).hexdigest()
    assert report["model"] == "fixture-model" and report["model_is_requested"] is True
    assert report["schema_valid_rate"] == 1 and report["field_accuracy"] == 1
    assert "synthetic-test-key-not-a-credential" not in output.read_text(encoding="utf-8")
