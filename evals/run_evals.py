"""AI 작성 초안 — 2026-10-01 팀 승인

Score request parsing without running models or arithmetic evaluations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
EVALSET = Path(__file__).with_name("evalset.jsonl")
SCHEMA = ROOT / "src/tricast/schemas/emulation_request.schema.json"
DRAFT = "AI 작성 초안 — 2026-10-01 팀 승인"
_MISSING = object()


def path_parts(path: str) -> list[str | int]:
    """Split dotted fields with explicit zero-based array indices."""
    if not re.fullmatch(r"[A-Za-z_]\w*(?:(?:\[\d+\])|(?:\.[A-Za-z_]\w*))*", path):
        raise ValueError(f"invalid field path: {path!r}")
    return [int(index) if index else name for name, index in re.findall(r"([A-Za-z_]\w*)|\[(\d+)\]", path)]


def get_path(document: Any, path: str, default: Any = None) -> Any:
    """Read a field; missing ancestors and null parents use ``default``."""
    value = document
    for part in path_parts(path):
        if isinstance(part, int):
            if not isinstance(value, list) or part >= len(value):
                return default
        elif not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def assumption_covers(assumptions: list[Any], path: str) -> bool:
    """Require an explicit field path, never a generic mention of defaults."""
    for assumption in assumptions:
        if isinstance(assumption, dict) and assumption.get("path") == path:
            return True
        if isinstance(assumption, str) and re.search(
            rf"(?<![\w.\[\]]){re.escape(path)}(?![\w.\[\]])", assumption,
        ):
            return True
    return False


def load_evalset(path: Path = EVALSET, *, text: str | None = None) -> list[dict[str, Any]]:
    """Load genuine JSONL: the draft notice is a field, not an invalid comment."""
    text = path.read_text(encoding="utf-8") if text is None else text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def score_item(item: dict[str, Any], result: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """Score a ParseResult-compatible object against explicitly annotated fields."""
    request = None if result is None or result.request is None else result.request.to_dict()
    validator = Draft202012Validator(schema)
    schema_errors = [error.message for error in validator.iter_errors(request)]
    fields = {}
    for path, expected in item["expected"].items():
        actual = get_path(request, path, _MISSING)
        # JSON booleans are not the numbers 0/1; object key order is irrelevant.
        match = actual is not _MISSING and json.dumps(actual, sort_keys=True) == json.dumps(
            expected, sort_keys=True,
        )
        fields[path] = {"expected": expected, "actual": None if actual is _MISSING else actual,
                        "missing": actual is _MISSING, "match": match}
    assumptions = request.get("assumptions", []) if isinstance(request, dict) else []
    if not isinstance(assumptions, list):
        assumptions = []
    inventions = [path for path in item["must_be_null"]
                  if get_path(request, path) is not None and not assumption_covers(assumptions, path)]
    questions = request.get("questions", []) if isinstance(request, dict) else []
    asked = isinstance(questions, list) and any(isinstance(q, str) and q.strip() for q in questions)
    return {
        "id": item["id"], "schema_valid": not schema_errors, "schema_errors": schema_errors,
        "fields": fields, "field_correct": sum(field["match"] for field in fields.values()),
        "field_total": len(fields), "inventions": inventions, "null_total": len(item["must_be_null"]),
        "ambiguous": item["ambiguous"], "asked_question": asked,
        "source": None if result is None else result.source,
        "parse_errors": ([] if result is None else
                         ["anthropic parse/validation error (details omitted)" for _ in result.errors]
                         if result.source == "anthropic" else list(result.errors)),
        "request": request,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(per_item: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average fields and checked null paths; undefined rates are null."""
    counts = {
        "items": len(per_item), "schema_valid": sum(row["schema_valid"] for row in per_item),
        "field_correct": sum(row["field_correct"] for row in per_item),
        "field_total": sum(row["field_total"] for row in per_item),
        "inventions": sum(len(row["inventions"]) for row in per_item),
        "null_total": sum(row["null_total"] for row in per_item),
        "ambiguous": sum(row["ambiguous"] for row in per_item),
        "ambiguity_detected": sum(row["ambiguous"] and row["asked_question"] for row in per_item),
    }
    return {
        "schema_valid_rate": _rate(counts["schema_valid"], counts["items"]),
        "field_accuracy": _rate(counts["field_correct"], counts["field_total"]),
        "invention_rate": _rate(counts["inventions"], counts["null_total"]),
        "ambiguity_recall": _rate(counts["ambiguity_detected"], counts["ambiguous"]),
        "counts": counts,
    }


def evaluate_items(
    items: list[dict[str, Any]], parse: Callable[..., Any], schema: dict[str, Any], *, llm: str = "offline",
) -> dict[str, Any]:
    """Keep parser failures in the denominator and record them per item."""
    Draft202012Validator.check_schema(schema)
    per_item = []
    for item in items:
        try:
            result = parse(item["text"], llm=llm)
        except Exception as exc:
            row = score_item(item, None, schema)
            # Exception messages can contain provider credentials or request headers.
            row["parse_errors"] = [f"parser raised {type(exc).__name__}"]
        else:
            row = score_item(item, result, schema)
        per_item.append(row)
    return {**summarize(per_item), "per_item": per_item}


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", choices=("offline", "anthropic"), default="offline")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.out is not None and args.out.exists():
        parser.exit(2, f"결과 파일이 이미 있습니다. 새 --out 경로를 사용하세요: {args.out}\n")
    if args.llm == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.exit(2, "ANTHROPIC_API_KEY가 없습니다. 키를 환경에 설정하거나 --llm offline을 사용하세요.\n")
    try:
        from tricast.agent import parser as request_parser
    except ModuleNotFoundError as exc:
        parser.exit(2, f"파서 또는 의존 모듈({exc.name})이 없습니다. Lane AG 통합과 설치를 확인하세요.\n")
    if not SCHEMA.is_file():
        parser.exit(2, "EmulationRequest 스키마가 없습니다. Lane AG 통합을 먼저 완료하세요.\n")
    schema_bytes, evalset_bytes = SCHEMA.read_bytes(), EVALSET.read_bytes()
    schema = json.loads(schema_bytes)
    items = load_evalset(text=evalset_bytes.decode("utf-8"))
    model = "offline" if args.llm == "offline" else request_parser.request_options()["model"]
    if args.limit is not None:
        items = items[:args.limit]
    started = datetime.now(timezone.utc).isoformat()
    git_sha, git_status = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    report = evaluate_items(items, request_parser.parse_request, schema, llm=args.llm)
    report = {"draft": DRAFT, "git_sha": git_sha,
              "git_dirty": bool(git_status) if git_status is not None else None,
              "llm": args.llm, "model": model, "model_is_requested": args.llm == "anthropic",
              "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
              "evalset_sha256": hashlib.sha256(evalset_bytes).hexdigest(),
              "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(), **report}
    print("metric                 | value")
    print("-----------------------|-------")
    for name in ("schema_valid_rate", "field_accuracy", "invention_rate", "ambiguity_recall"):
        value = report[name]
        print(f"{name:22} | {'N/A' if value is None else f'{value:.4f}'}")
    print(f"items                  | {len(items)}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"result                 | {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
