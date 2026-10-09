# AI 작성 — 2026-10-09 위임, 결함 주입 검토 후 고정 (docs/prompts/delegation_compare.md §3).
# 수정은 사람이 승인한다.
"""SPEC AC13 (verdict) and AC14 (refusal) golden cases: the PPL comparison of two evaluation records."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CASES = yaml.safe_load((ROOT / "tests/harness/golden_compare.yaml").read_text(encoding="utf-8"))["cases"]
CASE_KEYS = {"id", "ac", "kind", "input", "expected", "note"}
INPUT_KEYS = {"base", "emu", "limit"}
EXPECTED_KEYS = {"pass", "relative", "error"}


def _compare_ppl() -> Callable[..., dict[str, Any]]:
    # Only an absent module skips; a module that exists but fails to import is a failure.
    if importlib.util.find_spec("tricast.eval.compare") is None:
        pytest.skip("tricast.eval.compare is not implemented yet (AC13)")
    return importlib.import_module("tricast.eval.compare").compare_ppl


def _relative(actual: Any, expected: float) -> None:
    assert isinstance(actual, float), type(actual)
    if math.isinf(expected):
        assert actual == expected, actual
    else:
        assert abs(actual - expected) <= 1e-12 * abs(expected), (actual, expected)


def test_compare_manifest() -> None:
    assert CASES
    ids = [case.get("id") for case in CASES]
    assert len(ids) == len(set(ids)), ids
    for case in CASES:
        assert case.keys() == CASE_KEYS, case.get("id")
        assert case["ac"] in {"AC13", "AC14"}, case["id"]
        assert case["kind"] in {"normal", "boundary", "forbidden"}, case["id"]
        assert case["ac"] in case["note"], case["id"]
        assert {"base", "emu"} <= case["input"].keys() <= INPUT_KEYS, case["id"]
        expected = case["expected"]
        assert expected and expected.keys() <= EXPECTED_KEYS, case["id"]
        assert "error" not in expected or expected.keys() == {"error"}, case["id"]


@pytest.mark.parametrize("case", CASES, ids=[str(case.get("id")) for case in CASES])
def test_compare_golden(case: dict[str, Any]) -> None:
    compare_ppl = _compare_ppl()
    data = copy.deepcopy(case["input"])  # YAML anchors share one dict across cases
    kwargs = {"limit": data["limit"]} if "limit" in data else {}
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError, match=r"^" + re.escape(expected["error"]) + ":"):
            compare_ppl(data["base"], data["emu"], **kwargs)
        return
    result = compare_ppl(data["base"], data["emu"], **kwargs)
    if "limit" in data:
        assert result["limit"] == data["limit"], result
    if "pass" in expected:
        assert result["pass"] is expected["pass"], result
    if "relative" in expected:
        _relative(result["relative"], expected["relative"])
