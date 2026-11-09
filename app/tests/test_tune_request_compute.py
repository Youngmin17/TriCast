# AI 작성 — 2026-11-09, 독립 검토 지적(데모 표로는 연산 예산 경로를 가를 수 없음). 수정은 사람 승인.
"""SPEC AC28: a compute budget below the least MAC weight the request's constraints allow is a computed
conflict. A two-layer table whose methods have different B200 MAC weights (nvfp4 1/4, int8 1/2, weight-only
gptq_int4 1, bf16 1, nvint4 unrated), so a constant, the largest weight or ignored constraints all give the
wrong number; the demo table cannot tell them apart (all its methods run a BF16 GEMM)."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from app import tune_request as tr

MODEL = "Qwen/Qwen3-0.6B"
METHODS = ["bf16", "gptq_int4", "nvfp4", "int8", "nvint4"]


def _table() -> dict[str, Any]:
    """Layer 0 (3 parameters) has every method; layer 1 (1 parameter) only bf16, gptq_int4 and nvint4."""
    def layer(block: int, params: int, value: list[float | None]) -> dict[str, Any]:
        return {"block": block, "kind": "q_proj", "name": f"model.layers.{block}.self_attn.q_proj",
                "params": params, "in_features": 1024, "value": list(value), "kl": list(value)}
    return {"blocks": 2, "methods": METHODS, "meta": {"model_id": MODEL},
            "layers": [layer(0, 3, [0.0, 0.1, 0.2, 0.15, 0.3]), layer(1, 1, [0.0, 0.1, None, None, 0.3])]}


def _reply(**fields: Any) -> dict[str, Any]:
    base = {"outcome": "tune", "model": MODEL, "budget_bits": "16", "compute_budget": None,
            "keep_bf16_blocks": [], "exclude_methods": [], "only_methods": None, "questions": [],
            "reason": None, "assumptions": []}
    return {**base, **fields}


@pytest.mark.parametrize("request_fields, least", [
    ({}, Fraction(7, 16)),  # (3 x 1/4 nvfp4 + 1 x 1 gptq_int4 or bf16) / 4; nvint4 is unrated, not counted
    ({"only_methods": ["int8"]}, Fraction(5, 8)),  # (3 x 1/2 + 1 x 1 bf16) / 4: bf16 stays allowed
    ({"exclude_methods": ["nvfp4"]}, Fraction(5, 8)),  # int8 is then layer 0's least
    ({"keep_bf16_blocks": [0]}, Fraction(1)),
    ({"exclude_methods": ["bf16", "gptq_int4", "nvfp4", "int8"]}, None),  # layer 1 has only unrated nvint4
])
def test_cheapest_compute_is_the_least_rated_mac_weight_under_the_constraints(
        request_fields: dict[str, Any], least: Fraction | None) -> None:
    assert tr.cheapest_compute(tr.constrained(_table(), _reply(**request_fields))) == least


@pytest.mark.parametrize("request_fields, outcome, reason", [
    ({"compute_budget": "1/2"}, "tune", None),
    ({"compute_budget": "7/16"}, "tune", None),  # equal to the least: met
    ({"compute_budget": "3/8"}, "conflict", "the cheapest compute these constraints allow is 0.4375"),
    ({"compute_budget": "1/2", "only_methods": ["int8"]}, "conflict", "is 0.6250"),
    ({"compute_budget": "1/4", "exclude_methods": ["bf16", "gptq_int4", "nvfp4", "int8"]}, "tune", None),
    ({"compute_budget": "1/4", "budget_bits": None}, "needs_clarification", None),  # asked before computed
])
def test_a_compute_budget_below_the_least_is_a_computed_conflict(
        request_fields: dict[str, Any], outcome: str, reason: str | None) -> None:
    result = tr.check(_reply(**request_fields), _table(), [MODEL])
    assert result["outcome"] == outcome, result
    if reason is not None:
        assert reason in result["reason"] and result["reason"].endswith("(code)")
    if request_fields.get("exclude_methods") == ["bf16", "gptq_int4", "nvfp4", "int8"]:
        assert any("not checked" in note for note in result["checks"])


@pytest.mark.parametrize("field, value", [
    ("budget_bits", "1/0"), ("budget_bits", "1.5/2"), ("compute_budget", "3/0"), ("compute_budget", "0.5/2"),
])
def test_a_budget_the_schema_lets_through_but_no_number_reads_is_refused(field: str, value: str) -> None:
    """The schema pattern accepts these; Fraction does not. A RequestError (422), never a 500."""
    with pytest.raises(tr.RequestError, match=rf"^{field}: "):
        tr.check(_reply(**{field: value}), _table(), [MODEL])
