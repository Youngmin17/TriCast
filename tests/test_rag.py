"""BM25 retrieval cites the local numerical contract and provenance."""

import json

import pytest

from tricast.rag import search


def test_hopper_retrieves_preset_provenance() -> None:
    results = search("Hopper F=13", k=3)
    assert any(item["section"] == "nvidia_hopper_fp8" for item in results)
    assert all(set(item) == {"source", "section", "text", "score"} for item in results)
    assert results == search("Hopper F=13", k=3)
    assert all(item["score"] > 0 for item in results)
    json.dumps(results, allow_nan=False)


def test_rag_indexes_contract_schemes_and_glossary() -> None:
    assert any("ENGINE.md" in item["source"] for item in search("CoFDA FDA primitive", 10))
    assert any(item["section"] == "nvfp4" for item in search("nvfp4", 10))
    assert any(item["source"] == "AGENTS.md" for item in search("EmulationRequest assumptions", 10))


def test_search_empty_and_limits() -> None:
    assert search("", 5) == []
    assert search("unmatchedzzznumericalterm", 5) == []
    assert search("fp8", 0) == []
    assert len(search("fp8", 2)) <= 2
    with pytest.raises(ValueError, match="k"):
        search("fp8", -1)


def test_rag_indexes_kv_presets_and_error_report_contract() -> None:
    results = search("kivi2 residual cache key_axis", 10)
    assert any(item["section"] == "KV kivi2" for item in results)
    assert any("Error analysis" in item["section"] for item in search("logits KL SQNR cosine", 10))
