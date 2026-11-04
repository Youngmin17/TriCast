"""Layer cards and their retrieval: a layer profile becomes ledger records and a line of the layer's
document, the retrieval of a layer query returns that layer's document with its record ids, the cards join
table, profile and ids, and the tune loop's analysis carries the retrieved documents unless a mask hides
them. CPU only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app import layer_cards as lc
from app import ledger as lg
from app import quantizer_recipe as qr
from app import tune_loop as tl
from app.export_quantizer_demo import SHAPES
from app.tests.test_tune_agent import ENV
from app.tests.test_tune_loop import _session
from tricast.rag import search_evidence


def _profile(path: Path, model_sha: str = "c1899de") -> Path:
    layers = []
    for block in range(qr.BLOCKS):
        for index, kind in enumerate(qr.KINDS):
            out_features, in_features = SHAPES[kind]
            scale = 1 + block / 10 + index / 100
            layers.append({"name": qr.layer_name(block, kind), "block": block, "kind": kind,
                           "params": out_features * in_features, "in_features": in_features,
                           "out_features": out_features,
                           "weight": {stat: scale for stat in lg.PROFILE_STATS["weight"]},
                           "activation": {stat: 2 * scale for stat in lg.PROFILE_STATS["activation"]},
                           "sparsity": {rule: scale / 10 for rule in lg.PROFILE_STATS["sparsity"]}})
    meta = {**ENV, "model_id": "Qwen/Qwen3-0.6B", "model_sha": model_sha, "windows": 8, "seqlen": 512,
            "seed": 42, "source": {"dataset": "wikitext2", "split": "train"}, "env": ENV}
    path.write_text(json.dumps({"schema": "tricast.layer_profile/1", "status": "complete", "meta": meta,
                                "layers": layers}))
    return path


def _documented(tmp_path: Path) -> tuple[dict, Path]:
    config, ledger = _session(tmp_path)
    _profile(tmp_path / "profile.json")
    lg.ingest_and_document([tmp_path / "table.json", tmp_path / "profile.json"], ledger, tmp_path)
    return config, ledger


def test_a_profile_becomes_records_and_the_layer_document_retrieval_finds(tmp_path: Path) -> None:
    _, ledger = _documented(tmp_path)
    stats = [record for record in lg.read_ledger(ledger) if record["kind"] == "layer_stat"]
    per_layer = sum(len(names) for names in lg.PROFILE_STATS.values())
    assert len(stats) == qr.BLOCKS * len(qr.KINDS) * per_layer
    assert all(lg.verify(record, tmp_path) for record in stats[:40])  # each number reads back from its file
    docs = [str(lg.docs_path(ledger))]
    hits = search_evidence("layer model.layers.27.mlp.down_proj nvfp4", 2, ledgers=docs, catalogs=[])
    assert hits[0]["section"] == "layer model.layers.27.mlp.down_proj"
    text = hits[0]["text"]
    assert "layer profile" in text and "sparsity wanda_50" in text and "single-layer KL" in text
    wanda = next(record for record in stats if record["layer"] == "model.layers.27.mlp.down_proj"
                 and record["metric"] == "wanda_50")
    assert f"[{wanda['id']}]" in text


def test_cards_join_the_table_the_profile_and_the_record_ids(tmp_path: Path) -> None:
    _, ledger = _documented(tmp_path)
    payload = lc.cards(tmp_path / "table.json", tmp_path / "profile.json", ledger, tmp_path)
    assert len(payload["layers"]) == qr.BLOCKS * len(qr.KINDS) and payload["profile"]["seed"] == 42
    card = payload["layers"][-1]
    assert card["name"] == "model.layers.27.mlp.down_proj" and card["out_features"] * card["in_features"] == \
        card["params"]
    nvfp4 = card["methods"]["nvfp4"]
    assert nvfp4["bits"] == 4.5 and nvfp4["bytes"] == card["params"] * 4.5 / 8 and nvfp4["id"]
    assert set(nvfp4["options"]) == set(qr.METHOD_OPTIONS["nvfp4"]) and card["floor_id"]
    assert card["profile"]["sparsity"]["wanda_50"] == pytest.approx((1 + 2.7 + 0.06) / 10)
    assert all(card["profile_ids"].values())
    with pytest.raises(ValueError, match="another model revision"):
        lc.cards(tmp_path / "table.json", _profile(tmp_path / "other.json", "deadbee"), ledger, tmp_path)


def test_the_analysis_retrieves_the_loss_layers_documents_unless_masked(tmp_path: Path) -> None:
    config, _ = _documented(tmp_path)
    analysis = tl.analyze(config, tmp_path)
    retrieved = analysis["retrieved"]
    assert len(retrieved) == tl.RAG_LAYERS
    for item, layer in zip(retrieved, analysis["loss_concentration"], strict=False):
        assert item["query"].startswith(f"layer {layer['layer']} ")
        assert item["documents"][0]["section"] == f"layer {layer['layer']}"
        assert "[E" in item["documents"][0]["text"]
    seen = {}

    def runner(command, **kwargs):
        seen["prompt"] = kwargs["input"]
        reply = {"analysis": "a", "stop": True, "proposals": []}
        out = json.dumps({"type": "result", "structured_output": reply})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")
    for mask, shown in ((None, True), ("no-rag", False), ("anonymous", False)):
        masked = {**config, "llm": {"mask": [mask]}} if mask else config
        tl.claude_code(masked, tmp_path, runner)(analysis, 1, 1)
        assert ('"retrieved"' in seen["prompt"]) is shown, mask
