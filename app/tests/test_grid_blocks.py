"""Grids for models deeper than the page's 28 blocks (Llama-3.1-8B 32, Qwen3-8B 36): one row per decoder
block, rules that reproduce the grid, solve over a deeper table, and the layer map of a deeper model. Grids of
28 blocks are unchanged (the AC15 page tests). CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import autotune as at
from app import quantizer_recipe as qr
from app.tests.test_tune_agent import _table


def test_a_deeper_grid_builds_and_its_rules_reproduce_it() -> None:
    grid = qr.uniform("nvfp4", 36)
    grid[35][qr.KINDS.index("down_proj")] = "mxfp8"
    grid[30][qr.KINDS.index("q_proj")] = "int8"
    grid[0] = ["bf16"] * len(qr.KINDS)
    recipe = qr.build("deep", "nvfp4", grid)
    table = qr.fragments("nvfp4", {})
    assert qr.resolve("nvfp4", recipe["overrides"], 36, table=table) == grid
    # block 35's one cell is a name rule (match)
    assert any(rule.get("match") == qr.layer_name(35, "down_proj") for rule in recipe["overrides"])
    with pytest.raises(ValueError, match=">= 28 blocks"):
        qr.build("short", "nvfp4", qr.uniform("nvfp4", 27))


def test_solve_writes_one_row_per_block_of_a_deeper_model(tmp_path: Path) -> None:
    table = json.loads(_table(tmp_path, floor=0.001).read_text())
    extra = []
    for layer in table["layers"]:
        if layer["block"] < 8:  # blocks 28-35 copy blocks 0-7
            block = layer["block"] + 28
            extra.append({**layer, "block": block, "name": qr.layer_name(block, layer["kind"])})
    table["layers"] += extra
    table["kv"] += [{**row, "block": row["block"] + 28, "name": f"model.layers.{row['block'] + 28}.self_attn"}
                    for row in table["kv"] if row["block"] < 8]
    table["meta"].update(blocks=36, model_blocks=36)
    path = tmp_path / "deep.json"
    path.write_text(json.dumps(table))
    assert at.main(["solve", "--table", str(path), "--budget-bits", "5", "--name", "deep",
                    "--out", str(tmp_path / "out")]) == 0
    spec = json.loads((tmp_path / "out" / "grid.json").read_text())
    assert len(spec["grid"]) == 36
    solution = json.loads((tmp_path / "out" / "solution.json").read_text())
    chosen = {row["name"]: row["method"] for row in solution["layers"]}
    assert spec["grid"][35][qr.KINDS.index("down_proj")] == chosen[qr.layer_name(35, "down_proj")]
    assert solution["untabled_blocks_in_grid"] == [] and solution["grid_blocks_beyond_model"] == []
    layers = [at.Layer(row["name"], row["block"], row["kind"], row["params"], row["in_features"])
              for row in table["layers"]]
    assert at.predict(table, layers, table["meta"]["methods"], spec["grid"])[1] <= 5


def test_the_layer_map_of_a_deeper_model_is_not_refused() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=30, num_attention_heads=2, num_key_value_heads=1,
                                      max_position_embeddings=64)
    layers, blocks = at.model_layers(transformers.LlamaForCausalLM(config))
    assert blocks == 30 and len(layers) == 30 * len(qr.KINDS) and layers[-1].block == 29
