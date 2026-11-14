"""The packages step 5 of the tuning page shows (scripts/quantizer/pack_index.py): an LLM package is listed
with its model and recipe file only when every layer decoded to TriCast's weight, its files agree with each
other and its PPL is within 0.1% of TriCast's; an ONNX file only when its bytes are the ones its record
describes. Synthetic records, no model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pack_index", ROOT / "scripts" / "quantizer" / "pack_index.py")
PI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PI
SPEC.loader.exec_module(PI)

RECIPE = "name: test\ndefaults:\n  weight: int4_g128_zp\n"
SHA = hashlib.sha256(RECIPE.encode("utf-8")).hexdigest()
LAYERS = 196


def _package(tmp_path: Path, manifest_change: dict | None = None, **checks) -> Path:
    package = tmp_path / "q06b"
    package.mkdir(parents=True)
    config = {
        "schema": "tricast.pack/1",
        "tied": {"lm_head.weight": "model.embed_tokens.weight"},
        "recipe": {"sha256": SHA, "yaml": RECIPE},
        "layers": {f"model.layers.{i // 7}.l{i % 7}": {"kind": "affine"} for i in range(LAYERS)},
    }
    (package / "quant_config.json").write_text(json.dumps(config))
    digest = hashlib.sha256((package / "quant_config.json").read_bytes()).hexdigest()
    base = {
        "decoded_equals_tricast_weight": {"layers": LAYERS, "mismatched": []},
        "ppl_tricast": 26.4458,
        "ppl_package": 26.4412,
        "ppl_relative_difference": 26.4412 / 26.4458 - 1,
        "gpu": "NVIDIA H100 NVL",
        "model_sha": "c1899de2",
        "seconds": 319.2,
    }
    manifest = {
        "model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de2",
        "recipe_sha256": SHA,
        "files": {
            "model.safetensors": {"bytes": 1000, "sha256": "cd" * 32},  # on the cluster, not here
            "quant_config.json": {"bytes": 10, "sha256": digest},
        },
        "bytes": 1010,
        "weights_bytes": 1000,
        "quantized_layers": LAYERS,
        "layer_kinds": {"affine": 192, "dense": 4},
        "fallback_layers": {f"model.layers.{i}.self_attn.k_proj": "hqq keeps no codes" for i in range(4)},
        "checks": {**base, **checks},
        **(manifest_change or {}),
    }
    (package / "manifest.json").write_text(json.dumps(manifest))
    # a captured stdout of `python -m tricast.pack run`: the JSON, then its marker line
    run = {
        "package": "/scratch/packages/q06b",
        "mode": "codes",
        "ppl": 26.4456,
        "gpu_memory_loaded_bytes": 600,
        "gpu_memory_peak_eval_bytes": 900,
    }
    (tmp_path / "q06b.run_codes.json").write_text(json.dumps(run, indent=1) + "\nPACK_RUN_DONE\n")
    return package


def test_a_checked_package_is_listed_with_its_model_and_recipe(tmp_path: Path) -> None:
    entry = PI.llm_entry(_package(tmp_path))
    assert (entry["model"], entry["recipe_sha256"]) == ("Qwen/Qwen3-0.6B", SHA)
    assert entry["decoded_equal"] == {"layers": LAYERS, "mismatched": 0}
    assert entry["fallback"] == {"count": 4, "reasons": ["hqq keeps no codes"]}
    assert entry["tied"] == ["lm_head.weight"]
    assert entry["ppl_codes"] == 26.4456 and entry["gpu_memory_loaded_bytes"] == 600
    assert entry["ppl_relative_difference"] == pytest.approx(26.4412 / 26.4458 - 1)
    assert entry["original_bytes"] == 1503300328  # Qwen3-0.6B's BF16 checkpoint (app.autotune_export)
    manifest_bytes = (tmp_path / "q06b" / "manifest.json").stat().st_size
    assert entry["bytes"] == 1010 + manifest_bytes


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        (
            {
                "decoded_equals_tricast_weight": {
                    "layers": LAYERS,
                    "mismatched": ["model.layers.0.mlp.up_proj"],
                }
            },
            "decode",
        ),
        ({"decoded_equals_tricast_weight": {"layers": 0, "mismatched": []}}, "covered 0 of 196"),
        ({"ppl_package": 26.4458 * 1.0011}, "within 0.1%"),
        ({"ppl_package": 26.4458 * 0.9989}, "within 0.1%"),
        # the gap is computed from the two PPLs, not taken from the record
        ({"ppl_package": 264.0, "ppl_relative_difference": 0.0}, "within 0.1%"),
        ({"ppl_package": None}, "no measured PPL"),
        ({"ppl_tricast": float("nan")}, "no measured PPL"),
    ],
)
def test_a_package_that_fails_a_check_is_not_listed(tmp_path: Path, checks: dict, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        PI.llm_entry(_package(tmp_path, **checks))


def test_files_that_disagree_with_each_other_are_not_listed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="other recipes"):
        PI.llm_entry(_package(tmp_path / "a", {"recipe_sha256": "ef" * 32}))
    package = _package(tmp_path / "b")
    (package / "quant_config.json").write_text((package / "quant_config.json").read_text() + " ")
    with pytest.raises(SystemExit, match="quant_config.json is not the file"):
        PI.llm_entry(package)
    package = _package(tmp_path / "c")
    run = package.with_name("q06b.run_codes.json")
    run.write_text(run.read_text().replace("/scratch/packages/q06b", "/scratch/packages/q8b"))
    with pytest.raises(SystemExit, match="another package"):
        PI.llm_entry(package)


def _onnx(work: Path, data: bytes, recorded: bytes) -> None:
    (work / "onnx").mkdir(parents=True)
    (work / "onnx" / "u_int4.onnx").write_bytes(data)
    record = {
        "setup": "u_int4",
        "file": "u_int4.onnx",
        "bytes": len(data),
        "sha256": hashlib.sha256(recorded).hexdigest(),
        "stored": {"model.0.conv.weight": {"dtype": "int4"}, "model.1.conv.weight": {"dtype": "int4"}},
        "metrics": {"AP_50_95": 0.30},
        "pytorch_metrics": {"AP_50_95": 0.30},
        "raw_max_abs_diff": [0.001, 0.002],
        "demo_boxes": {},
        "evaluation_images": 4872,
        "env": {"onnxruntime": "1.23.2", "git_sha": "6bc78e6"},
    }
    (work / "onnx" / "onnx_u_int4.json").write_text(json.dumps(record))


def test_an_onnx_file_is_copied_and_listed_with_its_record(tmp_path: Path) -> None:
    _onnx(tmp_path / "work", b"onnx bytes", b"onnx bytes")
    copy = tmp_path / "page"
    entries = PI.onnx_entries("yolo11n", tmp_path / "work" / "onnx", copy)
    assert entries["u_int4"]["file"] == "packages/yolo11n/u_int4.onnx"
    assert entries["u_int4"]["kinds"] == {"int4": 2} and entries["u_int4"]["raw_max_abs_diff"] == 0.002
    assert (copy / "u_int4.onnx").read_bytes() == b"onnx bytes"


def test_an_onnx_file_that_differs_from_its_record_is_refused(tmp_path: Path) -> None:
    _onnx(tmp_path / "work", b"other bytes", b"onnx bytes")
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        PI.onnx_entries("yolo11n", tmp_path / "work" / "onnx", tmp_path / "page")
    assert not (tmp_path / "page").exists()


def test_the_original_models_file_keeps_its_record_but_is_not_copied(tmp_path: Path) -> None:
    _onnx(tmp_path / "work", b"fp32 bytes", b"fp32 bytes")
    record = tmp_path / "work" / "onnx" / "onnx_u_int4.json"
    data = json.loads(record.read_text())
    record.write_text(json.dumps({**data, "stored": {}}))  # no quantized conv
    entries = PI.onnx_entries("yolo11n", tmp_path / "work" / "onnx", tmp_path / "page")
    assert entries["u_int4"]["file"] is None and entries["u_int4"]["kinds"] == {}
    assert not (tmp_path / "page").exists()
