"""Catalog contract (app/README.md): algorithm schemas, canonical designs, presets and recipe composition."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from app.catalog import (
    ALGORITHMS,
    FORMATS,
    PRESETS,
    baseline_recipe,
    build_catalog,
    canonical_mma,
    compose_recipe,
    mma_label,
    preset_for,
    preset_mma,
    recipe_record,
)
from tricast import load_recipe, patch_model
from tricast.mma.spec import PRESETS as MMA_PRESETS
from tricast.mma.spec import MMASpec
from tricast.nn import iter_emuconv2d, iter_emulinear, unpatch_model

REPO = Path(__file__).resolve().parents[2]
TASKS = ("llm.generate", "vision.detect", "vision.classify")
VISION = TASKS[1:]


def korean_error(call, *args) -> str:
    with pytest.raises(ValueError) as info:
        call(*args)
    assert re.search("[가-힣]", str(info.value)) and "—" not in str(info.value), str(info.value)
    return str(info.value)


def strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in strings(item)]
    return []


def test_canonical_fills_defaults() -> None:
    assert canonical_mma({"algorithm": "cofda"}) == {"algorithm": "cofda", "f_bits": 13, "chunk_size": 32,
                                                     "c_mode": "fused", "promote_interval": 0,
                                                     "norm_rounding": "rtz"}
    assert canonical_mma({"algorithm": "gdfs"}) == {"algorithm": "gdfs", "f_bits": 35, "g_bits": 6,
                                                    "group_size": 16, "k_tile": 64, "norm_rounding": "rtz"}
    assert canonical_mma({"algorithm": "fp64"}) == {"algorithm": "fp64"}
    assert canonical_mma({"algorithm": "fp32_fma"}) == {"algorithm": "fp32_fma"}


def test_canonical_keeps_only_keys_whose_when_holds() -> None:
    assert canonical_mma({"algorithm": "cofda", "c_mode": "decoupled"})["f2_bits"] == 23
    assert canonical_mma({"algorithm": "cofda", "c_mode": "decoupled", "f2_bits": 20})["f2_bits"] == 20
    assert "f2_bits" not in canonical_mma({"algorithm": "cofda", "c_mode": "fused", "f2_bits": 20})


def test_canonical_stores_integers() -> None:
    m = canonical_mma({"algorithm": "cofda", "f_bits": 7.0, "chunk_size": 16.0})
    assert (m["f_bits"], m["chunk_size"]) == (7, 16)
    assert type(m["f_bits"]) is int and type(m["chunk_size"]) is int


@pytest.mark.parametrize("mma", [
    {"algorithm": "int_exact"},
    {"algorithm": "cofda", "g_bits": 6},
    {"algorithm": "cofda", "f_bits": 0},
    {"algorithm": "cofda", "f_bits": 49},
    {"algorithm": "cofda", "f_bits": 7.5},
    {"algorithm": "cofda", "f_bits": "7"},
    {"algorithm": "cofda", "f_bits": True},
    {"algorithm": "cofda", "chunk_size": 24},
    {"algorithm": "cofda", "c_mode": "both"},
    {"algorithm": "cofda", "promote_interval": 32},
    {"algorithm": "gdfs", "group_size": 64},
])
def test_canonical_rejects_values_outside_the_schema(mma: dict) -> None:
    korean_error(canonical_mma, mma)


def test_canonical_round_trip() -> None:
    for preset_id in PRESETS:
        m = preset_mma(preset_id)
        assert canonical_mma(m) == m and canonical_mma(canonical_mma(m)) == m


def test_promote_interval_must_be_a_multiple_of_the_chunk() -> None:
    message = korean_error(compose_recipe, {"algorithm": "cofda", "chunk_size": 128, "promote_interval": 64},
                           "fp8_tensor", "llm.generate")
    assert "배수" in message
    for chunk, interval in ((32, 128), (64, 64), (64, 256)):
        mma = {"algorithm": "cofda", "chunk_size": chunk, "promote_interval": interval}
        compose_recipe(mma, "fp8_tensor", "llm.generate")


def test_promotion_is_defined_for_fused_chunks_only() -> None:
    korean_error(compose_recipe, {"algorithm": "cofda", "c_mode": "decoupled", "promote_interval": 128},
                 "fp8_tensor", "llm.generate")


@pytest.mark.parametrize(("group_size", "k_tile", "ok"), [(8, 256, False), (32, 16, False), (16, 128, True),
                                                          (8, 64, True), (32, 32, True)])
def test_gdfs_tile_holds_one_to_eight_groups(group_size: int, k_tile: int, ok: bool) -> None:
    mma = {"algorithm": "gdfs", "group_size": group_size, "k_tile": k_tile}
    if ok:
        compose_recipe(mma, "fp8_tensor", "llm.generate")
    else:
        korean_error(compose_recipe, mma, "fp8_tensor", "llm.generate")


@pytest.mark.parametrize(("mma", "format_id", "ok"), [
    ({"algorithm": "cofda", "promote_interval": 128}, "mxfp8_e4m3", False),
    ({"algorithm": "cofda", "promote_interval": 64}, "fp8_block128", True),
    ({"algorithm": "cofda", "promote_interval": 256}, "fp8_block128", False),
    ({"algorithm": "cofda", "promote_interval": 128}, "fp8_tensor", True),
    ({"algorithm": "gdfs", "group_size": 32, "k_tile": 64}, "nvfp4", False),
    ({"algorithm": "gdfs", "group_size": 16}, "nvfp4", True),
    ({"algorithm": "gdfs", "group_size": 32, "k_tile": 64}, "mxfp4", True),
])
def test_promotion_and_groups_stay_inside_scale_domains(mma: dict, format_id: str, ok: bool) -> None:
    if ok:
        compose_recipe(mma, format_id, "llm.generate")
    else:
        korean_error(compose_recipe, mma, format_id, "llm.generate")


def test_every_preset_composes_and_keeps_its_arithmetic() -> None:
    for preset_id, preset in PRESETS.items():
        mma = preset_mma(preset_id)
        source = (MMA_PRESETS[preset.mma_preset] if preset.mma_preset
                  else load_recipe(preset.recipe).defaults.mma)
        assert MMASpec(**mma) == source.with_(), preset_id
        assert preset_for(mma, preset.format) == preset_id
        for task in TASKS:
            load_recipe(compose_recipe(mma, preset.format, task))


def test_presets_record_their_bundled_recipe() -> None:
    expected = {"hopper": "hopper_fp8_w8a8", "ada": "ada_fp8_w8a8", "blackwell_fp8": "blackwell_fp8_w8a8",
                "blackwell_fp4": "nvfp4_w_a", "deepseek_promote": "deepseek_fp8_block",
                "f7_fused": "fp8_f7_lowacc", "f7_decoupled": "fp8_f7_decoupled", "fp64": None}
    found = {preset_id: recipe_record(compose_recipe(preset_mma(preset_id), preset.format, "llm.generate"))
             ["bundled"] for preset_id, preset in PRESETS.items()}
    assert found == expected
    rht = compose_recipe(preset_mma("blackwell_fp4"), "mxfp4_rht", "llm.generate")
    assert recipe_record(rht)["bundled"] == "mxfp4_rht"


def test_every_combination_composes_or_raises_and_vision_rejects_transforms() -> None:
    rejected = set()
    for algorithm in ALGORITHMS:
        for format_id in FORMATS:
            for task in TASKS:
                try:
                    load_recipe(compose_recipe({"algorithm": algorithm}, format_id, task))
                except ValueError as exc:
                    assert re.search("[가-힣]", str(exc))
                    rejected.add((algorithm, format_id, task))
    assert rejected == {(algorithm, "mxfp4_rht", task) for algorithm in ALGORITHMS for task in VISION}


def designs() -> list[tuple[dict, str | None]]:
    defaults = [({"algorithm": algorithm}, format_id) for algorithm in ALGORITHMS for format_id in FORMATS]
    return defaults + [(preset_mma(preset_id), preset.format) for preset_id, preset in PRESETS.items()]


@pytest.mark.parametrize("vision", [False, True])
def test_composed_recipes_run_on_the_reference_backend(vision: bool) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(32, 4, 2)) if vision else nn.Sequential(nn.Linear(128, 8))
    sample = torch.randn(1, 32, 3, 3) if vision else torch.randn(3, 128)
    for mma, format_id in designs():
        if vision and FORMATS[format_id].transform != "none":
            continue
        recipe = compose_recipe(mma, format_id, "vision.detect" if vision else "llm.generate")
        patch_model(model, load_recipe(recipe), backend="reference", include_conv2d=vision)
        try:
            assert len(list(iter_emuconv2d(model) if vision else iter_emulinear(model))) == 1
            with torch.no_grad():
                assert torch.isfinite(model(sample)).all(), recipe["name"]
        finally:
            unpatch_model(model)


def test_baseline_recipe() -> None:
    assert baseline_recipe("native", "fp8_tensor") is None
    fp64 = baseline_recipe("same_quant_fp64", "fp8_tensor")
    assert fp64["defaults"]["mma"] == {"algorithm": "fp64"}
    assert (fp64["defaults"]["weight"], fp64["defaults"]["activation"]) == ("fp8_tensor", "fp8_tensor")
    assert fp64 == compose_recipe({"algorithm": "fp64"}, "fp8_tensor", "llm.generate")
    assert baseline_recipe("same_quant_fp64", None)["defaults"]["weight"] is None
    korean_error(baseline_recipe, "fp32", "fp8_tensor")
    korean_error(compose_recipe, {"algorithm": "cofda"}, "fp7", "llm.generate")
    korean_error(compose_recipe, {"algorithm": "cofda"}, "fp8_tensor", "audio.transcribe")


def test_mma_labels() -> None:
    assert mma_label({"algorithm": "cofda", "f_bits": 7}) == "CoFDA · F7 · CS32 · fused"
    assert mma_label(preset_mma("deepseek_promote")) == "CoFDA · F13 · CS32 · fused · FP32 승격 128"
    assert mma_label(preset_mma("blackwell_fp4")) == "GDFS · F35 · G6 · GS16 · KT64"
    assert mma_label({"algorithm": "fp64"}) == "FP64 정확 누산"
    assert mma_label({"algorithm": "cofda", "c_mode": "decoupled", "f_bits": 7}) == \
        "CoFDA · F7 · CS32 · decoupled · F2 23"


def test_recipe_record_round_trips_yaml() -> None:
    recipe = compose_recipe(preset_mma("hopper"), "fp8_tensor", "llm.generate")
    record = recipe_record(recipe)
    assert record["name"] == recipe["name"] and yaml.safe_load(record["yaml"]) == recipe


def test_catalog_contract() -> None:
    device = {"kind": "cpu", "name": None, "backend": "reference"}
    catalog = json.loads(json.dumps(build_catalog("live", device), allow_nan=False))
    assert set(catalog) == {"tricast", "mode", "device", "tasks", "models", "algorithms", "presets",
                            "formats", "baselines"}
    assert catalog["mode"] == "live" and catalog["device"] == device
    assert [task["id"] for task in catalog["tasks"]] == list(TASKS)
    assert [algorithm["id"] for algorithm in catalog["algorithms"]] == ["cofda", "gdfs", "fp32_fma", "fp64"]
    assert None in [fmt["id"] for fmt in catalog["formats"]]
    for algorithm in catalog["algorithms"]:
        assert algorithm["formats"] == [fmt["id"] for fmt in catalog["formats"]]
        for param in algorithm["params"]:
            assert {"key", "label", "kind", "default", "help"} <= set(param)
            if param["kind"] == "int":
                assert param["min"] <= param["ui_min"] <= param["default"] <= param["ui_max"] <= param["max"]
            else:
                assert param["default"] in param["options"]
    for preset in catalog["presets"]:
        assert set(preset) == {"id", "label", "source", "mma", "format", "status", "provenance",
                               "status_note"}
        assert preset["provenance"] and preset["status_note"]
        assert canonical_mma(preset["mma"]) == preset["mma"]
    models = {model["id"]: model for model in catalog["models"]}
    assert models["meta-llama/Llama-3.2-1B"]["revision"] == "4e20de362430cd3b72f300e6b0f18e50e7166e08"
    assert models["resnet18"]["support"] == "operator"
    assert [baseline["id"] for baseline in catalog["baselines"]] == ["native", "same_quant_fp64"]


def test_preset_status_matches_support_matrix() -> None:
    matrix = yaml.safe_load((REPO / "support_matrix.yaml").read_text(encoding="utf-8"))["mma"]["presets"]
    for preset in PRESETS.values():
        if preset.mma_preset in matrix:
            check = matrix[preset.mma_preset].get("silicon_check")
            expected = "partial_mismatch" if check == "partial_mismatch" else "modeled"
            assert preset.status == expected, preset.id
    assert {preset.id for preset in PRESETS.values() if preset.status == "design_point"} == {"f7_fused",
                                                                                             "f7_decoupled"}
    assert PRESETS["fp64"].status == "reference"


def test_ui_text_has_no_em_dash_or_internal_history() -> None:
    catalog = build_catalog("demo", {"kind": "none", "name": None, "backend": None})
    provenance = {preset["provenance"] for preset in catalog["presets"]}
    shown = [text for text in strings({key: value for key, value in catalog.items() if key != "tricast"})
             if text not in provenance]
    assert not [text for text in shown if "—" in text or re.search(r"20\d\d-\d\d-\d\d|결정", text)]
    for preset_id, preset in PRESETS.items():
        mma_preset = preset.mma_preset
        if mma_preset is not None:
            assert MMA_PRESETS[mma_preset].provenance in provenance, preset_id
