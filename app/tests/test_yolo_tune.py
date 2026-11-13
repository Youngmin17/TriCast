"""CPU-only checks of scripts/quantizer/yolo_tune.py's pure parts and of the exported YOLO11n tuning dataset.

The pipeline itself (table / evaluate) runs on a cluster CPU host; here: bits and bytes accounting, the
method -> QuantSpec mapping and its fake-quant grid, the calibration / evaluation split, the knapsack wiring
against brute force on a tiny synthetic table, the D objective on hand-made outputs, and the exported JSON's
schema and internal consistency.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("yolo_tune", ROOT / "scripts" / "quantizer" / "yolo_tune.py")
YT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = YT  # its dataclass looks the module up
SPEC.loader.exec_module(YT)
EXPORT = ROOT / "app" / "web" / "demo" / "autotune" / "yolo11n.json"


def test_stored_bits_and_layer_cost() -> None:
    layer = YT.ConvLayer("model.0.conv", 16 * 27, 27, 16)
    assert YT.stored_bits("fp16", 27) == 16
    assert YT.stored_bits("int4", 9) == 4 + Fraction(16, 9)
    assert YT.layer_cost(layer, "fp16") == 16 * 432
    for method, bits in (("int8", 8), ("int4", 4), ("int3", 3)):
        # b bits per element + one fp16 scale per output channel
        assert YT.layer_cost(layer, method) == 432 * bits + 16 * 16


def test_average_bits_and_bytes() -> None:
    layers = [YT.ConvLayer("a", 16 * 27, 27, 16), YT.ConvLayer("b", 32 * 9, 9, 32)]
    assignment = {"a": "int8", "b": "int4"}
    total = 432 * 8 + 16 * 16 + 288 * 4 + 16 * 32
    assert YT.total_bits(layers, assignment) == total
    assert YT.avg_bits(layers, assignment) == Fraction(total, 720)
    assert YT.model_bytes(layers, assignment, other_params=48) == (total + 16 * 48) // 8
    assert YT.original_bytes(layers, other_params=48) == 2 * (720 + 48)
    assert YT.model_bytes(layers, YT.uniform_assignment(layers, "fp16"), 48) == YT.original_bytes(layers, 48)
    assert YT.method_counts(assignment) == {"fp16": 0, "int8": 1, "int4": 1, "int3": 0}
    assert YT.parse_fraction(YT.fraction_text(Fraction(7, 3))) == Fraction(7, 3)
    with pytest.raises(YT.TuneError):
        YT.total_bits(layers, {"a": "int8"})


def test_method_to_spec() -> None:
    assert YT.weight_spec("fp16") is None
    for method, qmax in (("int8", 127), ("int4", 7), ("int3", 3)):
        spec = YT.weight_spec(method)
        assert spec.format.kind == "int" and spec.format.bits == YT.METHODS[method]
        assert spec.format.signed and spec.format.symmetric and spec.format.qmax == qmax == -spec.format.qmin
        assert spec.granularity == "row" and spec.zero_point == "none" and spec.mma_input == "scaled"
        assert spec.scale.method == "absmax" and spec.scale.format.name == "fp16"
        assert spec.rounding.name == "RNE" and spec.scale.rounding.name == "RNE"
    with pytest.raises(YT.TuneError):
        YT.weight_spec("int2")


@pytest.mark.parametrize("shape", [(4, 2, 3, 3), (8, 1, 3, 3), (6, 5, 1, 1)])
@pytest.mark.parametrize("method", ["int8", "int4", "int3"])
def test_fake_quant_weight_grid(shape: tuple[int, ...], method: str) -> None:
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(shape, generator=generator) * 0.1
    quantized = YT.fake_quant_weight(weight, method)
    assert quantized.shape == weight.shape and quantized.dtype == weight.dtype
    qmax = 2 ** (YT.METHODS[method] - 1) - 1
    for row, source in zip(quantized.reshape(shape[0], -1), weight.reshape(shape[0], -1), strict=True):
        # one scale per output channel over K = in/groups*kh*kw: the fp16-rounded absmax / qmax
        scale = (source.abs().max().double() / qmax).half().double()
        levels = row.double() / scale
        assert torch.equal(levels, levels.round()) and levels.abs().max() <= qmax
        assert torch.equal(levels, (source.double() / scale).round().clamp(-qmax, qmax))


def test_calibration_and_evaluation_split() -> None:
    ids = sorted(set(range(1, 4998)) | {5001, 8021})
    calibration = YT.calibration_ids(ids)
    assert calibration == YT.calibration_ids(list(reversed(ids)))
    assert calibration == sorted(random.Random(42).sample(sorted(set(ids) - {1000, 5001, 8021}), 128))
    assert len(set(calibration)) == 128 and not set(calibration) & set(YT.DEMO_IDS)
    evaluation = YT.evaluation_ids(ids, calibration)
    assert len(evaluation) == len(ids) - 128 and set(YT.DEMO_IDS) <= set(evaluation)
    assert not set(evaluation) & set(calibration)


def test_heuristic_and_uniform_assignment() -> None:
    layers = [YT.ConvLayer(name, 16, 1, 16) for name in ("model.0.conv", "model.1.conv", "model.23.cv2.0.2")]
    assert YT.heuristic_assignment(layers, ["model.0.conv", "model.23.cv2.0.2"]) == {
        "model.0.conv": "int8", "model.1.conv": "int4", "model.23.cv2.0.2": "int8"}
    assert YT.baseline_assignment("u_int3", layers, []) == {layer.name: "int3" for layer in layers}
    assert set(YT.baseline_assignment("native", layers, []).values()) == {"fp16"}
    with pytest.raises(YT.TuneError):
        YT.heuristic_assignment(layers, ["model.9.conv"])


def test_knapsack_matches_brute_force() -> None:
    rng = random.Random(7)
    layers = [YT.ConvLayer(f"l{i}", out * k, k, out) for i, (out, k) in
              enumerate([(16, 27), (32, 9), (64, 16), (16, 144)])]
    table = {}
    for layer in layers:
        base = rng.uniform(1e-4, 1e-2)
        table[layer.name] = {"int8": base * 0.01, "int4": base, "int3": base * rng.uniform(2, 6)}
    params = sum(layer.params for layer in layers)
    for reference in ("int4", "int3"):
        budget = YT.avg_bits(layers, YT.uniform_assignment(layers, reference))
        assignment, solution = YT.solve(layers, table, budget * params)
        bits = YT.avg_bits(layers, assignment)
        assert bits <= budget and solution.bound == 0
        best = min(YT.predicted_d(dict(zip(table, combo, strict=True)), table)
                   for combo in itertools.product(YT.METHODS, repeat=len(layers))
                   if YT.avg_bits(layers, dict(zip(table, combo, strict=True))) <= budget)
        assert solution.value == pytest.approx(best, rel=1e-12)
        assert YT.predicted_d(assignment, table) == pytest.approx(solution.value, rel=1e-12)
        # at the uniform setup's own size the uniform choice is feasible, so auto predicts no worse
        assert solution.value <= YT.predicted_d(YT.uniform_assignment(layers, reference), table) + 1e-15


def test_distortion() -> None:
    reference = torch.zeros(2, 84, 3)
    reference[:, 4:, :] = 0.1
    reference[0, 4, 0] = 0.9  # anchor 0 of image 0: weight 0.9, the others 0.1
    assert torch.equal(YT.distortion(reference, reference.clone()), torch.zeros(2, dtype=torch.float64))
    output = reference.clone()
    output[0, 5, 0] += 0.2  # class error 0.04 on the weight-0.9 anchor
    output[1, 0, 2] += 64.0  # box error 64^2 / 640^2 = 0.01 on a weight-0.1 anchor
    d = YT.distortion(output=output, reference=reference)
    assert d[0].item() == pytest.approx(0.9 * 0.04 / 1.1)
    assert d[1].item() == pytest.approx(0.1 * 0.01 / 0.3)


def test_exported_dataset() -> None:
    assert EXPORT.is_file(), f"{EXPORT} has not been exported"
    assert EXPORT.stat().st_size < 1_500_000
    data = json.loads(EXPORT.read_text())
    assert YT.validate(data) == []
    assert data["model"]["conv_layers"] == len(data["layers"]) > 0
    by_id = {setup["id"]: setup for setup in data["setups"]}
    for auto, reference in YT.AUTO.items():
        budget = YT.parse_fraction(by_id[reference]["bits_fraction"])
        assert YT.parse_fraction(by_id[auto]["budget_bits"]) == budget  # the knapsack's exact budget
        assert YT.parse_fraction(by_id[auto]["bits_fraction"]) <= budget
        assert by_id[auto]["bytes"] <= by_id[reference]["bytes"]
    for image in data["images"]:
        assert (ROOT / "app" / "web" / "demo" / image["file"]).is_file()
