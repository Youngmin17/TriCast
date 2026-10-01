"""ULP distances on exact encodings, and the accumulator-only ULP column of the layer report."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tests.conftest import wide_fp32
from tricast.analysis import layer_report, ulp_distance, ulp_error
from tricast.formats import BF16, FP16, FP32
from tricast.mma.spec import MMASpec


def f32(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def next_up(value: float) -> float:
    return torch.nextafter(f32(value), f32(math.inf)).item()


def signed_magnitude(bits: torch.Tensor, sign_bit: int) -> torch.Tensor:
    """Independent oracle: the sign-magnitude encoding read from a dtype view."""
    bits = bits.to(torch.int64)
    magnitude = bits & ((1 << sign_bit) - 1)
    return torch.where(bits < 0, -magnitude, magnitude)


def check_pairs(pairs: list[tuple[float, float, int]], fmt) -> None:
    actual, expected = f32(*(a for a, _, _ in pairs)), f32(*(e for _, e, _ in pairs))
    distance = ulp_distance(actual, expected, fmt)
    assert distance.dtype == torch.int64
    assert distance.tolist() == [d for _, _, d in pairs]
    assert torch.equal(ulp_distance(expected, actual, fmt), distance)


def test_ulp_distance_fp32_hand_values() -> None:
    tiny, big = 2.0**-149, torch.finfo(torch.float32).max
    check_pairs([
        (1.0, next_up(1.0), 1),
        (1.0, 1.0, 0),
        (0.0, -0.0, 0),
        (0.0, tiny, 1),
        (-tiny, -0.0, 1),
        (-tiny, tiny, 2),                       # -tiny, 0, +tiny
        (2.0**-126 - tiny, 2.0**-126, 1),       # largest subnormal, smallest normal
        (2.0**126, 2.0**127, 2**23),            # one binade
        (big, big - 2.0**104, 1),               # top of the range
        (-1.0, 1.0, 2 * 0x3F800000),            # across the sign change
    ], FP32)


def test_ulp_distance_bf16_hand_values() -> None:
    tiny, big = 2.0**-133, torch.finfo(torch.bfloat16).max
    pairs = [
        (1.0, 1.0 + 2.0**-7, 1),
        (0.0, -0.0, 0),
        (-tiny, tiny, 2),
        (2.0**-126 - tiny, 2.0**-126, 1),
        (2.0**126, 2.0**127, 2**7),
        (big, big - 2.0**120, 1),
        (-1.0, 1.0, 2 * 0x3F80),
    ]
    check_pairs(pairs, BF16)
    check_pairs(pairs, "bf16")
    actual = torch.tensor([a for a, _, _ in pairs], dtype=torch.bfloat16)
    expected = torch.tensor([e for _, e, _ in pairs], dtype=torch.bfloat16)
    assert ulp_distance(actual, expected, BF16).tolist() == [d for _, _, d in pairs]


def test_ulp_distance_fp16_hand_values() -> None:
    tiny = 2.0**-24
    check_pairs([
        (1.0, 1.0 + 2.0**-10, 1),
        (0.0, tiny, 1),
        (-tiny, tiny, 2),
        (2.0**-14 - tiny, 2.0**-14, 1),
        (65504.0, 65472.0, 1),
        (-65504.0, 65504.0, 2 * 0x7BFF),
    ], FP16)


def test_ulp_distance_matches_fp32_encoding(gen: torch.Generator) -> None:
    a, b = wide_fp32(4096, gen, 0, 255), wide_fp32(4096, gen, 0, 255)  # zero, subnormal ... max
    oracle = (signed_magnitude(a.view(torch.int32), 31) - signed_magnitude(b.view(torch.int32), 31)).abs()
    assert torch.equal(ulp_distance(a, b), oracle)
    below_max = wide_fp32(4096, gen, 0, 254)
    neighbour = torch.nextafter(below_max, torch.full_like(below_max, math.inf))
    assert bool((ulp_distance(below_max, neighbour) == 1).all())


@pytest.mark.parametrize("dtype, fmt", [(torch.bfloat16, BF16), (torch.float16, FP16)])
def test_ulp_distance_matches_16_bit_encodings(gen: torch.Generator, dtype: torch.dtype, fmt) -> None:
    raw = torch.randint(0, 2**16, (2, 4096), generator=gen)
    values = torch.where(raw >= 2**15, raw - 2**16, raw).to(torch.int16).view(dtype)
    keep = torch.isfinite(values).all(dim=0)
    a, b = values[0, keep], values[1, keep]
    oracle = (signed_magnitude(a.view(torch.int16), 15) - signed_magnitude(b.view(torch.int16), 15)).abs()
    assert torch.equal(ulp_distance(a, b, fmt), oracle)
    assert torch.equal(ulp_distance(a.float(), b.double(), fmt), oracle)


def test_ulp_distance_follows_the_format_not_a_dtype() -> None:
    zero, smallest_normal = f32(0.0), f32(2.0**-6)
    assert ulp_distance(zero, smallest_normal, "fp8_e4m3").item() == 8  # seven subnormals between
    assert ulp_distance(f32(448.0), f32(-448.0), "fp8_e4m3").item() == 2 * 0x7E
    assert ulp_distance(zero, smallest_normal, "e4m3:nosub").item() == 1
    assert ulp_distance(f32(-(2.0**-6)), smallest_normal, "e4m3:nosub").item() == 2
    # float8_e4m3fn holds 2^-7 as a subnormal; the no-subnormal grid does not.
    with pytest.raises(ValueError, match="grid"):
        ulp_distance(f32(2.0**-7), zero, "e4m3:nosub")


@pytest.mark.parametrize("values, fmt", [
    (torch.tensor([1.0 + 2.0**-30], dtype=torch.float64), FP32),  # finer than fp32
    (torch.tensor([1e39], dtype=torch.float64), FP32),             # beyond fp32's range
    (f32(1.0 + 2.0**-10), BF16),                                    # between bf16 neighbours
    (f32(2.0**-140), BF16),                                         # below bf16's smallest subnormal
    (f32(65520.0), FP16),                                           # rounds to Inf in fp16
    (f32(1e5), FP16),                                               # beyond fp16's range
])
def test_off_grid_values_are_rejected(values: torch.Tensor, fmt) -> None:
    on_grid = torch.zeros_like(values)
    with pytest.raises(ValueError, match="actual has values that are not on the"):
        ulp_distance(values, on_grid, fmt)
    with pytest.raises(ValueError, match="expected has values that are not on the"):
        ulp_error(on_grid, values, fmt)


def test_ulp_distance_rejects_nonfinite_shapes_and_non_float_formats() -> None:
    with pytest.raises(ValueError, match="finite"):
        ulp_distance(f32(math.inf), f32(1.0))
    with pytest.raises(ValueError, match="finite"):
        ulp_distance(f32(1.0), f32(math.nan))
    with pytest.raises(ValueError, match="shapes"):
        ulp_distance(f32(1.0, 2.0), f32(1.0))
    with pytest.raises(ValueError, match="float format"):
        ulp_distance(f32(1.0), f32(1.0), "int8")


def test_ulp_error_statistics_hand_calculation() -> None:
    steps = [0, 1, 2, 3, 10]
    expected = f32(1.0, 2.0, 4.0, -8.0, 0.5)
    # Within one sign, consecutive encodings are consecutive grid values.
    actual = (expected.view(torch.int32) + torch.tensor(steps, dtype=torch.int32)).view(torch.float32)
    result = ulp_error(actual, expected)
    # p99: rank (5 - 1) * 0.99 = 3.96 -> 3 + 0.96 * (10 - 3).
    assert result == {"max": 10, "mean": 3.2, "p99": 9.72, "exact_fraction": 0.2, "n": 5,
                      "nonfinite_mismatch": 0}
    assert result["p99"] == pytest.approx(np.percentile(steps, 99), rel=1e-15)
    assert ulp_error(actual, expected, "fp32") == result


def test_ulp_error_p99_matches_numpy_linear_percentile(gen: torch.Generator) -> None:
    expected = wide_fp32(1001, gen, 100, 150)
    steps = torch.randint(0, 50, (1001,), generator=gen, dtype=torch.int32)
    actual = (expected.view(torch.int32) + steps).view(torch.float32)
    result = ulp_error(actual, expected)
    assert result["p99"] == pytest.approx(np.percentile(steps.numpy(), 99), rel=1e-15)
    assert result["max"] == int(steps.max()) and result["mean"] == steps.sum().item() / 1001


def test_ulp_error_nonfinite_classification() -> None:
    nan, inf = math.nan, math.inf
    actual = f32(nan, inf, -inf, inf, 1.0, nan, nan, 1.0, 2.0)
    expected = f32(nan, inf, -inf, -inf, inf, 1.0, inf, next_up(1.0), 2.0)
    # exact: NaN/NaN, +Inf/+Inf, -Inf/-Inf and 2.0/2.0; mismatched: opposite infinities,
    # finite/Inf, NaN/finite and NaN/Inf; finite distances: 1 and 0.
    result = ulp_error(actual, expected)
    assert result == {"max": 1, "mean": 0.5, "p99": 0.99, "exact_fraction": 4 / 9, "n": 9,
                      "nonfinite_mismatch": 4}
    json.dumps(result, allow_nan=False)
    special_only = ulp_error(f32(nan, inf), f32(nan, -inf))
    assert special_only == {"max": None, "mean": None, "p99": None, "exact_fraction": 0.5, "n": 2,
                            "nonfinite_mismatch": 1}
    json.dumps(special_only, allow_nan=False)


def test_ulp_error_rejects_empty_and_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        ulp_error(f32(), f32())
    with pytest.raises(ValueError, match="matching"):
        ulp_error(f32(1.0, 2.0), f32(1.0))


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(4, 8)
        self.proj = nn.Linear(8, 4)
        self.unused = nn.Linear(8, 4)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


@pytest.fixture
def model() -> TinyLM:
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return TinyLM()


def recipe(mma: dict, **changes: object) -> dict:
    return {"name": "ulp-report", "defaults": {"mma": mma, **changes}, "backend": "reference",
            "include": ["proj", "unused"]}


LOW_PRECISION = {"algorithm": "cofda", "f_bits": 4, "chunk_size": 4, "out_format": "fp32"}


@pytest.mark.parametrize("out_format", ["fp32", "bf16"])
def test_report_mma_ulp_is_zero_for_fp64_accumulation(model: TinyLM, out_format: str) -> None:
    data = recipe({"preset": "fp64", "out_format": out_format}, weight="fp8_tensor", activation="fp8_tensor")
    result = layer_report(model, data, input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    proj, unused = result["layers"]
    assert proj["mma_ulp"] == {"max": 0, "mean": 0.0, "p99": 0.0, "exact_fraction": 1.0, "n": 16,
                               "nonfinite_mismatch": 0, "format": out_format}
    assert proj["output"]["mse"] > 0.0  # quantization error stays out of the accumulator metric
    assert unused["status"] == "not_observed" and unused["mma_ulp"] is None
    assert "| MMA ULP max / mean |" in result["markdown"]
    assert f"| 0 / 0 ({out_format}) |" in result["markdown"]
    assert "| unused (not observed) | - | - | - | - | - | - | - | - |" in result["markdown"]
    json.dumps(result, allow_nan=False)


def test_report_mma_ulp_compares_the_forward_gemm_with_fp64(model: TinyLM, monkeypatch) -> None:
    import tricast.analysis as analysis
    import tricast.nn.linear as linear_module

    original = analysis.gemm
    forward, measured = [], []

    def recording(calls: list):
        def gemm(a, b, spec, **kwargs):
            out = original(a, b, spec, **kwargs)
            calls.append((spec, kwargs, out.clone()))
            return out
        return gemm

    monkeypatch.setattr(linear_module, "gemm", recording(forward))
    monkeypatch.setattr(analysis, "gemm", recording(measured))
    result = layer_report(model, recipe(LOW_PRECISION), input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    [(forward_spec, forward_kwargs, forward_out)] = forward
    [(emulated_spec, emulated_kwargs, emulated), (exact_spec, exact_kwargs, exact)] = measured
    # The measured output is the forward's own GEMM, bit for bit; only the accumulator differs.
    assert emulated_spec == forward_spec and torch.equal(emulated, forward_out)
    assert exact_spec == MMASpec("fp64", out_format="fp32")
    assert emulated_kwargs["bias"] is forward_kwargs["bias"] is exact_kwargs["bias"]
    assert forward_kwargs["bias"] is not None
    ulp = result["layers"][0]["mma_ulp"]
    assert ulp == {**ulp_error(emulated, exact), "format": "fp32"}
    assert ulp["max"] > 0 and ulp["exact_fraction"] < 1.0
    assert ulp["n"] == 16 and ulp["nonfinite_mismatch"] == 0
    assert f"| {ulp['max']} / " in result["markdown"]


def test_report_mma_ulp_pools_windows(model: TinyLM, monkeypatch) -> None:
    import tricast.analysis as analysis

    original = analysis.gemm
    outputs = []

    def gemm(a, b, spec, **kwargs):
        out = original(a, b, spec, **kwargs)
        outputs.append(out.clone())
        return out

    monkeypatch.setattr(analysis, "gemm", gemm)
    result = layer_report(model, recipe(LOW_PRECISION), input_ids=[[0, 1, 2, 3], [3, 1, 0, 2]],
                          samples=2, seqlen=4)
    windows = [(outputs[i], outputs[i + 1]) for i in (0, 2)]
    distances = torch.cat([ulp_distance(emulated, exact).flatten() for emulated, exact in windows])
    per_window = [ulp_error(emulated, exact) for emulated, exact in windows]
    assert result["layers"][0]["mma_ulp"] == {
        "max": int(distances.max()), "mean": int(distances.sum()) / distances.numel(),
        "p99": max(window["p99"] for window in per_window),
        "exact_fraction": int((distances == 0).sum()) / distances.numel(), "n": 32,
        "nonfinite_mismatch": 0, "format": "fp32"}


def test_report_with_outliers_uses_the_effective_weight_and_both_paths(model: TinyLM, monkeypatch) -> None:
    import tricast.analysis as analysis
    from tricast.analysis import _operand_value, error_metrics

    calls = []
    original = analysis.gemm

    def gemm(activation, weight, spec, *args, **kwargs):
        calls.append(spec.algorithm)
        return original(activation, weight, spec, *args, **kwargs)

    monkeypatch.setattr(analysis, "gemm", gemm)
    data = recipe({"preset": "fp64", "out_format": "bf16"}, weight="fp8_tensor", activation="fp8_tensor",
                  outliers={"fraction": 0.25, "format": "fp32"})
    result = layer_report(model, data, input_ids=[[0, 1, 2, 3]], samples=1, seqlen=4)
    proj = result["layers"][0]
    assert proj["name"] == "proj"
    # fp64 on the main path in both runs; the outlier path is the same fp32 FMA chain in both.
    assert proj["mma_ulp"]["max"] == 0 and proj["mma_ulp"]["n"] == 16
    assert calls == ["fp64", "fp32_fma", "fp64", "fp32_fma"]
    # The weight error is taken on main + outliers: the 8 largest of 32 weights are kept exactly
    # in fp32, so only the other 24 carry fp8 error.
    import copy

    from tricast import load_recipe, patch_model

    patched = copy.deepcopy(model)
    patch_model(patched, load_recipe(data))
    layer = patched.proj
    effective = _operand_value(layer._weight_operand) + _operand_value(layer._outlier_operand)
    assert proj["weight"] == error_metrics(model.proj.weight.detach(), effective)
    outliers = _operand_value(layer._outlier_operand) != 0
    assert int(outliers.sum()) == 8
    assert torch.equal(effective[outliers], model.proj.weight.detach()[outliers])
