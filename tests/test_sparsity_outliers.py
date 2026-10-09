"""Weight sparsity and outlier preservation: selection and tie rules, recipe validation with
paths, the EmuLinear operands and forward composition, and the bundled recipes (CPU, reference)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest
import torch
from torch import nn

from tricast.formats import BF16, FP16, FP32
from tricast.mma.api import as_operand, gemm
from tricast.mma.spec import MMASpec
from tricast.nn import EmuLinear, iter_emulinear, patch_model
from tricast.quant.api import quantize
from tricast.quant.spec import QuantSpec, WeightAlgoSpec, get_scheme
from tricast.quant.structure import OutlierSpec, SparsityKind, SparsitySpec, outlier_mask, sparsity_mask
from tricast.recipe import LinearSpec, load_recipe
from tricast.reference.cast import round_to_format

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "src/tricast/schemas/recipe.schema.json").read_text())
NM = {"kind": "n:m", "n": 2, "m": 4}
W = {"weight": "fp8_tensor"}


def _layer(defaults: dict, in_features: int = 12, out_features: int = 5) -> EmuLinear:
    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features)
    spec = load_recipe({"name": "layer", "defaults": defaults}).defaults
    return EmuLinear(linear, spec, "layer", backend="reference").eval()


# --- selection rules ------------------------------------------------------------------------

def test_nm_keeps_largest_with_lower_k_on_ties_and_partial_last_group():
    weight = torch.tensor([
        [1.0, -1.0, 1.0, 0.5, 3.0, -3.0, 3.0],  # three-way tie keeps K 0, 1; partial group keeps 4, 5
        [0.0, -0.0, 0.0, 0.0, 0.0, 9.0, -9.0],  # all-zero group keeps K 0, 1
    ])
    assert sparsity_mask(weight, SparsitySpec("n:m", n=2, m=4)).tolist() == [
        [True, True, False, False, True, True, False],
        [True, True, False, False, False, True, True],
    ]
    # A last group of length L < n keeps min(n, L) = L weights.
    short = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.5]])
    assert sparsity_mask(short, SparsitySpec("n:m", n=3, m=4)).tolist() == [[True, True, True, False, True]]


@pytest.mark.parametrize("n,m,k", [(2, 4, 13), (1, 3, 9), (5, 8, 20), (7, 64, 70)])
def test_nm_matches_brute_force_ranking(n, m, k, gen):
    weight = torch.randint(-3, 4, (5, k), generator=gen).float() * 0.5  # many ties, signed zeros
    expected = torch.zeros(5, k, dtype=torch.bool)
    for row in range(5):
        for start in range(0, k, m):
            ranked = sorted(range(start, min(start + m, k)), key=lambda j: (-abs(weight[row, j].item()), j))
            expected[row, ranked[:n]] = True
    assert torch.equal(sparsity_mask(weight, SparsitySpec("n:m", n=n, m=m)), expected)
    assert torch.equal(sparsity_mask(weight.bfloat16(), SparsitySpec("n:m", n=n, m=m)), expected)


def test_unstructured_prunes_floor_ratio_smallest_lower_index_first():
    weight = torch.tensor([[0.5, -0.25, 0.25, 1.0, -0.5], [0.25, 2.0, -1.0, 0.0, 0.25]])
    # floor(0.45 * 10) = 4 pruned: |w| 0 (flat 8), then the 0.25 ties at flat 1, 2, 5 — not 9.
    assert sparsity_mask(weight, SparsitySpec("unstructured", ratio=0.45)).tolist() == [
        [True, False, False, True, True], [False, True, True, False, True]]
    assert sparsity_mask(weight, SparsitySpec("unstructured", ratio=0.05)).all()  # floor(0.5) = 0
    # The ratio is read as written: 0.29 of 100 is 29 (the binary product is 28.999...).
    ramp = torch.arange(1, 101, dtype=torch.float32).reshape(4, 25)
    assert int((~sparsity_mask(ramp, SparsitySpec("unstructured", ratio=0.29))).sum()) == 29


def test_outliers_take_ceil_fraction_largest_kept_lower_index_first():
    weight = torch.tensor([[3.0, -1.0, 3.0, 0.5], [-3.0, 2.0, 0.25, 0.0]])
    # ceil(0.25 * 8) = 2 of the three |3| ties: flat 0 and 2.
    assert outlier_mask(weight, OutlierSpec(0.25)).tolist() == [[True, False, True, False],
                                                                [False, False, False, False]]
    assert int(outlier_mask(weight, OutlierSpec(0.3)).sum()) == 3  # ceil(2.4)
    keep = torch.tensor([[False, True, False, True], [False, False, False, True]])
    pruned = weight.masked_fill(~keep, 0)
    # ceil(0.3 * 8) = 3 = every kept weight, including the kept zero at flat 7; the pruned zeros at
    # lower flat indices would win that tie without the mask.
    assert torch.equal(outlier_mask(pruned, OutlierSpec(0.3), keep), keep)
    assert not torch.equal(outlier_mask(pruned, OutlierSpec(0.3)), keep)
    assert torch.equal(outlier_mask(pruned, OutlierSpec(0.9), keep), keep)  # capped at the 3 kept
    ramp = torch.arange(1000, dtype=torch.float32).reshape(10, 100)
    assert int(outlier_mask(ramp, OutlierSpec(0.005)).sum()) == 5  # not ceil(5.000...1) = 6


def test_nan_ranks_as_the_largest_magnitude():
    weight = torch.tensor([[1.0, float("nan"), -2.0, 0.5]])
    unstructured = SparsitySpec("unstructured", ratio=0.5)
    assert sparsity_mask(weight, unstructured).tolist() == [[False, True, True, False]]
    assert sparsity_mask(weight, SparsitySpec("n:m", n=1, m=4)).tolist() == [[False, True, False, False]]
    assert outlier_mask(weight, OutlierSpec(0.25)).tolist() == [[False, True, False, False]]


# --- recipe and spec validation -------------------------------------------------------------

@pytest.mark.parametrize("data,message", [
    ({"defaults": {"sparsity": {"kind": "n:m", "n": 0, "m": 4}}}, "defaults.sparsity.n:"),
    ({"defaults": {"sparsity": {"kind": "n:m", "n": 4, "m": 4}}}, "defaults.sparsity.n:"),
    ({"defaults": {"sparsity": {"kind": "n:m", "n": 2, "m": 65}}}, "defaults.sparsity.m:"),
    ({"defaults": {"sparsity": {"kind": "n:m", "n": 2}}}, "defaults.sparsity.m: required"),
    ({"defaults": {"sparsity": {"kind": "unstructured"}}}, "defaults.sparsity.ratio: required"),
    ({"defaults": {"sparsity": {"kind": "unstructured", "ratio": 1.0}}}, "defaults.sparsity.ratio:"),
    ({"defaults": {"sparsity": {"kind": "unstructured", "ratio": 0}}}, "defaults.sparsity.ratio:"),
    ({"defaults": {"sparsity": {"kind": "none", "n": 2}}}, "defaults.sparsity.n: not used"),
    ({"defaults": {"sparsity": {"kind": "2:4"}}}, "defaults.sparsity.kind:"),
    ({"defaults": {**W, "outliers": {"fraction": 0}}}, "defaults.outliers.fraction:"),
    ({"defaults": {**W, "outliers": {"fraction": 1.5}}}, "defaults.outliers.fraction:"),
    ({"defaults": {**W, "outliers": {"format": "bf16"}}}, "defaults.outliers: 'fraction' is a required"),
    ({"defaults": {**W, "outliers": {"fraction": 0.1, "format": "int8"}}}, "defaults.outliers.format:"),
    ({"defaults": {**W, "outliers": {"fraction": 0.1, "format": "nope"}}}, "defaults.outliers.format:"),
    ({"defaults": {"outliers": {"fraction": 0.1}}}, "defaults.outliers: requires a weight QuantSpec"),
    ({"defaults": {**W, "weight_algo": "gptq", "sparsity": NM}},
     "defaults.sparsity: not supported with gptq"),
    ({"defaults": {**W, "weight_algo": "gptq", "outliers": {"fraction": 0.1}}},
     "defaults.outliers: not supported with gptq"),
    ({"defaults": W, "overrides": [{"match": "*", "sparsity": {"kind": "n:m", "n": 3, "m": 2}}]},
     "overrides[0].sparsity.n:"),
    ({"defaults": {**W, "sparsity": NM}, "overrides": [{"match": "*", "weight_algo": "gptq"}]},
     "overrides[0].sparsity: not supported with gptq"),
    ({"defaults": {**W, "outliers": {"fraction": 0.1}}, "overrides": [{"match": "*", "weight": None}]},
     "overrides[0].outliers: requires a weight QuantSpec"),
])
def test_invalid_structure_reports_recipe_path(data, message):
    with pytest.raises(ValueError) as error:
        load_recipe({"name": "invalid", **data})
    assert message in str(error.value)


def test_specs_validate_when_built_in_python():
    with pytest.raises(ValueError, match="^outliers: requires a weight QuantSpec"):
        LinearSpec(outliers=OutlierSpec(0.1))
    with pytest.raises(ValueError, match="^sparsity: not supported with gptq"):
        LinearSpec(weight=get_scheme("fp8_tensor"), weight_algo=WeightAlgoSpec("gptq"),
                   sparsity=SparsitySpec("unstructured", ratio=0.5))
    with pytest.raises(ValueError, match="^ratio: not used by kind 'n:m'"):
        SparsitySpec("n:m", n=2, m=4, ratio=0.5)
    assert OutlierSpec(0.1, "fp16").format == FP16 and OutlierSpec(0.1).format == BF16


def test_schema_sparsity_kinds_match_spec():
    assert set(SCHEMA["$defs"]["sparsity_kind"]["enum"]) == set(get_args(SparsityKind))


def test_overrides_replace_sparsity_and_can_disable_outliers():
    recipe = load_recipe({"name": "structure", "defaults": {
        **W, "sparsity": NM, "outliers": {"fraction": 0.01, "format": "fp16"}}, "overrides": [
        {"match": "a", "sparsity": {"kind": "unstructured", "ratio": 0.5}, "outliers": None},
        {"match": "b", "sparsity": {"kind": "none"}, "outliers": {"fraction": 0.02}},
    ]})
    a, b, c = (recipe.spec_for(name) for name in ("a", "b", "c"))
    assert a.sparsity == SparsitySpec("unstructured", ratio=0.5) and a.outliers is None
    assert b.sparsity == SparsitySpec() and b.outliers == OutlierSpec(0.02, FP16)  # format inherited
    assert c.sparsity == SparsitySpec("n:m", n=2, m=4) and c.outliers == OutlierSpec(0.01, FP16)
    assert recipe.to_dict()["defaults"]["sparsity"] == NM
    again = load_recipe(recipe.to_dict())
    assert again.sha256 == recipe.sha256 and again.spec_for("a") == a and again.spec_for("b") == b


def test_recipes_without_structure_keep_their_json_form():
    defaults = load_recipe("hopper_fp8_w8a8").to_dict()["defaults"]
    assert set(defaults) == {"weight", "activation", "mma", "transform", "weight_algo"}
    explicit = load_recipe({"name": "x", "defaults": {**W, "sparsity": {"kind": "none"}, "outliers": None}})
    assert explicit.sha256 == load_recipe({"name": "x", "defaults": W}).sha256


# --- EmuLinear operands and forward -------------------------------------------------------

def test_operands_split_the_pruned_weight_exactly():
    layer = _layer({"weight": {"format": "fp32", "scale": None}, "sparsity": NM,
                    "outliers": {"fraction": 0.2, "format": "fp32"}})
    weight = layer.weight.detach()
    keep = sparsity_mask(weight, layer.spec.sparsity)
    pruned = weight.masked_fill(~keep, 0)
    selected = outlier_mask(pruned, layer.spec.outliers, keep)
    main = layer._weight_operand.values.float()
    extra = layer._outlier_operand.values.float()
    assert int(selected.sum()) == math.ceil(0.2 * weight.numel()) == 12
    assert torch.equal(main + extra, pruned)
    assert not main[~keep | selected].any() and not extra[~selected].any()
    assert extra[selected].all()


def test_main_scales_exclude_outliers_and_outliers_round_to_format():
    layer = _layer({"weight": "fp8_row", "outliers": {"fraction": 0.1}})
    weight = layer.weight.detach()
    selected = outlier_mask(weight, layer.spec.outliers)
    expected = as_operand(quantize(weight.masked_fill(selected, 0), layer.spec.weight, backend="reference"))
    dense = as_operand(quantize(weight, layer.spec.weight, backend="reference"))
    assert torch.equal(layer._weight_operand.values.float(), expected.values)
    assert torch.equal(layer._weight_operand.scale, expected.scale)
    assert not torch.equal(layer._weight_operand.scale, dense.scale)
    assert layer._outlier_operand.fmt == BF16
    assert torch.equal(layer._outlier_operand.values.float(),
                       round_to_format(weight.masked_fill(~selected, 0), BF16))


def test_pruned_weights_are_exact_zeros_under_float_zero_points():
    # Affine float zero points do not map 0 back to 0, so the operand is zeroed explicitly.
    layer = _layer({"weight": {"format": "uint4", "granularity": "row", "scale": {"format": "fp16"},
                               "zero_point": "float", "mma_input": "dequant"},
                    "sparsity": {"kind": "unstructured", "ratio": 0.5}})
    keep = sparsity_mask(layer.weight.detach(), layer.spec.sparsity)
    unmasked = as_operand(quantize(layer.weight.detach().masked_fill(~keep, 0), layer.spec.weight,
                                   backend="reference"))
    assert unmasked.values[~keep].any()  # what the operand would hold without the explicit zeros
    assert not layer._weight_operand.values[~keep].any()
    assert torch.equal(layer._weight_operand.values.float()[keep], unmasked.values[keep])


@pytest.mark.parametrize("defaults", [
    {"weight": "fp8_tensor", "activation": "fp8_tensor", "mma": "nvidia_hopper_fp8",
     "sparsity": NM, "outliers": {"fraction": 0.05}},
    {"weight": "nvfp4", "activation": "nvfp4",
     "mma": {"preset": "nvidia_blackwell_fp4", "out_format": "fp16"},
     "sparsity": {"kind": "unstructured", "ratio": 0.3}, "outliers": {"fraction": 0.02, "format": "fp16"}},
    {"weight": "fp8_row", "activation": "fp8_row",
     "mma": {"preset": "nvidia_hopper_fp8", "out_format": "fp32"}, "outliers": {"fraction": 0.1}},
], ids=["fp8-2of4-bf16", "nvfp4-unstructured-fp16", "fp8-row-fp32"])
def test_forward_with_outliers_is_the_manual_composition(defaults, gen, monkeypatch):
    import tricast.nn.linear as linear_module

    layer = _layer(defaults, in_features=40, out_features=6)
    spec = layer.spec
    calls = []
    original = linear_module.gemm

    def tracked(a, b, mma, **kwargs):
        calls.append((mma.algorithm, mma.out_format))
        return original(a, b, mma, **kwargs)

    monkeypatch.setattr(linear_module, "gemm", tracked)
    x = torch.randn(3, 40, generator=gen)
    with torch.no_grad():
        actual = layer(x)

    weight = layer.weight.detach()
    keep = sparsity_mask(weight, spec.sparsity)
    pruned = weight.masked_fill(~keep, 0)
    selected = outlier_mask(pruned, spec.outliers, keep)
    main_weight = as_operand(quantize(pruned.masked_fill(selected, 0), spec.weight, backend="reference"))
    outlier_weight = as_operand(quantize(pruned.masked_fill(~selected, 0),
                                         QuantSpec(spec.outliers.format, scale=None), backend="reference"))
    activation = as_operand(quantize(x, spec.activation, backend="reference"))
    main = gemm(activation, main_weight, replace(spec.mma, out_format=FP32), bias=layer.bias,
                backend="reference")
    extra = gemm(activation, outlier_weight, MMASpec("fp32_fma", out_format=FP32), backend="reference")
    expected = round_to_format(main + extra, spec.mma.out_format, "rne", saturate=False)
    assert calls == [(spec.mma.algorithm, FP32), ("fp32_fma", FP32)]
    assert extra.abs().sum() > 0
    assert actual.dtype == torch.float32
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_sparsity_alone_keeps_a_single_gemm(monkeypatch):
    import tricast.nn.linear as linear_module

    layer = _layer({"weight": "fp8_tensor", "mma": {"preset": "fp64", "out_format": "fp32"},
                    "sparsity": {"kind": "unstructured", "ratio": 0.5}})
    calls = []
    original = linear_module.gemm
    monkeypatch.setattr(linear_module, "gemm", lambda *a, **k: calls.append(a[2]) or original(*a, **k))
    x = torch.randn(2, 12)
    with torch.no_grad():
        actual = layer(x)
    assert calls == [layer.spec.mma] and layer._outlier_operand is None
    weight = layer.weight.detach()
    pruned = weight.masked_fill(~sparsity_mask(weight, layer.spec.sparsity), 0)
    expected = gemm(x, quantize(pruned, layer.spec.weight, backend="reference"), layer.spec.mma,
                    bias=layer.bias, backend="reference")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("defaults", [{**W, "sparsity": NM}, {**W, "outliers": {"fraction": 0.1}}])
def test_training_with_grad_is_rejected_inference_runs(defaults):
    layer = _layer(defaults)
    x = torch.randn(2, 12)
    layer.train()
    with pytest.raises(NotImplementedError, match="inference-only"):
        layer(x)
    with torch.no_grad():
        assert torch.isfinite(layer(x)).all()
    layer.eval()
    assert torch.isfinite(layer(x)).all()


def test_failed_recalibration_restores_the_outlier_operand(monkeypatch):
    from tricast.calibration import calibrate

    class TokenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 16)
            self.a = nn.Linear(16, 16)
            self.b = nn.Linear(16, 8)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(self.embed(input_ids)))

    torch.manual_seed(0)
    model = TokenModel().eval()
    recipe = load_recipe({"name": "sq", "defaults": {
        **W, "activation": "fp8_tensor", "transform": "smoothquant", "outliers": {"fraction": 0.1},
        "mma": {"preset": "fp64", "out_format": "fp32"}}})
    patch_model(model, recipe, backend="reference")
    ids = torch.tensor([[1, 2, 3, 4]])
    calibrate(model, recipe, input_ids=ids, seqlen=4, samples=1, seed=42)
    saved = {name: (layer._weight_operand, layer._outlier_operand) for name, layer in iter_emulinear(model)}
    assert all(outliers is not None for _, outliers in saved.values())
    finish = EmuLinear.finish_calibration

    def failing(self, *args, **kwargs):
        finish(self, *args, **kwargs)  # rebuilds both operands, then fails
        if self.name == "b":
            raise RuntimeError("injected")

    monkeypatch.setattr(EmuLinear, "finish_calibration", failing)
    with pytest.raises(RuntimeError, match="injected"):
        calibrate(model, recipe, input_ids=ids, seqlen=4, samples=1, seed=42)
    for name, layer in iter_emulinear(model):
        assert layer._weight_operand is saved[name][0] and layer._outlier_operand is saved[name][1]


# --- bundled recipes ------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(32, 48)
        self.down = nn.Linear(48, 16)
        self.lm_head = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.down(torch.relu(self.up(x))))


@pytest.mark.parametrize("name", ["fp8_2of4_sparse", "nvfp4_outliers"])
def test_bundled_structure_recipes_patch_and_run(name):
    recipe = load_recipe(name)
    torch.manual_seed(42)
    model = _TinyModel().eval()
    report = patch_model(model, recipe, backend="reference")
    assert [patched for patched, _ in report.patched] == ["up", "down"] and report.skipped == ["lm_head"]
    with torch.no_grad():
        y = model(torch.randn(2, 3, 32))
    assert y.shape == (2, 3, 8) and torch.isfinite(y).all()
    for _, layer in iter_emulinear(model):
        weight = layer.weight.detach()
        values = layer._weight_operand.values.float()
        if name == "fp8_2of4_sparse":
            assert layer.spec.sparsity == SparsitySpec("n:m", n=2, m=4) and layer.spec.outliers is None
            keep = sparsity_mask(weight, layer.spec.sparsity)
            assert int(keep.sum()) == weight.numel() // 2 and not values[~keep].any()
            assert layer._outlier_operand is None
        else:
            assert layer.spec.outliers == OutlierSpec(0.005, BF16) and layer.spec.sparsity.kind == "none"
            selected = outlier_mask(weight, layer.spec.outliers)
            assert int(selected.sum()) == math.ceil(0.005 * weight.numel())  # 8 of 1536, 4 of 768
            assert not values[selected].any()
            assert torch.equal(layer._outlier_operand.values != 0, selected)
            assert layer._outlier_operand.fmt == BF16


@pytest.mark.parametrize("field", ["n", "m"])
def test_sparsity_rejects_non_integer_counts(field: str) -> None:
    sparsity = {"kind": "n:m", "n": 2, "m": 4, field: {"n": 2.0, "m": 4.0}[field]}
    with pytest.raises(ValueError, match=rf"^defaults\.sparsity\.{field}: expected an integer"):
        load_recipe({"name": "float-count", "defaults": {"weight": "fp8_tensor", "sparsity": sparsity}})


@pytest.mark.parametrize("key", ["weight", "outliers"])
def test_override_format_mapping_replaces_the_default_format(key: str) -> None:
    from tricast.formats import get_format

    half = {"kind": "float", "name": "half", "ebits": 5, "mbits": 10}
    defaults = {"weight": {"format": "e5m10:nosub"}, "outliers": {"fraction": 0.01, "format": "e5m10:nosub"}}
    changes = ({"weight": {"format": half}} if key == "weight"
               else {"outliers": {"fraction": 0.01, "format": half}})
    recipe = load_recipe({"name": "format-override", "defaults": defaults,
                          "overrides": [{"match": "a", **changes}]})
    spec = recipe.spec_for("a")
    resolved = spec.weight.format if key == "weight" else spec.outliers.format
    assert resolved == get_format(half)  # not the default's nosub variant
