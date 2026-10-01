"""EmuLinear with sparsity and outliers: Triton output bit-identical to the reference backend.

The reference layer runs on CPU and the Triton layer on CUDA from the same weights, so structure
selection (stable sorts), both GEMMs, the fp32 add and the final cast are compared across devices
and backends. Recorders on the Triton entry points show that the kernels ran.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from tests.conftest import bit_equal
from tricast.nn import EmuLinear
from tricast.quant.structure import OutlierSpec, SparsitySpec, outlier_mask, sparsity_mask
from tricast.recipe import load_recipe

pytestmark = pytest.mark.gpu

K = 70  # a partial last n:m group, NVFP4 group and GDFS tile
CASES = {
    "fp8-2of4-outliers-bf16": {
        "weight": "fp8_tensor", "activation": "fp8_tensor", "mma": "nvidia_hopper_fp8",
        "sparsity": {"kind": "n:m", "n": 2, "m": 4}, "outliers": {"fraction": 0.05}},
    "nvfp4-unstructured-outliers-fp16": {
        "weight": "nvfp4", "activation": "nvfp4",
        "mma": {"preset": "nvidia_blackwell_fp4", "out_format": "fp16"},
        "sparsity": {"kind": "unstructured", "ratio": 0.3}, "outliers": {"fraction": 0.02, "format": "fp16"}},
    "fp8-row-1of3-outliers-fp32-out": {
        "weight": "fp8_row", "activation": "fp8_row",
        "mma": {"preset": "nvidia_hopper_fp8", "out_format": "fp32"},
        "sparsity": {"kind": "n:m", "n": 1, "m": 3}, "outliers": {"fraction": 0.1}},
}


@pytest.fixture
def _bits(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    return x.view({8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.int8}[x.element_size()])


def recorded(monkeypatch) -> list[str]:
    """Names of the Triton entry points called (GEMM and quantize/cast)."""
    mma = pytest.importorskip("tricast.kernels.mma")
    quantize = pytest.importorskip("tricast.kernels.quantize")
    calls: list[str] = []
    for module, name in ((mma, "gemm_triton"), (quantize, "quantize_triton")):
        original = getattr(module, name)

        def record(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, record)
    return calls


@pytest.mark.parametrize("rows", [3, 40])  # small-M _gemv and _gemm paths of the main GEMM
@pytest.mark.parametrize("case", list(CASES))
def test_structured_layer_triton_matches_reference(case, rows, recorded):
    spec = load_recipe({"name": case, "defaults": CASES[case]}).defaults
    gen = torch.Generator().manual_seed(42)
    linear = nn.Linear(K, 24)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(24, K, generator=gen) * 0.05)
        linear.bias.copy_(torch.randn(24, generator=gen))
    x = torch.randn(rows, K, generator=gen)
    reference = EmuLinear(copy.deepcopy(linear), spec, "reference", backend="reference").eval()
    kernel = EmuLinear(copy.deepcopy(linear).cuda(), spec, "triton", backend="triton").eval()
    for name in ("_weight_operand", "_outlier_operand"):
        expected_op, actual_op = getattr(reference, name), getattr(kernel, name)
        assert torch.equal(_bits(actual_op.values.cpu()), _bits(expected_op.values))  # -0 != +0
        assert (actual_op.scale is None) == (expected_op.scale is None)
        if expected_op.scale is not None:
            assert torch.equal(_bits(actual_op.scale.cpu()), _bits(expected_op.scale))
    recorded.clear()
    with torch.no_grad():
        expected = reference(x)
        actual = kernel(x.cuda()).cpu()
    assert recorded.count("gemm_triton") == 2  # the main and the outlier GEMM
    assert recorded.count("quantize_triton") == (1 if spec.mma.out_format.name == "fp32" else 2)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert bit_equal(actual, expected)


@pytest.mark.parametrize("sparsity", [SparsitySpec("n:m", n=2, m=4), SparsitySpec("n:m", n=5, m=64),
                                      SparsitySpec("unstructured", ratio=0.37)])
def test_selection_is_device_independent(sparsity):
    gen = torch.Generator().manual_seed(7)
    weight = torch.randint(-4, 5, (129, 1000), generator=gen).float() * 0.25  # heavy ties
    weight.view(-1)[::997] = float("nan")  # NaN ranks as the largest magnitude on both devices
    keep = sparsity_mask(weight, sparsity)
    assert torch.equal(sparsity_mask(weight.cuda(), sparsity).cpu(), keep)
    pruned = weight.masked_fill(~keep, 0)
    outliers = OutlierSpec(0.01)
    assert torch.equal(outlier_mask(pruned.cuda(), outliers, keep.cuda()).cpu(),
                       outlier_mask(pruned, outliers, keep))
