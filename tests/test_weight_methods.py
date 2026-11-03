"""Weight methods that replace the whole weight (tricast.recipe.WEIGHT_METHODS) against their definitions.

NF4 (tricast_native): the code table and thresholds against bitsandbytes 0.50.2 (functional.get_4bit_type and
the float literals of csrc/kernels.cu dQuantizeNF4), the code choice against a brute-force nearest-code
reference with exact distances, bitsandbytes' CPU path, and (GPU only) bitsandbytes' CUDA kernels bit for bit.
HQQ (official_adapter): the adapter is the official HQQLinear call with the documented settings. RaZeR
(tricast_native): bit for bit the pinned official clone's quant_nvfp4_razer_e4m3. Then the recipe path.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from tricast.nn.linear import EmuLinear
from tricast.recipe import WEIGHT_METHODS, WeightMethodSpec, load_recipe
from tricast.weight_quant import replace_weight
from tricast.weight_quant.nf4 import NF4_CODE, NF4_THRESHOLDS, nf4, nf4_dequantize, nf4_quantize
from tricast.weight_quant.razer import razer_nvfp4

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "src" / "tricast" / "schemas" /
                     "recipe.schema.json").read_text(encoding="utf-8"))
# dQuantizeNF4 in bitsandbytes 0.50.2 csrc/kernels.cu, in threshold order (codes i | i + 1)
KERNEL_LITERALS = ("-0.8480964004993439", "-0.6106329262256622", "-0.4599952697753906",
                   "-0.33967943489551544", "-0.23460740596055984", "-0.13791173323988914",
                   "-0.045525018125772476", "0.03979014977812767", "0.1202552504837513", "0.2035212516784668",
                   "0.2920137718319893", "0.3893125355243683", "0.5016634166240692", "0.6427869200706482",
                   "0.8614784181118011")
ABOVE_MIDPOINT = {1, 3, 4, 8}  # thresholds half an fp32 ulp above the exact midpoint of their codes


def weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {"normal": torch.randn(256, 512) * 0.02,
            "student_t2": torch.distributions.StudentT(2.0).sample((256, 512)) * 0.02,
            "bf16": (torch.randn(128, 256) * 0.05).bfloat16()}


def f32(value: float) -> float:
    return torch.tensor(value, dtype=torch.float32).item()


def nearest_f32(decimal: str) -> float:
    """The fp32 nearest the exact decimal (what a C float literal is)."""
    exact = Fraction(decimal)
    start = torch.tensor(float(exact), dtype=torch.float32)
    down, up = torch.tensor(-2.0), torch.tensor(2.0)
    candidates = [torch.nextafter(start, down), start, torch.nextafter(start, up)]
    return min((c.item() for c in candidates), key=lambda c: abs(Fraction(c) - exact))


def brute_force_nf4(x: torch.Tensor, blocksize: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest NF4 code of the kernel's fp32 ``x * (1 / absmax)``, distances exact in fp64, ties to the lower
    code."""
    blocks = x.reshape(-1, blocksize).float()
    absmax = blocks.abs().amax(dim=1)
    scaled = blocks * (1.0 / absmax)[:, None]
    distance = (scaled.double().unsqueeze(-1) - torch.tensor(NF4_CODE, dtype=torch.float64)).abs()
    nearest = distance == distance.min(dim=-1, keepdim=True).values
    return nearest.to(torch.int64).argmax(dim=-1).reshape(x.shape), absmax  # argmax: the first (lower) code


# ---------- NF4 ----------
def test_nf4_constants_are_the_bitsandbytes_ones() -> None:
    assert all(f32(value) == value for value in NF4_CODE) and list(NF4_CODE) == sorted(NF4_CODE)
    assert [nearest_f32(text) for text in KERNEL_LITERALS] == list(NF4_THRESHOLDS)
    for i, threshold in enumerate(NF4_THRESHOLDS):
        midpoint = (Fraction(NF4_CODE[i]) + Fraction(NF4_CODE[i + 1])) / 2
        assert (Fraction(threshold) > midpoint) == (i in ABOVE_MIDPOINT), i
    functional = pytest.importorskip("bitsandbytes.functional")
    assert torch.equal(functional.get_4bit_type("nf4", device="cpu").float(), torch.tensor(NF4_CODE))


@pytest.mark.parametrize("name", ["normal", "student_t2", "bf16"])
def test_nf4_is_the_nearest_code(name: str) -> None:
    x = weights()[name]
    index, absmax = nf4_quantize(x)
    expected, expected_absmax = brute_force_nf4(x)
    assert torch.equal(index.long(), expected) and torch.equal(absmax, expected_absmax)
    assert len(index.unique()) == 16  # every code is used
    dequant = torch.tensor(NF4_CODE)[expected].reshape(-1, 64) * expected_absmax[:, None]
    assert torch.equal(nf4(x).view(torch.int16), dequant.reshape(x.shape).bfloat16().view(torch.int16))


def test_nf4_boundaries_follow_the_kernel_tree() -> None:
    """At each threshold t (scaled = x exactly: the block's absmax is 1): the fp32 below t takes code i, the
    one above code i + 1, t itself code i — where the exact nearest code agrees except at the four thresholds
    above their midpoint (nearest: code i + 1)."""
    below = [torch.nextafter(torch.tensor(t), torch.tensor(-2.0)).item() for t in NF4_THRESHOLDS]
    above = [torch.nextafter(torch.tensor(t), torch.tensor(2.0)).item() for t in NF4_THRESHOLDS]
    probes = torch.tensor([value for triple in zip(below, NF4_THRESHOLDS, above, strict=True)
                           for value in triple], dtype=torch.float32)
    block = torch.zeros(64)
    block[0] = 1.0
    block[1:1 + probes.numel()] = probes
    index = nf4_quantize(block)[0][1:1 + probes.numel()].reshape(-1, 3).tolist()
    nearest = brute_force_nf4(block)[0][1:1 + probes.numel()].reshape(-1, 3).tolist()
    assert index == [[i, i, i + 1] for i in range(15)]
    assert nearest == [[i, i + (i in ABOVE_MIDPOINT), i + 1] for i in range(15)]


def test_nf4_all_zero_block_is_the_kernels_nan_branch() -> None:
    """absmax 0: 0 * (1 / 0) is NaN, every test of the tree is false, code 0 (-1.0) times 0 = -0.0."""
    x = torch.zeros(2, 64)
    x[1, 3] = 0.5
    index, absmax = nf4_quantize(x)
    assert index[0].tolist() == [0] * 64 and absmax.tolist() == [0.0, 0.5]
    out = nf4(x)
    assert torch.all(out[0] == 0) and torch.all(torch.signbit(out[0])) and out[1, 3].item() == 0.5


def test_nf4_partial_last_block_has_its_own_absmax() -> None:
    x = torch.arange(1, 81, dtype=torch.float32)  # 64 + 16
    index, absmax = nf4_quantize(x)
    assert absmax.tolist() == [64.0, 80.0] and index.shape == x.shape
    assert torch.equal(nf4_dequantize(index, absmax)[[63, 79]], torch.tensor([64.0, 80.0]))


def test_nf4_matches_the_bitsandbytes_cpu_path() -> None:
    functional = pytest.importorskip("bitsandbytes.functional")
    for name, x in weights().items():
        w = x.bfloat16()
        packed, state = functional.quantize_4bit(w, blocksize=64, quant_type="nf4")
        reference = functional.dequantize_4bit(packed, state)
        assert torch.equal(state.absmax, nf4_quantize(w)[1]), name
        assert torch.equal(reference.view(torch.int16), nf4(w).view(torch.int16)), name


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="compares with bitsandbytes' CUDA NF4 kernels, which need a CUDA GPU")
def test_nf4_matches_bitsandbytes_cuda_bit_for_bit() -> None:
    functional = pytest.importorskip("bitsandbytes.functional")
    for name, x in weights().items():
        w = x.bfloat16().cuda()
        packed, state = functional.quantize_4bit(w, blocksize=64, quant_type="nf4")
        reference = functional.dequantize_4bit(packed, state)
        index, absmax = nf4_quantize(w)
        flat = index.reshape(-1)
        assert torch.equal(packed.reshape(-1), (flat[0::2] << 4) | flat[1::2]), name  # high nibble first
        assert torch.equal(state.absmax, absmax), name
        assert torch.equal(reference.view(torch.int16), nf4(w).view(torch.int16)), name


# ---------- HQQ ----------
def test_hqq_adapter_is_the_official_call() -> None:
    quantize = pytest.importorskip("hqq.core.quantize")
    from tricast.weight_quant.hqq import hqq_int4

    for name, x in weights().items():
        config = quantize.BaseQuantizeConfig(nbits=4, group_size=64)
        assert config["weight_quant_params"] == {"nbits": 4, "channel_wise": True, "group_size": 64,
                                                 "optimize": True, "round_zero": True, "axis": 1,
                                                 "view_as_float": False}
        layer = quantize.HQQLinear.from_weights(x.clone(), None, config, compute_dtype=torch.bfloat16,
                                                device="cpu")
        ours = hqq_int4(x)
        assert ours.dtype == torch.bfloat16 and torch.equal(ours.view(torch.int16),
                                                            layer.dequantize().view(torch.int16)), name
        groups = x.numel() // 64  # one BF16 scale and one BF16 zero per 64 along the input dim
        assert {key: (layer.meta[key].dtype, tuple(layer.meta[key].shape)) for key in ("scale", "zero")} == {
            "scale": (torch.bfloat16, (groups, 1)), "zero": (torch.bfloat16, (groups, 1))}
        assert max(len(group.unique()) for group in ours.reshape(-1, 64)) <= 16
        plain = dict(config, weight_quant_params={**config["weight_quant_params"], "optimize": False})
        unoptimized = quantize.HQQLinear.from_weights(x.clone(), None, plain, compute_dtype=torch.bfloat16,
                                                      device="cpu").dequantize()
        assert not torch.equal(ours, unoptimized), name  # the proximal solver ran


def test_hqq_adapter_refuses_another_release(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("hqq")
    from tricast.weight_quant import hqq

    monkeypatch.setattr(hqq, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match=r"pinned to hqq==0\.2\.8\.post1"):
        hqq.hqq_int4(torch.ones(4, 64))


# ---------- RaZeR ----------
def test_razer_is_the_official_function_bit_for_bit() -> None:
    from tricast.official import repos

    official = repos.load_module("razer", "quantize.quantizer")
    cases = weights()
    torch.manual_seed(1)
    tiny = torch.randn(64, 128)
    tiny[:32] *= 1e-6  # blocks far below the tensor amax: the 2^-9 block-scale clamp
    tiny[40] = 0  # all-zero blocks
    cases["clamped"] = tiny
    for name, x in cases.items():
        ours = razer_nvfp4(x)
        reference = official.quant_nvfp4_razer_e4m3(x, groupsize=16)
        assert ours.dtype == reference.dtype == torch.bfloat16 and ours.shape == x.shape
        assert torch.equal(ours.view(torch.int16), reference.view(torch.int16)), name
        remapped = (ours != official.quant_nvfp4(x, groupsize=16)).float().mean().item()
        assert 0 < remapped < 0.5, (name, remapped)  # the special value replaces some FP4 values


# ---------- the recipe path ----------
def test_schema_and_spec_list_the_weight_methods() -> None:
    assert SCHEMA["$defs"]["weight_method"]["enum"] == list(WEIGHT_METHODS)
    for kind in WEIGHT_METHODS:
        for given in (kind, {"kind": kind}, {"kind": kind, "block_size": 64, "damp": 0.0, "act_order": True}):
            spec = load_recipe({"name": kind, "defaults": {"weight": None, "weight_algo": given}}).defaults
            assert type(spec.weight_algo) is WeightMethodSpec and spec.weight_algo.kind == kind
            assert not load_recipe({"name": kind, "defaults": {"weight": None, "weight_algo": given}}
                                   ).needs_calibration
    with pytest.raises(ValueError, match=r"defaults\.weight: must be null with weight_algo nf4"):
        load_recipe({"name": "both", "defaults": {"weight": "nvfp4", "weight_algo": "nf4"}})
    with pytest.raises(ValueError, match=r"overrides\[0\]\.weight: must be null with weight_algo razer"):
        load_recipe({"name": "both", "defaults": {"weight": "nvfp4"},
                     "overrides": [{"match": "*", "weight_algo": "razer"}]})
    with pytest.raises(ValueError, match="unknown weight algorithm"):
        WeightMethodSpec("gptq")


@pytest.mark.parametrize("kind", WEIGHT_METHODS)
def test_a_patched_layer_uses_the_replaced_weight(kind: str) -> None:
    if kind == "hqq":
        pytest.importorskip("hqq")
    torch.manual_seed(0)
    linear = torch.nn.Linear(128, 64)
    activation = {"scheme": "nvfp4", "mma_input": "dequant"} if kind == "razer" else None
    spec = load_recipe({"name": kind, "defaults": {"weight": None, "activation": activation,
                                                   "weight_algo": kind}}).defaults
    layer = EmuLinear(linear, spec, "layer")
    expected = replace_weight(linear.weight.detach(), spec.weight_algo)
    assert expected.dtype == torch.bfloat16 and not torch.equal(expected.float(), linear.weight.detach())
    assert torch.equal(layer._weight_operand.values.view(torch.int16), expected.view(torch.int16))
    assert layer._weight_operand.fmt.name == "bf16"
    x = torch.randn(3, 128)
    out = layer(x)
    plain = EmuLinear(linear, load_recipe({"name": "plain", "defaults": {"weight": None}}).defaults, "plain")
    assert torch.isfinite(out).all() and not torch.equal(out, plain(x))


@pytest.mark.parametrize("kind", WEIGHT_METHODS)
def test_a_replacing_method_refuses_paths_that_would_not_see_its_weight(kind: str) -> None:
    """Review 2026-11-03 F1/F2: official GPTQ would quantize the replaced weight twice, and a transform fit
    (AWQ's scale search) quantizes with the weight spec, which is null for these methods."""
    with pytest.raises(ValueError, match="official_gptq: cannot be combined with weight_algo"):
        load_recipe({"name": "twice", "defaults": {"weight": None, "weight_algo": kind},
                     "official_gptq": {"wbits": 4}})
    for transform in ("awq", "smoothquant", "hadamard"):
        with pytest.raises(ValueError, match="transform: not supported with weight_algo"):
            load_recipe({"name": "fit", "defaults": {"weight": None, "weight_algo": kind,
                                                     "transform": transform}})
