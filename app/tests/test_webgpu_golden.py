"""Browser (WebGPU) golden vectors and the real-operand exporter, on the CPU in a few seconds."""

from __future__ import annotations

import json

import pytest
import torch

from app import export_operands, webgpu_golden
from tricast.formats import FP8_E4M3, FP32
from tricast.mma.api import gemm
from tricast.mma.operand import Operand
from tricast.mma.spec import MMASpec


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(webgpu_golden.OUT.read_text())


def _words(hex_bits: str) -> list[int]:
    return [int(hex_bits[i:i + 8], 16) for i in range(0, len(hex_bits), 8)]


def test_golden_covers_the_requested_settings(golden):
    cases = golden["cases"]
    assert len(cases) == len(webgpu_golden.plan()) >= 250
    mma = [case["mma"] for case in cases]
    assert {m["f_bits"] for m in mma} >= set(webgpu_golden.F_BITS)
    assert {m["chunk_size"] for m in mma} >= set(webgpu_golden.CHUNKS)
    assert {m["c_mode"] for m in mma} == {"fused", "decoupled"}
    assert {m["norm_rounding"] for m in mma} == {"rtz", "rne"}
    assert {m["f2_bits"] for m in mma if m["c_mode"] == "decoupled"} >= set(webgpu_golden.F2_BITS)
    assert all(("f2_bits" in m) == (m["c_mode"] == "decoupled") and m["promote_interval"] == 0 for m in mma)
    assert {case["K"] for case in cases} >= set(webgpu_golden.KS)
    tags = {tag for case in cases for tag in case["tags"]}
    assert {"nan", "cancel", "subnormal", "emax_mf0", "maxnormal", "scale_tiny", "scale_huge",
            "scale_negative"} <= tags
    bits = [word for case in cases for word in _words(case["expected"])]
    assert len(bits) == sum(case["M"] * case["N"] for case in cases)
    exponent = [(word >> 23) & 0xFF for word in bits]
    assert any(e == 0xFF and word & 0x7FFFFF for e, word in zip(exponent, bits, strict=True))  # NaN
    assert any(word & 0x7FFFFFFF == 0x7F800000 for word in bits)  # Inf from the epilogue
    assert any(e == 0 and word & 0x7FFFFF for e, word in zip(exponent, bits, strict=True))  # subnormal
    assert 0x80000000 in bits  # -0 (negative scale times a zero sum)


def _cheap(index: int) -> bool:
    item = webgpu_golden.plan()[index]
    return item.K <= 128 and -(-item.K // item.chunk_size) <= 64


def _fp64_nan_class(case: dict) -> dict:
    """``case`` with every ``fp64`` NaN as 7fc00000 (NaN sign and payload follow the CPU: arm64 ≠ x86_64).

    The same relaxation as webgpu_check.html: fp64 NaN is compared by class only; ``expected`` stays bitwise.
    """
    nan = lambda w: w & 0x7F800000 == 0x7F800000 and w & 0x7FFFFF
    return {**case, "fp64": "".join("7fc00000" if nan(w) else f"{w:08x}" for w in _words(case["fp64"]))}


@pytest.mark.parametrize("tag", ["grid", "nan", "cancel", "subnormal", "emax_mf0", "scale_tiny", "scale_huge",
                                 "scale_negative", "chunk_npot", "boundary", "shape"])
def test_golden_case_is_reproducible(golden, tag):
    """The first cheap rtz and rne cases of a tag, regenerated from their index, equal the stored ones."""
    plans = webgpu_golden.plan()
    for rounding in ("rtz", "rne"):
        index = next(i for i, item in enumerate(plans)
                     if item.tag == tag and item.norm_rounding == rounding and _cheap(i))
        assert _fp64_nan_class(webgpu_golden.build_case(index)) == _fp64_nan_class(golden["cases"][index])


def test_golden_grid_reproduces_wide_sums_in_both_c_modes(golden):
    plans = webgpu_golden.plan()
    for mode in ("fused", "decoupled"):
        index = next(i for i, item in enumerate(plans)
                     if item.tag == "grid" and item.c_mode == mode and item.f_bits > 23 and _cheap(i))
        assert webgpu_golden.build_case(index) == golden["cases"][index]


def _operand(codes: torch.Tensor, scale: float) -> Operand:
    values = codes.view(torch.float8_e4m3fn).float()
    return Operand(values, FP8_E4M3, torch.tensor(scale, dtype=torch.float32), FP32, "tensor")


def _read(path, dtype: torch.dtype, *shape: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).reshape(*shape)


def test_export_operands_with_an_injected_tiny_model(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    import tricast

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.randint(0, 64, (1, 7))
    kwargs = {"model_key": "tiny", "model_id": "tiny-llama", "label": "Tiny", "revision": None,
              "prompt": "(random ids)", "out_dir": tmp_path, "backends": ("reference",)}
    export_operands.export_model(model, input_ids, **kwargs)
    export_operands.export_model(model, input_ids, **kwargs)  # re-export replaces packs by id
    index = json.loads((tmp_path / "index.json").read_text())
    assert [pack["id"] for pack in index["packs"]] == ["tiny.l0.q_proj", "tiny.l0.down_proj"]
    hopper = ('[["algorithm","cofda"],["c_mode","fused"],["chunk_size",32],["f_bits",13],'
              '["norm_rounding","rtz"],["promote_interval",0]]')
    weights = {"tiny.l0.q_proj": model.model.layers[0].self_attn.q_proj.weight,
               "tiny.l0.down_proj": model.model.layers[0].mlp.down_proj.weight}
    for pack in index["packs"]:
        M, N, K = pack["M"], pack["N"], pack["K"]
        assert (M, N, K) == ((7, 32, 32) if pack["id"].endswith("q_proj") else (7, 32, 48))
        a = _read(tmp_path / pack["a_file"], torch.uint8, M, K)
        b = _read(tmp_path / pack["b_file"], torch.uint8, N, K)
        qb = tricast.quantize(weights[pack["id"]].detach().float(), "fp8_tensor")
        assert torch.equal(b.view(torch.float8_e4m3fn).float(), qb.values.float())
        assert pack["scale_b"] == qb.scale.item()
        meta = json.loads((tmp_path / pack["meta_file"]).read_text())
        assert meta["scale_a"] == f"{torch.tensor(pack['scale_a']).view(torch.int32).item() & 0xFFFFFFFF:08x}"
        assert meta["env"]["model_id"] == "tiny-llama" and meta["layer"] == pack["layer"]
        assert hopper in pack["server"] and len(pack["server"]) == 3
        for key, name in pack["server"].items():
            spec = MMASpec(**dict(json.loads(key)), out_format="fp32")
            expected = gemm(_operand(a, pack["scale_a"]), _operand(b, pack["scale_b"]), spec,
                            backend="reference")
            assert torch.equal(_read(tmp_path / name, torch.int32, M, N), expected.view(torch.int32))
