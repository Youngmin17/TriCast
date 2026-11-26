"""True bit-width packing and decoding: narrow int, fixed-point, float, pow2-scale and RaZeR formats."""

from __future__ import annotations

import json

import pytest
import torch

from tricast import bitpack as bp
from tricast import load_recipe, patch_model
from tricast import pack as tp
from tricast.formats import E8M0, FloatFormat, Pow2Format, get_format
from tricast.nn.patch import decoder_layers
from tricast.reference.cast import round_to_format

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_pack_and_unpack_bits_for_every_subbyte_width(bits: int) -> None:
    count = 259
    codes = torch.randint(
        0, 1 << bits, (count,), generator=torch.Generator().manual_seed(bits), dtype=torch.int64
    )
    packed = bp.pack_bits(codes, bits)
    expected_bytes = count if bits == 8 else -(-count // 8) * bits
    assert packed.dtype == torch.uint8 and packed.numel() == expected_bytes
    restored = bp.unpack_bits(packed, bits, count).to(torch.int64)
    assert torch.equal(restored, codes)


@pytest.mark.parametrize(
    "fmt_name",
    [
        "fp4_e2m1",
        "fp6_e2m3",
        "fp6_e3m2",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e4m3fnuz",
        "fp8_e5m2fnuz",
        "ue4m3",
        "e3m2:nosub",
    ],
)
def test_narrow_float_codebook_round_trips_every_grid_value(fmt_name: str) -> None:
    fmt = get_format(fmt_name)
    assert isinstance(fmt, FloatFormat)
    raw = torch.linspace(-500.0, 500.0, 4097, dtype=torch.float32)
    if not fmt.signed:
        raw = raw.abs()
    on_grid = round_to_format(raw, fmt)
    codes = bp.encode_float_codes(on_grid, fmt)
    packed = bp.pack_bits(codes, fmt.bits)
    expected_bytes = on_grid.numel() if fmt.bits == 8 else -(-on_grid.numel() // 8) * fmt.bits
    assert packed.numel() == expected_bytes
    decoded = bp.float_codebook(fmt)[bp.unpack_bits(packed, fmt.bits, on_grid.numel()).long()]
    assert bp.same_bits(decoded, on_grid)
    with pytest.raises(ValueError, match="not on the"):
        bp.encode_float_codes(torch.tensor([0.1234567], dtype=torch.float32), fmt)


def test_pow2_scale_codes_pack_to_one_byte_and_round_trip() -> None:
    assert isinstance(E8M0, Pow2Format)
    exps = torch.tensor([-127, -14, -1, 0, 1, 15, 127], dtype=torch.int64)
    vals = (2.0 ** exps.double()).float()
    vals = torch.cat([vals, torch.tensor([float("nan")], dtype=torch.float32)])
    packed, meta = bp.encode_scale_tensor(vals, E8M0)
    assert packed.dtype == torch.uint8 and packed.numel() == vals.numel()
    assert meta["scale_dtype"] == "packed_pow2" and meta["scale_bits"] == 8
    decoded = bp.decode_scale_tensor(packed, meta)
    assert torch.equal(decoded[:-1], vals[:-1]) and bool(torch.isnan(decoded[-1]))


def test_multi_format_llama_packs_mxfp_nvfp_bfp_fp8_and_razer_without_dense_fallback(tmp_path) -> None:
    torch.manual_seed(7)
    cfg = transformers.LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=32,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    recipe_dict = {
        "name": "all_formats_pack",
        "defaults": {
            "weight": {"scheme": "mxfp4", "mma_input": "dequant"},
            "activation": None,
            "mma": "fp32_fma",
        },
        "overrides": [
            {
                "match": "model.layers.0.self_attn.q_proj",
                "weight": {"scheme": "nvfp4", "mma_input": "dequant"},
            },
            {
                "match": "model.layers.0.self_attn.k_proj",
                "weight": {"scheme": "nvfp4", "format": "int4", "mma_input": "dequant"},
            },
            {
                "match": "model.layers.0.self_attn.v_proj",
                "weight": {"scheme": "mxfp6_e2m3", "mma_input": "dequant"},
            },
            {
                "match": "model.layers.0.self_attn.o_proj",
                "weight": {"scheme": "mxfp8_e4m3", "mma_input": "dequant"},
            },
            {
                "match": "model.layers.0.mlp.gate_proj",
                "weight": {"scheme": "msfp12", "mma_input": "dequant"},
            },
            {
                "match": "model.layers.0.mlp.up_proj",
                "weight": {"scheme": "mxint8", "mma_input": "dequant"},
            },
            {
                "match": "model.layers.0.mlp.down_proj",
                "weight": {"scheme": "fp8_block128", "block": [32, 32], "mma_input": "dequant"},
            },
            {
                "match": "model.layers.1.self_attn.q_proj",
                "weight": {"scheme": "int8_tensor", "mma_input": "dequant"},
            },
            {
                "match": "model.layers.1.mlp.up_proj",
                "weight": None,
                "weight_algo": "razer",
            },
        ],
    }
    recipe = load_recipe(recipe_dict)
    patch_model(model, recipe)
    capture = tp.Capture(model)
    capture.requantize_uncalibrated()
    assert not capture.fallback, f"unexpected dense fallback: {capture.fallback}"

    decoder = decoder_layers(model)
    for name, layer in capture.layers.items():
        enc = capture.encoded[name]
        assert enc.kind != "dense", name
        entry = {"kind": enc.kind, "shape": list(enc.shape), **enc.meta}
        decoded = tp.decode(entry, enc.tensors, recipe.spec_for(name, decoder=decoder))
        assert tp.same_bits(decoded, tp.effective_weight(layer)), name

    # Check exact code byte sizes match bit-widths (4b -> N/2, 6b -> 3N/4, 8b -> N)
    assert capture.encoded["model.layers.0.self_attn.q_proj"].tensors["codes"].numel() == (64 * 64) * 4 // 8
    assert capture.encoded["model.layers.0.self_attn.v_proj"].tensors["codes"].numel() == (32 * 64) * 6 // 8
    assert capture.encoded["model.layers.0.self_attn.o_proj"].tensors["codes"].numel() == (64 * 64) * 8 // 8
    assert capture.encoded["model.layers.1.mlp.up_proj"].kind == "razer"
    assert capture.encoded["model.layers.1.mlp.up_proj"].tensors["codes"].numel() == (128 * 64) * 4 // 8

    tokens = torch.tensor([[1, 4, 7, 11, 19]], dtype=torch.long)
    with torch.no_grad():
        ref_logits = model(tokens).logits
    capture.close()

    manifest = tp.write(model, capture, json.dumps(recipe_dict), "tiny/llama", "rev", tmp_path)
    assert manifest["fallback_layers"] == {}
    packed_load, _ = tp.load(tmp_path, dtype=torch.float32, mode="load")
    packed_codes, _ = tp.load(tmp_path, dtype=torch.float32, mode="codes")
    with torch.no_grad():
        out_load = packed_load(tokens).logits
        out_codes = packed_codes(tokens).logits
    assert torch.equal(out_load, out_codes)
    assert torch.nn.functional.cosine_similarity(out_load.flatten(), ref_logits.flatten(), dim=0) > 0.99999
