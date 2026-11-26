"""True-bitwidth ONNX export (tricast.onnx_export): 2/3/4/6/8-bit integers and floats, NF4, MXFP4, NVFP4,
BFP, RaZeR, input transforms (AWQ, Hadamard, Random Hadamard), Conv2d + grouped Conv2d, package-to-ONNX,
and com.microsoft::MatMulNBits kernel fusion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tricast import calibrate, load_recipe, patch_model
from tricast import pack as tp
from tricast.cli import main as cli_main
from tricast.nn.linear import EmuLinear

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("ml_dtypes")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from tricast import onnx_export as ox  # noqa: E402

IDS = [[1, 5, 9, 13, 17, 21, 25, 29]]


def _llama(tied: bool = False):
    torch.manual_seed(42)
    cfg = transformers.LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=tied,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def _tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i + 4 for i in range(124)})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=64,
    )


MIXED_RECIPE = {
    "name": "onnx_mixed_all",
    "defaults": {
        "weight": {"scheme": "int4_g128_zp", "format": "uint3", "group_size": 32},
        "activation": None,
        "weight_algo": "gptq",
        "mma": "fp32_fma",
    },
    "overrides": [
        {
            "match": "model.layers.0.self_attn.q_proj",
            "transform": "random_hadamard",
            "weight": {"scheme": "int4_g128_zp", "format": "uint2", "group_size": 32},
            "weight_algo": "gptq",
        },
        {
            "match": "model.layers.0.self_attn.k_proj",
            "transform": {"kind": "hadamard"},
            "weight": {"scheme": "mxfp4", "mma_input": "dequant"},
            "weight_algo": "rtn",
        },
        {
            "match": "model.layers.0.self_attn.v_proj",
            "weight": {"scheme": "nvfp4", "mma_input": "dequant"},
            "weight_algo": "rtn",
        },
        {
            "match": "model.layers.0.self_attn.o_proj",
            "transform": {"kind": "awq", "share_inputs": True, "grid": 5},
            "weight": {"scheme": "int4_g128_zp", "format": "uint4", "group_size": 32},
            "weight_algo": "awq_clip",
        },
        {
            "match": "model.layers.0.mlp.gate_proj",
            "weight": {"scheme": "mxfp6_e2m3", "mma_input": "dequant"},
            "weight_algo": "rtn",
        },
        {
            "match": "model.layers.0.mlp.up_proj",
            "weight": None,
            "weight_algo": "nf4",
        },
        {
            "match": "model.layers.0.mlp.down_proj",
            "weight": {"scheme": "int8_row", "mma_input": "dequant"},
            "weight_algo": "rtn",
        },
        {
            "match": "model.layers.1.self_attn.q_proj",
            "weight": {"scheme": "fp8_row", "mma_input": "dequant"},
            "weight_algo": "rtn",
        },
        {
            "match": "model.layers.1.self_attn.k_proj",
            "weight": {"scheme": "msfp12", "mma_input": "dequant"},
            "weight_algo": "rtn",
        },
        {
            "match": "model.layers.1.mlp.up_proj",
            "weight": None,
            "weight_algo": "razer",
        },
    ],
    "calibration": {"samples": 2, "seqlen": 8, "seed": 42, "sequential": False},
}


def test_mixed_bitwidth_llama_exports_at_true_bits_and_matches_tricast(tmp_path: Path) -> None:
    model = _llama()
    recipe = load_recipe(MIXED_RECIPE)
    patch_model(model, recipe)
    capture = tp.Capture(model)
    capture.requantize_uncalibrated()
    calibrate(model, recipe, input_ids=IDS, samples=2, seqlen=8, seed=42)
    capture.requantize_uncalibrated()
    capture.close()

    tokens = torch.tensor([[1, 6, 11, 15, 23, 31], [2, 7, 12, 0, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        before_logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits.clone()

    onnx_path = tmp_path / "mixed.onnx"
    manifest = ox.export_onnx(
        model,
        onnx_path,
        live_encoded=capture.encoded,
        recipe_text=json.dumps(MIXED_RECIPE),
        model_id="tiny/llama",
        revision="rev1",
    )
    # Caller's PyTorch model was not mutated
    assert sum(isinstance(m, EmuLinear) for m in model.modules()) == 14
    with torch.no_grad():
        after_logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
    assert torch.equal(before_logits, after_logits)

    assert manifest["quantized_layers"] == 14
    assert manifest["fallback_layers"] == {}
    assert manifest["checks"]["decoded_equals_tricast_weight"] == {
        "checked_layers": 14,
        "matmul_nbits_layers": 0,
        "matmul_nbits_not_bit_exact": [],
        "mismatched": [],
    }
    layers = manifest["layers"]
    assert layers["model.layers.0.self_attn.q_proj"]["encoding"] == "bitpack_int2"
    assert layers["model.layers.1.self_attn.o_proj"]["encoding"] == "bitpack_int3"
    assert layers["model.layers.0.self_attn.o_proj"]["encoding"] == "dequantize_linear_uint4"
    assert layers["model.layers.0.mlp.gate_proj"]["encoding"] == "bitpack_fp6_e2m3"
    assert layers["model.layers.0.mlp.up_proj"]["encoding"] == "bitpack_nf4"
    assert layers["model.layers.0.mlp.down_proj"]["encoding"] == "dequantize_linear_int8"
    assert layers["model.layers.1.self_attn.q_proj"]["encoding"] == "dequantize_linear_float8e4m3fn"
    assert layers["model.layers.1.mlp.up_proj"]["encoding"] == "bitpack_razer"

    # Verify raw initializer byte sizes inside the ONNX file reflect true bit-widths (not FP16/FP32!)
    proto = onnx.load(str(onnx_path))
    init_bytes = {init.name: len(init.raw_data) for init in proto.graph.initializer if init.raw_data}
    # q_proj is [64, 64] = 4096 params at 2 bits -> 1024 bytes!
    assert init_bytes["model_layers_0_self_attn_q_proj__codes"] == (64 * 64) * 2 // 8
    # gate_proj is [128, 64] = 8192 params at 6 bits -> 6144 bytes!
    assert init_bytes["model_layers_0_mlp_gate_proj__codes"] == (128 * 64) * 6 // 8
    # o_proj is [64, 64] = 4096 params at native UINT4 -> 2048 bytes!
    assert init_bytes["model_layers_0_self_attn_o_proj__q"] == (64 * 64) * 4 // 8
    # layers.1.o_proj is [64, 64] = 4096 params at 3 bits -> 1536 bytes!
    assert init_bytes["model_layers_1_self_attn_o_proj__codes"] == (64 * 64) * 3 // 8

    # Dynamic batch=2, seq=6 (different from default export [1, 4]) with right-padding mask
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = torch.from_numpy(
        sess.run(None, {"input_ids": tokens.numpy(), "attention_mask": mask.numpy()})[0]
    )
    cos0 = torch.nn.functional.cosine_similarity(ort_logits[0].flatten(), before_logits[0].flatten(), dim=0)
    cos1 = torch.nn.functional.cosine_similarity(
        ort_logits[1, :3].flatten(), before_logits[1, :3].flatten(), dim=0
    )
    assert cos0 > 0.99999 and (ort_logits[0] - before_logits[0]).abs().max() < 1e-3
    assert cos1 > 0.99999 and (ort_logits[1, :3] - before_logits[1, :3]).abs().max() < 1e-3


def test_package_to_onnx_and_matmul_nbits_kernel_modes(tmp_path: Path) -> None:
    model = _llama(tied=True)
    pkg_recipe = {
        "name": "pkg_to_onnx",
        "defaults": {
            "weight": {"scheme": "int4_g128_zp", "format": "uint4", "group_size": 32},
            "activation": None,
            "weight_algo": "gptq",
            "mma": "fp32_fma",
        },
        "overrides": [
            {
                "match": "model.layers.0.self_attn.q_proj",
                "weight": {"scheme": "int4_g128_zp", "format": "uint2", "group_size": 32},
            },
            {
                "match": "model.layers.0.self_attn.k_proj",
                "weight": {"scheme": "int4_g128_zp", "format": "uint3", "group_size": 32},
            },
            {"match": "model.layers.1.mlp.up_proj", "weight": None, "weight_algo": "nf4"},
        ],
        "calibration": {"samples": 2, "seqlen": 8, "seed": 42, "sequential": False},
    }
    recipe = load_recipe(pkg_recipe)
    patch_model(model, recipe)
    capture = tp.Capture(model)
    capture.requantize_uncalibrated()
    calibrate(model, recipe, input_ids=IDS, samples=2, seqlen=8, seed=42)
    capture.requantize_uncalibrated()
    capture.close()

    pkg_dir = tmp_path / "pkg"
    tp.write(model, capture, json.dumps(pkg_recipe), "tiny/llama", "rev", pkg_dir, None, _tokenizer())
    packed_model, _ = tp.load(pkg_dir, dtype=torch.float32)

    std_onnx = tmp_path / "std.onnx"
    nbits_onnx = tmp_path / "nbits.onnx"
    assert cli_main(["export-onnx", "--package", str(pkg_dir), "--out", str(std_onnx)]) == 0
    m_nbits = ox.export_package_onnx(pkg_dir, nbits_onnx, kernel="matmul_nbits")
    assert m_nbits["checks"]["decoded_equals_tricast_weight"]["mismatched"] == []
    # 12 of 14 layers are 2-bit or 4-bit group-32 affine -> MatMulNBits; 1 is 3-bit and 1 is NF4 -> standard!
    assert m_nbits["checks"]["decoded_equals_tricast_weight"]["matmul_nbits_layers"] == 12
    assert m_nbits["checks"]["decoded_equals_tricast_weight"]["checked_layers"] == 2

    tokens = torch.tensor([[1, 8, 12, 19, 27]], dtype=torch.long)
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        ref = packed_model(input_ids=tokens, attention_mask=mask, use_cache=False).logits

    def ort_logits(path: Path) -> torch.Tensor:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feeds = {"input_ids": tokens.numpy(), "attention_mask": mask.numpy()}
        return torch.from_numpy(sess.run(None, feeds)[0])

    # standard: every weight decodes bit for bit, the graph mirrors PackedLinear (fp32 GEMM, BF16 output)
    assert (ort_logits(std_onnx) - ref).abs().max() < 1e-5
    # MatMulNBits skips the recipe's BF16 rounding of the dequantized weight: the manifest says so per layer,
    # and the weight deviation is at most that rounding (half a BF16 ulp of the weight)
    nbits_layers = {
        n: i for n, i in m_nbits["layers"].items() if i["encoding"].startswith("ort_matmul_nbits")
    }
    assert (
        sorted(nbits_layers)
        == m_nbits["checks"]["decoded_equals_tricast_weight"]["matmul_nbits_not_bit_exact"]
    )
    assert all(0 < i["weight_max_abs_diff"] < 2**-8 for i in nbits_layers.values())
    got = ort_logits(nbits_onnx)
    assert torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0) > 0.9999

    gen = ox.run_onnx(nbits_onnx, tokenizer_source=pkg_dir, prompts=["t1 t2 t3"], max_new_tokens=4)
    assert len(gen["answers"]) == 1 and isinstance(gen["answers"][0]["text"], str)


def test_conv2d_and_grouped_conv2d_mixed_bitwidths_export_to_onnx(tmp_path: Path) -> None:
    class TinyVision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.dw = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=4)
            self.pw = nn.Conv2d(8, 16, kernel_size=1)
            self.head = nn.Linear(16, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.nn.functional.silu(self.stem(x))
            h = torch.nn.functional.silu(self.dw(h))
            h = torch.nn.functional.silu(self.pw(h))
            return self.head(h.mean(dim=(2, 3)))

    torch.manual_seed(11)
    net = TinyVision().eval()
    recipe = load_recipe(
        {
            "name": "vision_mixed",
            "defaults": {"weight": {"scheme": "int8_row", "mma_input": "dequant"}, "mma": "fp32_fma"},
            "overrides": [
                {
                    "match": "dw",
                    "weight": {"format": "int3", "granularity": "tensor", "mma_input": "dequant"},
                },
                {
                    "match": "pw",
                    "weight": {"format": "int4", "granularity": "row", "mma_input": "dequant"},
                },
                {
                    "match": "head",
                    "weight": {"format": "int2", "granularity": "row", "mma_input": "dequant"},
                },
            ],
            "include": ["stem", "dw", "pw", "head"],
        }
    )
    patch_model(net, recipe, include_conv2d=True)
    x = torch.randn(2, 3, 12, 12, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        ref = net(x)

    onnx_path = tmp_path / "vision.onnx"
    manifest = ox.export_onnx(
        net,
        onnx_path,
        sample_inputs=x,
        input_names=["images"],
        output_names=["scores"],
        dynamic_axes={"images": {0: "batch"}},
    )
    assert manifest["quantized_layers"] == 4
    assert manifest["checks"]["decoded_equals_tricast_weight"]["mismatched"] == []
    assert manifest["layers"]["dw"]["groups"] == 4 and manifest["layers"]["dw"]["bits"] == 3
    assert manifest["layers"]["pw"]["encoding"] == "dequantize_linear_int4"
    assert manifest["layers"]["head"]["encoding"] == "bitpack_int2"

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = torch.from_numpy(sess.run(None, {"images": x.numpy()})[0])
    assert (got - ref).abs().max() < 1e-4


def test_external_data_split_and_unsupported_recipes_are_refused(tmp_path: Path) -> None:
    torch.manual_seed(3)
    mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64)).eval()
    patch_model(
        mlp,
        load_recipe(
            {
                "name": "mlp4",
                "defaults": {"weight": {"scheme": "int4_g128", "group_size": 16}, "mma": "fp32_fma"},
                "include": ["0", "2"],
            }
        ),
    )
    x = torch.randn(3, 128)
    ext_onnx = tmp_path / "ext" / "mlp.onnx"
    manifest = ox.export_onnx(mlp, ext_onnx, sample_inputs=x, external_data_threshold=256)
    assert "mlp.onnx.data" in manifest["files"]
    sess = ort.InferenceSession(str(ext_onnx), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        assert (torch.from_numpy(sess.run(None, {"input_0": x.numpy()})[0]) - mlp(x)).abs().max() < 1e-5

    # Unsupported features are refused before writing any ONNX file
    for bad_defaults, match in [
        ({"weight": {"scheme": "int8_row"}, "activation": {"scheme": "int8_row"}}, "activation quantization"),
        ({"weight": {"scheme": "int8_row"}, "sparsity": {"kind": "n:m", "n": 2, "m": 4}}, "sparsity"),
        (
            {"weight": {"scheme": "int8_row"}, "mma": {"algorithm": "fp32_fma", "out_format": "fp8_e4m3"}},
            "output format fp8_e4m3",
        ),
    ]:
        bad_net = nn.Sequential(nn.Linear(16, 16)).eval()
        patch_model(
            bad_net,
            load_recipe({"name": "bad", "defaults": {"activation": None, **bad_defaults}, "include": ["0"]}),
        )
        bad_out = tmp_path / "bad.onnx"
        with pytest.raises(ValueError, match=match):
            ox.export_onnx(bad_net, bad_out, sample_inputs=torch.randn(1, 16))
        assert not bad_out.exists()
