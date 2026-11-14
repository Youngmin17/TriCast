"""Low-bit model packages (tricast.pack): codes bit-packed and read back, every decoded layer equal to the
weight TriCast's emulation multiplies with, and a written package that loads as a model giving TriCast's
outputs. CPU, a two-block Llama."""

from __future__ import annotations

import json

import pytest
import torch

from tricast import calibrate, load_recipe, patch_model
from tricast import pack as tp
from tricast.nn.patch import decoder_layers

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


def _model(tied: bool = False):
    torch.manual_seed(42)
    config = transformers.LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=tied,
    )
    return transformers.LlamaForCausalLM(config).eval()


def _tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i + 4 for i in range(252)})
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


# one recipe with each stored kind: GPTQ uint3/uint4 with group scales and zero points (the default), a
# random-Hadamard GPTQ layer, an int8 per-row layer and an NF4 layer
RECIPE = {
    "name": "pack_test",
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
            "weight": {"scheme": "int4_g128_zp", "format": "uint4", "group_size": 32},
        },
        {"match": "model.layers.0.mlp.down_proj", "weight": {"scheme": "int8_row"}, "weight_algo": "rtn"},
        {"match": "model.layers.1.mlp.up_proj", "weight": None, "weight_algo": "nf4"},
    ],
    "calibration": {"samples": 2, "seqlen": 8, "seed": 42, "sequential": False},
}
IDS = [[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]]


def _quantized(tied: bool = False):
    model = _model(tied)
    recipe = load_recipe(RECIPE)
    patch_model(model, recipe)
    capture = tp.Capture(model)
    capture.requantize_uncalibrated()
    calibrate(model, recipe, input_ids=IDS, samples=2, seqlen=8, seed=42)
    capture.requantize_uncalibrated()
    return model, recipe, capture


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_codes_pack_into_bits_and_come_back(bits):
    codes = torch.randint(
        0, 1 << bits, (1001,), generator=torch.Generator().manual_seed(bits), dtype=torch.int64
    )
    data = tp.pack_bits(codes, bits)
    assert data.dtype == torch.uint8 and data.numel() == (-(-1001 // 8) * bits if bits < 8 else 1001)
    assert torch.equal(tp.unpack_bits(data, bits, 1001).to(torch.int64), codes)
    with pytest.raises(ValueError, match="do not fit"):
        tp.pack_bits(torch.tensor([1 << bits]), bits)


def test_every_layer_decodes_to_the_weight_tricast_multiplies_with():
    model, recipe, capture = _quantized()
    decoder = decoder_layers(model)
    kinds = {}
    for name, layer in capture.layers.items():
        encoded = capture.encoded[name]
        entry = {"kind": encoded.kind, "shape": list(encoded.shape), **encoded.meta}
        weight = tp.decode(entry, encoded.tensors, recipe.spec_for(name, decoder=decoder))
        assert tp.same_bits(weight, tp.effective_weight(layer)), name
        kinds[name] = (encoded.kind, encoded.meta.get("bits"))
    assert kinds["model.layers.0.mlp.down_proj"] == ("affine", 8)
    assert kinds["model.layers.1.mlp.up_proj"] == ("nf4", 4)
    assert kinds["model.layers.0.self_attn.q_proj"] == ("affine", 4)
    assert kinds["model.layers.1.self_attn.o_proj"] == ("affine", 3)
    assert not capture.fallback


@pytest.mark.parametrize("tied", [False, True])
def test_a_written_package_loads_and_gives_tricasts_outputs(tmp_path, tied):
    model, _, capture = _quantized(tied)
    tokens = torch.tensor(IDS)
    with torch.no_grad():
        reference = model(tokens).logits
    capture.close()
    manifest = tp.write(
        model, capture, json.dumps(RECIPE), "tiny/llama", "rev", tmp_path, {"note": "test"}, _tokenizer()
    )
    config = json.loads((tmp_path / tp.CONFIG).read_text())
    assert set(config["layers"]) == set(capture.layers) and manifest["quantized_layers"] == 14
    assert manifest["files"][tp.WEIGHTS]["bytes"] == (tmp_path / tp.WEIGHTS).stat().st_size
    assert (config["tied"] == {"lm_head.weight": "model.embed_tokens.weight"}) == tied
    packed, tokenizer = tp.load(tmp_path, dtype=torch.float32)
    assert tokenizer is not None and tokenizer("t1 t2")["input_ids"]
    assert sum(isinstance(m, tp.PackedLinear) for m in packed.modules()) == 14
    with torch.no_grad():
        logits = packed(tokens).logits
    # the same weights and output rounding; only the order of the fp32 sums differs (TriCast's FMA chain in K
    # order vs PyTorch's matmul), which can move a BF16 output by one unit
    cos = torch.nn.functional.cosine_similarity(logits.flatten(), reference.flatten(), dim=0)
    assert cos > 0.99999 and (logits - reference).abs().max() < 1e-3
    codes_mode, _ = tp.load(tmp_path, dtype=torch.float32, mode="codes")
    with torch.no_grad():
        assert torch.equal(codes_mode(tokens).logits, logits)


def test_the_package_is_smaller_than_its_bf16_weights(tmp_path):
    model, _, capture = _quantized()
    capture.close()
    tp.write(model, capture, json.dumps(RECIPE), "tiny/llama", "rev", tmp_path)
    config = json.loads((tmp_path / tp.CONFIG).read_text())
    from safetensors import safe_open

    with safe_open(str(tmp_path / tp.WEIGHTS), "pt") as handle:
        stored = {key: handle.get_tensor(key) for key in handle.keys()}
    quantized_bytes = sum(
        t.numel() * t.element_size() for k, t in stored.items() if k.rsplit(".", 1)[0] in config["layers"]
    )
    params = sum(e["shape"][0] * e["shape"][1] for e in config["layers"].values())
    assert quantized_bytes < params * 2 / 3  # well under BF16 (2 bytes a weight): ~3–4 bits plus scales


def test_sparsity_is_refused():
    model = _model()
    recipe = load_recipe(
        {
            "name": "sparse",
            "defaults": {
                "weight": {"scheme": "int8_row"},
                "activation": None,
                "sparsity": {"kind": "n:m", "n": 2, "m": 4},
            },
        }
    )
    patch_model(model, recipe)
    capture = tp.Capture(model)
    with pytest.raises(ValueError, match="sparsity"):
        capture.requantize_uncalibrated()


def test_a_weight_rounded_to_negative_zero_decodes_to_positive_zero_and_nothing_else_moves():
    from types import SimpleNamespace

    from tricast.mma.api import as_operand
    from tricast.nn.linear import _operand_value
    from tricast.quant.api import quantize
    from tricast.quant.spec import QuantSpec, ScaleSpec

    spec = QuantSpec(format="int4", granularity="row", scale=ScaleSpec(format="fp32"), rounding="rne")
    weight = torch.tensor([[-0.01, 1.0, -0.5, 0.004], [0.3, -0.02, -1.0, 0.0]])
    qt = quantize(weight, spec, backend="reference")
    encoded = tp.encode_qtensor(qt)
    decoded = tp.decode(
        {"kind": encoded.kind, "shape": list(encoded.shape), **encoded.meta},
        encoded.tensors,
        SimpleNamespace(weight=spec),
    )
    tricast = _operand_value(as_operand(qt, compact=True))
    moved = decoded.view(torch.int32) != tricast.view(torch.int32)
    assert moved.any()  # TriCast keeps -0.0 for a small negative weight; integer codes cannot
    assert bool(((decoded == 0) & (tricast == 0))[moved].all())  # only zeros differ, and only in sign
    assert tp.same_bits(decoded, tricast) and not tp.same_bits(decoded, tricast + 1e-7)
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(42))
    assert torch.equal(x @ decoded.T, x @ tricast.T)  # a sum that starts from +0.0 does not see the sign


@pytest.mark.parametrize(
    ("defaults", "message"),
    [
        ({"weight": {"scheme": "int8_row"}, "activation": {"scheme": "int8_row"}}, "activation quantization"),
        (
            {"weight": {"scheme": "int8_row"}, "mma": {"algorithm": "fp32_fma", "out_format": "fp8_e4m3"}},
            "output format fp8_e4m3",
        ),
    ],
)
def test_what_a_package_cannot_run_is_refused(defaults, message):
    model = _model()
    patch_model(model, load_recipe({"name": "refused", "defaults": {"activation": None, **defaults}}))
    capture = tp.Capture(model)
    with pytest.raises(ValueError, match=message):
        capture.requantize_uncalibrated()


def test_a_stored_recipe_with_a_kv_spec_does_not_load(tmp_path):
    model, _, capture = _quantized()
    capture.close()
    tp.write(model, capture, json.dumps(RECIPE), "tiny/llama", "rev", tmp_path, None, _tokenizer())
    config = json.loads((tmp_path / tp.CONFIG).read_text())
    config["recipe"]["yaml"] = json.dumps({**RECIPE, "kv": {"preset": "kivi2", "mode": "fakequant"}})
    (tmp_path / tp.CONFIG).write_text(json.dumps(config))
    with pytest.raises(ValueError, match="KV-cache"):
        tp.load(tmp_path, dtype=torch.float32)


def test_a_new_package_does_not_inherit_old_checks(tmp_path):
    model, _, capture = _quantized()
    capture.close()
    tp.write(model, capture, json.dumps(RECIPE), "tiny/llama", "rev", tmp_path, {"ppl_package": 1.0})
    manifest = tp.write(model, capture, json.dumps(RECIPE), "tiny/llama", "rev", tmp_path, {"note": "again"})
    assert manifest["checks"] == {"note": "again"}


def test_a_dense_weight_is_kept_exactly_and_a_bias_survives_a_dtype_change():
    weight = torch.tensor([[1.001, 2.003], [0.5, -0.25]])
    encoded = tp.encode_dense(weight)
    assert encoded.meta["bits"] == 32 and torch.equal(encoded.tensors["weight"], weight)
    assert tp.encode_dense(weight.to(torch.bfloat16).float()).meta["bits"] == 16
    entry = {"kind": "dense", "shape": [2, 2], **encoded.meta}
    layer = tp.PackedLinear("l", entry, dict(encoded.tensors), None, torch.tensor([0.5, 1.0]))
    layer = layer.to(torch.bfloat16)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    expected = (x.float() @ weight.T + torch.tensor([0.5, 1.0])).to(torch.bfloat16)
    assert torch.equal(layer(x), expected)
