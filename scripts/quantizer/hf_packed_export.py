"""Export a single-shard, tied Qwen3 or Llama-3.2-1B checkpoint for stock W4A16 compressed-tensors loaders.

Input is either the original dense HF checkpoint or the expressly authorized approximation
from gguf_export.py dense-hf. Neither route is a lossless transcoding of a mixed recipe.
Linear projections are quantized offline; no custom model code is required at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP = 128
WEIGHTS = "model.safetensors"
SIDECARS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def expected_shapes(config: dict[str, Any]) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Allowlisted source and its complete parameter dimensions, including GQA and tied embeddings."""
    model = config.get("model_type")
    if (
        model not in ("qwen3", "llama")
        or config.get("architectures") != [{"qwen3": "Qwen3ForCausalLM", "llama": "LlamaForCausalLM"}[model]]
        or config.get("tie_word_embeddings") is not True
        or config.get("attention_bias", False)
        or config.get("mlp_bias", False)
        or config.get("quantization_config") is not None
    ):
        raise ValueError(
            "only original dense, tied Qwen3 or Llama-3.2-1B causal LMs without bias are supported"
        )
    h, m, v = config["hidden_size"], config["intermediate_size"], config["vocab_size"]
    count, nh, nk = config["num_hidden_layers"], config["num_attention_heads"], config["num_key_value_heads"]
    if any(type(x) is not int or x <= 0 for x in (h, m, v, count, nh, nk)) or h % nh or nh % nk:
        raise ValueError("invalid model dimensions")
    dim = config.get("head_dim") or h // nh
    norms = {"input_layernorm": h, "post_attention_layernorm": h}
    if model == "qwen3":
        norms.update({"self_attn.q_norm": dim, "self_attn.k_norm": dim})
    elif (h, m, v, count, nh, nk, dim) != (2048, 8192, 128256, 16, 32, 8, 64) or (
        config.get("rope_scaling") or {}
    ).get("rope_type") != "llama3":
        raise ValueError("only the Llama-3.2-1B architecture is supported")
    shapes = dict(
        zip(
            PROJECTIONS,
            ((nh * dim, h), (nk * dim, h), (nk * dim, h), (h, nh * dim), (m, h), (m, h), (h, m)),
            strict=True,
        )
    )
    projections = {f"model.layers.{i}.{p}.weight": s for i in range(count) for p, s in shapes.items()}
    dense = {"model.embed_tokens.weight": (v, h), "model.norm.weight": (h,)}
    dense.update({f"model.layers.{i}.{p}.weight": (size,) for i in range(count) for p, size in norms.items()})
    return projections, dense


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tied_head(head: torch.Tensor, embedding: torch.Tensor) -> None:
    """Only elide an explicitly serialized alias when every original storage bit agrees."""
    if head.dtype != embedding.dtype or head.shape != embedding.shape:
        raise ValueError("tied lm_head and embedding have different layout")
    a = head.reshape(-1).view(torch.uint8)
    b = embedding.reshape(-1).view(torch.uint8)
    for x, y in zip(a.split(1 << 22), b.split(1 << 22), strict=True):
        if not torch.equal(x, y):
            raise ValueError("tied lm_head differs bitwise from embedding; refusing to drop it")


def w4a16(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RTN using the stored BF16 scale, signed [-7,7], low nibble first with bias 8."""
    if weight.ndim != 2 or weight.shape[1] % GROUP or not weight.is_floating_point():
        raise ValueError(f"invalid projection {weight.dtype} {weight.shape} for group {GROUP}")
    rows, _ = weight.shape
    w = weight.float().reshape(rows, -1, GROUP)
    if not torch.isfinite(w).all():
        raise ValueError("nonfinite weight")
    peak = w.abs().amax(-1)
    scales = torch.where(peak == 0, 1.0, peak / 7).to(torch.bfloat16)
    if not (torch.isfinite(scales) & (scales > 0)).all():
        raise ValueError("scale does not fit BF16")
    signed = torch.round(w / scales.float().unsqueeze(-1)).clamp(-7, 7)
    unsigned = signed.reshape(rows, -1, 8).to(torch.int32) + 8
    packed = torch.zeros_like(unsigned[:, :, 0], memory_format=torch.contiguous_format)
    for i in range(8):
        packed.bitwise_or_(unsigned[:, :, i] << (i * 4))
    # Check every code, including sign bits in the most significant nibble.
    for i in range(8):
        if not torch.equal((packed >> (i * 4)) & 15, unsigned[:, :, i]):
            raise ValueError("signed W4 offset/packing round-trip failed")
    return packed, scales.contiguous()


def quant_config() -> dict[str, Any]:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": GROUP,
                    "symmetric": True,
                    "dynamic": False,
                },
            }
        },
    }


@torch.inference_mode()
def export(src: Path, out: Path, *, decoded: bool, allow_requantization: bool) -> dict[str, Any]:
    if decoded and not allow_requantization:
        raise ValueError("decoded TriCast weights require --allow-requantization; original codes will change")
    if not decoded and allow_requantization:
        raise ValueError("do not use --allow-requantization with original dense HF")
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    report_file = src / "tricast_dense_manifest.json"
    if decoded != report_file.is_file():
        raise ValueError("source kind does not match TriCast dense manifest presence")
    if (src / "quant_config.json").exists() or (src / "model.safetensors.index.json").exists():
        raise ValueError("use a single-shard, dense HF checkpoint; no TriCast packs or sharded weights")
    config = json.loads((src / "config.json").read_text())
    linear, dense = expected_shapes(config)
    if decoded and config["model_type"] != "qwen3":
        raise ValueError("TriCast decoded bridge is only supported for Qwen3")
    if not (src / "tokenizer.json").is_file() or not (src / "tokenizer_config.json").is_file():
        raise ValueError("missing tokenizer files")
    source_hashes = {k: sha256(src / k) for k in (*SIDECARS, WEIGHTS) if (src / k).is_file()}
    lineage: dict[str, Any] = {"kind": "dense_hf", "file_sha256": source_hashes}
    if decoded:
        report = json.loads(report_file.read_text())
        if report.get("schema") != "tricast.dense_hf/1":
            raise ValueError("unrecognized dense source manifest")
        declared = {k: v["sha256"] for k, v in report["files"].items()}
        if declared != source_hashes:
            raise ValueError("decoded checkpoint does not match its manifest")
        lineage.update(
            kind="requantized_tricast_dense_approximation",
            dense_manifest_sha256=sha256(report_file),
            source_revision=report["revision"],
            recipe_sha256=report["recipe_sha256"],
            source_pack_manifest_sha256=report["source_pack_manifest_sha256"],
        )
    data: dict[str, torch.Tensor] = {}
    with safe_open(src / WEIGHTS, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        has_explicit_head = "lm_head.weight" in keys
        if has_explicit_head:
            verify_tied_head(f.get_tensor("lm_head.weight"), f.get_tensor("model.embed_tokens.weight"))
            keys.remove("lm_head.weight")
        expected = linear | dense
        if keys != expected.keys():
            raise ValueError(f"unexpected/missing parameters: {sorted(keys ^ expected.keys())}")
        for key in sorted(keys):
            w = f.get_tensor(key)
            if tuple(w.shape) != expected[key]:
                raise ValueError(f"{key}: expected shape {expected[key]}, got {tuple(w.shape)}")
            if key in linear:
                prefix = key.removesuffix("weight")
                packed, scale = w4a16(w)
                data[prefix + "weight_packed"] = packed
                data[prefix + "weight_scale"] = scale
                data[prefix + "weight_shape"] = torch.tensor(w.shape, dtype=torch.int64)
            else:
                if not w.is_floating_point():
                    raise ValueError(f"non-floating source tensor: {key}")
                value = w.to(torch.bfloat16).contiguous()
                if not all(torch.isfinite(t).all() for t in value.flatten().split(1 << 20)):
                    raise ValueError(f"nonfinite BF16 parameter: {key}")
                data[key] = value
    config["torch_dtype"] = "bfloat16"
    if "dtype" in config:
        config["dtype"] = "bfloat16"
    config["quantization_config"] = quant_config()
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hf-w4a16-", dir=out.parent) as tmp:
        stage = Path(tmp) / "checkpoint"
        stage.mkdir()
        save_file(data, str(stage / WEIGHTS), metadata={"format": "pt"})
        for name in SIDECARS:
            if name != "config.json" and name in source_hashes:
                shutil.copyfile(src / name, stage / name)
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        result = {
            "schema": "tricast.hf_w4a16/1",
            "source": lineage,
            "scheme": "symmetric RTN W4A16",
            "source_codes_preserved": False,
            "group_size": GROUP,
            "scale_dtype": "bfloat16",
            "linear_layers": len(linear),
            "tied_embeddings_and_norms": "bfloat16",
            "explicit_tied_head_verified": has_explicit_head,
            "packed_weight_bytes": sum(v.nbytes for k, v in data.items() if k.endswith(".weight_packed")),
            "weights_bytes": (stage / WEIGHTS).stat().st_size,
            "weights_sha256": sha256(stage / WEIGHTS),
            "runtime_checked": False,
        }
        (stage / "tricast_w4a16_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        stage.rename(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dense-hf", type=Path, help="TriCast decoded HF; requires --allow-requantization")
    source.add_argument(
        "--original-hf", type=Path, help="original single-shard dense HF Qwen3 or Llama-3.2-1B checkpoint"
    )
    parser.add_argument("--allow-requantization", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = export(
        args.dense_hf or args.original_hf,
        args.out,
        decoded=args.dense_hf is not None,
        allow_requantization=args.allow_requantization,
    )
    print(f"HF_W4A16_EXPORT_OK layers={result['linear_layers']} weights_bytes={result['weights_bytes']}")


if __name__ == "__main__":
    main()
