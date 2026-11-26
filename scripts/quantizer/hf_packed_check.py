"""Offline W4A16 checkpoint audit against dense weights, using the official CT unpacker.

The temporary decompressions here are verification on CPU, not an inference path.
The source can be the --dense-hf or --original-hf input used by hf_packed_export.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from hf_packed_export import SIDECARS, WEIGHTS, expected_shapes, sha256, verify_tied_head
from safetensors import safe_open


def check_codes(w: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    out_features, in_features = w.shape
    if (
        packed.dtype != torch.int32
        or packed.shape != (out_features, in_features // 8)
        or scales.dtype != torch.bfloat16
        or scales.shape != (out_features, in_features // 128)
    ):
        raise ValueError("bad W4 tensor layout")
    groups = w.float().unfold(1, 128, 128)
    maxabs = groups.abs().amax(dim=-1)
    expected_scale = (maxabs / 7).to(torch.bfloat16).masked_fill(maxabs == 0, 1)
    if not torch.equal(scales, expected_scale):
        raise ValueError("BF16 scales differ")
    expected = (w.float() / scales.float().repeat_interleave(128, dim=1)).round().clamp(-7, 7).to(torch.int8)
    codes = unpack_from_int32(packed, 4, w.shape)
    if not torch.equal(codes, expected):
        raise ValueError("official unpacked codes differ from reference RTN")
    return codes


def measurement(
    w: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, codes: torch.Tensor, artifact_sha: str
) -> dict[str, Any]:
    from app.packed_targets import CT_FORMAT

    dense = w.double()
    diff = codes.double().reshape(*scale.shape, 128) * scale.double().unsqueeze(-1) - dense.reshape(
        *scale.shape, 128
    )
    denominator = dense.square().sum().item()
    if not denominator > 0:
        raise ValueError("relative weight SSE needs a nonzero source tensor")
    score = diff.square().sum().item() / denominator
    if not math.isfinite(score):
        raise ValueError("nonfinite weight SSE")
    # Digest layout: packed words (native little-endian), then the raw BF16 scale bytes.
    h = hashlib.sha256()
    for buffer in (packed, scale):
        h.update(buffer.contiguous().view(torch.uint8).numpy())
    return {
        "shape_nk": list(w.shape),
        "measurements": {
            CT_FORMAT: {
                "relative_sse": score,
                "artifact_sha256": artifact_sha,
                "payload_sha256": h.hexdigest(),
                "payload_bytes": packed.nbytes + scale.nbytes,
            }
        },
    }


def check_keyset(keys: set[str], expected: set[str]) -> None:
    if keys != expected:
        raise ValueError(f"missing/unexpected keys: {sorted(keys ^ expected)}")


@torch.inference_mode()
def audit(src: Path, ckpt: Path, table_out: Path | None = None) -> dict[str, Any]:
    config = json.loads((ckpt / "config.json").read_text())
    manifest = json.loads((ckpt / "tricast_w4a16_manifest.json").read_text())
    q = config["quantization_config"]
    weights = q["config_groups"]["group_0"]["weights"]
    if (
        q["quant_method"],
        q["format"],
        q["quantization_status"],
        q["ignore"],
        weights["type"],
        weights["num_bits"],
        weights["symmetric"],
        weights["strategy"],
        weights["group_size"],
    ) != ("compressed-tensors", "pack-quantized", "compressed", ["lm_head"], "int", 4, True, "group", 128):
        raise ValueError("configuration is not the supported compressed-tensors scheme")
    scheme = q["config_groups"]["group_0"]
    if (
        set(q["config_groups"]) != {"group_0"}
        or scheme["targets"] != ["Linear"]
        or scheme["input_activations"] is not None
        or scheme["output_activations"] is not None
        or weights["dynamic"] is not False
    ):
        raise ValueError("unexpected target, activation or dynamic quantization")
    base = json.loads((src / "config.json").read_text())
    projection_shapes, dense_shapes = expected_shapes(base)
    if table_out is not None and base["model_type"] != "qwen3":
        raise ValueError("solver table is only supported for Qwen3")
    expected_config = dict(base, torch_dtype="bfloat16", quantization_config=q)
    if "dtype" in base:
        expected_config["dtype"] = "bfloat16"
    if config != expected_config:
        raise ValueError("target model architecture changed")
    provenance = manifest["source"]["file_sha256"]
    for name in SIDECARS:
        if (src / name).is_file() != (ckpt / name).is_file():
            raise ValueError(f"missing/unexpected sidecar: {name}")
        if (src / name).is_file():
            digest = sha256(src / name)
            if digest != provenance[name] or (name != "config.json" and digest != sha256(ckpt / name)):
                raise ValueError(f"sidecar mismatch: {name}")
    if provenance.get(WEIGHTS) != sha256(src / WEIGHTS) or manifest["weights_sha256"] != sha256(
        ckpt / WEIGHTS
    ):
        raise ValueError("source or packed weights changed")
    expected_bytes = 0
    controls: list[str] = []
    rows: dict[str, Any] = {}
    with safe_open(src / WEIGHTS, framework="pt") as source, safe_open(ckpt / WEIGHTS, framework="pt") as ct:
        linears = set(projection_shapes)
        source_keys = set(source.keys()) - {"lm_head.weight"}
        check_keyset(source_keys, linears | dense_shapes.keys())
        for name, shape in (projection_shapes | dense_shapes).items():
            if tuple(source.get_slice(name).get_shape()) != shape:
                raise ValueError(f"{name}: invalid source shape")
        others = set(source.keys()) - linears
        if "lm_head.weight" in others:
            if config.get("tie_word_embeddings") is not True:
                raise ValueError("cannot omit untied lm_head")
            verify_tied_head(
                source.get_tensor("lm_head.weight"), source.get_tensor("model.embed_tokens.weight")
            )
            others.remove("lm_head.weight")
        expected = others | {
            k.removesuffix("weight") + suffix
            for k in linears
            for suffix in ("weight_scale", "weight_shape", "weight_packed")
        }
        check_keyset(set(ct.keys()), expected)
        for name in sorted(linears):
            prefix = name.removesuffix("weight")
            w = source.get_tensor(name)
            shape = ct.get_tensor(prefix + "weight_shape")
            packed, scale = ct.get_tensor(prefix + "weight_packed"), ct.get_tensor(prefix + "weight_scale")
            if shape.dtype != torch.int64 or shape.tolist() != list(w.shape) or w.shape[1] % 128:
                raise ValueError(f"{name}: unexpected shape")
            codes = check_codes(w, packed, scale)
            if table_out is not None:
                rows[name] = measurement(w, packed, scale, codes, manifest["weights_sha256"])
            expected_bytes += w.numel() // 2
            if not controls:
                # Flip a real stored nibble in a detached copy; never mutate the artifact.
                bad = packed[:1].clone()
                bad[0, 0] ^= 1
                try:
                    check_codes(w[:1], bad, scale[:1])
                except ValueError as err:
                    if "unpacked codes differ" not in str(err):
                        raise
                    controls.append("one code-bit flip rejected")
                try:
                    check_keyset(set(ct.keys()) | {name}, expected)
                except ValueError:
                    controls.append("dense projection sidecar rejected")
        for name in sorted(others):
            original, saved = source.get_tensor(name), ct.get_tensor(name)
            if saved.dtype != torch.bfloat16 or saved.shape != original.shape:
                raise ValueError(f"{name}: dense BF16 layout mismatch")
            for a, b in zip(original.flatten().split(1 << 20), saved.flatten().split(1 << 20), strict=True):
                if not torch.equal(a.to(torch.bfloat16), b):
                    raise ValueError(f"{name}: dense BF16 values differ")
    if len(controls) != 2 or expected_bytes != manifest["packed_weight_bytes"]:
        raise ValueError("control or byte-count check failed")
    if table_out is not None:
        from app.packed_targets import CT, OBJECTIVE, SCHEMA, layer_names

        if base.get("model_type") != "qwen3" or rows.keys() != set(
            layer_names(CT, config["num_hidden_layers"])
        ):
            raise ValueError("not a complete Qwen3 decoder")
        table = {
            "schema": SCHEMA,
            "target": CT,
            "architecture": "qwen3",
            "blocks": config["num_hidden_layers"],
            "reference_sha256": provenance[WEIGHTS],
            "objective": OBJECTIVE,
            "layers": rows,
        }
        with table_out.open("x", encoding="utf-8") as f:
            json.dump(table, f, indent=1, allow_nan=False)
            f.write("\n")

    return {
        "status": "packed_codes_verified",
        "source_weights_sha256": provenance[WEIGHTS],
        "weights_sha256": manifest["weights_sha256"],
        "linear_projections": len(linears),
        "packed_bytes": expected_bytes,
        "source_kind": manifest["source"]["kind"],
        "negative_controls": controls,
        "torch": torch.__version__,
        "compressed_tensors": version("compressed-tensors"),
        "unpacker": f"{unpack_from_int32.__module__}.{unpack_from_int32.__name__}",
        "claims_runtime_execution": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-hf", type=Path, required=True, help="exact dense input used for export")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table-out", type=Path, help="optional target-native W4A16 selection table")
    args = parser.parse_args()
    result = audit(args.dense_hf, args.checkpoint, args.table_out)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"HF_W4A16_STORAGE_VERIFIED layers={result['linear_projections']} bytes={result['packed_bytes']}")


if __name__ == "__main__":
    main()
