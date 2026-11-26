"""Repack a group-128 CT W4A16 Qwen3 or Llama-3.2-1B as *standard* GGUF Q4_0, without requantizing.

A corresponding F32 GGUF from gguf_export.py provides tokenizer/architecture metadata and
an oracle for tensor/rotary row ordering. Each 128-weight group becomes four standard
32-weight Q4_0 blocks with *the same* scale. Reject any scale that would round in FP16.
Embeddings remain BF16 and tied; normalization vectors are stored as their exact F32 values.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from hf_packed_export import SIDECARS, WEIGHTS, expected_shapes, quant_config, sha256, verify_tied_head
from safetensors import safe_open


def alias_order(ref: np.ndarray, w: torch.Tensor, name: str, cfg: dict) -> np.ndarray | None:
    """Prove the row layout from the original dense tensor, not by inferring from the name alone."""
    if ref.shape != tuple(w.shape):
        raise ValueError(f"{name}: template shape differs")

    # Check every FP32 bit, including sign of zero. Chunk the embedding comparison.
    def agrees(row_order: np.ndarray | None) -> bool:
        for start in range(0, ref.shape[0], 2048):
            sl = slice(start, start + 2048)
            x = w[sl] if row_order is None else w[row_order[sl]]
            if not np.array_equal(ref[sl].view(np.uint32), x.float().numpy().view(np.uint32)):
                return False
        return True

    if agrees(None):
        return None
    if ".self_attn.q_proj." in name or ".self_attn.k_proj." in name:
        n = cfg["num_attention_heads"] if ".q_proj." in name else cfg["num_key_value_heads"]
        ids = np.arange(w.shape[0])
        for a, b in ((2, -1), (-1, 2)):
            perm = ids.reshape(n, a, b).swapaxes(1, 2).reshape(-1)
            if agrees(perm):
                return perm
    raise ValueError(f"{name}: cannot establish equivalent F32 GGUF tensor layout")


def ct_codes(packed: torch.Tensor) -> np.ndarray:
    p = packed.numpy()
    if p.dtype != np.int32 or p.ndim != 2:
        raise ValueError("expected I32 weight_packed")
    shifts = np.arange(8, dtype=np.uint32) * 4
    return ((p.view(np.uint32)[:, :, None] >> shifts) & 15).astype(np.uint8).reshape(len(p), -1)


def q4_blocks(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    n, k = codes.shape
    if k % 128 or scales.shape != (n, k // 128):
        raise ValueError("not group-128 W4")
    fp16 = scales.astype("<f2")
    if not (np.isfinite(scales).all() and (scales > 0).all() and np.array_equal(fp16.astype("f4"), scales)):
        raise ValueError("Q4_0 FP16 scales would change original dequantized values")
    groups = codes.reshape(n, -1, 32)
    data = np.empty((n, k // 32, 18), dtype=np.uint8)
    data[:, :, :2] = fp16.repeat(4, axis=1)[:, :, None].view(np.uint8)
    data[:, :, 2:] = groups[:, :, :16] | (groups[:, :, 16:] << 4)
    return data.reshape(n, -1)


def f32_bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def check_original_metadata(template: Any, src: Path, repo: Path, out: Path, arch: str) -> None:
    """Compare every model/tokenizer metadata value with a fresh official vocab-only export."""
    import gguf

    tokenizer = json.loads((src / "tokenizer_config.json").read_text())
    if tokenizer.get("auto_map") or json.loads((src / "config.json").read_text()).get("auto_map"):
        raise ValueError("custom HF code not supported")
    subprocess.run(
        [
            sys.executable,
            str(repo / "convert_hf_to_gguf.py"),
            str(src),
            "--vocab-only",
            "--outtype",
            "f32",
            "--outfile",
            str(out),
        ],
        check=True,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )
    reference = gguf.GGUFReader(out)
    try:

        def relevant(reader: Any) -> set[str]:
            return {
                k
                for k in reader.fields
                if k.startswith((arch + ".", "tokenizer.")) or k == "general.architecture"
            }

        if relevant(template) != relevant(reference):
            raise ValueError("GGUF template lacks matching architecture/tokenizer metadata keys")
        for key in relevant(reference):
            a, b = template.fields[key], reference.fields[key]
            if a.types != b.types or a.contents() != b.contents():
                raise ValueError(f"GGUF template has different tokenizer/architecture metadata: {key}")
    finally:
        reference.data._mmap.close()


def repack(
    src: Path, ckpt: Path, template: Path, repo: Path, dest: Path, *, original: bool = False
) -> dict[str, Any]:
    sys.path.insert(0, str(repo.resolve() / "gguf-py"))
    import gguf

    sidecar = Path(str(dest) + ".manifest.json")
    if os.path.lexists(dest) or os.path.lexists(sidecar) or sys.byteorder != "little":
        raise ValueError("requires a new GGUF/manifest path and a little-endian host")
    cm = json.loads((ckpt / "tricast_w4a16_manifest.json").read_text())
    cfg = json.loads((ckpt / "config.json").read_text())
    src_cfg = json.loads((src / "config.json").read_text())
    origin = cm["source"]
    expected_origin = "dense_hf" if original else "requantized_tricast_dense_approximation"
    if cfg["quantization_config"] != quant_config() or origin["kind"] != expected_origin:
        raise ValueError("expected group-128 W4 from selected source kind")
    expected_shapes(src_cfg)
    arch = src_cfg["model_type"]
    if not original and arch != "qwen3":
        raise ValueError("only Qwen3 supports the TriCast dense-HF source route")
    expected_config = dict(src_cfg, torch_dtype="bfloat16", quantization_config=quant_config())
    if "dtype" in src_cfg:
        expected_config["dtype"] = "bfloat16"
    if cfg != expected_config:
        raise ValueError("checkpoint architecture differs from source")
    template_sha = sha256(template)
    if not original:
        src_manifest = sha256(src / "tricast_dense_manifest.json")
        tm = json.loads(Path(str(template) + ".manifest.json").read_text())
        if (
            tm["schema"] != "tricast.gguf/1"
            or tm["dense_hf_manifest_sha256"] != src_manifest
            or origin["dense_manifest_sha256"] != src_manifest
            or template_sha != tm["sha256"]
        ):
            raise ValueError("TriCast-derived template lineage mismatch")
    if (
        sha256(src / WEIGHTS) != origin["file_sha256"][WEIGHTS]
        or sha256(ckpt / WEIGHTS) != cm["weights_sha256"]
    ):
        raise ValueError("checkpoint or source digest changed")
    for name in SIDECARS:
        if name in origin["file_sha256"]:
            if sha256(src / name) != origin["file_sha256"][name]:
                raise ValueError(f"{name}: changed dense HF sidecar")
            if name != "config.json" and sha256(ckpt / name) != origin["file_sha256"][name]:
                raise ValueError(f"{name}: changed CT sidecar")
    T = gguf.GGMLQuantizationType
    template_reader = gguf.GGUFReader(template)
    if template_reader.fields["general.architecture"].contents() != arch:
        raise ValueError("template architecture mismatch")
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH[arch.upper()], cfg["num_hidden_layers"])
    types: Counter[str] = Counter()
    orders: dict[str, np.ndarray | None] = {}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".w4-gguf-", dir=dest.parent) as tmp,
        safe_open(ckpt / WEIGHTS, framework="pt") as ct,
        safe_open(src / WEIGHTS, framework="pt") as dense,
    ):
        path = Path(tmp) / "out.gguf"
        if original:
            check_original_metadata(
                template_reader, src, repo.resolve(), Path(tmp) / "reference-vocab.gguf", arch
            )
        extra: dict[str, np.ndarray] = {}
        if arch == "llama":
            # The converter synthesizes this RoPE table from config, not a HF parameter.
            # Recreate it with the *same official implementation*, not a guessed formula.
            llama = runpy.run_path(str(repo / "convert_hf_to_gguf.py"))["LlamaModel"]
            converter = llama(src, gguf.LlamaFileType.ALL_F32, path, hparams=src_cfg)
            extra = {k: v.numpy() for k, v in converter.generate_extra_tensors()}
            if set(extra) != {"rope_freqs.weight"}:
                raise ValueError("unexpected generated tensors for Llama-3.2-1B")
        writer = gguf.GGUFWriter(path, arch)
        for name, field in template_reader.fields.items():
            if name.startswith("GGUF.") or name in (
                "general.architecture",
                "general.file_type",
                "general.quantization_version",
            ):
                continue
            ty = field.types[0]
            sub = field.types[-1] if ty == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, field.contents(), ty, sub_type=sub)
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        template_keys = {t.name for t in template_reader.tensors}
        src_keys = set(dense.keys())
        if "lm_head.weight" in src_keys:
            verify_tied_head(
                dense.get_tensor("lm_head.weight"), dense.get_tensor("model.embed_tokens.weight")
            )
            if "output.weight" not in template_keys:
                src_keys.remove("lm_head.weight")
        names = {mapping.get_name(k, try_suffixes=(".weight",)): k for k in src_keys}
        if (
            set(names) | set(extra) != template_keys
            or len(names) != len(src_keys)
            or names.keys() & extra.keys()
        ):
            raise ValueError("source tensor map differs from F32 template")
        consumed: set[str] = set()
        for t in template_reader.tensors:
            if t.name in extra:
                if t.tensor_type != T.F32 or not f32_bits_equal(t.data, extra[t.name]):
                    raise ValueError("template RoPE tensor differs from official converter")
                writer.add_tensor(t.name, extra[t.name], raw_dtype=T.F32)
                types["F32"] += 1
                continue
            hf = names[t.name]
            w = dense.get_tensor(hf)
            if t.tensor_type != T.F32:
                raise ValueError("expected an all-F32 template")
            order = alias_order(t.data, w, hf, cfg)
            if hf == "lm_head.weight":
                if t.name != "output.weight" or order is not None:
                    raise ValueError("invalid optional tied output in template")
                continue  # llama.cpp uses token_embd.weight when output.weight is absent.
            orders[hf] = order
            prefix = hf.removesuffix("weight")
            if prefix + "weight_packed" in ct.keys():
                codes = ct_codes(ct.get_tensor(prefix + "weight_packed"))
                scales = ct.get_tensor(prefix + "weight_scale")
                shape = ct.get_tensor(prefix + "weight_shape").tolist()
                if shape != list(w.shape) or list(codes.shape) != shape or scales.dtype != torch.bfloat16:
                    raise ValueError(f"{hf}: invalid W4 tensor dimensions/type")
                if order is not None:
                    codes, scales = codes[order], scales[order]
                arr, ty = q4_blocks(codes, scales.float().numpy()), T.Q4_0
                consumed.update(prefix + s for s in ("weight_scale", "weight_shape", "weight_packed"))
            else:
                value = ct.get_tensor(hf)
                if value.dtype != torch.bfloat16 or value.shape != w.shape or order is not None:
                    raise ValueError(f"{hf}: unexpected dense CT tensor")
                if hf == "model.embed_tokens.weight":
                    arr, ty = value.view(torch.uint8).numpy(), T.BF16
                elif value.ndim == 1:
                    arr, ty = value.float().numpy(), T.F32
                else:
                    raise ValueError(f"{hf}: unquantized projection")
                consumed.add(hf)
            types[ty.name] += 1
            writer.add_tensor(t.name, np.ascontiguousarray(arr), raw_dtype=ty)
        if consumed != set(ct.keys()) or types["Q4_0"] != 7 * cfg["num_hidden_layers"] or types["BF16"] != 1:
            raise ValueError("unexpected or missing CT tensors")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        # Verify the artifact reopened from disk using upstream dequantize (not our nibble decoder).
        r = gguf.GGUFReader(path)
        checked = 0
        extras_checked = set()
        for t in r.tensors:
            if t.name in extra:
                if t.tensor_type != T.F32 or not f32_bits_equal(t.data, extra[t.name]):
                    raise ValueError("changed generated RoPE tensor")
                extras_checked.add(t.name)
                continue
            hf, order = names[t.name], orders[names[t.name]]
            if t.tensor_type == T.Q4_0:
                prefix = hf.removesuffix("weight")
                codes = ct_codes(ct.get_tensor(prefix + "weight_packed")).astype(np.float32) - 8
                scales = ct.get_tensor(prefix + "weight_scale").float().numpy()
                expected = (codes.reshape(*scales.shape, 128) * scales[:, :, None]).reshape(codes.shape)
                if order is not None:
                    expected = expected[order]
                if not f32_bits_equal(gguf.dequantize(t.data, T.Q4_0), expected):
                    raise ValueError(f"{hf}: upstream GGUF decoder differs from source CT weights")
            else:
                w = ct.get_tensor(hf)
                if tuple(t.shape[::-1]) != tuple(w.shape):
                    raise ValueError(f"{hf}: uncompressed shape mismatch")
                for start in range(0, w.shape[0], 2048):
                    sl = slice(start, start + 2048)
                    actual = gguf.dequantize(t.data[sl], t.tensor_type)
                    if not f32_bits_equal(actual, w[sl].float().numpy()):
                        raise ValueError(f"{hf}: BF16/F32 value changed")
            checked += 1
        if checked != len(orders) or extras_checked != set(extra):
            raise ValueError("incomplete validation")
        r.data._mmap.close()
        template_reader.data._mmap.close()
        report = {
            "schema": "tricast.hf_w4a16.gguf/1",
            "source_ct_sha256": cm["weights_sha256"],
            "source_template_sha256": template_sha,
            "source_kind": origin["kind"],
            "architecture": arch,
            "generated_tensors_verified": sorted(extras_checked),
            "gguf_converter_sha256": sha256(repo / "convert_hf_to_gguf.py"),
            "tensor_counts": dict(types),
            "reconstructed_tensors_exact": checked,
            "permuted_tensors": [k for k, v in orders.items() if v is not None],
            "requantized_from_ct": False,
            "mixed_tricast_codes_preserved": False,
            "logits_verified": False,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        facts = Path(tmp) / "manifest.json"
        facts.write_text(json.dumps(report, indent=2) + "\n")
        os.link(path, dest)
        os.link(facts, sidecar)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dense-hf", type=Path, help="TriCast-derived HF input used to create CT checkpoint")
    source.add_argument("--original-hf", type=Path, help="original HF input used to create CT checkpoint")
    for name in ("checkpoint", "template", "llama-cpp", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = repack(
        a.original_hf or a.dense_hf,
        a.checkpoint,
        a.template,
        a.llama_cpp,
        a.out,
        original=a.original_hf is not None,
    )
    print(f"CT_GGUF_EXACT_WEIGHTS tensors={result['reconstructed_tensors_exact']} bytes={result['bytes']}")


if __name__ == "__main__":
    main()
