"""Opt-in dense deployment bridge for a TriCast Qwen3 package.

`dense-hf` decodes the supplied package, not the original HF checkpoint. An FP32
Linear can fold input transforms algebraically, but cannot encode the packed
codes or per-layer MMA output rounding. `gguf` uses the official llama.cpp
converter on that checkpoint. `requantize` optionally runs llama-quantize to create
a new Q4_K_M or Q8_0 approximation. None of these preserves the original low-bit codes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch

from tricast.pack import CHECKPOINT_FILES, CONFIG, MANIFEST, SCHEMA, WEIGHTS, PackedLinear, load
from tricast.transforms import LinearTransform

REPORT = "tricast_dense_manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def verify(package: Path) -> tuple[dict, dict]:
    manifest = read(package / MANIFEST)
    files = manifest["files"]
    for name in {
        CONFIG,
        WEIGHTS,
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        *CHECKPOINT_FILES,
    }:
        p = package / name
        if not p.is_file() and name not in (
            CONFIG,
            WEIGHTS,
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            if name in files:
                raise ValueError(f"missing declared package file: {name}")
            continue
        if (
            name not in files
            or not p.is_file()
            or p.stat().st_size != files[name]["bytes"]
            or digest(p) != files[name]["sha256"]
        ):
            raise ValueError(f"missing or changed package file: {name}")
    c = read(package / CONFIG)
    if manifest["schema"] != SCHEMA or c["schema"] != SCHEMA or manifest["revision"] != c["revision"]:
        raise ValueError("inconsistent package metadata")
    if (
        manifest["recipe_sha256"] != c["recipe"]["sha256"]
        or hashlib.sha256(c["recipe"]["yaml"].encode()).hexdigest() != manifest["recipe_sha256"]
    ):
        raise ValueError("recipe SHA-256 mismatch")
    if read(package / "config.json")["model_type"] != "qwen3":
        raise ValueError("dense bridge validated for Qwen3 only")
    return c, manifest


def folded_weight(w: torch.Tensor, t: LinearTransform) -> torch.Tensor:
    """Return W T^T for inputs x T: random Hadamard is D H, not H D."""
    if t.kind == "none":
        return w.float()
    w = w.double()
    if t.kind in ("hadamard", "random_hadamard"):
        v = LinearTransform("hadamard", t.block).apply_activation(w)
        if t.kind == "random_hadamard":
            if t.diag is None or not torch.all(t.diag.abs() == 1):
                raise ValueError("invalid random Hadamard sign diagonal")
            v = v * t.diag.double()
    elif t.kind in ("awq", "smoothquant"):
        if t.diag is None or not torch.isfinite(t.diag).all() or (t.diag <= 0).any():
            raise ValueError("invalid scaling diagonal")
        v = w / t.diag.double()
    else:
        raise ValueError(f"unknown input transform: {t.kind}")
    return v.float()


@torch.inference_mode()
def materialize(package: Path, dst: Path) -> None:
    c, orig = verify(package)
    model, _ = load(package, dtype=torch.float32)
    layers = {n: mod for n, mod in model.named_modules() if isinstance(mod, PackedLinear)}
    if set(layers) != set(c["layers"]) or not layers:
        raise ValueError("did not recover every packed layer")
    counts: Counter = Counter()
    rounding: Counter = Counter()
    worst_err = 0.0
    g = torch.Generator().manual_seed(42)
    for name, layer in layers.items():
        if (
            layer.spec.activation is not None
            or layer.spec.outliers is not None
            or layer.spec.sparsity.kind != "none"
        ):
            raise ValueError(f"{name}: only weight-only packages supported")
        if layer.out_dtype not in (torch.float32, torch.float16, torch.bfloat16, None):
            raise ValueError(f"unsupported MMA output: {layer.out_dtype}")
        t = layer.transform
        counts[t.kind] += 1
        rounding[str(layer.out_dtype)] += 1
        w = layer._weight().detach()
        W = folded_weight(w, t)
        if not torch.isfinite(W).all():
            raise ValueError(f"{name}: nonfinite folded weights")
        # Independent algebraic oracle at fixed inputs before any output casts;
        # normalize by the absolute dot-product scale, not a nearly-zero result.
        x = torch.randn(2, w.shape[1], dtype=torch.float64, generator=g)
        a = t.apply_activation(x) @ w.double().T
        b = x @ W.double().T
        scale = (x.abs() @ W.double().abs().T).clamp_min(1e-30)
        err = ((a - b).abs() / scale).amax().item()
        if not err <= 2e-7:
            raise ValueError(f"{name}: fold validation failed: {err}")
        worst_err = max(worst_err, err)
        mod = torch.nn.Linear(
            layer.in_features, layer.out_features, bias=layer.bias is not None, device="meta"
        )
        mod.weight = torch.nn.Parameter(W.contiguous(), requires_grad=False)
        if layer.bias is not None:
            mod.bias = torch.nn.Parameter(layer.bias.detach().float().contiguous(), requires_grad=False)
        pname, _, child = name.rpartition(".")
        setattr(model.get_submodule(pname), child, mod)
    # Re-establish ties in storage, not by value-based deduplication.
    model.tie_weights()
    for alias, source in c.get("tied", {}).items():
        if model.get_parameter(alias).data_ptr() != model.get_parameter(source).data_ptr():
            raise ValueError(f"tied tensor not preserved: {alias}")
    model.save_pretrained(dst, safe_serialization=True, max_shard_size="4GB")
    for name in CHECKPOINT_FILES:
        if name not in ("config.json", "generation_config.json") and (package / name).is_file():
            shutil.copyfile(package / name, dst / name)
    if (package / "generation_config.json").is_file():
        shutil.copyfile(package / "generation_config.json", dst / "generation_config.json")
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(dst.iterdir()) if p.is_file()
    }
    write(
        {
            "schema": "tricast.dense_hf/1",
            "model": c["model"],
            "revision": c["revision"],
            "source_pack_manifest_sha256": digest(package / MANIFEST),
            "source_weights_sha256": orig["files"][WEIGHTS]["sha256"],
            "recipe_sha256": orig["recipe_sha256"],
            "transforms": dict(counts),
            "source_output_dtypes": dict(rounding),
            "tied": c.get("tied", {}),
            "decoded_layers": len(layers),
            "audit_seed": 42,
            "worst_normalized_fold_error": worst_err,
            "audit_threshold": 2e-7,
            "original_low_bit_storage_preserved": False,
            "per_layer_output_rounding_preserved": False,
            "storage": "dense FP32 HF safetensors; not original HF or a low-bit HF quantization backend",
            "files": files,
        },
        dst / REPORT,
    )


def check_types(tensors: list[tuple[int, str]], fmt: str) -> None:
    """This bridge handles Qwen3: vectors stay F32; all matrices must use the requested block family."""
    allowed = {"F32": {"F32"}, "Q4_K_M": {"Q4_K", "Q6_K"}, "Q8_0": {"Q8_0"}}
    if fmt not in allowed:
        raise ValueError(f"unsupported GGUF target {fmt}")
    matrices = {ty for rank, ty in tensors if rank > 1}
    if not matrices or matrices != allowed[fmt] or any(ty != "F32" for rank, ty in tensors if rank <= 1):
        raise ValueError(f"GGUF physical tensor types do not match {fmt}: {matrices}")


def tensor_layout(path: Path, repo: Path, fmt: str | None = None) -> tuple[dict, Counter]:
    """Return detached dimensions/types; close GGUFReader's mmap before NFS staging cleanup."""
    sys.path.insert(0, str(repo.resolve() / "gguf-py"))
    from gguf import GGUFReader, LlamaFileType

    reader = GGUFReader(str(path))
    try:
        shapes = {t.name: tuple(map(int, t.shape)) for t in reader.tensors}
        if not shapes or len(shapes) != len(reader.tensors):
            raise ValueError("empty or duplicate GGUF tensors")
        if fmt is not None:
            declared = reader.fields["general.file_type"]
            expected = {
                "F32": LlamaFileType.ALL_F32,
                "Q4_K_M": LlamaFileType.MOSTLY_Q4_K_M,
                "Q8_0": LlamaFileType.MOSTLY_Q8_0,
            }[fmt]
            if int(declared.parts[declared.data[0]][0]) != int(expected):
                raise ValueError(f"GGUF general.file_type is not {fmt}")
            check_types([(len(t.shape), t.tensor_type.name) for t in reader.tensors], fmt)
        return shapes, Counter(t.tensor_type.name for t in reader.tensors)
    finally:
        reader.data._mmap.close()  # upstream uses numpy.memmap and has no public close API


def audit_file(path: Path, repo: Path) -> tuple[dict, dict]:
    facts = read(Path(str(path) + ".manifest.json"))
    if (
        facts["schema"] != "tricast.gguf/1"
        or path.stat().st_size != facts["bytes"]
        or digest(path) != facts["sha256"]
    ):
        raise ValueError("GGUF size or digest does not match its manifest")
    shapes, types = tensor_layout(path, repo, facts.get("requantization", "F32"))
    if dict(types) != facts["tensor_counts"]:
        raise ValueError("GGUF physical types differ from manifest")
    return facts, shapes


def requantize(src: Path, tool: Path, repo: Path, threads: int, dst: Path, fmt: str) -> dict:
    if threads < 1:
        raise ValueError("--threads must be positive")
    metadata, old_shapes = audit_file(src, repo)
    if set(metadata["tensor_counts"]) != {"F32"}:
        raise ValueError("only an all-F32 derived GGUF may be requantized")
    subprocess.run([str(tool.resolve()), str(src), str(dst), fmt, str(threads)], check=True)
    new_shapes, types = tensor_layout(dst, repo, fmt)
    if old_shapes != new_shapes:
        raise ValueError("the output is missing model tensors or dimensions differ")
    return {
        "schema": "tricast.gguf/1",
        "source_pack_manifest_sha256": metadata["source_pack_manifest_sha256"],
        "parent_dense_gguf_sha256": metadata["sha256"],
        "quantizer_sha256": digest(tool),
        "original_low_bit_storage_preserved": False,
        "per_layer_output_rounding_preserved": False,
        "requantization": fmt,
        "tensor_counts": dict(types),
        "bytes": dst.stat().st_size,
        "sha256": digest(dst),
    }


def publish(file: Path, info: dict, out: Path, sidecar: Path) -> None:
    manifest = file.parent / "manifest.json"
    write(info, manifest)
    os.link(file, out)
    try:
        os.link(manifest, sidecar)
    except OSError:
        out.unlink()
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    h = sub.add_parser("dense-hf", help="materialize a stock HF checkpoint from the supplied TriCast package")
    h.add_argument("--package", type=Path, required=True)
    g = sub.add_parser("gguf", help="convert an audited dense-hf directory to standard F32 GGUF")
    g.add_argument("--dense-hf", type=Path, required=True)
    g.add_argument("--llama-cpp", type=Path, required=True, help="trusted upstream llama.cpp checkout")
    r = sub.add_parser("requantize", help="explicitly re-quantize a derived F32 GGUF")
    r.add_argument("--type", choices=("Q4_K_M", "Q8_0"), default="Q4_K_M")
    r.add_argument("--gguf", type=Path, required=True)
    r.add_argument("--llama-quantize", type=Path, required=True)
    r.add_argument("--llama-cpp", type=Path, required=True)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--allow-requantization", action="store_true")
    audit = sub.add_parser(
        "audit", help="inspect a published GGUF against its file manifest without rewriting it"
    )
    audit.add_argument("--gguf", type=Path, required=True)
    audit.add_argument("--llama-cpp", type=Path, required=True)
    for a in (h, g, r):
        a.add_argument("--out", type=Path, required=True)
    for a in (h, g):
        a.add_argument("--allow-dense-approximation", action="store_true")
    args = p.parse_args()
    if args.command == "audit":
        info, _ = audit_file(args.gguf, args.llama_cpp)
        print(f"GGUF_AUDIT_PASS sha256={info['sha256']} tensors={info['tensor_counts']}", flush=True)
        return
    if args.command == "requantize" and not args.allow_requantization:
        p.error(f"{args.type} changes numerical weights; pass --allow-requantization")
    if args.command != "requantize" and not args.allow_dense_approximation:
        p.error("cannot preserve bitwidths or MMA output rounding; pass --allow-dense-approximation")
    out: Path = args.out.absolute()
    sidecar = Path(str(out) + ".manifest.json")
    if os.path.lexists(out) or (args.command != "dense-hf" and os.path.lexists(sidecar)):
        raise FileExistsError(f"refusing to overwrite {out} or its manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tricast-", dir=out.parent) as tmp:
        stage = Path(tmp)
        if args.command == "dense-hf":
            path = stage / "hf"
            path.mkdir()
            materialize(args.package, path)
            path.rename(out)
            marker = "DENSE_HF_EXPORTED"
        elif args.command == "requantize":
            f = stage / "model.gguf"
            info = requantize(args.gguf, args.llama_quantize, args.llama_cpp, args.threads, f, args.type)
            publish(f, info, out, sidecar)
            marker = f"GGUF_{args.type}_EXPORTED"
        else:
            hf = args.dense_hf
            facts = read(hf / REPORT)
            if facts["schema"] != "tricast.dense_hf/1":
                raise ValueError("expected a TriCast dense HF bridge directory")
            # Reject an unlisted extra checkpoint too (the converter scans by glob).
            if {p.name for p in hf.iterdir()} != {*facts["files"], REPORT}:
                raise ValueError("extra or missing dense HF file")
            for name, spec in facts["files"].items():
                if (
                    Path(name).name != name
                    or (hf / name).stat().st_size != spec["bytes"]
                    or digest(hf / name) != spec["sha256"]
                ):
                    raise ValueError(f"dense file mismatch: {name}")
            converter = args.llama_cpp.resolve() / "convert_hf_to_gguf.py"
            f = stage / "model.gguf"
            subprocess.run(
                [sys.executable, str(converter), str(hf), "--outtype", "f32", "--outfile", str(f)], check=True
            )
            _, types = tensor_layout(f, converter.parent, "F32")
            if set(types) != {"F32"}:
                raise ValueError("not an all-F32 GGUF")
            info = {
                "schema": "tricast.gguf/1",
                "source_pack_manifest_sha256": facts["source_pack_manifest_sha256"],
                "dense_hf_manifest_sha256": digest(hf / REPORT),
                "converter_sha256": digest(converter),
                "tensor_counts": dict(types),
                "original_low_bit_storage_preserved": False,
                "per_layer_output_rounding_preserved": False,
                "bytes": f.stat().st_size,
                "sha256": digest(f),
            }
            publish(f, info, out, sidecar)
            marker = "GGUF_F32_EXPORTED"
    print(f"{marker} {out}", flush=True)


if __name__ == "__main__":
    main()
