"""Verify original TriCast package bytes and bounded tile inference, independently of export.

The oracle decodes each original weight once to FP32 outside measured tiled inference;
it uses the same transformed inputs, F.linear and output casts as pack mode=codes.
Report logit differences rather than calling differently shaped GEMMs bit-exact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from tricast import pack
from tricast.bitpack import same_bits


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def metrics(
    model: torch.nn.Module, ids: torch.Tensor, *, raw: bool = False
) -> tuple[float, torch.Tensor | None]:
    logits = model(input_ids=ids, use_cache=False).logits
    # FP32 token NLL, sum in FP64. Do not promote the full vocabulary sequence to FP64.
    vals = [
        F.cross_entropy(
            logits[:, a:b, :].float().reshape(-1, logits.shape[-1]),
            ids[:, a + 1 : b + 1].reshape(-1),
            reduction="none",
        )
        .double()
        .sum()
        .item()
        for a in range(0, ids.shape[1] - 1, 64)
        for b in [min(a + 64, ids.shape[1] - 1)]
    ]
    nll = math.fsum(vals)
    if not math.isfinite(nll):
        raise ValueError("nonfinite next-token NLL")
    return nll, logits.detach().cpu() if raw else None


@torch.inference_mode()
def run(a: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(a.device)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    if a.seqlen < 2:
        raise ValueError("seqlen must be at least 2")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((a.package / pack.MANIFEST).read_bytes())
    for name in (pack.CONFIG, pack.WEIGHTS, "config.json"):
        if manifest["files"][name]["sha256"] != pack._sha256_file(a.package / name):
            raise ValueError(f"source digest changed: {name}")
    data = []
    for path in a.inputs:
        with np.load(path, allow_pickle=False) as f:
            ids = f["input_ids"]
            if ids.ndim != 2 or ids.shape[1] < 2 or ids.dtype.kind not in "iu":
                raise ValueError(f"{path}: expected integer [B,S>=2]")
            if "attention_mask" in f and not np.all(f["attention_mask"] == 1):
                raise ValueError("padded inputs not supported")
            data.extend(torch.from_numpy(row.copy()).long().view(1, -1).to(device) for row in ids)
    corpus = np.load(a.corpus_ids, mmap_mode="r", allow_pickle=False) if a.corpus_ids else None
    if corpus is not None and (corpus.ndim != 1 or corpus.dtype.kind not in "iu" or len(corpus) < a.seqlen):
        raise ValueError("need at least one full corpus window of integer ids")
    result: dict[str, Any] = {
        "schema": "tricast.tiled-audit/1",
        "oracle": "full FP32 projection operands decoded once, codes-mode GEMMs; not BF16 deployment",
        "memory_scope": "PyTorch CUDA allocator including inputs, workspace, HF state; not device RSS",
        "device": str(device),
        "dtype": a.dtype,
        "torch": torch.__version__,
        "seed": 42,
        "host": platform.node(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tile_rows_requested": a.tile_rows,
        "package_manifest_sha256": pack._sha256_file(a.package / pack.MANIFEST),
        "weights_sha256": manifest["files"][pack.WEIGHTS]["sha256"],
        "recipe_sha256": manifest["recipe_sha256"],
        "inputs_sha256": [pack._sha256_file(p) for p in a.inputs],
        "corpus_sha256": pack._sha256_file(a.corpus_ids) if corpus is not None else None,
        "source_sha256": pack._sha256_file(Path(__file__)),
        "arms": {},
    }
    candidate: list[torch.Tensor] = []
    arms = ("tiled", "oracle") if a.arm == "both" else (a.arm,)
    for arm in arms:
        preload_bytes = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
            preload_bytes = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model, _ = pack.load(
            a.package,
            device=a.device,
            mode="tiled" if arm == "tiled" else "codes",
            dtype=dtype,
            tile_rows=a.tile_rows,
        )
        mods = {n: m for n, m in model.named_modules() if isinstance(m, pack.PackedLinear)}
        if len(mods) != manifest["quantized_layers"]:
            raise ValueError("package projection coverage mismatch")
        if arm == "oracle":
            # Decode once for oracle forwards; do not take mode=load's BF16 GEMM shortcut.
            for m in mods.values():
                w = m._weight().float()
                for key in list(m._buffers):
                    if key.startswith("stored_"):
                        delattr(m, key)
                m.register_buffer("weight", w)
                m.mode = "load"
        report: dict[str, Any] = {"load_s": time.perf_counter() - start, "case_nll_sum": []}
        if device.type == "cuda":
            torch.cuda.synchronize()
            report["preload_cuda_bytes"] = preload_bytes
            report["load_peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
            report["resident_cuda_bytes"] = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        if arm == "oracle" and candidate:
            report["logit_parity"] = []
        for idx, ids in enumerate(data):
            nll, logits = metrics(model, ids, raw=True)
            report["case_nll_sum"].append(nll)
            if arm == "tiled" and a.arm == "both":
                candidate.append(logits)
            elif candidate:
                x, y = candidate[idx].float(), logits.float()
                report["logit_parity"].append(
                    {
                        "max_abs": (x - y).abs().max().item(),
                        "bit_equal": torch.equal(x, y),
                        "cosine": F.cosine_similarity(
                            x.flatten().double(), y.flatten().double(), dim=0
                        ).item(),
                        "same_argmax": int((x.argmax(-1) == y.argmax(-1)).sum()),
                        "positions": ids.numel(),
                    }
                )
        if corpus is not None:
            sums = [
                metrics(model, torch.tensor(corpus[i : i + a.seqlen].tolist(), device=device)[None])[0]
                for i in range(0, len(corpus) - a.seqlen + 1, a.seqlen)
            ]
            tokens = len(sums) * (a.seqlen - 1)
            report["corpus"] = {
                "n_tokens": tokens,
                "windows": len(sums),
                "seqlen": a.seqlen,
                "tail": len(corpus) % a.seqlen,
                "ppl": math.exp(math.fsum(sums) / tokens),
                "sum_nll": math.fsum(sums),
            }
        if device.type == "cuda":
            torch.cuda.synchronize()
            report["forward_peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
        if arm == "tiled":
            # This intentionally full decoded oracle is OUTSIDE all runtime load/forward peak measurements.
            counts: Counter = Counter()
            scanned = []
            with safe_open(a.package / pack.WEIGHTS, framework="pt") as src:
                for name, m in mods.items():
                    stored = {k[7:]: t for k, t in m._buffers.items() if k.startswith("stored_")}
                    if any(digest(t) != digest(src.get_tensor(f"{name}.{k}")) for k, t in stored.items()):
                        raise ValueError(f"{name}: resident representation differs from the checkpoint")
                    original = pack.decode(m.entry, stored, m.spec)
                    for row in range(0, m.out_features, m.tile_rows):
                        stop = min(row + m.tile_rows, m.out_features)
                        if not same_bits(original[row:stop], m.weight_tile(row, stop)):
                            raise ValueError(f"{name}:{row}:{stop}: tile values differ from pack.decode")
                    if m.tile_calls == 0 or not 0 < m.max_tile_elements <= m.tile_rows * m.in_features:
                        raise ValueError(f"{name}: tiled inference not observed")
                    if m.entry["kind"] != "dense" and m.max_tile_elements >= original.numel():
                        raise ValueError(f"{name}: whole projection decoded")
                    counts[f"{m.entry['kind']}/{m.entry['bits']}"] += 1
                    scanned.append(
                        {
                            "name": name,
                            "kind": m.entry["kind"],
                            "bits": m.entry["bits"],
                            "tile_calls": m.tile_calls,
                            "max_tile_elements": m.max_tile_elements,
                            "weight_elements": original.numel(),
                            "stored_bytes": sum(t.numel() * t.element_size() for t in stored.values()),
                        }
                    )
                    del original, stored
            report.update(payload_and_tile_check="all_layers_equal", counts=dict(counts), layers=scanned)
        result["arms"][arm] = report
        del model, mods, m
        gc.collect()
    return result


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--arm", choices=("tiled", "oracle", "both"), default="both")
    p.add_argument("--tile-rows", type=int, default=128)
    p.add_argument("--corpus-ids", type=Path)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite prior results")
    r = run(args)
    args.out.write_text(json.dumps(r, allow_nan=False, indent=1) + "\n")
    print(f"TILED_AUDIT_DONE arms={list(r['arms'])} out={args.out}")
