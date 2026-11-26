"""Decoder-only plans over measured, physically packed Qwen3 projection tensors.

Build the GGUF candidate table with `python -m app.packed_targets --reference F32.gguf
--candidate Q4_K_M.gguf Q8_0.gguf --out table.json` (gguf-py from the converter
checkout on PYTHONPATH). This reads already quantized files and compares their
decoded weights, not logits. It does not run or certify any inference backend.

Objective (OBJECTIVE): the sum over projections of each measured candidate's relative weight SSE, a
plan for one deployment target's already-quantized files, not Auto-tune's model-level KL (AGENTS §2).
Plans are written with runtime_verified=false; quality is measured on the target afterwards (user
decision 2026-11-26).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from app import quantizer_recipe as qr

GGUF = "llama.cpp-cpu-gguf"
CT = "vllm-cuda-ct-w4a16"
TARGETS = (GGUF, CT)
SCHEMA = "tricast.autotune.packed-table/1"
GGUF_KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
# ggml/src/ggml-common.h: entries per block and sizeof(block), including scales.
GGUF_BLOCKS = {"Q4_K": (256, 144), "Q6_K": (256, 210), "Q8_0": (32, 34)}
CT_FORMAT = "ct-pack-int4-rtn-symmetric-g128-bf16"
OBJECTIVE = "sum_relative_weight_sse"


def layer_names(target: str, blocks: int) -> list[str]:
    if type(blocks) is not int or not 1 <= blocks <= 1024:
        raise ValueError("invalid number of Qwen3 decoder blocks")
    if target == GGUF:
        return [f"blk.{b}.{kind}.weight" for b in range(blocks) for kind in GGUF_KINDS]
    if target == CT:
        return [qr.layer_name(b, k) + ".weight" for b in range(blocks) for k in qr.KINDS]
    raise ValueError(f"unknown target: {target}")


def _digest(s: Any) -> bool:
    return isinstance(s, str) and re.fullmatch(r"[0-9a-f]{64}", s) is not None


def _payload(target: str, kind: str, n: int, k: int) -> int:
    if target == GGUF and kind in GGUF_BLOCKS:
        block, size = GGUF_BLOCKS[kind]
        if k % block == 0:
            return n * (k // block) * size
    if target == CT and kind == CT_FORMAT and k % 128 == 0 and n % 64 == 0:
        # Packed int32 words + one BF16 scale/group. weight_shape and container overhead excluded.
        return n * (k // 2 + k // 128 * 2)
    raise ValueError(f"{target}: unsupported packed layout or shape: {kind} ({n}, {k})")


def solve(table: dict[str, Any], target: str, bits: Fraction, resolution: Fraction | None) -> dict[str, Any]:
    from app.autotune import solve_mckp

    if (
        not isinstance(table, dict)
        or table.get("schema") != SCHEMA
        or table.get("target") != target
        or table.get("objective") != OBJECTIVE
        or table.get("architecture") != "qwen3"
    ):
        raise ValueError(f"{target} requires a native {SCHEMA} table; do not rename fake-quant KL columns")
    if bits <= 0 or not _digest(table.get("reference_sha256")):
        raise ValueError("positive bit budget and SHA-256 of reference weights required")
    names = layer_names(target, table.get("blocks"))
    rows = table.get("layers")
    if not isinstance(rows, dict) or rows.keys() != set(names):
        raise ValueError("packed table must cover every Qwen3 decoder projection, with no extras")
    choices, costs, scores = [], [], []
    params = 0
    for name in names:
        row = rows[name]
        shape = row.get("shape_nk") if isinstance(row, dict) else None
        if not isinstance(shape, list) or len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(f"{name}: missing positive [out_features, in_features]")
        n, k = shape
        params += n * k
        options = row.get("measurements")
        if not isinstance(options, dict) or not options:
            raise ValueError(f"{name}: no packed target measurement")
        # Deterministic ordering is independent of JSON object insertion order.
        kinds = sorted(options)
        for kind in kinds:
            m = options[kind]
            expected = _payload(target, kind, n, k)
            if (
                not isinstance(m, dict)
                or type(m.get("payload_bytes")) is not int
                or m["payload_bytes"] != expected
            ):
                raise ValueError(f"{name}: {kind} measured payload does not match its block layout")
            if not _digest(m.get("artifact_sha256")) or not _digest(m.get("payload_sha256")):
                raise ValueError(f"{name}: {kind} has no measured artifact/payload SHA")
            score = m.get("relative_sse")
            if type(score) not in (int, float) or not math.isfinite(score) or score < 0:
                raise ValueError(f"{name}: missing finite nonnegative target weight error for {kind}")
        choices.append(kinds)
        costs.append([8 * options[t]["payload_bytes"] for t in kinds])
        scores.append([options[t]["relative_sse"] for t in kinds])
    minimum = sum(min(row) for row in costs)
    if params * bits < minimum:
        raise ValueError(f"budget {bits} bpw below minimum packed payload {Fraction(minimum, params)} bpw")
    sol = solve_mckp(
        [[c - min(row) for c in row] for row in costs], scores, params * bits - minimum, resolution
    )
    selected = {}
    for name, kinds, index in zip(names, choices, sol.choice, strict=True):
        kind = kinds[index]
        selected[name] = {
            "layout": kind,
            "shape_nk": rows[name]["shape_nk"],
            **{
                k: rows[name]["measurements"][kind][k]
                for k in ("artifact_sha256", "payload_sha256", "payload_bytes", "relative_sse")
            },
        }
    return {
        "schema": "tricast.autotune.packed-plan/1",
        "target": target,
        "architecture": "qwen3",
        "reference_sha256": table["reference_sha256"],
        "scope": "decoder projection weight codes and scales only; not total model size or resident memory",
        "objective": OBJECTIVE,
        "objective_value": sol.value,
        "budget_bits": str(bits),
        "resolution_bits": str(resolution) if resolution is not None else None,
        "payload_bytes": int((sol.cost + minimum) / 8),
        "avg_bits": float((sol.cost + minimum) / params),
        "runtime_verified": False,
        "layers": selected,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path) -> dict[str, Any]:
    info = json.loads(Path(str(path) + ".manifest.json").read_text())
    if (
        info.get("schema") != "tricast.gguf/1"
        or info.get("bytes") != path.stat().st_size
        or info.get("sha256") != _hash_file(path)
    ):
        raise ValueError(f"{path}: size or digest does not match TriCast manifest")
    return info


def gguf_table(reference: Path, candidates: Sequence[Path]) -> dict[str, Any]:
    import numpy as np
    from gguf import GGUFReader, dequantize

    ref_meta = _checked(reference)
    ref = GGUFReader(str(reference))
    if ref.get_field("general.architecture").contents() != "qwen3":
        raise ValueError("only Qwen3 projection names are mapped")
    blocks = int(ref.get_field("qwen3.block_count").contents())
    names = layer_names(GGUF, blocks)
    base = {t.name: t for t in ref.tensors}
    if set(ref_meta.get("tensor_counts", {})) != {"F32"} or any(
        t.tensor_type.name != "F32" for t in ref.tensors
    ):
        raise ValueError("reference must be an all-F32 GGUF checkpoint")
    if not set(names) <= base.keys():
        raise ValueError("dense GGUF is missing a decoder projection")
    rows = {name: {"shape_nk": list(map(int, base[name].shape[::-1])), "measurements": {}} for name in names}
    for path in candidates:
        meta = _checked(path)
        if meta.get("parent_dense_gguf_sha256") != ref_meta["sha256"]:
            raise ValueError(f"{path}: not a quantization of the supplied dense GGUF")
        source = GGUFReader(str(path))
        tensors = {t.name: t for t in source.tensors}
        if tensors.keys() != base.keys():
            raise ValueError(f"{path}: tensor set differs from dense GGUF")
        for name in names:
            t, w = tensors[name], base[name]
            n, k = rows[name]["shape_nk"]
            kind = t.tensor_type.name
            if not np.array_equal(t.shape, w.shape) or t.n_bytes != _payload(GGUF, kind, n, k):
                raise ValueError(f"{path}:{name}: incompatible shape/payload")
            buf = np.asarray(t.data)
            original = np.asarray(w.data, dtype=np.float64).reshape(n, k)
            diff = dequantize(buf, t.tensor_type).reshape(n, k).astype(np.float64) - original
            denom = float(np.sum(original * original))
            if denom <= 0:
                raise ValueError(f"{name}: relative SSE is undefined for a zero reference tensor")
            score = float(np.sum(diff * diff)) / denom
            options = rows[name]["measurements"]
            if kind in options:
                raise ValueError(f"{name}: ambiguous duplicate {kind} candidate")
            options[kind] = {
                "payload_bytes": int(t.n_bytes),
                "relative_sse": score,
                "artifact_sha256": meta["sha256"],
                "payload_sha256": hashlib.sha256(buf).hexdigest(),
            }
    return {
        "schema": SCHEMA,
        "target": GGUF,
        "architecture": "qwen3",
        "blocks": blocks,
        "reference_sha256": ref_meta["sha256"],
        "objective": OBJECTIVE,
        "layers": rows,
    }


if __name__ == "__main__":
    from app.autotune import _dump

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _dump(args.out, gguf_table(args.reference, args.candidate))
    print(f"GGUF_PACKED_TABLE_DONE out={args.out}")
