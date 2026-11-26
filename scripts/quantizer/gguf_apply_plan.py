"""Assemble a stock little-endian GGUF from already packed bytes in same-parent artifacts.

All projections come from the measured selection; everything else (including the
embedding and tokenizer) comes from the explicitly named Q4_K_M base. The optional
general.file_type preset is omitted: this mixed file is not a Q4_K_M preset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import gguf
from app.packed_targets import GGUF, _checked, _hash_file, solve


def _by_name(reader: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors) or not tensors:
        raise ValueError("invalid GGUF tensor names")
    return tensors


def _metadata(reader: gguf.GGUFReader) -> dict[str, Any]:
    # Ignore the synthetic GGUFReader header fields, not model metadata.
    return {k: v for k, v in reader.fields.items() if not k.startswith("GGUF.") and k != "general.file_type"}


def apply(plan_path: Path, table_path: Path, base: Path, candidates: list[Path], out: Path) -> dict[str, Any]:
    if sys.byteorder != "little":
        raise ValueError("packed tensor copy requires a native little-endian host")
    manifest_path = Path(str(out) + ".manifest.json")
    if out.exists() or manifest_path.exists():
        raise ValueError("refusing to overwrite existing output or manifest")
    plan = json.loads(plan_path.read_bytes())
    raw = table_path.read_bytes()
    if plan.get("table_sha256") != hashlib.sha256(raw).hexdigest() or plan.get("target") != GGUF:
        raise ValueError("not the matching GGUF plan and table")
    resolution = plan.get("resolution_bits")
    expected = solve(
        json.loads(raw),
        GGUF,
        Fraction(plan["budget_bits"]),
        None if resolution is None else Fraction(resolution),
    )
    # Old plans that used default exact resolution did not record the optional field.
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("plan no longer matches the supplied table/budget")
    selected = expected["layers"]
    sources = {}
    for path in [base, *candidates]:
        meta = _checked(path)
        if meta.get("parent_dense_gguf_sha256") != plan["reference_sha256"]:
            raise ValueError(f"{path} does not descend from the plan's dense reference")
        reader = gguf.GGUFReader(path)
        if reader.endianess != gguf.GGUFEndian.LITTLE or "split.count" in reader.fields:
            raise ValueError("supported input: single, little-endian GGUF")
        if path == base:
            if meta.get("requantization") != "Q4_K_M":
                raise ValueError("explicit base must be a Q4_K_M artifact")
            base_sha = meta["sha256"]
        sources[meta["sha256"]] = (reader, _by_name(reader))
    source, all_tensors = sources[base_sha]
    if (
        source.get_field("general.architecture").contents() != "qwen3"
        or int(source.get_field("qwen3.block_count").contents()) != json.loads(raw)["blocks"]
    ):
        raise ValueError("base architecture differs from the table")
    if not set(selected) <= all_tensors.keys() or not all(
        s[1].keys() == all_tensors.keys() for s in sources.values()
    ):
        raise ValueError("candidate tensor names differ from base")
    tensors = dict(all_tensors)
    for name, m in selected.items():
        if m["artifact_sha256"] not in sources:
            raise ValueError(f"{name}: missing measured candidate artifact")
        t = sources[m["artifact_sha256"]][1][name]
        if (
            t.tensor_type.name != m["layout"]
            or list(map(int, t.shape[::-1])) != m["shape_nk"]
            or tuple(t.shape) != tuple(all_tensors[name].shape)
            or t.n_bytes != m["payload_bytes"]
            or hashlib.sha256(t.data).hexdigest() != m["payload_sha256"]
        ):
            raise ValueError(f"{name}: shape, type or bytes differ from measured candidate")
        tensors[name] = t
    base_fields = _metadata(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".packed-plan-") as tmp:
        stage = Path(tmp) / "model.gguf"
        writer = gguf.GGUFWriter(stage, "qwen3", endianess=source.endianess)
        try:
            # Match upstream gguf_new_metadata.py's typed KV / raw tensor copy API.
            for key, field in base_fields.items():
                if key == "general.architecture":
                    continue
                tp = field.types[0]
                sub = field.types[-1] if tp == gguf.GGUFValueType.ARRAY else None
                writer.add_key_value(key, field.contents(), tp, sub_type=sub)
            if source.alignment != 32:
                raise ValueError("non-default alignment not supported")
            for t in tensors.values():
                writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.n_bytes, t.tensor_type)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            for t in tensors.values():
                writer.write_tensor_data(t.data)
        finally:
            writer.close()
        saved = gguf.GGUFReader(stage)
        try:
            contents = _by_name(saved)
            saved_fields = _metadata(saved)
            if (
                "general.file_type" in saved.fields
                or contents.keys() != tensors.keys()
                or saved_fields.keys() != base_fields.keys()
            ):
                raise ValueError("saved metadata or names differ")
            for key, field in base_fields.items():
                if saved_fields[key].types != field.types or saved_fields[key].contents() != field.contents():
                    raise ValueError(f"{key}: changed metadata")
            for name, t in tensors.items():
                result = contents[name]
                if (
                    result.tensor_type != t.tensor_type
                    or tuple(result.shape) != tuple(t.shape)
                    or hashlib.sha256(result.data).digest() != hashlib.sha256(t.data).digest()
                ):
                    raise ValueError(f"{name}: saved bytes/shape/type changed")
        finally:
            saved.data._mmap.close()
        manifest = {
            "schema": "tricast.gguf.packed-plan/1",
            "sha256": _hash_file(stage),
            "bytes": stage.stat().st_size,
            "parent_dense_gguf_sha256": plan["reference_sha256"],
            "base_sha256": base_sha,
            "plan_sha256": _hash_file(plan_path),
            "table_sha256": plan["table_sha256"],
            "selected_tensors": len(selected),
            "untuned_tensors_from_base": len(tensors) - len(selected),
            "selected_payload_bytes": plan["payload_bytes"],
            "tensor_counts": dict(Counter(t.tensor_type.name for t in tensors.values())),
            "runtime_verified": False,
        }
        sidecar = Path(tmp) / "manifest.json"
        sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
        os.link(stage, out)
        os.link(sidecar, manifest_path)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "table", "base", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--candidate", nargs="*", type=Path, default=[])
    args = p.parse_args()
    result = apply(args.plan, args.table, args.base, args.candidate, args.out)
    print(
        f"GGUF_PACKED_PLAN_APPLIED tensors={result['selected_tensors']} "
        f"bytes={result['bytes']} sha256={result['sha256']} out={args.out}"
    )
