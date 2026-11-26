"""Serialize a TriCast low-bit ONNX as .ort, verifying the disk payload and replaying real input NPZs."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import platform
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf.message import Message
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.InferenceSession import (
    InferenceSession as FlatSession,
)


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def external_tensors(msg: Message) -> Iterator[onnx.TensorProto]:
    """Include nested graphs and tensor-valued attributes, not only named initializers."""
    if isinstance(msg, onnx.TensorProto):
        if onnx.external_data_helper.uses_external_data(msg):
            yield msg
        return
    for field, value in msg.ListFields():
        if field.type == field.TYPE_MESSAGE:
            repeated = (
                field.is_repeated if hasattr(field, "is_repeated") else field.label == field.LABEL_REPEATED
            )
            for child in value if repeated else (value,):
                yield from external_tensors(child)


def payloads(path: Path, metadata: dict) -> tuple[dict, dict]:
    """Identify original code by exporter declaration, not by ONNX element type (INT3 is packed UINT8)."""
    declared = {}
    widened = {}
    if metadata.get("schema") == "tricast.onnx/1":
        files = metadata["files"]
        for name, row in metadata["layers"].items():
            if name not in metadata["fallback_layers"]:
                names = row["weight_initializers"]
                if len(names) != 1:
                    raise ValueError("only single-initializer weight codes supported")
                declared[names[0]] = (row["bits"], row["params"])
    else:
        files = {path.name: {"bytes": metadata["bytes"], "sha256": metadata["sha256"]}}
        for name, row in metadata["stored"].items():
            storage_bits = {"int4": 4, "int8": 8, "packed_int3": 3}[row["dtype"]]
            if storage_bits != row["bits"]:
                widened[name] = {"recipe_bits": row["bits"], "storage_bits": storage_bits}
            declared[row.get("initializer", name + "__q")] = (storage_bits, row["params"])
    if not declared:
        raise ValueError("manifest did not declare any weight codes")
    if path.name not in files:
        raise ValueError("manifest does not identify the supplied ONNX")
    for key, spec in files.items():
        part = path.parent / key
        if Path(key).name != key or not part.is_file():
            raise ValueError("invalid ONNX artifact filename in manifest")
        if part.stat().st_size != spec["bytes"] or sha(part) != spec["sha256"]:
            raise ValueError(f"ONNX artifact hash/length differs from manifest: {key}")
    model = onnx.load(path, load_external_data=False)
    init = {x.name: x for x in model.graph.initializer}
    for tensor in external_tensors(model):
        location = onnx.external_data_helper.ExternalDataInfo(tensor).location
        if location not in files or location == path.name:
            raise ValueError(f"unverified external tensor: {location}")
    onnx.external_data_helper.load_external_data_for_model(model, str(path.parent))
    signatures = {}
    for name, (bits, count) in declared.items():
        x = init[name]
        if len(x.raw_data) != (count * bits + 7) // 8:
            raise ValueError(f"declared width differs from the raw payload: {name}")
        signatures[name] = (
            int(x.data_type),
            tuple(x.dims),
            len(x.raw_data),
            hashlib.sha256(x.raw_data).digest(),
        )
    return signatures, widened


def audit(path: Path, signatures: dict) -> None:
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        if not FlatSession.InferenceSessionBufferHasIdentifier(buf, 0):
            raise ValueError("missing ORTM FlatBuffer identifier")
        g = FlatSession.GetRootAs(buf, 0).Model().Graph()
        found = set()
        for i in range(g.InitializersLength()):
            t = g.Initializers(i)
            name = t.Name().decode()
            if name not in signatures:
                continue
            dims = tuple(t.Dims(j) for j in range(t.DimsLength()))
            sig = (t.DataType(), dims, t.RawDataLength(), hashlib.sha256(t.RawDataAsNumpy()).digest())
            if sig != signatures[name]:
                raise ValueError(f"serialized weight differs: {name}")
            found.add(name)
        if found != signatures.keys():
            raise ValueError(f"{len(signatures.keys() - found)} missing weight payloads")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path, nargs="+", help="NPZ files keyed by ONNX input name")
    p.add_argument("--threads", default=4, type=int)
    a = p.parse_args()
    if a.out.exists() or a.threads < 1:
        p.error("supply a new output directory and positive threads")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sigs, widened = payloads(a.onnx, json.loads(a.manifest.read_text()))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = a.threads, 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    with tempfile.TemporaryDirectory(dir=a.out.parent) as t:
        stage = Path(t) / "stage"
        stage.mkdir()
        out = stage / "model.ort"
        opts.optimized_model_filepath = str(out)
        opts.add_session_config_entry("session.save_model_format", "ORT")
        source = ort.InferenceSession(str(a.onnx), opts, providers=["CPUExecutionProvider"])
        audit(out, sigs)
        reopen = ort.SessionOptions()
        reopen.intra_op_num_threads, reopen.inter_op_num_threads = a.threads, 1
        reopen.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        target = ort.InferenceSession(str(out), reopen, providers=["CPUExecutionProvider"])
        cases = []
        for npz in a.inputs:
            with np.load(npz, allow_pickle=False) as f:
                feeds = {k: f[k] for k in f}
            expected, actual = source.run(None, feeds), target.run(None, feeds)
            if len(expected) != len(actual):
                raise ValueError("output count mismatch")
            equality, differences = [], []
            for x, y in zip(expected, actual, strict=True):
                if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise ValueError("bad shape or non-finite output")
                equality.append(bool(np.array_equal(x, y)))
                differences.append(float(np.abs(x - y).max()))
            cases.append(
                {"input": npz.name, "sha256": sha(npz), "output_equal": equality, "max_abs": differences}
            )
        report = {
            "onnx_sha256": sha(a.onnx),
            "artifact_sha256": sha(out),
            "artifact_bytes": out.stat().st_size,
            "weight_layers_exact": len(sigs),
            "already_wider_in_source": widened,
            "weight_bytes_exact": sum(v[2] for v in sigs.values()),
            "ort": ort.__version__,
            "host": platform.node(),
            "execution_provider": target.get_providers(),
            "optimization": "disabled for conversion and comparison",
            "cases": cases,
        }
        (stage / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if not all(all(c["output_equal"]) for c in cases):
            raise ValueError(f"ORT output changed: {cases}")
        if a.out.exists():
            raise FileExistsError(a.out)
        os.rename(stage, a.out)
    print(f"ORT_EXPORT_VERIFIED {a.out} codes={len(sigs)} cases={len(cases)}")


if __name__ == "__main__":
    main()
