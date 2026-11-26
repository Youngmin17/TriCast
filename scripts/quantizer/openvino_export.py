"""Export a TriCast ONNX to OpenVINO IR; audit the emitted constants and reopen on CPU.

Example: python scripts/quantizer/openvino_export.py --task causal-lm --onnx qwen.onnx \
    --tokenizer qwen-package --out /path/to/new-directory
Detector: --task detector --image sample1.jpg --image sample2.jpg (supply real images).
Requires numpy, openvino, onnx, onnxruntime; Pillow for detection, transformers for tokenization.
Never silently treats an expanded float IR as preserving the input's low-bit representation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import socket
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: Path, start: int = 0, size: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(start)
        remaining = path.stat().st_size - start if size is None else size
        while remaining:
            chunk = f.read(min(remaining, 1 << 22))
            if not chunk:
                raise ValueError("unexpected EOF in " + str(path))
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def code_signatures(path: Path, source_sha256: str) -> list[tuple[str, int, str]]:
    """Only TriCast's weight payloads; zero points, scales, and the dense fallback are not codes."""
    import onnx

    manifest = path.with_name(path.name + ".manifest.json")
    expected: dict[str, int] | None = None
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        if m["files"][path.name]["sha256"] != source_sha256:
            raise ValueError("ONNX bytes differ from the named TriCast manifest")
        expected = {}
        for layer in m["layers"].values():
            if layer.get("fallback"):
                continue
            names = layer["weight_initializers"]
            if len(names) != 1 or names[0] in expected:
                raise ValueError("expected one distinct weight payload per quantized layer")
            expected[names[0]] = layer["weight_bytes"]
    model = onnx.load(str(path), load_external_data=False)
    out = []
    for t in model.graph.initializer:
        if expected is not None:
            if t.name not in expected:
                continue
        elif not t.name.endswith(("__codes", "__q")) or t.name.endswith(
            ("__zp__codes", "__sc__codes", "__zpf__codes")
        ):
            continue
        if t.external_data:
            fields = {e.key: e.value for e in t.external_data}
            source = (path.parent / fields["location"]).resolve()
            if not source.is_relative_to(path.parent.resolve()):
                raise ValueError("external initializer is outside the ONNX directory")
            start = int(fields.get("offset", 0))
            size = int(fields.get("length", source.stat().st_size - start))
            sha = digest(source, start, size)
        else:
            data = t.raw_data
            if not data:
                raise ValueError(f"missing raw weight payload: {t.name}")
            size, sha = len(data), hashlib.sha256(data).hexdigest()
        out.append((t.name, size, sha))
    if expected is not None and {name: size for name, size, _ in out} != expected:
        raise ValueError("weight initializers do not match the TriCast manifest")
    if not out:
        raise ValueError("no known TriCast weight payloads; cannot audit this file")
    return out


def ir_audit(xml: Path, codes: list[tuple[str, int, str]]) -> dict[str, Any]:
    """Account for physical byte ranges, not XML precision labels or runtime plugin promises."""
    root = ET.parse(xml).getroot()
    by_id = {n.attrib["id"]: n for n in root.findall("./layers/layer")}
    parents = {
        (e.get("to-layer"), e.get("to-port")): e.get("from-layer") for e in root.findall("./edges/edge")
    }
    bins = xml.with_suffix(".bin")
    sizes = {c[1] for c in codes}
    sigs = set()
    entries: dict[tuple[int, int], str] = {}
    bytes_by_type: Counter[str] = Counter()
    for n in by_id.values():
        if n.get("type") != "Const":
            continue
        data = n.find("data")
        if data is None or "offset" not in data.attrib:
            continue
        key = (int(data.attrib["offset"]), int(data.attrib["size"]))
        if not 0 <= key[0] <= key[0] + key[1] <= bins.stat().st_size:
            raise ValueError("OpenVINO constant range outside BIN")
        if key in entries:
            continue
        typ = data.attrib["element_type"]
        entries[key] = typ
        bytes_by_type[typ] += key[1]
        if key[1] in sizes:
            sigs.add((key[1], digest(bins, *key)))

    def operand(layer: ET.Element, idx: int) -> ET.Element | None:
        p = layer.findall("input/port")
        if len(p) <= idx:
            return None
        return by_id.get(parents.get((layer.get("id"), p[idx].get("id"))))

    # Direct float constant RHS/Conv kernels prove serialized dense storage; scalar scales cannot qualify.
    floats = []
    for n in by_id.values():
        if n.get("type") not in {"MatMul", "Convolution", "GroupConvolution", "FullyConnected"}:
            continue
        rhs = operand(n, 1)
        while rhs is not None and rhs.get("type") in {"Convert", "Transpose", "Reshape"}:
            rhs = operand(rhs, 0)
        if rhs is not None and rhs.get("type") == "Const":
            d = rhs.find("data")
            if d is not None and d.get("element_type") in {"f16", "f32", "bf16", "f64"}:
                floats.append(n.attrib["name"])
    matches = [name for name, size, sha in codes if (size, sha) in sigs]
    return {
        "onnx_code_payloads": len(codes),
        "onnx_code_bytes": sum(v[1] for v in codes),
        "matched_code_payloads": len(matches),
        "matched_names": matches,
        "ir_constant_bytes": bytes_by_type,
        "float_constant_weight_ops": floats,
        "ir_bin_bytes": bins.stat().st_size,
        "classification": "original_code_bytes_present"
        if len(matches) == len(codes)
        else "not_preserved_dense_operands_found"
        if floats
        else "byte_preservation_not_established",
    }


def rewrite_pow2_unpack(m: Any) -> dict[int, int]:
    """Replace recognized byte-lane shifts with exact lookup rows, preserving the original packed bytes."""
    from onnx import TensorProto
    from onnx import helper as H
    from onnx import numpy_helper as N

    prod = {y: n for n in m.graph.node for y in n.output}
    ts = {v.name: v for v in m.graph.initializer}
    new = []
    counts = Counter()
    tables = {}
    for op in m.graph.node:
        if op.op_type != "BitwiseAnd":
            new.append(op)
            continue
        shift = prod.get(op.input[0])
        mask = ts.get(op.input[1])
        if (
            shift is None
            or mask is None
            or shift.op_type != "BitShift"
            or dict((a.name, H.get_attribute_value(a)) for a in shift.attribute).get("direction") != b"RIGHT"
        ):
            new.append(op)
            continue
        shape = prod.get(shift.input[0])
        rsh = ts.get(shift.input[1])
        if (
            shape is None
            or shape.op_type != "Reshape"
            or rsh is None
            or mask.data_type != TensorProto.UINT8
            or shape.input[0] not in ts
            or shape.input[1] not in ts
        ):
            new.append(op)
            continue
        arr = N.to_array(mask)
        if arr.size != 1 or int(arr.item()) not in (3, 15):
            new.append(op)
            continue
        bits = int(arr.item()).bit_length()
        steps = np.arange(0, 8, bits, dtype=np.uint8)
        raw = ts[shape.input[0]]
        if (
            raw.data_type != TensorProto.UINT8
            or len(raw.dims) != 1
            or not raw.name.endswith("__codes")
            or not np.array_equal(N.to_array(ts[shape.input[1]]), [-1, 1])
            or not np.array_equal(N.to_array(rsh), steps)
        ):
            new.append(op)
            continue
        table_name = f"tricast_ov_unpack_lut_{bits}"
        if bits not in tables:
            values = (np.arange(256, dtype=np.uint8)[:, None] >> steps) & np.uint8((1 << bits) - 1)
            tables[bits] = N.from_array(values, table_name)
        # Preserve the exact existing raw byte tensor, including packed group zero points.
        idx = op.output[0] + "_lut_input_i32"
        if any(name in prod or name in ts for name in (idx, table_name)):
            raise ValueError("name collision in rewritten unpack graph")
        new.extend(
            [
                H.make_node("Cast", [raw.name], [idx], to=TensorProto.INT32),
                H.make_node("Gather", [table_name, idx], list(op.output), axis=0),
            ]
        )
        counts[bits] += 1
    if not counts:
        raise ValueError("no UINT8 power-of-two unpack patterns found")
    # Retain only reachable definitions. The superseded Shift/Reshape nodes must really be absent.
    needed = {t.name for t in m.graph.output}
    live = []
    for n in reversed(new):
        if needed.intersection(n.output):
            live.append(n)
            needed.update(x for x in n.input if x)
    init = [t for t in list(m.graph.initializer) + list(tables.values()) if t.name in needed]
    del m.graph.node[:]
    m.graph.node.extend(reversed(live))
    del m.graph.initializer[:]
    m.graph.initializer.extend(init)
    return counts


def gather_split_conv_inputs(model: Any) -> list[str]:
    """Work around the CPU 2026.4.1 compressed-1x1 matcher losing the Split output index.

    Only replace a static channel segment feeding a Conv with an exactly equivalent Gather; no
    model weights, activation values or batch/spatial dimensions are changed.
    """
    import openvino.opset13 as ops

    rewritten = []
    for conv in model.get_ordered_ops():
        if conv.get_type_name() != "Convolution":
            continue
        src = conv.input_value(0)
        split = src.get_node()
        if split.get_type_name() not in {"VariadicSplit", "Split"}:
            continue
        weights = conv.input_value(1).get_partial_shape()
        if not weights.is_static or tuple(weights.to_shape()[2:]) != (1, 1):
            continue
        axis = split.input_value(1).get_node()
        if axis.get_type_name() != "Constant" or int(np.asarray(axis.get_data()).item()) != 1:
            continue
        shapes = [out.get_partial_shape() for out in split.outputs()]
        if any(not sh.rank.is_static or sh.rank.get_length() != 4 or not sh[1].is_static for sh in shapes):
            continue
        widths = [sh[1].get_length() for sh in shapes]
        index = src.get_index()
        start = sum(widths[:index])
        take = ops.gather(
            split.input_value(0),
            np.arange(start, start + widths[index], dtype=np.int64),
            np.array(1, dtype=np.int64),
        )
        conv.input(0).replace_source_output(take.output(0))
        rewritten.append(conv.get_friendly_name())
    if not rewritten:
        raise ValueError("no 1x1 convolution inputs matched a fixed channel split")
    model.validate_nodes_and_infer_types()
    return rewritten


def lm_cases(tok_path: Path) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    for tag, prompt in [
        ("short", "A triangle has"),
        (
            "long",
            "The role of quantization in machine learning "
            "is to reduce how many bits are needed to store a numerical weight while keeping "
            "the output of a trained network useful for inference.",
        ),
    ]:
        ids = tok(prompt, return_tensors="np").input_ids.astype(np.int64)
        yield tag, {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def detector_cases(paths: list[Path]) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    from PIL import Image

    for i, path in enumerate(paths):
        w = (320, 640)[i % 2]
        with Image.open(path) as im:
            im = im.convert("RGB")
            r = w / max(im.size)
            scaled = im.resize((round(im.width * r), round(im.height * r)), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (w, w), (114, 114, 114))
            canvas.paste(scaled, ((w - scaled.width) // 2, (w - scaled.height) // 2))
            arr = np.array(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
        yield f"{path.name}:{w}", {"images": np.ascontiguousarray(arr)}


def compare(ref: np.ndarray, got: np.ndarray, lm: bool) -> dict[str, Any]:
    if got.shape != ref.shape or not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise ValueError(f"non-finite outputs or shape mismatch {ref.shape}, {got.shape}")
    a, b = ref.flatten().astype("f8"), got.flatten().astype("f8")
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    out = {"shape": list(got.shape), "max_abs": float(np.abs(a - b).max()), "cosine": cos}
    if lm:
        out.update(
            top1_matches=int((got.argmax(-1) == ref.argmax(-1)).sum()),
            positions=int(ref.shape[1]),
            ort_next=int(ref[0, -1].argmax()),
            openvino_next=int(got[0, -1].argmax()),
        )
        out["strict_match"] = (
            cos > 0.99999 and out["max_abs"] < 1e-3 and out["ort_next"] == out["openvino_next"]
        )
    else:
        # Test the raw decoded YOLO head against ORT. This is NOT a new COCO mAP evaluation.
        out.update(
            box_max_abs=float(np.abs(got[:, :4] - ref[:, :4]).max()),
            class_score_max_abs=float(np.abs(got[:, 4:] - ref[:, 4:]).max()),
        )
        a1, b1 = ref[:, 4:], got[:, 4:]
        aa, bb = a1.max(1) >= 0.25, b1.max(1) >= 0.25
        out["ort_confident_anchors"], out["openvino_confident_anchors"] = int(aa.sum()), int(bb.sum())
        out["same_confident_anchors_and_classes"] = bool(
            np.array_equal(aa, bb) and np.array_equal(a1.argmax(1)[aa], b1.argmax(1)[bb])
        )
        out["strict_match"] = bool(np.allclose(got, ref, rtol=1e-5, atol=1e-5))
    return out


def _mean_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for t, label in zip(logits, labels, strict=True):
        x = t.astype("f8")
        top = x.max()
        total += float(np.log(np.exp(x - top).sum()) + top - x[label])
    return total / len(labels)


def heldout_cases(
    compiled: Any, baseline: Any, inputs: list[Path], refs_path: Path | None
) -> list[dict[str, Any]]:
    """Compare teacher-forced next-token decisions, excluding the final input (which has no label)."""
    refs = {r["input"]: r for r in json.loads(refs_path.read_text())["cases"]} if refs_path else {}
    rows = []
    seen = set()
    for path in inputs:
        with np.load(path, allow_pickle=False) as z:
            ids, mask = z["input_ids"], z["attention_mask"]
        if (
            ids.dtype != np.int64
            or ids.ndim != 2
            or ids.shape[0] != 1
            or ids.shape[1] < 2
            or mask.shape != ids.shape
            or mask.dtype != np.int64
            or not np.all(mask == 1)
        ):
            raise ValueError("expected batch=1, int64 input IDs and an all-ones int64 mask")
        sha = hashlib.sha256(ids.tobytes()).hexdigest()
        if sha in seen:
            raise ValueError("duplicate held-out input")
        seen.add(sha)
        feeds = {"input_ids": ids, "attention_mask": mask}
        rt = baseline.run(None, feeds)[0]
        ov = compiled(feeds)[compiled.output(0)]
        if (
            rt.shape != ov.shape
            or rt.shape[:2] != ids.shape
            or not (np.isfinite(rt).all() and np.isfinite(ov).all())
        ):
            raise ValueError("unexpected or nonfinite logits")
        rt, ov = rt[0, :-1], ov[0, :-1]
        rt_top, ov_top = rt.argmax(-1), ov.argmax(-1)
        labels = ids[0, 1:]

        row = {
            "input": path.name,
            "ids_sha256": sha,
            "positions": len(labels),
            "matches_vs_onnx": int((ov_top == rt_top).sum()),
            "max_abs_vs_onnx": float(np.max(np.abs(rt - ov))),
            "onnx_mean_nll": _mean_nll(rt, labels),
            "openvino_mean_nll": _mean_nll(ov, labels),
        }
        if refs_path:
            rec = refs[path.name]
            ref = np.asarray(rec["ref_top"])
            if rec["ids_sha256"] != sha or ref.shape != ov_top.shape:
                raise ValueError("TriCast reference is for a different input or positions")
            row.update(matches_vs_package=int((ov_top == ref).sum()), package_mean_nll=rec["ref_mean_nll"])
        rows.append(row)
    return rows


def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    import onnx
    import onnxruntime as ort
    import openvino as ov

    report.update(
        openvino=ov.__version__,
        ort=ort.__version__,
        onnx=onnx.__version__,
        execution_device="CPU",
        omp_threads=args.threads,
        compress_to_fp16=False,
    )
    path = args.onnx
    report.update(onnx_sha256=digest(path), onnx_bytes=path.stat().st_size)
    codes = code_signatures(path, report["onnx_sha256"])
    gc.collect()
    report["stage"] = "convert"
    report["serialization"] = args.serialization
    xml = args.out / "model.xml"
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        converter_input = path
        if args.lut_unpack:
            source_info = json.loads(path.with_name(path.name + ".manifest.json").read_text())
            if source_info.get("model") != "Qwen/Qwen3-0.6B" or source_info.get("kernel") != "standard":
                raise ValueError(
                    "--lut-unpack is validated for the standard TriCast Qwen/Qwen3-0.6B export only"
                )
            graph = onnx.load(str(path))
            report["rewritten_unpack_nodes_by_bits"] = rewrite_pow2_unpack(graph)
            converter_input = Path(tmp) / "lookup.onnx"
            onnx.save_model(
                graph,
                str(converter_input),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="data.bin",
                size_threshold=1 << 20,
            )
            del graph
            gc.collect()
        model = (
            ov.Core().read_model(str(converter_input))
            if args.serialization == "as-is"
            else ov.convert_model(str(converter_input))
        )
        if args.gather_split_conv_inputs:
            report["split_conv_gathers"] = gather_split_conv_inputs(model)
        report["stage"] = "serialize_ir"
        if args.serialization == "as-is":
            ov.serialize(model, str(xml), str(xml.with_suffix(".bin")))
        else:
            ov.save_model(model, str(xml), compress_to_fp16=False)
        del model
        gc.collect()
    report.update(xml_sha256=digest(xml), bin_sha256=digest(xml.with_suffix(".bin")))
    report["storage"] = ir_audit(xml, codes)
    report["stage"] = "reopen_compile"
    core = ov.Core()
    compiled = core.compile_model(
        core.read_model(str(xml)),
        "CPU",
        {
            "INFERENCE_PRECISION_HINT": "f32",
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
            "INFERENCE_NUM_THREADS": args.threads,
            "NUM_STREAMS": "1",
        },
    )
    report["execution_devices"] = list(compiled.get_property("EXECUTION_DEVICES"))
    report["inference_precision_hint"] = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    report["dynamic_quantization_group_size_requested"] = 0
    report["stage"] = "compare"
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = args.threads
    opt.inter_op_num_threads = 1
    baseline = ort.InferenceSession(str(path), sess_options=opt, providers=["CPUExecutionProvider"])
    lm = args.task == "causal-lm"
    results = []
    report["cases"] = results
    for label, feeds in lm_cases(args.tokenizer) if lm else detector_cases(args.image):
        row: dict[str, Any] = {"case": label, "steps": []}
        results.append(row)
        for _ in range(2 if lm else 1):
            ref = baseline.run(None, feeds)[0]
            got = compiled(feeds)[compiled.output(0)]
            row["steps"].append(
                {
                    "input_shapes": {k: list(v.shape) for k, v in feeds.items()},
                    "input_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in feeds.items()},
                    **compare(ref, got, lm),
                }
            )
            if lm:
                ids = np.append(feeds["input_ids"], ref[0, -1].argmax()).reshape(1, -1).astype(np.int64)
                feeds = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    if args.heldout:
        report["stage"] = "heldout"
        report["heldout"] = heldout_cases(compiled, baseline, args.heldout, args.reference_decisions)
        if args.reference_decisions:
            report["package_decisions_sha256"] = digest(args.reference_decisions)
    if report["xml_sha256"] != digest(xml) or report["bin_sha256"] != digest(xml.with_suffix(".bin")):
        raise ValueError("serialized IR files changed during inference")
    report["stage"] = "completed"
    report["strict_match"] = all(step["strict_match"] for case in results for step in case["steps"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new destination directory; never overwrite")
    p.add_argument("--task", choices=["causal-lm", "detector"], required=True)
    p.add_argument(
        "--serialization",
        choices=["standard", "as-is"],
        default="standard",
        help="standard: official converter; as-is: preserve imported graph before offline optimization",
    )
    p.add_argument(
        "--gather-split-conv-inputs",
        action="store_true",
        help="work around the OpenVINO 2026.4.1 CPU compressed Conv1x1/Split matcher",
    )
    p.add_argument(
        "--lut-unpack",
        action="store_true",
        help="Qwen3-0.6B as-is IR: lower 2/4-bit unpack to a byte lookup; no re-quantization",
    )
    p.add_argument(
        "--heldout", type=Path, nargs="+", help="LM .npz with input_ids and all-ones attention_mask"
    )
    p.add_argument(
        "--reference-decisions", type=Path, help="optional TriCast per-position record keyed by input SHA"
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--image", type=Path, action="append", default=[])
    args = p.parse_args()
    if args.threads < 1 or (args.task == "causal-lm" and not args.tokenizer):
        p.error("positive --threads and a local --tokenizer for causal-lm are required")
    if args.task == "detector" and not args.image:
        p.error("detector requires at least one real --image")
    if args.lut_unpack and (args.serialization != "as-is" or args.task != "causal-lm"):
        p.error("--lut-unpack requires --task causal-lm --serialization as-is")
    if (args.heldout and args.task != "causal-lm") or (args.reference_decisions and not args.heldout):
        p.error("--heldout requires causal-lm; --reference-decisions requires --heldout")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "stage": "prepare",
        "task": args.task,
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (args.out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        "OPENVINO_IR_" + ("MEASURED" if report["stage"] == "completed" else "FAILED"),
        report["stage"],
        report.get("error", ""),
        flush=True,
    )
    return 0 if report.get("strict_match") else 1


if __name__ == "__main__":
    raise SystemExit(main())
