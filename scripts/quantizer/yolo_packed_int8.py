"""Lower a TriCast DQ-weight YOLO11n to stock ONNX QLinearConv CPU inference.

Original signed Conv codes are sign-extended to S8 (not requantized); FP32
per-output-channel scales stay unchanged. U8 activations and int32 bias make
this a new inference recipe, not an exact implementation of weight-only FP32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh
from scripts.quantizer import yolo_tune as y


def session(
    model: bytes | str, threads: int, profile: Path | None = None, optimized: Path | None = None
) -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    if profile:
        o.enable_profiling = True
        o.profile_file_prefix = str(profile)
    if optimized:
        o.optimized_model_filepath = str(optimized)
    return ort.InferenceSession(model, o, providers=["CPUExecutionProvider"])


def convs(m: onnx.ModelProto, source_keys: set[str]) -> list[onnx.NodeProto]:
    selected = [n for n in m.graph.node if n.op_type == "Conv" and n.input[1] in source_keys]
    if len(selected) != len(source_keys) or {n.input[1] for n in selected} != source_keys:
        raise ValueError("baseline weight IDs do not match source Conv inputs one-to-one")
    for n in m.graph.node:
        if n.op_type == "Conv" and n.input[1] not in source_keys and "dfl" not in n.input[1]:
            raise ValueError(f"unexpected unquantized Conv: {n.name}")
    return selected


def ranges(m: onnx.ModelProto, selected: list, args: argparse.Namespace, calib: list[int]) -> tuple:
    # Instrument only the min and max scalars, not full activation tensors. The
    # same CPU thread budget applies to calibration and inference.
    names = sorted({v for n in selected for v in (n.input[0], n.output[0])})
    aug = onnx.ModelProto()
    aug.CopyFrom(m)
    del aug.graph.output[:]
    for i, name in enumerate(names):
        for kind in ("Min", "Max"):
            out = f"tricast_calib_{i}_{kind}"
            aug.graph.node.append(h.make_node("Reduce" + kind, [name], [out], keepdims=0))
            aug.graph.output.append(h.make_tensor_value_info(out, T.FLOAT, []))
    sess = session(aug.SerializeToString(), args.threads)
    lo, hi = np.zeros(len(names)), np.zeros(len(names))  # include zero, as required for Q/DQ
    source_hash = hashlib.sha256()
    for i in range(0, len(calib), 2):
        samples = []
        for image_id in calib[i : i + 2]:
            image = y.image_file(args.coco, image_id).read_bytes()
            source_hash.update(image_id.to_bytes(8, "little") + hashlib.sha256(image).digest())
            samples.append(y.square_input(image, 32).numpy())
        x = np.asarray(samples, np.float32) / 255
        values = np.asarray(sess.run(None, {"images": x})).reshape(len(names), 2)
        if not np.isfinite(values).all() or np.any(values[:, 0] > values[:, 1]):
            raise ValueError("nonfinite or reversed calibration ranges")
        lo, hi = np.minimum(lo, values[:, 0]), np.maximum(hi, values[:, 1])
    return {n: (float(a), float(b)) for n, a, b in zip(names, lo, hi, strict=True)}, source_hash.hexdigest()


def lower(m: onnx.ModelProto, selected: list, obs: dict[str, tuple]) -> dict:
    by_weight = {n.input[1]: n for n in selected}
    init = {t.name: t for t in m.graph.initializer}
    producers = {o: n for n in m.graph.node for o in n.output}
    consumers = Counter(i for n in m.graph.node for i in n.input)
    rewritten, rows, activation_params = [], {}, {}

    def array(key: str) -> np.ndarray:
        return nh.to_array(init[key])

    def put(value: Any, name: str) -> str:
        if name in init or name in producers:
            raise ValueError(f"new name collision: {name}")
        t = nh.from_array(np.asarray(value), name)
        m.graph.initializer.append(t)
        init[name] = t
        return name

    def params(name: str) -> tuple[str, str]:
        if name not in activation_params:
            lo, hi = obs[name]
            scale = np.float32((hi - lo) / 255 if hi != lo else 1.0)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid activation scale")
            zp = np.uint8(np.clip(np.rint(-lo / scale), 0, 255))
            activation_params[name] = (put(scale, name + "__a8_scale"), put(zp, name + "__a8_zp"))
        return activation_params[name]

    for n in m.graph.node:
        if n.op_type == "DequantizeLinear" and n.output[0] in by_weight:
            continue
        if n.op_type != "Conv" or n.input[1] not in by_weight:
            rewritten.append(n)
            continue
        x, w = n.input[:2]
        d = producers[w]
        if (
            d.op_type != "DequantizeLinear"
            or len(d.input) != 3
            or consumers[w] != 1
            or {a.name: h.get_attribute_value(a) for a in d.attribute} != {"axis": 0}
        ):
            raise ValueError(f"{w}: need single-use signed channel-axis DQ")
        q, s, z = map(array, d.input)
        if (
            init[d.input[0]].data_type not in (T.INT4, T.INT8)
            or q.ndim != 4
            or s.shape != (q.shape[0],)
            or z.shape != s.shape
            or np.any(z.astype(np.int8) != 0)
            or not np.isfinite(s).all()
            or s.dtype != np.float32
            or (s <= 0).any()
        ):
            raise ValueError(f"{w}: unsupported code type/zero-point/scales")
        s8 = q.astype(np.int8)
        if not np.array_equal(q.astype(np.float32), s8.astype(np.float32)):
            raise ValueError(f"{w}: widening changed original codes")
        xi, yi = params(x), params(n.output[0])
        qx, qy = w + "__a8_input", w + "__a8_output"
        rewritten.append(h.make_node("QuantizeLinear", [x, *xi], [qx], name=qx))
        wi = [put(s8, w + "__s8"), d.input[1], put(np.zeros_like(s, dtype=np.int8), w + "__s8_zp")]
        inputs = [qx, *xi, *wi, *yi]
        if len(n.input) == 3 and n.input[2]:
            bias = array(n.input[2])
            denom = float(array(xi[0])) * s.astype(np.float64)
            b = np.rint(bias / denom)
            if b.shape != s.shape or not np.isfinite(b).all() or (np.abs(b) >= 2**31).any():
                raise ValueError(f"{w}: invalid int32 bias")
            inputs.append(put(b.astype(np.int32), w + "__s32_bias"))
        new = h.make_node("QLinearConv", inputs, [qy], name=n.name + "__a8w8")
        new.attribute.extend(n.attribute)
        rewritten.extend([new, h.make_node("DequantizeLinear", [qy, *yi], [n.output[0]], name=qy + "__dq")])
        rows[new.name] = {
            "source_weight": w,
            "source_code_type": T.DataType.Name(init[d.input[0]].data_type),
            "s8_bytes": s8.nbytes,
            "scales_sha256": hashlib.sha256(s.tobytes()).hexdigest(),
        }
    del m.graph.node[:]
    m.graph.node.extend(rewritten)
    live = {k for n in rewritten for k in n.input} | {v.name for v in m.graph.output}
    kept = [v for v in m.graph.initializer if v.name in live]
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)
    onnx.checker.check_model(m)
    return rows


def audit(path: Path, profile: Path, expected: int) -> dict:
    m = onnx.load(path)
    conv = [n for n in m.graph.node if n.op_type == "QLinearConv"]
    init = {v.name: v for v in m.graph.initializer}
    if len(conv) != expected or len({n.name for n in conv}) != expected:
        raise ValueError("optimized graph lost quantized Conv")
    evidence = {}
    for n in conv:
        w = init.get(n.input[3])
        if w is None or w.data_type not in (T.UINT8, T.INT8) or len(w.dims) != 4:
            raise ValueError(f"optimized {n.name}: no static 8-bit operand")
        evidence[n.name] = {
            "weight": w.name,
            "type": T.DataType.Name(w.data_type),
            "shape": list(w.dims),
            "bytes": math.prod(w.dims),
        }
    events = [
        e
        for e in json.loads(profile.read_text())
        if e.get("cat") == "Node"
        and e.get("args", {}).get("op_name") == "QLinearConv"
        and e["name"].endswith("_kernel_time")
    ]
    recorded = {e["name"].removesuffix("_kernel_time"): e["args"]["provider"] for e in events}
    if set(recorded) != set(evidence) or set(recorded.values()) != {"CPUExecutionProvider"}:
        raise ValueError(f"partial CPU dispatch: {len(recorded)}/{len(evidence)}")
    return {
        "optimized": evidence,
        "profiled_cpu_conv": len(recorded),
        "profile_sha256": y.sha256_file(profile),
        "optimized_sha256": y.sha256_file(path),
    }


def main(a: argparse.Namespace) -> None:
    import torch

    if not a.allow_activation_quantization:
        raise ValueError("activation rounding changes: explicitly pass --allow-activation-quantization")
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise ValueError("output must be empty")
    baseline = json.loads(a.baseline.read_text())
    src = y.sha256_file(a.onnx)
    if baseline["sha256"] != src or baseline["evaluation_images"] != 4872:
        raise ValueError("baseline is not the matching 4872-image source ONNX")
    y.setup_torch(a.threads)
    annotations = a.coco / "annotations/instances_val2017.json"
    ids = y.annotation_ids(annotations)
    calib = y.calibration_ids(ids)
    heldout = y.evaluation_ids(ids, calib)
    if len(calib) != 128 or len(heldout) != 4872 or set(calib) & set(heldout):
        raise ValueError("unexpected split or calibration leakage")
    m = onnx.load(a.onnx)
    layers = convs(m, set(baseline["stored"]))
    limits, image_hash = ranges(m, layers, a, calib)
    rows = lower(m, layers, limits)
    dst, optimized = a.out / "model.onnx", a.out / "optimized-cpu.onnx"
    onnx.save(m, dst)
    for n in m.graph.node:
        if n.op_type == "Conv" and "dfl" not in n.input[1]:
            raise ValueError(f"{n.name}: learned Conv not quantized")
    sess = session(str(dst), a.threads, a.out / "dispatch", optimized)
    example = (
        y.square_input(y.image_file(a.coco, y.DEMO_IDS[0]).read_bytes(), 32).numpy()[None].astype(np.float32)
        / 255
    )
    output = sess.run(None, {"images": example})[0]
    if not np.isfinite(output).all() or output.shape[:2] != (1, 84):
        raise ValueError("invalid real-image detector output")
    trace = Path(sess.end_profiling())
    evidence = audit(optimized, trace, len(rows))
    common = {
        "source_sha256": src,
        "baseline_sha256": y.sha256_file(a.baseline),
        "sha256": y.sha256_file(dst),
        "route": "lossless signed Conv codes widened to INT8; same scales; new U8 activations and I32 bias",
        "quality_equivalence_claimed": False,
        "float_conv_exception": "fixed DFL arange decoder",
        "calib_ids": calib,
        "calib_images_sha256": image_hash,
        "calibration_minmax": limits,
        "annotation_sha256": y.sha256_file(annotations),
        "layers": rows,
        "dispatch": evidence,
        "env": {
            "node": platform.node(),
            "cpu": y.cpu_name(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "ort": ort.__version__,
            "threads": a.threads,
            "workers": a.workers,
            "cmd": sys.argv,
            "source_sha256": y.sha256_file(Path(__file__)),
        },
    }
    y.write_json(a.out / "probe.json", common)
    print(f"YOLO_A8W8_DISPATCH_PASS conv={len(rows)}", flush=True)
    del sess
    sess = session(str(dst), a.threads)
    evaluator = y.load_evaluator()
    dataset = evaluator.COCOValidation(
        a.coco / "val2017", annotations, a.out, y.EVAL["batch"], y.IMGSZ, 32, None
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=y.EVAL["batch"],
        shuffle=False,
        num_workers=a.workers,
        collate_fn=evaluator.coco_collate,
        generator=torch.Generator().manual_seed(y.SEED),
    )
    predictions, seen = [], []
    with torch.inference_mode():
        for i, (inputs, metadata) in enumerate(loader):
            x = inputs.float() / 255
            raw = torch.from_numpy(sess.run(None, {"images": x.numpy()})[0])
            if not torch.isfinite(raw).all():
                raise ValueError("nonfinite held-out predictions")
            predictions.extend(
                evaluator.serialize_predictions(
                    raw,
                    metadata,
                    tuple(x.shape),
                    dataset.category_map,
                    y.EVAL["conf"],
                    y.EVAL["iou"],
                    y.EVAL["max_det"],
                )
            )
            seen.extend(v["image_id"] for v in metadata)
            if (i + 1) % 100 == 0:
                print(f"YOLO_A8W8_COCO_PROGRESS batches={i + 1}", flush=True)
    if sorted(seen) != ids:
        raise ValueError("incomplete rectangular evaluation plan")
    metrics = evaluator.evaluate_coco(annotations, predictions, heldout, a.out / "predictions.json")
    y.write_json(
        a.out / "result.json",
        {
            **common,
            "metrics": {k: metrics[k] for k in y.METRIC_KEYS},
            "source_metrics": baseline["metrics"],
            "metric_delta": {k: metrics[k] - baseline["metrics"][k] for k in y.METRIC_KEYS},
            "evaluated_images": len(heldout),
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "heldout_sha256": hashlib.sha256(json.dumps(heldout).encode()).hexdigest(),
        },
    )
    print(f"YOLO_A8W8_COCO_MEASURED n={len(heldout)} AP={metrics['AP_50_95']}", flush=True)


def rss() -> dict[str, int]:
    """Linux process RSS and high-water mark. Both include non-weight memory."""
    result = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, val = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            n, unit = val.split()
            if unit != "kB":
                raise ValueError("unknown /proc/status memory units")
            result[key + "_bytes"] = int(n) * 1024
    if len(result) != 2:
        raise RuntimeError("VmRSS/VmHWM missing in Linux /proc/self/status")
    return result


def bench_worker(a: argparse.Namespace) -> None:
    nsamples = []
    for f in a.inputs:
        if f.suffix == ".npz":
            with np.load(f, allow_pickle=False) as z:
                x = z["images"]
        else:
            x = np.load(f, allow_pickle=False)
        if x.dtype != np.float32 or x.ndim != 4 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError(f"{f}: need NCHW float32 RGB input")
        nsamples.append((f, np.ascontiguousarray(x)))
    before = rss()
    sess = session(str(a.model), a.threads)
    loaded = rss()
    cases = []
    for file, x in nsamples:
        feed = {"images": x}
        for _ in range(3):
            sess.run(None, feed)
        warm = rss()
        times, output_hash = [], None
        for _ in range(5):
            start = time.perf_counter_ns()
            (out,) = sess.run(None, feed)
            elapsed = time.perf_counter_ns() - start
            if out.shape[:2] != (x.shape[0], 84) or not np.isfinite(out).all():
                raise ValueError("invalid YOLO output")
            digest = hashlib.sha256(out.tobytes()).hexdigest()
            if output_hash is not None and digest != output_hash:
                raise ValueError("non-deterministic output")
            output_hash = digest
            times.append(elapsed / 1e6)
            del out
        cases.append(
            {
                "input": str(file),
                "input_sha256": y.sha256_file(file),
                "shape": list(x.shape),
                "output_sha256": output_hash,
                "ms": times,
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "p99_ms": float(np.percentile(times, 99)),
                "rss_after_warmup": warm,
                "rss_after_measurements": rss(),
            }
        )
    result = {
        "model_sha256": y.sha256_file(a.model),
        "file_bytes": a.model.stat().st_size,
        "rss_with_inputs_before_session": before,
        "rss_immediately_after_session_load": loaded,
        "rss_at_end": rss(),
        "cases": cases,
        "process_id": os.getpid(),
        "host": platform.node(),
        "cpu": y.cpu_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "ort": ort.__version__,
        "threads": a.threads,
        "providers": sess.get_providers(),
    }
    y.write_json(a.out, result)
    print("YOLO_CPU_FRESH_PROCESS_MEASURED", flush=True)


def bench(a: argparse.Namespace) -> None:
    proof = json.loads(a.proof.read_text())
    if (
        y.sha256_file(a.source) != proof["source_sha256"]
        or y.sha256_file(a.packed) != proof["sha256"]
        or proof["dispatch"]["profiled_cpu_conv"] != len(proof["layers"])
        or not proof["layers"]
    ):
        raise ValueError("artifact or CPU dispatch proof mismatch")
    if a.threads < 1 or a.threads > len(os.sched_getaffinity(0)):
        raise ValueError("invalid intra-op thread budget")
    a.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for label, path in [("mixed-weight-fp32-conv", a.source), ("a8w8-qlinearconv", a.packed)]:
        out = a.out / (label + ".json")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_bench_worker",
            "--model",
            str(path),
            "--inputs",
            *map(str, a.inputs),
            "--out",
            str(out),
            "--threads",
            str(a.threads),
        ]
        subprocess.run(cmd, check=True)
        results[label] = json.loads(out.read_text())
    source, packed = results.values()
    input_key = lambda arm: [(c["input_sha256"], c["shape"]) for c in arm["cases"]]
    if input_key(source) != input_key(packed) or source["cpu_affinity"] != packed["cpu_affinity"]:
        raise ValueError("unmatched inputs or CPU affinity")
    y.write_json(
        a.out / "result.json",
        {
            "arms": results,
            "proof_sha256": y.sha256_file(a.proof),
            "code_sha256": y.sha256_file(Path(__file__)),
            "cmd": sys.argv,
            "protocol": (
                "fresh process/session; CPU EP; fixed threads; each input: 3 warmups + 5 repeats; "
                "synchronous inference only"
            ),
            "memory_scope": (
                "Linux process RSS and HWM include Python, inputs, intermediate tensors, allocator and "
                "ORT/prepack. These are not weight-only bytes."
            ),
            "median_speedup_by_input": [
                b["median_ms"] / q["median_ms"] for b, q in zip(source["cases"], packed["cases"], strict=True)
            ],
        },
    )
    print("YOLO_CPU_PAIRED_BENCH_MEASURED", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] in (["bench"], ["_bench_worker"]):
        master = sys.argv[1] == "bench"
        parser = argparse.ArgumentParser(description="Compare stock ORT CPU latency and process RSS")
        for key in ("source", "packed", "proof") if master else ("model",):
            parser.add_argument("--" + key, type=Path, required=True)
        parser.add_argument("--inputs", nargs="+", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--threads", type=int, default=8)
        (bench if master else bench_worker)(parser.parse_args(sys.argv[2:]))
        sys.exit(0)
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("onnx", "baseline", "coco", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--allow-activation-quantization", action="store_true")
    main(p.parse_args())
