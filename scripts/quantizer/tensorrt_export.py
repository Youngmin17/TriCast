"""Build a TensorRT 10.x plan and compare its GPU outputs with the original ONNX on ORT CPU.

The direct route does not edit the graph. For detector compatibility testing ONLY,
--decode-conv-weights explicitly materializes static Conv kernels as FP32; never
advertise this route as preserving low-bit storage. Output directories are never reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


def ort_cpu(model: str | bytes) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
    return ort.InferenceSession(model, providers=["CPUExecutionProvider"], sess_options=opts)


def materialize_conv(path: Path, dest: Path) -> dict[str, Any]:
    """Evaluate only static operand subgraphs, not the model's data-dependent path."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(path)
    nodes = list(model.graph.node)
    initial = {ini.name: ini for ini in model.graph.initializer}
    producers = {o: i for i, n in enumerate(nodes) for o in n.output if o}
    deps = {x for n in nodes if n.op_type == "Conv" for x in n.input[1:2] if x not in initial}
    if not deps:
        raise ValueError("no computed convolution weights found")

    def walk(outputs: set[str], stop: set[str]) -> tuple[set[int], set[str]]:
        pending, visited, active = list(outputs), set(), set()
        while pending:
            name = pending.pop()
            if not name or name in visited:
                continue
            visited.add(name)
            if name in stop or name not in producers:
                continue
            idx = producers[name]
            active.add(idx)
            pending.extend(nodes[idx].input)
        return active, visited

    active, values = walk(deps, set(initial))
    if values.intersection(v.name for v in model.graph.input):
        raise ValueError("a convolution weight depends on dynamic input")
    ops = [nodes[i] for i in sorted(active)]
    if any(n.domain or n.op_type.startswith("Random") for n in ops):
        raise ValueError("refusing custom or random weight calculation")
    outs = sorted(deps)
    wgraph = helper.make_graph(
        ops,
        "static_conv_weights",
        [],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in outs],
        [v for k, v in initial.items() if k in values],
    )
    weights = helper.make_model(wgraph, opset_imports=model.opset_import)
    weights.ir_version = model.ir_version
    arrays = ort_cpu(weights.SerializeToString()).run(outs, {})
    if not all(a.dtype == np.float32 and a.ndim == 4 and np.isfinite(a).all() for a in arrays):
        raise ValueError("invalid decoded conv kernel")
    kept, consumed = walk({v.name for v in model.graph.output}, set(initial) | deps)
    del model.graph.node[:]
    model.graph.node.extend(nodes[i] for i in sorted(kept))
    del model.graph.initializer[:]
    model.graph.initializer.extend(v for k, v in initial.items() if k in consumed)
    model.graph.initializer.extend(numpy_helper.from_array(a, n) for n, a in zip(outs, arrays, strict=True))
    retained_values = [v for v in model.graph.value_info if v.name in consumed and v.name not in deps]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_values)
    onnx.checker.check_model(model)
    onnx.save(model, dest)
    return {
        "materialized_fp32_conv_weights": len(outs),
        "fp32_weight_bytes": sum(a.nbytes for a in arrays),
        "onnx_bytes": dest.stat().st_size,
        "onnx_sha256": sha256(dest),
    }


def input_cases(args: argparse.Namespace) -> list[tuple[str, dict[str, np.ndarray]]]:
    if args.model == "qwen":
        sequences = [[1, 2, 3], [8, 2, 5, 1, 1, 9, 3, 5, 2, 7, 15, 963]]
        if args.tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            sequences = [tok.encode(p) for p in ("A triangle has", "Quantization reduces weight storage.")]
        result = []
        for i, seq in enumerate(sequences):
            ids = np.asarray([seq], dtype=np.int64)
            result.append((f"prompt-{i}", {"input_ids": ids, "attention_mask": np.ones_like(ids)}))
        return result
    if args.image_npy:
        return [
            (p.stem, {"images": np.ascontiguousarray(np.load(p, allow_pickle=False), dtype=np.float32)})
            for p in args.image_npy
        ]
    rng = np.random.default_rng(42)
    return [(f"synthetic-{i}", {"images": rng.random((1, 3, 640, 640), dtype=np.float32)}) for i in range(2)]


def run(args: argparse.Namespace, report: dict) -> None:
    import onnxruntime as ort
    import tensorrt as trt
    import torch

    source, dest = args.onnx, args.out_dir
    torch.cuda.init()
    report.update(
        tensorrt=trt.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        ort=ort.__version__,
        gpu=torch.cuda.get_device_name(0),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    log = trt.Logger(trt.Logger.WARNING)
    engine_source = source
    if args.decode_conv_weights:
        if args.model != "yolo":
            raise ValueError("--decode-conv-weights is for YOLO only")
        report["stage"] = "decode"
        engine_source = dest / "explicit-dense.onnx"
        report["decoded_conv_graph"] = materialize_conv(source, engine_source)
    report["stage"] = "parse"
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, log)
    parsed = parser.parse_from_file(str(engine_source.resolve()))
    report["parser_errors"] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed:
        raise RuntimeError("TensorRT rejected the ONNX graph")
    report["stage"] = "profile"
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib << 30)
    options = builder.create_optimization_profile()
    cases = input_cases(args)
    report["profile"] = {}
    for j in range(network.num_inputs):
        inp = network.get_input(j)
        shapes = sorted(d[inp.name].shape for _, d in cases)
        lo = tuple(min(sz[axis] for sz in shapes) for axis in range(len(shapes[0])))
        hi = tuple(max(sz[axis] for sz in shapes) for axis in range(len(shapes[0])))
        bounds = (lo, shapes[0], hi)
        if args.yolo_profile:
            if args.model != "yolo" or inp.name != "images":
                raise ValueError("--yolo-profile requires a YOLO images input")
            sizes = [tuple(map(int, spec.split(","))) for spec in args.yolo_profile]
            if any(len(sz) != 3 or min(sz) < 1 or sz[1] % 32 or sz[2] % 32 for sz in sizes):
                raise ValueError("expected three B,H,W shapes with H,W multiples of 32")
            bounds = tuple((b, 3, h, w) for b, h, w in sizes)
            if not all(
                a <= v <= c
                for shape in [*shapes, bounds[1]]
                for a, v, c in zip(bounds[0], shape, bounds[2], strict=True)
            ):
                raise ValueError(f"calibration input or opt shape outside requested profile {bounds}")
        if min(inp.shape) < 0:
            # TRT 10.x set_shape() raises on bad bounds and returns None on success.
            options.set_shape(inp.name, *bounds)
            applied = tuple(tuple(v) for v in options.get_shape(inp.name))
            if applied != bounds:
                raise ValueError(f"optimization profile {inp.name}: requested {bounds}, got {applied}")
        report["profile"][inp.name] = bounds
    if any(min(network.get_input(i).shape) < 0 for i in range(network.num_inputs)):
        if config.add_optimization_profile(options) < 0:
            raise RuntimeError("failed to add optimization profile")
    report["stage"] = "build"
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("builder returned no engine; see stderr for layer diagnostics")
    plan = dest / "model.plan"
    with plan.open("xb") as stream:
        stream.write(bytes(blob))
    report.update(plan_sha256=sha256(plan), plan_bytes=plan.stat().st_size)
    del blob, parser, network, options, config, builder
    report["stage"] = "deserialize"
    runtime = trt.Runtime(log)
    engine = runtime.deserialize_cuda_engine(plan.read_bytes())
    if not engine:
        raise RuntimeError("serialized plan could not be reloaded")
    inspector = engine.create_engine_inspector()
    layers = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    (dest / "engine-inspector.json").write_text(layers)
    report["engine_layers"] = engine.num_layers
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    session = ort_cpu(str(source))
    stream = torch.cuda.current_stream()
    report["stage"] = "compare"
    report["comparisons"] = []
    for label, data in cases:
        expected = session.run(outputs, data)
        buffers = {}
        for name, value in data.items():
            if not ctx.set_input_shape(name, value.shape):
                raise ValueError(f"invalid input shape: {name} {value.shape}")
            buffers[name] = torch.from_numpy(value).cuda().contiguous()
        for name in outputs:
            dims = tuple(ctx.get_tensor_shape(name))
            if min(dims) < 0:
                raise ValueError(f"unresolved output: {name} {dims}")
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
            buffers[name] = torch.empty(dims, dtype=dtype, device="cuda")
        for name, tensor in buffers.items():
            if not ctx.set_tensor_address(name, tensor.data_ptr()):
                raise ValueError(f"cannot bind {name}")
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("GPU inference returned false")
        stream.synchronize()
        for out_name, ref in zip(outputs, expected, strict=True):
            pred = buffers[out_name].cpu().numpy()
            if ref.shape != pred.shape or not (np.isfinite(pred).all() and np.isfinite(ref).all()):
                raise ValueError(f"non-finite or incompatible {label} {out_name}")
            delta = np.abs(ref.astype(np.float64) - pred.astype(np.float64))
            ok = bool(np.allclose(ref, pred, rtol=1e-4, atol=1e-3))
            evidence = {
                "input": label,
                "output": out_name,
                "shape": list(ref.shape),
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
                "allclose": ok,
                "input_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in data.items()},
            }
            if args.model == "qwen":
                evidence["top1_last_equal"] = bool(
                    np.array_equal(ref[:, -1].argmax(-1), pred[:, -1].argmax(-1))
                )
            report["comparisons"].append(evidence)
    report["comparison_pass"] = all(
        x["allclose"] and x.get("top1_last_equal", True) for x in report["comparisons"]
    )
    report["stage"] = "complete"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "yolo"), required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--decode-conv-weights", action="store_true")
    parser.add_argument(
        "--image-npy", action="append", type=Path, help="YOLO normalized float NCHW; repeatable"
    )
    parser.add_argument("--tokenizer", type=Path, help="local HF tokenizer for Qwen prompts")
    parser.add_argument(
        "--yolo-profile",
        nargs=3,
        metavar=("MIN", "OPT", "MAX"),
        help="explicit detector engine profile; each shape is B,H,W (multiples of stride 32)",
    )
    parser.add_argument("--workspace-gib", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "model": args.model,
        "stage": "environment",
        "hostname": platform.node(),
        "python": sys.version,
        "onnx": str(args.onnx),
        "bytes": args.onnx.stat().st_size,
        "sha256": sha256(args.onnx),
        "route": "explicit-fp32-convolution" if args.decode_conv_weights else "unchanged-onnx",
        "precision": "strongly typed, TF32 disabled; engine weight bitwidth not certified",
        "thresholds": {"atol": 1e-3, "rtol": 1e-4},
        "command": sys.argv,
    }
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (args.out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if "error" in report:
        print(f"TENSORRT_EXPORT_FAILED stage={report['stage']} reason={report['error']}", flush=True)
        return 1
    if not report["comparison_pass"]:
        print("TENSORRT_PLAN_RAN_PARITY_FAILED", flush=True)
        return 2
    print("TENSORRT_PLAN_RELOAD_INFERENCE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
