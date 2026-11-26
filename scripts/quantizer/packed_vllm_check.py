"""Audit Qwen3 and Llama decoder projections inside stock vLLM, TP=1, eager, no prefix caching.

CUDA events are correlated to ops *within each projection* for both prefill and
decode. Packed residency is checked on registered parameters/buffers before and
after inference. Temporary tiles/registers, activation/KV memory and quantized
ALU instructions are distinct from the persistent weight storage contract.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

ARCHITECTURES = {"qwen3": "Qwen3ForCausalLM", "llama": "LlamaForCausalLM"}
LEAVES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
MARK = "tricast_projection/"
COMPUTE = re.compile(r"marlin|machete|gem[mv]|mul_mat|mat_vec|mmvq|mmq|cutlass|xmma|s168|cublas|nvjet_", re.I)
OPS = {
    "compressed-tensors": re.compile(r"::(gptq_marlin_gemm|marlin_gemm|machete_mm|allspark_w8a16_gemm)$"),
    "gguf": re.compile(r"::ggml_mul_mat(_vec)?_a8$"),
    "dense": re.compile(r"aten::(mm|addmm|matmul|linear)$"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(4 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=1) + "\n")
    tmp.replace(path)


def projections(model: torch.nn.Module, n_layers: int) -> dict[str, torch.nn.Module]:
    layers = {name: m for name, m in model.named_modules() if name.split(".")[-1] in LEAVES}
    keys = [(int(re.search(r"(?:^|\.)layers\.(\d+)\.", n)[1]), n.split(".")[-1]) for n in layers]
    if sorted(keys) != sorted((i, n) for i in range(n_layers) for n in LEAVES):
        raise ValueError("missing, duplicated or unsupported decoder projection names")
    return layers


def resident(model: torch.nn.Module, layers: dict, kind: str) -> dict:
    rows, global_storage = {}, {}
    for t in list(model.parameters()) + list(model.buffers()):
        s = t.untyped_storage()
        global_storage[str(t.device), s.data_ptr()] = s.nbytes()
    for name, m in layers.items():
        # TP=1; dimensions are vLLM's uncompressed Linear dimensions.
        n = m.input_size * m.output_size
        tensors = dict(m.named_parameters()) | dict(m.named_buffers())
        key = {"gguf": "qweight", "compressed-tensors": "weight_packed", "dense": "weight"}[kind]
        weight = tensors.get(key)
        stores, details = {}, {}
        for k, t in tensors.items():
            s = t.untyped_storage()
            stores[str(t.device), s.data_ptr()] = s.nbytes()
            details[k] = [str(t.dtype), list(t.shape), s.nbytes()]
        ok = weight is not None and weight.device.type == "cuda"
        if ok and kind == "dense":
            ok = weight.is_floating_point() and weight.numel() == n
        elif ok:
            # Reject a compact integer view backed by a dense allocation or a
            # second registered dense copy (including a nonstandard param name).
            payload = weight.untyped_storage().nbytes()
            ok = not weight.is_floating_point() and 0 < payload < n * 2
            if kind == "compressed-tensors":
                ok = ok and payload * 2 == n
            else:
                from gguf import GGMLQuantizationType as QType

                qt = m.qweight_type
                types = list(qt.shard_weight_type.values()) if weight.shard_id else [qt.weight_type]
                ok = (
                    ok
                    and bool(types)
                    and all(QType(t).name in {"Q4_0", "Q4_K", "Q6_K", "Q8_0"} for t in types)
                )
            ok = ok and sum(stores.values()) < n * 2
        rows[name] = {
            "accepted": bool(ok),
            "params": details,
            "logical_elements": n,
            "registered_storage_bytes": sum(stores.values()),
            "method": type(m.quant_method).__name__,
            "required_packed_matmuls": (len(weight.shard_id) or 1) if kind == "gguf" and ok else 1,
        }
    return {
        "layers": rows,
        "passed": all(r["accepted"] for r in rows.values()),
        "registered_model_cuda_bytes": sum(v for (d, _), v in global_storage.items() if d.startswith("cuda")),
        "projection_registered_bytes": sum(r["registered_storage_bytes"] for r in rows.values()),
    }


def annotate(stack: ExitStack, layers: dict, counts: dict) -> None:
    import torch

    for name, module in layers.items():

        def pre(m: torch.nn.Module, args: tuple, label: str = name) -> None:
            x = args[0]
            tokens = x.numel() // x.shape[-1]
            counts[label].append(tokens)
            phase = "decode" if tokens == 1 else "prefill"
            span = torch.profiler.record_function(f"{MARK}{phase}/{label}")
            span.__enter__()
            m._tricast_audit_span = span

        def post(m: torch.nn.Module, args: tuple, result: Any) -> None:
            m._tricast_audit_span.__exit__(None, None, None)
            del m._tricast_audit_span

        stack.callback(module.register_forward_pre_hook(pre).remove)
        stack.callback(module.register_forward_hook(post, always_call=True).remove)


def dispatch(prof: Any, layers: dict, kind: str) -> dict:
    """Inspect actual Kineto CUDA kernels, not CPU op annotations alone."""
    import torch

    rows, signatures = {}, Counter()
    events = prof.events()
    for scope in events:
        if scope.device_type != torch.autograd.DeviceType.CPU or not scope.name.startswith(MARK):
            continue
        tag = scope.name[len(MARK) :]
        children = list(scope.cpu_children)
        ops_with_kernels, disallowed = [], set()
        while children:
            e = children.pop()
            children.extend(e.cpu_children)
            if kind != "dense" and (OPS["dense"].fullmatch(e.name) or "ggml_dequantize" in e.name):
                disallowed.add(e.name)
            if not OPS[kind].search(e.name):
                continue
            # The actual launch may be owned by the op itself or a nested frontend op.
            under, kernels = [e], set()
            while under:
                sub = under.pop()
                under.extend(sub.cpu_children)
                if not sub.is_legacy:
                    kernels.update(k.name for k in sub.kernels if k.duration > 0)
            # Drop casts/copies/workspace initialization. Correlation to the expected
            # operator disambiguates generic CUTLASS symbol names from dense GEMMs.
            compute = sorted(k for k in kernels if COMPUTE.search(k))
            if compute:
                ops_with_kernels.append(e.name)
                signatures.update((e.name, k) for k in compute)
        needed = (len(layers[tag.split("/", 1)[1]].qweight.shard_id) or 1) if kind == "gguf" else 1
        rows.setdefault(tag, []).append(
            {
                "ops": ops_with_kernels,
                "disallowed": sorted(disallowed),
                "passed": len(ops_with_kernels) >= needed and not disallowed,
            }
        )
    expected = {f"{p}/{n}" for p in ("prefill", "decode") for n in layers}
    return {
        "passed": set(rows) == expected and all(r["passed"] for v in rows.values() for r in v),
        "observed_scopes": len(rows),
        "expected_scopes": len(expected),
        "missing": sorted(expected - set(rows)),
        "layers": rows,
        "kernel_signatures": [
            {"op": op, "kernel": k, "calls_with_signature": v} for (op, k), v in signatures.most_common()
        ],
    }


def replay(result_file: Path, trace: Path, out: Path) -> None:
    """Re-evaluate a Kineto trace offline, without altering/repeating the run."""
    if out.exists() or result_file.resolve().parent != trace.resolve().parent:
        raise ValueError("use a new result file and inputs from the same run directory")
    record = json.loads(result_file.read_text())
    trace_digest = sha(trace)
    binding = "legacy_unbound" if "trace_sha256" not in record else "live_digest_verified"
    if binding != "legacy_unbound" and record["trace_sha256"] != trace_digest:
        raise ValueError("raw trace SHA-256 differs from trace recorded by model worker")
    print(f"VLLM_TRACE_BINDING {binding}", flush=True)
    with gzip.open(trace, "rt") if trace.suffix == ".gz" else trace.open() as f:
        events = json.load(f)["traceEvents"]
    kind = record["kind"]
    traces, signatures = {}, Counter()
    scopes = [e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith(MARK)]
    groups: dict[tuple, list] = {}
    kernels: dict[Any, set] = {}
    for e in events:
        cat, ext = e.get("cat"), e.get("args", {}).get("External id")
        if cat == "cpu_op" and e.get("ph") == "X":
            groups.setdefault((e["pid"], e["tid"]), []).append(e)
        if cat == "kernel" and ext is not None and e.get("dur", 0) > 0 and COMPUTE.search(e["name"]):
            kernels.setdefault(ext, set()).add(e["name"])
    indexed = {}
    for key, group in groups.items():
        group.sort(key=lambda v: v["ts"])
        indexed[key] = ([e["ts"] for e in group], group)
    for scope in scopes:
        tag = scope["name"][len(MARK) :]
        times, grp = indexed.get((scope["pid"], scope["tid"]), ([], []))
        ts, end = scope["ts"], scope["ts"] + scope["dur"]
        part = [
            e
            for e in grp[bisect.bisect_left(times, ts) : bisect.bisect_right(times, end)]
            if e["ts"] + e["dur"] <= end + 0.002
        ]
        bad = sorted(
            {
                e["name"]
                for e in part
                if kind != "dense" and (OPS["dense"].fullmatch(e["name"]) or "ggml_dequantize" in e["name"])
            }
        )
        confirmed = []
        for op in part:
            if not OPS[kind].search(op["name"]):
                continue
            # Kineto records the external correlation id of the launching CPU
            # op (possibly nested within the registered top-level quantized op).
            names = set()
            for child in part:
                if op["ts"] <= child["ts"] and child["ts"] + child["dur"] <= op["ts"] + op["dur"] + 0.002:
                    names.update(kernels.get(child.get("args", {}).get("External id"), ()))
            if names:
                confirmed.append(op["name"])
                signatures.update((op["name"], k) for k in names)
        # Older GGUF snapshots didn't save shard layout. For them keep the
        # stricter three/two rule, never infer fusion from a missing call.
        leaf = tag.rsplit(".", 1)[-1]
        legacy_needed = {"qkv_proj": 3, "gate_up_proj": 2}.get(leaf, 1) if kind == "gguf" else 1
        needed = record["before"]["layers"][tag.split("/", 1)[1]].get(
            "required_packed_matmuls", legacy_needed
        )
        traces.setdefault(tag, []).append(
            {"ops": confirmed, "disallowed": bad, "passed": len(confirmed) >= needed and not bad}
        )
    expected = {f"{phase}/{n}" for n in record["before"]["layers"] for phase in ("prefill", "decode")}
    dispatch_pass = (
        bool(expected) and set(traces) == expected and all(r["passed"] for v in traces.values() for r in v)
    )
    # Also bind to the original run's observed op/kernel identities and census.
    original = {(s["op"], s["kernel"]) for s in record["dispatch"]["kernel_signatures"]}
    # The original matcher did not recognize cuBLAS's Hopper nvjet symbol.
    # Only that exact, independently correlated dense signature may be new.
    extra = set(signatures) - original
    signatures_match = (
        bool(signatures)
        and original <= set(signatures)
        and all(kind == "dense" and k.startswith("nvjet_") for _, k in extra)
    )
    resident_pass = (
        record["resident_passed"] and record["before"]["passed"] and record["before"] == record["after"]
    )
    result = {
        "kind": kind,
        "input_result_sha256": sha(result_file),
        "trace_sha256": trace_digest,
        "trace_binding": binding,
        "script_sha256": sha(Path(__file__)),
        "scope": "recomputed from CPU-only scopes on existing trace",
        "original_gpu_signature_set_matches": signatures_match,
        "resident_passed": resident_pass,
        "dispatch_passed": dispatch_pass,
        "expected_scopes": len(expected),
        "observed_scopes": len(traces),
        "missing": sorted(expected - set(traces)),
        "layers": traces,
        "kernel_signatures": [
            {"op": o, "kernel": k, "observations": v} for (o, k), v in signatures.most_common()
        ],
        "exact_profiled_shape": record.get("exact_profiled_shape"),
        "passed": bool(
            dispatch_pass and signatures_match and resident_pass and record.get("exact_profiled_shape", True)
        ),
    }
    write(out, result)
    if not result["passed"]:
        raise ValueError(f"record/trace did not certify packed execution; see {out}")
    print("VLLM_DISPATCH_REPLAY_PASS", flush=True)


def main(a: argparse.Namespace) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams, __version__

    cfg = json.loads((a.tokenizer / "config.json").read_text())
    family = cfg.get("model_type")
    if family not in ARCHITECTURES or cfg.get("architectures") != [ARCHITECTURES[family]]:
        raise ValueError(
            f"not a supported stock causal-LM architecture: {family}, {cfg.get('architectures')}"
        )
    fields = (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    )
    if any(not isinstance(cfg.get(k), int) or cfg[k] < 1 for k in fields[2:]):
        raise ValueError("invalid decoder config or vocab size")
    tokenizer_hash = sha(a.tokenizer / "tokenizer.json")
    if a.kind != "gguf":
        actual = json.loads((a.model / "config.json").read_text())
        if any(actual.get(k) != cfg.get(k) for k in fields):
            raise ValueError("checkpoint architecture/dimensions/vocab differ from tokenizer config")
        if sha(a.model / "tokenizer.json") != tokenizer_hash:
            raise ValueError("checkpoint and supplied tokenizer use different serialized vocabularies")
    if a.kv_mib <= 0 or a.seqlen < 2:
        raise ValueError("invalid KV budget or corpus context")
    cases = []
    for path in a.inputs:
        with np.load(path, allow_pickle=False) as f:
            ids = f["input_ids"]
            if ids.ndim != 2 or ids.dtype.kind not in "iu" or ids.shape[1] < 2 or not ids.shape[0]:
                raise ValueError(f"{path}: need nonempty integer [B,S>=2]")
            if int(ids.min()) < 0 or int(ids.max()) >= cfg["vocab_size"]:
                raise ValueError(f"{path}: token id outside model vocabulary")
            if "attention_mask" in f and not np.all(f["attention_mask"] == 1):
                raise ValueError("padded inputs not supported")
            cases.extend((path.name, idx, row.tolist()) for idx, row in enumerate(ids))
    longest = max((ids for _, _, ids in cases), key=len)
    if a.probe_tokens is not None and not 2 <= a.probe_tokens <= len(longest):
        raise ValueError(f"--probe-tokens requires 2..{len(longest)} tokens from one supplied input")
    probe = longest[: a.probe_tokens if a.probe_tokens is not None else 128]
    files = [a.model] if a.model.is_file() else sorted(a.model.glob("*.safetensors"))
    if not files:
        raise ValueError("no checkpoint weights found")
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise ValueError("output must be empty")
    result: dict = {
        "kind": a.kind,
        "host": platform.node(),
        "torch": torch.__version__,
        "vllm": __version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "cmd": sys.argv,
        "seed": 42,
        "model": str(a.model),
        "model_sha256": {f.name: sha(f) for f in files},
        "tokenizer_sha256": tokenizer_hash,
        "decoder_config": {k: cfg[k] for k in fields},
        "inputs_sha256": {str(f): sha(f) for f in a.inputs},
        "script_sha256": sha(Path(__file__)),
        "kv_cache_bytes": a.kv_mib << 20,
        "timing_protocol": "TP1 eager B1 prefix cache off; fixed KV budget; 3 warmups + 5 measurements",
        "scope": f"{family} decoder projections; not a native INT4 ALU or total device memory audit",
    }
    model_len = max(max(len(ids) for _, _, ids in cases) + 8, a.seqlen + 1 if a.corpus_ids else 0)
    options = (
        {"load_format": "gguf", "quantization": "gguf", "hf_config_path": str(a.tokenizer)}
        if a.kind == "gguf"
        else {"load_format": "safetensors"}
    )
    eng = LLM(
        model=str(a.model),
        tokenizer=str(a.tokenizer),
        **options,
        tensor_parallel_size=1,
        seed=42,
        dtype="bfloat16",
        max_model_len=model_len,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        kv_cache_memory_bytes=a.kv_mib << 20,
        gpu_memory_utilization=0.23,
    )
    if eng.apply_model(lambda m: os.getpid()) != [os.getpid()]:
        raise RuntimeError("profiler must run in the model worker process")
    model = eng.apply_model(lambda m: m)[0]  # inproc only, no tensor serialization
    if type(model).__name__ != ARCHITECTURES[family]:
        raise ValueError(f"worker loaded unexpected model class: {type(model).__name__}")
    result["worker_model_class"] = type(model).__name__
    layers = projections(model, cfg["num_hidden_layers"])
    result["before"] = resident(model, layers, a.kind)

    def gen(ids: list[int], length: int, probs: int | None = None) -> Any:
        params = SamplingParams(
            temperature=0,
            seed=42,
            max_tokens=length,
            prompt_logprobs=probs,
            ignore_eos=True,
            detokenize=False,
        )
        req = eng.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        if not req.finished or req.prompt_token_ids != ids or len(req.outputs[0].token_ids) != length:
            raise RuntimeError("incomplete request")
        return req

    times = []
    for idx in range(8):
        if idx == 3:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        gen(probe, 8)
        torch.cuda.synchronize()
        if idx >= 3:
            times.append(1000 * (time.perf_counter() - start))
    result["timing_ms"] = {
        "input_tokens": len(probe),
        "output_tokens": 8,
        "samples": times,
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "p99": float(np.percentile(times, 99)),
    }
    result["gpu_torch_allocator"] = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "live_allocated_bytes": torch.cuda.memory_allocated(),
    }
    tokens_by_layer: dict[str, list[int]] = {name: [] for name in layers}
    with ExitStack() as stack:
        annotate(stack, layers, tokens_by_layer)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            gen(probe, 2)
            torch.cuda.synchronize()
    trace_path = a.out / "dispatch.trace.json.gz"
    prof.export_chrome_trace(str(trace_path))
    result["trace_sha256"] = sha(trace_path)
    result["dispatch"] = dispatch(prof, layers, a.kind)
    result["after"] = resident(model, layers, a.kind)
    result["resident_passed"] = result["before"]["passed"] and result["after"] == result["before"]
    result["dispatch"]["tokens_by_layer"] = tokens_by_layer
    result["exact_profiled_shape"] = all(v == [len(probe), 1] for v in tokens_by_layer.values())
    result["probe_passed"] = (
        result["resident_passed"] and result["dispatch"]["passed"] and result["exact_profiled_shape"]
    )
    write(a.out / "probe.json", result)

    def score(ids: list[int], top: bool) -> tuple[list[float], list[int]]:
        req = gen(ids, 1, 1 if top else 0)
        if len(req.prompt_logprobs) != len(ids) or req.prompt_logprobs[0] is not None:
            raise RuntimeError("missing logprobs for original prefix")
        nll, tops = [], []
        for gold, p in zip(ids[1:], req.prompt_logprobs[1:], strict=True):
            nll.append(-float(p[gold].logprob))
            if top:
                tops.append(max(p, key=lambda k: p[k].logprob))
        if not np.all(np.isfinite(nll)):
            raise RuntimeError("non-finite prompt logprobs")
        return nll, tops

    result["cases"] = []
    for source, idx, ids in cases:
        nll, tops = score(ids, True)
        result["cases"].append(
            {"input": source, "row": idx, "n_tokens": len(nll), "sum_nll": math.fsum(nll), "top": tops}
        )
    if a.corpus_ids:
        corpus = np.load(a.corpus_ids, mmap_mode="r", allow_pickle=False)
        if corpus.ndim != 1 or corpus.dtype.kind not in "iu" or len(corpus) < a.seqlen:
            raise ValueError("corpus must be 1D token IDs with at least one full window")
        sums = [
            math.fsum(score(corpus[i : i + a.seqlen].tolist(), False)[0])
            for i in range(0, len(corpus) - a.seqlen + 1, a.seqlen)
        ]
        count = len(sums) * (a.seqlen - 1)
        result["corpus"] = {
            "path": str(a.corpus_ids),
            "sha256": sha(a.corpus_ids),
            "seqlen": a.seqlen,
            "windows": len(sums),
            "tail": len(corpus) % a.seqlen,
            "n_tokens": count,
            "sum_nll": math.fsum(sums),
            "ppl": math.exp(math.fsum(sums) / count),
        }
    result["passed"] = result["probe_passed"]
    write(a.out / "result.json", result)
    if not result["passed"]:
        raise RuntimeError(f"packed execution audit did not pass; see {a.out}/result.json")
    print("VLLM_PACKED_AUDIT_PASS" if a.kind != "dense" else "VLLM_DENSE_CONTROL_PASS", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["replay"]:
        parser = argparse.ArgumentParser(description=replay.__doc__)
        for flag in ("result", "trace", "out"):
            parser.add_argument("--" + flag, required=True, type=Path)
        args = parser.parse_args(sys.argv[2:])
        replay(args.result, args.trace, args.out)
        sys.exit(0)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--kind", choices=tuple(OPS), required=True)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpus-ids", type=Path)
    p.add_argument(
        "--probe-tokens",
        type=int,
        help="exact timed/profiled prefix from longest supplied input "
        "(default: up to 128); rejects unavailable lengths or chunked projection calls",
    )
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--kv-mib", type=int, default=384)
    main(p.parse_args())
