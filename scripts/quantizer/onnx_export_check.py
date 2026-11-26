"""True-bitwidth ONNX export of a real checkpoint on CPU (no GPU, no calibration data needed).

    python scripts/quantizer/onnx_export_check.py --model Qwen/Qwen3-0.6B --revision SHA --out runs/DIR
    python scripts/quantizer/onnx_export_check.py --model /path/to/hf --input-ids 1,2,3,4,5 --out runs/DIR

With a local checkpoint, --input-ids avoids any tokenizer or Hub dependency. Decode verification concerns
the weights, not the accumulation order of a hardware MMA. The E2E check compares finite logits using the
existing mixed-LM integration test's gates (cosine > 0.99999, max absolute error < 1e-3), and additionally
requires the same greedy continuations. A failing comparison is recorded in result.json and exits nonzero.

Applies a calibration-free mixed recipe (native INT4 / INT8, packed 3-bit integers, MXFP4, NF4, RaZeR)
and exports the patched model with
``kernel=standard`` (every layer checked bit for bit in onnxruntime before the file is written), then records:
file sizes against the FP32 / BF16 size of the same Linear weights, the ONNX logits against TriCast's emulated
logits on a few prompts, and a greedy continuation from both. Writes ``<out>/result.json`` and the ONNX files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

RECIPE = {
    "name": "onnx_true_bits_mixed",
    "description": "calibration-free mixed recipe for low-bit weight export",
    "defaults": {
        "weight": {"scheme": "int4_g128"},
        "activation": None,
        "weight_algo": "rtn",
        "mma": "fp32_fma",
    },
    "overrides": [
        {"layers": "0,-1", "weight": {"scheme": "int8_row", "mma_input": "dequant"}},
        {"match": "*.mlp.down_proj", "weight": None, "weight_algo": "nf4"},
        {"match": "*.self_attn.q_proj", "weight": {"scheme": "mxfp4", "mma_input": "dequant"}},
        {"match": "*.self_attn.k_proj", "weight": {"scheme": "mxfp4", "mma_input": "dequant"}},
        {"layers": "8-19", "modules": ["gate_proj"], "weight": {"scheme": "int4_g128_zp", "format": "uint3"}},
        {"layers": "8-19", "modules": ["up_proj"], "weight": None, "weight_algo": "razer"},
    ],
}
PROMPTS = ["The capital of France is", "Quantization reduces the size of a neural network by"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except OSError:
        return None


def _checkpoint_files(path: Path) -> dict[str, str]:
    # Hash local checkpoints (even if they have not been published to a Hub revision).
    globs = (
        "*.safetensors",
        "*.bin",
        "*.index.json",
        "config.json",
        "tokenizer*",
        "*.model",
        "vocab.*",
        "merges.txt",
        "special_tokens_map.json",
    )
    return {
        f.name: _sha(f) for f in sorted({f for pattern in globs for f in path.glob(pattern) if f.is_file()})
    }


def _storage(path: Path, capture: Any, manifest: dict) -> tuple[dict, list[str]]:
    import onnx

    # Keep large external data on disk. ORT independently loads and uses it for the forward below.
    graph = onnx.load(str(path), load_external_data=False).graph
    inits = {t.name: t for t in graph.initializer}
    # Only count initializers on a live dependency path to the logits, not unused decode scaffolding.
    parents = {out: node.input for node in graph.node for out in node.output}
    pending = [out.name for out in graph.output]
    live: set[str] = set()
    while pending:
        current = pending.pop()
        if current and current not in live:
            live.add(current)
            pending.extend(parents.get(current, ()))
    claimed: set[str] = set()
    layers, errors = {}, []
    for name, layer in capture.layers.items():
        algo = layer.spec.weight_algo.kind
        bits = 4 if algo in ("nf4", "razer") else layer.spec.weight.format.bits
        params = layer.weight.numel()
        minimum = (params * bits + 7) // 8
        info = manifest["layers"].get(name, {})
        native = info.get("encoding", "").startswith("dequantize_linear_")
        # Capture consists solely of Linear layers (one code tensor each). Read the actual mapping;
        # graph-builder collision suffixes and different native formats need no special name convention.
        keys = info.get("weight_initializers")
        if not isinstance(keys, list) or len(keys) != 1 or keys[0] in claimed:
            errors.append(f"{name}: missing, ambiguous or reused code initializer: {keys}")
            continue
        key = keys[0]
        claimed.add(key)
        t = inits.get(key)
        if t is None or key not in live:
            errors.append(f"{name}: missing or unused code initializer {key}")
            continue
        metadata = {e.key: e.value for e in t.external_data}
        size = int(metadata["length"]) if t.data_location == onnx.TensorProto.EXTERNAL else len(t.raw_data)
        dtype = onnx.TensorProto.DataType.Name(t.data_type)
        layout_ok = (
            (
                size == minimum
                and math.prod(t.dims) == params
                and dtype == info["encoding"].removeprefix("dequantize_linear_").upper()
            )
            if native
            else (size == minimum and dtype == "UINT8" and math.prod(t.dims) == size)
        )
        if (
            not layout_ok
            or info.get("bits") != bits
            or info.get("params") != params
            or info.get("weight_bytes") != size
        ):
            errors.append(f"{name}: expected {params} x {bits}-bit codes, found {size} bytes of {dtype}")
        layers[name] = {
            "bits": bits,
            "params": params,
            "code_bytes": size,
            "minimum_bytes": minimum,
            "initializer": key,
            "dtype": dtype,
        }
    if manifest.get("quantized_weight_bytes") != sum(entry["code_bytes"] for entry in layers.values()):
        errors.append("manifest total differs from the checked initializer payloads")
    return layers, errors


@torch.inference_mode()
def _compare(model: Any, session: Any, ids: torch.Tensor, steps: int) -> dict:
    cur = ids
    tricast_tokens, onnx_tokens, checks = [], [], []
    for _ in range(steps):
        mask = torch.ones_like(cur)
        ref = model(input_ids=cur, attention_mask=mask, use_cache=False).logits.float()
        out = torch.from_numpy(
            session.run(None, {"input_ids": cur.numpy(), "attention_mask": mask.numpy()})[0]
        )
        if out.shape != ref.shape or not (torch.isfinite(out).all() and torch.isfinite(ref).all()):
            checks.append(
                {"seq": cur.size(1), "passed": False, "error": "shape mismatch or non-finite logits"}
            )
            break
        diff = float((out - ref).abs().max())
        cosine = float(
            torch.nn.functional.cosine_similarity(out.flatten().double(), ref.flatten().double(), dim=0)
        )
        a, b = int(ref[0, -1].argmax()), int(out[0, -1].argmax())
        tricast_tokens.append(a)
        onnx_tokens.append(b)
        checks.append(
            {
                "seq": cur.size(1),
                "max_abs_diff": diff,
                "cosine": cosine,
                "top1_agreement": float((ref.argmax(-1) == out.argmax(-1)).float().mean()),
                "passed": cosine > 0.99999 and diff < 1e-3 and a == b,
            }
        )
        if a != b:
            break  # No subsequent logits would be conditioned on the same prefix.
        cur = torch.cat([cur, cur.new_tensor([[a]])], dim=1)
    return {
        "input_ids": ids.flatten().tolist(),
        "tricast_ids": tricast_tokens,
        "onnx_ids": onnx_tokens,
        "steps": checks,
        "passed": len(checks) == steps and all(step["passed"] for step in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", help="pin the Hub checkpoint and tokenizer to this revision")
    parser.add_argument("--recipe", type=Path, help="optional calibration-free, weight-only YAML recipe")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU and comparison-session threads")
    parser.add_argument(
        "--input-ids",
        action="append",
        metavar="ID,ID,...",
        help="compare these token IDs instead of loading a tokenizer (repeatable)",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error("--max-new-tokens and --threads must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be empty (do not mix a new run with stale success evidence)")

    import onnx
    import onnxruntime as ort
    import transformers
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tricast import load_recipe, patch_model
    from tricast import onnx_export as ox
    from tricast import pack as tp

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    local = Path(args.model).is_dir()
    if local and args.revision:
        parser.error("--revision applies to Hub IDs only; local checkpoint files are hashed")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, revision=args.revision, local_files_only=local
    ).eval()
    revision = getattr(model.config, "_commit_hash", None)
    if not local and not revision:
        raise ValueError("could not establish the resolved Hub revision")
    files = _checkpoint_files(Path(args.model)) if local else {}
    tokenizer = None
    if args.input_ids:
        try:
            inputs = [(None, torch.tensor([[int(i) for i in raw.split(",")]])) for raw in args.input_ids]
        except ValueError:
            parser.error("--input-ids takes comma-separated integer token IDs")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, local_files_only=local)
        inputs = [(prompt, tokenizer(prompt, return_tensors="pt")["input_ids"]) for prompt in PROMPTS]
    if any(ids.min().item() < 0 or ids.max().item() >= model.config.vocab_size for _, ids in inputs):
        parser.error("input token ID outside this model's vocabulary")
    recipe_text = (
        args.recipe.read_text(encoding="utf-8") if args.recipe else json.dumps(RECIPE, sort_keys=True)
    )
    recipe_dict = yaml.safe_load(recipe_text)
    recipe = load_recipe(recipe_dict)
    if recipe.needs_calibration or recipe.kv is not None:
        parser.error("this evidence harness requires a calibration-free, weight-only recipe")
    report = patch_model(model, recipe)
    capture = tp.Capture(model)
    try:
        capture.requantize_uncalibrated()
    finally:
        capture.close()
    linear_params = sum(layer.weight.numel() for layer in capture.layers.values())
    onnx_path = args.out / "model.onnx"
    tick = time.monotonic()
    manifest = ox.export_onnx(
        model,
        onnx_path,
        live_encoded=capture.encoded,
        recipe_text=recipe_text,
        model_id=args.model,
        revision=revision,
    )
    export_seconds = time.monotonic() - tick

    layers, failures = _storage(onnx_path, capture, manifest)
    decode = manifest["checks"]["decoded_equals_tricast_weight"]
    if (
        not layers
        or set(capture.layers) != set(manifest["layers"])
        or capture.fallback
        or manifest["fallback_layers"]
        or decode["mismatched"]
        or decode["checked_layers"] != len(layers)
    ):
        failures.append("not every selected weight was stored as codes and decoded bit-exactly")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    prompts = []
    for prompt, ids in inputs:
        comparison = _compare(model, session, ids, args.max_new_tokens)
        if tokenizer is not None:
            comparison.update(
                {
                    "prompt": prompt,
                    "tricast_text": tokenizer.decode(comparison["tricast_ids"]),
                    "onnx_text": tokenizer.decode(comparison["onnx_ids"]),
                }
            )
        if not comparison["passed"]:
            failures.append(f"prompt {len(prompts) + 1}: logits or greedy token parity failed")
        prompts.append(comparison)
    if not any(step["seq"] != 4 for prompt in prompts for step in prompt["steps"]):
        failures.append("no comparison used a sequence length different from the four-token tracing input")
    dirty = _git("status", "--porcelain")
    result = {
        "passed": not failures,
        "failures": failures,
        "model": args.model,
        "revision": revision,
        "local_checkpoint_sha256": files,
        "recipe": recipe_dict,
        "recipe_sha256": manifest["recipe_sha256"],
        "patched_layers": len(report.patched),
        "unused_overrides": report.unused_overrides,
        "linear_params": linear_params,
        "linear_fp32_bytes": linear_params * 4,
        "linear_bf16_bytes": linear_params * 2,
        "linear_code_bytes": sum(entry["code_bytes"] for entry in layers.values()),
        "stored_layers": layers,
        "onnx_files": manifest["files"],
        "onnx_bytes": manifest["bytes"],
        "encodings": manifest["encodings"],
        "fallback_layers": manifest["fallback_layers"],
        "decode_check": decode,
        "ort_forward_4_tokens": manifest["checks"]["ort_forward"],
        "prompts": prompts,
        "export_seconds": round(export_seconds, 1),
        "total_seconds": round(time.monotonic() - started, 1),
        "env": {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": None if dirty is None else bool(dirty),
            "source_sha256": {
                str(p.relative_to(ROOT)): _sha(p)
                for p in [Path(__file__).resolve(), *sorted((ROOT / "src/tricast").rglob("*.py"))]
            },
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
            "machine": f"{platform.system()} {platform.machine()}",
            "threads": args.threads,
            "seed": 42,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    (args.out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "onnx_bytes",
                    "linear_code_bytes",
                    "encodings",
                    "decode_check",
                    "export_seconds",
                    "failures",
                )
            },
            ensure_ascii=False,
        )
    )
    marker = "ONNX_EXPORT_CHECK_DONE" if result["passed"] else "ONNX_EXPORT_CHECK_FAIL"
    print(f"{marker} out={args.out}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
