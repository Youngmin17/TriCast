"""Full WikiText-2 and Winogrande evaluation of a pinned, local Llama snapshot.

Run on an isolated UCL GPU. Native and recipe runs share model bytes, tokenizer,
input corpus, dtype and attention implementation. No model is downloaded here.
Run costs are recorded for reproducibility, not presented as speed benchmarks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from numbers import Real
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from validate_model_families import ArithmeticTrace, bit_equal, finite, max_diff, require

from tricast import load_recipe, patch_model
from tricast.eval.envinfo import capture_env
from tricast.eval.lmeval import evaluate
from tricast.eval.ppl import _dataset_texts, perplexity
from tricast.nn.linear import EmuLinear
from tricast.nn.patch import unpatch_model


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


class ForwardLogitsWitness:
    """Check every root-model output, including all PPL and task forwards."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.checked_forwards = 0
        self.handle = None
        self.pre_handle = None
        self.trace: QualityArithmeticTrace | None = None
        self.before: list[dict[str, int] | None] = []
        self.phase = "unclassified"
        self.phase_counts: Counter[str] = Counter()
        self.checked_arithmetic_forwards = 0

    def __enter__(self) -> ForwardLogitsWitness:
        def before(module: torch.nn.Module, inputs: tuple) -> None:
            self.before.append(None if self.trace is None else dict(self.trace.layer_calls))

        def check(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
            logits = getattr(output, "logits", None)
            require(isinstance(logits, torch.Tensor), "full forward returned no logits tensor")
            finite(logits, "full model forward logits")
            previous = self.before.pop()
            if self.trace is not None:
                require(previous is not None, "missing arithmetic root-forward snapshot")
                missing = [name for name in self.trace.expected
                           if self.trace.layer_calls[name] - previous.get(name, 0) <= 0]
                require(not missing, f"arithmetic bypass: layers absent from current root forward: {missing}")
                self.checked_arithmetic_forwards += 1
            self.checked_forwards += 1
            self.phase_counts[self.phase] += 1

        self.pre_handle = self.model.register_forward_pre_hook(before)
        self.handle = self.model.register_forward_hook(check)
        return self

    def __exit__(self, *args: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        if self.pre_handle is not None:
            self.pre_handle.remove()

    def summary(self, minimum: int) -> dict:
        require(self.checked_forwards >= minimum, "missing full-model finite forward checks")
        return {"all_logits_finite": True, "checked_forwards": self.checked_forwards,
                "checked_forwards_by_phase": dict(self.phase_counts),
                "all_selected_layers_checked_per_forward": self.checked_arithmetic_forwards,
                "scope": "all root model forwards during full PPL/tasks; patched sample included"}


class QualityArithmeticTrace(ArithmeticTrace):
    """Add per-invocation dispatch proof without changing the shared tracer."""

    def __init__(self, model: torch.nn.Module, *, backend: str) -> None:
        super().__init__(model, compare=False)
        self.backend = backend
        self.kernel_calls: Counter[str] = Counter()
        self.checked_invocations: Counter[str] = Counter()
        self.extra = contextlib.ExitStack()

    def __enter__(self) -> QualityArithmeticTrace:
        super().__enter__()
        try:
            traced_matmul = EmuLinear._matmul

            def checked_matmul(layer: EmuLinear, *args: Any, **kwargs: Any) -> torch.Tensor:
                name = layer.name
                require(name in self.expected, f"untracked emulated layer: {name}")
                before = self.per_layer_algorithms.get(name, Counter()).copy()
                before_kernel = self.kernel_calls[name]
                output = traced_matmul(layer, *args, **kwargs)
                observed = self.per_layer_algorithms.get(name, Counter())
                delta = {algorithm: count - before[algorithm] for algorithm, count in observed.items()}
                algorithm = self.expected[name]
                require(delta.get(algorithm, 0) >= 1,
                        f"arithmetic bypass: {name} lacked selected {algorithm} in current invocation")
                allowed = {algorithm, "fp32_fma"} if name in self.outlier_layers else {algorithm}
                require(observed.keys() <= allowed, f"{name}: unexpected MMA algorithm")
                if self.backend == "triton":
                    require(self.kernel_calls[name] - before_kernel == sum(delta.values()),
                            f"arithmetic bypass: {name} lacked Triton dispatch in current invocation")
                self.checked_invocations[name] += 1
                return output

            self.extra.enter_context(patch.object(EmuLinear, "_matmul", checked_matmul))
            if self.backend == "triton":
                from tricast.kernels import mma as kernels

                original_kernel = kernels.gemm_triton

                def dispatched(*args: Any, **kwargs: Any) -> torch.Tensor:
                    require(self._active in self.expected, "untracked Triton MMA dispatch")
                    output = original_kernel(*args, **kwargs)
                    self.kernel_calls[self._active] += 1
                    return output

                self.extra.enter_context(patch.object(kernels, "gemm_triton", dispatched))
        except Exception:
            self.extra.close()
            super().__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        self.extra.__exit__(*args)
        super().__exit__(*args)

    def summary(self) -> dict:
        summary = super().summary()
        require(all(self.checked_invocations[name] == count for name, count in self.layer_calls.items()),
                "some emulated layer invocations lacked dispatch checks")
        return {**summary, "selected_layer_algorithms": self.expected,
                "dispatch_backend": self.backend,
                "checked_invocations_per_layer": dict(self.checked_invocations),
                "triton_kernel_dispatch_calls_per_layer": dict(self.kernel_calls),
                "dispatch_check_scope": "every selected layer invocation, no cumulative surplus"}


def evaluate_full(model: torch.nn.Module, tokenizer: Any, texts: list[str], args: Any,
                  forward_witness: ForwardLogitsWitness) -> dict:
    forward_witness.phase = "ppl"
    metrics = {"ppl": perplexity(model, tokenizer, texts=texts, seqlen=args.seqlen,
                                 batch_size=1, device=args.device)}
    require(forward_witness.phase_counts["ppl"] == metrics["ppl"]["n_windows"],
            "PPL requires exactly one checked full-model forward per complete window")
    if args.tasks:
        forward_witness.phase = "tasks"
        task_metrics = evaluate(model, tokenizer, tasks=args.tasks.split(","), batch_size=1,
                                num_fewshot=0, limit=None, log_samples=False)
        require(forward_witness.phase_counts["tasks"] > 0, "tasks executed no checked model forwards")
        for task in args.tasks.split(","):
            scores = task_metrics.get("results", {}).get(task)
            require(isinstance(scores, dict) and bool(scores), f"missing task metrics: {task}")
            numbers = [v for v in scores.values() if isinstance(v, Real) and not isinstance(v, bool)]
            require(bool(numbers) and all(math.isfinite(v) for v in numbers),
                    f"missing/nonfinite scored metrics: {task}")
            counts = task_metrics.get("n-samples", {}).get(task, {})
            require(type(counts.get("original")) is int and counts["original"] > 0
                    and counts.get("effective") == counts["original"],
                    f"task did not evaluate its full split: {task}")
        if "winogrande" in args.tasks.split(","):
            samples = task_metrics["n-samples"]["winogrande"]
            require(samples["original"] == 1267 and samples["effective"] == 1267,
                    "Winogrande must evaluate all 1267 validation examples")
        metrics["lm_eval"] = task_metrics
    require(math.isfinite(metrics["ppl"]["ppl"]), "nonfinite PPL")
    return metrics


def run(args: Any, output: Path) -> None:
    snapshot = Path(args.snapshot).resolve()
    require(Path(args.snapshot).name == args.revision,
            "snapshot directory name must equal the supplied official revision")
    require(len(args.revision) == 40 and all(c in "0123456789abcdef" for c in args.revision),
            "revision must be a 40-character Git SHA")
    config = json.loads((snapshot / "config.json").read_text())
    require(config.get("model_type") == "llama", "expected a Llama architecture")
    model_files = sorted(p for p in snapshot.iterdir() if p.is_file())
    require(any(p.suffix == ".safetensors" for p in model_files), "missing snapshot weights")
    manifest = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in model_files}
    write_json(output / "model_manifest.json", manifest)
    provenance = json.loads(Path(args.upstream_manifest).read_text())
    require(provenance["repo_id"] == args.model_id and provenance["revision"] == args.revision,
            "upstream provenance does not match requested model/revision")
    for p in model_files:
        expected = provenance["files"].get(p.name)
        require(expected is not None, f"file not present in upstream manifest: {p.name}")
        require(expected["size"] == p.stat().st_size, f"upstream size mismatch: {p.name}")
        if expected.get("sha256"):
            require(expected["sha256"] == manifest[p.name]["sha256"],
                    f"upstream LFS SHA256 mismatch: {p.name}")
        else:
            blob = f"blob {p.stat().st_size}\0".encode() + p.read_bytes()
            require(hashlib.sha1(blob).hexdigest() == expected["blob_id"],
                    f"upstream Git blob mismatch: {p.name}")
    write_json(output / "upstream_model_manifest.json", provenance)
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith("cuda"):
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "CUDA requested: isolate exactly one available GPU")
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    env = capture_env(extra={"model_id": args.model_id, "model_sha": args.revision,
        "model_kind": "official_hf_snapshot", "snapshot": str(snapshot), "seed": 42,
        "dtype": args.dtype, "device": args.device, "backend": args.backend,
        "attention_implementation": "sdpa", "command": sys.argv,
        "harness_sha256": file_hash(Path(__file__)),
        "arithmetic_trace_harness_sha256": file_hash(Path(__file__).with_name("validate_model_families.py")),
        "upstream_model_manifest_sha256": file_hash(output / "upstream_model_manifest.json"),
        "model_manifest_sha256": file_hash(output / "model_manifest.json"),
        "source_archive_sha256": args.source_archive_sha256})
    write_json(output / "env.json", env)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    texts, source_fingerprint = _dataset_texts("wikitext2", "test")
    require(len(texts) == 4358, "full WikiText-2 raw-v1 test split requires all 4358 rows")
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt")["input_ids"]
    total_windows = encoded.numel() // args.seqlen
    require(total_windows > 1, "full WikiText-2 test corpus required")
    env["dataset"] = {"id": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
        "split": "test", "source_fingerprint": source_fingerprint,
        "source_row_count": len(texts),
        "joined_text_sha256": hashlib.sha256(joined.encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(encoded.contiguous().numpy().tobytes()).hexdigest(),
        "full_token_count": encoded.numel(), "complete_windows": total_windows,
        "seqlen": args.seqlen, "window_convention": "nonoverlap; omit tail and first target"}
    write_json(output / "env.json", env)
    sample = encoded[:, :64].to(args.device)
    result: dict = {"status": "running", "scope": "full_pretrained_quality_not_performance",
                    "runs": {}, "limitations": ["attention matmuls stay native in this quality run",
                    "results apply to the pinned checkpoint and named datasets/recipes only"]}
    write_json(output / "result.json", result)
    with torch.no_grad():
        native_sample = model(sample, use_cache=False).logits.detach().clone()
    finite(native_sample, "native sample logits")
    native_metrics: dict | None = None
    for name in ["native", *(name for name in args.recipes.split(",") if name)]:
        print(f"LLAMA_QUALITY_START {name}", flush=True)
        started = time.perf_counter()
        recipe = None if name == "native" else load_recipe(name)
        record: dict = {"status": "running", "recipe": None if recipe is None else recipe.to_dict(),
                        "recipe_sha256": None if recipe is None else recipe.sha256}
        result["runs"][name] = record
        write_json(output / "result.json", result)
        try:
            if recipe is not None:
                report = patch_model(model, recipe, backend=args.backend)
                require(bool(report.patched), "recipe selected no Linear modules")
                record["patched_layers"] = [layer for layer, _ in report.patched]
            with torch.inference_mode(), ForwardLogitsWitness(model) as forward_witness:
                if recipe is None:
                    metrics = evaluate_full(model, tokenizer, texts, args, forward_witness)
                    native_metrics = metrics
                else:
                    with QualityArithmeticTrace(model, backend=args.backend) as trace:
                        forward_witness.trace = trace
                        forward_witness.phase = "sample"
                        emulated = model(sample, use_cache=False).logits
                        finite(emulated, "patched sample logits")
                        record["sample_logit_max_diff_vs_native"] = max_diff(emulated, native_sample)
                        metrics = evaluate_full(model, tokenizer, texts, args, forward_witness)
                    record["arithmetic_trace"] = trace.summary()
                record["full_forward_logits"] = forward_witness.summary(total_windows)
            if recipe is not None:
                assert native_metrics is not None
                relative = abs(metrics["ppl"]["ppl"] / native_metrics["ppl"]["ppl"] - 1)
                record["ppl_relative_difference_vs_native"] = relative
                if name == "bf16_passthrough":
                    require(relative <= 1e-3, "SPEC AC4: passthrough full PPL differs by >1e-3")
            require(metrics["ppl"]["n_windows"] == total_windows,
                    "quality run omitted complete WikiText-2 windows")
            require(metrics["ppl"]["n_tokens"] == total_windows * (args.seqlen - 1)
                    and metrics["ppl"]["dataset_fingerprint"] == env["dataset"]["joined_text_sha256"]
                    and metrics["ppl"]["forward_mode"] == "full_window",
                    "PPL token/fingerprint/forward accounting mismatch")
            record.update(status="passed", metrics=metrics)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            result["status"] = "failed"
            raise
        finally:
            unpatch_model(model)
            with torch.no_grad():
                restored = model(sample, use_cache=False).logits
            bit_equal(restored, native_sample, "native sample after unpatch")
            record["native_after_unpatch_bit_equal"] = True
            record["elapsed_seconds_not_benchmark"] = time.perf_counter() - started
            write_json(output / "result.json", result)
        print(f"LLAMA_QUALITY_PASS {name}", flush=True)
    result["status"] = "passed"
    write_json(output / "result.json", result)
    print("LLAMA_QUALITY_DONE status=passed", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--upstream-manifest", required=True,
                        help="HF model_info files_metadata at the pinned official revision")
    parser.add_argument("--recipes", default="bf16_passthrough,hopper_fp8_w8a8")
    parser.add_argument("--tasks", default="winogrande",
                        help="Comma-separated full tasks; empty for PPL only")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--backend", choices=("triton", "reference"), default="triton")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-archive-sha256")
    args = parser.parse_args()
    recipes = [name for name in args.recipes.split(",") if name]
    if "native" in recipes or len(set(recipes)) != len(recipes):
        parser.error("recipes must be distinct; native is evaluated automatically")
    for name in recipes:
        load_recipe(name)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "result.json").exists(), "refusing to overwrite an existing run")
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                run(args, output)
            except Exception as exc:
                traceback.print_exc()
                path = output / "result.json"
                result = json.loads(path.read_text()) if path.exists() else {"stage": "setup"}
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                write_json(path, result)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
