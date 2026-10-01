"""Compare quantization and accumulation choices on a single Qwen3 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[1]
SCHEMES = (
    "bf16", "fp8_tensor", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4", "nvfp4", "nvfp4_4o6",
    "msfp12", "int4_g128_zp",
)
DEFAULT_RECIPES = (
    "bf16_passthrough", "hopper_fp8_w8a8", "fp8_f7_lowacc", "fp8_f7_decoupled", "mxfp8_w_a",
    "mxfp4_w_a", "nvfp4_w_a", "nvfp4_4o6", "nvfp4_outliers", "fp8_2of4_sparse", "w4a16_g128_zp_gptq",
)
RECIPES = (*DEFAULT_RECIPES, "blackwell_fp8_w8a8", "fp8_ema_static", "mxfp4_rht")
GENERATION_RECIPES = (None, "hopper_fp8_w8a8", "fp8_f7_lowacc", "mxfp4_w_a")
PROMPTS = (
    "양자화가 무엇인지 고등학생에게 두 문장으로 설명해 주세요.",
    "Explain in two sentences why an accumulator's precision matters in a dot product.",
)
SEED = 42
SEQLEN = 2048


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda", help="single torch device, e.g. cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--quick", action="store_true", help="use at most the first 16 PPL windows")
    parser.add_argument("--max-windows", type=positive_int, help="limit PPL windows (2048 tokens each)")
    parser.add_argument("--recipes", default=",".join(DEFAULT_RECIPES), help="comma-separated PPL recipes")
    parser.add_argument("--skip-gptq", action="store_true", help="remove GPTQ from the PPL recipe list")
    parser.add_argument("--calib-samples", type=positive_int, default=32,
                        help="calibration windows for recipes that need it (GPTQ, static observers)")
    parser.add_argument("--calib-seqlen", type=positive_int, default=512,
                        help="tokens per calibration window")
    parser.add_argument("--out", type=Path, default=Path("runs") / datetime.now(timezone.utc).strftime(
        "demo_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args(argv)
    args.recipes = list(dict.fromkeys(name.strip() for name in args.recipes.split(",") if name.strip()))
    if not args.recipes or set(args.recipes) - set(RECIPES):
        parser.error(f"--recipes must select from: {', '.join(RECIPES)}")
    if args.skip_gptq:
        args.recipes = [name for name in args.recipes if name != "w4a16_g128_zp_gptq"]
    if not args.recipes:
        parser.error("no PPL recipes remain after --skip-gptq")
    if args.quick:
        args.max_windows = min(args.max_windows or 16, 16)
    return args


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def attempt(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = {"name": name, "status": "ok", **operation()}
    except Exception as exc:
        result = {"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {**result, "elapsed_s": time.perf_counter() - start}


def synchronize(device: torch.device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_digest(tensor: torch.Tensor | None) -> str | None:
    import torch

    if tensor is None:
        return None
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


class OriginalLinears:
    """Retain original modules and verify every linear weight/bias after unpatching."""

    def __init__(self, model: Any, tokenizer: Any = None, calibration: dict | None = None) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.calibration = calibration or {}
        self.layers = {
            name: (module, tensor_digest(module.weight), tensor_digest(module.bias))
            for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
        }
        self.restore_checks = 0
        self.backend = "triton" if next(model.parameters()).is_cuda else "reference"

    def restore(self) -> None:
        from tricast.nn.patch import unpatch_model

        unpatch_model(self.model)
        for name, (original, weight_hash, bias_hash) in self.layers.items():
            module = self.model.get_submodule(name)
            if (module is not original or tensor_digest(module.weight) != weight_hash
                    or tensor_digest(module.bias) != bias_hash):
                raise RuntimeError(f"original linear was not restored bit-for-bit: {name}")
        self.restore_checks += 1

    @contextmanager
    def recipe(self, name: str | None) -> Iterator[dict[str, Any]]:
        from tricast import calibrate, load_recipe, patch_model

        self.restore()
        try:
            evidence: dict[str, Any] = {"recipe": name or "native", "patched_modules": [],
                                        "backend": self.backend if name else "native_pytorch",
                                        "kernel_verified": False}
            if name is not None:
                recipe = load_recipe(name)
                evidence["recipe_sha256"] = recipe.sha256
                evidence["patch_report"] = json_value(patch_model(self.model, recipe, backend=self.backend))
                if recipe.needs_calibration:
                    # WikiText-2 train windows; the evaluated split is test.
                    evidence["calibration"] = json_value(calibrate(
                        self.model, recipe, self.tokenizer, **self.calibration))
                evidence["patched_modules"] = [
                    key for key, (original, _, _) in self.layers.items()
                    if self.model.get_submodule(key) is not original
                ]
                if not evidence["patched_modules"]:
                    raise RuntimeError("patch_model replaced no linear modules")
            calls: dict[str, int] = {}

            def count_call(key: str) -> Callable:
                def hook(_module: Any, _inputs: Any, _output: Any) -> None:
                    calls[key] = calls.get(key, 0) + 1
                return hook

            handles = [self.model.get_submodule(key).register_forward_hook(count_call(key))
                       for key in evidence["patched_modules"]]
            try:
                yield evidence
                missing = set(evidence["patched_modules"]) - calls.keys()
                if missing:
                    raise RuntimeError(f"patched linear modules never executed: {sorted(missing)}")
                evidence["forward_calls"] = calls
            finally:
                for handle in handles:
                    handle.remove()
        finally:
            self.restore()


def logical_bits(qtensor: Any) -> float:
    """Ideal format payload, not actual fp32 containers or allocator memory."""
    spec = qtensor.spec
    count = qtensor.values.numel()
    bits = count * spec.format.bits
    if qtensor.scale is not None:
        fmt = spec.scale.format
        bits += qtensor.scale.numel() * (fmt.ebits if fmt.kind == "pow2" else fmt.bits)
    if qtensor.global_scale is not None:
        bits += qtensor.global_scale.numel() * 32
    if qtensor.zero_point is not None:
        bits += qtensor.zero_point.numel() * (spec.format.bits if spec.zero_point == "int" else 32)
    return bits / count


def format_comparison(model: Any) -> dict[str, Any]:
    import torch

    from tricast import get_scheme, quantize

    name, layer = next((name, layer) for name, layer in model.named_modules()
                       if name.endswith("self_attn.q_proj"))
    weight = layer.weight.detach().float()
    backend = "triton" if weight.is_cuda else "reference"

    def measure(scheme: str) -> dict[str, Any]:
        qweight = quantize(weight, get_scheme(scheme), backend=backend)
        restored = qweight.dequantize().double()
        if not torch.isfinite(restored).all():
            raise ValueError("non-finite dequantized weights")
        delta = weight.double() - restored
        noise = delta.square().sum().item()
        signal = weight.double().square().sum().item()
        sqnr = math.inf if noise == 0 else (10 * math.log10(signal / noise) if signal else -math.inf)
        return {"sqnr_db": sqnr, "max_abs_error": delta.abs().max().item(),
                "bits_per_element": logical_bits(qweight)}

    return {"layer": name, "shape": list(weight.shape),
            "rows": [attempt(scheme, lambda scheme=scheme: measure(scheme)) for scheme in SCHEMES]}


def accumulation_comparison(model: Any, tokenizer: Any, device: torch.device) -> dict[str, Any]:
    import torch

    from tricast import gemm, get_preset, get_scheme, quantize
    from tricast.analysis import ulp_error

    layers = [(name, layer) for name, layer in model.named_modules() if name.endswith("mlp.down_proj")]
    name, layer = layers[min(10, len(layers) - 1)]
    captured = []

    def capture(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:4].clone())

    handle = layer.register_forward_pre_hook(capture)
    try:
        inputs = tokenizer("Explain why numerical precision matters.", return_tensors="pt").to(device)
        model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"activation hook did not fire: {name}")
    backend = "triton" if device.type == "cuda" else "reference"
    x = quantize(captured[0].float(), get_scheme("fp8_tensor"), backend=backend)
    weight = quantize(layer.weight[:32].detach().float(), get_scheme("fp8_tensor"), backend=backend)
    hopper = get_preset("nvidia_hopper_fp8")
    presets = [(key, get_preset(key)) for key in (
        "fp64", "nvidia_blackwell_fp8", "nvidia_hopper_fp8", "nvidia_ada_fp8")]
    presets += [("hopper_f7_fused", hopper.with_(f_bits=7, c_mode="fused")),
                ("hopper_f7_decoupled", hopper.with_(f_bits=7, c_mode="decoupled", f2_bits=23))]
    reference = gemm(x, weight, get_preset("fp64").with_(out_format="fp32"), backend=backend)
    if not torch.isfinite(reference).all():
        raise ValueError("non-finite fp64 reference output")
    reference_norm = reference.double().norm().item()
    reference_bits = reference.to(torch.bfloat16).contiguous().view(torch.int16)

    def measure(spec: Any) -> dict[str, Any]:
        spec = spec.with_(out_format="fp32")
        for _ in range(3):
            gemm(x, weight, spec, backend=backend)
        durations = []
        for _ in range(5):
            synchronize(device)
            start = time.perf_counter()
            output = gemm(x, weight, spec, backend=backend)
            synchronize(device)
            durations.append((time.perf_counter() - start) * 1000)
        if not torch.isfinite(output).all():
            raise ValueError("non-finite GEMM output")
        error = (output.double() - reference.double()).norm().item()
        relative = error / reference_norm if reference_norm else (0.0 if error == 0 else math.inf)
        bits = output.to(torch.bfloat16).contiguous().view(torch.int16)
        ulp = ulp_error(output, reference, fmt="fp32")
        ordered = sorted(durations)
        return {"relative_frobenius": relative,
                "bf16_bit_mismatch": (bits != reference_bits).double().mean().item(),
                "max_ulp_fp32": ulp["max"], "mean_ulp_fp32": ulp["mean"],
                "median_ms": statistics.median(durations), "mean_ms": statistics.mean(durations),
                "p99_ms": ordered[3] + 0.96 * (ordered[4] - ordered[3]), "samples_ms": durations,
                "mma": asdict(spec)}

    return {"layer": name, "hook_calls": len(captured), "backend": backend, "kernel_verified": False,
            "shape_mnk": [x.shape[0], weight.shape[0], x.shape[-1]],
            "rows": [attempt(key, lambda spec=spec: measure(spec)) for key, spec in presets]}


def quality_comparison(model: Any, tokenizer: Any, originals: OriginalLinears,
                       args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from tricast.eval.ppl import perplexity

    def evaluate(name: str) -> dict[str, Any]:
        with originals.recipe(name) as evidence:
            kwargs = {} if args.max_windows is None else {"max_windows": args.max_windows}
            synchronize(device)
            result = perplexity(model, tokenizer, dataset="wikitext2", seqlen=SEQLEN, **kwargs)
            synchronize(device)
            result = asdict(result) if is_dataclass(result) else result
            ppl = float(result["ppl"] if isinstance(result, dict) else result)
            if not math.isfinite(ppl) or ppl <= 0:
                raise ValueError(f"invalid perplexity: {ppl}")
        counts = {}
        if isinstance(result, dict):
            counts = {key: result.get(key) for key in ("n_windows", "n_tokens")}
        return {"ppl": ppl, **counts, "evaluation": result, "evidence": evidence, "originals_restored": True}

    return {"dataset": "wikitext2", "split": "test", "seqlen": SEQLEN, "max_windows": args.max_windows,
            "rows": [attempt(name, lambda name=name: evaluate(name)) for name in args.recipes]}


def generation_comparison(model: Any, tokenizer: Any, originals: OriginalLinears,
                          device: torch.device) -> dict[str, Any]:
    def generate(name: str | None) -> dict[str, Any]:
        texts = []
        with originals.recipe(name) as evidence:
            for prompt in PROMPTS:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
                pad_id = tokenizer.pad_token_id
                output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=48,
                                        pad_token_id=pad_id if pad_id is not None else tokenizer.eos_token_id)
                generated = output[0, inputs.input_ids.shape[1]:]
                texts.append(tokenizer.decode(generated, skip_special_tokens=True))
            synchronize(device)
        return {"texts": texts, "evidence": evidence, "originals_restored": True}

    return {"prompts": list(PROMPTS), "max_new_tokens": 48, "do_sample": False, "enable_thinking": False,
            "rows": [attempt(name or "native", lambda name=name: generate(name))
                     for name in GENERATION_RECIPES]}


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        return text.replace("&", "&amp;").replace("<", "&lt;").replace("|", "&#124;").replace("\n", "<br>")

    return "\n".join("| " + " | ".join(cell(item) for item in row) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def render_report(results: dict[str, Any]) -> str:
    sections = ["# TriCast Qwen3 demo", "Numerical design exploration, not native tensor-core throughput.",
                "## Environment", "```json\n" + json.dumps(json_value(results["environment"]),
                                                          indent=2, ensure_ascii=False) + "\n```"]
    columns = {
        "formats": [("Scheme", "name"), ("SQNR (dB)", "sqnr_db"), ("Max abs error", "max_abs_error"),
                    ("Logical bits/element", "bits_per_element")],
        "accumulation": [("MMA", "name"), ("Relative Frobenius", "relative_frobenius"),
                         ("BF16 bit mismatch (fraction)", "bf16_bit_mismatch"),
                         ("Max ULP (fp32)", "max_ulp_fp32"), ("Mean ULP (fp32)", "mean_ulp_fp32"),
                         ("Median ms", "median_ms"),
                         ("Mean ms", "mean_ms"), ("p99 ms (5 samples)", "p99_ms")],
        "quality": [("Recipe", "name"), ("PPL", "ppl"), ("Windows", "n_windows"),
                    ("Scored tokens", "n_tokens"), ("Elapsed s (single run)", "elapsed_s")],
    }
    for key, stage in results["stages"].items():
        sections.append(f"## {key}")
        if stage["status"] == "failed":
            sections.append(table(["Status", "Error"], [["failed", stage["error"]]]))
            continue
        rows = stage.get("rows", [])
        if "layer" in stage:
            shape = stage.get("shape_mnk", stage.get("shape"))
            sections.append(f"Layer: `{stage['layer']}`; shape: {shape}.")
        if key in columns:
            cols = [*columns[key], ("Status", "status"), ("Error", "error")]
            sections.append(table([title for title, _ in cols],
                                  [[row.get(field, "—") for _, field in cols] for row in rows]))
        elif key == "generation":
            sections.append(table(["Prompt", *[row["name"] for row in rows]], [
                [prompt, *[row["texts"][i] if row["status"] == "ok" else row["error"] for row in rows]]
                for i, prompt in enumerate(stage["prompts"])
            ]))
        else:
            sections.append(table(["Status", "Elapsed s"], [[stage["status"], stage["elapsed_s"]]]))
    sections += ["## Interpretation", "Logical bits include scale and zero-point payload, not tensor "
                 "containers or allocator memory. UE4M3 uses the contract's 7 logical bits; integer zero "
                 "points use the element width. This is not a physical NVFP4 packing claim.",
                 "MMA uses four activation rows and 32 output channels (or fewer), full K, FP32 outputs "
                 "and a separate BF16 bit comparison. Warmup 3, measure 5; p99 is linearly interpolated.",
                 "PPL elapsed time includes patch, evaluation and full linear-weight restoration checks; "
                 "it is one observation, not a speed comparison. Quick PPL is not full-test PPL.",
                 "Forward hooks prove patched modules ran, not that a particular GPU kernel ran. "
                 "GPU parity/profiling and silicon validation remain separate gates."]
    return "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    results: dict[str, Any] = {"environment": {
        "arguments": vars(args), "seed": SEED, "deterministic_algorithms": True,
        "mma_timing": {"warmup": 3, "repeat": 5},
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, "stages": {}}
    model = tokenizer = originals = device = None

    def setup() -> dict[str, Any]:
        nonlocal model, tokenizer, originals, device
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; no CPU fallback or model download attempted")
            import triton  # noqa: F401
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype), attn_implementation="eager",
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=getattr(model.config, "_commit_hash", None),
        )
        originals = OriginalLinears(model, tokenizer, {"samples": args.calib_samples,
                                                       "seqlen": args.calib_seqlen})
        results["environment"].update({"model": args.model,
                                        "model_commit": getattr(model.config, "_commit_hash", None),
                                        "dtype": args.dtype, "device": str(device),
                                        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
        return {"linear_count": len(originals.layers)}

    print("[setup] Load model and snapshot original linear weights", flush=True)
    results["stages"]["setup"] = attempt("setup", setup)

    def environment() -> dict[str, Any]:
        from tricast.eval.envinfo import capture_env

        model_sha = getattr(model.config, "_commit_hash", None) if model else None
        return capture_env(extra={"model_id": args.model, "model_sha": model_sha})

    results["environment"]["capture"] = attempt("environment", environment)
    operations = (
        ("formats", lambda: format_comparison(model)),
        ("accumulation", lambda: accumulation_comparison(model, tokenizer, device)),
        ("quality", lambda: quality_comparison(model, tokenizer, originals, args, device)),
        ("generation", lambda: generation_comparison(model, tokenizer, originals, device)),
    )
    for index, (name, operation) in enumerate(operations, 1):
        print(f"[{index}/5] {name}", flush=True)
        if results["stages"]["setup"]["status"] != "ok":
            results["stages"][name] = {"status": "failed", "error": "model setup failed"}
            continue
        import torch

        with torch.inference_mode():
            results["stages"][name] = attempt(name, operation)
    if originals is not None:
        results["stages"]["restore"] = attempt("restore", lambda: {"checked": originals.restore() is None})
        results["environment"]["original_weight_restore_checks"] = originals.restore_checks
    results["environment"]["dataset_fingerprints"] = sorted({
        row["evaluation"]["dataset_fingerprint"]
        for row in results["stages"]["quality"].get("rows", [])
        if row["status"] == "ok" and isinstance(row["evaluation"], dict)
        and row["evaluation"].get("dataset_fingerprint")
    })
    print("[5/5] Save results.json, report.md and env.json", flush=True)
    failed = results["environment"]["capture"]["status"] == "failed"
    failed |= any(stage["status"] == "failed"
                  or any(row["status"] == "failed" for row in stage.get("rows", []))
                  for stage in results["stages"].values())
    results["status"] = "partial_failure" if failed else "ok"
    report = render_report(results)
    for filename, value in (("results.json", results), ("env.json", results["environment"])):
        (args.out / filename).write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"DEMO_DONE status={results['status']} out={args.out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
