"""CPU adversarial gate checks; NOT kernel, GPU, dataset or model-quality proof.

Uses tiny tensors and controlled dispatch stubs to reject counter-surplus bypass,
missing kernel dispatch and nonfinite logits after an initially finite forward.
Requires the Llama evaluation dependencies, but never loads a checkpoint or data.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from tricast import load_recipe, patch_model
from tricast.eval.envinfo import capture_env
from tricast.nn import linear
from tricast.nn.linear import EmuLinear
from tricast.nn.patch import unpatch_model


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_checks() -> list[dict[str, Any]]:
    import evaluate_llama_quality as llama
    import evaluate_vision_quality as vision

    import tricast

    checks: list[dict[str, Any]] = []

    def reject(label: str, callback: Callable[[], Any], message: str) -> None:
        try:
            callback()
        except AssertionError as error:
            if message not in str(error):
                raise AssertionError(f"{label}: unexpected failure: {error}") from error
            checks.append({"name": label, "passed": True, "rejection": str(error)})
        else:
            raise AssertionError(f"{label}: invalid gate accepted")

    witness = vision.DispatchWitness.__new__(vision.DispatchWitness)
    witness.expected, witness.allowed = {"probe": "fp32_fma"}, {"probe": {"fp32_fma"}}
    witness.layers, witness.kernel_calls = Counter(probe=1), Counter(probe=2)
    witness.gemms, witness.checked_batches = {"probe": Counter(fp32_fma=2)}, 0
    before = witness.snapshot()
    reject("vision_missing_current_layer", lambda: witness.check_batch(before), "not called")
    witness.layers["probe"] += 1
    reject("vision_prior_surplus_cannot_mask_gemm", lambda: witness.check_batch(before), "lacked selected")
    witness.gemms["probe"]["fp32_fma"] += 1
    reject("vision_missing_current_kernel", lambda: witness.check_batch(before), "kernel dispatch")
    witness.kernel_calls["probe"] += 1
    witness.check_batch(before)
    assert witness.checked_batches == 1
    checks.append({"name": "vision_valid_current_dispatch", "passed": True})
    before = witness.snapshot()
    witness.layers["probe"] += 1
    witness.gemms["probe"].update(fp32_fma=1, fp64=1)
    witness.kernel_calls["probe"] += 2
    reject("vision_unexpected_algorithm", lambda: witness.check_batch(before), "unexpected MMA")

    class ToyLM(nn.Module):
        nonfinite = False

        def forward(self, tokens: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=torch.full((1, 2, 4), float("nan") if self.nonfinite else 0.))

    toy = ToyLM()
    with torch.inference_mode(), llama.ForwardLogitsWitness(toy) as forwards:
        toy(torch.zeros(1))
        assert forwards.summary(1)["checked_forwards"] == 1
        reject("llama_missing_finite_forward_count", lambda: forwards.summary(2), "missing full-model")
        toy.nonfinite = True
        reject("llama_nonfinite_nonsample_forward", lambda: toy(torch.zeros(1)), "nonfinite")
    assert not toy._forward_hooks
    checks.append({"name": "llama_finite_hook_removed", "passed": True})

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 2, bias=False)).eval()
    patch_model(model, load_recipe("bf16_passthrough"), backend="reference")
    fake_kernel = ModuleType("tricast.kernels.mma")
    fake_kernel.gemm_triton = lambda a, b, spec, **kw: torch.zeros(a.rows, b.rows)
    fake_package = ModuleType("tricast.kernels")
    fake_package.__path__ = []
    fake_package.mma = fake_kernel
    mode = {"dispatch": True}

    def fake_gemm(a: Any, b: Any, spec: Any, **kwargs: Any) -> torch.Tensor:
        return (fake_kernel.gemm_triton(a, b, spec) if mode["dispatch"] else torch.zeros(a.rows, b.rows))

    original_matmul = EmuLinear._matmul
    try:
        with (
            patch.dict(sys.modules, {"tricast.kernels": fake_package, "tricast.kernels.mma": fake_kernel}),
            patch.object(tricast, "kernels", fake_package, create=True),
            patch.object(linear, "gemm", fake_gemm),
        ):
            with torch.inference_mode(), llama.QualityArithmeticTrace(model, backend="triton") as trace:
                model(torch.ones(1, 4))
                trace.summary()
                mode["dispatch"] = False
                reject("llama_finite_fake_gemm_without_kernel", lambda: model(torch.ones(1, 4)),
                       "Triton dispatch")
        with patch.object(EmuLinear, "_matmul", return_value=torch.zeros(1, 2)):
            with torch.inference_mode(), llama.QualityArithmeticTrace(model, backend="reference") as trace:
                trace.per_layer_algorithms["0"] = Counter(fp32_fma=2)
                trace.layer_calls["0"] = 1
                reject("llama_prior_surplus_cannot_mask_invocation", lambda: model(torch.ones(1, 4)),
                       "lacked selected")
    finally:
        unpatch_model(model)
    assert EmuLinear._matmul is original_matmul
    checks.append({"name": "llama_dispatch_patch_removed", "passed": True})

    class ProjectedToy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self.bypass = False

        def forward(self, tokens: torch.Tensor) -> SimpleNamespace:
            logits = (torch.nn.functional.linear(tokens, self.proj.weight) if self.bypass
                      else self.proj(tokens))
            return SimpleNamespace(logits=logits[:, None, :])

    projected = ProjectedToy().eval()
    patch_model(projected, load_recipe("bf16_passthrough"), backend="reference")
    try:
        with torch.inference_mode(), llama.ForwardLogitsWitness(projected) as root:
            with llama.QualityArithmeticTrace(projected, backend="reference") as trace:
                root.trace, root.phase = trace, "sample"
                projected(torch.ones(1, 4))
                root.phase, projected.bypass = "ppl", True
                reject("llama_sample_cannot_mask_later_whole_layer_bypass",
                       lambda: projected(torch.ones(1, 4)), "absent from current root forward")
                assert root.phase_counts == {"sample": 1}
    finally:
        unpatch_model(projected)
    assert not projected._forward_hooks and not projected._forward_pre_hooks
    checks.append({"name": "llama_root_snapshot_hooks_removed", "passed": True})

    def evaluate_case(ppl_calls: int, task_calls: int, effective: int = 1267,
                      original: int = 1267, score: float = .5) -> dict:
        native = ToyLM().eval()

        def fake_ppl(model: nn.Module, *args: Any, **kwargs: Any) -> dict:
            for _ in range(ppl_calls):
                model(torch.zeros(1))
            return {"ppl": 2., "n_windows": 2}

        def fake_tasks(model: nn.Module, *args: Any, **kwargs: Any) -> dict:
            for _ in range(task_calls):
                model(torch.zeros(1))
            return {"results": {"winogrande": {"acc,none": score}},
                    "n-samples": {"winogrande": {"original": original, "effective": effective}}}

        args = SimpleNamespace(seqlen=4, device="cpu", tasks="winogrande")
        with patch.object(llama, "perplexity", fake_ppl), patch.object(llama, "evaluate", fake_tasks):
            with torch.inference_mode(), llama.ForwardLogitsWitness(native) as root:
                metrics = llama.evaluate_full(native, None, ["stub corpus"], args, root)
                assert root.phase_counts == {"ppl": 2, "tasks": task_calls}
                return metrics

    reject("llama_missing_ppl_window_forward", lambda: evaluate_case(1, 1), "exactly one")
    reject("llama_extra_ppl_window_forward", lambda: evaluate_case(3, 1), "exactly one")
    reject("llama_tasks_no_model_forward", lambda: evaluate_case(2, 0), "no checked model")
    reject("llama_partial_task_split", lambda: evaluate_case(2, 1, 1), "full split")
    reject("llama_wrong_winogrande_split_count", lambda: evaluate_case(2, 1, 1266, 1266), "all 1267")
    reject("llama_nonfinite_task_metric", lambda: evaluate_case(2, 1, score=float("nan")), "nonfinite scored")
    evaluate_case(2, 1)
    checks.append({"name": "llama_valid_phase_and_sample_accounting", "passed": True,
                   "scope": "stub evaluator verifies gate logic, not real dataset quality"})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any((args.out / name).exists() for name in ("env.json", "result.json")):
        parser.error("choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "running", "scope": __doc__}
    env = capture_env(extra={"command": sys.argv, "scope": __doc__, "seed": 42,
        "harness_sha256": digest(Path(__file__)),
        "reviewed_harnesses": {name: digest(Path(__file__).with_name(name)) for name in (
            "evaluate_vision_quality.py", "evaluate_llama_quality.py", "validate_model_families.py")}})
    (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    with (args.out / "stdout.log").open("w") as stdout, (args.out / "stderr.log").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                result.update(status="passed", checks=run_checks())
            except Exception as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
