"""Exact attention arithmetic and HF eager integration validation (no downloads).

Run on UCL with HF Transformers 4.55.2. Outputs verify explicit QK/PV matrix
semantics and random tiny Llama/Qwen3 integration, not pretrained model quality::

    python scripts/e2e/validate_attention.py --device cpu --backend reference --out runs/<UTC>/attn_cpu
    CUDA_VISIBLE_DEVICES=0 python scripts/e2e/validate_attention.py \
        --device cuda --backend triton --out runs/<UTC>/attn_gpu
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from tricast import FP8_E4M3, MMASpec, ObserverSpec, QuantSpec, ScaleSpec, load_recipe, quantize
from tricast.eval.envinfo import capture_env
from tricast.mma.api import as_operand, gemm
from tricast.nn import (
    AttentionSpec,
    emulated_attention,
    iter_emuattention,
    patch_attention,
    unpatch_attention,
)
from tricast.nn import attention as attention_module
from tricast.nn.patch import patch_model, unpatch_model
from tricast.quant.spec import TransformSpec, WeightAlgoSpec
from tricast.quant.structure import OutlierSpec, SparsitySpec
from tricast.recipe import LinearSpec

RECIPES = ("fp8_f7_lowacc", "mxfp4_w_a", "int8_row_w8a8", "bf16_passthrough", "fp64_reference")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def bits_equal(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    require(left.shape == right.shape and left.dtype == right.dtype, f"{label}: shape/dtype mismatch")
    a, b = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    require(
        torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)),
        f"{label}: bit mismatch",
    )


def tensor_hash(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tensor.shape, tensor.dtype)).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    return hashlib.sha256(
        json.dumps(
            [(name, tensor_hash(tensor)) for name, tensor in sorted(model.state_dict().items())]
        ).encode()
    ).hexdigest()


class GemmTrace:
    """Compare every actual quantized QK/PV GEMM to CPU reference bit for bit."""

    def __init__(self) -> None:
        self.algorithms: Counter[str] = Counter()
        self.shapes: Counter[str] = Counter()
        self.parity_calls = 0
        self.exact_difference = 0.0
        self.context: Any = None

    def __enter__(self) -> GemmTrace:
        original = attention_module.gemm

        def tracked(a: Any, b: Any, spec: MMASpec, **kwargs: Any) -> torch.Tensor:
            output = original(a, b, spec, **kwargs)
            require(bool(torch.isfinite(output).all()), "attention GEMM output nonfinite")
            operands = []
            for value in (a, b):
                operand = as_operand(value)
                operands.append(
                    replace(
                        operand,
                        values=operand.values.detach().cpu(),
                        scale=None if operand.scale is None else operand.scale.detach().cpu(),
                        alpha=None if operand.alpha is None else operand.alpha.detach().cpu(),
                    )
                )
            reference = gemm(*operands, spec, backend="reference")
            bits_equal(output, reference, f"{spec.algorithm} same-operand GEMM")
            exact = gemm(*operands, MMASpec("fp64", out_format=spec.out_format), backend="reference")
            self.exact_difference = max(
                self.exact_difference, float((reference.double() - exact.double()).abs().max())
            )
            self.parity_calls += 1
            self.algorithms[spec.algorithm] += 1
            self.shapes[str((operands[0].values.shape, operands[1].values.shape))] += 1
            return output

        self.context = patch.object(attention_module, "gemm", tracked)
        self.context.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.context.__exit__(*args)

    def result(self, algorithm: str | tuple[str, ...], calls: int | None = None) -> dict:
        require(self.parity_calls > 0, "attention arithmetic bypass: no emulated GEMM")
        expected = {algorithm} if isinstance(algorithm, str) else set(algorithm)
        require(set(self.algorithms) == expected, "attention: wrong selected accumulator algorithm")
        if calls is not None:
            require(
                self.parity_calls == calls,
                f"attention: expected {calls} QK/PV GEMMs, got {self.parity_calls}",
            )
        return {
            "reference_bit_parity_calls": self.parity_calls,
            "reference_max_diff": 0.0,
            "algorithms": dict(self.algorithms),
            "matrix_shapes": dict(self.shapes),
            "same_quant_operands_fp64_max_diff": self.exact_difference,
            "bypass_detected": False,
        }


def additive_mask(batch: int, heads: int, queries: int, keys: int, device: str) -> torch.Tensor:
    query_positions = torch.arange(queries, device=device) + keys - queries
    key_positions = torch.arange(keys, device=device)
    masked = key_positions[None, :] > query_positions[:, None]
    return torch.zeros(batch, heads, queries, keys, device=device).masked_fill(masked, float("-inf"))


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: AttentionSpec,
    mask: torch.Tensor | None,
    groups: int,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-reference quant/GEMM oracle; native scaling/mask/softmax stay on the target device."""

    def reference_matrix(a: torch.Tensor, b: torch.Tensor, linear_spec: LinearSpec) -> torch.Tensor:
        a, b = a.detach().cpu(), b.detach().cpu()
        a_op = (
            a if linear_spec.activation is None else quantize(a, linear_spec.activation, backend="reference")
        )
        b_op = b if linear_spec.weight is None else quantize(b, linear_spec.weight, backend="reference")
        return gemm(a_op, b_op, linear_spec.mma, backend="reference").to(q.device)

    scores = torch.empty((*q.shape[:-1], k.shape[-2]), device=q.device, dtype=q.dtype)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            kv_head = head // groups
            scores[batch, head] = reference_matrix(q[batch, head], k[batch, kv_head], spec.qk).to(q.dtype)
    scores = scores * q.shape[-1] ** -0.5
    if mask is not None:
        scores = scores + mask[..., : k.shape[-2]]
    probability = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.empty_like(q)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            output[batch, head] = reference_matrix(
                probability[batch, head],
                v[batch, head // groups].T,
                spec.pv,
            ).to(q.dtype)
    return output.transpose(1, 2).contiguous(), probability


def validate_matrices(device: str, backend: str) -> list[dict]:
    results = []
    for recipe_name in RECIPES:
        spec = AttentionSpec(load_recipe(recipe_name).defaults, load_recipe(recipe_name).defaults)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for queries, keys, use_mask in ((3, 3, True), (2, 6, True), (1, 7, False)):
                torch.manual_seed(42)
                q = torch.randn(2, 4, queries, 8, device=device, dtype=dtype)
                k = torch.randn(2, 2, keys, 8, device=device, dtype=dtype)
                v = torch.randn_like(k)
                mask = additive_mask(2, 4, queries, keys, device) if use_mask else None
                if mask is not None and keys > queries:
                    mask[0, :, :, 0] = float("-inf")  # padding plus cached causal positions
                with GemmTrace() as trace:
                    actual, probabilities = emulated_attention(
                        q,
                        k,
                        v,
                        spec,
                        scaling=8**-0.5,
                        num_key_value_groups=2,
                        attention_mask=mask,
                        backend=backend,
                    )
                expected, expected_probabilities = explicit_attention(q, k, v, spec, mask, 2, backend)
                bits_equal(actual, expected, "explicit attention matrix layout/output")
                bits_equal(probabilities, expected_probabilities, "explicit native softmax/mask")
                require(bool(torch.isfinite(actual).all()), "attention matrix output nonfinite")
                if mask is not None:
                    require(
                        bool((probabilities[mask.isneginf()] == 0).all()), "attention masked token leaked"
                    )
                results.append(
                    {
                        "recipe": recipe_name,
                        "dtype": str(dtype),
                        "queries": queries,
                        "keys": keys,
                        "input_sha256": [tensor_hash(t) for t in (q, k, v)],
                        "explicit_layout_bit_equal": True,
                        "masked_probabilities_zero": True,
                        **trace.result(spec.qk.mma.algorithm, 16),
                    }
                )
    for qk_recipe, pv_recipe in (("fp8_f7_lowacc", "fp64_reference"), ("fp64_reference", "fp8_f7_lowacc")):
        spec = AttentionSpec(load_recipe(qk_recipe).defaults, load_recipe(pv_recipe).defaults)
        torch.manual_seed(42)
        q = torch.randn(2, 4, 2, 8, device=device)
        k = torch.randn(2, 2, 5, 8, device=device)
        v = torch.randn_like(k)
        mask = additive_mask(2, 4, 2, 5, device)
        with GemmTrace() as trace:
            actual, probabilities = emulated_attention(
                q, k, v, spec, scaling=8**-0.5, num_key_value_groups=2, attention_mask=mask, backend=backend
            )
        expected, expected_probabilities = explicit_attention(q, k, v, spec, mask, 2, backend)
        bits_equal(actual, expected, "independent mixed QK/PV arithmetic")
        bits_equal(probabilities, expected_probabilities, "mixed QK probabilities")
        require(
            trace.algorithms == {spec.qk.mma.algorithm: 8, spec.pv.mma.algorithm: 8},
            "mixed attention: QK and PV did not each execute their own algorithm eight times",
        )
        results.append(
            {
                "qk_recipe": qk_recipe,
                "pv_recipe": pv_recipe,
                "input_sha256": [tensor_hash(t) for t in (q, k, v)],
                "explicit_layout_bit_equal": True,
                **trace.result((spec.qk.mma.algorithm, spec.pv.mma.algorithm), 16),
            }
        )
    return results


def build_model(family: str) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

    options = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        eos_token_id=None,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config = LlamaConfig(**options) if family == "llama" else Qwen3Config(**options)
    config._attn_implementation = "eager"
    return (LlamaForCausalLM(config) if family == "llama" else Qwen3ForCausalLM(config)).eval()


def validate_model(family: str, device: str, backend: str, new_tokens: int) -> dict:
    torch.manual_seed(42)
    model = build_model(family).to(device)
    untouched = copy.deepcopy(model)
    identity = state_hash(model)
    originals = dict(model.named_parameters())
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], device=device)
    native = model(ids, attention_mask=mask, use_cache=False).logits
    results = []
    for recipe_name in RECIPES:
        recipe = load_recipe(recipe_name)
        spec = AttentionSpec(recipe.defaults, recipe.defaults)
        report = patch_attention(model, spec, backend=backend)
        require(bool(report.patched), "attention: no HF module patched")
        require(not patch_attention(model, spec, backend=backend).patched, "attention: patch not idempotent")
        require(state_hash(model) == identity, "attention: patch changed parameters/state_dict")
        require(
            all(dict(model.named_parameters())[name] is parameter for name, parameter in originals.items()),
            "attention: Parameter sharing/identity changed",
        )
        with GemmTrace() as trace:
            output = model(ids, attention_mask=mask, use_cache=False).logits
        bits_equal(
            untouched(ids, attention_mask=mask, use_cache=False).logits,
            native,
            "attention: other model instance changed",
        )
        copied = copy.deepcopy(model)
        bits_equal(copied(ids, attention_mask=mask, use_cache=False).logits, output, "attention deepcopy")
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        bits_equal(loaded(ids, attention_mask=mask, use_cache=False).logits, output, "attention pickle")
        with GemmTrace() as cache_trace:
            prefill = model(ids, attention_mask=mask, use_cache=True)
            require(prefill.past_key_values.get_seq_length() == 4, "attention: cache prefill wrong length")
            continuation = model(
                torch.tensor([[8], [9]], device=device),
                attention_mask=torch.cat((mask, torch.ones(2, 1, device=device)), dim=-1),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            require(
                continuation.past_key_values.get_seq_length() == 5, "attention: cache append wrong length"
            )
        with GemmTrace() as generation_trace:
            generated = model.generate(
                ids[1:],
                attention_mask=mask[1:],
                min_new_tokens=new_tokens,
                max_new_tokens=new_tokens,
                eos_token_id=None,
                pad_token_id=0,
                do_sample=False,
                use_cache=True,
            )
        require(generated.shape == (1, 4 + new_tokens), "attention: generation did not complete")
        require(bool(torch.isfinite(output).all()), "attention HF logits nonfinite")
        result = {
            "recipe": recipe_name,
            "recipe_sha256": recipe.sha256,
            "patched": report.patched,
            "forward": trace.result(spec.qk.mma.algorithm, 8),
            "cache": cache_trace.result(spec.qk.mma.algorithm, 16),
            "generation": generation_trace.result(spec.qk.mma.algorithm, 4 * new_tokens),
            "generated_ids": generated.cpu().tolist(),
            "generated_new_tokens": new_tokens,
            "deepcopy_and_whole_model_pickle_bit_equal": True,
        }
        if recipe_name == "fp8_f7_lowacc":
            exact_spec = AttentionSpec(
                replace(spec.qk, mma=MMASpec("fp64", out_format=spec.qk.mma.out_format)),
                replace(spec.pv, mma=MMASpec("fp64", out_format=spec.pv.mma.out_format)),
            )
            unpatch_attention(model)
            patch_attention(model, exact_spec, backend=backend)
            exact = model(ids, attention_mask=mask, use_cache=False).logits
            difference = float((output.double() - exact.double()).abs().max())
            require(difference > 0, "attention: accumulator change did not reach final logits")
            result["logits_vs_same_quant_spec_fp64_max_diff"] = difference
        unpatch_attention(model)
        bits_equal(
            model(ids, attention_mask=mask, use_cache=False).logits, native, "attention: restored native"
        )
        require(not list(iter_emuattention(model)), "attention: unpatch left instances marked")
        result["native_after_unpatch_bit_equal"] = True
        results.append(result)
    # Independently patch projection and attention, then reverse each without losing the other.
    projections = patch_model(model, "fp64_reference", backend=backend)
    patch_attention(
        model,
        AttentionSpec(
            LinearSpec(mma=MMASpec("fp64", out_format="fp32")),
            LinearSpec(mma=MMASpec("fp64", out_format="fp32")),
        ),
        backend=backend,
    )
    finite = model(ids, attention_mask=mask, use_cache=False).logits
    require(
        bool(torch.isfinite(finite).all()) and bool(projections.patched), "attention/projection coexistence"
    )
    unpatch_attention(model)
    unpatch_model(model)
    bits_equal(model(ids, attention_mask=mask, use_cache=False).logits, native, "both adapters restored")
    return {
        "scope": "random_tiny_config_not_pretrained_quality",
        "state_sha256": identity,
        "input_sha256": tensor_hash(ids),
        "config": model.config.to_dict(),
        "recipes": results,
        "projection_adapter_coexistence": True,
        "instance_isolation": True,
    }


def validate_guards(device: str, backend: str) -> dict:
    from transformers.cache_utils import DynamicCache, StaticCache

    spec = AttentionSpec(LinearSpec(), LinearSpec())
    model = build_model("llama").to(device)
    layer = model.model.layers[0].self_attn
    aliases = nn.ModuleDict({"first": layer, "second": layer})
    patch_attention(aliases, spec, backend=backend)
    require(aliases["first"] is aliases["second"], "attention: shared aliases broken")
    unpatch_attention(aliases)
    require("forward" not in layer.__dict__, "attention: class forward not restored")
    rejected = []
    for name, mutation in (
        ("non_eager", lambda item: setattr(item.config, "_attn_implementation", "sdpa")),
        ("sliding_window", lambda item: setattr(item, "sliding_window", 4)),
        ("hook", lambda item: item.register_forward_hook(lambda *args: None)),
        ("custom_forward", lambda item: setattr(item, "forward", item.forward)),
    ):
        candidate = build_model("llama").to(device)
        mutation(candidate.model.layers[0].self_attn)
        try:
            patch_attention(candidate, spec, backend=backend)
        except NotImplementedError:
            require(not list(iter_emuattention(candidate)), f"{name}: failed patch was not atomic")
            rejected.append(name)
        else:
            raise AssertionError(f"attention: unsupported {name} accepted")
    patch_attention(model, spec, backend=backend)
    model.train()
    try:
        model(torch.tensor([[1, 2]], device=device))
    except NotImplementedError:
        rejected.append("training")
    else:
        raise AssertionError("attention: training accepted")
    model.eval()
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=4, device=device, dtype=torch.float32
    )
    try:
        model(torch.tensor([[1, 2]], device=device), past_key_values=cache, use_cache=True)
    except NotImplementedError as error:
        require("DynamicCache" in str(error), "static cache rejection occurred outside the adapter")
        require(cache.get_seq_length() == 0, "rejected static cache was modified")
        rejected.append("static_cache")
    else:
        raise AssertionError("attention: StaticCache accepted")
    processed_cache = DynamicCache()
    processed_cache.cache_processor = object()
    try:
        model(torch.tensor([[1, 2]], device=device), past_key_values=processed_cache, use_cache=True)
    except NotImplementedError as error:
        require("unprocessed" in str(error), "processed cache rejection occurred outside the adapter")
        require(processed_cache.get_seq_length() == 0, "rejected processed cache was modified")
        rejected.append("processed_dynamic_cache")
    else:
        raise AssertionError("attention: processed DynamicCache accepted")
    unsupported = {
        "transform": LinearSpec(transform=TransformSpec("hadamard")),
        "gptq": LinearSpec(weight=QuantSpec(FP8_E4M3), weight_algo=WeightAlgoSpec("gptq")),
        "observer": LinearSpec(activation=QuantSpec(FP8_E4M3, observer=ObserverSpec("minmax"))),
        "history_observer": LinearSpec(activation=QuantSpec(FP8_E4M3, observer=ObserverSpec("history"))),
        "element_sr": LinearSpec(activation=QuantSpec(FP8_E4M3, rounding="sr")),
        "scale_sr": LinearSpec(weight=QuantSpec(FP8_E4M3, scale=ScaleSpec(rounding="sr"))),
        "sparsity": LinearSpec(sparsity=SparsitySpec("n:m", n=2, m=4)),
        "outlier": LinearSpec(weight=QuantSpec(FP8_E4M3), outliers=OutlierSpec(0.25)),
    }
    for name, bad_spec in unsupported.items():
        for side in ("qk", "pv"):
            try:
                AttentionSpec(
                    bad_spec if side == "qk" else LinearSpec(), bad_spec if side == "pv" else LinearSpec()
                )
            except NotImplementedError:
                rejected.append(f"{side}_{name}")
            else:
                raise AssertionError(f"attention: unsupported {side} {name} accepted")
    q = torch.randn(1, 2, 2, 8, device=device)
    k = torch.randn(1, 1, 2, 8, device=device)
    with torch.enable_grad():
        frozen_model = build_model("llama").to(device).requires_grad_(False)
        patch_attention(frozen_model, spec, backend=backend)
        historical_cache = DynamicCache()
        historical_cache.update(
            torch.randn(1, 1, 1, 8, device=device, requires_grad=True),
            torch.randn(1, 1, 1, 8, device=device, requires_grad=True),
            0,
        )
        try:
            frozen_model(
                torch.tensor([[1, 2]], device=device), past_key_values=historical_cache, use_cache=True
            )
        except NotImplementedError as error:
            require("cached autograd history" in str(error), "cache history rejection happened elsewhere")
            require(historical_cache.get_seq_length() == 1, "autograd history rejection appended cache")
            rejected.append("cached_autograd_history_atomic")
        else:
            raise AssertionError("attention: cached autograd history accepted")
        finally:
            unpatch_attention(frozen_model)
        untouched_cache = DynamicCache()
        try:
            model(torch.tensor([[1, 2]], device=device), past_key_values=untouched_cache, use_cache=True)
        except NotImplementedError as error:
            require("autograd" in str(error), "HF autograd rejection happened outside the adapter")
            require(untouched_cache.get_seq_length() == 0, "autograd rejection appended cache values")
            rejected.append("autograd_cache_update_atomic")
        else:
            raise AssertionError("attention: HF autograd accepted")
        try:
            emulated_attention(
                q.detach().requires_grad_(),
                k,
                k,
                spec,
                scaling=8**-0.5,
                num_key_value_groups=2,
                attention_mask=additive_mask(1, 1, 2, 2, device),
                backend=backend,
            )
        except NotImplementedError as error:
            require("autograd" in str(error), "autograd rejection happened outside the adapter")
            rejected.append("autograd")
        else:
            raise AssertionError("attention: autograd accepted")
    for name, bad_mask in (
        ("missing_causal_mask", None),
        ("boolean_mask", torch.ones(1, 1, 2, 2, device=device, dtype=torch.bool)),
    ):
        try:
            emulated_attention(
                q,
                k,
                k,
                spec,
                scaling=8**-0.5,
                num_key_value_groups=2,
                attention_mask=bad_mask,
                backend=backend,
            )
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"attention: unsupported {name} accepted")
    # The evidence gate must fail if the adapter silently substitutes a native matmul.
    with patch.object(attention_module, "attention_gemm", lambda a, b, spec, **kwargs: a @ b.T):
        with GemmTrace() as trace:
            model(torch.tensor([[1, 2]], device=device))
        try:
            trace.result(spec.qk.mma.algorithm)
        except AssertionError:
            rejected.append("native_bypass_negative_control")
        else:
            raise AssertionError("attention: bypass negative control was not detected")
    unpatch_attention(model)
    return {"rejected_or_detected": rejected, "shared_aliases_preserved": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--backend", choices=("reference", "triton"), required=True)
    parser.add_argument("--new-tokens", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "triton" and args.device != "cuda":
        parser.error("Triton requires CUDA")
    if not 17 <= args.new_tokens <= 60:
        parser.error("--new-tokens must be [17,60]; generation is not a 16-token smoke test")
    if any((args.out / name).exists() for name in ("env.json", "result.json")):
        parser.error("select a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1, "isolate one available CUDA GPU"
        )
        if args.backend == "triton":
            require(torch.cuda.get_device_capability()[0] >= 8, "attention Triton needs sm_80+")
    env = capture_env(
        extra={
            "command": [sys.executable, *sys.argv],
            "seed": 42,
            "scope": "random_tiny_models_no_dataset_or_pretrained_quality",
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    result: dict[str, Any] = {"status": "running", "scope": "exact_arithmetic_and_integration_not_quality"}
    started = time.monotonic()
    try:
        with torch.no_grad():
            result["matrices"] = validate_matrices(args.device, args.backend)
            for family in ("llama", "qwen3"):
                print(f"ATTENTION_START {family}", flush=True)
                result[family] = validate_model(family, args.device, args.backend, args.new_tokens)
                print(f"ATTENTION_PASS {family}", flush=True)
            result["guards"] = validate_guards(args.device, args.backend)
        result["status"] = "passed"
    except Exception as error:
        result.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        result["elapsed_seconds_not_benchmark"] = time.monotonic() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"ATTENTION_DONE status={result['status']} out={args.out}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
