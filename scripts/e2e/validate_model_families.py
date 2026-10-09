"""Reproducible, download-free model-family integration validation.

Run on UCL, not as a substitute for the repository CPU/GPU test suites::

    python scripts/e2e/validate_model_families.py --suites conv,llama,qwen \
        --device cpu --backend reference --out runs/<UTC>/model_families_cpu
    CUDA_VISIBLE_DEVICES=0 python scripts/e2e/validate_model_families.py \
        --suites resnet,yolo --device cuda --backend triton \
        --out runs/<UTC>/model_families_vision

Models use deterministic random initial weights unless --yolo-weights names a
local checkpoint. This validates execution,
arithmetic propagation and reversibility, NOT pretrained quality/PPL/mAP or
native tensor-core equivalence/performance. Requested missing dependencies fail.
Expected APIs: torch>=2.4, transformers>=4.55, a torch-compatible torchvision,
and ultralytics 8.3+ (YOLO11). CUDA execution also requires Triton>=3.4/sm_80+.
Tiny LLMs have one 32-wide layer; vision uses full ResNet18/YOLO11n, image32.
Elapsed seconds are run costs, not benchmark measurements.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from tricast import MMASpec, gemm, load_recipe, patch_model, quantize
from tricast.eval.envinfo import capture_env
from tricast.mma.api import as_operand
from tricast.mma.operand import Operand
from tricast.nn.linear import EmuLinear
from tricast.nn.patch import iter_emuconv2d, iter_emulinear, unpatch_model
from tricast.recipe import LinearSpec

RECIPES = ("fp8_f7_lowacc", "mxfp4_w_a", "int8_row_w8a8", "bf16_passthrough", "fp64_reference")
SUITES = ("conv", "llama", "qwen", "resnet", "yolo")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def tensor_hash(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((values.shape, values.dtype)).encode())
    digest.update(values.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def tensors(output: Any) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        return [tensor for item in output for tensor in tensors(item)]
    if isinstance(output, dict):
        return [tensor for item in output.values() for tensor in tensors(item)]
    return []


def finite(output: Any, label: str) -> list[torch.Tensor]:
    result = tensors(output)
    require(bool(result), f"{label}: no output tensors")
    require(all(bool(torch.isfinite(t).all()) for t in result), f"{label}: nonfinite output")
    return result


def bit_equal(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    require(a.shape == b.shape and a.dtype == b.dtype, f"{label}: shape/dtype mismatch")
    left, right = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    require(torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)),
            f"{label}: bit mismatch; max_diff={(left.double() - right.double()).abs().max().item()}")


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    require(a.shape == b.shape, "max_diff: compared output shapes differ")
    return float((a.detach().double() - b.detach().double()).abs().max().item())


def cpu_operand(operand: Operand) -> Operand:
    return replace(operand, values=operand.values.detach().cpu(),
                   scale=None if operand.scale is None else operand.scale.detach().cpu(),
                   alpha=None if operand.alpha is None else operand.alpha.detach().cpu())


def engines(model: nn.Module) -> list[EmuLinear]:
    layers = [layer for _, layer in iter_emulinear(model)]
    layers.extend(engine for _, conv in iter_emuconv2d(model) for engine in conv._engines)
    return list({id(layer): layer for layer in layers}.values())


def fp64_control(model: nn.Module, forward: Callable[[], Any]) -> tuple[Any, dict]:
    saved = [(layer, layer.spec, id(layer._weight_operand), id(layer._weight_noise))
             for layer in engines(model)]
    try:
        for layer, spec, _, _ in saved:
            layer.spec = replace(spec, mma=MMASpec("fp64", out_format=spec.mma.out_format))
        output = forward()
    finally:
        for layer, spec, _, _ in saved:
            layer.spec = spec
    require(all(id(layer._weight_operand) == operand_id for layer, _, operand_id, _ in saved),
            "fp64 control regenerated a cached quantized weight operand")
    require(all(id(layer._weight_noise) == noise_id for layer, _, _, noise_id in saved),
            "fp64 control regenerated stochastic weight rounding noise")
    return output, {
        "cached_weight_operand_identity_preserved": True,
        "weight_rounding_noise_identity_preserved": True,
        "scope": "same input/weights/quantization specs; downstream activation operands may differ",
    }


class ArithmeticTrace:
    """Count actual layer/GEMM calls; optionally compare identical quant operands."""

    def __init__(self, model: nn.Module, *, compare: bool = False) -> None:
        self.expected = {layer.name: layer.spec.mma.algorithm for _, layer in iter_emulinear(model)}
        self.outlier_layers = {layer.name for _, layer in iter_emulinear(model)
                               if layer.spec.outliers is not None}
        for _, conv in iter_emuconv2d(model):
            # .to()/deepcopy can leave caches empty until the first forward.
            names = {f"{conv.name}.group{group}" for group in range(conv.groups)}
            self.expected.update((name, conv.spec.mma.algorithm) for name in names)
            if conv.spec.outliers is not None:
                self.outlier_layers.update(names)
        self.compare = compare
        self.layer_calls: Counter[str] = Counter()
        self.algorithms: Counter[str] = Counter()
        self.per_layer_algorithms: dict[str, Counter[str]] = {}
        self.same_operand_max_diff = 0.0
        self.reference_parity_calls = 0
        self._active = ""
        self._contexts: list[Any] = []

    def __enter__(self) -> ArithmeticTrace:
        original_matmul = EmuLinear._matmul
        from tricast.nn import linear

        original_gemm = linear.gemm

        def traced_matmul(layer: EmuLinear, operand: Operand, *args: Any, **kwargs: Any) -> torch.Tensor:
            self.layer_calls[layer.name] += 1
            prior, self._active = self._active, layer.name
            try:
                return original_matmul(layer, operand, *args, **kwargs)
            finally:
                self._active = prior

        def traced_gemm(a: Operand, b: Operand, spec: MMASpec, **kwargs: Any) -> torch.Tensor:
            self.algorithms[spec.algorithm] += 1
            self.per_layer_algorithms.setdefault(self._active, Counter())[spec.algorithm] += 1
            output = original_gemm(a, b, spec, **kwargs)
            finite(output, f"{self._active} GEMM")
            if self.compare:
                # CPU reference avoids thousands of tiny CUDA launches; operand bits/scales
                # are copied intact, not requantized or reconstructed from dequant values.
                a_cpu, b_cpu = cpu_operand(a), cpu_operand(b)
                bias = kwargs.get("bias")
                bias = None if bias is None else bias.detach().cpu()
                reference = gemm(a_cpu, b_cpu, spec, bias=bias, backend="reference")
                bit_equal(output.cpu(), reference, f"{self._active} reference parity")
                exact = gemm(a_cpu, b_cpu, MMASpec("fp64", out_format=spec.out_format),
                             bias=bias, backend="reference")
                finite(exact, f"{self._active} same-operand fp64")
                self.same_operand_max_diff = max(self.same_operand_max_diff, max_diff(reference, exact))
                self.reference_parity_calls += 1
            return output

        self._contexts = [patch.object(EmuLinear, "_matmul", traced_matmul),
                          patch.object(linear, "gemm", traced_gemm)]
        for context in self._contexts:
            context.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        for context in reversed(self._contexts):
            context.__exit__(*args)

    def summary(self) -> dict:
        missing = sorted(self.expected.keys() - self.layer_calls.keys())
        require(bool(self.expected), "no EmuLinear engines selected")
        require(not missing, f"arithmetic bypass: selected layers were not called: {missing}")
        for name, algorithm in self.expected.items():
            observed = self.per_layer_algorithms.get(name, Counter())
            require(observed[algorithm] >= self.layer_calls[name],
                    f"arithmetic bypass: {name} did not execute {algorithm} on every layer call")
            allowed = {algorithm, "fp32_fma"} if name in self.outlier_layers else {algorithm}
            require(observed.keys() <= allowed,
                    f"arithmetic bypass: {name} executed unexpected algorithms {dict(observed)}")
        return {"bypass_detected": False, "layer_calls": dict(self.layer_calls),
                "gemm_algorithm_calls": dict(self.algorithms),
                "gemm_algorithm_calls_per_layer": {name: dict(counts)
                                                    for name, counts in self.per_layer_algorithms.items()},
                "reference_bit_parity_calls": self.reference_parity_calls,
                "same_quant_operands_fp64_max_diff": self.same_operand_max_diff if self.compare else None}


def lowered_conv(conv: nn.Conv2d, x: torch.Tensor, spec: LinearSpec) -> torch.Tensor:
    """Independent explicit lowering; no EmuConv2d/EmuLinear forward is used."""
    padding = conv._reversed_padding_repeated_twice
    mode = "constant" if conv.padding_mode == "zeros" else conv.padding_mode
    padded = F.pad(x, padding, mode=mode)
    patches = F.unfold(padded, conv.kernel_size, dilation=conv.dilation, stride=conv.stride)
    n, _, locations = patches.shape
    group_k = patches.shape[1] // conv.groups
    group_n = conv.out_channels // conv.groups
    patches = patches.reshape(n, conv.groups, group_k, locations)
    outputs = []
    for group in range(conv.groups):
        weight = conv.weight[group * group_n:(group + 1) * group_n].reshape(group_n, group_k)
        weight_op = as_operand(weight if spec.weight is None else quantize(weight, spec.weight,
                                                                         backend="reference"))
        bias = None if conv.bias is None else conv.bias[group * group_n:(group + 1) * group_n]
        activation = patches[:, group].transpose(1, 2)
        quant = spec.activation
        per_image = (quant is not None and quant.scale is not None and
                     (quant.granularity == "tensor" or quant.scale.two_level or
                      (quant.granularity == "block" and quant.block[0] > 1)))
        sequences = list(activation) if n > 1 and per_image else [activation]
        values = []
        for sequence in sequences:
            operand = sequence if quant is None else quantize(sequence.reshape(-1, group_k), quant,
                                                             backend="reference")
            value = gemm(operand, weight_op, spec.mma, bias=bias, backend="reference")
            values.append(value.reshape(-1, locations, group_n))
        outputs.append(torch.cat(values, dim=0))
    height = (padded.shape[-2] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    width = (padded.shape[-1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    return torch.cat(outputs, dim=-1).transpose(1, 2).reshape(n, conv.out_channels, height, width).to(x.dtype)


def validate_conv(device: str, backend: str, recipes: list[str]) -> dict:
    cases = [
        ("dense_bias", dict(in_channels=3, out_channels=4, kernel_size=3, padding=1)),
        ("group_stride", dict(in_channels=4, out_channels=6, kernel_size=3, groups=2,
                              stride=2, padding=(1, 2), bias=False)),
        ("depthwise_dilation", dict(in_channels=3, out_channels=6, kernel_size=3,
                                    groups=3, dilation=2, padding=2)),
        ("reflect_padding", dict(in_channels=2, out_channels=3, kernel_size=3, padding=1,
                                 padding_mode="reflect")),
        ("replicate_padding", dict(in_channels=2, out_channels=3, kernel_size=3, padding=1,
                                   padding_mode="replicate")),
        ("circular_padding", dict(in_channels=2, out_channels=3, kernel_size=3, padding=1,
                                  padding_mode="circular")),
        ("same_even_kernel", dict(in_channels=2, out_channels=3, kernel_size=2, padding="same")),
        ("valid_padding", dict(in_channels=2, out_channels=3, kernel_size=3, padding="valid")),
    ]
    results = []
    for recipe_name in recipes:
        recipe = load_recipe(recipe_name)
        for name, options in cases:
            torch.manual_seed(42)
            original = nn.Conv2d(**options).eval().to(device)
            model = nn.Sequential(original).eval()
            x = torch.randn(2, original.in_channels, 7, 8, device=device)
            model_identity, input_identity = state_hash(model), tensor_hash(x)
            identity = (id(original), id(original.weight), id(original.bias))
            native = original(x)
            report = patch_model(model, recipe, backend=backend, include_conv2d=True)
            require(len(report.patched) == 1, f"{name}: Conv2d not selected")
            with ArithmeticTrace(model) as trace:
                actual = model(x)
            expected = lowered_conv(copy.deepcopy(original).cpu(), x.cpu(), recipe.defaults)
            bit_equal(actual.cpu(), expected, f"{recipe_name}/{name}: explicit lowered reference")
            unpatch_model(model)
            restored = model[0]
            require((id(restored), id(restored.weight), id(restored.bias)) == identity,
                    f"{name}: original module/parameter identity not restored")
            restored.to(device)
            bit_equal(restored(x), native, f"{name}: native output after unpatch")
            summary = trace.summary()
            require(summary["gemm_algorithm_calls"].get(recipe.defaults.mma.algorithm, 0) > 0,
                    f"{recipe_name}/{name}: selected algorithm was not called")
            results.append({"recipe": recipe_name, "recipe_sha256": recipe.sha256,
                            "case": name, "options": options, "output_shape": list(actual.shape),
                            "state_sha256": model_identity, "input_sha256": input_identity,
                            "outputs_finite": True, "reference_max_diff": 0.0,
                            "reference_bit_equal": True, "restored_identity": True,
                            "native_after_unpatch_bit_equal": True, **summary})
    return {"cases": results, "contracts": validate_conv_contracts(device, backend)}


def validate_conv_contracts(device: str, backend: str) -> list[dict]:
    """Public integration guards outside the protected repository test tree."""
    torch.manual_seed(42)
    recipe = load_recipe("fp8_f7_lowacc")
    original = nn.Conv2d(2, 3, 3, padding=1).eval().to(device)
    model = nn.Module()
    model.add_module("left", original)
    model.add_module("right", original)
    x = torch.randn(2, 2, 7, 8, device=device)
    native = original(x)
    parameter_ids = (id(original.weight), id(original.bias))
    checkpoint = {name: value.clone() for name, value in model.state_dict().items()}
    report = patch_model(model, recipe, backend=backend)
    require(not report.patched and model.left is original and model.right is original,
            "Conv2d opt-in contract was broken")
    require(set(report.skipped) == {"left", "right"}, "disabled Conv2d aliases were not reported")
    patch_model(model, recipe, backend=backend, include_conv2d=True)
    require(model.left is model.right, "shared Conv2d alias identity was broken")
    require(model.state_dict().keys() == checkpoint.keys(), "Conv2d state_dict keys changed")
    require(not patch_model(model, recipe, backend=backend, include_conv2d=True).patched,
            "second patch was not idempotent")
    model.load_state_dict(checkpoint, strict=True)
    with ArithmeticTrace(model) as trace:
        left, right = model.left(x), model.right(x)
        chw = model.left(x[0])
    bit_equal(left, right, "alias outputs")
    bit_equal(chw, left[0], "CHW versus first image")
    oracle = lowered_conv(copy.deepcopy(original).cpu(), x.cpu(), recipe.defaults)
    bit_equal(left.cpu(), oracle, "alias/load_state_dict lowered reference")
    contracts = [{"case": "opt_in_alias_idempotence_checkpoint_CHW", "passed": True,
                  "reason": "default opt-in report, shared identity, unchanged keys, strict load, bit parity",
                  **trace.summary()}]

    model.to(dtype=torch.float16)
    require(not model.left._engines, "dtype move retained stale private engines")
    with ArithmeticTrace(model) as move_trace:
        half_output = model.left(x.half())
    half_oracle = lowered_conv(copy.deepcopy(original).cpu().half(), x.cpu().half(), recipe.defaults)
    bit_equal(half_output.cpu(), half_oracle, "half dtype rebuilt cache")
    model.float().cpu().to(device)
    require(not model.left._engines, "device move retained stale private engines")
    with ArithmeticTrace(model) as restored_trace:
        restored_output = model.left(x)
    # The fp16 round trip changes stored weights, so compare to the current
    # parameter bits, not the pre-move native baseline.
    oracle = lowered_conv(copy.deepcopy(model.left._original).cpu(), x.cpu(), recipe.defaults)
    bit_equal(restored_output.cpu(), oracle, "device move rebuilt cache")
    contracts.append({"case": "dtype_device_cache_invalidation", "passed": True,
                      "reason": "cache cleared and fresh half/fp32 GEMMs equal explicit lowering",
                      "half_trace": move_trace.summary(), "restored_trace": restored_trace.summary()})
    model.train()
    try:
        with torch.enable_grad():
            model.left(x)
    except NotImplementedError as error:
        contracts.append({"case": "training_rejected", "passed": True, "reason": str(error)})
    else:
        raise AssertionError("Conv2d training with gradients was not explicitly rejected")
    finally:
        model.eval()
    unpatch_model(model)
    require(model.left is original and model.right is original, "unpatch lost original shared module")
    require((id(original.weight), id(original.bias)) == parameter_ids,
            "unpatch lost original Parameter identity")
    # Restore the original checkpoint after the deliberate dtype round trip.
    model.load_state_dict(checkpoint, strict=True)
    bit_equal(original(x), native, "native identity/checkpoint round trip")

    for selector, override in (("conflicting_skip", {"match": "right", "skip": True}),
                               ("conflicting_mma", {"match": "right", "mma": {"preset": "fp64"}})):
        before = state_hash(model)
        try:
            patch_model(model, replace(recipe, overrides=[override]), backend=backend, include_conv2d=True)
        except ValueError as error:
            require(model.left is original and model.right is original and state_hash(model) == before,
                    f"{selector}: rejection was not atomic")
            contracts.append({"case": selector, "passed": True, "reason": str(error)})
        else:
            raise AssertionError(f"{selector}: incompatible aliases were accepted")

    class CustomConv(nn.Conv2d):
        pass

    rejected = [("transform", load_recipe("nvfp4_smoothquant"), nn.Conv2d(2, 3, 3)),
                ("gptq", load_recipe("w4a16_g128_zp_gptq"), nn.Conv2d(2, 3, 3)),
                ("observer", load_recipe("fp8_ema_static"), nn.Conv2d(2, 3, 3)),
                ("custom_subclass", recipe, CustomConv(2, 3, 3)),
                ("forward_hook", recipe, nn.Conv2d(2, 3, 3))]
    for name, unsupported_recipe, conv in rejected:
        if name == "forward_hook":
            conv.register_forward_hook(lambda module, inputs, output: output + 1)
        candidate = nn.Sequential(nn.Linear(4, 4), conv).eval().to(device)
        before = state_hash(candidate)
        try:
            patch_model(candidate, unsupported_recipe, backend=backend, include_conv2d=True)
        except NotImplementedError as error:
            require(isinstance(candidate[0], nn.Linear) and candidate[1] is conv,
                    f"{name}: unsupported patch partially replaced modules")
            require(state_hash(candidate) == before, f"{name}: unsupported patch changed model state")
            contracts.append({"case": f"{name}_rejected_atomically", "passed": True, "reason": str(error)})
        else:
            raise AssertionError(f"{name}: unsupported Conv2d contract was accepted")

    # Weight-only SR has deterministic activations: deepcopy must retain the
    # cached weight noise, rather than draw a different quantized model.
    require(recipe.defaults.weight is not None, "SR guard needs an explicit weight QuantSpec")
    sr_spec = replace(recipe.defaults, weight=replace(recipe.defaults.weight, rounding="sr"))
    sr_recipe = replace(recipe, name="guard_fp8_weight_sr", defaults=sr_spec)
    sr_model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1)).eval().to(device)
    patch_model(sr_model, sr_recipe, backend=backend, include_conv2d=True)
    sr_output = sr_model(x)
    sr_copy = copy.deepcopy(sr_model)
    with ArithmeticTrace(sr_copy) as copy_trace:
        copied_output = sr_copy(x)
    bit_equal(copied_output, sr_output, "deepcopy cached stochastic weight model")
    require(all(tensor_hash(a._weight_noise) == tensor_hash(b._weight_noise)
                for a, b in zip(engines(sr_model), engines(sr_copy), strict=True)),
            "deepcopy changed cached stochastic weight noise")
    contracts.append({"case": "deepcopy_weight_sr_cache_preserved", "passed": True,
                      "reason": "copied weight noise preserved; deterministic activations bit-identical",
                      **copy_trace.summary()})
    return contracts


def build_llm(family: str) -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

    options = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
               "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
               "max_position_embeddings": 128, "bos_token_id": 1, "eos_token_id": None,
               "pad_token_id": 0, "tie_word_embeddings": False, "attention_dropout": 0.0}
    config = LlamaConfig(**options) if family == "llama" else Qwen3Config(**options, head_dim=8)
    config._attn_implementation = "eager"
    return (LlamaForCausalLM(config) if family == "llama" else Qwen3ForCausalLM(config)).eval()


def validate_llm(family: str, args: argparse.Namespace) -> dict:
    torch.manual_seed(42)
    model = build_llm(family).to(args.device)
    identity = state_hash(model)
    originals = {name: layer for name, layer in model.named_modules() if isinstance(layer, nn.Linear)}
    ids = torch.arange(2, 18, device=args.device).reshape(2, 8)
    native = model(ids, use_cache=False).logits
    finite(native, f"{family}: native")
    results = []
    for recipe_name in args.recipes:
        recipe = load_recipe(recipe_name)
        report = patch_model(model, recipe, backend=args.backend)
        require(bool(report.patched), f"{family}/{recipe_name}: no selected Linear layers")
        with ArithmeticTrace(model, compare=True) as forward_trace:
            output = model(ids, use_cache=False).logits
        forward_summary = forward_trace.summary()
        require(set(forward_summary["gemm_algorithm_calls"]) == {recipe.defaults.mma.algorithm},
                f"{family}/{recipe_name}: a different accumulator algorithm executed")
        if recipe_name == "fp8_f7_lowacc":
            require(forward_summary["same_quant_operands_fp64_max_diff"] > 0,
                    f"{family}: low-precision accumulator had no measured effect (possible bypass)")
        exact_logits, control = fp64_control(model, lambda: model(ids, use_cache=False).logits)
        logits_accumulator_diff = max_diff(output, exact_logits)
        if recipe_name == "fp8_f7_lowacc":
            require(logits_accumulator_diff > 0,
                    f"{family}: low-precision accumulator did not propagate to final logits")
        with ArithmeticTrace(model) as generation_trace:
            generated = model.generate(ids[:1], attention_mask=torch.ones_like(ids[:1]),
                                       min_new_tokens=args.new_tokens, max_new_tokens=args.new_tokens,
                                       do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0)
        require(generated.shape[-1] == ids.shape[-1] + args.new_tokens,
                f"{family}/{recipe_name}: generation did not complete {args.new_tokens} tokens")
        generation_summary = generation_trace.summary()
        require(set(generation_summary["gemm_algorithm_calls"]) == {recipe.defaults.mma.algorithm},
                f"{family}/{recipe_name}: generation bypassed selected accumulator algorithm")
        finite([output, exact_logits, generated], f"{family}/{recipe_name}")
        unpatch_model(model)
        require(all(model.get_submodule(name) is layer for name, layer in originals.items()),
                f"{family}/{recipe_name}: original Linear identity not restored")
        bit_equal(model(ids, use_cache=False).logits, native, f"{family}: native after unpatch")
        results.append({"recipe": recipe_name, "recipe_sha256": recipe.sha256,
                        "algorithm": recipe.defaults.mma.algorithm,
                        "patched_count": len(report.patched), "logits_shape": list(output.shape),
                        "logits_vs_native_max_diff": max_diff(output, native),
                        "logits_vs_fp64_same_quant_spec_max_diff": logits_accumulator_diff,
                        "fp64_control": control,
                        "forward": forward_summary, "generation": generation_summary,
                        "generated_new_tokens": args.new_tokens, "generated_ids": generated.cpu().tolist(),
                        "native_after_unpatch_bit_equal": True})
    return {"scope": "random_initialization_tiny_config_not_pretrained_quality",
            "state_sha256": identity, "input_sha256": tensor_hash(ids),
            "config": model.config.to_dict(), "recipes": results}


def validate_vision(family: str, args: argparse.Namespace) -> dict:
    torch.manual_seed(42)
    if family == "resnet":
        from torchvision.models import resnet18

        model = resnet18(weights=None).eval()
        architecture = "ResNet18"
    else:
        if args.yolo_weights is not None:
            from ultralytics import YOLO

            model = YOLO(str(args.yolo_weights)).model.eval()
            config = getattr(model, "yaml", {})
            architecture = Path(str(config.get("yaml_file", type(model).__name__))).stem
        else:
            from ultralytics.nn.tasks import DetectionModel

            model = DetectionModel("yolo11n.yaml", ch=3, nc=80, verbose=False).eval()
            architecture = "YOLO11n"
    model = model.to(args.device)
    identity = state_hash(model)
    originals = {name: module for name, module in model.named_modules()
                 if isinstance(module, (nn.Linear, nn.Conv2d))}
    x = torch.randn(1, 3, args.image_size, args.image_size, device=args.device)
    native = finite(model(x), f"{family}: native")
    recipe = load_recipe("fp8_f7_lowacc")
    report = patch_model(model, recipe, backend=args.backend, include_conv2d=True)
    selected_conv = sum(isinstance(originals.get(name), nn.Conv2d) for name, _ in report.patched)
    require(selected_conv > 0, f"{family}: no Conv2d arithmetic selected")
    with ArithmeticTrace(model) as trace:
        output = finite(model(x), f"{family}: emulated")
    summary = trace.summary()
    exact, control = fp64_control(model, lambda: model(x))
    exact_outputs = finite(exact, f"{family}: fp64 control")
    require(len(exact_outputs) == len(output), f"{family}: fp64 control changed output structure")
    accumulator_differences = [max_diff(a, b) for a, b in zip(output, exact_outputs, strict=True)]
    require(any(value > 0 for value in accumulator_differences),
            f"{family}: low-precision accumulation did not propagate to final outputs")
    require(len(native) == len(output), f"{family}: changed output structure")
    differences = [max_diff(a, b) for a, b in zip(output, native, strict=True)]
    unpatch_model(model)
    require(all(model.get_submodule(name) is module for name, module in originals.items()),
            f"{family}: original module identity not restored")
    restored = finite(model(x), f"{family}: restored")
    for a, b in zip(restored, native, strict=True):
        bit_equal(a, b, f"{family}: native output after unpatch")
    checkpoint = args.yolo_weights if family == "yolo" else None
    return {"architecture": architecture, "scope": "full_forward_not_mAP_or_accuracy",
            "initialization": "checkpoint" if checkpoint is not None else "seeded_random",
            "checkpoint_sha256": (None if checkpoint is None else
                                  hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
            "state_sha256": identity, "input_sha256": tensor_hash(x), "input_shape": list(x.shape),
            "recipe": recipe.name, "recipe_sha256": recipe.sha256,
            "patched_count": len(report.patched), "patched_conv2d_count": selected_conv,
            "output_shapes": [list(t.shape) for t in output], "outputs_finite": True,
            "outputs_vs_native_max_diff": differences,
            "outputs_vs_fp64_same_quant_spec_max_diff": accumulator_differences,
            "fp64_control": control, "native_after_unpatch_bit_equal": True, **summary}


def selection(value: str, choices: tuple[str, ...]) -> list[str]:
    selected = value.split(",")
    if not selected or any(item not in choices for item in selected) or len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError(f"select unique comma-separated values from {', '.join(choices)}")
    return selected


def hex_hash(value: str, lengths: tuple[int, ...]) -> str:
    if len(value) not in lengths or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise argparse.ArgumentTypeError(f"expected a full hexadecimal hash of length {lengths}")
    return value.lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suites", type=lambda value: selection(value, SUITES), default=list(SUITES))
    parser.add_argument("--recipes", type=lambda value: selection(value, RECIPES), default=list(RECIPES))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--backend", choices=("reference", "triton"), required=True)
    parser.add_argument("--new-tokens", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--yolo-weights", type=Path, help="local .pt checkpoint, no auto-download")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--base-git-sha", type=lambda value: hex_hash(value, (40, 64)))
    parser.add_argument("--source-archive-sha256", type=lambda value: hex_hash(value, (64,)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.new_tokens <= 16 or args.new_tokens + 8 > 128:
        parser.error("--new-tokens must be in [17, 120] (not a 16-token smoke test)")
    if args.image_size < 32 or args.image_size % 32:
        parser.error("--image-size must be a multiple of 32, at least 32")
    if args.backend == "triton" and args.device != "cuda":
        parser.error("Triton validation requires --device cuda")
    if args.device == "cpu" and any(suite in args.suites for suite in ("resnet", "yolo")):
        parser.error("Full vision emulation is intentionally UCL GPU-only; split CPU and GPU suites")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.yolo_weights is not None:
        if ("yolo" not in args.suites or not args.yolo_weights.is_file()
                or args.yolo_weights.suffix.lower() != ".pt"):
            parser.error("--yolo-weights needs the yolo suite and an existing local .pt checkpoint")
    if any((args.out / name).exists() for name in ("env.json", "result.json")):
        parser.error("--out already contains env.json/result.json; select a fresh run directory")
    # A caller may create the directory first to redirect stdout/stderr into it.
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    optional_versions = {}
    for package in ("torchvision", "ultralytics"):
        try:
            optional_versions[package] = version(package)
        except PackageNotFoundError:
            optional_versions[package] = None
    env = capture_env(extra={"command": [sys.executable, *sys.argv], "seed": 42, "device": args.device,
                             "backend": args.backend, "threads": args.threads,
                             "optional_versions": optional_versions,
                             "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "base_git_sha": args.base_git_sha,
                             "source_archive_sha256": args.source_archive_sha256,
                             "models": "seeded_random_or_explicit_local_yolo_checkpoint_no_auto_download",
                             "mkl_cbwr": os.environ.get("MKL_CBWR"),
                             "dataset": "deterministic_synthetic_inputs_no_quality_claim"})
    (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    result: dict[str, Any] = {
        "status": "running", "scope": "integration_not_accuracy_or_performance", "suites": {},
        "limitations": ["Attention, norms, nonlinearities and pooling remain native",
                        "Conv2d uses unfold+GEMM, not native Tensor Core Conv parity"],
    }
    started = time.monotonic()
    try:
        if args.device == "cuda":
            require(torch.cuda.is_available(), "requested CUDA unavailable")
            require(torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to exactly one isolated GPU")
            if args.backend == "triton":
                require(torch.cuda.get_device_capability()[0] >= 8, "Triton integration requires sm_80+")
        with torch.no_grad():
            for suite in args.suites:
                suite_started = time.monotonic()
                print(f"MODEL_FAMILY_START {suite}", flush=True)
                if suite == "conv":
                    evidence = validate_conv(args.device, args.backend, args.recipes)
                elif suite in ("llama", "qwen"):
                    evidence = validate_llm(suite, args)
                else:
                    evidence = validate_vision(suite, args)
                evidence["elapsed_seconds_not_benchmark"] = time.monotonic() - suite_started
                result["suites"][suite] = evidence
                print(f"MODEL_FAMILY_PASS {suite}", flush=True)
                (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["elapsed_seconds_not_benchmark"] = time.monotonic() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"MODEL_FAMILIES_DONE status={result['status']} out={args.out}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
