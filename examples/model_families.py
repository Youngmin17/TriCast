"""Patch Llama/Qwen linear arithmetic or opt in to ResNet/YOLO convolution lowering.

Download-free architecture examples (all random initialized, not quality eval)::

    python examples/model_families.py --family llama --device cpu --backend reference
    python examples/model_families.py --family qwen --device cuda --backend triton
    python examples/model_families.py --family resnet --device cuda --backend triton
    python examples/model_families.py --family yolo --device cuda --backend triton

Load real HF weights explicitly with --model-id and --revision, or a local
Ultralytics checkpoint with --weights. The validator also accepts a local YOLO
checkpoint with --yolo-weights and never downloads it automatically. Attention/normalization and
nonlinear operations remain native; Conv2d is an opt-in unfold+GEMM lowering.
Use tricast ppl/eval/report for dataset-backed LLM quality measurements.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from tricast import load_recipe, patch_model
from tricast.nn.patch import unpatch_model


def output_shapes(output: Any) -> list[list[int]]:
    if isinstance(output, torch.Tensor):
        if not torch.isfinite(output).all():
            raise RuntimeError("model produced nonfinite output")
        return [list(output.shape)]
    if isinstance(output, (tuple, list)):
        return [shape for item in output for shape in output_shapes(item)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=("llama", "qwen", "resnet", "yolo"), required=True)
    parser.add_argument("--recipe", default="fp8_f7_lowacc")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--backend", choices=("reference", "triton"), required=True)
    parser.add_argument("--model-id", help="HF model ID/local directory; text families only")
    parser.add_argument("--revision", help="HF commit SHA for reproducible pretrained loads")
    parser.add_argument("--weights", help="local YOLO checkpoint; otherwise use random yolo11n.yaml")
    args = parser.parse_args()
    if args.backend == "triton" and args.device != "cuda":
        parser.error("--backend triton requires --device cuda")
    if args.family in ("resnet", "yolo") and args.device == "cpu":
        parser.error("Full vision emulation is expensive: use an isolated UCL GPU")
    if args.model_id and args.family not in ("llama", "qwen"):
        parser.error("--model-id is for Llama/Qwen; use --weights for YOLO")
    if args.model_id and not args.revision:
        parser.error("--model-id requires an explicit --revision")
    if args.weights and args.family != "yolo":
        parser.error("--weights is for YOLO only")
    if args.weights and not Path(args.weights).is_file():
        parser.error("--weights must name an existing local checkpoint")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    text = args.family in ("llama", "qwen")
    if text:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaConfig,
            LlamaForCausalLM,
            Qwen3Config,
            Qwen3ForCausalLM,
        )

        if args.model_id:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision,
                                                        torch_dtype=torch.float32,
                                                        attn_implementation="eager")
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
            inputs = tokenizer("Explain how an MMA accumulator affects a matrix product.",
                               return_tensors="pt").input_ids
        else:
            options = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                       "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2}
            config = LlamaConfig(**options) if args.family == "llama" else Qwen3Config(**options, head_dim=8)
            config._attn_implementation = "eager"
            model = LlamaForCausalLM(config) if args.family == "llama" else Qwen3ForCausalLM(config)
            inputs = torch.arange(2, 10).unsqueeze(0)
    elif args.family == "resnet":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        inputs = torch.randn(1, 3, 32, 32)
    else:
        if args.weights:
            from ultralytics import YOLO

            model = YOLO(args.weights).model
        else:
            from ultralytics.nn.tasks import DetectionModel

            model = DetectionModel("yolo11n.yaml", ch=3, nc=80, verbose=False)
        inputs = torch.randn(1, 3, 32, 32)
    model = model.eval().to(args.device)
    inputs = inputs.to(args.device)
    recipe = load_recipe(args.recipe)
    report = patch_model(model, recipe, backend=args.backend, include_conv2d=not text)
    if not report.patched:
        raise RuntimeError("recipe selected no supported modules")
    try:
        with torch.no_grad():
            output = model(inputs)
            shapes = output_shapes(output.logits if text else output)
        print({"family": args.family, "recipe": recipe.name, "patched_modules": len(report.patched),
               "output_shapes": shapes, "scope": "pretrained_forward" if args.model_id or args.weights
               else "random_initialized_architecture_only"})
    finally:
        unpatch_model(model)


if __name__ == "__main__":
    main()
