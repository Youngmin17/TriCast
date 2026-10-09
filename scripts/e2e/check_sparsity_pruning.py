"""2:4 magnitude pruning without TriCast (plain torch, weights pruned in place) against TriCast's
`sparsity` option with no quantization, on the same WikiText-2 windows (CUDA).

If both give the same perplexity, a collapse of the 2:4 recipes comes from the pruning itself, not
from the emulation. Qwen3-0.6B, first 4 windows: dense 23.3331, both pruned 73434.3212 (V100).
Writes result.json and env.json to OUT_DIR.

    python scripts/e2e/check_sparsity_pruning.py [OUT_DIR]
"""
import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tricast import load_recipe, patch_model
from tricast.eval.envinfo import capture_env
from tricast.eval.ppl import perplexity

out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sparsity_crosscheck")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).cuda().eval()


def prune_2of4(weight: torch.Tensor) -> torch.Tensor:
    rows, k = weight.shape
    groups = weight.reshape(rows, k // 4, 4)
    keep = groups.abs().float().argsort(dim=-1, descending=True, stable=True)[..., :2]
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, keep, True)
    return (groups * mask).reshape(rows, k)


native = copy.deepcopy(base)
with torch.no_grad():
    for name, module in native.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            module.weight.copy_(prune_2of4(module.weight))
emulated = copy.deepcopy(base)
prune_only = {"name": "prune-only",
              "defaults": {"sparsity": {"kind": "n:m", "n": 2, "m": 4}, "mma": "fp32_fma"}}
patch_model(emulated, load_recipe(prune_only))
results = {}
for label, model in (("dense", base), ("torch_2of4", native), ("tricast_2of4_fp32acc", emulated)):
    results[label] = perplexity(model, tok, dataset="wikitext2", seqlen=2048, max_windows=4)
    print(f"PPL {label} {results[label]['ppl']:.4f} windows {results[label]['n_windows']}", flush=True)
out.mkdir(parents=True, exist_ok=True)
env = capture_env("Qwen/Qwen3-0.6B", {"device": torch.cuda.get_device_name(),
                                      "dataset_fingerprint": results["dense"]["dataset_fingerprint"]})
(out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
(out / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"CROSSCHECK_DONE {out}", flush=True)
