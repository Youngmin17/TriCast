"""Per recipe file, the WikiText-2 test PPL of a model measured outside the job runner (Qwen3-8B), for the
tuning app's bundle (``app.export_autotune_demo --outputs``): the existing way's recipe files (one method on
every layer) and Auto-tune's, each on a fresh model.

    # 1 the uniform recipe files exactly as the page hands them over (app/autotune_recipe.grid_recipe)
    python scripts/quantizer/tune_outputs.py uniform --blocks 36 --methods gptq_int4,gptq_int3 --out DIR
    # 2 on a GPU node (tricast env): per recipe file, a fresh model, the recipe applied the way the job
    #   runner applies it (patch, calibration), WikiText-2 test PPL and greedy answers to PROMPTS from that
    #   same model; --native also records the unquantized model (BF16)
    python scripts/quantizer/tune_outputs.py record --model Qwen/Qwen3-8B --revision REV --out RECORDS \
        [--native] DIR/u_gptq_int3.yaml runs/auto_3.15/recipe.yaml …
    # 3 here: the bundle takes each grid's PPL from RECORDS by recipe-file SHA-256
    python -m app.export_autotune_demo --table … --measured … --outputs RECORDS --out …

The page does not show the answers; they stay in the records as the evidence of which setups collapse (a
PPL alone does not say whether a model still writes sentences). ``record`` prints one line per recipe and
ends with ``OUTPUTS_DONE ok=<n> failed=<k>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

# two Korean, one arithmetic with one right answer (10), one English, one code: where a broken model
# shows first
PROMPTS = (
    "양자화가 무엇인지 고등학생에게 두 문장으로 설명해 줘.",
    "사과가 12개 있었는데 5개를 먹고 친구에게 3개를 더 받았어. 지금 사과는 몇 개인지 풀이와 함께 답해 줘.",
    "대한민국의 수도는 어디이고, 그 도시의 대표적인 관광지 두 곳을 알려 줘.",
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
)
MAX_NEW_TOKENS = 160
# the table's calibration settings (app.autotune.calibration_settings with the table command's --calib-*):
# the uniform files must be byte-identical to the ones the page builds from the bundle's table meta
CALIBRATION = {"dataset": "wikitext2", "samples": 128, "seed": 42, "seqlen": 2048, "sequential": False}
# GPTQ fits every layer at once (sequential: false, as the table and the 0.6B runs): on an 8B model the fp64
# Hessians take ~58 GB, over tricast's 4 GiB default storage budget, so the budget is raised (memory only:
# the fitted weights are the same); it needs a GPU with that much free memory (H200 141 GB)
HESSIAN_MAX_BYTES = 96 * 2**30


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def uniform(blocks: int, methods: list[str], out: Path, mma: str) -> None:
    """``u_<method>.yaml`` for each method, through the page's own builder (a uniform grid of a baseline)."""
    from app import autotune_recipe as ar
    from app import quantizer_recipe as qr

    out.mkdir(parents=True, exist_ok=True)
    bundle = {"table": {"blocks": blocks, "kinds": list(qr.KINDS), "methods": ["bf16", *methods],
                        "path": "table.json", "meta": {"calibration": CALIBRATION, "mma": mma}},
              "solutions": [], "baselines": [{"id": f"u_{m}", "method": m} for m in methods]}
    for method in methods:
        recipe = ar.grid_recipe(bundle, qr.uniform(method, blocks))
        (out / recipe["file"]).write_text(recipe["yaml"], encoding="utf-8")
        print(f"{out / recipe['file']} sha256={recipe['sha256']}")


def _answers(model: Any, tokenizer: Any, device: str, max_new_tokens: int) -> list[dict[str, Any]]:
    """Greedy answers, chat template with thinking off; ``ids``: the new tokens, ``text``: decoded."""
    import torch

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.as_tensor(encoded).to(device)
        torch.manual_seed(42)
        with torch.no_grad():
            sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                      num_beams=1, max_new_tokens=max_new_tokens, pad_token_id=pad)
        new = sequence[0, ids.shape[1]:].tolist()
        text = tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        out.append({"prompt": prompt, "ids": new, "text": text})
    return out


def _record_one(model_id: str, revision: str, path: Path | None, device: str, smoke: bool,
                cache: Path | None = None) -> dict[str, Any]:
    """One fresh model; ``path`` None = the unquantized model. ``smoke``: 2 PPL windows and 8 new tokens, to
    check the pipeline on a small model (``app.export_autotune_demo`` refuses such a record). ``cache``: see
    ``prepare``."""
    import torch

    from tricast.eval import perplexity

    model, tokenizer, row, env = prepare(model_id, revision, path, device, cache)
    row.update(smoke=smoke, recorder_sha256=_sha256(Path(__file__).read_bytes()))
    ppl = perplexity(model, tokenizer, device=device, seqlen=2048, max_windows=2 if smoke else None)
    row["ppl"] = {"dataset": "wikitext2", "split": "test", "seqlen": 2048,
                  **{key: ppl.get(key) for key in ("ppl", "n_tokens", "n_windows", "dataset_fingerprint")}}
    row["answers"] = _answers(model, tokenizer, device, 8 if smoke else MAX_NEW_TOKENS)
    row["env"] = env
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def prepare(model_id: str, revision: str, path: Path | None, device: str,
            cache: Path | None = None) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """A fresh model with the recipe file at ``path`` applied the way the job runner applies it (patch, then
    calibration); ``path`` None = the unquantized model. Returns the model in eval mode, its tokenizer, the
    record's identity fields (name, file SHA-256, recipe hash, patched layer count, calibration) and the
    environment. ``cache``: tricast's bit-identical per-layer calibration cache (``calibrate(cache_dir=…)``,
    node-local disk): a recipe whose layers another recipe already fitted the same way reuses those results.
    scripts/quantizer/tune_bench.py prepares its models here too: its benchmarks belong to the same setups."""
    import random

    import numpy as np
    import torch

    from tricast import calibrate, load_recipe, patch_model
    from tricast.eval import capture_env
    from tricast.eval.runner import _load_model

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    dtype = "bfloat16" if device == "cuda" else "float32"
    env = capture_env(model_id, {"seed": 42, "dtype": dtype, "device": device})
    model, tokenizer = _load_model(model_id, dtype, device, revision=revision)
    env["model_sha"] = getattr(model.config, "_commit_hash", None) or env.get("model_sha")
    row: dict[str, Any] = {"model": model_id, "revision": revision}
    if path is not None:
        data = path.read_bytes()
        recipe = load_recipe(str(path))
        report = patch_model(model, recipe)  # firing evidence: how many Linears the recipe replaced
        row.update(name=path.stem if path.name != "recipe.yaml" else path.parent.name, file=path.name,
                   file_sha256=_sha256(data), recipe_hash=recipe.sha256, patched_layers=len(report.patched))
        if recipe.needs_calibration:
            summary = calibrate(model, recipe, tokenizer=tokenizer, device=device,
                                hessian_max_bytes=HESSIAN_MAX_BYTES, cache_dir=cache)
            keys = ("dataset", "split", "samples", "seqlen", "seed", "dataset_fingerprint")
            hits = (summary.get("cache") or {}).get("hits")
            row["calibration"] = {**{key: summary.get(key) for key in keys},
                                  "hessian_max_bytes": HESSIAN_MAX_BYTES,
                                  "cache_hits": None if hits is None else len(hits)}
    else:
        row.update(name="native", file=None, file_sha256=None, recipe_hash=None, patched_layers=0)
    model.eval()
    return model, tokenizer, row, env


def record(model_id: str, revision: str, paths: list[Path], native: bool, out: Path, device: str,
           smoke: bool = False, cache: Path | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    jobs: list[Path | None] = ([None] if native else []) + list(paths)
    ok = failed = 0
    for path in jobs:
        name = "native" if path is None else (path.stem if path.name != "recipe.yaml" else path.parent.name)
        target = out / f"{name}.json"
        if target.is_file():
            print(f"SKIP {name} (recorded)", flush=True)
            ok += 1
            continue
        started = time.time()
        try:
            row = _record_one(model_id, revision, path, device, smoke, cache)
            row["seconds"] = round(time.time() - started, 1)
            target.write_text(json.dumps(row, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"OK {name} ppl={row['ppl']['ppl']:.4f} patched={row['patched_layers']} {row['seconds']}s",
                  flush=True)
            ok += 1
        except Exception:  # one failing recipe does not stop the others; its reason is in the log
            traceback.print_exc()
            print(f"FAILED {name}", flush=True)
            failed += 1
    print(f"OUTPUTS_DONE ok={ok} failed={failed}", flush=True)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    u = sub.add_parser("uniform")
    u.add_argument("--blocks", type=int, required=True)
    u.add_argument("--methods", required=True)
    u.add_argument("--mma", default="fp32_fma")
    u.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("record")
    r.add_argument("--model", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--native", action="store_true")
    r.add_argument("--device", default="cuda")
    r.add_argument("--smoke", action="store_true", help="2 PPL windows, 8 new tokens (pipeline check only)")
    r.add_argument("--calib-cache", type=Path, help="per-layer calibration cache directory (node-local disk)")
    r.add_argument("recipes", nargs="*", type=Path)
    args = parser.parse_args()
    if args.command == "uniform":
        uniform(args.blocks, args.methods.split(","), args.out, args.mma)
        return 0
    return record(args.model, args.revision, args.recipes, args.native, args.out, args.device, args.smoke,
                  args.calib_cache)


if __name__ == "__main__":
    raise SystemExit(main())
