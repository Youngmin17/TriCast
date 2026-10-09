"""Latency of one emulated decode step (one new token per step, M = 1) on a Hugging Face model.

Prefills a WikiText-2 test prompt through the patched model, then decodes greedily through the KV
cache. Protocol: WARMUP untimed steps, then STEPS timed steps (`torch.cuda.synchronize` +
`perf_counter`), median / mean / p99 reported with the host load and the number of other processes
on the GPU (decode steps are host-bound, so both move the numbers); the generated token ids are kept
so that runs of two commits can be checked for identical output. The source commit is whichever
`tricast` is on the import path, so the same script measures an older checkout through `PYTHONPATH`.

    python scripts/bench/bench_decode.py --recipe hopper_fp8_w8a8 --out runs/bench_decode
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from tricast import load_recipe, patch_model
from tricast.eval.envinfo import capture_env
from tricast.eval.ppl import _dataset_texts


def percentile(samples: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def contention() -> dict:
    """Host load and other processes on this GPU: decode steps are host-bound, so both move them."""
    uuid = str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", ""))
    others = None  # unknown without a device UUID or nvidia-smi
    try:
        if uuid:
            rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                            "--format=csv,noheader"], text=True, timeout=10).splitlines()
            pids = [row.split(",")[-1].strip() for row in rows if uuid in row]
            others = sum(1 for pid in pids if pid != str(os.getpid()))
    except (OSError, subprocess.SubprocessError):
        pass
    return {"loadavg_1_5_15": list(os.getloadavg()), "cpus": os.cpu_count(), "other_processes_on_gpu": others}


def decode(model, ids: torch.Tensor, steps: int) -> tuple[float, list[float], list[int]]:
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(ids, use_cache=True)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - start
        past, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
        times, tokens = [], []
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(token, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            past, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
            tokens.append(int(token))
    return prefill, times, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--recipe", action="append", help="repeatable; 'native' = unpatched model")
    parser.add_argument("--prompt-len", type=int, default=700)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts, fingerprint = _dataset_texts("wikitext2", "test")
    ids = tokenizer("\n\n".join(texts[:400]), return_tensors="pt").input_ids[:, :args.prompt_len].cuda()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.recipe or ["hopper_fp8_w8a8"]:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
        if name != "native":
            recipe = load_recipe(name)
            if recipe.needs_calibration:
                raise SystemExit(f"{name} needs calibration; this benchmark takes calibration-free recipes")
            patch_model(model, recipe)
        decode(model, ids, args.warmup)  # compiles and autotunes the prefill and M=1 shapes
        prefill, times, tokens = decode(model, ids, args.warmup + args.steps)
        timed = [1e3 * t for t in times[args.warmup:]]
        result = {"recipe": name, "prompt_tokens": int(ids.shape[1]), "warmup": args.warmup,
                  "steps": args.steps, "contention": contention(), "prefill_s": prefill,
                  "decode_ms": {"median": statistics.median(timed), "mean": statistics.fmean(timed),
                                "p99": percentile(timed, 0.99), "min": min(timed), "max": max(timed)},
                  "tokens": tokens}
        results.append(result)
        print(f"DECODE {name}: median {result['decode_ms']['median']:.2f} ms, mean "
              f"{result['decode_ms']['mean']:.2f}, p99 {result['decode_ms']['p99']:.2f}; "
              f"prefill {prefill:.3f} s ({ids.shape[1]} tokens)", flush=True)
        del model
        torch.cuda.empty_cache()
    env = capture_env(args.model, {"dataset_fingerprint": fingerprint,
                                   "device": torch.cuda.get_device_name()})
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"BENCH_DECODE_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
