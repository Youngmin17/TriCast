"""Throughput of TriCast's Triton MMA emulation against NADPE's hand-written CUDA kernel.

Both run the same FP8 E4M3 GEMM under the same accumulation configuration on the same GPU;
outputs are checked bit for bit before timing. Protocol: 3 warm-up calls, then 5 timed calls
(CUDA events), median / mean / p99 reported, determinism settings and env captured.

    python scripts/bench/bench_mma_vs_nadpe.py --nadpe-so <build_fp8/nadpe_mma_emu.so> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch

from tricast import BF16, FP8_E4M3, FP32, MMASpec
from tricast.eval.envinfo import capture_env
from tricast.kernels.mma import gemm_triton
from tricast.mma.operand import Operand

# (label, algorithm id in NADPE, MMASpec) — NADPE FP8 tiles K by 32, so GDFS uses k_tile 32.
CONFIGS = [
    ("hopper CoFDA F=13 CS=32", 2, MMASpec("cofda", f_bits=13, chunk_size=32, out_format=BF16)),
    ("CoFDA C-decoupled F=13 CS=32", 3,
     MMASpec("cofda", f_bits=13, chunk_size=32, c_mode="decoupled", f2_bits=23, out_format=BF16)),
    ("GDFS G=32 F=25 GS=16", 1, MMASpec("gdfs", f_bits=25, g_bits=32, group_size=16, k_tile=32,
                                         out_format=BF16)),
]
SHAPES = [(2048, 1024, 3072), (2048, 3072, 1024), (4096, 4096, 4096)]  # Qwen3-0.6B down/up, square


def load_nadpe(path: str):
    loader = importlib.machinery.ExtensionFileLoader("nadpe_mma_emu", path)
    spec = importlib.util.spec_from_file_location("nadpe_mma_emu", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def timed(fn, warmup: int = 3, repeat: int = 5) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summary(times: list[float], macs: int) -> dict:
    ordered = sorted(times)
    median = statistics.median(ordered)
    return {"median_ms": median, "mean_ms": statistics.fmean(ordered),
            "p99_ms": ordered[min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))],
            "tmac_per_s": macs / (median * 1e-3) / 1e12, "samples_ms": ordered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nadpe-so", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False  # float8 .to() under determinism
    nadpe = load_nadpe(args.nadpe_so)
    rows = []
    for m, n, k in SHAPES:
        a8 = (torch.randn(m, k, device="cuda") * 2).to(torch.float8_e4m3fn)
        w8 = (torch.randn(n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
        one = torch.ones(1, device="cuda")
        a_op = Operand(a8.float(), FP8_E4M3, one.reshape(()), FP32, "tensor")
        w_op = Operand(w8.float(), FP8_E4M3, one.reshape(()), FP32, "tensor")
        for label, algorithm, spec in CONFIGS:
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

            run_nadpe = partial(nadpe.mma_emu_scaled_fp8_mm, out, a8, w8.t(), one, one, None, algorithm,
                                spec.f_bits, spec.g_bits, spec.group_size, spec.chunk_size)
            run_triton = partial(gemm_triton, a_op, w_op, spec)
            run_nadpe()
            same = torch.equal(run_triton().view(torch.int16), out.view(torch.int16))
            t_nadpe = summary(timed(run_nadpe), m * n * k)
            t_triton = summary(timed(run_triton), m * n * k)
            rows.append({"shape": [m, n, k], "config": label, "bit_identical": same,
                         "nadpe": t_nadpe, "triton": t_triton,
                         "triton_over_nadpe": t_triton["tmac_per_s"] / t_nadpe["tmac_per_s"]})
            print(f"{label:30s} {m}x{n}x{k}  bit_identical={same}  "
                  f"nadpe {t_nadpe['tmac_per_s']:.3f} TMAC/s  triton {t_triton['tmac_per_s']:.3f} TMAC/s  "
                  f"ratio {rows[-1]['triton_over_nadpe']:.2f}", flush=True)
    (args.out / "result.json").write_text(json.dumps({"rows": rows}, indent=1))
    (args.out / "env.json").write_text(json.dumps(capture_env(extra={"nadpe_so": args.nadpe_so}),
                                                  indent=1, default=str))
    print("BENCH_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
