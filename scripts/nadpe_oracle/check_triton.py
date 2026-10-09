"""Triton MMA kernels against the NADPE golden vectors, bit for bit (needs a CUDA device).

tests/test_golden.py checks the exact reference against tests/data/nadpe on CPU; this script
runs the same cases through tricast.kernels.mma.gemm_triton:

    python scripts/nadpe_oracle/check_triton.py 'tests/data/nadpe/fp8_*.pt' 'tests/data/nadpe/fp4_*.pt'

Prints one NADPE_TRITON line per glob (cases and mismatches per algorithm) and a final
NADPE_TRITON_DONE line; exits 1 if any case mismatches or a glob matches no file.
"""

from __future__ import annotations

import collections
import glob
import sys

import torch

from tricast.formats import BF16, E8M0, FP4_E2M1, FP8_E4M3, FP32, UE4M3
from tricast.kernels.mma import gemm_triton
from tricast.mma.operand import Operand
from tricast.mma.spec import MMASpec

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def operands(vectors: dict, name: str) -> tuple[list[Operand], int, str]:
    """Operands of one vector file, NADPE's K tile, and a family label."""
    if name.startswith("fp8_"):
        ops = [Operand(vectors[codes].view(torch.float8_e4m3fn).float().cuda(), FP8_E4M3,
                       torch.tensor(float(vectors[scale]), dtype=torch.float32).cuda(), FP32, "tensor")
               for codes, scale in (("a_codes", "scale_a"), ("w_codes", "scale_b"))]
        return ops, 32, "fp8"
    block, scale_format = int(vectors["block_size"]), vectors["scale_format"]
    ops = []
    for side in ("a", "w"):
        values = E2M1[vectors[f"{side}_codes"].long()]
        codes = vectors[f"{side}_scale_codes"]
        if scale_format == "ue4m3":  # sign bit masked; 0x00 / 0x80 are zero scales
            scale, fmt = (codes & 0x7F).contiguous().view(torch.float8_e4m3fn).float(), UE4M3
        else:
            scale, fmt = torch.ldexp(torch.ones(codes.shape), codes.long() - 127).float(), E8M0
        alpha = (torch.tensor(float(vectors["alpha"])).cuda()
                 if side == "a" and scale_format == "ue4m3" else None)
        ops.append(Operand(values.cuda(), FP4_E2M1, scale.cuda(), fmt, "k", block, alpha))
    return ops, 64, f"fp4_{scale_format}"


def spec_of(case: dict, k_tile: int) -> MMASpec:
    algorithm = int(case["algorithm"])  # NADPE ids: 1 GDFS, 2 CoFDA C-fused, 3 CoFDA C-decoupled
    if algorithm == 1:
        return MMASpec("gdfs", f_bits=int(case["f_bits"]), g_bits=int(case["g_bits"]),
                       group_size=int(case["group_size"]), k_tile=k_tile, out_format=BF16)
    return MMASpec("cofda", f_bits=int(case["f_bits"]), chunk_size=int(case["chunk_size"]),
                   c_mode="decoupled" if algorithm == 3 else "fused", f2_bits=23, out_format=BF16)


def check(pattern: str) -> int:
    total, mismatched = collections.Counter(), collections.Counter()
    paths = sorted(glob.glob(pattern))
    for path in paths:
        vectors = torch.load(path, weights_only=True)
        ops, k_tile, family = operands(vectors, path.rsplit("/", 1)[-1])
        for case in vectors["cases"]:
            spec = spec_of(case, k_tile)
            out = gemm_triton(*ops, spec).cpu().view(torch.int16)
            key = f"{family}:{spec.algorithm}{'_decoupled' if spec.c_mode == 'decoupled' else ''}"
            total[key] += 1
            mismatched[key] += not torch.equal(out, case["out_bits"].to(torch.int16))
    print("NADPE_TRITON", pattern, "files", len(paths), "cases", dict(total),
          "mismatched", {k: v for k, v in mismatched.items() if v}, flush=True)
    return sum(mismatched.values()) + (not paths)


def main() -> None:
    patterns = sys.argv[1:] or ["tests/data/nadpe/fp8_*.pt", "tests/data/nadpe/fp4_*.pt"]
    print("device", torch.cuda.get_device_name(), flush=True)
    failures = sum(check(pattern) for pattern in patterns)
    print("NADPE_TRITON_DONE failures", failures, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
