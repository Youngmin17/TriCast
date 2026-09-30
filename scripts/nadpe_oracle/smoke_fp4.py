#!/usr/bin/env python
"""Launch smoke test of the NADPE NVFP4 / MXFP4 emulation ops on sm_80 (GPU job).

All block scales are 1.0 (UE4M3 0x38 / E8M0 127) and alpha = 1, so the swizzled
scale layout cannot affect the result; outputs are compared with an exact fp64
matmul of the decoded E2M1 values. This is a launch/plausibility check only,
not a golden-vector generator.
"""
import json
import os
import sys

import torch
from gen_vectors import load_ext

HERE = os.path.dirname(os.path.abspath(__file__))
SENTINEL = 0x7FAB
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def round_up(x, y):
    return (x + y - 1) // y * y


def decode(nib):
    mag = torch.tensor(E2M1, dtype=torch.float64)[(nib & 7).long()]
    return torch.where((nib & 8) != 0, -mag, mag)


def pack(nib):  # low nibble = even k, high nibble = odd k (scaled_*fp4_mm.cuh)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()


def check(ext, out, exact, label):
    err = ext.last_cuda_error()
    torch.cuda.synchronize()
    bits = out.view(torch.int16).cpu()
    y = out.double().cpu()
    cos = float((y * exact).sum() / (y.norm() * exact.norm()))
    row = dict(case=label, cuda_error=err, untouched=int((bits == SENTINEL).sum()),
               y_finite=bool(torch.isfinite(y).all()), cos_vs_exact=cos,
               max_abs_diff=float((y - exact).abs().max()))
    print("SMOKE " + json.dumps(row), flush=True)
    return row


def main():
    ext = load_ext(os.path.join(HERE, "build_fp4", "nadpe_mma_emu_fp4.so"))
    dev = torch.device("cuda", 0)
    M, N, K = 33, 64, 128  # N % 32 == 0, K % 64 == 0 (MXFP4) / % 32 (NVFP4)
    torch.manual_seed(4321)
    a_nib = torch.randint(0, 16, (M, K), dtype=torch.uint8)
    w_nib = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    exact = decode(a_nib) @ decode(w_nib).t()
    A, B = pack(a_nib).to(dev), pack(w_nib).to(dev)
    rows = []

    def fresh():
        out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        out.view(torch.int16).fill_(SENTINEL)
        return out

    sf_nv = torch.full((round_up(M, 128), round_up(K // 16, 4)), 0x38, dtype=torch.uint8)
    sfb_nv = torch.full((round_up(N, 128), round_up(K // 16, 4)), 0x38, dtype=torch.uint8)
    A_sf = sf_nv.view(torch.float8_e4m3fn).to(dev)
    B_sf = sfb_nv.view(torch.float8_e4m3fn).to(dev)
    alpha = torch.tensor([1.0], dtype=torch.float32, device=dev)
    for alg, f, g in [(2, 25, 6), (2, 13, 6), (2, 3, 6), (1, 35, 6), (1, 7, 3)]:
        out = fresh()
        ext.mma_emu_scaled_nvfp4_mm(out, A, B, A_sf, B_sf, alpha, alg, f, g)
        rows.append(check(ext, out, exact, f"nvfp4 alg={alg} F={f} G={g}"))

    A_sf = torch.full((round_up(M, 128), round_up(K // 32, 4)), 127, dtype=torch.uint8, device=dev)
    B_sf = torch.full((round_up(N, 128), round_up(K // 32, 4)), 127, dtype=torch.uint8, device=dev)
    for alg, f, g, gs, cs in [(2, 25, 6, 32, 32), (2, 13, 6, 32, 16), (1, 35, 6, 32, 32),
                              (1, 7, 3, 16, 32)]:
        out = fresh()
        ext.mma_emu_scaled_mxfp4_mm(out, A, B, A_sf, B_sf, alg, f, g, gs, cs)
        rows.append(check(ext, out, exact, f"mxfp4 alg={alg} F={f} G={g} GS={gs} CS={cs}"))

    ok = all(not r["cuda_error"] and r["untouched"] == 0 and r["y_finite"] for r in rows)
    print(f"SMOKE_FP4_DONE ok={ok} cases={len(rows)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
