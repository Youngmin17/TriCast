#!/usr/bin/env python
"""Harness self-check: an independent numpy model of the NADPE FP8 kernels.

CPU only. Re-derives every stored case of vectors/fp8_*.pt from the E4M3 codes
with integer arithmetic written from a reading of mma_emu/ (not a copy), and
compares against the CUDA kernel's bf16 bits. It checks the oracle harness
(operand layout, scale order, K tiling) and pins down the semantics TriCast has
to reproduce:
  * product significand = sig_a*sig_b shifted from radix 6 to F (or G) bits;
    a right shift (F<6 / G<6) can leave significand 0 while the product still
    counts as non-zero for max_exp;
  * alignment = right shift of the magnitude to max_exp (RZ), shifts >= 64 -> 0;
  * fixed -> fp32: normalize, then keep min(F,23) fraction bits (RZ);
  * C-fused CoFDA: running fp32 acc joins the chunk at F bits; an all-zero chunk
    leaves acc unchanged;
  * C-decoupled: chunk summed at F bits from 0, then merged into acc at F2=23;
  * GDFS: GS-groups at G bits -> re-radix to F (shift F-G) -> one F-bit FDA over
    the BK/GS group operands of each BK=32 tile plus acc;
  * epilogue: fp32 scale_a * (scale_b * acc), then bf16 round-to-nearest-even.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

NEG = -9999
BK = 32
F2 = 23


def decode(codes):
    c = codes.astype(np.int64)
    sign = np.where(c & 0x80, -1, 1).astype(np.int64)
    e = (c >> 3) & 0xF
    m = c & 0x7
    nan = (c & 0x7F) == 0x7F
    zero = (e == 0) & (m == 0)
    assert not nan.any(), "NaN codes are not modelled"
    sub = (e == 0) & ~zero
    sig = np.where(zero, 0, np.where(sub, m, 8 + m))
    exp = np.where(zero, 0, np.where(sub, -6, e - 7))
    return sign, sig, exp, zero


def shift(x, s):
    return x << s if s >= 0 else x >> (-s)


def products(A, B, k0, k1, frac):
    """fp8_multiply_predecoded<frac> over k in [k0,k1): arrays [M,N,L]."""
    sa, ga, ea, za = (t[:, k0:k1] for t in A)
    sb, gb, eb, zb = (t[:, k0:k1] for t in B)
    zero = za[:, None, :] | zb[None, :, :]
    sig = shift(ga[:, None, :] * gb[None, :, :], frac - 6)
    sig = np.where(zero, 0, sig)
    exp = np.where(zero, 0, ea[:, None, :] + eb[None, :, :])
    sign = sa[:, None, :] * sb[None, :, :]
    return sign, sig, exp, zero


def f32_to_operand(bits, F):
    bits = bits.astype(np.int64)
    ab = bits & 0x7FFFFFFF
    zero = ab == 0
    sign = np.where(bits & 0x80000000, -1, 1).astype(np.int64)
    be = (ab >> 23) & 0xFF
    man = ab & 0x7FFFFF
    if ((be == 255) & ~zero).any() or ((be == 0) & ~zero).any():
        raise NotImplementedError("inf/nan/subnormal accumulator")
    sig = shift((1 << 23) | man, F - 23)
    return sign, np.where(zero, 0, sig), np.where(zero, 0, be - 127), zero


def fixed_to_f32_bits(total, max_exp, F):
    nz = total != 0
    a = np.abs(total)
    if (a >= (1 << 53)).any():
        raise OverflowError("fixed-point sum beyond float64-exact range")
    _, e2 = np.frexp(a.astype(np.float64))
    lead = e2.astype(np.int64) - 1
    be = lead + max_exp - F + 127
    if (nz & ((be <= 0) | (be >= 255))).any():
        raise NotImplementedError("subnormal/overflow result")
    mant = np.where(lead >= 23, a >> np.clip(lead - 23, 0, 63),
                    a << np.clip(23 - lead, 0, 63)) & 0x7FFFFF
    tb = 23 - min(F, 23)
    if tb > 0:
        mant &= 0x7FFFFF & ~((1 << tb) - 1)
    bits = np.where(total < 0, 0x80000000, 0) | (np.clip(be, 0, 255) << 23) | mant
    return np.where(nz, bits, 0)


def chunked_accumulate(ops, c_bits, F):
    osign, osig, oexp, ozero = ops
    csign, csig, cexp, czero = f32_to_operand(c_bits, F)
    mx = np.where(ozero, NEG, oexp).max(axis=-1)
    mx = np.where(czero, mx, np.maximum(mx, cexp))
    nnz = (~ozero).sum(axis=-1) + (~czero)
    d = np.clip(mx[..., None] - oexp, 0, None)
    al = np.where(ozero | (d >= 64), 0, osig >> np.minimum(d, 63))
    total = (osign * al).sum(axis=-1)
    dc = np.clip(mx - cexp, 0, None)
    total = total + np.where(czero | (dc >= 64), 0, csign * (csig >> np.minimum(dc, 63)))
    return np.where(nnz == 0, c_bits, fixed_to_f32_bits(total, mx, F))


def ref_cofda(A, B, K, F, CS, decoupled):
    M, N = A[0].shape[0], B[0].shape[0]
    c = np.zeros((M, N), dtype=np.int64)
    for k0 in range(0, K, CS):
        ops = products(A, B, k0, min(k0 + CS, K), F)
        if not decoupled:
            c = chunked_accumulate(ops, c, F)
        else:
            part = chunked_accumulate(ops, np.zeros_like(c), F)
            ps = tuple(t[..., None] for t in f32_to_operand(part, F2))
            c = chunked_accumulate(ps, c, F2)
    return c


def ref_gdfs(A, B, K, F, G, GS):
    M, N = A[0].shape[0], B[0].shape[0]
    c = np.zeros((M, N), dtype=np.int64)
    for kt in range(0, K, BK):
        gsign, gsig, gexp, gzero = [], [], [], []
        for g in range(BK // GS):
            k0 = kt + g * GS
            if k0 >= K:
                z = np.zeros((M, N), dtype=np.int64)
                gsign.append(z + 1)
                gsig.append(z)
                gexp.append(z)
                gzero.append(z == 0)
                continue
            sign, sig, exp, zero = products(A, B, k0, min(k0 + GS, K), G)
            mx = np.where(zero, NEG, exp).max(axis=-1)
            d = np.clip(mx[..., None] - exp, 0, None)
            al = np.where(zero | (d >= 64), 0, sig >> np.minimum(d, 63))
            msum = (sign * al).sum(axis=-1)
            z = zero.all(axis=-1) | (msum == 0)
            gsign.append(np.where(msum < 0, -1, 1))
            gsig.append(np.where(z, 0, shift(np.abs(msum), F - G)))
            gexp.append(np.where(z, 0, mx))
            gzero.append(z)
        ops = tuple(np.stack(t, axis=-1) for t in (gsign, gsig, gexp, gzero))
        c = chunked_accumulate(ops, c, F)
    return c


def epilogue_bf16_bits(c_bits, sa, sb):
    acc = c_bits.astype(np.uint32).view(np.float32)
    v = np.float32(sa) * (np.float32(sb) * acc)
    if not np.isfinite(v).all():
        raise NotImplementedError("non-finite output")
    b = v.view(np.uint32).astype(np.int64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return r.astype(np.uint16).view(np.int16)


def ref_case(A, B, K, case, scale_a, scale_b):
    """bf16 bits (int16 [M,N]) of one stored case, from decoded operands."""
    alg, F = case["algorithm"], case["f_bits"]
    if alg == 1:
        acc = ref_gdfs(A, B, K, F, case["g_bits"], case["group_size"])
    else:
        acc = ref_cofda(A, B, K, F, case["chunk_size"], decoupled=(alg == 3))
    return epilogue_bf16_bits(acc, scale_a, scale_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "vectors"))
    ap.add_argument("--out", default=None, help="write result json here")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.vectors, "fp8_*.pt")))
    assert paths, "no vector files"
    t0 = time.time()
    n_cases = n_bad_cases = n_bad_elems = 0
    rows, first_bad = [], []
    for path in paths:
        d = torch.load(path, weights_only=True)
        A = decode(d["a_codes"].numpy())
        B = decode(d["w_codes"].numpy())
        K = d["a_codes"].shape[1]
        bad = 0
        for c in d["cases"]:
            ref = ref_case(A, B, K, c, d["scale_a"], d["scale_b"])
            got = c["out_bits"].numpy()
            m = int((ref != got).sum())
            n_cases += 1
            if m:
                bad += 1
                n_bad_elems += m
                if len(first_bad) < 10:
                    i = np.argwhere(ref != got)[0]
                    first_bad.append(dict(file=os.path.basename(path),
                                          case={k: c[k] for k in c if k != "out_bits"},
                                          mismatches=m, first_idx=i.tolist(),
                                          ref=int(ref[tuple(i)]) & 0xFFFF,
                                          got=int(got[tuple(i)]) & 0xFFFF))
        n_bad_cases += bad
        rows.append(dict(file=os.path.basename(path), cases=len(d["cases"]), mismatched_cases=bad))
        print(f"REFCHECK {os.path.basename(path)} cases={len(d['cases'])} mismatched_cases={bad}",
              flush=True)
    res = dict(files=len(paths), cases=n_cases, mismatched_cases=n_bad_cases,
               mismatched_elems=n_bad_elems, seconds=round(time.time() - t0, 1),
               rows=rows, first_mismatches=first_bad)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("REFCHECK_RESULT " + json.dumps({k: v for k, v in res.items() if k != "rows"}),
          flush=True)
    return 0 if n_bad_cases == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
