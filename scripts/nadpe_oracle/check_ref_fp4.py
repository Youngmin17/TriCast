#!/usr/bin/env python
"""Harness self-check for the FP4 vectors: independent numpy model (CPU only).

Re-derives every stored case of vectors/fp4_*.pt from the LOGICAL codes with
integer arithmetic written from a reading of mma_emu/ (FP8 accumulator model
reused from check_ref.py), and compares with the kernel's bf16 bits. Semantics
modelled:
  * E2M1: sign bit 3; mag 0 = zero; e=0 -> sig 1, exp 0; else sig 2+m, exp e-1
  * UE4M3 block scale: MSB masked; masked 0 -> zero scale; e=0 -> sig m, exp -6;
    else sig 8+m, exp e-7 (Q1.3). E8M0: code 0 -> ZERO scale; else exp = code-127
  * GDFS: GS-group at G bits (sig_a*sig_b << (G-2), RZ alignment), then
    nvfp4: |sum|*sig_sa*sig_sb re-radixed by F-(G+6); mxfp4: |sum| by F-G;
    exponent += scale exponents; a re-radixed significand of 0 still counts as
    non-zero; one F-bit FDA per BK=64 tile over its 64/GS group operands + acc
  * CoFDA: per product nvfp4: sig_a*sig_b*sig_sa*sig_sb re-radixed by F-8 (a
    product truncated to 0 IS zero); mxfp4: sig_a*sig_b << (F-2); CS chunks
  * epilogue: nvfp4 fp32 acc*alpha then bf16 RNE; mxfp4 bf16 RNE of acc
--mutations additionally reports how many cases each single-rule change breaks.
"""
import argparse
import glob
import json
import os
import sys
import time

import check_ref as R
import numpy as np
import torch


def decode_e2m1(codes):
    c = codes.astype(np.int64)
    sign = np.where(c & 8, -1, 1).astype(np.int64)
    mag = c & 7
    zero = mag == 0
    e, m = (mag >> 1) & 3, mag & 1
    sig = np.where(zero, 0, np.where(e == 0, m, 2 + m))
    exp = np.where(zero, 0, np.where(e == 0, 0, e - 1))
    return sign, sig, exp, zero


def decode_scale(codes, fmt, msb_sign=False):
    c = codes.astype(np.int64)
    sign = np.ones_like(c)
    if fmt == "nvfp4":
        cm = c & 0x7F
        assert not (cm == 0x7F).any(), "NaN scale not modelled"
        zero = cm == 0
        e, m = (cm >> 3) & 0xF, cm & 7
        sig = np.where(e == 0, m, 8 + m)
        exp = np.where(e == 0, -6, e - 7)
        if msb_sign:
            sign = np.where(c & 0x80, -1, 1)
    else:
        assert not (c == 255).any(), "NaN scale not modelled"
        zero = c == 0
        sig = np.ones_like(c)
        exp = c - 127
    return zero, sig, exp, sign


def _scale_at(S, b, axis):
    return tuple((t[:, b][:, None] if axis == 0 else t[:, b][None, :]) for t in S)


def ref_gdfs(fmt, A, B, SA, SB, K, F, G, GS, bk=64, reradix_zero=False):
    sa, ga, ea, za = A
    sb, gb, eb, zb = B
    M, N = ga.shape[0], gb.shape[0]
    blk = 16 if fmt == "nvfp4" else 32
    c = np.zeros((M, N), dtype=np.int64)
    for kt in range(0, K, bk):
        ops = []
        for g in range(bk // GS):
            k0 = kt + g * GS
            if k0 >= K:
                z = np.zeros((M, N), dtype=np.int64)
                ops.append((z + 1, z, z, z == 0))
                continue
            k1 = min(k0 + GS, K)
            zero = za[:, None, k0:k1] | zb[None, :, k0:k1]
            sig = np.where(zero, 0, (ga[:, None, k0:k1] * gb[None, :, k0:k1]) << (G - 2))
            exp = np.where(zero, 0, ea[:, None, k0:k1] + eb[None, :, k0:k1])
            sign = sa[:, None, k0:k1] * sb[None, :, k0:k1]
            mx = np.where(zero, R.NEG, exp).max(axis=-1)
            d = np.clip(mx[..., None] - exp, 0, None)
            al = np.where(zero | (d >= 64), 0, sig >> np.minimum(d, 63))
            msum = (sign * al).sum(axis=-1)
            zA, gA, eA, sA = _scale_at(SA, k0 // blk, 0)
            zB, gB, eB, sB = _scale_at(SB, k0 // blk, 1)
            gz = zero.all(axis=-1) | (msum == 0) | zA | zB
            mag = np.abs(msum)
            gsig = R.shift(mag * gA * gB, F - (G + 6)) if fmt == "nvfp4" else R.shift(mag, F - G)
            if reradix_zero:
                gz = gz | (gsig == 0)
            gsign = np.where(msum < 0, -1, 1) * sA * sB
            ops.append((gsign, np.where(gz, 0, gsig), np.where(gz, 0, mx + eA + eB), gz))
        stacked = tuple(np.stack(t, axis=-1) for t in zip(*ops, strict=False))
        c = R.chunked_accumulate(stacked, c, F)
    return c


def ref_cofda(fmt, A, B, SA, SB, K, F, CS, trunc_zero=True):
    sa, ga, ea, za = A
    sb, gb, eb, zb = B
    M, N = ga.shape[0], gb.shape[0]
    blk = 16 if fmt == "nvfp4" else 32
    c = np.zeros((M, N), dtype=np.int64)
    for k0 in range(0, K, CS):
        k1 = min(k0 + CS, K)
        zA, gA, eA, sA = (t[..., None] for t in _scale_at(SA, k0 // blk, 0))
        zB, gB, eB, sB = (t[..., None] for t in _scale_at(SB, k0 // blk, 1))
        zero = za[:, None, k0:k1] | zb[None, :, k0:k1] | zA | zB
        raw = ga[:, None, k0:k1] * gb[None, :, k0:k1]
        sig = R.shift(raw * gA * gB, F - 8) if fmt == "nvfp4" else R.shift(raw, F - 2)
        if trunc_zero:
            zero = zero | (sig == 0)
        exp = ea[:, None, k0:k1] + eb[None, :, k0:k1] + eA + eB
        sign = sa[:, None, k0:k1] * sb[None, :, k0:k1] * sA * sB
        c = R.chunked_accumulate((sign, np.where(zero, 0, sig), np.where(zero, 0, exp), zero), c, F)
    return c


def epilogue(c_bits, alpha, fmt):
    acc = c_bits.astype(np.uint32).view(np.float32)
    v = acc * np.float32(alpha) if fmt == "nvfp4" else acc
    if not np.isfinite(v).all():
        raise NotImplementedError("non-finite output")
    b = v.view(np.uint32).astype(np.int64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return r.astype(np.uint16).view(np.int16)


def ref_case(d, case, msb_sign=False, bk=64, reradix_zero=False, trunc_zero=True):
    fmt = d["meta"]["format"]
    A = decode_e2m1(d["a_codes"].numpy())
    B = decode_e2m1(d["w_codes"].numpy())
    SA = decode_scale(d["a_scale_codes"].numpy(), fmt, msb_sign)
    SB = decode_scale(d["w_scale_codes"].numpy(), fmt, msb_sign)
    K = d["a_codes"].shape[1]
    if case["algorithm"] == 1:
        acc = ref_gdfs(fmt, A, B, SA, SB, K, case["f_bits"], case["g_bits"], case["group_size"],
                       bk=bk, reradix_zero=reradix_zero)
    else:
        acc = ref_cofda(fmt, A, B, SA, SB, K, case["f_bits"], case["chunk_size"],
                        trunc_zero=trunc_zero)
    return epilogue(acc, d["alpha"], fmt)


MUTATIONS = {
    "UE4M3 MSB read as a negative sign (nvfp4)": dict(msb_sign=True),
    "GDFS: F-bit FDA per group (tile = GS) instead of per BK=64 tile": dict(bk="GS"),
    "GDFS: F-bit FDA per 128-wide tile": dict(bk=128),
    "GDFS: re-radixed significand 0 counts as zero": dict(reradix_zero=True),
    "CoFDA nvfp4: product truncated to 0 still non-zero": dict(trunc_zero=False),
}


def e8m0_tiny_rule_hits(d):
    """outputs where NADPE gives +-0 in every case but reading E8M0 code 0 as 2^-127
    gives a non-zero bf16 value (exact float64 product, then bf16 rounding)."""
    E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    def val(c, s):
        v = E2M1[c & 7] * np.where(c & 8, -1.0, 1.0)
        return v * np.repeat(np.exp2(s.astype(np.float64) - 127), d["block_size"], axis=1)
    a, w = d["a_codes"].numpy(), d["w_codes"].numpy()
    y = val(a, d["a_scale_codes"].numpy()) @ val(w, d["w_scale_codes"].numpy()).T
    y32 = y.astype(np.float32).view(np.uint32).astype(np.int64)
    bf = ((y32 + 0x7FFF + ((y32 >> 16) & 1)) >> 16) & 0x7FFF
    zero_all = np.all([(c["out_bits"].numpy() & 0x7FFF) == 0 for c in d["cases"]], axis=0)
    return int((zero_all & (bf != 0)).sum()), int(zero_all.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "vectors"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mutations", action="store_true")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.vectors, "fp4_*.pt")))
    assert paths, "no fp4 vector files"
    data = [(os.path.basename(p), torch.load(p, weights_only=True)) for p in paths]
    t0 = time.time()
    n_cases = n_bad = n_elems = 0
    rows, first_bad = [], []
    for name, d in data:
        bad = 0
        for c in d["cases"]:
            ref = ref_case(d, c)
            got = c["out_bits"].numpy()
            m = int((ref != got).sum())
            n_cases += 1
            if m:
                bad += 1
                n_elems += m
                if len(first_bad) < 10:
                    i = np.argwhere(ref != got)[0]
                    first_bad.append(dict(file=name, case={k: c[k] for k in c if k != "out_bits"},
                                          mismatches=m, first_idx=i.tolist(),
                                          ref=int(ref[tuple(i)]) & 0xFFFF,
                                          got=int(got[tuple(i)]) & 0xFFFF))
        n_bad += bad
        rows.append(dict(file=name, cases=len(d["cases"]), mismatched_cases=bad))
        print(f"REFCHECK_FP4 {name} cases={len(d['cases'])} mismatched_cases={bad}", flush=True)
    res = dict(files=len(paths), cases=n_cases, mismatched_cases=n_bad, mismatched_elems=n_elems,
               seconds=round(time.time() - t0, 1), rows=rows, first_mismatches=first_bad)
    if args.mutations:
        table = {}
        for label, kw in MUTATIONS.items():
            cnt = {}
            for _name, d in data:
                fmt = d["meta"]["format"]
                for c in d["cases"]:
                    kk = dict(kw)
                    if kk.get("bk") == "GS":
                        kk["bk"] = c["group_size"] or 64
                    key = f"{fmt}/alg{c['algorithm']}"
                    hit, tot = cnt.get(key, (0, 0))
                    ref = ref_case(d, c, **kk)
                    cnt[key] = (hit + int((ref != c["out_bits"].numpy()).any()), tot + 1)
            table[label] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(cnt.items())}
            print(f"MUT {label}: " + " ".join(f"{k}={v}" for k, v in table[label].items()),
                  flush=True)
        e8 = {name: e8m0_tiny_rule_hits(d) for name, d in data if d["scale_format"] == "e8m0"}
        table["E8M0 code 0 read as 2^-127 (outputs NADPE=0 in all cases -> non-zero bf16)"] = {
            k: f"{v[0]}/{v[1]}" for k, v in e8.items()}
        print("MUT E8M0 code0=2^-127: " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in e8.items()),
              flush=True)
        res["mutations"] = table
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("REFCHECK_FP4_RESULT " + json.dumps({k: v for k, v in res.items()
                                               if k not in ("rows", "mutations")}), flush=True)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
