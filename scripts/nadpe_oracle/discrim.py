#!/usr/bin/env python
"""How discriminative are the golden vectors? (CPU only)

Each mutation changes one semantic detail of the check_ref.py numpy model; the
table reports how many stored cases then stop matching the CUDA oracle. A
mutation with 0 mismatching cases is a detail these vectors cannot detect.
"""
import argparse
import glob
import json
import os
import sys

import check_ref as R
import numpy as np
import torch

ORIG = {k: getattr(R, k) for k in ("fixed_to_f32_bits", "products", "ref_gdfs",
                                   "ref_cofda", "epilogue_bf16_bits", "chunked_accumulate")}


def fixed_variant(mode):
    def fn(total, max_exp, F):
        nz = total != 0
        a = np.abs(total)
        _, e2 = np.frexp(a.astype(np.float64))
        lead = e2.astype(np.int64) - 1
        be = lead + max_exp - F + 127
        mant = np.where(lead >= 23, a >> np.clip(lead - 23, 0, 63),
                        a << np.clip(23 - lead, 0, 63)) & 0x7FFFFF
        tb = 23 - min(F, 23)
        if tb > 0 and mode == "rne":
            full = mant | (1 << 23)
            keep = full >> tb
            rem = full & ((1 << tb) - 1)
            half = 1 << (tb - 1)
            keep = keep + ((rem > half) | ((rem == half) & ((keep & 1) == 1)))
            carry = keep >> (24 - tb)
            keep = np.where(carry > 0, keep >> 1, keep)
            be = be + carry
            mant = (keep << tb) & 0x7FFFFF
        bits = np.where(total < 0, 0x80000000, 0) | (np.clip(be, 0, 255) << 23) | mant
        return np.where(nz, bits, 0)
    return fn


def products_zero_after_trunc(A, B, k0, k1, frac):
    sign, sig, exp, zero = ORIG["products"](A, B, k0, k1, frac)
    zero = zero | (sig == 0)
    return sign, sig, np.where(zero, 0, exp), zero


def gdfs_with_bk(bk_of_gs):
    def fn(A, B, K, F, G, GS):
        old = R.BK
        R.BK = bk_of_gs(GS)
        try:
            return ORIG["ref_gdfs"](A, B, K, F, G, GS)
        finally:
            R.BK = old
    return fn


def cofda_decoupled_ieee_merge(A, B, K, F, CS, decoupled):
    if not decoupled:
        return ORIG["ref_cofda"](A, B, K, F, CS, False)
    M, N = A[0].shape[0], B[0].shape[0]
    c = np.zeros((M, N), dtype=np.float32)
    for k0 in range(0, K, CS):
        ops = R.products(A, B, k0, min(k0 + CS, K), F)
        part = R.chunked_accumulate(ops, np.zeros((M, N), dtype=np.int64), F)
        c = c + part.astype(np.uint32).view(np.float32)  # fp32 RNE add
    return c.view(np.uint32).astype(np.int64)


def epilogue_scale_product_first(c_bits, sa, sb):
    acc = c_bits.astype(np.uint32).view(np.float32)
    v = (np.float32(sa) * np.float32(sb)) * acc
    b = v.view(np.uint32).astype(np.int64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return r.astype(np.uint16).view(np.int16)


def rshift_rne(x, d):
    d = np.minimum(d, 63)
    q = x >> d
    r = x - (q << d)
    half = np.where(d > 0, np.left_shift(1, np.maximum(d - 1, 0)), 0)
    up = (d > 0) & ((r > half) | ((r == half) & ((q & 1) == 1)))
    return q + up


def chunked_accumulate_rne_align(ops, c_bits, F):
    osign, osig, oexp, ozero = ops
    csign, csig, cexp, czero = R.f32_to_operand(c_bits, F)
    mx = np.where(ozero, R.NEG, oexp).max(axis=-1)
    mx = np.where(czero, mx, np.maximum(mx, cexp))
    nnz = (~ozero).sum(axis=-1) + (~czero)
    d = np.clip(mx[..., None] - oexp, 0, None)
    al = np.where(ozero | (d >= 64), 0, rshift_rne(osig, d))
    total = (osign * al).sum(axis=-1)
    dc = np.clip(mx - cexp, 0, None)
    total = total + np.where(czero | (dc >= 64), 0, csign * rshift_rne(csig, dc))
    return np.where(nnz == 0, c_bits, R.fixed_to_f32_bits(total, mx, F))


def gdfs_variant(A, B, K, F, G, GS, reradix_zero_is_zero=False, rne_align=False):
    M, N = A[0].shape[0], B[0].shape[0]
    c = np.zeros((M, N), dtype=np.int64)
    acc = chunked_accumulate_rne_align if rne_align else R.chunked_accumulate
    for kt in range(0, K, R.BK):
        gs_, gg, ge, gz = [], [], [], []
        for g in range(R.BK // GS):
            k0 = kt + g * GS
            if k0 >= K:
                z = np.zeros((M, N), dtype=np.int64)
                gs_.append(z + 1)
                gg.append(z)
                ge.append(z)
                gz.append(z == 0)
                continue
            sign, sig, exp, zero = R.products(A, B, k0, min(k0 + GS, K), G)
            mx = np.where(zero, R.NEG, exp).max(axis=-1)
            d = np.clip(mx[..., None] - exp, 0, None)
            sh = rshift_rne(sig, d) if rne_align else sig >> np.minimum(d, 63)
            al = np.where(zero | (d >= 64), 0, sh)
            msum = (sign * al).sum(axis=-1)
            gsig = R.shift(np.abs(msum), F - G)
            z = zero.all(axis=-1) | (msum == 0)
            if reradix_zero_is_zero:
                z = z | (gsig == 0)
            gs_.append(np.where(msum < 0, -1, 1))
            gg.append(np.where(z, 0, gsig))
            ge.append(np.where(z, 0, mx))
            gz.append(z)
        ops = tuple(np.stack(t, axis=-1) for t in (gs_, gg, ge, gz))
        c = acc(ops, c, F)
    return c


MUTATIONS = {
    "control: gdfs_variant with no change (expect 0)": {
        "ref_gdfs": lambda A, B, K, F, G, GS: gdfs_variant(A, B, K, F, G, GS)},
    "alignment: RNE right shift instead of RZ (all algorithms)": {
        "chunked_accumulate": chunked_accumulate_rne_align,
        "ref_gdfs": lambda A, B, K, F, G, GS: gdfs_variant(A, B, K, F, G, GS, rne_align=True)},
    "GDFS: group operand with significand 0 after F-G re-radix counts as zero": {
        "ref_gdfs": lambda A, B, K, F, G, GS: gdfs_variant(A, B, K, F, G, GS,
                                                           reradix_zero_is_zero=True)},
    "fixed_to_fp32: no F-bit truncation (keep 23)": {"fixed_to_f32_bits": fixed_variant("none")},
    "fixed_to_fp32: RNE at F bits instead of RZ": {"fixed_to_f32_bits": fixed_variant("rne")},
    "products: significand 0 after >> counts as zero": {"products": products_zero_after_trunc},
    "GDFS: inter-group FDA per 64-wide tile (BK=64)": {"ref_gdfs": gdfs_with_bk(lambda gs: 64)},
    "GDFS: inter-group FDA per group (BK=GS)": {"ref_gdfs": gdfs_with_bk(lambda gs: gs)},
    "C-decoupled: merge = IEEE fp32 RNE add (not F2=23 RZ)": {"ref_cofda": cofda_decoupled_ieee_merge},
    "epilogue: (scale_a*scale_b)*acc": {"epilogue_bf16_bits": epilogue_scale_product_first},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "vectors"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = []
    for path in sorted(glob.glob(os.path.join(args.vectors, "fp8_*.pt"))):
        d = torch.load(path, weights_only=True)
        data.append((os.path.basename(path), d, R.decode(d["a_codes"].numpy()),
                     R.decode(d["w_codes"].numpy())))

    # structural facts straight from the oracle outputs
    alg2_vs_3 = [0, 0]
    distinct = {}
    for name, d, _, _ in data:
        by_key = {(c["algorithm"], c["f_bits"], c["chunk_size"]): c["out_bits"] for c in d["cases"]}
        for (alg, f, cs), bits in by_key.items():
            if alg == 2:
                alg2_vs_3[1] += 1
                alg2_vs_3[0] += int(not torch.equal(bits, by_key[(3, f, cs)]))
        distinct[name] = len({c["out_bits"].numpy().tobytes() for c in d["cases"]})

    table = {}
    for label, patch in MUTATIONS.items():
        for k, v in patch.items():
            setattr(R, k, v)
        try:
            cnt = {}
            for _name, d, A, B in data:
                K = d["a_codes"].shape[1]
                kind = d["meta"]["input_kind"]
                for c in d["cases"]:
                    ref = R.ref_case(A, B, K, c, d["scale_a"], d["scale_b"])
                    key = f"alg{c['algorithm']}/{kind}"
                    hit, tot = cnt.get(key, (0, 0))
                    cnt[key] = (hit + int((ref != c["out_bits"].numpy()).any()), tot + 1)
        finally:
            for k in patch:
                setattr(R, k, ORIG[k])
        table[label] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(cnt.items())}
        print(f"MUT {label}: " + " ".join(f"{k}={v}" for k, v in table[label].items()),
              flush=True)
    res = dict(mutations=table, cfused_vs_decoupled_differ=f"{alg2_vs_3[0]}/{alg2_vs_3[1]}",
               distinct_outputs_per_file=distinct)
    print("DISCRIM_RESULT " + json.dumps({k: v for k, v in res.items() if k != "mutations"}),
          flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
