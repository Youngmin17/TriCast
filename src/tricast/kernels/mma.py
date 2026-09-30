"""CUDA-core GEMM emulation: sequential K order, no dot/Tensor Core fallback.

FDA streams each chunk twice. GDFS retains at most eight fixed-point groups in
statically indexed SSA tuples (separate registers, not a shared-memory K tile).
Scale descriptors are (kind, K-domain, domain-count, format, E8M0-field0).
All multiplication/addition outside explicit FMA sites is separately rounded.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ..formats import BF16, FP16, FP32, FloatFormat, Format, IntFormat, Pow2Format, container_dtype
from ..mma.operand import Operand
from ..mma.spec import MMASpec
from .cast_core import fmt_constexprs, round_to_format
from .ieee import add_rn, f64_to_f32_rn, fma_rn, mul_rn
from .mma_core import (
    aligned,
    c_operand,
    decode_f32,
    finish,
    merge,
    pack_f32,
    product,
    scale_term,
    scan,
    shift,
)


@triton.jit
def _load_scale(S, rows, k, size, K, DESC: tl.constexpr):
    if DESC[0] == "none":
        s = tl.full(rows.shape, 1, tl.float32)
    elif DESC[0] == "tensor":
        s = tl.broadcast_to(tl.load(S).to(tl.float32), rows.shape)
    elif DESC[0] == "row":
        s = tl.load(S + rows, rows < size, other=1).to(tl.float32)
    else:
        s = tl.load(S + rows * DESC[2] + k // DESC[1], (rows < size) & (k < K), other=1)
    return s


@triton.jit
def _k_scale(S, rows, k, size, K, DESC: tl.constexpr):
    if DESC[0] == "k":
        s = _load_scale(S, rows, k, size, K, DESC)
    else:
        s = tl.full(rows.shape, 1, tl.float32)
    return s


@triton.jit
def _term(A, B, SA, SB, rm, rn, k, end, M, N, K,
          AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
          F: tl.constexpr, PRODUCT_SCALE: tl.constexpr):
    valid = (k < K) & (k < end)
    a = tl.load(A + k * M + rm, (rm < M) & valid, other=0).to(tl.float32)
    b = tl.load(B + k * N + rn, (rn < N) & valid, other=0).to(tl.float32)
    term = product(a[:, None], b[None, :], AF, BF)
    if PRODUCT_SCALE:
        sa = _k_scale(SA, rm, k, M, K, AD)[:, None]
        sb = _k_scale(SB, rn, k, N, K, BD)[None, :]
        # Indexing a constexpr tuple yields a plain tuple; re-wrap it or the call binding fails.
        term = scale_term(term, sa, sb, tl.constexpr(AD[3]), tl.constexpr(BD[3]),
                          AF[4] + BF[4], F, AD[4], BD[4])
    else:
        neg, e, m, nz, nan, inf = term
        term = neg, e, shift(m, F - AF[4] - BF[4]), nz, nan, inf
    neg, e, m, nz, nan, inf = term
    valid = valid & (rm[:, None] < M) & (rn[None, :] < N)
    return neg, e, m, nz & valid, nan & valid, inf & valid


@triton.jit
def _chunk(A, B, SA, SB, rm, rn, start, end, c,
           M, N, K,
           AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
           F: tl.constexpr, CS: tl.constexpr, RNE: tl.constexpr, PRODUCT_SCALE: tl.constexpr):
    ct = c_operand(c, F)
    e, nan, pos, neg = scan(tl.full(c.shape, -1000000, tl.int32), False, False, False, ct)
    for i in tl.static_range(CS):
        t = _term(A, B, SA, SB, rm, rn, start + i, end, M, N, K, AF, BF, AD, BD, F, PRODUCT_SCALE)
        e, nan, pos, neg = scan(e, nan, pos, neg, t)
    total = aligned(ct, e)
    for i in tl.static_range(CS):
        t = _term(A, B, SA, SB, rm, rn, start + i, end, M, N, K, AF, BF, AD, BD, F, PRODUCT_SCALE)
        total += aligned(t, e)
    return finish(total, e, c, nan, pos, neg, F, RNE)


@triton.jit
def _group(A, B, SA, SB, rm, rn, start, M, N, K,
           AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
           F: tl.constexpr, G: tl.constexpr, GS: tl.constexpr, GROUP_SCALE: tl.constexpr):
    e = tl.full((rm.shape[0], rn.shape[0]), -1000000, tl.int32)
    nan = tl.full(e.shape, False, tl.int1)
    pos = tl.full(e.shape, False, tl.int1)
    neg = tl.full(e.shape, False, tl.int1)
    for i in tl.static_range(GS):
        t = _term(A, B, SA, SB, rm, rn, start + i, K, M, N, K, AF, BF, AD, BD, G, False)
        e, nan, pos, neg = scan(e, nan, pos, neg, t)
    total = tl.full(e.shape, 0, tl.int64)
    for i in tl.static_range(GS):
        t = _term(A, B, SA, SB, rm, rn, start + i, K, M, N, K, AF, BF, AD, BD, G, False)
        total += aligned(t, e)
    nan |= pos & neg
    inf = (pos | neg) & ~nan
    negative = tl.where(inf, neg, total < 0)
    mag = tl.where(total < 0, -total, total).to(tl.uint64)
    term = negative, e, mag, (total != 0) & ~nan & ~inf, nan, inf
    if GROUP_SCALE:
        sa = _k_scale(SA, rm, start, M, K, AD)[:, None]
        sb = _k_scale(SB, rn, start, N, K, BD)[None, :]
        term = scale_term(term, sa, sb, tl.constexpr(AD[3]), tl.constexpr(BD[3]),
                          G, F, AD[4], BD[4], True)
    else:
        term = negative, e, shift(mag, F - G), (total != 0) & ~nan & ~inf, nan, inf
    tn, te, tm, tz, tnan, ti = term
    valid = (start < K) & (rm[:, None] < M) & (rn[None, :] < N)
    return tn, te, tm, tz & valid, tnan & valid, ti & valid


# Fast path (finite operands, narrow products): exponents of zero values carry _ZERO_E, so a
# product with a zero operand never wins a chunk maximum and needs no mask; any Inf/NaN in the
# chunk's rows, columns or running value sends that chunk to the exact general path instead.
_ZERO_E = tl.constexpr(-(1 << 20))
_NO_TERM = tl.constexpr(-(1 << 22))
_REAL_TERM = tl.constexpr(-(1 << 19))


@triton.jit
def _fast_side(P, S, rows, k, end, size, K, FMT: tl.constexpr, DESC: tl.constexpr, POW2: tl.constexpr):
    """One K column of one operand: sign, exponent, int32 significand, and an Inf/NaN flag.
    POW2 folds a power-of-two product scale into the exponent (NADPE E8M0 path)."""
    v = tl.load(P + k * size + rows, (rows < size) & (k < K) & (k < end), other=0).to(tl.float32)
    bits = v.to(tl.uint32, bitcast=True)
    special = (bits & 0x7F800000) == 0x7F800000
    zero = (bits & 0x7FFFFFFF) == 0
    if FMT[2] or FMT[1] < -126:
        neg, e, m = decode_f32(v, FMT[0], FMT[1], FMT[2], FMT[3])
        m = m.to(tl.int32)
    else:
        # decode_f32 in int32 for float formats whose grid lies in fp32's range (fp8/fp6/fp4,
        # bf16, fp16, fp32): a format subnormal is an fp32 normal (or, for bf16, an fp32
        # subnormal), so the significand only ever shifts right.
        raw_e = ((bits >> 23) & 255).to(tl.int32)
        e32 = tl.where(raw_e == 0, -126, raw_e - 127)
        sig = ((bits & 0x7FFFFF) | tl.where(raw_e != 0, 0x800000, 0)).to(tl.int32)
        e = tl.maximum(e32, FMT[1])
        m = sig >> tl.minimum(e - e32 + (23 - FMT[0]), 31)
        e = tl.where(m == 0, 0, e)  # as decode_f32 (an off-grid nonzero keeps exponent 0)
        neg = (bits >> 31) != 0
    if POW2:
        s = _k_scale(S, rows, k, size, K, DESC)
        sbits = s.to(tl.uint32, bitcast=True)
        special |= (sbits & 0x7F800000) == 0x7F800000
        _, es, _ = decode_f32(s, 0, -149, False, 0)
        zero |= (sbits & 0x7FFFFFFF) == 0
        if DESC[4]:
            zero |= sbits == 0x00400000  # E8M0 field 0 (2^-127) contributes zero
        e = e + es
    e = tl.where(zero, _ZERO_E, e)
    m = tl.where(zero, 0, m)
    return neg, e, m, special


@triton.jit
def _aligned_sum(A, B, SA, SB, rm, rn, start, end, emax, total, M, N, K,
                 AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
                 RADIX: tl.constexpr, COUNT: tl.constexpr, POW2: tl.constexpr, ACC: tl.constexpr,
                 TERM: tl.constexpr, TERM_SHIFT_MAX: tl.constexpr):
    """Pass 2 of an FDA over COUNT products at radix RADIX: one variable shift per product,
    computed in TERM (int32 whenever one aligned product fits) and summed in ACC."""
    C: tl.constexpr = RADIX - AF[4] - BF[4]
    for i in range(COUNT):  # a runtime loop: unrolled 32-product bodies spill registers
        na, ea, ma, _ = _fast_side(A, SA, rm, start + i, end, M, K, AF, AD, POW2)
        nb, eb, mb, _ = _fast_side(B, SB, rn, start + i, end, N, K, BF, BD, POW2)
        m = ma.to(TERM)[:, None] * mb.to(TERM)[None, :]
        s = (C - (emax - (ea[:, None] + eb[None, :]))).to(TERM)
        # Both select arms are evaluated: keep every shift count inside [0, TERM_SHIFT_MAX].
        left = m << tl.minimum(tl.maximum(s, 0), TERM_SHIFT_MAX)
        right = m >> tl.minimum(tl.maximum(-s, 0), TERM_SHIFT_MAX)
        v = tl.where(s >= 0, left, right)
        total += tl.where(na[:, None] ^ nb[None, :], -v, v).to(ACC)
    return total


@triton.jit
def _chunk_max(A, B, SA, SB, rm, rn, start, end, emax, M, N, K,
               AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
               COUNT: tl.constexpr, POW2: tl.constexpr):
    """Pass 1: the maximum product exponent, and whether any operand in the span is Inf/NaN."""
    xa = rm < 0
    xb = rn < 0
    for i in range(COUNT):
        _, ea, _, sa = _fast_side(A, SA, rm, start + i, end, M, K, AF, AD, POW2)
        _, eb, _, sb = _fast_side(B, SB, rn, start + i, end, N, K, BF, BD, POW2)
        emax = tl.maximum(emax, ea[:, None] + eb[None, :])
        xa |= sa
        xb |= sb
    return emax, tl.max(xa.to(tl.int32), 0) + tl.max(xb.to(tl.int32), 0)


@triton.jit
def _chunk_finite(A, B, SA, SB, rm, rn, start, end, c, cnz, emax, M, N, K,
                  AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
                  F: tl.constexpr, CS: tl.constexpr, RNE: tl.constexpr, POW2: tl.constexpr,
                  ACC: tl.constexpr, SHIFT_MAX: tl.constexpr, TERM: tl.constexpr,
                  TERM_SHIFT_MAX: tl.constexpr):
    """Pass 2 and normalization of a chunk whose products are all finite."""
    cneg, ce, cm, _, _, _ = c_operand(c, F)
    v = cm.to(ACC) >> tl.minimum(tl.maximum(emax - ce, 0), SHIFT_MAX).to(ACC)
    total = tl.where(cnz, tl.where(cneg, -v, v), 0)
    total = _aligned_sum(A, B, SA, SB, rm, rn, start, end, emax, total, M, N, K,
                         AF, BF, AD, BD, F, CS, POW2, ACC, TERM, TERM_SHIFT_MAX)
    return finish(total.to(tl.int64), tl.where(emax > _REAL_TERM, emax, -1000000), c,
                  False, False, False, F, RNE)


@triton.jit
def _chunk_fast(A, B, SA, SB, rm, rn, start, end, c,
                M, N, K,
                AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
                F: tl.constexpr, CS: tl.constexpr, RNE: tl.constexpr, POW2: tl.constexpr,
                ACC: tl.constexpr, SHIFT_MAX: tl.constexpr, TERM: tl.constexpr,
                TERM_SHIFT_MAX: tl.constexpr, FINITE: tl.constexpr):
    """_chunk with int32/int64 arithmetic when no Inf/NaN is involved; bit-identical results.

    FINITE (every operand value and scale finite, checked on the host) compiles no fallback: the
    running value is then the only possible Inf/NaN (an fp32 overflow of an earlier chunk), which
    the general path returns unchanged next to finite products."""
    cbits = c.to(tl.uint32, bitcast=True)
    _, ce, _, cnz, _, _ = c_operand(c, F)
    emax = tl.where(cnz, ce, _NO_TERM)
    emax, special = _chunk_max(A, B, SA, SB, rm, rn, start, end, emax, M, N, K, AF, BF, AD, BD, CS, POW2)
    if FINITE:
        out = _chunk_finite(A, B, SA, SB, rm, rn, start, end, c, cnz, emax, M, N, K,
                            AF, BF, AD, BD, F, CS, RNE, POW2, ACC, SHIFT_MAX,
                            TERM, TERM_SHIFT_MAX)
        c_special = (cbits & 0x7F800000) == 0x7F800000
        c_nan = c_special & ((cbits & 0x7FFFFF) != 0)
        out = tl.where(c_special, tl.where(c_nan, float("nan"), c), out)
    else:
        special += tl.max(tl.max(((cbits & 0x7F800000) == 0x7F800000).to(tl.int32), 1), 0)
        if special > 0:
            out = _chunk(A, B, SA, SB, rm, rn, start, end, c, M, N, K, AF, BF, AD, BD, F, CS, RNE, POW2)
        else:
            out = _chunk_finite(A, B, SA, SB, rm, rn, start, end, c, cnz, emax, M, N, K,
                                AF, BF, AD, BD, F, CS, RNE, POW2, ACC, SHIFT_MAX,
                                TERM, TERM_SHIFT_MAX)
    return out


@triton.jit
def _group_finite(A, B, SA, SB, rm, rn, start, emax, M, N, K,
                  AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
                  F: tl.constexpr, G: tl.constexpr, GS: tl.constexpr, GROUP_SCALE: tl.constexpr,
                  ACC: tl.constexpr, SHIFT_MAX: tl.constexpr, TERM: tl.constexpr,
                  TERM_SHIFT_MAX: tl.constexpr):
    """The group operand of GS finite products (same tuple as _group)."""
    total = tl.full(emax.shape, 0, ACC)
    total = _aligned_sum(A, B, SA, SB, rm, rn, start, K, emax, total, M, N, K,
                         AF, BF, AD, BD, G, GS, False, ACC, TERM, TERM_SHIFT_MAX)
    nonzero = total != 0
    negative = total < 0
    mag = tl.where(negative, -total, total).to(tl.int64).to(tl.uint64)
    e = tl.where(emax > _REAL_TERM, emax, -1000000)
    nan = negative & False
    term = negative, e, mag, nonzero, nan, nan
    if GROUP_SCALE:
        sa = _k_scale(SA, rm, start, M, K, AD)[:, None]
        sb = _k_scale(SB, rn, start, N, K, BD)[None, :]
        term = scale_term(term, sa, sb, tl.constexpr(AD[3]), tl.constexpr(BD[3]),
                          G, F, AD[4], BD[4], True)
    else:
        term = negative, e, shift(mag, F - G), nonzero, nan, nan
    tn, te, tm, tz, tnan, ti = term
    valid = (start < K) & (rm[:, None] < M) & (rn[None, :] < N)
    return tn, te, tm, tz & valid, tnan & valid, ti & valid


@triton.jit
def _group_fast(A, B, SA, SB, rm, rn, start, M, N, K,
                AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
                F: tl.constexpr, G: tl.constexpr, GS: tl.constexpr, GROUP_SCALE: tl.constexpr,
                ACC: tl.constexpr, SHIFT_MAX: tl.constexpr, TERM: tl.constexpr,
                TERM_SHIFT_MAX: tl.constexpr, FINITE: tl.constexpr):
    """_group with a narrow group sum when no Inf/NaN is involved; bit-identical results.
    FINITE (operands and scales checked finite on the host) compiles no fallback."""
    emax = tl.full((rm.shape[0], rn.shape[0]), _NO_TERM, tl.int32)
    emax, special = _chunk_max(A, B, SA, SB, rm, rn, start, K, emax, M, N, K, AF, BF, AD, BD, GS, False)
    if FINITE:
        term = _group_finite(A, B, SA, SB, rm, rn, start, emax, M, N, K, AF, BF, AD, BD,
                             F, G, GS, GROUP_SCALE, ACC, SHIFT_MAX, TERM, TERM_SHIFT_MAX)
    else:
        if special > 0:
            term = _group(A, B, SA, SB, rm, rn, start, M, N, K, AF, BF, AD, BD, F, G, GS, GROUP_SCALE)
        else:
            term = _group_finite(A, B, SA, SB, rm, rn, start, emax, M, N, K, AF, BF, AD, BD,
                                 F, G, GS, GROUP_SCALE, ACC, SHIFT_MAX, TERM, TERM_SHIFT_MAX)
    return term


@triton.jit
def _gdfs_tile(A, B, SA, SB, rm, rn, start, c,
               M, N, K,
               AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
               F: tl.constexpr, G: tl.constexpr, GS: tl.constexpr, KT: tl.constexpr,
               RNE: tl.constexpr, GROUP_SCALE: tl.constexpr,
               FAST: tl.constexpr, ACC: tl.constexpr, SHIFT_MAX: tl.constexpr, TERM: tl.constexpr,
               TERM_SHIFT_MAX: tl.constexpr, FINITE: tl.constexpr):
    ct = c_operand(c, F)
    e, nan, pos, neg = scan(tl.full(c.shape, -1000000, tl.int32), False, False, False, ct)
    # Compile-time indexing makes these S0..S7/E0..E7, with no dynamic register indexing.
    groups = ()
    for j in tl.static_range(KT // GS):
        if FAST:
            t = _group_fast(A, B, SA, SB, rm, rn, start + j * GS, M, N, K, AF, BF, AD, BD, F, G, GS,
                            GROUP_SCALE, ACC, SHIFT_MAX, TERM, TERM_SHIFT_MAX, FINITE)
        else:
            t = _group(A, B, SA, SB, rm, rn, start + j * GS, M, N, K, AF, BF, AD, BD, F, G, GS, GROUP_SCALE)
        groups += (t,)
        e, nan, pos, neg = scan(e, nan, pos, neg, t)
    total = aligned(ct, e)
    for j in tl.static_range(KT // GS):
        total += aligned(groups[j], e)
    return finish(total, e, c, nan, pos, neg, F, RNE)


_CONFIGS = [
    triton.Config({"BM": 16, "BN": 16}, num_warps=2),
    triton.Config({"BM": 16, "BN": 32}, num_warps=4),
    triton.Config({"BM": 32, "BN": 32}, num_warps=8),
]
# Tile shape never changes numerics; TRICAST_AUTOTUNE=0 compiles one config (tests, sweeps
# over many distinct specs) instead of benchmarking all of them for every new shape.
if os.environ.get("TRICAST_AUTOTUNE", "1") == "0":
    _CONFIGS = _CONFIGS[:1]


# M enters the tuning key only as a power-of-two bucket, so varying sequence lengths reuse one
# tuning result; shapes are runtime arguments, so they never trigger recompilation.
@triton.autotune(configs=_CONFIGS, key=["M_BUCKET", "N", "K", "MODE"])
@triton.jit
def _gemm(A, B, SA, SB, ALPHA_A, ALPHA_B, BIAS, OUT,
          M, N, K, M_BUCKET,
          AF: tl.constexpr, BF: tl.constexpr, AD: tl.constexpr, BD: tl.constexpr,
          MODE: tl.constexpr, F: tl.constexpr, CS: tl.constexpr, F2: tl.constexpr,
          G: tl.constexpr, GS: tl.constexpr, KT: tl.constexpr, RNE: tl.constexpr,
          PI: tl.constexpr, APPLY: tl.constexpr, DECOUPLED: tl.constexpr,
          HAS_ALPHA_A: tl.constexpr, HAS_ALPHA_B: tl.constexpr, HAS_BIAS: tl.constexpr,
          OUT_FMT: tl.constexpr, FAST: tl.constexpr, ACC: tl.constexpr, SHIFT_MAX: tl.constexpr,
          TERM: tl.constexpr, TERM_SHIFT_MAX: tl.constexpr,
          POW2: tl.constexpr, FINITE: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr):
    # A final partial chunk can step beyond K before its load mask is applied.
    K = tl.cast(K, tl.int64)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    c = tl.full((BM, BN), 0, tl.float32)
    if MODE == "cofda":
        if PI > 0:
            for start in range(0, K, PI):
                p = tl.full((BM, BN), 0, tl.float32)
                for j in range(0, tl.minimum(PI, K - start), CS):
                    if FAST:
                        p = _chunk_fast(A, B, SA, SB, rm, rn, start + j, tl.minimum(start + PI, K), p,
                                        M, N, K, AF, BF, AD, BD, F, CS, RNE, POW2, ACC, SHIFT_MAX,
                                        TERM, TERM_SHIFT_MAX, FINITE)
                    else:
                        p = _chunk(A, B, SA, SB, rm, rn, start + j, tl.minimum(start + PI, K), p,
                                   M, N, K, AF, BF, AD, BD, F, CS, RNE, APPLY == "product")
                if APPLY == "promote":
                    sa = _k_scale(SA, rm, start, M, K, AD)[:, None]
                    sb = _k_scale(SB, rn, start, N, K, BD)[None, :]
                    w = mul_rn(tl.broadcast_to(sa, (BM, BN)), tl.broadcast_to(sb, (BM, BN)))
                else:
                    w = tl.full((BM, BN), 1, tl.float32)
                c = fma_rn(p, w, c)
        else:
            for start in range(0, K, CS):
                if DECOUPLED:
                    if FAST:
                        p = _chunk_fast(A, B, SA, SB, rm, rn, start, K, tl.full((BM, BN), 0, tl.float32),
                                        M, N, K, AF, BF, AD, BD, F, CS, RNE, POW2, ACC, SHIFT_MAX,
                                        TERM, TERM_SHIFT_MAX, FINITE)
                    else:
                        p = _chunk(A, B, SA, SB, rm, rn, start, K, tl.full((BM, BN), 0, tl.float32),
                                   M, N, K, AF, BF, AD, BD, F, CS, RNE, APPLY == "product")
                    c = merge(c, p, F2, RNE)
                elif FAST:
                    c = _chunk_fast(A, B, SA, SB, rm, rn, start, K, c,
                                    M, N, K, AF, BF, AD, BD, F, CS, RNE, POW2, ACC, SHIFT_MAX,
                                    TERM, TERM_SHIFT_MAX, FINITE)
                else:
                    c = _chunk(A, B, SA, SB, rm, rn, start, K, c,
                               M, N, K, AF, BF, AD, BD, F, CS, RNE, APPLY == "product")
    elif MODE == "gdfs":
        for start in range(0, K, KT):
            c = _gdfs_tile(A, B, SA, SB, rm, rn, start, c,
                           M, N, K, AF, BF, AD, BD, F, G, GS, KT, RNE, APPLY == "group",
                           FAST, ACC, SHIFT_MAX, TERM, TERM_SHIFT_MAX, FINITE)
    elif MODE == "int_exact":
        if FAST:  # finite integers (validated): signed products summed in ACC
            total = tl.full((BM, BN), 0, ACC)
            for k in range(K):
                a = tl.load(A + k * M + rm, rm < M, other=0).to(tl.float32)
                b = tl.load(B + k * N + rn, rn < N, other=0).to(tl.float32)
                na, _, ma = decode_f32(a, AF[0], AF[1], AF[2], AF[3])
                nb, _, mb = decode_f32(b, BF[0], BF[1], BF[2], BF[3])
                ia = tl.where(na, -ma.to(ACC), ma.to(ACC))
                ib = tl.where(nb, -mb.to(ACC), mb.to(ACC))
                total += ia[:, None] * ib[None, :]
            c = pack_f32(total.to(tl.int64), -AF[4] - BF[4], 23, True, True)
        else:
            total = tl.full((BM, BN), 0, tl.int64)
            for k in range(K):
                t = _term(A, B, SA, SB, rm, rn, k, K, M, N, K, AF, BF, AD, BD, AF[4] + BF[4], False)
                total += tl.where(t[0], -t[2].to(tl.int64), t[2].to(tl.int64))
            c = pack_f32(total, -AF[4] - BF[4], 23, True, True)
    else:
        if MODE == "fp64":
            acc = tl.full((BM, BN), 0, tl.float64)
        else:
            acc = tl.full((BM, BN), 0, tl.float32)
        for k in range(K):
            a = tl.load(A + k * M + rm, rm < M, other=0).to(tl.float32)
            b = tl.load(B + k * N + rn, rn < N, other=0).to(tl.float32)
            if APPLY == "operand":
                a = mul_rn(a, _k_scale(SA, rm, k, M, K, AD))
                b = mul_rn(b, _k_scale(SB, rn, k, N, K, BD))
            if MODE == "fp64":
                acc = tl.fma(a[:, None].to(tl.float64), b[None, :].to(tl.float64), acc)
            else:
                acc = fma_rn(tl.broadcast_to(a[:, None], (BM, BN)),
                             tl.broadcast_to(b[None, :], (BM, BN)), acc)
        if MODE == "fp64":
            c = f64_to_f32_rn(acc)
        else:
            c = acc
    # Tensor and row scales remain in the epilogue even when the other side varies in K.
    if BD[0] == "tensor" or BD[0] == "row":
        c = mul_rn(tl.broadcast_to(_load_scale(SB, rn, 0, N, K, BD)[None, :], (BM, BN)), c)
    if AD[0] == "tensor" or AD[0] == "row":
        c = mul_rn(tl.broadcast_to(_load_scale(SA, rm, 0, M, K, AD)[:, None], (BM, BN)), c)
    if HAS_ALPHA_A or HAS_ALPHA_B:
        aa = tl.full((), 1, tl.float32)
        ab = tl.full((), 1, tl.float32)
        if HAS_ALPHA_A:
            aa = tl.load(ALPHA_A).to(tl.float32)
        if HAS_ALPHA_B:
            ab = tl.load(ALPHA_B).to(tl.float32)
        alpha = mul_rn(tl.broadcast_to(aa, (BM, BN)), tl.broadcast_to(ab, (BM, BN)))
        c = mul_rn(alpha, c)
    if HAS_BIAS:
        c = add_rn(c, tl.broadcast_to(tl.load(BIAS + rn, rn < N, other=0)[None, :].to(tl.float32), (BM, BN)))
    if OUT_FMT is not None:
        c = round_to_format(c, tl.full(c.shape, 0, tl.uint32), OUT_FMT, 0, False)
    tl.store(OUT + rm[:, None] * N + rn[None, :], c.to(OUT.dtype.element_ty),
             (rm[:, None] < M) & (rn[None, :] < N))


def _format(fmt: Format) -> tuple[int, int, bool, int, int]:
    if isinstance(fmt, FloatFormat):
        return fmt.mbits, fmt.emin, False, 0, fmt.mbits
    if isinstance(fmt, IntFormat):
        return 0, 0, True, fmt.frac_bits, fmt.frac_bits
    return 0, -149, False, 0, 0


def _scale_desc(op: Operand) -> tuple:
    arithmetic_fmt = (op.scale_fmt or FP32) if op.scale_kind == "k" else FP32
    return (op.scale_kind, op.k_domain, triton.cdiv(op.K, op.k_domain) if op.scale_kind == "k" else 1,
            _format(arithmetic_fmt), op.scale_kind == "k" and isinstance(arithmetic_fmt, Pow2Format)
            and arithmetic_fmt.ebits == 8 and arithmetic_fmt.bias == 127)


def _integer_bits(fmt: Format) -> int:
    if isinstance(fmt, IntFormat):
        return max(abs(fmt.qmin), abs(fmt.qmax)).bit_length() + max(0, -fmt.frac_bits)
    return 1


def _headroom(a: Operand, b: Operand, spec: MMASpec, apply: str) -> None:
    def check(radix: int, integer_bits: int, width: int) -> None:
        if radix + integer_bits + width.bit_length() > 62:
            raise ValueError("MMA aligned sum exceeds int64 headroom (F + int_bits + ceil(log2(n+1)) > 62)")

    ib = _integer_bits(a.fmt) + _integer_bits(b.fmt)
    sb = sum(_integer_bits(op.scale_fmt or FP32) for op in (a, b)
             if op.scale_kind == "k" and not isinstance(op.scale_fmt, Pow2Format))
    if spec.algorithm == "cofda":
        check(spec.f_bits, ib + (sb if apply == "product" else 0), spec.chunk_size)
        if spec.c_mode == "decoupled" and not spec.promote_interval:
            check(spec.f2_bits, 1, 1)
    elif spec.algorithm == "gdfs":
        check(spec.g_bits, ib, spec.group_size)
        check(spec.f_bits, ib + (spec.group_size - 1).bit_length() + (sb if apply == "group" else 0),
              spec.groups_per_tile)
    elif spec.algorithm == "int_exact":
        if not isinstance(a.fmt, IntFormat) or not isinstance(b.fmt, IntFormat):
            raise ValueError("int_exact requires integer operands")
        check(0, ib, a.K)
        if not bool(torch.isfinite(a.values).all()) or not bool(torch.isfinite(b.values).all()):
            raise ValueError("int_exact requires finite integer operands")


def _fast_path(a: Operand, b: Operand, spec: MMASpec, apply: str) -> tuple[bool, object, int, bool]:
    """(FAST, ACC dtype, SHIFT_MAX, POW2) for the finite-operand fast path of CoFDA and GDFS.

    Products must fit int32 (significand radix plus integer bits of both operands); the aligned
    sum uses int32 when F (G for GDFS) plus its headroom fits in 30 bits, else int64. Product-level
    scales are supported only when they are powers of two (a pure exponent add)."""
    ib = _integer_bits(a.fmt) + _integer_bits(b.fmt)
    if _format(a.fmt)[4] + _format(b.fmt)[4] + ib > 31:
        return False, tl.int64, 63, False
    pow2 = apply == "product" and all(op.scale_kind != "k" or isinstance(op.scale_fmt, Pow2Format)
                                      for op in (a, b))
    if spec.algorithm == "int_exact":  # operands validated finite; no specials to route
        narrow = ib + a.K.bit_length() <= 30
        return True, tl.int32 if narrow else tl.int64, 31 if narrow else 63, False
    if spec.algorithm == "cofda" and (apply != "product" or pow2):
        radix, width = spec.f_bits, spec.chunk_size
    elif spec.algorithm == "gdfs":
        radix, width = spec.g_bits, spec.group_size
    else:
        return False, tl.int64, 63, False
    narrow = radix + ib + (width + 1).bit_length() <= 30
    return True, tl.int32 if narrow else tl.int64, 31 if narrow else 63, pow2


def _term_width(a: Operand, b: Operand, spec: MMASpec) -> tuple[object, int]:
    """(dtype, shift cap) for one aligned product: its magnitude stays below 2^(R + int bits), R
    the FDA radix (F, or G for GDFS groups), so it is int32 whenever that fits even if the sum of
    a chunk needs int64."""
    ib = _integer_bits(a.fmt) + _integer_bits(b.fmt)
    radix = spec.g_bits if spec.algorithm == "gdfs" else spec.f_bits
    return (tl.int32, 31) if radix + ib <= 30 else (tl.int64, 63)


def _validate_operand(op: Operand) -> None:
    if not op.values.is_cuda or not op.values.is_floating_point():
        raise ValueError("gemm_triton requires floating CUDA grid-value tensors")
    if op.scale_kind not in ("none", "tensor", "row", "k"):
        raise ValueError(f"unknown scale_kind {op.scale_kind!r}")
    if op.scale is not None:
        if op.scale_kind == "tensor":
            shape = ()
        elif op.scale_kind == "row":
            shape = (op.rows, 1)
        else:
            shape = (op.rows, triton.cdiv(op.K, op.k_domain))
        if tuple(op.scale.shape) != shape:
            raise ValueError(f"scale shape must be {shape}, got {tuple(op.scale.shape)}")
        if op.scale.device != op.values.device:
            raise ValueError("scale and values must be on the same CUDA device")
    if op.alpha is not None and (op.alpha.numel() != 1 or op.alpha.device != op.values.device):
        raise ValueError("alpha must be a scalar on the operand device")


def gemm_triton(a: Operand, b: Operand, spec: MMASpec, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Emulate independently per output; inputs must already be on their format grids.

    K-major packing preserves grid values in fp32. No reference computation is
    used here: only the shared scale-policy resolver is imported from that layer.
    GPU compilation and bit-exact parity are tested in tests/gpu/test_triton_mma.py.
    """
    from ..reference.mma import resolve_scale_apply

    if a.K != b.K or a.values.device != b.values.device:
        raise ValueError("operands must have the same K and CUDA device")
    if max(a.values.numel(), b.values.numel(), a.rows * b.rows) >= 2**31:
        raise ValueError("Triton MMA requires fewer than 2^31 input/output elements (int32 indexing)")
    _validate_operand(a)
    _validate_operand(b)
    apply = resolve_scale_apply(spec, a, b)
    dtype = container_dtype(spec.out_format)
    out_fmt = None if spec.out_format in (FP32, BF16, FP16) else fmt_constexprs(spec.out_format)["FMT"]
    if spec.algorithm == "int_exact" and (a.scale_kind == "k" or b.scale_kind == "k"):
        raise ValueError("int_exact does not support K-varying scales")
    _headroom(a, b, spec, apply)
    fast, acc, shift_max, pow2 = _fast_path(a, b, spec, apply)
    term, term_shift_max = _term_width(a, b, spec)
    # Finite operands (the normal case; cached for weights) get a kernel without Inf/NaN fallback.
    finite = fast and a.all_finite() and b.all_finite()
    if bias is not None and (bias.shape != (b.rows,) or bias.device != a.values.device):
        raise ValueError("bias must have shape [N] on the operand device")
    out = torch.empty((a.rows, b.rows), device=a.values.device, dtype=dtype)
    if a.rows == 0 or b.rows == 0:
        return out
    at, bt = a.k_major(), b.k_major()
    sa = a.scale.to(torch.float32).contiguous() if a.scale is not None else at
    sb = b.scale.to(torch.float32).contiguous() if b.scale is not None else bt
    aa = a.alpha.to(torch.float32).contiguous() if a.alpha is not None else at
    ab = b.alpha.to(torch.float32).contiguous() if b.alpha is not None else bt
    bias_arg = bias.to(torch.float32).contiguous() if bias is not None else at
    with torch.cuda.device(a.values.device):
        _gemm[lambda meta: (triton.cdiv(a.rows, meta["BM"]), triton.cdiv(b.rows, meta["BN"]))](
            at, bt, sa, sb, aa, ab, bias_arg, out, a.rows, b.rows, a.K,
            triton.next_power_of_2(min(a.rows, 4096)),
            _format(a.fmt), _format(b.fmt), _scale_desc(a), _scale_desc(b),
            spec.algorithm, spec.f_bits, spec.chunk_size, spec.f2_bits, spec.g_bits,
            spec.group_size, spec.k_tile, spec.norm_rounding == "rne", spec.promote_interval,
            apply, spec.c_mode == "decoupled", a.alpha is not None, b.alpha is not None,
            bias is not None, out_fmt, fast, acc, shift_max, term, term_shift_max, pow2, finite,
            enable_fp_fusion=False,
        )
    return out
