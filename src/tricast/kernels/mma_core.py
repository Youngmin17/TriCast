"""Integer FDA primitives implementing tricast.reference.mma, without floating rescaling.

Terms are (negative, exponent, magnitude, nonzero, nan, inf). Nonzero describes
inputs *before* radix truncation. Wide products retain both limbs until the one
specified truncation; the caller validates that aligned sums fit signed int64.
"""

import triton
import triton.language as tl


@triton.jit
def rshift(x, d):
    """Logical uint64 shift; no branch evaluates an LLVM over-wide shift."""
    return tl.where(d > 63, 0, x.to(tl.uint64) >> tl.minimum(tl.maximum(d, 0), 63))


@triton.jit
def lshift(x, d):
    return tl.where(d > 63, 0, x.to(tl.uint64) << tl.minimum(tl.maximum(d, 0), 63))


@triton.jit
def shift(x, d):
    """Positive d shifts left, negative d truncates right."""
    return tl.where(d >= 0, lshift(x, d), rshift(x, -d))


@triton.jit
def leading_bit(x):
    z = tl.inline_asm_elementwise(
        "clz.b64 $0, $1;", constraints="=r,l", args=[x.to(tl.uint64)],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    return 63 - z


@triton.jit
def classify(v):
    bits = v.to(tl.float32).to(tl.uint32, bitcast=True)
    mag = bits & 0x7FFFFFFF
    return (bits >> 31) != 0, mag == 0, mag > 0x7F800000, mag == 0x7F800000


@triton.jit
def decode_f32(v, MBITS: tl.constexpr, EMIN: tl.constexpr, IS_INT: tl.constexpr, FRAC: tl.constexpr):
    """Decode finite grid values; specials have zero magnitude, not a float cast.

    Pow2 uses MBITS=0, EMIN=-149. Float subnormals remain format-native;
    integer significands are |k| and their exponent is always zero.
    """
    bits = v.to(tl.float32).to(tl.uint32, bitcast=True)
    raw_e = (bits >> 23) & 255
    sig = ((bits & 0x7FFFFF) | tl.where(raw_e != 0, 0x800000, 0)).to(tl.uint64)
    e = tl.where(raw_e == 0, -126, raw_e.to(tl.int32) - 127)
    if IS_INT:
        m = shift(sig, e + FRAC - 23)
        e = tl.full(v.shape, 0, tl.int32)
    else:
        if EMIN >= -126:
            native_e = tl.maximum(e, EMIN)
        else:
            # Running c and pow2 normalize fp32-container subnormals.
            actual_e = tl.where(raw_e == 0, leading_bit(sig) - 149, e)
            native_e = tl.maximum(actual_e, EMIN)
        m = shift(sig, e - native_e + MBITS - 23)
        e = native_e
    m = tl.where(raw_e == 255, 0, m)
    e = tl.where(m == 0, 0, e)
    return ((bits >> 31) != 0) & ((bits & 0x7FFFFFFF) != 0), e, m


@triton.jit
def c_operand(c_f32, F: tl.constexpr):
    """Normalize fp32 subnormals before moving the significand to radix F."""
    neg, zero, nan, inf = classify(c_f32)
    _, e, m = decode_f32(c_f32, 23, -149, False, 0)
    m = shift(m, F - 23)
    return neg, e, m, ~(zero | nan | inf), nan, inf


@triton.jit
def round_shift(m, d, RNE: tl.constexpr):
    q = rshift(m, d)
    if RNE:
        # d=64 is legal mathematically, but not as a machine shift count.
        mask = lshift(tl.full((), 1, tl.uint64), d) - 1
        rem = m & mask
        half = lshift(tl.full((), 1, tl.uint64), d - 1)
        inc = (d > 0) & (d <= 64) & ((rem > half) | ((rem == half) & ((q & 1) != 0)))
        q += inc.to(tl.uint64)
    return tl.where(d > 0, q, lshift(m, -d))


@triton.jit
def pack_f32(S, exponent, PRECISION: tl.constexpr, RNE: tl.constexpr, SUB_RNE: tl.constexpr):
    """Assemble S * 2**exponent; FDA and integer conversion differ below normal."""
    negative = S < 0
    m = tl.where(negative, -S, S).to(tl.uint64)
    lead = leading_bit(m)
    b = lead + exponent + 127
    sig = round_shift(m, lead - PRECISION, RNE)
    carry = sig >= (1 << (PRECISION + 1))
    sig = tl.where(carry, rshift(sig, 1), sig)
    normal = ((b + carry.to(tl.int32)).to(tl.uint32) << 23) | (
        (sig.to(tl.uint32) << (23 - PRECISION)) & 0x7FFFFF
    )
    normal = tl.where(b + carry.to(tl.int32) >= 255, 0x7F800000, normal)
    sub = round_shift(m, -exponent - 149, SUB_RNE).to(tl.uint32)
    mag = tl.where(b <= 0, sub, normal)
    bits = mag | (negative.to(tl.uint32) << 31)
    bits = tl.where(S == 0, 0, bits).to(tl.uint32)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def fixed_to_f32(S_int64, Emax, F: tl.constexpr, NORM_RNE: tl.constexpr):
    """NADPE normalization, with one-step normal RNE and truncating subnormals."""
    precision: tl.constexpr = F if F < 23 else 23
    return pack_f32(S_int64, Emax - F, precision, NORM_RNE, False)


@triton.jit
def mul_wide(lo, hi, m):
    """128-bit unsigned product by a <=24-bit scale significand."""
    high = tl.inline_asm_elementwise(
        "mul.hi.u64 $0, $1, $2;", constraints="=l,l,l", args=[lo, m],
        dtype=tl.uint64, is_pure=True, pack=1,
    )
    return lo * m, high + hi * m


@triton.jit
def wide_radix(lo, hi, D: tl.constexpr):
    """Move a 128-bit significand by D bits; the result must fit int64."""
    if D >= 0:
        out = lshift(lo, D)
    elif D > -64:
        out = rshift(lo, -D) | lshift(hi, 64 + D)
    else:
        out = rshift(hi, -D - 64)
    return out


@triton.jit
def scale_term(term, sa, sb, SA_FMT: tl.constexpr, SB_FMT: tl.constexpr,
               RADIX: tl.constexpr, F: tl.constexpr, FIELD0_A: tl.constexpr, FIELD0_B: tl.constexpr,
               GROUP: tl.constexpr = False):
    neg, e, m, nz, nan, inf = term
    na, ea, ma = decode_f32(sa, SA_FMT[0], SA_FMT[1], SA_FMT[2], SA_FMT[3])
    nb, eb, mb = decode_f32(sb, SB_FMT[0], SB_FMT[1], SB_FMT[2], SB_FMT[3])
    _, za, nana, infa = classify(sa)
    _, zb, nanb, infb = classify(sb)
    if FIELD0_A:
        za |= sa.to(tl.uint32, bitcast=True) == 0x00400000
    if FIELD0_B:
        zb |= sb.to(tl.uint32, bitcast=True) == 0x00400000
    zero = (~nz & ~inf) | za | zb
    any_inf = inf | infa | infb
    nan |= nana | nanb
    if not GROUP:
        nan |= zero & any_inf
    inf = any_inf & ~zero & ~nan
    lo, hi = mul_wide(m.to(tl.uint64), tl.full((), 0, tl.uint64), ma)
    lo, hi = mul_wide(lo, hi, mb)
    out = wide_radix(lo, hi, F - RADIX - SA_FMT[4] - SB_FMT[4])
    return neg ^ na ^ nb, e + ea + eb, out, ~zero & ~nan & ~inf, nan, inf


@triton.jit
def product(a, b, AF: tl.constexpr, BF: tl.constexpr):
    na, ea, ma = decode_f32(a, AF[0], AF[1], AF[2], AF[3])
    nb, eb, mb = decode_f32(b, BF[0], BF[1], BF[2], BF[3])
    _, za, nana, infa = classify(a)
    _, zb, nanb, infb = classify(b)
    zero = za | zb
    nan = nana | nanb | ((infa | infb) & zero)
    inf = (infa | infb) & ~zero & ~nan
    return na ^ nb, ea + eb, ma * mb, ~zero & ~nan & ~inf, nan, inf


@triton.jit
def scan(emax, nan, pos, neg_inf, term):
    neg, e, _, nz, tn, ti = term
    return tl.maximum(emax, tl.where(nz, e, -1000000)), nan | tn, pos | (ti & ~neg), neg_inf | (ti & neg)


@triton.jit
def aligned(term, emax):
    neg, e, m, nz, _, _ = term
    mag = tl.where(nz, rshift(m, emax - e), 0).to(tl.int64)
    return tl.where(neg, -mag, mag)


@triton.jit
def finish(S, emax, c, nan, pos, neg_inf, F: tl.constexpr, RNE: tl.constexpr):
    out = fixed_to_f32(S, emax, F, RNE)
    out = tl.where(emax == -1000000, c, out)
    out = tl.where(pos, float("inf"), out)
    out = tl.where(neg_inf, -float("inf"), out)
    return tl.where(nan | (pos & neg_inf), float("nan"), out)


@triton.jit
def merge(c, p, F: tl.constexpr, RNE: tl.constexpr):
    ct = c_operand(c, F)
    pt = c_operand(p, F)
    e, nan, pos, neg = scan(tl.full(c.shape, -1000000, tl.int32), False, False, False, ct)
    e, nan, pos, neg = scan(e, nan, pos, neg, pt)
    return finish(aligned(ct, e) + aligned(pt, e), e, c, nan, pos, neg, F, RNE)
