// WGSL for the browser GEMMs, exported as strings so the module loads without fetch.
//
// COFDA_WGSL: TriCast's CoFDA accumulation and epilogue for
// FP8 E4M3 operands, bit-exact with tricast.reference.mma (the executable definition). WGSL has neither 64-bit integers nor f64,
// so exact sums are i32 (when the chunk bound fits) or 64-bit two's complement in vec2<u32> (lo, hi),
// and the epilogue's fp32 multiplies are done in integers (no reliance on the GPU's float rounding,
// subnormal or NaN behaviour). One invocation computes one output; x = column n, y = row m.
//
// Layout: A row-major, KW = ceil(K / 4) words per row (byte k of row m in word m·KW + k/4, byte k % 4);
// B k4-major, word (k/4)·N + n holds codes k..k+3 of row n. Bytes past K are zero codes.
//
// F32_WGSL: the "native-style" comparison, an f32 fma chain on dequantized operands (a·s_a, b·s_b).
// It relies on the GPU's f32 arithmetic and is not part of the bit-exact contract.

const SHARED = /* wgsl */ `
struct Params { M: u32, N: u32, K: u32, KW: u32, scale_a: u32, scale_b: u32, pad0: u32, pad1: u32 }

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;

fn code_a(m: u32, k: u32) -> u32 { return (A[m * P.KW + (k >> 2u)] >> ((k & 3u) * 8u)) & 0xFFu; }
fn code_b(n: u32, k: u32) -> u32 { return (B[(k >> 2u) * P.N + n] >> ((k & 3u) * 8u)) & 0xFFu; }
`;

export const COFDA_WGSL = SHARED + /* wgsl */ `
override F: u32 = 13u;          // f_bits: datapath fraction bits
override CS: u32 = 32u;         // chunk_size
override F2: u32 = 23u;         // f2_bits (decoupled)
override DECOUPLED: bool = false;
override RNE: bool = false;     // norm_rounding rne (else rtz)
override WIDE: bool = false;    // 64-bit chunk sums (CS·225·2^(F-6) ≥ 2^31)
override WORDS: bool = true;    // CS % 4 == 0: every chunk starts on a word

const NAN_BITS: u32 = 0x7FC00000u;
const ZERO64 = vec2<u32>(0u, 0u);

// E4M3 (fn): exponent field clamped to 1 (true exponent = field - 7), significand at radix 3.
fn ebias(code: u32) -> u32 { return max((code >> 3u) & 15u, 1u); }
fn sig3(code: u32) -> u32 { return select(code & 7u, (code & 7u) | 8u, (code & 0x78u) != 0u); }
fn nonzero(code: u32) -> bool { return (code & 0x7Fu) != 0u; }
fn is_nan_code(code: u32) -> bool { return (code & 0x7Fu) == 0x7Fu; }
fn word_has_nan(w: u32) -> bool { return (((w & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) != 0u; }

// 64-bit two's complement helpers on (lo, hi).
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn shl64(v: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) { return v; }
  if (s < 32u) { return vec2<u32>(v.x << s, (v.y << s) | (v.x >> (32u - s))); }
  if (s < 64u) { return vec2<u32>(0u, v.x << (s - 32u)); }
  return ZERO64;
}
fn shr64(v: vec2<u32>, s: u32) -> vec2<u32> {
  if (s == 0u) { return v; }
  if (s < 32u) { return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s); }
  if (s < 64u) { return vec2<u32>(v.y >> (s - 32u), 0u); }
  return ZERO64;
}
// v·2^s for a nonnegative v: left shifts exact, right shifts truncate, |s| >= 64 gives 0.
fn shift64(v: vec2<u32>, s: i32) -> vec2<u32> {
  if (s >= 64 || s <= -64) { return ZERO64; }
  if (s >= 0) { return shl64(v, u32(s)); }
  return shr64(v, u32(-s));
}
fn bit64(v: vec2<u32>, i: u32) -> bool {
  if (i < 32u) { return ((v.x >> i) & 1u) == 1u; }
  if (i < 64u) { return ((v.y >> (i - 32u)) & 1u) == 1u; }
  return false;
}
fn low_bits_set(v: vec2<u32>, count: u32) -> bool {
  if (count == 0u) { return false; }
  if (count < 32u) { return (v.x & ((1u << count) - 1u)) != 0u; }
  if (count == 32u) { return v.x != 0u; }
  if (count < 64u) { return v.x != 0u || (v.y & ((1u << (count - 32u)) - 1u)) != 0u; }
  return v.x != 0u || v.y != 0u;
}
// v >> s rounded to nearest, ties to even; the result fits 32 bits at every call site.
fn shr64_rne(v: vec2<u32>, s: u32) -> u32 {
  let kept = shr64(v, s).x;
  if (s == 0u) { return kept; }
  let up = bit64(v, s - 1u) && (low_bits_set(v, s - 1u) || (kept & 1u) == 1u);
  return kept + select(0u, 1u, up);
}

// The fp32 register as an FDA term (§4.3 step 1): subnormals normalised to a 24-bit significand.
struct Reg { zero: bool, neg: bool, e: i32, sig: u32 }
fn decode_reg(bits: u32) -> Reg {
  let field = (bits >> 23u) & 0xFFu;
  let frac = bits & 0x7FFFFFu;
  let neg = (bits >> 31u) == 1u;
  if (field == 0u) {
    if (frac == 0u) { return Reg(true, neg, 0, 0u); }
    let lead = firstLeadingBit(frac);
    return Reg(false, neg, i32(lead) - 149, frac << (23u - lead));
  }
  return Reg(false, neg, i32(field) - 127, frac | 0x800000u);
}
fn add_reg(sum: vec2<u32>, r: Reg, emax: i32, radix: u32) -> vec2<u32> {
  let t = shift64(vec2<u32>(r.sig, 0u), i32(radix) - 23 - (emax - r.e));
  return select(add64(sum, t), sub64(sum, t), r.neg);
}

// §4.3 step 5: to_fp32(S, Emax, radix, norm). S is the exact chunk sum at radix "radix".
fn to_fp32(sum: vec2<u32>, emax: i32, radix: u32) -> u32 {
  let neg = (sum.y >> 31u) == 1u;
  let mag = select(sum, sub64(ZERO64, sum), neg);
  if (mag.x == 0u && mag.y == 0u) { return 0u; }
  var lead: i32 = i32(firstLeadingBit(mag.x));
  if (mag.y != 0u) { lead = 32 + i32(firstLeadingBit(mag.y)); }
  let sign = select(0u, 0x80000000u, neg);
  let biased = lead + emax - i32(radix) + 127;
  if (biased <= 0) { return sign | shift64(mag, 149 + emax - i32(radix)).x; }  // subnormal: truncates
  let prec = min(radix, 23u);
  let amount = i32(prec) - lead;
  var kept: u32;
  if (amount >= 0) {
    kept = mag.x << u32(amount);
  } else if (RNE) {
    kept = shr64_rne(mag, u32(-amount));
  } else {
    kept = shr64(mag, u32(-amount)).x;
  }
  var e = biased;
  if (kept >= (2u << prec)) { kept = kept >> 1u; e = e + 1; }
  if (e >= 255) { return sign | 0x7F800000u; }
  return sign | (u32(e) << 23u) | ((kept << (23u - prec)) & 0x7FFFFFu);
}

// Pass 1 over a chunk: largest exponent-field sum of the non-zero products (0 = none), and NaN codes.
struct Pass1 { es: u32, nan: bool }
fn pass1(m: u32, n: u32, start: u32, end: u32) -> Pass1 {
  var es = 0u;
  var nan = false;
  if (WORDS) {
    let words_end = (end + 3u) >> 2u;
    for (var w = start >> 2u; w < words_end; w += 1u) {
      let wa = A[m * P.KW + w];
      let wb = B[w * P.N + n];
      nan = nan | word_has_nan(wa) | word_has_nan(wb);
      for (var j = 0u; j < 32u; j += 8u) {
        let ca = (wa >> j) & 0xFFu;
        let cb = (wb >> j) & 0xFFu;
        es = max(es, select(0u, ebias(ca) + ebias(cb), nonzero(ca) && nonzero(cb)));
      }
    }
  } else {
    for (var k = start; k < end; k += 1u) {
      let ca = code_a(m, k);
      let cb = code_b(n, k);
      nan = nan | is_nan_code(ca) | is_nan_code(cb);
      es = max(es, select(0u, ebias(ca) + ebias(cb), nonzero(ca) && nonzero(cb)));
    }
  }
  return Pass1(es, nan);
}

// Aligned product m_a·m_b at radix F, truncated to Emax: shift = base + field sum, base = F - 20 - Emax.
fn term32(ca: u32, cb: u32, base: i32) -> i32 {
  let mm = sig3(ca) * sig3(cb);
  let s = base + i32(ebias(ca) + ebias(cb));
  let t = select(mm >> u32(min(-s, 31)), mm << u32(max(s, 0)), s >= 0);
  return select(i32(t), -i32(t), ((ca ^ cb) & 0x80u) != 0u);
}
fn add_term64(sum: vec2<u32>, ca: u32, cb: u32, base: i32) -> vec2<u32> {
  let mm = sig3(ca) * sig3(cb);
  let s = base + i32(ebias(ca) + ebias(cb));
  var t = vec2<u32>(mm >> u32(min(-s, 31)), 0u);
  if (s >= 0) { t = shl64(vec2<u32>(mm, 0u), u32(s)); }
  return select(add64(sum, t), sub64(sum, t), ((ca ^ cb) & 0x80u) != 0u);
}

// Pass 2: S = Σ ±(m_F >> (Emax - e)) over the chunk, exact.
fn pass2(m: u32, n: u32, start: u32, end: u32, base: i32) -> vec2<u32> {
  var narrow: i32 = 0;
  var wide = ZERO64;
  if (WORDS) {
    let words_end = (end + 3u) >> 2u;
    for (var w = start >> 2u; w < words_end; w += 1u) {
      let wa = A[m * P.KW + w];
      let wb = B[w * P.N + n];
      for (var j = 0u; j < 32u; j += 8u) {
        let ca = (wa >> j) & 0xFFu;
        let cb = (wb >> j) & 0xFFu;
        if (WIDE) { wide = add_term64(wide, ca, cb, base); } else { narrow += term32(ca, cb, base); }
      }
    }
  } else {
    for (var k = start; k < end; k += 1u) {
      let ca = code_a(m, k);
      let cb = code_b(n, k);
      if (WIDE) { wide = add_term64(wide, ca, cb, base); } else { narrow += term32(ca, cb, base); }
    }
  }
  if (WIDE) { return wide; }
  return vec2<u32>(u32(narrow), select(0u, 0xFFFFFFFFu, narrow < 0));
}

// Decoupled second stage: fda([P at radix F2], c, F2).
fn fda_regs(p: u32, c: u32) -> u32 {
  let rp = decode_reg(p);
  let rc = decode_reg(c);
  if (rp.zero && rc.zero) { return c; }
  var emax: i32 = -100000;
  if (!rp.zero) { emax = rp.e; }
  if (!rc.zero) { emax = max(emax, rc.e); }
  var sum = ZERO64;
  if (!rp.zero) { sum = add_reg(sum, rp, emax, F2); }
  if (!rc.zero) { sum = add_reg(sum, rc, emax, F2); }
  return to_fp32(sum, emax, F2);
}

// IEEE binary32 multiply, round to nearest even, in integers (subnormals, overflow to Inf).
fn mul24(a: u32, b: u32) -> vec2<u32> {
  let al = a & 0xFFFFu;
  let ah = a >> 16u;
  let bl = b & 0xFFFFu;
  let bh = b >> 16u;
  let ll = al * bl;
  let mid = ah * bl + al * bh;
  let lo = ll + (mid << 16u);
  return vec2<u32>(lo, ah * bh + (mid >> 16u) + select(0u, 1u, lo < ll));
}
fn mul_f32(x: u32, y: u32) -> u32 {
  let sign = (x ^ y) & 0x80000000u;
  let ex = (x >> 23u) & 0xFFu;
  let ey = (y >> 23u) & 0xFFu;
  let fx = x & 0x7FFFFFu;
  let fy = y & 0x7FFFFFu;
  if ((ex == 255u && fx != 0u) || (ey == 255u && fy != 0u)) { return NAN_BITS; }
  let zx = ex == 0u && fx == 0u;
  let zy = ey == 0u && fy == 0u;
  if (ex == 255u || ey == 255u) {
    if (zx || zy) { return NAN_BITS; }
    return sign | 0x7F800000u;
  }
  if (zx || zy) { return sign; }
  let rx = decode_reg(x);
  let ry = decode_reg(y);
  let p = mul24(rx.sig, ry.sig);                       // [2^46, 2^48)
  let top = select(46u, 47u, (p.y >> 15u) != 0u);
  var be = rx.e + ry.e + i32(top) - 46 + 127;          // biased exponent of the leading bit
  var s = top - 23u;
  if (be <= 0) {                                       // subnormal result: quantum 2^-149
    s = s + u32(1 - be);
    return sign | shr64_rne(p, s);                     // 2^23 here is the smallest normal
  }
  var kept = shr64_rne(p, s);
  if (kept >= 0x1000000u) { kept = kept >> 1u; be = be + 1; }
  if (be >= 255) { return sign | 0x7F800000u; }
  return sign | (u32(be) << 23u) | (kept & 0x7FFFFFu);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (n >= P.N || m >= P.M) { return; }
  var c: u32 = 0u;  // the fp32 register, +0
  var nan = false;
  for (var start = 0u; start < P.K; start += CS) {
    let end = min(start + CS, P.K);
    let p1 = pass1(m, n, start, end);
    if (p1.nan) { nan = true; break; }  // a NaN product makes the register NaN for good
    if (DECOUPLED) {
      var p = 0u;
      if (p1.es != 0u) {
        let emax = i32(p1.es) - 14;
        p = to_fp32(pass2(m, n, start, end, i32(F) - 20 - emax), emax, F);
      }
      c = fda_regs(p, c);
    } else {
      let rc = decode_reg(c);
      if (p1.es == 0u && rc.zero) { continue; }  // nothing non-zero: c unchanged
      var emax: i32 = -100000;
      var sum = ZERO64;
      if (p1.es != 0u) { emax = i32(p1.es) - 14; }
      if (!rc.zero) { emax = max(emax, rc.e); }
      if (p1.es != 0u) { sum = pass2(m, n, start, end, i32(F) - 20 - emax); }
      if (!rc.zero) { sum = add_reg(sum, rc, emax, F); }
      c = to_fp32(sum, emax, F);
    }
  }
  OUT[m * P.N + n] = select(mul_f32(P.scale_a, mul_f32(P.scale_b, c)), NAN_BITS, nan);
}
`;

export const F32_WGSL = SHARED + /* wgsl */ `
fn e4m3(code: u32) -> f32 {
  let field = (code >> 3u) & 15u;
  let mant = code & 7u;
  if ((code & 0x7Fu) == 0x7Fu) { return bitcast<f32>(code | 0x7FC00000u); }
  if (field == 0u) { return select(1.0, -1.0, (code & 0x80u) != 0u) * f32(mant) * 0.001953125; }
  return bitcast<f32>(((code & 0x80u) << 24u) | ((field + 120u) << 23u) | (mant << 20u));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (n >= P.N || m >= P.M) { return; }
  let sa = bitcast<f32>(P.scale_a);
  let sb = bitcast<f32>(P.scale_b);
  var acc = 0.0;
  for (var w = 0u; w < P.KW; w += 1u) {
    let wa = A[m * P.KW + w];
    let wb = B[w * P.N + n];
    for (var j = 0u; j < 4u; j += 1u) {
      if (w * 4u + j < P.K) {
        acc = fma(e4m3((wa >> (8u * j)) & 0xFFu) * sa, e4m3((wb >> (8u * j)) & 0xFFu) * sb, acc);
      }
    }
  }
  OUT[m * P.N + n] = bitcast<u32>(acc);
}
`;
