// Exact JavaScript reference for TriCast's CoFDA GEMM on FP8 E4M3 operands: the browser-side twin of
// tricast.reference.mma (the executable definition). D = A·Bᵀ, A [M, K] and B [N, K] as E4M3
// (e4m3fn) codes, per-tensor fp32 scales applied in the epilogue, fp32 output as bit patterns.
// app/web/demo/webgpu/golden.json (app/webgpu_golden.py, TriCast's reference backend) checks it bit for bit.
//
// Integers stay exact: an aligned term is below 2^50, so it is an exact Number. A chunk sum is accumulated
// in a Number when its bound fits 2^53 (every F ≤ 43, larger F with short chunks), otherwise in a BigInt.

const NAN_BITS = 0x7fc00000;
const INF_BITS = 0x7f800000;

// Settings the browser path accepts (WebGPU v1): cofda, fp32 output, per-tensor scales, no promotion.
export const LIMITS = {maxBits: 48, maxChunk: 128};

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
export function bitsToFloat(bits) {
  u32[0] = bits;
  return f32[0];
}
export function floatToBits(value) {
  if (Number.isNaN(value)) return NAN_BITS;
  f32[0] = value;
  return u32[0];
}

// Powers of two from repeated doubling (exact), indexed by exponent + 64 for exponents in [-64, 64].
const POW2 = new Float64Array(129);
POW2[64] = 1;
for (let i = 1; i <= 64; i += 1) {
  POW2[64 + i] = POW2[63 + i] * 2;
  POW2[64 - i] = POW2[65 - i] / 2;
}
const pow2 = (e) => POW2[64 + e];

// E4M3 (fn) decode tables: exponent field clamped to 1 (true exponent = EXP - 7), significand at radix 3.
const EXP = new Uint8Array(256);
const MAN = new Uint8Array(256);
const SGN = new Uint8Array(256);
const ISNAN = new Uint8Array(256);
const VALUE = new Float64Array(256);
for (let code = 0; code < 256; code += 1) {
  const field = (code >> 3) & 15;
  EXP[code] = Math.max(field, 1);
  MAN[code] = field ? 8 | (code & 7) : code & 7;
  SGN[code] = code >> 7;
  ISNAN[code] = (code & 0x7f) === 0x7f ? 1 : 0;
  VALUE[code] = ISNAN[code] ? NaN : (SGN[code] ? -1 : 1) * MAN[code] * 2 ** (EXP[code] - 10);
}

export function decodeE4M3(code) {
  return VALUE[code & 0xff];
}

function fail(message) {
  throw new Error(message);
}

function integer(value, lo, hi, name) {
  if (!Number.isInteger(value) || value < lo || value > hi) {
    fail(`${name}는 ${lo}–${hi} 사이의 정수여야 합니다 (받은 값: ${value})`);
  }
  return value;
}

// Validates an MMA setting for the browser path and returns its parsed form; throws in Korean otherwise.
export function checkMma(mma) {
  if (!mma || typeof mma !== "object") fail("mma 설정이 없습니다");
  if (mma.algorithm !== "cofda") {
    fail(`브라우저 GPU 경로는 cofda 알고리즘만 지원합니다 (받은 값: ${mma.algorithm})`);
  }
  const fBits = integer(mma.f_bits, 1, LIMITS.maxBits, "f_bits");
  const chunk = integer(mma.chunk_size, 1, LIMITS.maxChunk, "chunk_size");
  if (mma.c_mode !== "fused" && mma.c_mode !== "decoupled") {
    fail(`c_mode는 fused 또는 decoupled여야 합니다 (받은 값: ${mma.c_mode})`);
  }
  const decoupled = mma.c_mode === "decoupled";
  const f2Bits = decoupled ? integer(mma.f2_bits, 1, LIMITS.maxBits, "f2_bits") : 0;
  if (mma.promote_interval !== undefined && mma.promote_interval !== 0) {
    fail(`승격 주기(promote_interval = ${mma.promote_interval})는 브라우저 GPU 경로에서 지원하지 않습니다 — 0만 가능합니다`);
  }
  if (mma.norm_rounding !== "rtz" && mma.norm_rounding !== "rne") {
    fail(`norm_rounding은 rtz 또는 rne여야 합니다 (받은 값: ${mma.norm_rounding})`);
  }
  if (mma.out_format !== undefined && mma.out_format !== "fp32") {
    fail(`출력 형식은 fp32만 지원합니다 (받은 값: ${mma.out_format})`);
  }
  if (mma.scale_apply !== undefined && mma.scale_apply !== "auto" && mma.scale_apply !== "epilogue") {
    fail(`텐서 단위 스케일은 epilogue에서만 적용됩니다 (scale_apply: ${mma.scale_apply})`);
  }
  return {fBits, chunk, decoupled, f2Bits, rne: mma.norm_rounding === "rne"};
}

// Validates shapes and operand buffers; returns the fp32-rounded scales.
export function checkArgs(args) {
  const {a, b, M, N, K} = args;
  integer(M, 1, 2 ** 31 - 1, "M");
  integer(N, 1, 2 ** 31 - 1, "N");
  integer(K, 1, 2 ** 31 - 1, "K");
  if (!(a instanceof Uint8Array) || a.length !== M * K) fail(`A는 길이 M·K = ${M * K}인 Uint8Array여야 합니다`);
  if (!(b instanceof Uint8Array) || b.length !== N * K) fail(`B는 길이 N·K = ${N * K}인 Uint8Array여야 합니다`);
  if (typeof args.scaleA !== "number" || typeof args.scaleB !== "number") fail("scaleA, scaleB는 숫자여야 합니다");
  return {scaleA: Math.fround(args.scaleA), scaleB: Math.fround(args.scaleB)};
}

// ⌊log2 x⌋ for an integer 1 ≤ x < 2^53.
function log2floor(x) {
  if (x < 4294967296) return 31 - Math.clz32(x);
  return 63 - Math.clz32(Math.floor(x / 4294967296));
}

// x · 2^s for a nonnegative integer Number: left shifts are exact, right shifts truncate, |s| ≥ 64 gives 0.
function shift(x, s) {
  if (s >= 64 || s <= -64) return 0;
  return s >= 0 ? x * pow2(s) : Math.floor(x * pow2(s));
}

// The fp32 register as an FDA term (§4.3 step 1): subnormals normalised, 24-bit significand.
function decodeF32(bits) {
  const field = (bits >>> 23) & 0xff;
  const frac = bits & 0x7fffff;
  const neg = bits >>> 31 === 1;
  if (field === 0) {
    if (frac === 0) return {zero: true, neg, e: 0, sig: 0};
    const lead = 31 - Math.clz32(frac);
    return {zero: false, neg, e: lead - 149, sig: frac * pow2(23 - lead)};
  }
  return {zero: false, neg, e: field - 127, sig: frac + 0x800000};
}

const isNaNBits = (bits) => (bits & 0x7f800000) === 0x7f800000 && (bits & 0x7fffff) !== 0;
const isInfBits = (bits) => (bits & 0x7fffffff) === INF_BITS;

// §4.3 step 5, to_fp32(S, Emax, F, norm) for an exact Number sum |S| < 2^53.
function toFp32Number(total, emax, radix, rne) {
  if (total === 0) return 0;
  const sign = total < 0 ? 0x80000000 : 0;
  const mag = Math.abs(total);
  const lead = log2floor(mag);
  const biased = lead + emax - radix + 127;
  if (biased <= 0) return (sign | shift(mag, 149 + emax - radix)) >>> 0;  // subnormal: truncates
  const precision = Math.min(radix, 23);
  const amount = precision - lead;
  let kept;
  if (amount >= 0) {
    kept = mag * pow2(amount);
  } else {
    kept = Math.floor(mag * pow2(amount));
    if (rne) {
      const rem = mag - kept * pow2(-amount);
      const half = pow2(-amount - 1);
      if (rem > half || (rem === half && kept % 2 === 1)) kept += 1;
    }
  }
  let exponent = biased;
  if (kept >= pow2(precision + 1)) {
    kept /= 2;
    exponent += 1;
  }
  if (exponent >= 255) return (sign | INF_BITS) >>> 0;
  return (sign | (exponent << 23) | ((kept * pow2(23 - precision)) & 0x7fffff)) >>> 0;
}

// The same for a BigInt sum (chunks whose bound exceeds 2^53).
function toFp32Big(total, emax, radix, rne) {
  if (total === 0n) return 0;
  const sign = total < 0n ? 0x80000000 : 0;
  const mag = total < 0n ? -total : total;
  const lead = mag.toString(2).length - 1;
  const biased = lead + emax - radix + 127;
  const shiftBig = (x, s) => (s >= 64 || s <= -64 ? 0n : s >= 0 ? x << BigInt(s) : x >> BigInt(-s));
  if (biased <= 0) return (sign | Number(shiftBig(mag, 149 + emax - radix))) >>> 0;
  const precision = Math.min(radix, 23);
  const amount = precision - lead;
  let kept = shiftBig(mag, amount);
  if (amount < 0 && rne) {
    const discard = BigInt(-amount);
    const rem = mag - (kept << discard);
    const half = 1n << (discard - 1n);
    if (rem > half || (rem === half && (kept & 1n) === 1n)) kept += 1n;
  }
  let exponent = biased;
  if (kept >= 1n << BigInt(precision + 1)) {
    kept >>= 1n;
    exponent += 1;
  }
  if (exponent >= 255) return (sign | INF_BITS) >>> 0;
  return (sign | (exponent << 23) | Number((kept << BigInt(23 - precision)) & 0x7fffffn)) >>> 0;
}

// One FDA over the products a[ia + k]·b[ib + k], k in [start, end), and the fp32 register c (bits, or
// null for the decoupled first stage, whose c is +0). Returns fp32 bits.
function fdaChunk(a, ia, b, ib, start, end, c, radix, rne, wide) {
  let emax = -Infinity;
  for (let k = start; k < end; k += 1) {
    const ca = a[ia + k];
    const cb = b[ib + k];
    if (ISNAN[ca] || ISNAN[cb]) return NAN_BITS;  // NaN product (NaN × 0 included)
    if (MAN[ca] !== 0 && MAN[cb] !== 0) emax = Math.max(emax, EXP[ca] + EXP[cb] - 14);
  }
  const reg = c === null ? null : decodeF32(c);
  const regActive = reg !== null && !reg.zero;
  if (emax === -Infinity && !regActive) return c === null ? 0 : c;  // nothing non-zero: c unchanged
  if (regActive) emax = Math.max(emax, reg.e);
  const base = radix - 20 - emax;  // product shift = radix - 6 - (emax - e), e = EXP_a + EXP_b - 14
  let total = 0;
  let big = 0n;
  let pending = 0;
  for (let k = start; k < end; k += 1) {
    const ca = a[ia + k];
    const cb = b[ib + k];
    const m = MAN[ca] * MAN[cb];
    if (m === 0) continue;
    const s = base + EXP[ca] + EXP[cb];
    const term = s >= 0 ? m * pow2(s) : s > -8 ? m >> -s : 0;
    total += SGN[ca] !== SGN[cb] ? -term : term;
    if (wide && (pending += 1) === 4) {  // 4 terms < 2^52: flush before a Number sum could round
      big += BigInt(total);
      total = 0;
      pending = 0;
    }
  }
  if (regActive) {
    const term = shift(reg.sig, radix - 23 - (emax - reg.e));
    total += reg.neg ? -term : term;
  }
  if (wide) return toFp32Big(big + BigInt(total), emax, radix, rne);
  return toFp32Number(total, emax, radix, rne);
}

// Decoupled second stage: fda([P at radix F2], c, F2), both operands fp32 registers.
function fdaRegisters(p, c, radix, rne) {
  if (isNaNBits(p) || isNaNBits(c)) return NAN_BITS;
  if (isInfBits(p) || isInfBits(c)) {
    if (isInfBits(p) && isInfBits(c) && p !== c) return NAN_BITS;
    return isInfBits(p) ? p : c;
  }
  const rp = decodeF32(p);
  const rc = decodeF32(c);
  if (rp.zero && rc.zero) return c;
  const emax = Math.max(rp.zero ? -Infinity : rp.e, rc.zero ? -Infinity : rc.e);
  let total = 0;
  for (const r of [rp, rc]) {
    if (r.zero) continue;
    const term = shift(r.sig, radix - 23 - (emax - r.e));
    total += r.neg ? -term : term;
  }
  return toFp32Number(total, emax, radix, rne);
}

// Epilogue (§4.9): v = s_a · (s_b · acc), fp32 RN each; fp32 output keeps v. A double product of two
// fp32 values is exact, so Math.fround rounds it once.
function epilogue(acc, scaleA, scaleB) {
  return floatToBits(Math.fround(scaleA * Math.fround(scaleB * bitsToFloat(acc))));
}

// CoFDA GEMM (§4.4): chunks of chunk_size products from k = 0, c = +0.
export function cofdaGemm(args) {
  const cfg = checkMma(args.mma);
  const {scaleA, scaleB} = checkArgs(args);
  const {a, b, M, N, K} = args;
  const radix = cfg.fBits;
  const productBound = cfg.chunk * 225 * 2 ** (radix - 6);
  const wide = productBound + (cfg.decoupled ? 0 : 2 ** (radix + 1)) > 2 ** 53;
  const out = new Uint32Array(M * N);
  for (let m = 0; m < M; m += 1) {
    for (let n = 0; n < N; n += 1) {
      let c = 0;
      for (let start = 0; start < K; start += cfg.chunk) {
        const end = Math.min(start + cfg.chunk, K);
        if (isNaNBits(c)) break;  // §4.3 step 1: a NaN register stays NaN
        if (cfg.decoupled) {
          const p = fdaChunk(a, m * K, b, n * K, start, end, null, radix, cfg.rne, wide);
          c = fdaRegisters(p, c, cfg.f2Bits, cfg.rne);
        } else if (isInfBits(c)) {
          if (fdaChunk(a, m * K, b, n * K, start, end, null, radix, cfg.rne, wide) === NAN_BITS) c = NAN_BITS;
        } else {
          c = fdaChunk(a, m * K, b, n * K, start, end, c, radix, cfg.rne, wide);
        }
      }
      out[m * N + n] = epilogue(c, scaleA, scaleB);
    }
  }
  return out;
}

// TriCast's fp64 algorithm (§4.7): fp64 chain acc = a_k·b_k + acc from +0 in K order (every E4M3
// product is exact in fp64), one rounding to fp32, then the same epilogue. For E4M3 operands every
// partial sum is a multiple of 2^-18 below 2^(K's bits + 18), so the fp64 chain is the exact sum.
export function fp64Gemm(args) {
  const {scaleA, scaleB} = checkArgs(args);
  const {a, b, M, N, K} = args;
  const out = new Uint32Array(M * N);
  for (let m = 0; m < M; m += 1) {
    for (let n = 0; n < N; n += 1) {
      let acc = 0;
      const ia = m * K;
      const ib = n * K;
      for (let k = 0; k < K; k += 1) acc = VALUE[a[ia + k]] * VALUE[b[ib + k]] + acc;
      out[m * N + n] = epilogue(floatToBits(Math.fround(acc)), scaleA, scaleB);
    }
  }
  return out;
}

// fp32 ULP distance as tricast.analysis.ulp_distance: neighbouring values are 1 apart (also across zero),
// +0 and -0 are 0 apart. NaN/NaN and equal infinities count as 0; any other Inf/NaN pair has no finite
// distance and returns Infinity (TriCast's ulp_error counts those as nonfinite_mismatch).
export function ulpDistance(aBits, bBits) {
  const nanA = isNaNBits(aBits);
  const nanB = isNaNBits(bBits);
  if (nanA || nanB) return nanA && nanB ? 0 : Infinity;
  if (isInfBits(aBits) || isInfBits(bBits)) return aBits >>> 0 === bBits >>> 0 ? 0 : Infinity;
  const ordinal = (bits) => (bits >>> 31 ? -(bits & 0x7fffffff) : bits & 0x7fffffff);
  return Math.abs(ordinal(aBits) - ordinal(bBits));
}
