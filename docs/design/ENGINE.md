# TriCast engine — design and numerical semantics (contract)

This document is the contract every TriCast backend implements. The pure-PyTorch
reference (`tricast.reference`) is the executable form of it; Triton kernels
(`tricast.kernels`) must match the reference **bit for bit** on every input the
reference accepts. When this text and the reference disagree, fix one of them —
never tolerate the mismatch in a test.

Sources the semantics are taken from (cited per section):

- **NADPE** — *Not All Dot Products Are Equal* (MICRO'26), MMA-Emu CUDA kernels,
  `micro26-ae/csrc/quantization/mma_emu` (Apache-2.0). FDA / CoFDA / GDFS.
- **microxcaling** — `github.com/microsoft/microxcaling` (MIT). MX shared-exponent
  quantization, element rounding.
- **OCP MX v1.0**, **NVIDIA NVFP4** (TensorRT-Model-Optimizer ordering), **DeepSeek-V3**
  technical report §3.3 (FP8 promotion), **Transformer Engine** delayed scaling,
  **GPTQ**, **SmoothQuant**, **AWQ**, **QuaRot/RHT**, **Four Over Six** (NVFP4 adaptive block scaling).

---

## 0. Layering and module map

```
spec layer (pure Python, no torch kernels)
  tricast.formats        FloatFormat / IntFormat / Pow2Format, registry, parser
  tricast.rounding       Rounding enum
  tricast.quant.spec     QuantSpec, ScaleSpec, ObserverSpec, TransformSpec, WeightAlgoSpec, schemes
  tricast.mma.spec       MMASpec, presets
reference layer (torch, CPU or GPU, exact, slow)
  tricast.reference.cast       round_to_format, decode                 (§2)
  tricast.reference.quantize   scales, elements, QTensor construction  (§3)
  tricast.reference.mma        FDA/CoFDA/GDFS/fma/fp64/int             (§4)
kernel layer (Triton, CUDA cores, bit-exact to reference)
  tricast.kernels.cast_core    @triton.jit round-to-format helpers
  tricast.kernels.quantize     quantize kernels
  tricast.kernels.mma_core     @triton.jit decode / product / FDA helpers
  tricast.kernels.mma          emulated GEMM kernels
api layer (backend dispatch: "auto" = triton if CUDA+triton else reference)
  tricast.quant.api      quantize(), fake_quant()      tricast.quant.qtensor  QTensor
  tricast.quant.observer ObserverState                  tricast.mma.api        gemm()
algorithms
  tricast.transforms     hadamard / random_hadamard / smoothquant / awq     (§3.9)
  tricast.weight_quant   rtn / gptq                                         (§3.10)
integration
  tricast.recipe         Recipe (YAML/JSON, schema src/tricast/schemas/recipe.schema.json)
  tricast.nn             EmuLinear, patch_model                             (§6)
  tricast.calibrate      calibration pass (observers, transform stats, GPTQ Hessians)
  tricast.eval           ppl, lm-eval model "tricast", runner + env capture
  tricast.cli            `tricast` command
```

Conventions used below: `rows × K` is the 2-D view of an operand (`K` = reduction
axis = last dim; `nn.Linear` weight `[N, K]`, activation `[M, K]`). "fp32 RN" means an
IEEE binary32 operation with round-to-nearest-even (in Triton: `tl.math.div_rn`
for division; `*`, `+`, `tl.fma` are RN already). `⌊log2 x⌋` is always computed
exactly from the exponent bits, never from a floating `log2`.

---

## 1. Formats (`tricast.formats`)

- `FloatFormat(ebits, mbits, bias, special, signed, subnormals)`; `special ∈ {ieee, fn,
  fnuz, none}`; `emin = 1 - bias`; `emax` = top usable exponent; `max_normal`
  (`fn` loses the all-ones fraction at `emax`). Range must lie inside fp32.
- `IntFormat(bits, signed, symmetric, frac_bits)`: `value = k · 2^-frac_bits`.
  MX ints: `mxint8 = int8, frac 6` (max 127/64), block floating point mantissas:
  `bfp_m{m} = int{m}, frac m-2`.
- `Pow2Format` (E8M0): `2^e, e ∈ [-127, 127]`, no zero, NaN at field 255.
- Every value of every format is exactly representable in fp32; all tensors
  carrying grid values use an fp32 (or exactly-fitting bf16/fp16, see
  `container_dtype`) container.

## 2. Casting — `round_to_format(x, fmt, rounding, saturate, noise, sr_bits)`

Reference: `tricast/reference/cast.py` (verified bit-exact against `torch` casts to
`float8_e4m3fn/e5m2/e4m3fnuz/e5m2fnuz`, `bfloat16`, `float16`).

For finite `x`, with `q` the quantum exponent:

- float: `e = max(⌊log2|x|⌋, emin)`, `q = e - mbits`; int: `q = -frac_bits`.
- `y = |x| · 2^-q` (exact), `k = ⌊y⌋`, `f = y - k` (exact).
- increment `inc`: RNE `f > ½ ∨ (f = ½ ∧ k odd)`; RNA `f ≥ ½`; RTZ `0`;
  RUP `f > 0 ∧ x > 0`; RDN `f > 0 ∧ x < 0`;
  SR `u < f` with `u = ⌊noise / 2^(32-sr_bits)⌋ / 2^sr_bits`, `noise` a uint32 value
  per element (same noise ⇒ same result on every backend).
- `|r| = (k + inc) · 2^q`, sign copied from `x` (−0 kept except `fnuz`/unsigned).
- overflow (`|r| > max_normal` or `x = ±Inf`): saturate → `±max_normal`; else `ieee`
  formats give Inf when rounding points away from zero (RNE/RNA/SR always, RUP for
  `x > 0`, RDN for `x < 0`) and `max_normal` otherwise; `fn/fnuz` give NaN; `none`
  formats always saturate.
- ints clamp to `[qmin, qmax] · 2^-frac` (unsigned: negatives → 0).
- `subnormals=False`: `|x| < min_normal` flushes to ±0 *before* rounding (microxcaling
  `allow_denorm=False`).
- NaN in → NaN out. Pow2 formats round the value (RNE/RNA pick the nearer power,
  ties up), clamp below `2^emin`, NaN on negative/overflow (saturate clamps).

**Triton equivalent** (`cast_core`): operate on fp32 bits as int32/int64: extract
exponent and 24-bit significand, `shift = 23 - mbits + max(0, emin - e)` clamped to
25, `kept = sig >> shift`, `rem = sig & ((1<<shift)-1)`, `half = 1<<(shift-1)`,
same `inc` rules on `(rem, half, kept&1)` (SR: `inc = (noise >> (32-sr_bits)) <<
(32-sr_bits) < rem << (32-shift)` using 64-bit arithmetic), rebuild the fp32 bit
pattern from `kept+inc` and `q` with integer ops (no float multiply by `2^q`, which
flushes subnormals under FTZ). Do **not** emulate rounding with `.to(bf16).to(f32)`
round trips (constant-folded away).

## 3. Quantization — `quantize(x, spec) -> QTensor`

### 3.1 View and scale domains
`x` of shape `[..., K]` is viewed as `[rows, K]`, `rows = prod(shape[:-1])`.
Domains: `tensor` (all), `row` (each row), `group` (`group_size` consecutive
elements of a row; the last group of a row may be short), `block` (`br × bc` tiles of
`[rows, K]`; edge tiles may be short). Short domains use only their valid elements.

### 3.2 amax
`amax(D) = max |x|` over the domain (NaN propagates). With an `ObserverSpec`
(tensor granularity only) amax comes from the observer state (§3.11) instead of the
current tensor.

### 3.3 Scale methods (single level)
Let `M = max_normal(elem)`, `E = ⌊log2 M⌋` (`QuantSpec.emax_elem`), `SF` the scale format.

| method | scale `s` |
|---|---|
| `absmax` | `t = amax / M` (fp32 RN div) → `s = round_to_format(t, SF, scale.rounding, saturate=True)` |
| `pow2_floor` | `e = ⌊log2 amax⌋ - E` (amax = 0 → use `⌊log2 2^-126⌋`); clamp to SF: `e > emax_SF` → NaN scale (microxcaling), `e < emin_SF` → `emin_SF`; `s = 2^e` |
| `pow2_ceil` | smallest `e` with `2^e · M ≥ amax` (exact test), clamped as above |
| `percentile` | `absmax` on `amax_p = quantile(|D|, p/100)` (linear interpolation, fp64) |
| `mse` | for each `r` in `search` (default `linspace(1.0, 0.5, mse_grid)`): `absmax` on `r·amax` (fp32 RN mult), quantize `D`, `err = Σ (x - x̂)²` (fp64); keep the smallest `err`, ties → earliest `r`. `search=(1.0, 1.5)` = Four-over-Six |

Guards: `amax = 0` → `s = 1` for absmax-type methods; if a rounded absmax scale is 0
(underflow) → `s = min positive of SF`. Float scale formats clamp to `max_normal`.

### 3.4 Two-level (NVFP4, TensorRT-Model-Optimizer order)
`d2 = amax_tensor / (M · SF.max_normal)` (fp32 RN; product `M·SF.max` exact; `d2 = 0 →
1`). Per block: `t = (amax_b / M) / d2` (two fp32 RN divs), `s_b = round_to_format(t,
SF, scale.rounding, saturate=True)`. Effective scale `s_b · d2` (fp32 RN). QTensor keeps
`s_b` in `scale` and `d2` in `global_scale`.

### 3.5 Zero points (integer formats, `mma_input="dequant"` only)
`lo = min(0, min D)`, `hi = max(0, max D)`, `s* = (hi - lo) / (qmax - qmin)` → scale
format as in absmax; `int`: `z = clamp(RNE(qmin - lo / s), qmin, qmax)`; `float`: `z =
qmin - lo / s` (fp32). Elements `q = clamp(round(x / s) + z, qmin, qmax)` with the
element rounding mode applied to `x / s` and the integer addition exact.

### 3.6 Elements
`q = round_to_format(x / s_eff, elem, spec.rounding, saturate=spec.saturate,
noise, spec.sr_bits)` where `x / s_eff` is fp32 RN (`s_eff = s` or `s_b·d2`). A NaN
scale (pow2 overflow) makes its domain NaN. `scale=None` → `q = round_to_format(x,
elem, …)` directly.

### 3.7 QTensor
```
values        [rows, K] grid values of spec.format (unscaled), container dtype
scale         fp32 | None   tensor: []   row: [rows,1]   group: [rows, ceil(K/G)]
                            block: [ceil(rows/br), ceil(K/bc)]
zero_point    fp32 | None   same shape as scale
global_scale  fp32 scalar | None  (d2)
spec, shape   the QuantSpec and the original shape
```
`dequantize()`: `x̂ = (q - z) · s` (fp32 RN, per-element broadcast of `s`), two-level
`x̂ = (q · s_b) · d2`; reshaped to `shape`. `mma_operand()`: `scaled` → `(values, scale
layout)`; `dequant` → `round_to_format(x̂, dequant_format, RNE, saturate=False)`.

### 3.8 fake_quant and QAT
`fake_quant(x, spec) = mma_operand-equivalent value in fp32` (dequantized, and rounded
to `dequant_format` in dequant mode). Autograd: straight-through estimator —
`grad_x = grad_out` inside the representable range, `0` where the element saturated
(clipped STE). LSQ-style learnable scales are a planned extension.

### 3.9 Transforms (`TransformSpec`, applied per linear to both operands)
`y = x Wᵀ = (x T)(W T^{-ᵀ})ᵀ`. `hadamard`: `T = diag(H_b, …)`, `H_b` the Sylvester
Hadamard of size `b` scaled by `1/√b` (orthonormal, `T^{-ᵀ} = T`); `b = block` or
the largest power of two ≤ 128 dividing K. `random_hadamard`: `T = D·H` with `D`
a ±1 diagonal from `torch.Generator().manual_seed(seed)`. `smoothquant`: `T =
diag(1/s)`, `s_j = max|X_j|^α / max|W_j|^{1-α}` from calibration (`max|X_j|` over all
calibration tokens), `s_j` clamped to `[1e-5, 1e5]`, weights become `W diag(s)`.
`awq`: same diagonal form with `s_j = mean|X_j|^α` normalised by `sqrt(max s · min
s)`; `α` chosen from `linspace(0, 1, grid)` minimising `‖Q(W diag(s)) (X/s)ᵀ − W Xᵀ‖²`
on calibration activations. Transformed activations/weights are fp32 values that
are then quantized as usual.

### 3.10 Weight algorithms (`WeightAlgoSpec`)
`rtn`: `quantize(W, spec)`. `gptq`: Frantar et al. with `H = 2XᵀX/n` from calibration
activations (after the transform), dampening `damp · mean(diag H)`, optional
act-order, lazy batch updates of `block_size` columns, each column rounded with the
QuantSpec (group scales are computed when a group's first column is reached, from
the current partially-updated weights). Output is a regular QTensor, so every
format/scale/granularity combination supports GPTQ.

### 3.11 Observers (`ObserverSpec`, activations, tensor granularity)
State per EmuLinear: `amax`, `count`, `history`, reservoir `samples` (≤ `max_samples`).
Calibration mode updates state and quantizes dynamically; eval mode uses the frozen
static amax. `minmax`: running max. `ema`: `amax ← decay·amax + (1−decay)·amax_call`,
first call initialises. `history`: TE delayed scaling — the scale for a call uses
`reduce(history)` of *previous* calls (`max` or `most_recent`), the first call uses its
own amax, then the current amax is appended (length ≤ `history_len`); history keeps
updating in eval mode. `percentile`/`mse`: computed once from the reservoir at freeze.

## 4. MMA — `gemm(a, b, mma) -> out`

`a` = `[M, K]` activation operand, `b` = `[N, K]` weight operand, `out[m, n] = Σ_k a[m,k]·b[n,k]`
under `mma`. Each output element is computed independently; tile shapes never
change numerics.

### 4.1 Operands
An operand is `(values, format, scales)`. Unquantized tensors use the format of their
dtype (`format_of_dtype`). `decode(v, fmt) = (neg, e, m, R)` with `v = ±m · 2^(e−R)`
(`reference.cast.decode`): floats `e = max(⌊log2|v|⌋, emin)`, `R = mbits` (normals
`m ∈ [2^R, 2^(R+1))`, subnormals unnormalised); ints `e = 0`, `R = frac_bits`; pow2 `m =
1, R = 0`. Zero ⇔ `m = 0`.

### 4.2 Products (NADPE `fp8_multiply`, generalised)
`p = a·b`: `neg = neg_a ⊕ neg_b`, `e = e_a + e_b`, `m = m_a·m_b` at radix `R_a+R_b`
(exact). At datapath radix `F`: `m_F = m << (F − R)` if `F ≥ R`, else `m >> (R − F)`
(truncation). A product is **zero iff an operand is zero** (even if `m_F` becomes 0).
Product-level scales (§4.6) multiply the scale significands in and add exponents.

### 4.3 FDA primitive — `fda(terms, c, F, norm)` (NADPE `chunked_accumulate<F,N>`)
`terms`: products/group operands at radix `F`. `c`: fp32 running value.
1. `c` NaN → NaN. `c` operand: fp32 decode (subnormals normalised: `e < −126`,
   24-bit significand), significand moved to radix F (`>> (23−F)` or `<< (F−23)`).
2. Specials over terms and `c`: any NaN → NaN; `+Inf` and `−Inf` → NaN; else any Inf → that Inf.
3. `Emax = max e` over non-zero terms and non-zero `c`; if there is none, return `c`
   unchanged.
4. `S = Σ ±(m_F >> (Emax − e))` over non-zero terms and `c` (shift ≥ 64 ⇒ 0), exact int64.
5. `to_fp32(S, Emax, F, norm)`: `S = 0 → +0.0`. `L = ⌊log2|S|⌋`, `b = L + Emax − F + 127`.
   `b ≤ 0` (subnormal): `b < −23 → ±0`, else significand `⌊|S|·2^(23−L−(1−b))⌋` (no F
   truncation). `b ≥ 255 → ±Inf`. Normal: 23-bit fraction `⌊|S|·2^(23−L)⌋ mod 2^23`, then
   keep only the top `min(F, 23)` fraction bits: `rtz` truncates (NADPE); `rne`
   (TriCast extension) rounds the discarded bits of `|S|` to nearest-even, carrying
   into the exponent (→ ±Inf past `b = 254`).

### 4.4 CoFDA (`algorithm="cofda"`)
Chunks `[j·CS, (j+1)·CS)` from `k = 0`, the last one zero-padded; `c = +0.0`.
- `fused`: `c = fda(products(chunk), c, F)`.
- `decoupled`: `P = fda(products(chunk), +0.0, F)`; `c = fda([decode_fp32(P) at radix F2],
  c, F2)` (NADPE uses F2 = 23).
- `promote_interval = PI > 0` (DeepSeek-V3 / DeepGEMM): intervals `[i·PI, (i+1)·PI)`;
  per interval `P` = fused CoFDA over its chunks starting from `+0.0`; then
  `acc = fma(P, w, acc)` (fp32, single rounding) with `w = s_a·s_b` of the interval
  (fp32 RN; `1.0` without K-varying scales). Output `acc`.

### 4.5 GDFS (`algorithm="gdfs"`) and int64 headroom
Tiles `[t·KT, (t+1)·KT)`, groups of `GS` inside, `c = +0.0`. Per group: products at radix
`G`; `E_g` = max `e` over non-zero products; `S_g = Σ ±(m_G >> (E_g − e))`; all-zero
group ⇒ zero operand. Group operand: sign of `S_g`, magnitude `|S_g|·m_sa·m_sb` at
radix `G + R_sa + R_sb`, exponent `E_g + e_sa + e_sb`, moved to radix F (shift left or
truncate right); zero if `S_g = 0` or a scale is zero, NaN if a scale is NaN (NADPE
`apply_ue4m3_scales` / `apply_e8m0_scales`; without block scales the factors are 1).
Then `c = fda(group operands of the tile, c, F)` — the tile's `KT/GS` groups form one FDA.
**E8M0 field-0 rule:** an E8M0 scale equal to `2^−127` contributes zero (NADPE, matches
hardware). **Headroom:** every call validates `F + int_bits + ⌈log2(n+1)⌉ ≤ 62` where
`int_bits` bounds the integer part of an aligned term (2 for float×float, +1 per float
scale factor, significand widths for ints) and `n` the FDA width.

### 4.6 Where scales enter (`scale_apply`)
- tensor / row scales → `epilogue` (4.9).
- K-varying scales (group/block): `gdfs` → `group` (4.5; each GDFS group must lie
  inside one scale domain of each operand); `cofda` → `promote` if
  `promote_interval > 0` (interval inside one scale domain) else `product`;
  `fp32_fma` / `fp64` → `operand` (operands dequantized first, fp32 RN);
  `int_exact` → rejected.
- `dequant` operands carry no scales (already applied).
- Two-level `d2` factors always go to the epilogue as `alpha = d2_a · d2_b`.

### 4.7 Other algorithms
`fp32_fma`: `acc = fma(a_k, b_k, acc)` in fp32 for `k = 0..K−1` from `+0.0` — the
reference must be an exactly-once-rounded FMA (fp64 product + TwoSum-corrected
rounding); plain fp64 add-then-round is not acceptable. `fp64`: fp64 FMA chain,
one RNE rounding to fp32 at the end. `int_exact`: both operands integer, `acc = Σ
k_a·k_b` exact in int64, then `acc · 2^−(fa+fb)` rounded to fp32 RNE.

### 4.8 Specials
Products: NaN if an operand is NaN or `Inf × 0`; ±Inf if an operand is Inf. They feed
§4.3 step 2. `fp32_fma`/`fp64` follow IEEE.

### 4.9 Epilogue and output
`v = acc`; tensor/row scales: `v = s_a[m] · (s_b[n] · v)` (CUTLASS order, fp32 RN each);
two-level: `v = alpha · v`; bias: `v = v + bias[n]` (fp32 RN); `out =
round_to_format(v, out_format, RNE, saturate=False)` stored in the torch dtype of
`out_format` (bf16/fp16/fp32).

## 5. Triton kernels

- Must match the reference bit-exactly (NaN positions equal; `-0` vs `+0` may differ
  only where the reference documents it).
- Specialise on `tl.constexpr` (formats' `mbits/emin`, `F`, `CS`, `G`, `GS`, `KT`, modes);
  one compiled variant per configuration.
- Operands are passed K-major (`[K, M]`, `[K, N]` contiguous) so per-`k` vector loads are
  coalesced; weights are packed once at patch time.
- Streaming two-pass FDA per chunk (pass 1: `Emax` + specials, pass 2: aligned sum),
  `[BM, BN]` register tiles; GDFS keeps up to 8 group results in explicit
  constexpr-unrolled registers.
- Every variable shift is clamped (`tl.where(d > 63, 0, x >> min(d, 63))`); LLVM
  treats over-wide shifts as poison.
- fp32 division uses `tl.math.div_rn`; no rounding emulated by dtype round trips.
- SR uses the caller's `noise` tensor when given, else `tl.randint(seed, offsets)`.

## 6. Integration

### 6.1 Recipe (`tricast.recipe`, schema `src/tricast/schemas/recipe.schema.json`)
```yaml
name: hopper_fp8_w8a8
description: FP8 E4M3 per-tensor W8A8 on Hopper tensor-core accumulation
defaults:
  weight: {scheme: fp8_tensor}            # scheme name or explicit QuantSpec fields
  activation: {scheme: fp8_tensor}        # null = unquantized (input dtype)
  mma: {preset: nvidia_hopper_fp8}        # preset or explicit MMASpec fields
  transform: {kind: none}
  weight_algo: {kind: rtn}
include: ["*"]                            # fnmatch on model.named_modules() names
exclude: ["lm_head"]
overrides:                                # ordered, first match wins, merged over defaults
  - match: "*.mlp.down_proj"
    weight: {scheme: fp8_row}
calibration: {dataset: wikitext2, split: train, samples: 128, seqlen: 2048, seed: 0}
backend: auto                             # auto | triton | reference
```
QuantSpec fields: `format, granularity ("tensor" | "row" | "group:G" | "block:RxC"),
scale {format, method, rounding, two_level, percentile, mse_grid, search},
zero_point, rounding, sr_bits, saturate, mma_input, dequant_format, observer {kind,
decay, history_len, reduce, percentile}`. Validation errors name the recipe path.

### 6.2 EmuLinear / patch_model
`patch_model(model, recipe) -> PatchReport` replaces matching `nn.Linear` modules
with `EmuLinear` (weights quantized once, packed K-major, bias kept in fp32);
`forward` quantizes the activation (dynamic or observer amax), calls `gemm`, returns
the input dtype. Training mode: forward is the emulated GEMM, backward uses the
dequantized operands with the STE of §3.8. `unpatch_model` restores the originals.

### 6.3 Calibration (`tricast.calibrate`)
One pass over calibration tokens collects, per EmuLinear: observer state,
per-channel `max|X|`/`mean|X|` (smoothquant/awq), `XᵀX` (gptq). Then transforms are
fitted, weights re-quantized (GPTQ where requested; v1 uses full-precision inputs
for every layer), observers frozen.

### 6.4 Evaluation
- `tricast.eval.ppl.perplexity(model, tokenizer, dataset="wikitext2", seqlen=2048)`:
  GPTQ convention — join the test split with `"\n\n"`, tokenize once, non-overlapping
  `seqlen` windows, mean token NLL, `ppl = exp(nll)`.
- lm-eval model **`tricast`** (`lm_eval --model tricast --model_args
  pretrained=…,recipe=…`), a subclass of `HFLM` that patches (and calibrates) the model
  after loading; `tricast.eval.lmeval.evaluate(model, tokenizer, tasks, …)` for Python.
- `tricast.eval.runner`: config → runs (resume, per-run JSON) with `env.json`: git SHA +
  dirty flag, versions (python/torch/triton/transformers/lm_eval/datasets), GPU name +
  driver + CUDA, model repo + commit SHA, dataset fingerprints, recipe hash, seeds.

## 7. Verification ladder

| level | check | where |
|---|---|---|
| L0 | reference vs torch native casts (fp8 ×4, bf16, fp16) — bit-exact | `tests/test_cast_reference.py` |
| L0 | reference quantization vs microxcaling (`round=even/nearest/floor`) — bit-exact except documented `log2` corner cases | `tests/test_microxcaling_parity.py` (`oracle`) |
| L0 | reference MMA: hand-derived NADPE cases, invariants, `fp64` cross-checks | `tests/test_mma_reference.py` |
| L1 | Triton vs reference, bit-exact, random + adversarial inputs, every mode | `tests/gpu/` (`gpu`) |
| L2 | Triton vs NADPE CUDA kernels (standalone build) — bit-exact | `scripts/nadpe_oracle/` (`gpu`) |
| L3 | emulation vs silicon (`torch._scaled_mm` on sm_89+/sm_90) | `tricast.probe` (`gpu`) |
| L4 | end-to-end: bf16 passthrough recipe reproduces the HF model's PPL; known schemes land in the published range | `tricast.eval.runner` |

## 8. Implementation lanes (ownership — no lane edits another lane's files)

| lane | files |
|---|---|
| spec + cast reference (done) | `formats.py`, `rounding.py`, `quant/spec.py`, `mma/spec.py`, `reference/cast.py`, this doc |
| Q quantization reference | `quant/{qtensor,api,observer}.py`, `reference/quantize.py`, `tests/test_quantize_*.py`, `tests/test_observer*.py`, `tests/test_microxcaling_parity.py`, `tests/test_cast_reference.py` |
| M MMA reference | `reference/mma.py`, `mma/api.py`, `tests/test_mma_*.py` |
| KQ Triton quantize | `kernels/{__init__,cast_core,quantize}.py`, `tests/gpu/test_triton_quantize.py` |
| KM Triton MMA | `kernels/{mma_core,mma}.py`, `tests/gpu/test_triton_mma.py` |
| A algorithms | `transforms.py`, `weight_quant/*.py`, `tests/test_transforms.py`, `tests/test_gptq.py` |
| I integration | `recipe.py`, `schemas/recipe.schema.json`, `nn/*.py`, `calibrate.py`, `eval/*.py`, `cli.py`, `configs/**`, `tests/test_{recipe,nn_patch,eval,cli}.py`, `tests/conftest.py` |
