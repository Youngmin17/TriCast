# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Number formats (`ExMy` float with IEEE / fn / fnuz / no-special, custom bias, subnormals on/off; `intN`/`uintN`
  with fixed point; E8M0), six rounding modes including stochastic rounding with explicit noise.
- Exact reference backend (fp64 / int64) for casting, quantization and MMA accumulation.
- Quantization: tensor / row / group / 2-D block scales; absmax, power-of-two, MSE search, percentile,
  Four-over-Six; two-level NVFP4; integer (GPTQ/AWQ) and float (KIVI/HQQ) zero points; MX, NVFP4, BFP and
  FP8/INT schemes; static observers (min-max, EMA, delayed history, percentile, MSE).
- MMA accumulation models: CoFDA (C-fused / C-decoupled), GDFS, DeepSeek-style FP32 promotion, FP32 FMA
  chain, FP64, exact integer; hardware presets with provenance (Hopper, Ada, Blackwell FP8/FP4).
- Triton kernels for quantization and MMA emulation on CUDA cores, bit-identical to the reference.
- Hugging Face integration: `patch_model` with per-layer rules (`match` / `layers` / `modules`, `skip`),
  calibration (WikiText-2 / C4 / Pile), GPTQ (incl. sequential), AWQ, SmoothQuant, Hadamard rotations,
  STE for QAT, KIVI-style KV-cache quantization.
- Evaluation: WikiText-2 perplexity, lm-eval adapter (`tricast-lm-eval`, `tricast eval`), sweep runner with
  resume and environment capture, per-layer error report (`tricast report`).
- `tricast agent`: natural-language request to a validated recipe (Claude API or offline parser).
- NADPE golden vectors (1716 cases) and `scripts/nadpe_oracle/check_triton.py` for Triton-vs-NADPE checks.

### Changed
- MMA emulation kernel: integer fast path for finite operands (int32/int64 aligned sums, int32
  operand decode, runtime pass loops, no register spills). On an A100 the Hopper FP8 accumulator
  went from 0.011 to 0.15 TMAC/s (NADPE CUDA kernel: 0.15) and a Qwen3-0.6B 2048-token window
  from 83 s to 6 s; results stay bit-identical.
- MMA emulation kernel for small M (decoding, M ≤ 16; CoFDA, GDFS, exact integer): each chunk,
  group or K span is loaded as one tile and reduced in registers, and Inf/NaN flags are reduced in
  the kernel instead of a host finiteness check (a device sync per linear). A Qwen3-0.6B Hopper-FP8
  decode step on a V100 went from 371 to 165 ms (GEMM kernel time 254 to 40 ms); results stay
  bit-identical.
