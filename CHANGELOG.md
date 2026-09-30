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
