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
- Qwen3-0.6B results (`docs/results/qwen3_0.6b.md`): WikiText-2 perplexity of 24 recipes on an A100,
  KIVI perplexity and CoQA, and HellaSwag/CoQA of five recipes on a V100, each with its environment;
  `scripts/e2e/summarize.py` also tabulates lm-eval records.
- `scripts/bench/bench_decode.py`: decode-step latency of a patched model (median / mean / p99), with
  the host load, other processes on the GPU and the source commit recorded.
- Weight structure in recipes: `sparsity` (N:M or unstructured magnitude pruning) and `outliers` (the
  largest weights kept in a higher-precision format and added back through an fp32 path), with the
  bundled recipes `fp8_2of4_sparse` and `nvfp4_outliers` (team interviews, log 10).
- Accumulator ULP error: `tricast.analysis.ulp_distance` / `ulp_error`, the per-layer `mma_ulp` of
  `tricast report` (against fp64 accumulation of the same operands) and ULP columns in the demo
  (team interviews, log 9).
- Course deliverables grounded in the team's ten interviews: `docs/research/interviews.md`,
  `docs/ontology.yaml`, `docs/PROBLEM.md`, `docs/SPEC.md` (AC8–AC10).

### Changed
- MMA emulation kernel: integer fast path for finite operands (int32/int64 aligned sums, int32
  operand decode, runtime pass loops, no register spills). On an A100 the Hopper FP8 accumulator
  went from 0.011 to 0.15 TMAC/s (NADPE CUDA kernel: 0.15) and a Qwen3-0.6B 2048-token window
  from 83 s to 6 s; results stay bit-identical.
- MMA emulation kernel for small M (decoding, M ≤ 16; CoFDA, GDFS, exact integer): each chunk,
  group or K span is loaded as one tile and reduced in registers, and for CoFDA and GDFS the
  Inf/NaN flags are reduced in the kernel instead of a host finiteness check (a device sync per
  linear). A Qwen3-0.6B Hopper-FP8 decode step went from 375–401 to 168–170 ms on a V100 and from
  292–294 to 101 ms on an A100 (`scripts/bench/bench_decode.py`); results and generated tokens stay
  identical.
