# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Interview record: the ten-log record of 2026-10-01 was replaced on 2026-10-08 by the 28-log record now in
  `docs/research/interviews.md`; old logs 9 and 10 are now logs 01 and 02, so earlier entries that cite
  "log 9" or "log 10" refer to 01 and 02. The PROBLEM falsification approved on 2026-10-01 belongs to the
  ten-log record.
- Course deliverables moved to the standard paths: `docs/PROBLEM.md`, `docs/SPEC.md`,
  `docs/research/interviews.md` and `docs/ontology.yaml` (split out of `docs/ontology.md`). Golden
  cases use the SPEC v1.0 AC numbers (NADPE vectors are AC1), AGENTS.md gains stop conditions and
  ontology class names, and references to removed documents now point at existing files.
- The earlier design, results and slide documents were removed from `docs/`, and the git history was
  squashed into one commit. RAG retrieval reads `docs/SPEC.md`.
- The README now shows the 9-slide business overview as images (`assets/readme/`, slide 7
  animated), rendered from the v3 deck PDF, followed by a short Korean text section.

### Added
- `docs/PROBLEM.md` §3-0 lists behaviour, cost and workaround evidence and the unaffected group in the interview
  log's own words; `docs/ontology.md` §4-4 shows the interview passage behind each class. `tests/test_ontology.py`
  checks that every quote and tag in both tables appears in the cited log row.
- `scripts/harness/compare_fault_injection.py` reruns the AC13/AC14 fault injection on a copy;
  `scripts/bench/count_code_lines.py` reproduces the README code-size comparison; `docs/evidence/` keeps the
  2026-10-09 full-check record.
- `tricast.eval.compare.compare_ppl` (SPEC AC13 · AC14): the PPL verdict for two evaluation records, refused
  with the offending field path unless the model revision, dataset fingerprint and token count are valid and
  equal. Golden cases in `tests/harness/golden_compare.yaml`, harness `tests/test_compare_golden.py`, delegation
  record `docs/prompts/delegation_compare.md`.
- `tests/test_ac4_passthrough.py` judges SPEC AC4 on the CPU reference backend, with a lossy recipe as the
  negative control; `make test-app` and the `app` extra run the TriCast Studio tests.
- Course records restored at the course paths: the CUDA-core emulation spike
  (`docs/spikes/cuda_core_bit_exact_emulation.md`, team-ratified 2026-10-01) and the Wave 1 delegation
  prompts with their verification loop (`docs/prompts/delegation_examples.md`, `docs/prompts/wave1/`).
- `docs/prompts/glossary_check.md`: AGENTS.md removal experiment judged by the recipe loader; it found that
  the glossary did not say `layers` is a string, and the corrected glossary passes.
- `tests/test_spec_traceability.py` checks that every SPEC AC names test, config or script files that
  exist (AC12 is the one marked pending), that golden cases and AGENTS rules cite real ACs, and that the
  SPEC traceability table points at existing files.
- `docs/ontology.yaml` gains interview synonyms and a three-tier scope (managed / external reference /
  out of scope with reasons and evidence); `tests/test_ontology.py` checks that the out-of-scope items
  match SPEC §4's excluded column.
- pytest puts the checkout's `src/` first on `sys.path` (`pythonpath`), so in-process tests import this
  tree even when another tricast is installed; the lm-eval entry-point test still needs `pip install -e .`.
- Claude Code harness: `/check` and `/golden` are user-invoked only (`disable-model-invocation`) and report
  golden and auxiliary checks separately; `git push` asks first; the guard hook protects
  `src/tricast/reference/` instead of the removed design document.
- `tests/test_ontology.py` checks that every ontology class, attribute and relation cites an
  interview log that exists and that relations only target defined classes.
- README text below the slides: install, quick start, document map and verification status.
- `docs/SPEC.md` reference list: every source was opened and checked on 2026-10-08 (NADPE MICRO'26
  artifact doi:10.5281/zenodo.21505180, OCP MX v1.0, IEEE 754-2019, DeepSeek-V3, GPTQ, AWQ, KIVI and
  others). Preset provenance now names the NADPE DOI and source path; the DeepSeek preset states that
  its F and CS come from NADPE.
- `docs/prompts/delegation_golden_harness.md`: fault-injection check of the golden harness (five
  injected faults, all detected). `test_golden_manifest` now rejects unknown case keys, and
  `test_golden_vectors_present` checks the FP8 and FP4 NADPE vectors against `manifest.json` sha256.
- TriCast Studio (`app/`), a web app that compares native and emulated outputs of one input for a
  user-designed accumulation algorithm (CoFDA, GDFS, FP32 FMA, FP64) and input format, on Qwen3-0.6B,
  Llama-3.2-1B, YOLO11n and ResNet18. Three modes: recorded examples, live runs on a server GPU
  (FastAPI job queue), and a WebGPU lab whose WGSL CoFDA GEMM matches the exact JS reference bit for
  bit on 295 golden cases (1,470 outputs) and TriCast's Triton bits on a real Qwen3-0.6B layer.
- Project logo in both READMEs.
- Development harness: `make check` (ruff + CPU pytest), Claude Code hooks that ask before Edit/Write
  changes tests, golden data, shared specs or the gate files and lint every edited Python file, and a
  GitHub Actions workflow (lint, CPU tests, wheel build) switched by the `CI_ENABLED` repository
  variable and off by default.
- `make ci` replays the three CI jobs locally. On macOS arm64 (Python 3.11, torch 2.8.0 CPU, no
  microxcaling) at fa13ba1: lint clean, 2,229 passed / 899 skipped (CUDA and oracle only), wheel and
  sdist pass `twine check`.
- Opt-in inference-only `EmuConv2d` lowering through the existing quantization and MMA engine
  (`patch_model(..., include_conv2d=True)`), including groups/depthwise, padding modes and reversible patching.
- Separate opt-in `AttentionSpec` / `patch_attention` for Transformers 4.55.2 Llama/Qwen3 eager
  QK/PV arithmetic, with explicit per-product specs and plain DynamicCache guards. Projection patching
  can coexist; TriCast KV-cache combinations and training are unsupported. H200/A100 Triton and
  V100 reference passed 47 arithmetic cases, two tiny models across five recipes with 20-token
  generation each, and 29 guards; this does not establish pretrained attention quality.
- A narrow native Hopper FP8 instruction probe with raw operand/lane-mapping controls, exact
  native/reference/Triton comparisons, preserved mismatches and PTX/SASS witnesses. The mma.sync
  path was compiler-lowered to FP16; a separate WGMMA probe passed its native witness and 140/141
  numerical cases. The direct FP32-C mismatch is retained; overall native bit parity and other GPU
  generations are not verified.
- Model-family examples and a reproducible integration validator for Linear/Conv2d, Llama, Qwen,
  ResNet and YOLO, with bit parity, accumulator propagation controls and run metadata.
- Full paired YOLO11n COCO2017 validation: 5,000 images/625 batches, finite outputs, selected Triton
  dispatch and native restoration. `fp8_f7_lowacc` bbox AP fell from 39.4194% to 19.3953%
  (−20.0241 pp); full-evaluation execution passed, not an accuracy-preservation claim.
- Full Llama-3.2-1B projection/MLP WikiText-2 and Winogrande comparisons, retaining native SDPA.
  H200 `hopper_fp8_w8a8` PPL was 9.7573 → 9.8798. A100 `bf16_passthrough` passed AC4's `1e-3`
  relative PPL gate (`4.4411e-5`); A100 `fp8_f7_lowacc` PPL rose 9.7564 → 141.2488 and Winogrande
  fell 60.2210% → 49.5659%. All full execution gates passed, not an acceptable-quality claim.
  Each comparison uses its same-GPU native baseline; F7 changes both quantization and accumulation.
- Architecture diagrams, a model coverage guide and a twenty-six-slide Korean technical overview
  covering Tensor Core motivation, numerical/model-quality decisions, individual interview excerpts,
  Python-to-GPU Triton execution, configurable MMA, model application and recorded metric/demo results.
  The Triton section separates compiler/execution from numerical rules, quotes the actual
  alignment helper and FP8 recipe, and explains specialization, Linear forward and validation scope.
  The strengths comparison ties tile abstraction, arithmetic control, JIT reuse and autotune to
  implementation evidence. Input/CoFDA equations and promotion pseudocode use editable Office Math.
  Six arithmetic walkthrough slides explain FDA, CoFDA fused/decoupled, GDFS, periodic FP32
  promotion and comparison baselines, including the existing hand-derived 16/18 example.
  The deck follows hardware comparison, algorithms and LLM outputs, interviews, then Triton
  design/features/results, with a native side-by-side CUDA Core/Tensor Core comparison table.
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
- Qwen3-0.6B results (`docs/results/qwen3_0.6b.md`): WikiText-2 perplexity of 26 recipes on an A100,
  KIVI perplexity and CoQA, HellaSwag/CoQA of five recipes and the accumulator ULP of five FP8
  accumulators on a V100, each with its environment;
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
- Korean technical overview, 30 slides: agenda, validation dashboard, closing summary; annotated recipe
  and API code slides; Llama/YOLO results relabeled as `fp8_f7_lowacc` (F=7, C-fused); interview roles
  restored to the recorded logs; Qwen PPL split into quantization and accumulator shares.
- Synchronize the latest problem/specification review without relabeling new AC/V-plan contracts as
  verified; align the demo guide with its implemented calibration and retained execution record.
- Minimum Triton version is 3.4; 3.0 cannot compile the current constexpr tuple-based kernels.
- Recipe loading supports Python 3.10's location of the `Traversable` protocol.
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
- Recipe overrides replace a `sparsity`, `format` or `dequant_format` mapping instead of merging
  it field by field: a partial merge could combine two formats' fields into a third format.
- Interview log (`docs/research/interviews.md`): the respondents of logs 9 and 10 appear by role
  instead of by name.
- `README.md` is now the Korean README; the English one moved to `README.en.md`. Both gained a
  command table, the three-minute demo outline, the repository layout and the verification of the CPU
  suite; the Korean one also lists the course deliverables.
- Course deliverables approved by the team on 2026-10-01: AC1–AC10, the problem statement and its
  falsification, the absolute rules and definition of done, the golden cases, the evaluation set and
  judge rubric, and the spike's success conditions; ontology entries without interview or
  observation evidence were removed.
- Six CPU tests compared native (unemulated) results bit for bit and failed on Linux, where CPU
  BLAS and SDPA round differently by shape. They now allow that rounding only where native ops are
  involved (perplexity across batch sizes: relative 1e-12; STE gradients: relative 1e-6; SDPA
  single-chunk cache logits: 1e-6), and the KV-only report test uses a model whose logits the 2-bit
  cache measurably changes (KL 2.7e-8; it was rounding noise before).

### Fixed
- `app/tests/test_webgpu_golden.py` compares the `fp64` field's NaNs by class, as `webgpu_check.html`
  does. On x86_64 Linux (geneva) the `nan` case rebuilt `fff00000` where the macOS-recorded golden has
  `7ff00000` and failed at 3820b5b; with the change all 161 app tests pass there. `expected` stays bitwise.
- `tests/test_cli.py` passes `--device cpu`: its fake model loader ignores the device, so three CLI
  tests failed on any machine with a visible GPU (reproduced at 3820b5b on an A100).
- Named-task dataset fingerprint capture for lm-eval 0.4.13's `TaskManager.load` API. Five CPU
  provenance checks and the full UCL CPU suite passed (2,337 tests, one CUDA-only skip). Numerical
  and model execution are unchanged. Existing quality records retain their original source and
  empty fingerprints; a separately labeled after-run cache audit is not historical in-memory capture.
- Reference arithmetic on CUDA tensors: powers of two are built from their fp64 bit pattern instead of
  `torch.ldexp`, whose `pow(2, e)` is inexact on CUDA. Before, a few casts differed from the CPU
  (A100: 1 of ~500k bf16 values, up to 7 tf32) and decoding, hence every reference GEMM, raised on
  CUDA. The Triton path runs this code on the GPU for MSE/percentile scale search, zero points and
  dequantized operands, and GPTQ and the AWQ search call it directly, so the Qwen3-0.6B rows of those
  recipes were rerun: WikiText-2 PPL `nvfp4_4o6` 25.5801 → 25.5673, `w4a16_gptq_sequential`
  24.4976 → 24.0839, KIVI-2 23.0081 → 22.9740 (8 windows streamed through the cache: 19.6035 →
  19.5115), and the demo's 16-window `nvfp4_4o6` 24.3761 → 24.2218; the other rerun rows moved by at
  most 0.15% or within lm-eval's standard error, and `nvfp4_awq_shared` did not change. A reference
  cast on CUDA takes about 30% longer per call (idle V100).
