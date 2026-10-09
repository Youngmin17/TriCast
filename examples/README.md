# Qwen3 numerical-design demo

Compare **operand formats**, **accumulation arithmetic**, and **model quality** on
one model, without changing hardware. Numerical semantics come from the executable reference in
[src/tricast/reference/](../src/tricast/reference/).

## Install and run

Bundled recipes live in `src/tricast/recipes/` and load by name through
`tricast.recipe.load_recipe`; `tricast.recipe.list_recipes()` lists them. On a Linux CUDA
machine, use a CUDA-enabled PyTorch installation and install the evaluation and Triton extras:

```bash
PY=/scratch/uceeeee/conda_envs/tricast/bin/python
"$PY" -m pip install -e '.[triton,eval]'
PYTHONNOUSERSITE=1 "$PY" examples/demo_qwen3.py --help
```

Use a **single allocated GPU**. On UCL, the operator must launch through
`gpu_cap.sh run` and respect the two-GPU-per-server cap; the demo does not allocate
GPUs or use SSH. Within that allocation, the presentation command is:

```bash
PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /scratch/uceeeee/conda_envs/tricast/bin/python examples/demo_qwen3.py \
  --quick --skip-gptq
```

Defaults: `Qwen/Qwen3-0.6B`, `--device cuda`, `--dtype bfloat16`, seed 42. PyTorch
uses deterministic algorithms and eager attention. A CUDA run explicitly selects
TriCast's Triton backend: missing kernels fail rather than silently dispatching
the entire GEMM to the slow reference backend. Some scale-search routines still
use reference helpers on GPU tensors; this is not an all-Triton-kernel claim. `--help` needs only the Python standard library.
The model and WikiText-2 are downloaded on the **GPU host** if absent from its HF
cache; do not run this workload or download the checkpoint on the development Mac.

For a shorter rehearsal, reduce PPL work, not the sequence length:

```bash
PYTHONNOUSERSITE=1 "$PY" examples/demo_qwen3.py --quick --max-windows 2 \
  --recipes bf16_passthrough,hopper_fp8_w8a8,fp8_f7_lowacc,mxfp4_w_a \
  --out runs/qwen3_rehearsal
```

`--quick` uses at most the first **16 complete 2048-token windows** of WikiText-2
test, joined with `\n\n` and tokenized by the shared PPL evaluator. `--max-windows N`
further limits them; with `--quick`, the smaller of N and 16 wins. Without either
flag, PPL uses all complete test windows. A limit changes coverage, not the meaning
of a token's loss. Report the actual `n_windows` and `n_tokens`, not just the limit.
`--recipes` selects only the PPL table; the format, MMA, and generation comparisons
are fixed. `--out` must be a new directory to avoid overwriting an earlier run.

Default PPL recipes:

- `bf16_passthrough`, `hopper_fp8_w8a8`
- `fp8_f7_lowacc`, `fp8_f7_decoupled`
- `mxfp8_w_a`, `mxfp4_w_a`, `nvfp4_w_a`, `nvfp4_4o6`
- `w4a16_g128_zp_gptq`, unless `--skip-gptq` is supplied

Also selectable: `blackwell_fp8_w8a8`, `fp8_ema_static`, `mxfp4_rht`.

**Calibration:** GPTQ and frozen EMA require calibration. The demo calls `calibrate`
with its recorded dataset, samples, sequence length and seed before evaluating those recipes
(defaults: `--calib-samples 32`, `--calib-seqlen 512`). Use `--skip-gptq` to omit GPTQ work
for a shorter rehearsal, not because the API is unavailable. Calibration failures remain failed
rows; the demo does not substitute RTN or dynamic scales.

## What runs

| Stage | Work | Scheduling estimate, not a measurement |
| --- | --- | --- |
| Setup | Load one model/tokenizer; hash every original linear weight and bias | Tens of seconds to minutes; network/cache dependent |
| 1. Formats | First `self_attn.q_proj`, nine schemes, SQNR/error/encoded bits | Seconds to a minute, including possible JIT compilation |
| 2. Accumulation | Hook the 11th `mlp.down_proj` (last if fewer layers), six MMA variants | Seconds to minutes; full reduction dimension and cold JIT matter |
| 3. Quality | Up to 16 × 2048-token windows **per recipe** in quick mode | Minutes or longer; exact emulation may dominate |
| 4. Generation | Two prompts × four settings, greedy, up to 48 new tokens each | Seconds to minutes; exact decode emulation may dominate |
| 5. Artifacts | JSON and Markdown tables | Seconds |

Three minutes describes the presentation, **not the evaluation runtime**. The retained
2026-10-01 A100 recording includes calibration and took about 40 minutes for its quick
quality rows. Prepare downloads, JIT and evaluation before presenting; replay the saved report
if needed. A smaller PPL subset is not a full evaluation.

Accumulation uses the first four captured activation rows and first 32 weight rows
(or fewer), retaining **all K columns**. This bounds the standalone GEMM while
preserving its reduction length. Both submatrices are quantized once to
`fp8_tensor` and reused across every preset. They are not the full layer output.
All six MMA variants produce FP32 outputs; BF16 bit mismatch is computed separately
against the FP64-accumulation result rounded to BF16. Timers synchronize CUDA:
three warmups discarded, five measurements, median/mean/interpolated p99. These are
emulation timings, not native Hopper/Blackwell/Ada tensor-core throughput.

Before each recipe, the model is unpatched; after success or exception it is
unpatched again. Original module identities and SHA-256 digests of **every linear
weight and bias** must match. The report records patch reports and forward-hook
counts for every replaced module. This proves those modules ran, **not** GPU kernel
parity or absence of an internal numerical fallback. `kernel_verified=false`
remains explicit until separate GPU validation supplies that evidence.

## Outputs and failure handling

The default directory is `runs/demo_<UTC>/`:

- `results.json`: arguments, stage/row status, metrics, raw PPL metadata (including
  dataset fingerprint), MMA definitions and timing samples, generated texts,
  recipe-file hashes, patch/forward evidence, restoration checks.
- `report.md`: the same comparisons as readable tables and an environment summary.
- `env.json`: source SHA/dirty flag, package/CUDA/GPU information, seed, actual loaded
  model commit when available, arguments and evaluated dataset fingerprints.

Progress emits one line per stage, then the tables. Each failed comparison row
retains its exception and other rows/stages continue. Exit status is nonzero if
any stage/row/environment capture failed. Verify the terminal marker
`DEMO_DONE status=ok` **and** JSON statuses; neither a process exit nor a `.done`
file alone establishes correctness. `partial_failure` means exactly that.

PPL elapsed time includes patching, evaluation, and original-weight verification;
it is a **single-run observation**, not a throughput benchmark. Quick PPL and two
short completions do not establish full benchmark or bit-exact backend validation.

## Local authoring checks (no model download)

```bash
.venv/bin/python -m py_compile examples/demo_qwen3.py
.venv/bin/python -m ruff check examples/demo_qwen3.py
.venv/bin/python examples/demo_qwen3.py --help
```

Local authoring checks do not establish GPU correctness or full-model quality. Full Qwen results
are summarized in `support_matrix.yaml` (`model_families.qwen.pretrained_quality`), and the
arithmetic suites have their own coverage and source identities. The demo's
`kernel_verified=false` is not a bit-parity pass.
