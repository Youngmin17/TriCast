# Qwen3 numerical-design demo

Compare **operand formats**, **accumulation arithmetic**, and **model quality** on
one model, without changing hardware. Numerical semantics come from
[ENGINE.md](../docs/design/ENGINE.md); the
[three-minute presentation guide](../docs/demo/DEMO.md) explains the tables.

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

**Calibration limitation:** Lane D's allowed API does not include `calibrate`,
while GPTQ and frozen EMA require it. Those recipes are recorded as failed, not
silently evaluated with RTN or dynamic scales. Use `--skip-gptq` for the presentation
until the calibration API is authorized and wired in. Selecting `fp8_ema_static`
also reports this limitation. The other stages continue.

## What runs

| Stage | Work | Scheduling estimate, not a measurement |
| --- | --- | --- |
| Setup | Load one model/tokenizer; hash every original linear weight and bias | Tens of seconds to minutes; network/cache dependent |
| 1. Formats | First `self_attn.q_proj`, nine schemes, SQNR/error/encoded bits | Seconds to a minute, including possible JIT compilation |
| 2. Accumulation | Hook the 11th `mlp.down_proj` (last if fewer layers), six MMA variants | Seconds to minutes; full reduction dimension and cold JIT matter |
| 3. Quality | Up to 16 × 2048-token windows **per recipe** in quick mode | Minutes or longer; exact emulation may dominate |
| 4. Generation | Two prompts × four settings, greedy, up to 48 new tokens each | Seconds to minutes; exact decode emulation may dominate |
| 5. Artifacts | JSON and Markdown tables | Seconds |

The goal is a few-minute warm-cache demo on one GPU, **not a validated runtime
promise**. Measured stage times: **실측 후 채움**. Rehearse on the target GPU;
if it exceeds the presentation slot, show the saved report rather than claiming
that a smaller PPL subset is a full evaluation.

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

GPU correctness, full Qwen3 execution, measured runtime, and calibration integration
remain separate validation gates. No example result numbers are supplied here.
