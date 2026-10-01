# TriCast

**Emulate the arithmetic of low-precision matrix units — formats, quantization and the tensor-core
accumulator itself — bit for bit on CUDA cores, and measure what it does to a language model.**

[한국어 (main)](README.md) · [Engine contract](docs/design/ENGINE.md) · [Support matrix](support_matrix.yaml) ·
[Full results](docs/results/qwen3_0.6b.md) · [Demo script (Korean)](docs/demo/DEMO.md)

> Change one value in a recipe and get back, without rewriting a kernel, what that arithmetic does to the
> model's quality and to the accumulator's error — with the environment of the run recorded.

**Contents** · [One dot product, four answers](#one-dot-product-four-answers) · [Why](#why) ·
[What it models](#what-it-models) · [Quickstart](#quickstart) · [Results](#results-qwen3-06b) ·
[Three-minute demo](#three-minute-demo) · [Verification](#how-it-is-verified) ·
[Repository layout](#repository-layout) · [Limitations](#limitations)

---

## One dot product, four answers

The same 32-element FP8 (E4M3) dot product, accumulated the way four different datapaths do it
(`python examples/one_dot_product.py` reproduces it on CPU with the exact reference):

```
accumulator                      result (fp32)            bits
fp64, one rounding               291.274658203125         0x4391a328
Blackwell FP8   CoFDA F=25       291.2746276855469        0x4391a327
Hopper FP8      CoFDA F=13       291.25                   0x4391a000
narrow          CoFDA F=7        290.0                    0x43910000
```

A tensor core does not add products one at a time. It aligns a chunk of them to the largest
exponent, cuts every bit below a fixed fraction width `F`, sums the survivors exactly, and
normalises once:

```
             chunk max exponent Emax = 8, F = 10
p0  + 1.101 · 2^8      | 1.1010000000 |
p1  − 1.011 · 2^5      | 0.0010110000 |        shifted right by 3
p2  + 1.111 · 2^-1     | 0.0000000011 | 11     shifted right by 9; the bits past F are dropped
                         └─ F = 10 ──┘
```

That width, the chunk size, and whether the running accumulator joins the truncation are
design parameters. They are invisible in a spec sheet and they move model quality. TriCast makes
each of them a number you can set.

## Why

People who design and verify low-precision models and NPU datapaths rewrite code and CUDA kernels
whenever they change a bit width, a scaling method, the accumulation, sparsity, outlier handling or
the GPU generation, just to see the effect. Three of the five practitioners in the team's ten
interviews said so (logs 1, 9 and 10; [problem statement](docs/PROBLEM.md),
[interview record](docs/research/interviews.md), [spec](docs/SPEC.md), all in Korean).

## What it models

| layer | options |
|---|---|
| **number formats** | any float `ExMy` (IEEE / finite-NaN / fnuz / no-special, custom bias, subnormals on/off), integers and fixed point (`intN`, `uintN`, `frac=F`), power-of-two scales (E8M0); registry: fp32 · tf32 · bf16 · fp16 · fp8 e4m3/e5m2 (+fnuz) · fp6 e3m2/e2m3 · fp4 e2m1 · ue4m3 · int8/4/2 · mxint8/4 |
| **rounding** | nearest-even, nearest-away, toward zero, up, down, stochastic (with explicit noise and bit width) |
| **scaling** | per tensor / row (channel, token) / group / 2-D block; absmax, power-of-two floor/ceil (MX), MSE search, percentile, **Four-over-Six**; two-level (NVFP4); integer and float zero points |
| **schemes** | MXFP8/6/4, MXINT8/4, **NVFP4**, block floating point (MSFP12/16, `bfp<m>_b<block>`), FP8 tensor/row/group/block (DeepSeek), INT8, INT4 g128 ± zero point, **KIVI** 2/4-bit KV cache |
| **calibration** | static observers (min-max, **EMA**, Transformer-Engine delayed history, percentile, MSE); WikiText-2 / C4 / Pile samples |
| **algorithms** | **GPTQ** (any format, groups, act-order, sequential), **AWQ** and SmoothQuant (shared-input groups), Hadamard and random Hadamard rotations, STE for QAT |
| **weight structure** | N:M (e.g. 2:4) and unstructured magnitude sparsity; outliers kept in a higher-precision format beside the quantized weight (SpQR-style, separate fp32 path) |
| **MMA accumulation** | CoFDA (C-fused / C-decoupled), GDFS (two-level group sums), DeepSeek-style FP32 promotion, IEEE FP32 FMA chain, FP64, exact integer; block scales applied per product, per group, at promotion or in the epilogue |
| **hardware presets** | Hopper FP8 (F=13, CS=32), Ada FP8, Blackwell FP8 (F=25), Blackwell FP4 (GDFS G=6 F=35), DeepSeek FP8 promotion — every preset carries its source |
| **models and tasks** | any Hugging Face causal LM (`nn.Linear` layers, per-layer rules), WikiText-2 perplexity, every lm-eval task, per-layer error reports (MSE, SQNR, cosine, logits KL, accumulator ULP error against fp64 accumulation) |

## Quickstart

```bash
pip install -e ".[triton,eval]"          # Linux + CUDA for the Triton kernels; CPU runs the reference
pip install -e ".[agent]"                # only for `tricast agent` with the Claude API
```

```python
import torch, tricast
from tricast.eval.ppl import perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

report = tricast.patch_model(model, "hopper_fp8_w8a8")   # every decoder linear, Hopper accumulation
print(perplexity(model, tok, dataset="wikitext2"))
```

| command | what it does |
|---|---|
| `tricast formats` · `schemes` · `presets` | list number formats, quantization schemes, hardware presets |
| `tricast cast` | round values onto a format grid |
| `tricast recipe-check <name or path>` | validate a recipe; errors name the field path |
| `tricast ppl --model M --recipe R` | WikiText-2 perplexity |
| `tricast eval --model M --recipe R --tasks hellaswag,coqa` | lm-eval tasks |
| `tricast report --model M --recipe R` | per-layer MSE, SQNR, cosine and accumulator ULP error, model logits KL |
| `tricast run CONFIG.yaml` | run a recipe or sweep configuration (resumes where it stopped) |
| `tricast agent "<request>"` | natural-language request to a validated recipe |
| `tricast-lm-eval …` | the lm-eval CLI as is (`--model tricast --model_args pretrained=…,recipe=…`) |

Twenty-six named recipes live in `src/tricast/recipes/` (`hopper_fp8_w8a8`, `nvfp4_w_a`, `mxfp4_w_a`,
`fp8_2of4_sparse`, `nvfp4_outliers`, `w4a16_g128_zp_gptq`, `kivi2_kv`, …).

### A recipe

```yaml
name: my_accelerator
defaults:
  weight:     {scheme: nvfp4}
  activation: {format: fp8_e4m3, granularity: row}
  mma:        {algorithm: cofda, f_bits: 11, chunk_size: 64, c_mode: decoupled}
  sparsity:   {kind: "n:m", n: 2, m: 4}     # 2:4 weight sparsity
  outliers:   {fraction: 0.005, format: bf16}  # the 0.5% largest weights kept in bf16
overrides:
  - layers: "0,-1"                 # first and last decoder blocks: linear layers stay unquantized
    skip: true
  - modules: [down_proj]
    weight: {scheme: mxfp8_e4m3}
kv: {preset: kivi2, mode: cache}   # KIVI-2 KV cache in every block (kv.layers narrows it)
```

## Results (Qwen3-0.6B)

WikiText-2 test, all 146 windows of 2048 tokens, bf16 model on one A100 (native perplexity 20.966).
Weights and activations are FP8 E4M3 with one scale per tensor in every row; only the accumulator
changes.

| accumulator | perplexity | vs native |
| --- | ---: | ---: |
| exact (fp64) | 21.167 | +0.96% |
| Blackwell, F=25 | 21.184 | +1.04% |
| Hopper, F=13, chunks of 32 | 21.204 | +1.13% |
| F=7, running sum in a separate 23-bit register | 21.275 | +1.47% |
| F=7, running sum truncated with every chunk | **29.158** | **+39.1%** |

- The last two rows differ only in whether the running sum joins each chunk's truncation.
- Measured in ULPs of the bf16 output (V100, 1024 tokens), Hopper is 2.62 ULPs off on average and F=7
  C-decoupled 60.5, yet C-decoupled leaves the logits about as close to the native model (KL 0.024
  against 0.022). An ULP budget alone does not predict model quality.
- Pruning every linear to 2:4 by magnitude without fine-tuning collapses the model (65,188); the same
  pruning in plain PyTorch gives the same perplexity on the first 4 windows (73,434), so the collapse is
  the pruning. Keeping the 0.5% largest weights in bf16 barely moves NVFP4 (26.18 against 26.22).
- The emulated bf16 passthrough reproduces the native model to 0.004%.

Perplexity of 26 recipes, block formats, GPTQ, AWQ, KIVI and lm-eval (HellaSwag, CoQA), with the
environment of every run: [docs/results/qwen3_0.6b.md](docs/results/qwen3_0.6b.md).

## Three-minute demo

```bash
python examples/demo_qwen3.py --quick     # about 40 minutes on one A100 → report.md · results.json · env.json
```

| time | what is shown | key number |
|---|---|---|
| 0:00 | the problem | interview quote: the kernel is rewritten for every combination |
| 0:25 | one weight in nine formats | 4-bit rows span 18.6–21.2 dB SQNR depending on the scaling |
| 1:00 | the same FP8, different accumulators | F=7 C-fused relative error 0.158, C-decoupled 0.0078 — 20× |
| 1:40 | model quality (first 16 windows) | F=7 C-fused 27.58 against C-decoupled 20.51; 2:4 without retraining 47,112 |
| 2:15 | generated text | broken words under F=7 C-fused and MXFP4 |
| 2:45 | conclusion and artifacts | bit exactness is checked separately: 896 GPU tests and 1716 NADPE vectors |

Last run (commit `9e9bfed`, A100): `DEMO_DONE status=ok`, all 11 recipes. Screens, talking points and
caveats: [docs/demo/DEMO.md](docs/demo/DEMO.md) (Korean).

## How it is verified

Emulation is only useful if it is the arithmetic you think it is. Every claim below is a test in
this repository.

| check | evidence |
|---|---|
| reference casts vs PyTorch native conversions | fp8 ×4 formats, bf16, fp16 — bit-identical on 100k random values per format + edge cases |
| reference MX quantization vs `microsoft/microxcaling` | bit-identical (even / nearest / floor), except inputs microxcaling misclassifies via fp32 `log2` |
| reference MMA vs **NADPE** CUDA kernels (MICRO'26), built standalone | **1716 / 1716** cases bit-identical — FP8 CoFDA / C-decoupled / GDFS, NVFP4, MXFP4 |
| Triton kernels vs reference | GPU suite (`tests/gpu`, quantization + MMA, random, adversarial and special values): 896 passed on A100 and on V100 at `9e9bfed` |
| Triton MMA vs NADPE on full GEMMs | bit-identical on 2048×1024×3072, 2048×3072×1024 and 4096³ for CoFDA, C-decoupled and GDFS (`scripts/bench/bench_mma_vs_nadpe.py`) |
| CPU suite (`pytest --ignore=tests/gpu`) | 2,337 passed on Linux, torch 2.8 (`82db449`) |

Triton links libdevice with flush-to-zero; the kernels use IEEE PTX for every fp32 operation that
can meet a subnormal, and the reference computes each fp32 operation in fp64 and rounds once —
so results do not depend on the device.

## Repository layout

```
TriCast/
├─ src/tricast/                 the library
│  ├─ formats.py, rounding.py   number formats and rounding
│  ├─ quant/                    quantization specs, API, observers, sparsity and outliers (structure.py)
│  ├─ mma/                      MMA accumulation specs and hardware presets
│  ├─ reference/                exact reference (fp64 / int64) — the definition of the arithmetic
│  ├─ kernels/                  Triton kernels: quantization, MMA, small-M decoding
│  ├─ nn/                       Hugging Face patching (EmuLinear, patch_model)
│  ├─ kv/                       KV-cache quantization (KIVI)
│  ├─ weight_quant/             GPTQ
│  ├─ transforms.py             AWQ, SmoothQuant, Hadamard
│  ├─ calibration.py            calibration data and observer fitting
│  ├─ eval/                     perplexity, lm-eval adapter, sweep runner, run environment
│  ├─ analysis.py               per-layer errors and accumulator ULP report
│  ├─ agent/ · tools/ · rag.py  natural-language request → recipe agent
│  ├─ recipes/                  26 named recipes
│  └─ cli.py                    the tricast command
├─ tests/        CPU tests · gpu/ (Triton bit exactness) · data/nadpe/ (golden vectors) · harness/ (golden cases)
├─ configs/      e2e/ (configs that reproduce the results document) · sweeps/
├─ scripts/      bench/ (performance) · e2e/ (result tables, cross-checks) · nadpe_oracle/ (independent implementation)
├─ examples/     demo_qwen3.py (three-minute demo) · one_dot_product.py
├─ evals/        30 natural-language parsing cases, scorer and judge prompt
├─ docs/         spec, problem statement, ontology, interviews, engine contract, results, demo, spike
├─ AGENTS.md     rules for coding agents (CLAUDE.md only loads it)
└─ support_matrix.yaml   source of truth for what is implemented and verified
```

Two branches: `develop` for work and `main`, fast-forwarded once a state is verified. Changes are in
[CHANGELOG.md](CHANGELOG.md). The course deliverables (AI Capstone Design, in Korean) are listed in the
[Korean README](README.md#ai캡스톤-과제-산출물).

## Limitations

- Attention `QKᵀ` / `PV` matmuls are not emulated yet; KV-cache quantization is.
- Hardware presets come from published measurements (NADPE); TriCast has not yet compared them
  against silicon itself (`tricast.probe` is planned).
- Emulation on CUDA cores is slower than the native tensor-core GEMM it models: on an A100 the
  Hopper FP8 accumulator runs at 0.15 TMAC/s (NADPE's CUDA kernel: 0.15), one 2048-token Qwen3-0.6B
  window takes about 6 s, and one decoded token about 0.1 s (V100: 0.17 s), most of it host-side
  kernel launches. First use of a new configuration compiles Triton kernels.
- The agent that turns a natural-language request into a recipe (`tricast agent`) is tested with
  a mocked model client. The live Claude API path needs `pip install -e ".[agent]"` and an
  `ANTHROPIC_API_KEY`; without them `--llm auto` uses the offline parser and prints which parser ran.

## Built on

NADPE / MMA-Emu (MICRO'26, *Not All Dot Products Are Equal*), microsoft/microxcaling, the OCP
Microscaling spec, NVIDIA NVFP4, DeepSeek-V3 (FP8 promotion), GPTQ, AWQ, SmoothQuant, QuaRot,
KIVI, Four Over Six. MIT licensed.
