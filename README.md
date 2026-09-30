# TriCast

**Emulate the arithmetic of low-precision matrix units — formats, quantization and the tensor-core
accumulator itself — bit for bit on CUDA cores, and measure what it does to a language model.**

[한국어 README](README.ko.md) · [Engine contract](docs/design/ENGINE.md) · [Support matrix](support_matrix.yaml)

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

## What it models

| layer | options |
|---|---|
| **number formats** | any float `ExMy` (IEEE / finite-NaN / fnuz / no-special, custom bias, subnormals on/off), integers and fixed point (`intN`, `uintN`, `frac=F`), power-of-two scales (E8M0); registry: fp32 · tf32 · bf16 · fp16 · fp8 e4m3/e5m2 (+fnuz) · fp6 e3m2/e2m3 · fp4 e2m1 · ue4m3 · int8/4/2 · mxint8/4 |
| **rounding** | nearest-even, nearest-away, toward zero, up, down, stochastic (with explicit noise and bit width) |
| **scaling** | per tensor / row (channel, token) / group / 2-D block; absmax, power-of-two floor/ceil (MX), MSE search, percentile, **Four-over-Six**; two-level (NVFP4); integer and float zero points |
| **schemes** | MXFP8/6/4, MXINT8/4, **NVFP4**, block floating point (MSFP12/16, `bfp<m>_b<block>`), FP8 tensor/row/group/block (DeepSeek), INT8, INT4 g128 ± zero point, **KIVI** 2/4-bit KV cache |
| **calibration** | static observers (min-max, **EMA**, Transformer-Engine delayed history, percentile, MSE); WikiText-2 / C4 / Pile samples |
| **algorithms** | **GPTQ** (any format, groups, act-order, sequential), **AWQ** and SmoothQuant (shared-input groups), Hadamard and random Hadamard rotations, STE for QAT |
| **MMA accumulation** | CoFDA (C-fused / C-decoupled), GDFS (two-level group sums), DeepSeek-style FP32 promotion, IEEE FP32 FMA chain, FP64, exact integer; block scales applied per product, per group, at promotion or in the epilogue |
| **hardware presets** | Hopper FP8 (F=13, CS=32), Ada FP8, Blackwell FP8 (F=25), Blackwell FP4 (GDFS G=6 F=35), DeepSeek FP8 promotion — every preset carries its source |
| **models and tasks** | any Hugging Face causal LM (`nn.Linear` layers, per-layer rules), WikiText-2 perplexity, every lm-eval task, per-layer error reports (MSE, SQNR, cosine, logits KL) |

## How it is verified

Emulation is only useful if it is the arithmetic you think it is. Every claim below is a test in
this repository.

| check | evidence |
|---|---|
| reference casts vs PyTorch native conversions | fp8 ×4 formats, bf16, fp16 — bit-identical on 100k random values per format + edge cases |
| reference MX quantization vs `microsoft/microxcaling` | bit-identical (even / nearest / floor), except inputs microxcaling misclassifies via fp32 `log2` |
| reference MMA vs **NADPE** CUDA kernels (MICRO'26), built standalone | **1716 / 1716** cases bit-identical — FP8 CoFDA / C-decoupled / GDFS, NVFP4, MXFP4 |
| Triton kernels vs reference | GPU suite (`tests/gpu`, quantization + MMA, random, adversarial and special values): 824 passed on A100 and on V100 |
| Triton MMA vs NADPE on full GEMMs | bit-identical on 2048×1024×3072, 2048×3072×1024 and 4096³ for CoFDA, C-decoupled and GDFS (`scripts/bench/bench_mma_vs_nadpe.py`) |

Triton links libdevice with flush-to-zero; the kernels use IEEE PTX for every fp32 operation that
can meet a subnormal, and the reference computes each fp32 operation in fp64 and rounds once —
so results do not depend on the device.

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

```bash
tricast schemes                                   # what can be quantized, and how
tricast ppl    --model Qwen/Qwen3-0.6B --recipe nvfp4_w_a
tricast report --model Qwen/Qwen3-0.6B --recipe mxfp4_w_a        # per-layer MSE / SQNR / cosine / KL
python -m tricast.eval.lmeval --model tricast \
    --model_args pretrained=Qwen/Qwen3-0.6B,recipe=hopper_fp8_w8a8 --tasks hellaswag,coqa
python examples/demo_qwen3.py --quick             # formats, accumulators, PPL on 16 windows, generation
```

### A recipe

```yaml
name: my_accelerator
defaults:
  weight:     {scheme: nvfp4}
  activation: {format: fp8_e4m3, granularity: row}
  mma:        {algorithm: cofda, f_bits: 11, chunk_size: 64, c_mode: decoupled}
overrides:
  - layers: "0,-1"                 # first and last decoder blocks: linear layers stay unquantized
    skip: true
  - modules: [down_proj]
    weight: {scheme: mxfp8_e4m3}
kv: {preset: kivi2, mode: cache}   # KIVI-2 KV cache in every block (kv.layers narrows it)
```

## Results — Qwen3-0.6B

(filled from `examples/demo_qwen3.py` runs; see `docs/demo/DEMO.md`)

## Limitations

- Attention `QKᵀ` / `PV` matmuls are not emulated yet; KV-cache quantization is.
- Hardware presets come from published measurements (NADPE); TriCast has not yet compared them
  against silicon itself (`tricast.probe` is planned).
- Emulation on CUDA cores is slower than the native tensor-core GEMM it models: on an A100 the
  Hopper FP8 accumulator runs at 0.15 TMAC/s (NADPE's CUDA kernel: 0.15), one 2048-token Qwen3-0.6B
  window takes about 6 s. First use of a new configuration compiles Triton kernels.
- The agent that turns a natural-language request into a recipe (`tricast agent`) is tested with
  a mocked model client. The live Claude API path needs `pip install -e ".[agent]"` and an
  `ANTHROPIC_API_KEY`; without them `--llm auto` uses the offline parser and prints which parser ran.

## Built on

NADPE / MMA-Emu (MICRO'26, *Not All Dot Products Are Equal*), microsoft/microxcaling, the OCP
Microscaling spec, NVIDIA NVFP4, DeepSeek-V3 (FP8 promotion), GPTQ, AWQ, SmoothQuant, QuaRot,
KIVI, Four Over Six. MIT licensed.
