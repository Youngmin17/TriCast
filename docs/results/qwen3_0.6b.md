# Qwen3-0.6B under emulated number formats and accumulators

Perplexity and lm-eval numbers come from `tricast run` records (`configs/e2e/*.yaml`), joined with
`scripts/e2e/summarize.py`; the per-layer error figures come from `tricast report`, and the KV path
comparison from `scripts/e2e/check_kv_paths.py`. One run per row: the emulation is bit-exact and
deterministic, so a rerun reproduces the same perplexity to the last bit (`mxfp8_w_a` run again at a
later commit gave 21.294279307019544 both times). The exception was reference code that some recipes
run on the GPU — MSE scale search, zero points, GPTQ and the AWQ search — which was not exact on CUDA
until `9e9bfed` (CHANGELOG, Fixed); the rows of those recipes were rerun there and say so.
`native` is the unpatched Hugging Face model.

## Perplexity — WikiText-2 test, A100

146 windows of 2048 tokens (298,862 scored tokens, GPTQ convention), dataset fingerprint
`a46124b21ac53738`. Model `Qwen/Qwen3-0.6B@c1899de289a0` in bf16; NVIDIA A100-SXM4-80GB, CUDA 12.8,
torch 2.8.0+cu128, triton 3.4.0, transformers 4.55.2; TriCast `49a6a76` (clean tree), except the
exact-accumulation row `fp8_w8a8_fp64acc`, run at `032684b`, the two weight-structure rows, run at
`f3d6fe6` (`configs/e2e/qwen3_0.6b_ppl_d.yaml`), and `nvfp4_4o6`, `nvfp4_awq_shared` and the two GPTQ
rows, rerun at `9e9bfed`: `nvfp4_awq_shared` reproduced to the last digit, `nvfp4_4o6` moved from
25.5801 to 25.5673, `w4a16_g128_zp_gptq` from 24.7228 to 24.7281 and `w4a16_gptq_sequential` from
24.4976 to 24.0839.

### References

| run | PPL | vs native | what it checks |
| --- | ---: | ---: | --- |
| native | 20.9662 | | unpatched model |
| `bf16_passthrough` | 20.9669 | +0.004% | every linear through the emulator with an IEEE fp32 FMA chain (SPEC AC4: ≤ 0.1%) |
| `fp64_reference` | 20.9624 | −0.02% | unquantized operands, fp64 accumulation, one rounding |

### Same FP8 operands, different accumulators

All rows quantize weights and activations to FP8 E4M3 with one dynamic fp32 scale per tensor; only
the dot-product datapath changes. FP8 quantization alone costs +0.96% (exact accumulation); the
last column is what each accumulator adds on top.

| accumulator | PPL | vs native | vs exact accumulation |
| --- | ---: | ---: | ---: |
| exact (fp64), `fp8_w8a8_fp64acc` | 21.1673 | +0.96% |  |
| Blackwell: CoFDA F=25, chunks of 32 | 21.1844 | +1.04% | +0.08% |
| Ada: CoFDA F=13, chunks of 16 | 21.1960 | +1.10% | +0.14% |
| Hopper: CoFDA F=13, chunks of 32 | 21.2035 | +1.13% | +0.17% |
| F=7, running sum kept apart at 23 bits (C-decoupled) | 21.2749 | +1.47% | +0.51% |
| F=7, running sum truncated with every chunk (C-fused) | **29.1576** | **+39.07%** | **+37.75%** |

At F=7 the two datapaths see the same 7-bit-aligned products; what differs is whether the running
sum is also truncated to the chunk's alignment every 32 products. That choice alone moves the model
from +1.5% to +39%.

### Formats and scaling

| recipe | weights / activations | accumulator | PPL | vs native |
| --- | --- | --- | ---: | ---: |
| `fp8_delayed_history` | FP8, activation scale from the amax history | Hopper | 21.1747 | +0.99% |
| `fp8_ema_static` | FP8, calibrated EMA activation scale | Hopper | 21.2590 | +1.40% |
| `deepseek_fp8_block` | FP8, 1×128 activation and 128×128 weight blocks | Hopper, fp32 promotion every 128 | 21.2518 | +1.36% |
| `int8_row_w8a8` | INT8 per row | exact integer | 21.3647 | +1.90% |
| `mxfp8_w_a` | MXFP8 E4M3, E8M0 per 32 | CoFDA F=25 | 21.2943 | +1.57% |
| `mxfp6_w_a` | MXFP6 E3M2, E8M0 per 32 | CoFDA F=25 | 21.6150 | +3.09% |
| `mxfp4_w_a` | MXFP4 E2M1, E8M0 per 32 | GDFS G=6 F=35 | 33.9409 | +61.88% |
| `mxfp4_rht` | MXFP4 after a seeded random Hadamard rotation (blocks of 128 along K) | GDFS G=6 F=35 | 43.3677 | +106.85% |
| `msfp12_bfp` | block floating point, 4-bit mantissas, E8M0 per 16 | GDFS G=6 F=35 | 36.4076 | +73.65% |
| `nvfp4_w_a` | NVFP4 E2M1, UE4M3 per 16 and one fp32 scale per tensor | GDFS G=6 F=35 | 26.2197 | +25.06% |
| `nvfp4_4o6` | NVFP4, each block scaled so its amax maps to 4 or 6 (Four-over-Six) | GDFS G=6 F=35 | 25.5673 | +21.95% |
| `nvfp4_smoothquant` | NVFP4 after SmoothQuant (α = 0.5) | GDFS G=6 F=35 | 25.7532 | +22.83% |
| `nvfp4_awq_shared` | NVFP4 after AWQ scaling, one scale per group of projections sharing an input | GDFS G=6 F=35 | 24.3041 | +15.92% |
| `mixed_first_last_bf16` | MXFP4 in decoder blocks 1–26; blocks 0 and 27 unquantized | GDFS G=6 F=35 | 29.6883 | +41.60% |
| `w4a16_g128_zp_gptq` | GPTQ UINT4 weights, groups of 128 with zero points; bf16 activations | CoFDA F=23, chunks of 32 | 24.7281 | +17.94% |
| `w4a16_gptq_sequential` | the same weight format, GPTQ fitted block by block on the quantized model's outputs | IEEE fp32 FMA chain | 24.0839 | +14.87% |
| `nvfp4_outliers` | NVFP4; the 0.5% largest \|w\| kept in BF16 and added through an fp32 path | GDFS G=6 F=35 | 26.1794 | +24.86% |
| `fp8_2of4_sparse` | FP8 per tensor; weights pruned to 2:4 by magnitude, no fine-tuning | Hopper | 65,188 | collapses |

SmoothQuant, AWQ and the two GPTQ rows calibrate on 128 windows of 2048 tokens from WikiText-2
train (seed 42 for `nvfp4_smoothquant` and `w4a16_g128_zp_gptq`, 0 for the others, as their recipes
set). Keeping the first and last of the 28 decoder blocks unquantized reduces MXFP4's increase from
+61.9% to +41.6%. The two GPTQ rows differ in both calibration order and accumulator, so their gap
cannot be assigned to either. `w4a16_gptq_sequential` moved the most when the reference arithmetic
these recipes run on the GPU was made exact (24.4976 → 24.0839). On one V100 the code before and
after that fix gives 24.4368 and 24.1584, and the fixed code gave 24.1584 to the last digit again on
a second V100, so the change is the fix, not run-to-run noise. Each block is fitted on the outputs
of the blocks already quantized, so a difference in an early block reaches every later fit.

The last two rows change the weight structure — the sparsity ratio and the outlier-preservation
scheme that log 10 names.
Keeping the 0.5% largest weights in BF16 moves NVFP4 from 26.22 to 26.18 — a small effect, and on
the demo's first 16 windows it goes the other way (25.12 against 25.05). Pruning every linear to 2:4
by magnitude without fine-tuning collapses the model. Pruning the same weights in plain PyTorch,
without TriCast, gives the same perplexity on the first 4 windows (73,434.3212 both, dense 23.3331;
`scripts/e2e/check_sparsity_pruning.py` at `9e9bfed` on a V100-PCIE-16GB, which also writes
`env.json`), so the collapse is the pruning, not the emulation; the 2:4 workflow of Mishra et al.
retrains after pruning, which TriCast does not model.

The rotation makes MXFP4 worse here although it makes each operand easier to quantize.
`tricast report` on 2 × 512 WikiText-2 tokens (V100) shows why the operand metrics mislead: with the
rotation the activation SQNR rises in all 196 linear layers (median 17.2 → 18.8 dB) and the weight
SQNR barely moves (18.7 → 18.7 dB), but the SQNR of the layer outputs does not improve (median
16.4 → 16.3 dB, lower in 112 of 196 layers) and the logits KL divergence grows from 0.57 to 0.82. The
rotation itself preserves `x·Wᵀ` exactly (fp64 test in `tests/test_nn_patch.py`); the cause of the
output error is not established.

### Accumulator error in ULPs

`tricast report` on 2 × 512 WikiText-2 tokens (dataset fingerprint `5972d1f2…`) runs every linear's
GEMM again with fp64 accumulation of the same FP8 operands and counts the difference in ULPs of
the bf16 output (`mma_ulp`, ENGINE §6.5); 196 layers each. NVIDIA V100-PCIE-16GB, TriCast `a1b0ede`
(clean tree). Perplexity here is over these 1024 tokens only (unpatched: 25.481).

| accumulator | mean ULP | median of the per-layer p99 | largest per-layer p99 | outputs within 0 ULP | logits KL | PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blackwell, F=25 | 0.00 | 0 | 0 | 100.0% | 0.022 | 26.15 |
| Hopper, F=13, chunks of 32 | 2.62 | 3 | 8 | 83.3% | 0.022 | 26.11 |
| Ada, F=13, chunks of 16 | 2.63 | 3 | 8 | 83.2% | 0.021 | 26.20 |
| F=7, C-decoupled | 60.54 | 76 | 143 | 23.4% | 0.024 | 26.05 |
| F=7, C-fused | 319.52 | 8,092 | 31,423 | 2.7% | 0.297 | 35.25 |

At F=25 the accumulator's truncation stays below the bf16 output's resolution. The F=7 decoupled
datapath is tens of ULPs off yet leaves the logits nearly as close to the unpatched model as Hopper
does (KL 0.024 against 0.022); only the fused variant, whose running sum is truncated with every
chunk, moves the model. An ULP budget alone does not predict model quality, so the report gives
both; the interviews list ULP error and model accuracy among what each combination affects (pain 6,
logs 9 and 10).

## KIVI KV cache — WikiText-2 test, V100

Linear layers stay native, so only the KV cache changes. NVIDIA V100-PCIE-32GB, CUDA 12.8, torch
2.8.0+cu128, triton 3.4.0, transformers 4.55.2, lm_eval 0.4.13; TriCast `5c09f47` (clean tree) for the
native rows. The KIVI rows were rerun at `9e9bfed` (clean tree), because KIVI's zero points and fp16
dequantization ran reference code that was not exact on CUDA before; they moved by up to 0.5%.
The V100 native perplexity (20.9651) differs from the A100 one in the third decimal: attention and
normalization run natively and their kernels differ between the two GPUs.

`fakequant` scores every token as a decode step reading the quantized cache (ENGINE §3.13). Keys are
2- or 4-bit per channel in groups of 32 tokens, quantized once a block of 128 tokens is complete
(the remainder stays in full precision); values are quantized per token, except the latest 128.

| run | all 146 windows | vs native | first 8 windows, streamed through the cache | vs native |
| --- | ---: | ---: | ---: | ---: |
| native | 20.9651 | | 17.8377 | |
| KIVI-4 | 21.0119 | +0.22% | 17.8858 | +0.27% |
| KIVI-2 | 22.9740 | +9.58% | 19.5115 | +9.38% |

The one-forward (`fakequant`) and token-by-token (`cache`) paths compute the same attention given the
same K/V: `scripts/e2e/check_kv_paths.py` finds at most 6e-6 relative difference on Qwen3-0.6B layers
0 and 14 (fp32, 2048 tokens). Whole-model perplexities of the two paths still differ slightly: for
KIVI-2 by 0.42% (bf16, the same 8 windows) and 0.22% (fp32, 2 windows), against 0.03% and 1e-7 for
the native model. A batched forward and token-by-token decoding round the K/V projections
differently, and a 2-bit grid can turn such last-bit differences into different quantized values.

### CoQA through the quantized cache

lm-eval CoQA, all 500 dialogues (EM / F1), `cache` mode: the prompt is prefilled in full precision
and every generated token reads the quantized cache (ENGINE §3.13). A quantized cache is evaluated
one request at a time (lm-eval pads batches without an attention mask), so the native row here is
batch size 1 too; the same model at batch size 16 scores EM 0.5792 and F1 0.7071.

| run | CoQA EM | CoQA F1 |
| --- | ---: | ---: |
| native | 0.5770 ± 0.0194 | 0.7066 ± 0.0165 |
| KIVI-4 | 0.5738 ± 0.0194 | 0.7041 ± 0.0165 |
| KIVI-2 | 0.5733 ± 0.0195 | 0.6977 ± 0.0168 |

KIVI-2 raises perplexity by 9.6% but lowers CoQA EM by only 0.4 points, within one standard error.
The two evaluations quantize different things: perplexity scores every position against quantized
keys and values, while in CoQA only the few generated answer tokens read the quantized cache, and
the most recent values (128 tokens) and keys (the residual under 128) stay in full precision. Which
of these accounts for the gap was not measured.

## lm-eval — V100

HellaSwag (0-shot `acc_norm`) on its first 2000 of 10,042 items and CoQA (all 500 dialogues, EM /
F1), lm-eval 0.4.13; ± is lm-eval's standard error. Batch size 16, except recipes whose activation
scale spans tokens (one scale per tensor, or NVFP4's two-level scale): lm-eval pads without an
attention mask, so they run one request at a time (ENGINE §6.4). At batch size 1 the native model
scores CoQA EM 0.5770 instead of 0.5792, so batch-1 rows carry that much batch effect. NVIDIA
V100-PCIE-32GB, except `w4a16_g128_zp_gptq` on a V100-PCIE-16GB; TriCast `5c09f47`, except
`fp8_f7_lowacc`, which ran at `b68addd` (the small-M kernel: faster decoding, bit-identical results),
and `w4a16_g128_zp_gptq`, rerun at `9e9bfed` because GPTQ runs the reference code fixed there (at
`b68addd` it scored 0.4465 / 0.5425 / 0.6605).

| recipe | HellaSwag acc_norm | CoQA EM | CoQA F1 | batch | wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 0.4545 ± 0.0111 | 0.5792 ± 0.0194 | 0.7071 ± 0.0165 | 16 | 15 min |
| `hopper_fp8_w8a8` | 0.4500 ± 0.0111 | 0.5688 ± 0.0195 | 0.7025 ± 0.0165 | 1 | 273 min |
| `fp8_f7_lowacc` | 0.4290 ± 0.0111 | 0.5215 ± 0.0203 | 0.6407 ± 0.0179 | 1 | 111 min |
| `mxfp4_w_a` | 0.4290 ± 0.0111 | 0.4270 ± 0.0202 | 0.5569 ± 0.0188 | 16 | 134 min |
| `nvfp4_w_a` | 0.4325 ± 0.0111 | 0.5040 ± 0.0202 | 0.6246 ± 0.0182 | 1 | 336 min |
| `w4a16_g128_zp_gptq` | 0.4490 ± 0.0111 | 0.5418 ± 0.0198 | 0.6622 ± 0.0173 | 16 | 84 min |

Hopper FP8 stays within lm-eval's standard error of the native scores on both tasks. The other rows
lose 0.6–2.6 HellaSwag points but separate more on CoQA, which generates its answers: GPTQ W4A16
−3.7 EM points, F=7 C-fused −5.8, NVFP4 −7.5, MXFP4 −15.2. CoQA does not follow perplexity across
kinds of error: F=7 C-fused raises perplexity more than NVFP4 (+39% against +25%) yet loses fewer
CoQA points.

`hopper_fp8_w8a8` run again at `b68addd` on the same GPU model reproduced every metric and standard
error above exactly; its wall time fell from 273 to 169 minutes with the kernel changes between
the two commits (other jobs shared the host during both runs).

## Reproduce

```bash
tricast run configs/e2e/qwen3_0.6b_ppl_a.yaml      # references and accumulators
tricast run configs/e2e/qwen3_0.6b_ppl_b.yaml      # block-scaled formats, transforms, GPTQ
tricast run configs/e2e/qwen3_0.6b_ppl_c.yaml      # FP8 operands with exact accumulation
tricast run configs/e2e/qwen3_0.6b_ppl_d.yaml      # 2:4 sparsity and NVFP4 outliers
tricast report --model Qwen/Qwen3-0.6B --recipe hopper_fp8_w8a8 --dataset wikitext2 --samples 2 --seqlen 512
tricast run configs/e2e/qwen3_0.6b_kv.yaml         # KIVI, full test set (fakequant)
tricast run configs/e2e/qwen3_0.6b_kv_stream.yaml  # KIVI, first 8 windows through the cache
tricast run configs/e2e/qwen3_0.6b_kv_lmeval.yaml  # KIVI, CoQA through the cache
tricast run configs/e2e/qwen3_0.6b_lmeval_a.yaml   # lm-eval: Hopper FP8, FP8 F=7 fused
tricast run configs/e2e/qwen3_0.6b_lmeval_b.yaml   # lm-eval: MXFP4, NVFP4, GPTQ W4A16
python scripts/e2e/summarize.py runs/e2e/qwen3_0.6b_ppl_a runs/e2e/qwen3_0.6b_ppl_b
python scripts/e2e/check_kv_paths.py 14 2048       # fakequant vs cache attention, same K/V (layer, tokens)
```
