# Qwen3-0.6B under emulated number formats and accumulators

Perplexity and lm-eval numbers come from `tricast run` records (`configs/e2e/*.yaml`), joined with
`scripts/e2e/summarize.py`; the per-layer error figures come from `tricast report`, and the KV path
comparison from `scripts/e2e/check_kv_paths.py`. One run per row: the emulation is bit-exact and
deterministic, so a rerun reproduces the same perplexity to the last bit (`mxfp8_w_a` run again at a
later commit gave 21.294279307019544 both times). `native` is the unpatched Hugging Face model.

## Perplexity — WikiText-2 test, A100

146 windows of 2048 tokens (298,862 scored tokens, GPTQ convention), dataset fingerprint
`a46124b21ac53738`. Model `Qwen/Qwen3-0.6B@c1899de289a0` in bf16; NVIDIA A100-SXM4-80GB, CUDA 12.8,
torch 2.8.0+cu128, triton 3.4.0, transformers 4.55.2; TriCast `49a6a76` (clean tree), except the
exact-accumulation row `fp8_w8a8_fp64acc`, run at `032684b`.

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
| `nvfp4_4o6` | NVFP4, each block scaled so its amax maps to 4 or 6 (Four-over-Six) | GDFS G=6 F=35 | 25.5801 | +22.01% |
| `nvfp4_smoothquant` | NVFP4 after SmoothQuant (α = 0.5) | GDFS G=6 F=35 | 25.7532 | +22.83% |
| `nvfp4_awq_shared` | NVFP4 after AWQ scaling, one scale per group of projections sharing an input | GDFS G=6 F=35 | 24.3041 | +15.92% |
| `mixed_first_last_bf16` | MXFP4 in decoder blocks 1–26; blocks 0 and 27 unquantized | GDFS G=6 F=35 | 29.6883 | +41.60% |
| `w4a16_g128_zp_gptq` | GPTQ UINT4 weights, groups of 128 with zero points; bf16 activations | CoFDA F=23, chunks of 32 | 24.7228 | +17.92% |
| `w4a16_gptq_sequential` | the same weight format, GPTQ fitted block by block on the quantized model's outputs | IEEE fp32 FMA chain | 24.4976 | +16.84% |

SmoothQuant, AWQ and the two GPTQ rows calibrate on 128 windows of 2048 tokens from WikiText-2
train (seed 42 for `nvfp4_smoothquant` and `w4a16_g128_zp_gptq`, 0 for the others, as their recipes
set). Keeping the first and last of the 28 decoder blocks unquantized reduces MXFP4's increase from
+61.9% to +41.6%. The two GPTQ rows differ in both calibration order and accumulator, so their gap
cannot be assigned to either.

The rotation makes MXFP4 worse here although it makes each operand easier to quantize.
`tricast report` on 2 × 512 WikiText-2 tokens (V100) shows why the operand metrics mislead: with the
rotation the activation SQNR rises in all 196 linear layers (median 17.2 → 18.8 dB) and the weight
SQNR barely moves (18.7 → 18.7 dB), but the SQNR of the layer outputs does not improve (median
16.4 → 16.3 dB, lower in 112 of 196 layers) and the logits KL divergence grows from 0.57 to 0.82. The
rotation itself preserves `x·Wᵀ` exactly (fp64 test in `tests/test_nn_patch.py`); the cause of the
output error is not established.

## KIVI KV cache — WikiText-2 test, V100

Linear layers stay native, so only the KV cache changes. NVIDIA V100-PCIE-32GB, CUDA 12.8, torch
2.8.0+cu128, triton 3.4.0, transformers 4.55.2, lm_eval 0.4.13; TriCast `5c09f47` (clean tree).
The V100 native perplexity (20.9651) differs from the A100 one in the third decimal: attention and
normalization run natively and their kernels differ between the two GPUs.

`fakequant` scores every token as a decode step reading the quantized cache (ENGINE §3.13). Keys are
2- or 4-bit per channel in groups of 32 tokens, quantized once a block of 128 tokens is complete
(the remainder stays in full precision); values are quantized per token, except the latest 128.

| run | all 146 windows | vs native | first 8 windows, streamed through the cache | vs native |
| --- | ---: | ---: | ---: | ---: |
| native | 20.9651 | | 17.8377 | |
| KIVI-4 | 21.0041 | +0.19% | 17.8724 | +0.19% |
| KIVI-2 | 23.0081 | +9.74% | 19.6035 | +9.90% |

The one-forward (`fakequant`) and token-by-token (`cache`) paths compute the same attention given the
same K/V: `scripts/e2e/check_kv_paths.py` finds at most 6e-6 relative difference on Qwen3-0.6B layers
0 and 14 (fp32, 2048 tokens). Whole-model perplexities of the two paths still differ slightly: for
KIVI-2 by 0.10% (bf16, the same 8 windows) and 0.16% (fp32, 2 windows), against 0.03% and 1e-7 for
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
| KIVI-4 | 0.5762 ± 0.0194 | 0.7037 ± 0.0166 |
| KIVI-2 | 0.5713 ± 0.0195 | 0.6950 ± 0.0169 |

KIVI-2 raises perplexity by 9.7% but lowers CoQA EM by only 0.6 points, within one standard error.
The two evaluations quantize different things: perplexity scores every position against quantized
keys and values, while in CoQA only the few generated answer tokens read the quantized cache, and
the most recent values (128 tokens) and keys (the residual under 128) stay in full precision. Which
of these accounts for the gap was not measured.

## Reproduce

```bash
tricast run configs/e2e/qwen3_0.6b_ppl_a.yaml      # references and accumulators
tricast run configs/e2e/qwen3_0.6b_ppl_b.yaml      # block-scaled formats, transforms, GPTQ
tricast run configs/e2e/qwen3_0.6b_ppl_c.yaml      # FP8 operands with exact accumulation
tricast run configs/e2e/qwen3_0.6b_kv.yaml         # KIVI, full test set (fakequant)
tricast run configs/e2e/qwen3_0.6b_kv_stream.yaml  # KIVI, first 8 windows through the cache
tricast run configs/e2e/qwen3_0.6b_kv_lmeval.yaml  # KIVI, CoQA through the cache
python scripts/e2e/summarize.py runs/e2e/qwen3_0.6b_ppl_a runs/e2e/qwen3_0.6b_ppl_b
python scripts/e2e/check_kv_paths.py 14 2048       # fakequant vs cache attention, same K/V (layer, tokens)
```
