# Operand pack sources

`qwen3-0.6b.l0.q_proj.*`: the input activation and weight of `model.layers.0.self_attn.q_proj` in Qwen/Qwen3-0.6B
(revision `c1899de289a04d12100db370d81485cdf75e47ca`), Apache License 2.0 (https://huggingface.co/Qwen/Qwen3-0.6B), for the prompt
"The history of the printing press begins", quantized with TriCast `fp8_tensor` to FP8 E4M3 codes. The `*.u32.bin` files are
TriCast's own GEMM outputs (fp32 bits) for each MMA setting; `qwen3-0.6b.l0.q_proj.json` holds the shapes,
scales and the recording environment.
