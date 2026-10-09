"""Same Q/K/V into both KV paths: fakequant selectors vs token-by-token TriCastKVCache (fp32).

Captures one Qwen3-0.6B layer's post-RoPE Q/K/V from a native forward over the first WikiText-2
window, then computes attention (a) with the fakequant (query, key) selectors and (b) by streaming
the same K/V one token at a time through TriCastKVCache. With identical inputs the two must agree
to fp32 summation order; a larger gap would be a semantic mismatch, not rounding.
"""
import sys

import torch
import transformers.models.qwen3.modeling_qwen3 as mq
from transformers import AutoModelForCausalLM, AutoTokenizer

from tricast.eval.ppl import _dataset_texts
from tricast.kv.attention import _flush_mask
from tricast.kv.cache import TriCastKVCache, quantize_states
from tricast.quant.spec import get_kv_spec

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32,
                                             attn_implementation="eager").cuda().eval()
texts, _ = _dataset_texts("wikitext2", "test")
ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[:, :tokens].cuda()
captured = {}
original = mq.eager_attention_forward


def capture(module, query, key, value, attention_mask, **kwargs):
    if module.layer_idx == layer and not captured:
        captured.update(q=query.clone(), k=key.clone(), v=value.clone(), scaling=kwargs.get("scaling"))
    return original(module, query, key, value, attention_mask, **kwargs)


mq.eager_attention_forward = capture
with torch.inference_mode():
    model(ids)
mq.eager_attention_forward = original
q, k, v, scaling = captured["q"], captured["k"], captured["v"], captured["scaling"]
repeats = q.shape[1] // k.shape[1]
for name in ("kivi2", "kivi4"):
    kv = get_kv_spec(name)
    with torch.inference_mode():
        # (a) fakequant: quantize all tokens once, select per (query, key)
        positions = torch.arange(tokens, device=q.device)[None]
        key_mask = _flush_mask(positions, positions, kv.key, kv.key_axis, kv.residual)
        value_mask = _flush_mask(positions, positions, kv.value, kv.value_axis, kv.residual)
        kq = quantize_states(k, kv.key, kv.key_axis)
        vq = quantize_states(v, kv.value, kv.value_axis)
        kq, kr, vq, vr = (t.repeat_interleave(repeats, dim=1) for t in (kq, k, vq, v))
        scores = torch.where(key_mask, q @ kq.transpose(-1, -2), q @ kr.transpose(-1, -2)) * scaling
        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).tril()
        weights = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
        out_a = weights.masked_fill(~value_mask, 0) @ vq + weights.masked_fill(value_mask, 0) @ vr
        # (b) streaming: the cache returns the pre-write state plus the appended token
        cache = TriCastKVCache(kv, num_hidden_layers=1)
        rows = []
        for t in range(tokens):
            kt, vt = cache.update(k[:, :, t:t + 1], v[:, :, t:t + 1], 0)
            kt, vt = kt.repeat_interleave(repeats, dim=1), vt.repeat_interleave(repeats, dim=1)
            w = torch.softmax((q[:, :, t:t + 1] @ kt.transpose(-1, -2)) * scaling, dim=-1)
            rows.append(w @ vt)
        out_b = torch.cat(rows, dim=2)
    diff = (out_a - out_b).abs()
    print(f"layer {layer} {name}: max|a-b| {diff.max().item():.3e}  max|a| {out_a.abs().max().item():.3e}  "
          f"relative {(diff.max() / out_a.abs().max()).item():.3e}", flush=True)
print("KV_PATH_CHECK_DONE", flush=True)
