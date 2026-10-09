"""SPEC AC4, CPU-side check: bf16_passthrough keeps perplexity within 0.1% of the native model.

The GPU golden case (`e2e_bf16_passthrough_ppl`) judges the Triton path; this test judges the reference
backend on a tiny model, so a regression in the reference path shows up wherever the CPU suite runs. The
full-model AC4 measurement is the recorded Llama/Qwen WikiText-2 run. A randomly initialised tiny model
predicts almost uniformly, which hides quantization error, so its matrices are scaled up until a lossy
recipe moves perplexity past the limit; the second test keeps that discrimination honest.
"""

from __future__ import annotations

import pytest
import torch

from tricast.eval.ppl import perplexity
from tricast.nn.patch import iter_emulinear, patch_model, unpatch_model
from tricast.recipe import load_recipe

LIMIT = 1e-3  # SPEC AC4: abs(PPL_emu - PPL_base) / PPL_base <= 0.1%
WEIGHT_SCALE = 8.0  # sharpens the predictions; at 1.0 even fp8_f7_lowacc stays within LIMIT
TEXT = "t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15"


def _tiny_model() -> tuple[torch.nn.Module, object]:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    torch.manual_seed(42)
    config = transformers.LlamaConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=32, vocab_size=32, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_dropout=0.0,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 2:
                parameter.mul_(WEIGHT_SCALE)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i + 4 for i in range(28)})
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]",
        unk_token="[UNK]", model_max_length=32,
    )
    return model, tokenizer


def _relative_ppl_change(recipe: str) -> float:
    model, tokenizer = _tiny_model()
    kwargs = {"texts": [TEXT], "seqlen": 8, "device": "cpu"}
    baseline = perplexity(model, tokenizer, **kwargs)
    report = patch_model(model, load_recipe(recipe), backend="reference")
    layers = dict(iter_emulinear(model))
    assert report.patched and layers
    fired: set[str] = set()
    handles = [layer.register_forward_hook(lambda module, args, out, name=name: fired.add(name))
               for name, layer in layers.items()]
    try:
        emulated = perplexity(model, tokenizer, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        unpatch_model(model)
    assert fired == set(layers), "every patched layer must execute"
    assert emulated["dataset_fingerprint"] == baseline["dataset_fingerprint"]
    return abs(emulated["ppl"] - baseline["ppl"]) / baseline["ppl"]


def test_bf16_passthrough_keeps_ppl_within_ac4_limit() -> None:
    relative = _relative_ppl_change("bf16_passthrough")
    assert relative <= LIMIT, relative


def test_ac4_limit_rejects_a_lossy_recipe() -> None:
    """The same judgement must fail an accumulator that drops bits, or the first test proves nothing."""
    relative = _relative_ppl_change("fp8_f7_lowacc")
    assert relative > LIMIT, relative
