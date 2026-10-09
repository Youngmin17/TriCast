"""LLM text generation: a baseline (native, or the same format with FP64 accumulation) against a
TriCast-emulated design.

Both sides decode greedily (seed 42). For the per-position comparison the emulated model is then decoded
again with every step forced to the baseline's token: the same prompt prefill and one token per step through
the KV cache as generation itself, so position t sees only the prompt and the first t baseline tokens, and
an activation scale never spans tokens that a real decode step would not see. The baseline side of that
comparison is the baseline generation's own step logits.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .. import metrics
from ..catalog import MODELS
from .common import (
    SEED,
    TOP_K,
    BaselineCache,
    Progress,
    emulated,
    llm_dtype,
    now,
    recipe_key,
    recipe_label,
    report,
    require_finite,
    resolve_backend,
)

_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}


@dataclass(frozen=True)
class _Baseline:
    """What a later run with the same baseline needs: the continuation and each step's raw logits
    (float32, CPU). For at most 64 new tokens one entry is at most 64 × 151,936 × 4 B ≈ 39 MB (Qwen3-0.6B;
    Llama-3.2-1B ≈ 33 MB), so the 4-entry cache stays under 160 MB."""

    ids: list[int]
    logits: torch.Tensor
    seconds: float


BASELINES = BaselineCache()


def load_llm(model_id: str, device: str, revision: str | None) -> tuple[Any, Any]:
    """The model (bf16 on CUDA, fp32 elsewhere; eval; eager attention) and tokenizer at a pinned
    revision, cached per (model, device). ``HF_HUB_OFFLINE=1`` restricts loading to the local cache."""
    key = (model_id, str(device))
    if key not in _MODELS:
        from huggingface_hub import constants
        from transformers import AutoModelForCausalLM, AutoTokenizer

        offline = constants.HF_HUB_OFFLINE
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=offline)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=llm_dtype(device), attn_implementation="eager",
            local_files_only=offline,
        ).to(device).eval()
        _MODELS[key] = (model, tokenizer)
    return _MODELS[key]


def encode_prompt(tokenizer: Any, prompt: str, chat: bool) -> torch.Tensor:
    """Chat models get their chat template with thinking off (as examples/demo_qwen3.py); base models
    continue the raw prompt."""
    if not chat:
        return tokenizer(prompt, return_tensors="pt").input_ids
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def token_pieces(decode: Callable[[list[int]], str], ids: Sequence[int]) -> list[str]:
    """One piece per token, joining to exactly ``decode(ids)``.

    A token's piece is the text that becomes stable once it is decoded (growth of the common prefix of
    the decoded prefix and the full text), so a token ending inside a multibyte character gets ``""``
    and the character goes to the token that completes it."""
    text = decode(list(ids))
    pieces, done = [], 0
    for end in range(1, len(ids) + 1):
        reach = max(done, len(os.path.commonprefix([decode(list(ids[:end])), text])))
        pieces.append(text[done:reach])
        done = reach
    return pieces


class _ForceTokens:
    """Logits processor for greedy ``generate``: step i emits ``tokens[i]`` (every other token gets -inf).
    ``generate`` records each step's raw logits before processors run."""

    def __init__(self, tokens: list[int], prompt_length: int) -> None:
        self._tokens, self._start = tokens, prompt_length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        forced = torch.full_like(scores, -math.inf)
        forced[:, self._tokens[input_ids.shape[1] - self._start]] = 0
        return forced


def _generate(model: nn.Module, tokenizer: Any, prompt_ids: torch.Tensor, max_new_tokens: int,
              forced: list[int] | None = None) -> tuple[list[int], torch.Tensor]:
    """Greedy decoding: the new token ids and each step's raw logits ``[steps, vocab]`` (float32, CPU).

    With ``forced`` every step emits the next forced token instead of its argmax, through the same
    ``generate`` call (prompt prefill, then one token per step with the KV cache, batch 1)."""
    from transformers import LogitsProcessorList

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    processors = LogitsProcessorList([] if forced is None else [_ForceTokens(forced, prompt_ids.shape[1])])
    torch.manual_seed(SEED)
    with torch.no_grad():
        output = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, pad_token_id=pad,
                                logits_processor=processors, output_logits=True, return_dict_in_generate=True)
    ids = output.sequences[0, prompt_ids.shape[1]:].tolist()
    if forced is not None and ids != forced:
        raise RuntimeError("강제 디코딩이 baseline 토큰열을 따르지 않았습니다.")
    return ids, require_finite(torch.stack(output.logits, dim=1)[0].float().cpu(), "생성 logits")


def _output(label: str, tokenizer: Any, ids: list[int], logits: torch.Tensor) -> dict:
    """Output of one side: text, and per token its piece, log-probability and top-5 (ties: lower id)."""
    def decode(part: list[int]) -> str:
        return tokenizer.decode(part, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    logprobs = torch.log_softmax(logits.double(), dim=-1)
    values, indices = torch.sort(logprobs, dim=-1, descending=True, stable=True)
    tokens = []
    for position, (token, piece) in enumerate(zip(ids, token_pieces(decode, ids), strict=True)):
        top = [{"piece": tokenizer.decode([index], clean_up_tokenization_spaces=False), "p": math.exp(value)}
               for value, index in zip(values[position, :TOP_K].tolist(), indices[position, :TOP_K].tolist(),
                                       strict=True)]
        tokens.append({"id": token, "piece": piece, "logprob": logprobs[position, token].item(), "top": top})
    return {"label": label, "text": decode(ids), "tokens": tokens}


def run_generate(model_id: str, recipe: Any, baseline_recipe: Any | None, prompt: str, max_new_tokens: int,
                 device: str, backend: str | None = None, progress: Progress | None = None,
                 model: nn.Module | None = None, tokenizer: Any = None) -> dict:
    """Compare greedy generation of ``model_id`` under ``baseline_recipe`` (``None`` = native) and
    ``recipe``. ``model`` and ``tokenizer`` replace the pinned checkpoint (tests). ``progress(stage,
    fraction, partial=None)`` receives the baseline Output as soon as it exists. The baseline is
    reused from memory for the same model, prompt, token budget and baseline recipe."""
    backend = resolve_backend(device, backend)
    report(progress, "load", 0.0)
    if model is None:
        model, tokenizer = load_llm(model_id, device, MODELS[model_id].revision)
    prompt_ids = encode_prompt(tokenizer, prompt, MODELS[model_id].chat).to(device)
    key = (model_id, prompt, max_new_tokens, recipe_key(baseline_recipe))
    baseline = BASELINES.get(model, key)
    cached = baseline is not None
    if not cached:
        report(progress, "baseline", 0.05)
        start = now(device)
        with nullcontext() if baseline_recipe is None else emulated(model, baseline_recipe, backend):
            ids, logits = _generate(model, tokenizer, prompt_ids, max_new_tokens)
        baseline = _Baseline(ids, logits, now(device) - start)
        BASELINES.put(model, key, baseline)
    baseline_output = _output(recipe_label(baseline_recipe), tokenizer, baseline.ids, baseline.logits)
    report(progress, "baseline", 0.4, partial={"baseline": baseline_output})
    report(progress, "patch", 0.42)
    start = now(device)
    with emulated(model, recipe, backend) as evidence:
        report(progress, "emulated", 0.45)
        ids, logits = _generate(model, tokenizer, prompt_ids, max_new_tokens)
        report(progress, "emulated", 0.7)
        _, forced = _generate(model, tokenizer, prompt_ids, len(baseline.ids), forced=baseline.ids)
    emulated_s = now(device) - start
    report(progress, "metrics", 0.95)
    divergence = metrics.first_divergence(baseline.ids, ids)
    return {
        "baseline": baseline_output,
        "emulated": _output(recipe_label(recipe), tokenizer, ids, logits),
        "metrics": {"first_divergence": divergence,
                    "prefix_match": len(baseline.ids) if divergence is None else divergence,
                    "teacher_forced": metrics.teacher_forced(baseline.logits, forced)},
        "timing": {"baseline_s": baseline.seconds, "emulated_s": emulated_s},
        "evidence": evidence,
        "cached_baseline": cached,
    }
