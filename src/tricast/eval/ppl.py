"""Token-weighted perplexity on non-overlapping GPTQ-style windows."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F


def _model_logits(
    model: torch.nn.Module, input_ids: torch.Tensor, *, streaming: bool = False,
) -> torch.Tensor:
    """Use token-wise teacher forcing when evaluating a streaming KV-cache recipe."""
    patch = getattr(model, "_tricast_kv_patch", None)
    cache_mode = patch is not None and patch.kv.mode == "cache"
    if not streaming and not cache_mode:
        return model(input_ids=input_ids, use_cache=False).logits
    if cache_mode:
        from ..kv import make_cache

        cache = make_cache(model)
    else:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
    # A single full-window prefill attends before cache quantization and would
    # silently measure the baseline instead of the configured cache arithmetic.
    logits = []
    for index in range(input_ids.shape[1]):
        output = model(input_ids=input_ids[:, index:index + 1], past_key_values=cache, use_cache=True)
        logits.append(output.logits)
    return torch.cat(logits, dim=1)


def _dataset_texts(dataset: str, split: str) -> tuple[list[str], str | None]:
    from datasets import load_dataset

    if dataset == "wikitext2":
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset == "c4":
        split = "validation" if split == "test" else split
        if split not in {"train", "validation"}:
            raise ValueError("c4 split must be train, validation, or test (validation alias)")
        shards = "01024" if split == "train" else "00008"
        data = load_dataset(
            "allenai/c4", "en", data_files={split: f"en/c4-{split}.00000-of-{shards}.json.gz"},
            split=split, streaming=True,
        )
        return [row["text"] for row in islice(data, 1100)], getattr(data, "_fingerprint", None)
    elif dataset == "pile":
        data = load_dataset("NeelNanda/pile-10k", split=split)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; use wikitext2, c4, or pile")
    return list(data["text"]), getattr(data, "_fingerprint", None)


def perplexity(
    model,
    tokenizer,
    *,
    dataset: str = "wikitext2",
    split: str = "test",
    seqlen: int = 2048,
    max_windows: int | None = None,
    batch_size: int = 1,
    device=None,
    texts: Iterable[str] | None = None,
    streaming: bool = False,
) -> dict:
    """Join and tokenize once; omit incomplete windows and each window's first target.

    Built-in C4 follows GPTQ ``get_c4_new``: the first 1100 documents of validation shard 0,
    joined with a single space, then the first 256 * ``seqlen`` tokens. ``test``
    aliases ``validation``; an explicit ``train`` split uses the same bounds but
    is not the validation benchmark. ``max_windows`` can only reduce this limit.
    Other datasets and explicit ``texts`` retain the double-newline join and
    evaluate all complete windows unless ``max_windows`` is set.

    ``streaming`` feeds one token at a time through a KV cache. Cache-mode KV recipes always
    stream (a full-window prefill would never read the quantized cache); set it on the baseline
    run too, so both sides of a KV comparison use the same forward path.
    """
    if seqlen < 2 or batch_size < 1 or (max_windows is not None and max_windows < 1):
        raise ValueError("seqlen must be >= 2; batch_size and max_windows must be positive")
    source_fingerprint = None
    c4_convention = texts is None and dataset == "c4"
    if texts is None:
        texts, source_fingerprint = _dataset_texts(dataset, split)
    elif isinstance(texts, str):
        texts = [texts]
    joined = (" " if c4_convention else "\n\n").join(texts)
    fingerprint = source_fingerprint or hashlib.sha256(joined.encode("utf-8")).hexdigest()
    encoded = tokenizer(joined, return_tensors="pt")
    ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long).reshape(-1)
    n_windows = ids.numel() // seqlen
    if c4_convention:
        n_windows = min(n_windows, 256)
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    if not n_windows:
        raise ValueError(f"need at least {seqlen} tokens for one complete evaluation window")
    windows = ids[:n_windows * seqlen].reshape(n_windows, seqlen)
    if device is None:
        device = next(model.parameters()).device
    patch = getattr(model, "_tricast_kv_patch", None)
    streaming = streaming or (patch is not None and patch.kv.mode == "cache")
    modes = [(module, module.training) for module in model.modules()]
    nll_sum = 0.0
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, n_windows, batch_size):
                batch = windows[start:start + batch_size].to(device)
                logits = _model_logits(model, batch, streaming=streaming)[:, :-1]
                loss = F.cross_entropy(
                    logits.double().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum"
                )
                if not torch.isfinite(loss):
                    raise ValueError("perplexity encountered non-finite negative log likelihood")
                nll_sum += loss.item()
    finally:
        for module, training in modes:
            module.training = training
    n_tokens = n_windows * (seqlen - 1)
    nll = nll_sum / n_tokens
    return {
        "ppl": math.exp(nll),
        "nll": nll,
        "n_tokens": n_tokens,
        "n_windows": n_windows,
        "dataset_fingerprint": fingerprint,
        "forward_mode": "streaming_cache" if streaming else "full_window",
    }
