"""Attach KV emulation to Hugging Face's functional attention interface."""

from __future__ import annotations

import re
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from inspect import signature
from types import MethodType
from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..formats import FP32
from ..quant.spec import KVAxis, KVSpec, QuantSpec, get_kv_spec
from .cache import TriCastKVCache, quantize_states

_MISSING = object()


@dataclass
class _Patch:
    kv: KVSpec
    indices: set[int]
    names: list[str]
    configs: list[tuple[nn.Module, Any]]
    prepare_cache: Any = _MISSING
    forward_hook: Any = None
    clear_hook: Any = None
    token_mask: torch.Tensor | None = None
    cache: TriCastKVCache | None = None
    query_positions: torch.Tensor | None = None


def _token_mask(attention_mask: torch.Tensor | None, key: torch.Tensor) -> torch.Tensor | None:
    """Recover token visibility from HF's binary or additive attention masks."""
    if attention_mask is None:
        return None
    mask = attention_mask[..., :key.shape[-2]]
    if mask.ndim == 2:
        if not ((mask == 0) | (mask == 1)).all():
            raise NotImplementedError("KV quantization requires a binary 2D attention_mask")
        if mask.shape[-1] < key.shape[-2]:
            # Static caches expose unused slots beyond the supplied token mask.
            mask = torch.cat((mask, mask.new_zeros(mask.shape[0], key.shape[-2] - mask.shape[-1])), dim=-1)
        visible = mask.bool()
    elif mask.ndim == 4:
        if mask.dtype == torch.bool:
            visible = mask
        elif mask.is_floating_point():
            blocked = (mask == torch.finfo(mask.dtype).min) | (mask == -torch.inf)
            if not ((mask == 0) | blocked).all():
                raise NotImplementedError("KV quantization requires a boolean or zero/blocked 4D mask")
            visible = ~blocked
        else:
            raise NotImplementedError("KV quantization requires a boolean or additive 4D mask")
        visible = visible.any(dim=1).any(dim=1)
    else:
        raise NotImplementedError("KV quantization requires a 2D or 4D attention_mask")
    if visible.shape[-1] != key.shape[-2] or visible.shape[0] not in (1, key.shape[0]):
        raise ValueError("attention_mask does not cover the KV batch and tokens")
    return visible.expand(key.shape[0], -1)


def _flush_mask(
    positions: torch.Tensor, query_positions: torch.Tensor, spec: QuantSpec | None,
    axis: KVAxis, residual: int,
) -> torch.Tensor:
    """Read the stored prefix before query t appends its own full-precision KV."""
    if spec is None:
        return torch.zeros_like(positions[:, None, None, :] < query_positions[:, None, :, None])
    boundary = (query_positions // residual * residual if axis == "channel"
                else query_positions - residual)
    return ((positions[:, None, :] < boundary[:, :, None])
            & (positions[:, None, :] >= 0))[:, None]


def _attention(
    module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: torch.Tensor | None, *args: Any, **kwargs: Any,
) -> Any:
    patch, original, implementation = module._tricast_kv_attention
    kv = patch.kv
    if kv.mode == "cache":
        # update() returns the previously stored state plus raw appended tokens.
        # Prefill therefore uses native attention; quantization affects later calls.
        return original(module, query, key, value, attention_mask, *args, **kwargs)
    # A generated sliding-window mask hides valid history, not padding.
    # Only the caller's mask may remove tokens from the scale/flush clock.
    token_mask = _token_mask(patch.token_mask, key)
    if patch.token_mask is None:
        _token_mask(attention_mask, key)  # Validate supported mask values.
    if implementation == "eager" and attention_mask is not None and attention_mask.dtype == torch.bool:
        # HF eager expects additive masks, including on the unquantized fallback.
        attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill(
            ~attention_mask, torch.finfo(query.dtype).min,
        )
    raw_key, raw_value = key, value
    key = quantize_states(key, kv.key, kv.key_axis, token_mask).to(key.dtype)
    value = quantize_states(value, kv.value, kv.value_axis, token_mask).to(value.dtype)
    length, queries = key.shape[-2], query.shape[-2]
    positions = (torch.arange(length, device=key.device).expand(key.shape[0], -1)
                 if token_mask is None else token_mask.long().cumsum(-1) - 1)
    query_indices = patch.query_positions
    if query_indices is None:
        query_indices = torch.arange(length - queries, length, device=key.device)
    query_positions = positions.index_select(-1, query_indices.to(key.device))
    key_mask = _flush_mask(positions, query_positions, kv.key, kv.key_axis, kv.residual)
    value_mask = _flush_mask(positions, query_positions, kv.value, kv.value_axis, kv.residual)
    if token_mask is not None:
        key_mask = key_mask & token_mask[:, None, None, :]
        value_mask = value_mask & token_mask[:, None, None, :]
    if not key_mask.any() and not value_mask.any():
        return original(module, query, raw_key, raw_value, attention_mask, *args, **kwargs)
    if all(spec is None or (spec.scale is None and spec.format == FP32 and spec.dequant_format == FP32)
           for spec in (kv.key, kv.value)):
        return original(module, query, key, value, attention_mask, *args, **kwargs)
    if args:
        raise NotImplementedError("query-dependent KV attention requires keyword attention options")
    if kwargs.get("head_mask") is not None or kwargs.get("softcap") is not None:
        raise NotImplementedError("query-dependent KV attention does not support head_mask or softcap")

    # A key group uses only its G completed tokens; the (query, key) selector
    # prevents those statistics from affecting a query before the group's flush.
    repeats = query.shape[1] // key.shape[1]
    key, raw_key, value, raw_value = (
        tensor.repeat_interleave(repeats, dim=1) for tensor in (key, raw_key, value, raw_value)
    )
    scaling = kwargs.get("scaling", query.shape[-1] ** -0.5)
    scores = torch.where(key_mask, query @ key.transpose(-1, -2), query @ raw_key.transpose(-1, -2))
    scores = scores * scaling
    if attention_mask is None:
        # SDPA ordinarily applies this mask internally; the explicit score path
        # needs the same bottom-right alignment for cached multi-token queries.
        causal = torch.arange(length, device=key.device)[None, :] <= query_indices.to(key.device)[:, None]
        scores = scores.masked_fill(~causal, -torch.inf)
    elif attention_mask.dtype == torch.bool:
        scores = scores.masked_fill(~attention_mask[..., :length], -torch.inf)
    else:
        scores = scores + attention_mask[..., :length]
    weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    # SDPA defines fully blocked query rows as zero; eager finite-min masks keep
    # their usual softmax behavior. Such padding rows are never scale samples.
    weights = weights.masked_fill(torch.isneginf(scores).all(dim=-1, keepdim=True), 0)
    weights = nn.functional.dropout(weights, p=kwargs.get("dropout", 0.0), training=module.training)
    output = (weights.masked_fill(~value_mask, 0) @ value
              + weights.masked_fill(value_mask, 0) @ raw_value)
    return output.transpose(1, 2).contiguous(), weights


def _layer_indices(layers: str | None) -> set[int] | None:
    if layers is None:
        return None
    indices: set[int] = set()
    for item in layers.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", item)
        if match is None:
            raise ValueError("layers must contain indices or inclusive ranges, e.g. '0-3,27'")
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if end < start:
            raise ValueError("layer ranges must be ascending")
        indices.update(range(start, end + 1))
    return indices


def _original_attention(module: nn.Module) -> Callable[..., Any]:
    implementation = module.config._attn_implementation
    source = import_module(type(module).__module__)
    registry = getattr(source, "ALL_ATTENTION_FUNCTIONS", None)
    if registry is None:
        raise ValueError(f"{type(module).__name__} does not use AttentionInterface")
    if implementation == "eager":
        return source.eager_attention_forward
    return registry[implementation]


def make_cache(model: nn.Module) -> TriCastKVCache:
    """Create a fresh cache for a patched model; pass as ``past_key_values`` to forward."""
    patch = getattr(model, "_tricast_kv_patch", None)
    if patch is None or patch.kv.mode != "cache":
        raise ValueError("apply_kv(model, KVSpec(..., mode='cache')) before make_cache")
    count = model.config.get_text_config(decoder=True).num_hidden_layers
    return TriCastKVCache(patch.kv, layers=patch.indices, num_hidden_layers=count)


def apply_kv(model: nn.Module, kv: KVSpec, *, layers: str | None = None) -> list[str]:
    """Patch Llama/Qwen3-style attention and return the selected module names.

    Fakequant runs after RoPE/cache lookup. Only attention modules receive copied
    configs: the model's original implementation still builds causal/padding masks
    in Transformers 4.55 (unregistered mask implementations would skip masking).

    Cache mode hooks 4.55's ``_prepare_cache_for_generation``: its explicit-Cache
    early return accepts ``past_key_values`` without creating a DynamicCache.
    Each generation gets a fresh cache; explicit caches and ``use_cache=False``
    retain their normal HF behavior. Direct forward calls use ``make_cache``.
    Padded cache inputs are rejected before update: streaming writes do not carry
    a token mask. Fakequant groups valid tokens and reproduces the same per-query
    flush schedule as token-by-token cache mode, including the residual length.
    Assisted and contrastive generation are rejected before the first forward.
    Fakequant uses explicit mixed scores/PV for query-dependent KV values.
    Cache mode preserves native eager/SDPA attention over pre-write KV values;
    other attention implementations are unsupported rather than silently replaced.
    """
    kv = get_kv_spec(kv)
    indices = _layer_indices(layers)
    candidates = [(name, module) for name, module in model.named_modules()
                  if isinstance(getattr(module, "layer_idx", None), int)
                  and hasattr(module, "config")
                  and all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))]
    if not candidates:
        raise ValueError("no supported Llama/Qwen3-style attention modules found")
    available = {module.layer_idx for _, module in candidates}
    if indices is not None and not indices <= available:
        raise ValueError(f"unknown attention layer indices: {sorted(indices - available)}")
    selected = [(name, module) for name, module in candidates
                if indices is None or module.layer_idx in indices]
    if kv.mode == "cache" and not hasattr(model, "_prepare_cache_for_generation"):
        raise ValueError("cache mode needs a Hugging Face generation model")
    implementations = [module.config._attn_implementation for _, module in selected]
    existing = getattr(model, "_tricast_kv_patch", None)
    if existing is not None:
        originals_by_id = {id(module): config._attn_implementation for module, config in existing.configs}
        implementations = [originals_by_id.get(id(module), implementation)
                           for (_, module), implementation in zip(selected, implementations, strict=True)]
    if any(implementation not in ("eager", "sdpa") for implementation in implementations):
        raise NotImplementedError("TriCast KV emulation supports only eager and sdpa attention")
    remove_kv(model)
    originals = [_original_attention(module) for _, module in selected]
    patch = _Patch(kv, {module.layer_idx for _, module in selected},
                   [name for name, _ in selected], [])
    ALL_ATTENTION_FUNCTIONS.register("tricast_kv", _attention)
    for (_, module), original, implementation in zip(selected, originals, implementations, strict=True):
        patch.configs.append((module, module.config))
        if kv.mode == "fakequant":
            module.config = copy(module.config)
            module.config._attn_implementation = "tricast_kv"
        module._tricast_kv_attention = (patch, original, implementation)
    if kv.mode == "cache":
        original = model._prepare_cache_for_generation
        patch.prepare_cache = model.__dict__.get("_prepare_cache_for_generation", _MISSING)

        @wraps(original)
        def prepare_cache(
            self: nn.Module, generation_config: Any, model_kwargs: dict[str, Any],
            *args: Any, **kwargs: Any,
        ) -> Any:
            if generation_config.use_cache and model_kwargs.get("past_key_values") is None:
                if generation_config.cache_implementation is not None:
                    raise ValueError("TriCast KV cache cannot be combined with cache_implementation")
                model_kwargs["past_key_values"] = make_cache(self)
            assistant = args[0] if args else kwargs.get("assistant_model")
            if isinstance(model_kwargs.get("past_key_values"), TriCastKVCache):
                mode = generation_config.get_generation_mode(assistant)
                if mode == "assisted_generation":
                    raise NotImplementedError(
                        "speculative/assisted generation cannot roll back a quantized KV cache"
                    )
                if mode == "contrastive_search":
                    raise NotImplementedError("contrastive search is unsupported with a TriCast KV cache")
            return original(generation_config, model_kwargs, *args, **kwargs)

        model._prepare_cache_for_generation = MethodType(prepare_cache, model)
    forward_signature = signature(model.forward)

    def prepare_mask(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any],
    ) -> None:
        arguments = forward_signature.bind_partial(*args, **kwargs).arguments
        mask = arguments.get("attention_mask")
        patch.query_positions = arguments.get("cache_position")
        if patch.query_positions is None:
            inputs = arguments.get("input_ids")
            if inputs is None:
                inputs = arguments.get("inputs_embeds")
            if inputs is not None:
                past = arguments.get("past_key_values")
                start = past.get_seq_length() if past is not None else 0
                patch.query_positions = torch.arange(start, start + inputs.shape[1], device=inputs.device)
        if kv.mode == "fakequant":
            if mask is not None and mask.ndim == 4 and patch.query_positions is not None:
                # HF accepts caller-supplied noncausal masks unchanged. Refuse
                # those rather than silently violating the causal KV contract.
                visible = mask if mask.dtype == torch.bool else mask == 0
                future = (torch.arange(mask.shape[-1], device=mask.device)[None, :]
                          > patch.query_positions.to(mask.device)[:, None])
                if (visible & future).any():
                    raise NotImplementedError("fakequant requires a causal 4D mask")
            # Older SDPA backends unmask fully masked query rows. Preserve the
            # original padding mask rather than inferring validity from those rows.
            patch.token_mask = mask
        else:
            cache = arguments.get("past_key_values")
            patch.cache = cache if isinstance(cache, TriCastKVCache) else None
        if patch.cache is not None:
            if mask is not None and (mask.ndim != 2 or not (mask == 1).all()):
                raise NotImplementedError(
                    "TriCast quantized KV cache does not support padding or non-2D attention masks; "
                    "use unpadded inputs or mode='fakequant'"
                )
            for module, config in patch.configs:
                module.config = copy(config)
                module.config._attn_implementation = "tricast_kv"

    patch.forward_hook = model.register_forward_pre_hook(prepare_mask, with_kwargs=True)
    def clear_state(_module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        patch.token_mask = None
        patch.query_positions = None
        if patch.cache is not None:
            patch.cache.clear_attention_states()
            patch.cache = None
        if kv.mode == "cache":
            for module, config in patch.configs:
                module.config = config

    patch.clear_hook = model.register_forward_hook(clear_state, always_call=True)
    model._tricast_kv_patch = patch
    return patch.names.copy()


def remove_kv(model: nn.Module) -> None:
    """Restore configs and the generation cache hook; existing cache objects survive."""
    patch = getattr(model, "_tricast_kv_patch", None)
    if patch is None:
        return
    for module, config in patch.configs:
        module.config = config
        del module._tricast_kv_attention
    patch.forward_hook.remove()
    if patch.clear_hook is not None:
        patch.clear_hook.remove()
    if patch.kv.mode == "cache":
        if patch.prepare_cache is _MISSING:
            del model._prepare_cache_for_generation
        else:
            model._prepare_cache_for_generation = patch.prepare_cache
    del model._tricast_kv_patch
