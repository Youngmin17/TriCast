"""Call a transformers>=4.48 decoder (Llama, Qwen2/Qwen3 family) the way the 2023 official scripts do.

The scripts (wanda, SparseGPT) capture the first decoder layer's inputs with a ``Catcher`` swapped into
``model.model.layers[0]`` and then run each layer as ``layer(h, attention_mask=m, position_ids=p)[0]``.
Since transformers 4.48 a decoder layer needs ``position_embeddings=(cos, sin)`` from the model's rotary
embedding, returns a tensor instead of a tuple, and under SDPA the model may pass ``attention_mask=None``.
Only the model side is adapted here; the official code runs unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from torch import nn

_MISSING = object()


class _OfficialLayerList(nn.ModuleList):
    """``model.model.layers`` while official code runs. Qwen2/Qwen3 models read ``attention_type`` from
    every list entry, so a module swapped in for a layer (the official ``Catcher``) inherits it."""

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        kind = getattr(self[idx], "attention_type", None)
        if kind is not None and getattr(module, "attention_type", None) is None:
            module.attention_type = kind
        super().__setitem__(idx, module)


def _official_call(forward: Callable[..., Any], rotary: nn.Module) -> Callable[..., Any]:
    """``layer(h, attention_mask=m, position_ids=p)`` returns ``(output,)`` with ``position_embeddings``
    computed from ``position_ids`` exactly as the model computes them. Calls that carry
    ``position_embeddings`` come from the model's own forward and pass through unchanged."""

    def official_forward(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        if args or "position_embeddings" in kwargs:
            return forward(hidden_states, *args, **kwargs)
        if kwargs.get("position_ids") is None:
            raise ValueError("official-style decoder-layer call needs position_ids for the rotary embedding")
        kwargs["position_embeddings"] = rotary(hidden_states, kwargs["position_ids"])
        return (forward(hidden_states, **kwargs),)

    return official_forward


@contextmanager
def official_compat(model: nn.Module, seqlen: int) -> Iterator[list[str]]:
    """Adapt ``model`` for official 2023 code; yields the adaptations applied; restores everything on exit.

    Forces eager attention (a real 4-D mask tensor reaches the Catcher), wraps every decoder layer's
    forward (``_official_call``), sets ``model.seqlen`` and, when accelerate did not dispatch the model,
    ``model.hf_device_map = {}`` (the official code reads it unconditionally).
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    rotary = getattr(inner, "rotary_emb", None)
    if type(layers) is not nn.ModuleList or not isinstance(rotary, nn.Module):
        raise ValueError("official code needs model.model.layers (a ModuleList) and model.model.rotary_emb "
                         "(transformers>=4.48 Llama / Qwen2 / Qwen3 family)")
    config = model.config
    attention = config._attn_implementation
    originals = list(layers)
    forwards = [vars(layer).get("forward", _MISSING) for layer in originals]
    attributes = {key: vars(model).get(key, _MISSING) for key in ("seqlen", "hf_device_map")}
    applied = [f"attention {attention!r} -> 'eager' (4-D mask tensor for the Catcher)",
               "decoder-layer forward: official-style calls get position_embeddings = "
               "model.model.rotary_emb(hidden, position_ids) and return (hidden,)",
               "model.model.layers entries swapped in by official code inherit attention_type",
               f"model.seqlen = {seqlen}"]
    try:
        config._attn_implementation = "eager"
        model.seqlen = seqlen
        if attributes["hf_device_map"] is _MISSING:
            model.hf_device_map = {}
            applied.append("model.hf_device_map = {} (model not dispatched by accelerate)")
        layers.__class__ = _OfficialLayerList
        for layer in originals:
            layer.forward = _official_call(layer.forward, rotary)
        yield applied
    finally:
        layers.__class__ = nn.ModuleList
        for index, (layer, forward) in enumerate(zip(originals, forwards, strict=True)):
            if layers[index] is not layer:  # official code raised with its Catcher still installed
                layers[index] = layer
            if forward is _MISSING:
                vars(layer).pop("forward", None)
            else:
                layer.forward = forward
        for key, value in attributes.items():
            if value is _MISSING:
                vars(model).pop(key, None)
            else:
                setattr(model, key, value)
        config._attn_implementation = attention
