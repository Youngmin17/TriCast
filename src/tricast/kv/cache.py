"""Streaming KV emulation using the Transformers 4.55 per-layer cache API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicLayer

from ..quant.api import quantize
from ..quant.spec import KVAxis, KVSpec, QuantSpec, get_kv_spec


def quantize_states(
    states: torch.Tensor, spec: QuantSpec | None, axis: KVAxis,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize batch/head-local domains, grouping only real tokens.

    Token-axis tensor scaling means one scale per token and head, never one
    scale per update. Channel groups begin at the first unmasked token. Masked
    entries stay unchanged for scaled specs, including fully masked eager
    queries. Direct casts have no scale statistics and remain elementwise.
    """
    if spec is None or states.numel() == 0:
        return states
    if spec.scale is not None and token_mask is not None:
        if token_mask.shape != (states.shape[0], states.shape[-2]):
            raise ValueError("KV token mask must have shape [batch, tokens]")
    if (axis == "channel" and spec.scale is not None
            and spec.granularity in ("tensor", "block") and states.shape[0] * states.shape[1] > 1):
        # Standalone callers may use non-streamable channel domains, but their
        # statistics still cannot cross a batch row or an attention head.
        return torch.cat([
            torch.cat([
                quantize_states(states[b:b + 1, h:h + 1], spec, axis,
                                None if token_mask is None else token_mask[b:b + 1])
                for h in range(states.shape[1])
            ], dim=1)
            for b in range(states.shape[0])
        ], dim=0)
    if spec.scale is None:
        # Elementwise casts have no scale statistics for padding to affect.
        token_mask = None
    if token_mask is not None:
        token_mask = token_mask.to(device=states.device, dtype=torch.bool)
        if not token_mask.all() and spec.scale.two_level and spec.granularity != "tensor":
            raise NotImplementedError("masked KV quantization with two-level scales is unsupported")
        if axis == "channel" and spec.scale is not None and not token_mask.all():
            result = states.float().clone()
            for batch in range(states.shape[0]):
                selected = token_mask[batch].nonzero(as_tuple=True)[0]
                if selected.numel():
                    compact = states[batch:batch + 1].index_select(-2, selected)
                    result[batch:batch + 1].index_copy_(
                        -2, selected, quantize_states(compact, spec, axis),
                    )
            return result
    # All supported dynamic domains now occupy one row (or a subgroup of it).
    # Flattening batch and head therefore cannot share their statistics.
    if axis == "token" and spec.scale is not None and spec.granularity == "tensor":
        spec = spec.with_(granularity="row")
    view = states.transpose(-1, -2) if axis == "channel" else states
    qt = quantize(view.reshape(-1, view.shape[-1]), spec, backend="auto")
    result = qt.mma_operand().values.reshape(view.shape)
    result = result.transpose(-1, -2) if axis == "channel" else result
    if token_mask is not None:
        result = torch.where(token_mask[:, None, :, None], result, states)
    return result


def _join(quantized: torch.Tensor | None, residual: torch.Tensor | None) -> torch.Tensor | None:
    if quantized is None:
        return residual
    return torch.cat((quantized, residual), dim=-2)


class _KVLayer(DynamicLayer):
    """Keep independent K/V boundaries; stored quantized values are not requantized."""

    def __init__(self, kv: KVSpec | None) -> None:
        self.kv = kv
        self.quantized_keys: torch.Tensor | None = None
        self.quantized_values: torch.Tensor | None = None
        self.residual_keys: torch.Tensor | None = None
        self.residual_values: torch.Tensor | None = None
        # Keep the pre-write view alive through multi-token attention calls.
        self.attention_keys: torch.Tensor | None = None
        self.attention_values: torch.Tensor | None = None
        self.attention_prefix_lengths: tuple[int, int] = (0, 0)

    @property
    def keys(self) -> torch.Tensor | None:
        result = _join(self.quantized_keys, self.residual_keys)
        return None if result is None else result.to(self.residual_keys.dtype)

    @property
    def values(self) -> torch.Tensor | None:
        result = _join(self.quantized_values, self.residual_values)
        return None if result is None else result.to(self.residual_values.dtype)

    def _append(self, states: torch.Tensor, name: str) -> torch.Tensor:
        quantized = getattr(self, f"quantized_{name}")
        residual = getattr(self, f"residual_{name}")
        residual = states if residual is None else torch.cat((residual, states), dim=-2)
        visible = _join(quantized, residual).to(states.dtype)
        if states.shape[-2] > 1:
            setattr(self, f"attention_{name}", visible)
        spec = getattr(self.kv, "key" if name == "keys" else "value", None)
        if spec is not None:
            axis = self.kv.key_axis if name == "keys" else self.kv.value_axis
            count = (residual.shape[-2] // self.kv.residual * self.kv.residual
                     if axis == "channel" else max(0, residual.shape[-2] - self.kv.residual))
            if count:
                addition = quantize_states(residual[..., :count, :], spec, axis)
                quantized = addition if quantized is None else torch.cat(
                    (quantized, addition), dim=-2,
                )
                residual = residual[..., count:, :]
        setattr(self, f"quantized_{name}", quantized)
        setattr(self, f"residual_{name}", residual)
        return visible

    def clear_attention_states(self) -> None:
        """Release temporary pre-flush values after the attention call."""
        self.attention_keys = self.attention_values = None
        self.attention_prefix_lengths = (0, 0)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("KV states must be [batch, heads, tokens, head_dim]")
        if key_states.shape[:3] != value_states.shape[:3]:
            raise ValueError("K and V must have matching batch, heads and tokens")
        self.clear_attention_states()
        if key_states.shape[-2] > 1:
            self.attention_prefix_lengths = tuple(
                0 if tensor is None else tensor.shape[-2]
                for tensor in (self.quantized_keys, self.quantized_values)
            )
        return self._append(key_states, "keys"), self._append(value_states, "values")

    def get_seq_length(self, cache_position: torch.Tensor | None = None) -> int:
        return sum(t.shape[-2] for t in (self.quantized_keys, self.residual_keys) if t is not None)

    def _map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        for name in ("quantized_keys", "quantized_values", "residual_keys", "residual_values",
                     "attention_keys", "attention_values"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, fn(value))

    def reset(self) -> None:
        self.clear_attention_states()
        self.quantized_keys = self.quantized_values = None
        self.residual_keys = self.residual_values = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._map(lambda x: x.index_select(0, beam_idx.to(x.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._map(lambda x: x.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._map(lambda x: x.index_select(0, indices.to(x.device)))

    def _check_crop(self, max_length: int) -> int:
        length = self.get_seq_length()
        max_length = max(0, length + max_length) if max_length < 0 else max_length
        if self.kv is not None and 0 < max_length < length:
            for name in ("key", "value"):
                if getattr(self.kv, name) is None:
                    continue
                quantized = getattr(self, f"quantized_{name}s")
                prefix = 0 if quantized is None else quantized.shape[-2]
                target_prefix = (max_length // self.kv.residual * self.kv.residual
                                 if getattr(self.kv, f"{name}_axis") == "channel"
                                 else max(0, max_length - self.kv.residual))
                if target_prefix < min(prefix, max_length):
                    # The target would need originals that quantization discarded.
                    raise NotImplementedError(
                        "speculative/assisted generation cannot roll back a quantized KV cache "
                        "past a residual flush"
                    )
        return max_length

    def crop(self, max_length: int) -> None:
        length = self.get_seq_length()
        max_length = self._check_crop(max_length)
        if max_length >= length:
            return
        if max_length == 0:
            self.reset()
            return
        self.clear_attention_states()
        for name in ("keys", "values"):
            quantized = getattr(self, f"quantized_{name}")
            residual = getattr(self, f"residual_{name}")
            prefix = 0 if quantized is None else quantized.shape[-2]
            if quantized is not None:
                setattr(self, f"quantized_{name}", quantized[..., :max_length, :])
            if residual is not None:
                setattr(self, f"residual_{name}", residual[..., :max(0, max_length - prefix), :])


class TriCastKVCache(Cache):
    """A numerical cache emulator, not a packed memory-saving cache.

    Channel-axis specs flush their entire buffer every ``R`` tokens, retaining
    ``n % R`` full-precision tokens. Token-axis specs retain the most recent ``R``
    tokens. Quantized prefixes hold fp32 containers of dequant-format values;
    ``update`` returns the previously stored state plus the appended full-precision
    tokens in the input dtype. Only stored state is independent of update chunking.
    Unselected layers use full-precision dynamic storage.
    Cropping preserves the same stored-state contract: it cannot restore originals
    discarded by a flush, so such crops raise ``NotImplementedError``. Complete
    quantized channel buffers and still-valid residual tails can be removed.
    No full-precision rollback copies are kept.
    """

    def __init__(
        self, kv: KVSpec, *, layers: set[int] | None = None, num_hidden_layers: int = 1,
    ) -> None:
        self.kv = get_kv_spec(kv)
        if self.kv.mode != "cache":
            raise ValueError("TriCastKVCache requires mode='cache'")
        if num_hidden_layers < 1 or (layers is not None and any(i < 0 for i in layers)):
            raise ValueError("cache layer counts must be positive and indices nonnegative")
        self.selected_layers = None if layers is None else frozenset(layers)
        super().__init__(layer_classes=DynamicLayer)
        self.num_hidden_layers = num_hidden_layers
        self.append_new_layers(num_hidden_layers - 1)

    def append_new_layers(self, layer_idx: int) -> None:
        if layer_idx < 0:
            raise ValueError("layer_idx must be nonnegative")
        while len(self.layers) <= layer_idx:
            selected = self.selected_layers is None or len(self.layers) in self.selected_layers
            self.layers.append(_KVLayer(self.kv if selected else None))

    def clear_attention_states(self) -> None:
        """Release per-forward pre-flush tensors from all cache layers."""
        for layer in self.layers:
            layer.clear_attention_states()

    def crop(self, max_length: int) -> None:
        # Refuse before changing any layer, including unselected dynamic layers.
        for layer in self.layers:
            layer._check_crop(max_length)
        super().crop(max_length)
