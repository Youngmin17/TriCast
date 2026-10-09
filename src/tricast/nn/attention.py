"""Opt-in, instance-local Llama/Qwen3 eager-attention arithmetic emulation.

The integration preserves the Transformers 4.55.2 forward function, replacing
only its eager-attention callable in an isolated globals dictionary. No shared
Transformers dispatch table or class is modified.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..mma.api import gemm
from ..quant.api import quantize
from ..recipe import LinearSpec
from ..rounding import Rounding


@dataclass(frozen=True)
class AttentionSpec:
    """Explicit, independently chosen QK and PV matrix arithmetic.

    ``activation`` quantizes Q/P; ``weight`` quantizes K/the transposed V.
    A scale domain is one batch item and query head, never the whole batch.
    QK's K axis is head_dim; PV's K axis is cached key length. PV weight rows
    are output features, not cached tokens. No projection recipe is inferred.
    """

    qk: LinearSpec
    pv: LinearSpec

    def __post_init__(self) -> None:
        for name, spec in (("qk", self.qk), ("pv", self.pv)):
            if not isinstance(spec, LinearSpec):
                raise TypeError(f"attention.{name}: expected an explicit LinearSpec")
            if spec.transform.kind != "none" or spec.weight_algo.kind != "rtn":
                raise NotImplementedError(f"attention.{name}: transforms and GPTQ are not supported")
            if spec.sparsity.kind != "none" or spec.outliers is not None:
                raise NotImplementedError(f"attention.{name}: sparsity and outliers are not supported")
            if any(q is not None and q.observer is not None for q in (spec.activation, spec.weight)):
                raise NotImplementedError(f"attention.{name}: observers/calibration are not supported")
            if any(q is not None and (q.rounding is Rounding.SR or
                                      (q.scale is not None and q.scale.rounding is Rounding.SR))
                   for q in (spec.activation, spec.weight)):
                raise NotImplementedError(f"attention.{name}: stochastic rounding needs explicit noise; "
                                          "the attention adapter does not supply it")


@dataclass
class AttentionPatchReport:
    patched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _AttentionState:
    spec: AttentionSpec
    name: str
    backend: str


@dataclass
class _InstanceForward:
    # A module-level callable survives deepcopy/pickle; a dynamically bound
    # method named _attention_forward would not unpickle as module.forward.
    module: nn.Module

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return _attention_forward(self.module, *args, **kwargs)


def attention_gemm(
    activation: torch.Tensor, weight: torch.Tensor, spec: LinearSpec, *, backend: str,
) -> torch.Tensor:
    """One attention matrix multiply; the hook point for exact operand validation."""
    a = activation if spec.activation is None else quantize(activation, spec.activation, backend=backend)
    b = weight if spec.weight is None else quantize(weight, spec.weight, backend=backend)
    return gemm(a, b, spec.mma, backend=backend)


def emulated_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, spec: AttentionSpec, *,
    scaling: float, num_key_value_groups: int = 1, attention_mask: torch.Tensor | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal QK -> native mask/FP32 softmax -> PV, one GEMM per batch/head.

    GEMM out_format applies first, followed by a cast to the query dtype before
    the native HF scaling/mask epilogue. Probabilities are FP32-softmax rounded
    back to query.dtype. PV is cast to query.dtype before output reconstruction.
    A supplied mask is additive floating-point [B,1|H,Tq,Tk] and already encodes
    causality/padding; without a mask only a single query token is accepted.
    This inference-only primitive does not emulate softmax, scaling or RoPE.
    """
    if backend not in ("auto", "reference", "triton"):
        raise ValueError("attention.backend: expected auto, reference, or triton")
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("attention: expected query/key/value in [batch, heads, tokens, head_dim]")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        raise NotImplementedError("attention: autograd is not supported; use torch.no_grad()")
    if torch.is_grad_enabled() and attention_mask is not None and attention_mask.requires_grad:
        raise NotImplementedError("attention: mask autograd is not supported; use torch.no_grad()")
    batch, heads, queries, dim = query.shape
    if min(batch, heads, queries, dim) < 1 or key.shape[-2] < 1:
        raise ValueError("attention: empty dimensions are not supported")
    if num_key_value_groups < 1 or heads != key.shape[1] * num_key_value_groups:
        raise ValueError("attention: inconsistent grouped-query head counts")
    if key.shape != value.shape or key.shape[0] != batch or key.shape[-1] != dim:
        raise ValueError("attention: incompatible query/key/value shapes")
    if any(t.device != query.device or t.dtype != query.dtype for t in (key, value)):
        raise ValueError("attention: query/key/value device and dtype must agree")
    keys = key.shape[-2]
    if attention_mask is None and queries != 1:
        raise ValueError("attention: multi-token causal attention requires an explicit additive mask")
    if attention_mask is not None:
        if (attention_mask.ndim != 4 or not attention_mask.is_floating_point()
                or attention_mask.shape[0] not in (1, batch)
                or attention_mask.shape[1] not in (1, heads)
                or attention_mask.shape[2] != queries or attention_mask.shape[3] < keys
                or attention_mask.device != query.device):
            raise ValueError("attention: expected a floating additive [batch,1|heads,queries,keys] mask")
    scores = torch.stack([torch.stack([
        attention_gemm(query[b, h], key[b, h // num_key_value_groups], spec.qk, backend=backend)
        for h in range(heads)]) for b in range(batch)]).to(query.dtype)
    scores = scores * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[..., :keys]
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.stack([torch.stack([
        attention_gemm(probabilities[b, h], value[b, h // num_key_value_groups].T, spec.pv, backend=backend)
        for h in range(heads)]) for b in range(batch)]).to(query.dtype)
    return output.transpose(1, 2).contiguous(), probabilities


def _eager_attention(
    module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    attention_mask: torch.Tensor | None, scaling: float, dropout: float = 0.0, **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = module._tricast_attention
    if module.training or dropout:
        raise NotImplementedError(f"{state.name}: attention training/dropout is not supported; use eval()")
    return emulated_attention(query, key, value, state.spec, scaling=scaling,
                              num_key_value_groups=module.num_key_value_groups,
                              attention_mask=attention_mask, backend=state.backend)


def _attention_forward(module: nn.Module, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers.cache_utils import DynamicCache, DynamicLayer

    state = module._tricast_attention
    if module.training:
        raise NotImplementedError(f"{state.name}: attention training is not supported; use eval()")
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    positions = kwargs.get("position_embeddings", args[1] if len(args) > 1 else ())
    mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
    inputs = [hidden, mask, *(positions or ())]
    if torch.is_grad_enabled() and (any(parameter.requires_grad for parameter in module.parameters())
                                    or any(isinstance(value, torch.Tensor) and value.requires_grad
                                           for value in inputs)):
        raise NotImplementedError(f"{state.name}: attention autograd is not supported; use torch.no_grad()")
    if module.config._attn_implementation != "eager":
        raise NotImplementedError(f"{state.name}: emulated attention requires the eager mask/forward path")
    cache = kwargs.get("past_key_value", args[3] if len(args) > 3 else None)
    if cache is not None and type(cache) is not DynamicCache:
        raise NotImplementedError(f"{state.name}: only HF DynamicCache is supported")
    if cache is not None and (cache.cache_processor is not None
                              or any(type(layer) is not DynamicLayer for layer in cache.layers)):
        raise NotImplementedError(f"{state.name}: only unprocessed, plain-layer HF DynamicCache is supported")
    if torch.is_grad_enabled() and cache is not None and any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for layer in cache.layers for value in (layer.keys, layer.values)
    ):
        raise NotImplementedError(f"{state.name}: cached autograd history is not supported; "
                                  "use torch.no_grad()")
    original = type(module).forward
    namespace = dict(original.__globals__, eager_attention_forward=_eager_attention)
    # Reuse the audited upstream implementation for projections, norms, RoPE,
    # cache update and output projection; do not copy or modify its shared globals.
    forward = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__,
                           original.__closure__)
    forward.__kwdefaults__ = original.__kwdefaults__
    return forward(module, *args, **kwargs)


def _supported_types() -> tuple[type[nn.Module], ...]:
    import transformers
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    if transformers.__version__ != "4.55.2":
        raise NotImplementedError("attention integration is audited for Transformers 4.55.2 only")
    return LlamaAttention, Qwen3Attention


def patch_attention(
    model: nn.Module, spec: AttentionSpec, *, backend: str = "auto",
) -> AttentionPatchReport:
    """Bind opt-in inference attention to supported instances, with atomic preflight.

    Load HF models with attn_implementation='eager'. The model and config must
    be instance-local: shared attention module aliases remain aliases. Native
    projections can independently be patched with patch_model. Static/sliding
    caches, sliding attention, custom forward/hooks and training are rejected.
    """
    if not isinstance(spec, AttentionSpec):
        raise TypeError("attention.spec: expected AttentionSpec(qk=LinearSpec(...), pv=LinearSpec(...))")
    if backend not in ("auto", "reference", "triton"):
        raise ValueError("attention.backend: expected auto, reference, or triton")
    supported = _supported_types()
    report = AttentionPatchReport()
    selected: dict[int, tuple[str, nn.Module]] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, supported):
            continue
        if type(module) not in supported:
            raise NotImplementedError(f"{name}: custom attention subclasses are not supported")
        if hasattr(module, "_tricast_attention"):
            state = module._tricast_attention
            if state.spec != spec or state.backend != backend:
                raise ValueError(f"{name}: attention is already patched with a different spec/backend")
            report.skipped.append(name)
            continue
        if "forward" in module.__dict__:
            raise NotImplementedError(f"{name}: an instance-specific attention forward is not supported")
        if any(getattr(module, hook) for hook in ("_forward_pre_hooks", "_forward_hooks",
                                                 "_backward_pre_hooks", "_backward_hooks")):
            raise NotImplementedError(f"{name}: attention module hooks are not supported")
        if module.config._attn_implementation != "eager":
            raise NotImplementedError(f"{name}: load the model with attn_implementation='eager'")
        if getattr(module, "sliding_window", None) is not None:
            raise NotImplementedError(f"{name}: sliding-window attention is not supported")
        selected.setdefault(id(module), (name, module))
        report.patched.append(name)
    if not selected and not report.skipped:
        raise ValueError("attention: no supported Llama/Qwen3 attention modules found")
    for name, module in selected.values():
        module._tricast_attention = _AttentionState(spec, name, backend)
        module.forward = _InstanceForward(module)
    return report


def unpatch_attention(model: nn.Module) -> None:
    """Restore upstream class forwards without touching projections/Parameters/config."""
    for _, module in model.named_modules():
        if hasattr(module, "_tricast_attention"):
            del module.forward
            del module._tricast_attention


def iter_emuattention(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if hasattr(module, "_tricast_attention"):
            yield name, module
