"""Replace selected linear/Conv2d layers and retain their original modules."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field

from torch import nn

from ..quant.spec import QuantSpec
from ..recipe import LinearSpec, Recipe, load_recipe
from .conv import EmuConv2d
from .linear import EmuLinear


@dataclass
class PatchReport:
    patched: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    layers: list[dict] = field(default_factory=list)
    kv: list[str] = field(default_factory=list)
    decoder_layers: str | None = None  # the ModuleList that ``layers:`` indices refer to
    unused_overrides: list[int] = field(default_factory=list)  # overrides selecting no enabled operator


def decoder_layers(model: nn.Module) -> tuple[str, int] | None:
    """``(name, length)`` of the decoder-block ModuleList: the one whose length is the (text)
    config's layer count, looked for under ``get_decoder()`` when the model has it. ``None`` when
    no single list qualifies (then ``layers:`` selectors cannot be resolved)."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "get_text_config"):
        config = config.get_text_config()
    count = next((getattr(config, key) for key in ("num_hidden_layers", "n_layer", "num_layers")
                  if isinstance(getattr(config, key, None), int)), None)
    scope = ""
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        try:
            decoder = get_decoder()
        except (AttributeError, NotImplementedError):
            decoder = None
        scope = next((name for name, module in model.named_modules() if module is decoder), "")
    lists = [(name, len(module)) for name, module in model.named_modules()
             if isinstance(module, nn.ModuleList) and len(module) > 0
             and (not scope or name.startswith(scope + ".")) and (count is None or len(module) == count)]
    return lists[0] if len(lists) == 1 else None


def _quant_summary(spec: QuantSpec | None) -> dict | None:
    if spec is None:
        return None
    return {"format": spec.format.name, "granularity": spec.granularity,
            "group_size": spec.group_size, "rounding": spec.rounding.value, "mma_input": spec.mma_input}


def _layer_summary(name: str, spec: LinearSpec | None, reason: str | None = None) -> dict:
    return {"name": name, "weight": None if spec is None else _quant_summary(spec.weight),
            "activation": None if spec is None else _quant_summary(spec.activation),
            "mma": None if spec is None else {"algorithm": spec.mma.algorithm, "f_bits": spec.mma.f_bits,
                                              "g_bits": spec.mma.g_bits, "name": spec.mma.name},
            "transform": None if spec is None else spec.transform.kind,
            "weight_algo": None if spec is None else spec.weight_algo.kind,
            "sparsity": None if spec is None else _sparsity_summary(spec),
            "outliers": None if spec is None or spec.outliers is None else
            {"fraction": spec.outliers.fraction, "format": str(spec.outliers.format)},
            "skipped": reason}


def _sparsity_summary(spec: LinearSpec) -> str:
    sparsity = spec.sparsity
    if sparsity.kind == "n:m":
        return f"{sparsity.n}:{sparsity.m}"
    return f"unstructured {sparsity.ratio}" if sparsity.kind == "unstructured" else "none"


def patch_model(
    model: nn.Module, recipe: Recipe, *, backend: str | None = None, include_conv2d: bool = False,
) -> PatchReport:
    """Patch Linear layers and optionally inference-only Conv2d lowering.

    Conv2d is opt-in because unfold is memory-intensive and calibration/training
    recipes are not supported for it. Disabled convolutions are reported, not
    silently described as emulated.
    """
    recipe = load_recipe(recipe)
    backend = backend or recipe.backend
    report = PatchReport()
    replacements = []
    shared = {}
    conv_alias_specs = {}
    modules = list(model.named_modules(remove_duplicate=False))
    decoder = decoder_layers(model)
    uses_layers = any("layers" in entry for entry in recipe.overrides) or recipe.kv_layers is not None
    if decoder is None and uses_layers:
        raise ValueError("layers: selectors need the decoder-block list, but no single ModuleList matches "
                         "the model config's layer count; select modules with match: patterns instead")
    report.decoder_layers = None if decoder is None else decoder[0]
    num_layers = None if decoder is None else decoder[1]
    used = set()
    for name, module in modules:
        if isinstance(module, (EmuLinear, EmuConv2d)):
            report.skipped.append(name)
            report.layers.append(_layer_summary(name, module.spec, "already patched"))
            continue
        if isinstance(module, nn.Conv2d) and not include_conv2d:
            report.skipped.append(name)
            report.layers.append(_layer_summary(name, None, "Conv2d disabled (include_conv2d=False)"))
            continue
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        used.add(recipe.override_index(name, decoder=decoder))
        spec = recipe.spec_for(name, num_layers=num_layers, decoder=decoder)
        if isinstance(module, nn.Conv2d):
            if id(module) in conv_alias_specs and conv_alias_specs[id(module)] != spec:
                raise ValueError(f"{name}: a shared Conv2d cannot use different recipes "
                                 "or skips at its aliases")
            conv_alias_specs[id(module)] = spec
        if spec is None:
            report.skipped.append(name)
            report.layers.append(_layer_summary(name, None, recipe.skip_reason(name, decoder=decoder)))
            continue
        if not name:
            constructor = ("EmuConv2d.from_conv2d" if isinstance(module, nn.Conv2d)
                           else "EmuLinear.from_linear")
            raise ValueError(f"patch_model needs a parent module; use {constructor} for a root layer")
        if id(module) in shared:
            replacement = shared[id(module)]
            if replacement.spec != spec:
                kind = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"
                raise ValueError(f"{name}: a shared {kind} cannot use different recipes at its aliases")
        else:
            replacement = (EmuConv2d.from_conv2d(module, spec, name, backend)
                           if isinstance(module, nn.Conv2d) else
                           EmuLinear.from_linear(module, spec, name, backend))
            shared[id(module)] = replacement
        parent_name, _, child_name = name.rpartition(".")
        replacements.append((model.get_submodule(parent_name), child_name, replacement))
        report.patched.append((name, replacement.extra_repr()))
        report.layers.append(_layer_summary(name, spec))
    report.unused_overrides = [i for i in range(len(recipe.overrides)) if i not in used]
    if report.unused_overrides:
        operators = "Linear or Conv2d" if include_conv2d else "Linear"
        warnings.warn(f"recipe {recipe.name!r}: overrides {report.unused_overrides} "
                      f"select no {operators} module", stacklevel=2)
    if recipe.kv is not None:
        from ..kv import apply_kv

        report.kv = apply_kv(model, recipe.kv, layers=recipe.kv_layers)
        model._tricast_kv_patched = True
    for parent, child_name, replacement in replacements:
        setattr(parent, child_name, replacement)
    return report


def unpatch_model(model: nn.Module) -> None:
    if getattr(model, "_tricast_kv_patched", False):
        from ..kv import remove_kv

        remove_kv(model)
        del model._tricast_kv_patched
    for name, module in list(model.named_modules(remove_duplicate=False)):
        if isinstance(module, (EmuLinear, EmuConv2d)):
            original = module._original
            original.weight = module.weight
            if isinstance(module, EmuConv2d):
                original.bias = module.bias
            elif original.bias is not None:
                original.bias.data = module.bias.detach().to(module.weight).clone()
            original.train(module.training)
            parent_name, _, child_name = name.rpartition(".")
            setattr(model.get_submodule(parent_name), child_name, original)


def iter_emulinear(model: nn.Module) -> Iterator[tuple[str, EmuLinear]]:
    """Yield ``(name, layer)`` pairs in model traversal order."""
    for name, module in model.named_modules():
        if isinstance(module, EmuLinear):
            yield name, module


def iter_emuconv2d(model: nn.Module) -> Iterator[tuple[str, EmuConv2d]]:
    """Yield registered Conv2d wrappers, excluding their private linear caches."""
    for name, module in model.named_modules():
        if isinstance(module, EmuConv2d):
            yield name, module
