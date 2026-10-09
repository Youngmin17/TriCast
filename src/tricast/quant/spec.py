"""Quantization specs: how a tensor is mapped onto a format.

A 2-D operand is always viewed as ``[rows, K]`` with ``K`` the reduction axis
(``nn.Linear`` weights are ``[N, K]``, activations ``[M, K]``). The *scale domain*
is the set of elements sharing one scale:

=============  ================================================================
granularity    scale domain
=============  ================================================================
``tensor``     the whole tensor
``row``        one row (per-output-channel for weights, per-token for activations;
               ``channel`` and ``token`` are accepted synonyms)
``group``      ``group_size`` consecutive elements of a row along K (MX, NVFP4, BFP, g128)
``block``      a ``block[0] x block[1]`` tile of ``[rows, K]`` (e.g. 128x128 weight blocks)
=============  ================================================================

Exact scale/element semantics are defined by the reference implementation,
:mod:`tricast.reference.quantize`.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Literal

from ..formats import BF16, E8M0, FP4_E2M1, FP16, FP32, UE4M3, FloatFormat, Format, IntFormat, get_format
from ..rounding import Rounding

Granularity = Literal["tensor", "row", "group", "block"]
ScaleMethod = Literal["absmax", "pow2_floor", "pow2_ceil", "mse", "percentile"]
ZeroPoint = Literal["none", "int", "float"]
MMAInput = Literal["scaled", "dequant"]
ObserverKind = Literal["minmax", "ema", "history", "percentile", "mse"]
TransformKind = Literal["none", "hadamard", "random_hadamard", "smoothquant", "awq"]
WeightAlgo = Literal["rtn", "gptq"]

FOUR_OVER_SIX = (1.0, 1.5)  # NVFP4 "Four Over Six": map each block's amax to 6 or to 4

_GRANULARITY_SYNONYMS = {"channel": "row", "token": "row", "per_channel": "row",
                         "per_token": "row", "per_tensor": "tensor"}


@dataclass(frozen=True)
class ScaleSpec:
    """How the scale of each scale domain is chosen and stored.

    method
        ``absmax``      s = amax / max_elem, then rounded onto ``format``
        ``pow2_floor``  s = 2**(floor(log2 amax) - emax_elem)   (OCP MX / microxcaling "max")
        ``pow2_ceil``   s = 2**ceil(log2(amax / max_elem))      (never clips)
        ``mse``         absmax on ``r * amax`` for each ratio ``r`` in ``search`` (default: an
                        ``mse_grid``-point grid from 1.0 down to 0.5), keeping the ``r`` with
                        the smallest L2 error; ``search=(1.0, 1.5)`` is NVFP4 Four-over-Six
        ``percentile``  absmax on the ``percentile``-th percentile of ``|x|``
    two_level
        NVFP4-style: an fp32 per-tensor decode scale ``d2 = amax_tensor / (max_elem *
        format.max_normal)`` so block scales use the full range of ``format``.
    """

    format: Format = FP32
    method: ScaleMethod = "absmax"
    rounding: Rounding = Rounding.RNE
    two_level: bool = False
    percentile: float = 99.99
    mse_grid: int = 20
    search: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.method == "four_over_six":
            object.__setattr__(self, "method", "mse")
            object.__setattr__(self, "search", FOUR_OVER_SIX)
        object.__setattr__(self, "format", get_format(self.format))
        object.__setattr__(self, "rounding", Rounding.parse(self.rounding))
        object.__setattr__(self, "search", tuple(float(r) for r in self.search))
        if self.method not in ("absmax", "pow2_floor", "pow2_ceil", "mse", "percentile"):
            raise ValueError(f"unknown scale method {self.method!r}")
        if any(r <= 0 for r in self.search) or self.mse_grid < 1:
            raise ValueError("search ratios must be > 0 and mse_grid >= 1")
        if self.format.kind == "pow2" and self.method not in ("pow2_floor", "pow2_ceil"):
            raise ValueError("a power-of-two scale format needs method pow2_floor or pow2_ceil")
        if isinstance(self.format, IntFormat):
            raise ValueError("integer scale formats are not supported; use a float or e8m0")
        if not 0 < self.percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")


@dataclass(frozen=True)
class ObserverSpec:
    """Where a *static* scale's amax comes from (activations). ``None`` on a
    QuantSpec means dynamic "current" scaling: amax of the tensor at each call.

    ``minmax``      running max of per-call amax over calibration
    ``ema``         ``amax = decay * amax + (1 - decay) * amax_call`` (first call initialises)
    ``history``     Transformer-Engine delayed scaling: the scale for a call comes from the
                    previous ``history_len`` amaxes reduced by ``reduce`` (``max`` |
                    ``most_recent``); the first call falls back to its own amax
    ``percentile``  ``percentile`` of ``|x|`` over all calibration samples
    ``mse``         the ScaleSpec's mse search run on concatenated calibration samples
    """

    kind: ObserverKind = "ema"
    decay: float = 0.99
    history_len: int = 16
    reduce: Literal["max", "most_recent"] = "max"
    percentile: float = 99.99
    max_samples: int = 1 << 20

    def __post_init__(self) -> None:
        if self.kind not in ("minmax", "ema", "history", "percentile", "mse"):
            raise ValueError(f"unknown observer kind {self.kind!r}")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("ema decay must be in [0, 1)")
        if self.history_len < 1 or self.reduce not in ("max", "most_recent"):
            raise ValueError("history needs history_len >= 1 and reduce in {max, most_recent}")


@dataclass(frozen=True)
class TransformSpec:
    """Function-preserving rewrite of a linear layer applied before quantization:
    ``y = x W^T = (x T)(W T^-T)^T``.

    ``hadamard``         T = block-diagonal normalised Hadamard of size ``block`` along K
    ``random_hadamard``  T = D H with a seeded random ±1 diagonal D (RHT)
    ``smoothquant``      T = diag(1/s), ``s_j = max|X_j|**alpha / max|W_j|**(1-alpha)``
    ``awq``              diag scaling with ``alpha`` searched on a ``grid`` to minimise output error
    ``block=0`` picks the largest power of two ≤ 128 that divides K.
    ``share_inputs`` (smoothquant / awq): linears that read the same activation (q/k/v,
    gate/up) share one scale vector — the deployable form, where ``1/s`` folds into the
    preceding norm. ``False`` fits each linear on its own.
    """

    kind: TransformKind = "none"
    block: int = 0
    seed: int = 0
    alpha: float = 0.5
    grid: int = 20
    share_inputs: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("none", "hadamard", "random_hadamard", "smoothquant", "awq"):
            raise ValueError(f"unknown transform {self.kind!r}")
        if self.block and self.block & (self.block - 1):
            raise ValueError("hadamard block must be a power of two")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")


@dataclass(frozen=True)
class WeightAlgoSpec:
    """How weights are rounded. ``rtn``: independent round-to-nearest per element.
    ``gptq``: error-compensating column-by-column rounding with the calibration
    Hessian ``H = 2 X^T X`` (any element format / scale spec, groups included)."""

    kind: WeightAlgo = "rtn"
    block_size: int = 128
    damp: float = 0.01
    act_order: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("rtn", "gptq"):
            raise ValueError(f"unknown weight algorithm {self.kind!r}")


@dataclass(frozen=True)
class QuantSpec:
    """Quantization of one operand.

    ``scale=None`` means a direct cast onto ``format`` with no scaling (e.g. bf16).
    ``mma_input`` decides what the MMA datapath consumes:

    ``scaled``   element values on the ``format`` grid; scales are applied by the MMA
                 algorithm (epilogue for tensor/row scales, inside the K loop for
                 group/block scales). Zero points are not allowed here.
    ``dequant``  ``x_hat = dequant(q)`` rounded onto ``dequant_format`` (e.g. W4A16:
                 int4 weights dequantised to bf16 before a bf16 MMA).
    """

    format: Format
    granularity: Granularity = "tensor"
    group_size: int = 0
    block: tuple[int, int] = (0, 0)
    scale: ScaleSpec | None = field(default_factory=ScaleSpec)
    zero_point: ZeroPoint = "none"
    rounding: Rounding = Rounding.RNE
    sr_bits: int = 32
    saturate: bool = True
    mma_input: MMAInput = "scaled"
    dequant_format: Format = BF16
    observer: ObserverSpec | None = None

    def __post_init__(self) -> None:
        set_ = lambda k, v: object.__setattr__(self, k, v)  # noqa: E731 (frozen dataclass)
        set_("format", get_format(self.format))
        set_("dequant_format", get_format(self.dequant_format))
        set_("rounding", Rounding.parse(self.rounding))
        set_("granularity", _GRANULARITY_SYNONYMS.get(self.granularity, self.granularity))
        set_("block", tuple(self.block))
        if isinstance(self.observer, dict):
            set_("observer", ObserverSpec(**self.observer))
        if not 1 <= self.sr_bits <= 32:
            raise ValueError("sr_bits must be in [1, 32]")
        if self.observer is not None and (self.granularity != "tensor" or self.scale is None):
            raise ValueError("static observers need a scaled, per-tensor spec "
                             "(row/group scales of activations change every call)")
        if self.granularity not in ("tensor", "row", "group", "block"):
            raise ValueError(f"unknown granularity {self.granularity!r}")
        if self.granularity == "group" and self.group_size <= 0:
            raise ValueError("granularity 'group' needs group_size > 0")
        if self.granularity == "block" and min(self.block) <= 0:
            raise ValueError("granularity 'block' needs block=(rows, cols) > 0")
        if self.format.kind == "pow2":
            raise ValueError("e8m0 is a scale format, not an element format")
        if self.zero_point != "none":
            if not isinstance(self.format, IntFormat):
                raise ValueError("zero points need an integer element format")
            if self.mma_input != "dequant":
                raise ValueError("zero points need mma_input='dequant'")
            if self.scale is None or self.scale.two_level:
                raise ValueError("zero points need a single-level scale")
            if self.zero_point == "float" and self.format.signed:
                raise ValueError("float zero points (KIVI/HQQ offset = min) need an unsigned format")
            if self.observer is not None:
                # Observers track amax only; an affine scale needs min and max.
                raise ValueError("static observers support symmetric scales only (zero_point='none')")
            if self.scale.method != "absmax":
                warnings.warn(f"zero-point quantization takes its range from [min, max]; "
                              f"scale.method={self.scale.method!r} is not used", stacklevel=3)
        if self.scale is None and self.granularity != "tensor":
            raise ValueError("a direct cast (scale=None) has no granularity")
        if not isinstance(self.dequant_format, FloatFormat):
            raise ValueError("dequant_format must be a float format")

    @property
    def emax_elem(self) -> int:
        """``floor(log2(max_elem))`` — the exponent offset used by pow2 scale methods
        (equals ``format.emax`` for floats; 0 for MXINT8, whose max is 127/64)."""
        return math.frexp(self.format.max_normal)[1] - 1

    def with_(self, **changes) -> QuantSpec:
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, d: dict) -> QuantSpec:
        """Build from a recipe mapping. ``scheme`` expands a named scheme first."""
        d = dict(d)
        base = get_scheme(d.pop("scheme")) if "scheme" in d else None
        if "granularity" in d and isinstance(d["granularity"], str) and ":" in d["granularity"]:
            d.update(_parse_granularity(d.pop("granularity")))
        scale = d.pop("scale", "__keep__")
        if scale == "__keep__":
            scale = base.scale if base else ScaleSpec()
        elif scale is not None and not isinstance(scale, ScaleSpec):
            scale = ScaleSpec(**{**(vars(base.scale) if base and base.scale else {}), **scale})
        if "block" in d:
            d["block"] = tuple(d["block"])
        if base is not None:
            return replace(base, scale=scale, **d)
        return cls(scale=scale, **d)


KVAxis = Literal["channel", "token"]
KVMode = Literal["cache", "fakequant"]


@dataclass(frozen=True)
class KVSpec:
    """Quantization of the attention KV cache (KIVI, arXiv:2402.02750, and relatives).

    ``channel`` groups each head-dim channel over complete groups of tokens;
    dynamic channel specs therefore require ``group`` granularity. ``token``
    groups head-dim channels independently for each token and head: ``tensor``
    means a per-token scale here, and blocks may not span tokens. Scale domains
    never cross batch rows or heads. ``scale=None`` is a static unit-scale cast.

    After n real tokens, ``channel`` quantizes floor(n/R)*R tokens and keeps
    n mod R tokens in full precision (R > 0, and R is a multiple of group size G).
    ``token`` quantizes max(0, n-R) tokens and keeps the most recent min(n, R).
    ``cache`` attends with the pre-update state plus new full-precision tokens,
    then writes this state; a prompt prefill therefore attends in full precision.
    ``fakequant`` reproduces token-by-token cache decoding: query t uses the
    stored state of its t predecessors and its own full-precision key/value.
    """

    key: QuantSpec | None = None
    value: QuantSpec | None = None
    key_axis: KVAxis = "channel"
    value_axis: KVAxis = "token"
    residual: int = 128
    mode: KVMode = "cache"

    def __post_init__(self) -> None:
        for name in ("key", "value"):
            spec = getattr(self, name)
            if isinstance(spec, (str, dict)):
                object.__setattr__(self, name, get_scheme(spec) if isinstance(spec, str)
                                   else QuantSpec.from_dict(spec))
        if self.key is None and self.value is None:
            raise ValueError("a KVSpec needs a key or a value spec")
        if {self.key_axis, self.value_axis} - {"channel", "token"}:
            raise ValueError("key_axis and value_axis must be 'channel' or 'token'")
        if self.mode not in ("cache", "fakequant") or self.residual < 0:
            raise ValueError("mode must be 'cache' or 'fakequant' and residual >= 0")
        for name in ("key", "value"):
            spec = getattr(self, name)
            if spec is None:
                continue
            if spec.observer is not None or spec.mma_input != "dequant":
                raise ValueError("KV specs require dynamic or unit scales and mma_input='dequant'")
            if spec.rounding == Rounding.SR or (
                spec.scale is not None and spec.scale.rounding == Rounding.SR
            ):
                raise ValueError("stochastic rounding is unsupported for chunk-invariant KV specs")
            axis = getattr(self, f"{name}_axis")
            if axis == "channel" and self.residual == 0:
                raise ValueError("channel-axis KV specs require residual > 0")
            if spec.scale is None:
                continue
            if spec.scale.two_level:
                raise ValueError("two-level scales are unsupported for chunk-invariant KV specs")
            if axis == "channel" and spec.granularity != "group":
                raise ValueError("dynamic channel-axis KV specs require group granularity")
            if axis == "channel" and self.residual % spec.group_size != 0:
                raise ValueError("channel-axis KV residual must be a multiple of group_size")
            if axis == "token" and spec.granularity == "block" and spec.block[0] != 1:
                raise ValueError("token-axis KV blocks must have one row, not span tokens")


def _parse_granularity(text: str) -> dict:
    """``group:32`` -> group_size 32; ``block:128x128`` -> block (128, 128)."""
    kind, arg = text.split(":", 1)
    if kind == "group":
        return {"granularity": "group", "group_size": int(arg)}
    if kind == "block":
        r, c = arg.lower().split("x")
        return {"granularity": "block", "block": (int(r), int(c))}
    raise ValueError(f"cannot parse granularity {text!r}")


# --------------------------------------------------------------------------
# Named schemes
# --------------------------------------------------------------------------

def _mx(elem: Format) -> QuantSpec:
    return QuantSpec(elem, "group", group_size=32, scale=ScaleSpec(E8M0, "pow2_floor"))


def bfp(mbits: int, block: int) -> QuantSpec:
    """Block floating point: shared 8-bit exponent per ``block`` elements and
    ``mbits``-bit sign-magnitude mantissas (``mbits`` includes the sign)."""
    elem = IntFormat(f"bfp_m{mbits}", mbits, frac_bits=mbits - 2)
    return QuantSpec(elem, "group", group_size=block, scale=ScaleSpec(E8M0, "pow2_floor"))


SCHEMES: dict[str, QuantSpec] = {
    # OCP Microscaling (block 32, E8M0 shared scale)
    "mxfp8_e4m3": _mx("fp8_e4m3"), "mxfp8_e5m2": _mx("fp8_e5m2"),
    "mxfp6_e3m2": _mx("fp6_e3m2"), "mxfp6_e2m3": _mx("fp6_e2m3"),
    "mxfp4": _mx("fp4_e2m1"), "mxint8": _mx("mxint8"), "mxint4": _mx("mxint4"),
    # NVIDIA NVFP4: E2M1, block 16, UE4M3 block scale + fp32 tensor scale
    "nvfp4": QuantSpec(FP4_E2M1, "group", group_size=16, scale=ScaleSpec(UE4M3, "absmax", two_level=True)),
    "nvfp4_4o6": QuantSpec(FP4_E2M1, "group", group_size=16,
                           scale=ScaleSpec(UE4M3, "mse", two_level=True, search=FOUR_OVER_SIX)),
    # Block floating point (MSFP naming: sign + mantissa bits, block 16)
    "msfp12": bfp(4, 16), "msfp16": bfp(8, 16),
    # FP8 with fp32 scales
    "fp8_tensor": QuantSpec("fp8_e4m3", "tensor"),
    "fp8_tensor_ema": QuantSpec("fp8_e4m3", "tensor", observer=ObserverSpec("ema")),
    "fp8_tensor_delayed": QuantSpec("fp8_e4m3", "tensor", observer=ObserverSpec("history")),
    "fp8_row": QuantSpec("fp8_e4m3", "row"),
    "fp8_group128": QuantSpec("fp8_e4m3", "group", group_size=128),
    "fp8_block128": QuantSpec("fp8_e4m3", "block", block=(128, 128)),
    # Integer
    "int8_tensor": QuantSpec("int8", "tensor"),
    "int8_row": QuantSpec("int8", "row"),
    "int4_g128": QuantSpec("int4", "group", group_size=128, scale=ScaleSpec(FP16),
                           mma_input="dequant"),
    "int4_g128_zp": QuantSpec("uint4", "group", group_size=128, scale=ScaleSpec(FP16),
                              zero_point="int", mma_input="dequant"),
    # KIVI KV cache elements: asymmetric min/max, fp16 scale and float offset, group 32
    "kivi2": QuantSpec("uint2", "group", group_size=32, scale=ScaleSpec(FP16), zero_point="float",
                       mma_input="dequant", dequant_format=FP16),
    "kivi4": QuantSpec("uint4", "group", group_size=32, scale=ScaleSpec(FP16), zero_point="float",
                       mma_input="dequant", dequant_format=FP16),
    # Direct casts (no scale)
    "bf16": QuantSpec(BF16, scale=None), "fp16": QuantSpec(FP16, scale=None),
    "fp32": QuantSpec(FP32, scale=None), "tf32": QuantSpec("tf32", scale=None),
}

_BFP = re.compile(r"^bfp(\d+)_b(\d+)$")


def get_scheme(name: str | QuantSpec) -> QuantSpec:
    """Named scheme, or ``bfp{m}_b{block}`` for any block floating point."""
    if isinstance(name, QuantSpec):
        return name
    key = name.strip().lower()
    if key in SCHEMES:
        return SCHEMES[key]
    m = _BFP.match(key)
    if m:
        return bfp(int(m.group(1)), int(m.group(2)))
    raise ValueError(f"unknown scheme {name!r}; known: {sorted(SCHEMES)} or bfp<m>_b<block>")


KV_PRESETS: dict[str, KVSpec] = {
    # KIVI (Liu et al., ICML 2024), Algorithm 1: channel keys flush at residual 128;
    # token values retain the latest 128, both with group size 32 (KIVI §4.1).
    "kivi2": KVSpec(key="kivi2", value="kivi2", residual=128),
    "kivi4": KVSpec(key="kivi4", value="kivi4", residual=128),
    # vLLM v0.15.1 CacheConfig: fp8 (=E4M3), calculate_kv_scales=False and
    # no checkpoint scales uses static k_scale=v_scale=1.0 (not dynamic amax).
    # No residual: every appended token is quantized when stored.
    # https://docs.vllm.ai/en/v0.15.1/api/vllm/config/cache/#vllm.config.cache.CacheConfig.calculate_kv_scales
    "kv_fp8": KVSpec(key=QuantSpec("fp8_e4m3", scale=None, mma_input="dequant", dequant_format=FP32),
                     value=QuantSpec("fp8_e4m3", scale=None, mma_input="dequant", dequant_format=FP32),
                     key_axis="token", residual=0),
}


def get_kv_spec(spec: str | dict | KVSpec) -> KVSpec:
    """KV preset name, KVSpec, or a mapping of KVSpec fields (key/value as scheme or dict)."""
    if isinstance(spec, KVSpec):
        return spec
    if isinstance(spec, str):
        try:
            return KV_PRESETS[spec.strip().lower()]
        except KeyError:
            raise ValueError(f"unknown KV preset {spec!r}; known: {sorted(KV_PRESETS)}") from None
    d = dict(spec)
    base = KV_PRESETS[d.pop("preset")] if "preset" in d else None
    return replace(base, **d) if base else KVSpec(**d)
