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

Exact scale/element semantics live in ``docs/design/ENGINE.md`` §3; the
reference implementation is :mod:`tricast.reference.quantize`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Literal

from ..formats import BF16, E8M0, FP4_E2M1, FP16, FP32, UE4M3, Format, FloatFormat, IntFormat, get_format
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
    ``random_hadamard``  T = H D with a seeded random ±1 diagonal D (RHT)
    ``smoothquant``      T = diag(1/s), ``s_j = max|X_j|**alpha / max|W_j|**(1-alpha)``
    ``awq``              diag scaling with ``alpha`` searched on a ``grid`` to minimise output error
    ``block=0`` picks the largest power of two ≤ 128 that divides K.
    """

    kind: TransformKind = "none"
    block: int = 0
    seed: int = 0
    alpha: float = 0.5
    grid: int = 20

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
