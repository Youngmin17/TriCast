"""Number formats — the shared vocabulary of TriCast.

A format fixes exactly which real values are representable (its *grid*).
How values off the grid are mapped onto it is decided separately by a
:class:`~tricast.rounding.Rounding` mode and an overflow policy.

Three kinds of format exist:

* :class:`FloatFormat` — sign / exponent / fraction binary floats
  (fp32, bf16, fp16, fp8, fp6, fp4, the unsigned ue4m3 scale format, custom ExMy).
* :class:`IntFormat`   — integers, optionally fixed-point (``value = k * 2**-frac_bits``).
  MX int elements (MXINT8) and block floating point mantissas are fixed-point ints.
* :class:`Pow2Format`  — unsigned powers of two (the OCP MX E8M0 shared scale).

All formats are frozen dataclasses: hashable, comparable, usable as dict keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Special = Literal["ieee", "fn", "fnuz", "none"]

_SPECIALS = ("ieee", "fn", "fnuz", "none")


@dataclass(frozen=True)
class FloatFormat:
    """Binary floating-point format with ``ebits`` exponent and ``mbits`` fraction bits.

    ``special`` selects how the top of the encoding space is used:

    ``ieee``  top exponent field reserved for Inf/NaN (fp32, bf16, fp16, fp8_e5m2)
    ``fn``    finite + NaN: top exponent holds normals, all-ones fraction there is NaN (fp8_e4m3)
    ``fnuz``  finite + NaN encoded as negative zero, no -0 (fp8_e4m3fnuz, fp8_e5m2fnuz)
    ``none``  every encoding is a finite number (OCP MX fp6 / fp4)
    """

    name: str
    ebits: int
    mbits: int
    bias: int = field(default=-1)
    special: Special = "ieee"
    signed: bool = True
    subnormals: bool = True

    def __post_init__(self) -> None:
        if self.ebits < 1 or self.ebits > 8:
            raise ValueError(f"{self.name}: ebits must be in [1, 8], got {self.ebits}")
        if self.mbits < 0 or self.mbits > 23:
            raise ValueError(f"{self.name}: mbits must be in [0, 23], got {self.mbits}")
        if self.special not in _SPECIALS:
            raise ValueError(f"{self.name}: special must be one of {_SPECIALS}")
        if self.special == "fn" and self.mbits == 0:
            raise ValueError(f"{self.name}: 'fn' needs mbits >= 1 (use Pow2Format for ExM0)")
        if self.bias == -1:
            object.__setattr__(self, "bias", 2 ** (self.ebits - 1) - 1)
        if self.emin < -126 or self.emax > 127:
            raise ValueError(f"{self.name}: exponent range [{self.emin}, {self.emax}] exceeds fp32")

    kind = "float"

    @property
    def bits(self) -> int:
        return self.ebits + self.mbits + (1 if self.signed else 0)

    @property
    def precision(self) -> int:
        """Significand bits including the implicit leading one."""
        return self.mbits + 1

    @property
    def emin(self) -> int:
        """Exponent of the smallest normal number."""
        return 1 - self.bias

    @property
    def emax(self) -> int:
        """Exponent of the largest normal number."""
        top = 2**self.ebits - 1
        if self.special == "ieee":
            top -= 1
        return top - self.bias

    @property
    def max_normal(self) -> float:
        # fn reserves the all-ones fraction at the top exponent for NaN.
        lost = self.mbits - 1 if self.special == "fn" else self.mbits
        return (2.0 - 2.0**-lost) * 2.0**self.emax

    @property
    def min_normal(self) -> float:
        return 2.0**self.emin

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (self.emin - self.mbits) if self.subnormals else self.min_normal

    @property
    def has_inf(self) -> bool:
        return self.special == "ieee"

    @property
    def has_nan(self) -> bool:
        return self.special != "none"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class IntFormat:
    """Integer (optionally fixed-point) format: ``value = k * 2**-frac_bits``.

    Signed ``symmetric`` formats use ``k`` in ``[-(2**(bits-1) - 1), 2**(bits-1) - 1]``
    (sign-magnitude range, the microxcaling / MX convention); otherwise two's
    complement ``[-2**(bits-1), 2**(bits-1) - 1]``. Unsigned: ``[0, 2**bits - 1]``.
    """

    name: str
    bits: int
    signed: bool = True
    symmetric: bool = True
    frac_bits: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 24:
            raise ValueError(f"{self.name}: bits must be in [1, 24], got {self.bits}")
        if self.signed and self.bits < 2:
            raise ValueError(f"{self.name}: signed ints need bits >= 2")

    kind = "int"

    @property
    def qmin(self) -> int:
        if not self.signed:
            return 0
        return -(2 ** (self.bits - 1)) + (1 if self.symmetric else 0)

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.signed else 2**self.bits - 1

    @property
    def max_normal(self) -> float:
        return self.qmax * 2.0**-self.frac_bits

    @property
    def min_normal(self) -> float:
        return 2.0**-self.frac_bits

    @property
    def magnitude_bits(self) -> int:
        """Bits needed for ``|k|`` (the significand width seen by an MMA datapath)."""
        return self.bits - 1 if self.signed else self.bits

    has_inf = False
    has_nan = False

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Pow2Format:
    """Unsigned power-of-two format: ``value = 2**(field - bias)``.

    OCP MX E8M0: ``ebits=8``, ``bias=127``; field 255 is NaN, there is no zero,
    so representable exponents are ``[-127, 127]``.
    """

    name: str = "e8m0"
    ebits: int = 8
    bias: int = 127

    kind = "pow2"

    @property
    def emin(self) -> int:
        return -self.bias

    @property
    def emax(self) -> int:
        return 2**self.ebits - 2 - self.bias

    @property
    def max_normal(self) -> float:
        return 2.0**self.emax

    @property
    def min_normal(self) -> float:
        return 2.0**self.emin

    has_inf = False
    has_nan = True
    signed = False

    def __str__(self) -> str:
        return self.name


Format = FloatFormat | IntFormat | Pow2Format

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

FP32 = FloatFormat("fp32", 8, 23)
TF32 = FloatFormat("tf32", 8, 10)
BF16 = FloatFormat("bf16", 8, 7)
FP16 = FloatFormat("fp16", 5, 10)
FP8_E4M3 = FloatFormat("fp8_e4m3", 4, 3, special="fn")
FP8_E5M2 = FloatFormat("fp8_e5m2", 5, 2)
FP8_E4M3FNUZ = FloatFormat("fp8_e4m3fnuz", 4, 3, bias=8, special="fnuz")
FP8_E5M2FNUZ = FloatFormat("fp8_e5m2fnuz", 5, 2, bias=16, special="fnuz")
FP6_E3M2 = FloatFormat("fp6_e3m2", 3, 2, special="none")
FP6_E2M3 = FloatFormat("fp6_e2m3", 2, 3, special="none")
FP4_E2M1 = FloatFormat("fp4_e2m1", 2, 1, special="none")
UE4M3 = FloatFormat("ue4m3", 4, 3, special="fn", signed=False)
E8M0 = Pow2Format("e8m0")
INT8 = IntFormat("int8", 8)
INT4 = IntFormat("int4", 4)
INT2 = IntFormat("int2", 2)
UINT8 = IntFormat("uint8", 8, signed=False)
UINT4 = IntFormat("uint4", 4, signed=False)
MXINT8 = IntFormat("mxint8", 8, frac_bits=6)
MXINT4 = IntFormat("mxint4", 4, frac_bits=2)

REGISTRY: dict[str, Format] = {
    f.name: f
    for f in (
        FP32, TF32, BF16, FP16, FP8_E4M3, FP8_E5M2, FP8_E4M3FNUZ, FP8_E5M2FNUZ,
        FP6_E3M2, FP6_E2M3, FP4_E2M1, UE4M3, E8M0,
        INT8, INT4, INT2, UINT8, UINT4, MXINT8, MXINT4,
    )
}

ALIASES: dict[str, str] = {
    "float32": "fp32", "float": "fp32", "bfloat16": "bf16", "float16": "fp16", "half": "fp16",
    "e4m3": "fp8_e4m3", "e4m3fn": "fp8_e4m3", "fp8": "fp8_e4m3", "e5m2": "fp8_e5m2",
    "e4m3fnuz": "fp8_e4m3fnuz", "e5m2fnuz": "fp8_e5m2fnuz",
    "e3m2": "fp6_e3m2", "e2m3": "fp6_e2m3", "e2m1": "fp4_e2m1", "fp4": "fp4_e2m1",
    "ue8m0": "e8m0",
}

_EXMY = re.compile(r"^(u)?e(\d+)m(\d+)$")
_INTN = re.compile(r"^(u)?int(\d+)$")


def get_format(spec: str | Format | dict) -> Format:
    """Resolve a format from a name, a ``Format`` instance or a dict.

    String grammar (case-insensitive, options separated by ``:``)::

        fp8_e4m3 | e4m3 | bf16 | fp4 | e8m0 ...          registered names / aliases
        e3m4[:ieee|fn|fnuz|none][:bias=N][:nosub]         custom float; 'u' prefix = unsigned
        int5 | uint4 | int8:full | int8:frac=6            custom int ('full' = two's complement)

    A custom ExMy defaults to ``special='ieee'`` when ``E >= 5`` and ``'none'``
    otherwise (the OCP MX convention for narrow formats).
    """
    if isinstance(spec, (FloatFormat, IntFormat, Pow2Format)):
        return spec
    if isinstance(spec, dict):
        return _from_dict(spec)
    if not isinstance(spec, str):
        raise TypeError(f"cannot interpret {spec!r} as a format")

    raw, *opts = [p.strip() for p in spec.strip().lower().split(":")]
    head = ALIASES.get(raw, raw)
    if head in REGISTRY and not opts:
        return REGISTRY[head]

    kv = dict(o.split("=", 1) for o in opts if "=" in o)
    flags = {o for o in opts if "=" not in o}
    head = raw  # custom grammar is matched on what the user wrote (e.g. "e4m3:nosub")

    m = _EXMY.match(head)
    if m:
        _check_options(spec, flags, kv, {*_SPECIALS, "nosub"}, {"bias"})
        if len(flags & set(_SPECIALS)) > 1:
            raise ValueError(f"{spec!r}: choose one of {_SPECIALS}")
        unsigned, e, mb = m.group(1) is not None, int(m.group(2)), int(m.group(3))
        if e >= 1 and mb == 0 and not flags & set(_SPECIALS):
            return Pow2Format(name=spec, ebits=e, bias=int(kv.get("bias", 2 ** (e - 1) - 1)))
        special = next((f for f in flags if f in _SPECIALS), "ieee" if e >= 5 else "none")
        bias = int(kv["bias"]) if "bias" in kv else -1
        return FloatFormat(spec, e, mb, bias=bias, special=special,
                           signed=not unsigned, subnormals="nosub" not in flags)
    m = _INTN.match(head)
    if m:
        _check_options(spec, flags, kv, {"full"}, {"frac"})
        unsigned, bits = m.group(1) is not None, int(m.group(2))
        return IntFormat(spec, bits, signed=not unsigned, symmetric="full" not in flags,
                         frac_bits=int(kv.get("frac", 0)))
    if head in REGISTRY:
        raise ValueError(f"options {opts} are not supported on registered format {head!r}")
    raise ValueError(f"unknown format {spec!r}; registered: {sorted(REGISTRY)}")


def _check_options(spec: str, flags: set, kv: dict, allowed_flags: set, allowed_keys: set) -> None:
    unknown = sorted(flags - allowed_flags) + sorted(set(kv) - allowed_keys)
    if unknown:
        raise ValueError(f"{spec!r}: unsupported option(s) {unknown}; allowed: "
                         f"{sorted(allowed_flags)} and {sorted(k + '=' for k in allowed_keys)}")


def _from_dict(d: dict) -> Format:
    d = dict(d)
    kind = d.pop("kind", "float")
    if kind == "float":
        return FloatFormat(**d)
    if kind == "int":
        return IntFormat(**d)
    if kind == "pow2":
        return Pow2Format(**d)
    raise ValueError(f"unknown format kind {kind!r}")


def format_of_dtype(dtype) -> FloatFormat:
    """The FloatFormat whose grid equals a torch floating dtype."""
    import torch

    table = {torch.float32: FP32, torch.bfloat16: BF16, torch.float16: FP16}
    for name, fmt in (("float8_e4m3fn", FP8_E4M3), ("float8_e5m2", FP8_E5M2),
                      ("float8_e4m3fnuz", FP8_E4M3FNUZ), ("float8_e5m2fnuz", FP8_E5M2FNUZ)):
        if hasattr(torch, name):
            table[getattr(torch, name)] = fmt
    if dtype not in table:
        raise ValueError(f"no TriCast format for dtype {dtype}")
    return table[dtype]


def container_dtype(fmt: Format):
    """Smallest torch float dtype that holds every value of ``fmt`` exactly."""
    import torch

    for dtype, grid in ((torch.bfloat16, BF16), (torch.float16, FP16)):
        if _fits(fmt, grid):
            return dtype
    return torch.float32


def _fits(fmt: Format, grid: FloatFormat) -> bool:
    """True when every value of ``fmt`` is exactly representable in ``grid``.

    A value needs (a) no more significant bits than ``grid`` keeps, (b) an MSB
    exponent ``<= grid.emax`` and (c) an LSB exponent no finer than grid's
    smallest subnormal quantum ``grid.emin - grid.mbits``.
    """
    finest = grid.emin - grid.mbits
    if isinstance(fmt, FloatFormat):
        return (fmt.mbits <= grid.mbits and fmt.emax <= grid.emax
                and fmt.emin - fmt.mbits >= finest)
    if isinstance(fmt, IntFormat):
        span = max(abs(fmt.qmin), abs(fmt.qmax)).bit_length()
        return (span <= grid.precision and -fmt.frac_bits >= finest
                and fmt.max_normal <= grid.max_normal)
    return fmt.emin >= finest and fmt.emax <= grid.emax
