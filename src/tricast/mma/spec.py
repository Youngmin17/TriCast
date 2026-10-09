"""MMA accumulation specs: the arithmetic a matrix unit performs on a dot product.

Tensor cores are fixed-function; their alignment width, truncation and the
order in which partial sums meet the accumulator cannot be changed from
software. TriCast runs that arithmetic on CUDA cores instead, so each of those
choices becomes a parameter. The algorithm family follows NADPE (MICRO'26,
"Not All Dot Products Are Equal"; artifact https://doi.org/10.5281/zenodo.21505180);
the executable definition is :mod:`tricast.reference.mma`.

algorithm
    ``cofda``      Chain of Fused-Dot-Add: per chunk of ``chunk_size`` products, align
                   everything to the chunk's max exponent, truncate to ``f_bits``
                   fractional bits (RZ), sum exactly, normalise into an fp32 register
                   keeping ``f_bits`` fraction bits. ``c_mode`` says whether the running
                   accumulator joins that datapath (``fused``) or is merged afterwards at
                   ``f2_bits`` (``decoupled``). CoFDA with chunk = MMA K-width is FDA.
    ``gdfs``       Group-Dot-Fused-Sum: products of each ``group_size`` group summed at
                   ``g_bits``; the ``k_tile / group_size`` group results of a K tile (block
                   scales applied at group level) then go through one FDA at ``f_bits``.
    ``fp32_fma``   IEEE fp32 fused multiply-add chain in K order (a CUDA-core SGEMM).
    ``fp64``       fp64 FMA chain, rounded to fp32 once — the numerical reference.
    ``int_exact``  exact integer sum of integer products (IMMA-style), then fp32.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from ..formats import BF16, FloatFormat, Format, get_format

Algorithm = Literal["cofda", "gdfs", "fp32_fma", "fp64", "int_exact"]

# Where operand scales enter the arithmetic. "auto" resolves from
# the operands: tensor/row scales -> epilogue; K-varying (group/block) scales ->
# group (gdfs), promote (cofda with promote_interval), product (cofda), operand (fma/fp64).
ScaleApply = Literal["auto", "epilogue", "product", "group", "promote", "operand"]

MAX_F_BITS = 48  # keeps every aligned sum inside int64


@dataclass(frozen=True)
class MMASpec:
    algorithm: Algorithm = "cofda"
    f_bits: int = 23
    chunk_size: int = 32
    c_mode: Literal["fused", "decoupled"] = "fused"
    f2_bits: int = 23
    g_bits: int = 32
    group_size: int = 16
    k_tile: int = 64
    norm_rounding: Literal["rtz", "rne"] = "rtz"
    scale_apply: ScaleApply = "auto"
    promote_interval: int = 0
    out_format: Format = BF16
    name: str = ""
    provenance: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "out_format", get_format(self.out_format))
        if self.algorithm not in ("cofda", "gdfs", "fp32_fma", "fp64", "int_exact"):
            raise ValueError(f"unknown algorithm {self.algorithm!r}")
        if self.scale_apply not in ("auto", "epilogue", "product", "group", "promote", "operand"):
            raise ValueError(f"unknown scale_apply {self.scale_apply!r}")
        if self.promote_interval < 0 or (self.promote_interval and self.algorithm != "cofda"):
            raise ValueError("promote_interval (> 0) is a cofda option")
        if self.promote_interval and self.promote_interval % self.chunk_size:
            raise ValueError("promote_interval must be a multiple of chunk_size")
        if not isinstance(self.out_format, FloatFormat):
            raise ValueError("out_format must be a float format")
        if self.algorithm in ("cofda", "gdfs"):
            if not 1 <= self.f_bits <= MAX_F_BITS:
                raise ValueError(f"f_bits must be in [1, {MAX_F_BITS}]")
            if self.norm_rounding not in ("rtz", "rne"):
                raise ValueError("norm_rounding must be 'rtz' or 'rne'")
        if self.algorithm == "cofda":
            if self.chunk_size < 1:
                raise ValueError("chunk_size must be >= 1")
            if self.c_mode not in ("fused", "decoupled"):
                raise ValueError("c_mode must be 'fused' or 'decoupled'")
            if self.c_mode == "decoupled" and not 1 <= self.f2_bits <= MAX_F_BITS:
                raise ValueError(f"f2_bits must be in [1, {MAX_F_BITS}]")
        if self.algorithm == "gdfs":
            if not 1 <= self.g_bits <= MAX_F_BITS:
                raise ValueError(f"g_bits must be in [1, {MAX_F_BITS}]")
            if self.group_size < 1 or self.k_tile % self.group_size:
                raise ValueError("k_tile must be a positive multiple of group_size")
            if not 1 <= self.k_tile // self.group_size <= 8:
                raise ValueError("k_tile / group_size (groups per tile) must be in [1, 8]")

    @property
    def groups_per_tile(self) -> int:
        return self.k_tile // self.group_size

    def with_(self, **changes) -> MMASpec:
        return replace(self, name="", provenance="", **changes)

    @classmethod
    def from_dict(cls, d: dict) -> MMASpec:
        """``{"preset": name, **overrides}`` or explicit fields."""
        d = dict(d)
        if "preset" in d:
            base = get_preset(d.pop("preset"))
            return base.with_(**d) if d else base
        return cls(**d)


_NADPE = (
    "NADPE (MICRO'26, doi:10.5281/zenodo.21505180) "
    "micro26-ae csrc/quantization/mma_emu/core/design_space.cuh"
)

PRESETS: dict[str, MMASpec] = {
    "nvidia_ada_fp8": MMASpec(
        "cofda", f_bits=13, chunk_size=16, name="nvidia_ada_fp8",
        provenance=f"{_NADPE}: Ada CoFDA CS=16 F=13. Not yet checked on Ada silicon by TriCast."),
    "nvidia_hopper_fp8": MMASpec(
        "cofda", f_bits=13, chunk_size=32, name="nvidia_hopper_fp8",
        provenance=f"{_NADPE}: Hopper CS=32 F=13; NADPE Table 6 reports bit-exact scores and "
                   "logprobs vs native H100. Not yet checked on silicon by TriCast."),
    "nvidia_blackwell_fp8": MMASpec(
        "cofda", f_bits=25, chunk_size=32, name="nvidia_blackwell_fp8",
        provenance=f"{_NADPE}: Blackwell FP8 CS=32 F=25. Not yet checked on Blackwell silicon by TriCast."),
    "nvidia_blackwell_fp4": MMASpec(
        "gdfs", f_bits=35, g_bits=6, group_size=16, k_tile=64, name="nvidia_blackwell_fp4",
        provenance=f"{_NADPE}: Blackwell MXFP4/NVFP4 GDFS G=6 F=35 GS=16; K tile 64 is the NVFP4 "
                   "emulator BK in micro26-ae core/tiling.cuh, not a hardware figure. "
                   "Not yet checked on Blackwell silicon by TriCast."),
    "deepseek_fp8_promote128": MMASpec(
        "cofda", f_bits=13, chunk_size=32, promote_interval=128, name="deepseek_fp8_promote128",
        provenance="DeepSeek-V3 report (arXiv:2412.19437) §3.3.2: H800 FP8 WGMMA partials "
                   "(~14-bit accumulation) promoted to fp32 CUDA-core FMA every 128 K elements with block "
                   "scales; F=13 and CS=32 inside each interval are the NADPE Hopper values. "
                   "Not checked on silicon by TriCast."),
    "fp32_fma": MMASpec("fp32_fma", name="fp32_fma",
                        provenance="IEEE fp32 FMA chain in K order (a definition, not a hardware claim)."),
    "fp64": MMASpec("fp64", name="fp64",
                    provenance="fp64 FMA chain, one rounding to fp32 (a definition, not a hardware claim)."),
    "int_exact": MMASpec("int_exact", name="int_exact", provenance="Exact integer accumulation."),
}


def get_preset(name: str | MMASpec) -> MMASpec:
    if isinstance(name, MMASpec):
        return name
    try:
        return PRESETS[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown MMA preset {name!r}; known: {sorted(PRESETS)}") from None
