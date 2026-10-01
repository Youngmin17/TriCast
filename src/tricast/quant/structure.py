"""Weight structure specs: sparsity and outlier preservation, applied before quantization.

``EmuLinear`` prunes the (transformed) weight, then moves its outliers out of the weight that
gets quantized, so the quantization scales see neither. Selection only compares magnitudes, so
it is exact on every device; ``torch.sort`` ranks NaN as the largest magnitude, and stable sorts
make the lower index win every tie.

Spec errors name the offending field first (``n: ...``); recipe loading prefixes the recipe path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from ..formats import BF16, FloatFormat, Format, get_format

if TYPE_CHECKING:
    import torch

SparsityKind = Literal["none", "n:m", "unstructured"]


@dataclass(frozen=True)
class SparsitySpec:
    """Which weights are pruned to exact zeros (unused fields stay 0).

    ``none``          keep every weight
    ``n:m``           in each output row, keep the ``n`` largest ``|w|`` of every ``m`` consecutive
                      weights along K (1 <= n < m <= 64); ties keep the lower K index, and a partial
                      last group of length L keeps min(n, L)
    ``unstructured``  zero the floor(ratio * numel) smallest ``|w|`` of the whole tensor
                      (0 < ratio < 1); ties prune the lower flat index first
    """

    kind: SparsityKind = "none"
    n: int = 0
    m: int = 0
    ratio: float = 0.0

    def __post_init__(self) -> None:
        for name in ("n", "m"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name}: expected an integer, got {value!r}")
        uses = {"none": (), "n:m": ("n", "m"), "unstructured": ("ratio",)}
        if self.kind not in uses:
            raise ValueError(f"kind: expected none, n:m or unstructured, got {self.kind!r}")
        for name in ("n", "m", "ratio"):
            used = name in uses[self.kind]
            if used != bool(getattr(self, name)):
                raise ValueError(f"{name}: {'required' if used else 'not used'} by kind {self.kind!r}")
        if self.kind == "n:m":
            if not 2 <= self.m <= 64:
                raise ValueError(f"m: expected 2 <= m <= 64, got {self.m!r}")
            if not 1 <= self.n < self.m:
                raise ValueError(f"n: expected 1 <= n < m = {self.m}, got {self.n!r}")
        if self.kind == "unstructured":
            if not 0 < self.ratio < 1:
                raise ValueError(f"ratio: expected 0 < ratio < 1, got {self.ratio!r}")
            object.__setattr__(self, "ratio", float(self.ratio))


@dataclass(frozen=True)
class OutlierSpec:
    """SpQR-style weight outliers kept apart from the quantized weight.

    After sparsity, the ceil(fraction * numel) largest ``|w|`` of the kept weights (numel counts
    the whole tensor; never more than are kept; ties: lower flat index) are zeroed in the weight
    that gets quantized and rounded to ``format`` (RNE) as a separate operand. The layer adds the
    fp32 FMA-chain product of that operand to the fp32 main GEMM once, then rounds to the MMA
    ``out_format``.
    """

    fraction: float
    format: Format = BF16

    def __post_init__(self) -> None:
        if not 0 < self.fraction < 1:
            raise ValueError(f"fraction: expected 0 < fraction < 1, got {self.fraction!r}")
        object.__setattr__(self, "fraction", float(self.fraction))
        try:
            fmt = get_format(self.format)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"format: {exc}") from exc
        if not isinstance(fmt, FloatFormat):
            raise ValueError(f"format: expected a float format, got {fmt.name}")
        object.__setattr__(self, "format", fmt)


def _decimal(fraction: float) -> Fraction:
    """A ratio as the decimal it is written as: 0.29 of 100 is 29 weights, not floor of the
    binary product 28.999..., and 0.005 of 1000 is 5, not ceil of 5.000...1."""
    return Fraction(str(fraction))


def _magnitude(weight: torch.Tensor) -> torch.Tensor:
    """``|w|`` without rounding: fp32 holds every narrower float exactly, fp64 stays fp64."""
    import torch

    return (weight if weight.dtype == torch.float64 else weight.float()).abs()


def sparsity_mask(weight: torch.Tensor, spec: SparsitySpec) -> torch.Tensor:
    """Boolean mask of the weights ``spec`` keeps, shaped like ``weight`` ([..., K])."""
    import torch

    magnitude = _magnitude(weight)
    if spec.kind == "none":
        return torch.ones_like(magnitude, dtype=torch.bool)
    if spec.kind == "n:m":
        k = magnitude.shape[-1]
        rows = magnitude.reshape(-1, k)
        groups = -(-k // spec.m)
        # Padding ranks below every |w|, so a partial last group of length L keeps min(n, L).
        padded = torch.full((rows.shape[0], groups * spec.m), -1.0, dtype=rows.dtype, device=rows.device)
        padded[:, :k] = rows
        order = padded.reshape(-1, groups, spec.m).sort(dim=-1, descending=True, stable=True).indices
        keep = torch.zeros_like(order, dtype=torch.bool).scatter_(-1, order[..., :spec.n], True)
        return keep.reshape(rows.shape[0], -1)[:, :k].reshape(magnitude.shape)
    count = math.floor(_decimal(spec.ratio) * magnitude.numel())
    order = magnitude.flatten().sort(stable=True).indices
    keep = torch.ones(magnitude.numel(), dtype=torch.bool, device=magnitude.device)
    keep[order[:count]] = False
    return keep.reshape(magnitude.shape)


def outlier_mask(weight: torch.Tensor, spec: OutlierSpec, keep: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean mask of the outliers among the weights ``keep`` marks (all when ``None``)."""
    import torch

    magnitude = _magnitude(weight)
    available = magnitude.numel()
    if keep is not None:
        magnitude = magnitude.masked_fill(~keep, -1.0)  # pruned weights rank below every kept one
        available = int(keep.sum())
    count = min(math.ceil(_decimal(spec.fraction) * magnitude.numel()), available)
    order = magnitude.flatten().sort(descending=True, stable=True).indices
    mask = torch.zeros(magnitude.numel(), dtype=torch.bool, device=magnitude.device)
    mask[order[:count]] = True
    return mask.reshape(magnitude.shape)
