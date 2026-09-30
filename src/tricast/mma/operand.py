"""MMA operands: what one side of the dot product looks like to the datapath.

Shared by the reference (`tricast.reference.mma`) and the Triton kernels
(`tricast.kernels.mma`); built from a QTensor or a plain tensor by
`tricast.mma.api.as_operand`. Layout rules are ENGINE.md §4.1 and §4.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..formats import Format

ScaleKind = Literal["none", "tensor", "row", "k"]


def tensor_state(tensor: torch.Tensor) -> tuple:
    """What a cache derived from ``tensor`` depends on: which tensor, its storage and its in-place
    version counter. PyTorch does not version edits made through ``.data`` or inside
    ``torch.inference_mode``; after those, rebuild explicitly (``EmuLinear.refresh()``)."""
    version = None if tensor.is_inference() else tensor._version
    return id(tensor), tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), version


@dataclass
class Operand:
    """One MMA input viewed as ``[rows, K]``.

    values     fp32 ``[rows, K]`` grid values of ``fmt`` (already dequantized for
               ``mma_input="dequant"`` operands, whose ``fmt`` is the dequant format)
    scale      fp32 scale values on the ``scale_fmt`` grid, laid out by ``scale_kind``:
               ``tensor`` → shape ``[]``; ``row`` → ``[rows, 1]``;
               ``k`` (K-varying: group or 2-D block) → ``[rows, ceil(K / k_domain)]``,
               the scale of element ``(r, k)`` being ``scale[r, k // k_domain]``
    k_domain   width along K of one scale domain for ``scale_kind == "k"`` (else 0)
    alpha      two-level decode scale ``d2`` (fp32 scalar) applied in the epilogue
    """

    values: torch.Tensor
    fmt: Format
    scale: torch.Tensor | None = None
    scale_fmt: Format | None = None
    scale_kind: ScaleKind = "none"
    k_domain: int = 0
    alpha: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.values.dim() != 2:
            raise ValueError("operand values must be 2-D [rows, K]")
        if (self.scale is None) != (self.scale_kind == "none"):
            raise ValueError("scale and scale_kind disagree")
        if self.scale_kind == "k" and self.k_domain <= 0:
            raise ValueError("K-varying scales need k_domain > 0")

    @property
    def rows(self) -> int:
        return self.values.shape[0]

    @property
    def K(self) -> int:
        return self.values.shape[1]

    def k_major(self) -> torch.Tensor:
        """``values`` as contiguous fp32 ``[K, rows]`` for the Triton kernels. Packed on first use
        and reused while ``values`` is the same, unmodified tensor (weights are not repacked on
        every forward)."""
        state = tensor_state(self.values)
        cached = self.__dict__.get("_k_major")
        if cached is None or cached[0] != state:
            cached = (state, self.values.to(torch.float32).T.contiguous())
            self.__dict__["_k_major"] = cached
        return cached[1]

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_k_major", None)  # derived from values; rebuilt on first use
        return state

    def scale_per_element(self) -> torch.Tensor | None:
        """Scale of every element, ``[rows, K]`` fp32 (``None`` without scales)."""
        if self.scale is None:
            return None
        if self.scale_kind == "k":
            return self.scale.repeat_interleave(self.k_domain, dim=1)[:, : self.K]
        return self.scale.expand(self.rows, self.K)
