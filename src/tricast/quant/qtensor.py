"""Quantized values and the scale layout needed to reconstruct them."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mma.operand import Operand
from ..reference.cast import round_to_format
from ..rounding import Rounding
from .spec import QuantSpec


@dataclass
class QTensor:
    values: torch.Tensor
    scale: torch.Tensor | None
    zero_point: torch.Tensor | None
    global_scale: torch.Tensor | None
    spec: QuantSpec
    shape: tuple[int, ...]

    @property
    def rows(self) -> int:
        return self.values.shape[0]

    @property
    def K(self) -> int:
        return self.values.shape[1]

    def scale_per_element(self) -> torch.Tensor | None:
        """Expand the first-level scales to [rows, K]."""
        from ..reference.quantize import expand_scale

        if self.scale is None:
            return None
        return expand_scale(self.scale, self.spec, self.rows, self.K)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct the original shape, preserving fp32 operation order."""
        from ..reference.quantize import _add, _mul, _sub, expand_scale

        x = self.values.float()
        zero = None
        if self.zero_point is not None:
            zero = expand_scale(self.zero_point, self.spec, self.rows, self.K)
            if self.spec.zero_point == "int":
                x = _sub(x, zero)
        if self.scale is not None:
            x = _mul(x, self.scale_per_element())
        if zero is not None and self.spec.zero_point == "float":
            x = _add(x, zero)
        if self.global_scale is not None:
            x = _mul(x, self.global_scale)
        return x.reshape(self.shape).to(dtype)

    def mma_operand(self) -> Operand:
        """Build the scaled or pre-dequantized input to an MMA datapath."""
        spec = self.spec
        if spec.mma_input == "dequant":
            values = round_to_format(self.dequantize().reshape(self.rows, self.K), spec.dequant_format,
                                     Rounding.RNE, saturate=False)
            return Operand(values, spec.dequant_format)
        if self.scale is None:
            return Operand(self.values.float(), spec.format)
        scale, kind, width = self.scale, spec.granularity, 0
        if kind == "group":
            kind, width = "k", spec.group_size
        elif kind == "block":
            kind, width = "k", spec.block[1]
            scale = scale.repeat_interleave(spec.block[0], dim=0)[:self.rows]
        return Operand(self.values.float(), spec.format, scale, spec.scale.format,
                       kind, width, self.global_scale)

    def to(self, device: torch.device | str) -> QTensor:
        """Move all tensor fields to a device without changing the format."""
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device)

        return QTensor(move(self.values), move(self.scale), move(self.zero_point),
                       move(self.global_scale), self.spec, self.shape)
