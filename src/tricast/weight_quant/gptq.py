"""GPTQ column rounding with fp64 Hessian factorization and error updates."""

from __future__ import annotations

import math

import torch

from ..formats import container_dtype
from ..quant.qtensor import QTensor
from ..quant.spec import QuantSpec
from ..reference.cast import round_to_format
from ..reference.quantize import compute_scale, quantize_elements


def _dequantize_column(
    values: torch.Tensor,
    scale: torch.Tensor | None,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    *,
    float_zero_point: bool = False,
) -> torch.Tensor:
    """Match QTensor's fp32 rounding, including the two NVFP4 multiplies."""
    out = values.float()
    if zero_point is not None and not float_zero_point:
        out = (out.double() - zero_point.double()).float()
    if scale is not None:
        out = (out.double() * scale.double()).float()
    if global_scale is not None:
        out = (out.double() * global_scale.double()).float()
    if zero_point is not None and float_zero_point:
        out = (out.double() + zero_point.double()).float()
    return out


def _group_scale(
    weights: torch.Tensor, spec: QuantSpec, global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Choose a group's scales while keeping its original two-level decode scale."""
    sf = spec.scale
    if global_scale is None or sf.method in ("pow2_floor", "pow2_ceil"):
        scale, zp, _ = compute_scale(weights, spec)
        return scale, zp
    # compute_scale has no fixed-global-scale argument (ENGINE §3.12).
    data = weights.float()
    maximum = data.abs().amax(dim=1, keepdim=True)
    if sf.method == "percentile":
        maximum = torch.quantile(data.double().abs(), sf.percentile / 100, dim=1, keepdim=True).float()
    ratios = (sf.search or torch.linspace(1.0, 0.5, sf.mse_grid).tolist()) if sf.method == "mse" else (1.0,)
    best_scale, best_error = None, None
    for ratio in ratios:
        ratio_fp32 = torch.tensor(ratio, device=weights.device, dtype=torch.float32)
        amax = (maximum.double() * ratio_fp32.double()).float()
        raw = (amax.double() / spec.format.max_normal).float()
        raw = (raw.double() / global_scale.double()).float()
        scale = round_to_format(raw, sf.format, sf.rounding, saturate=True)
        minimum = getattr(sf.format, "min_subnormal", sf.format.min_normal)
        scale = torch.where(scale == 0, minimum, scale)
        scale = torch.where(amax == 0, 1.0, scale)
        if sf.method != "mse":
            return scale, None
        q = quantize_elements(data, spec, scale, global_scale=global_scale)
        restored = _dequantize_column(q, scale, None, global_scale)
        error = (data.double() - restored.double()).square().sum(dim=1, keepdim=True)
        if best_error is None:
            best_scale, best_error = scale, error
        else:
            better = error < best_error
            best_scale = torch.where(better, scale, best_scale)
            best_error = torch.where(better, error, best_error)
    return best_scale, None


@torch.no_grad()
def gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    spec: QuantSpec,
    *,
    block_size: int = 128,
    damp: float = 0.01,
    act_order: bool = False,
) -> QTensor:
    """Quantize ``[N, K]`` weights using ``H = 2 X.T @ X / n``.

    Tensor/row scales and the two-level global scale are fixed from the original
    weights. Groups use current compensated weights, except that act-order fixes
    groups before permutation so the output retains the regular QTensor layout.
    """
    if W.ndim != 2 or min(W.shape) == 0 or not W.is_floating_point():
        raise ValueError("W must be a nonempty floating-point matrix [N, K]")
    columns = W.shape[1]
    if H.shape != (columns, columns):
        raise ValueError("H must have shape [K, K]")
    if block_size < 1 or not isinstance(block_size, int):
        raise ValueError("block_size must be a positive integer")
    if not math.isfinite(damp) or damp < 0:
        raise ValueError("damp must be finite and nonnegative")
    if spec.granularity == "block":
        raise ValueError("GPTQ does not support 2-D block granularity")
    if not torch.isfinite(W).all() or not torch.isfinite(H).all():
        raise ValueError("W and H must be finite")
    if not torch.equal(H, H.T) or (H.diagonal() < 0).any():
        raise ValueError("H must be a symmetric positive semidefinite Hessian")

    original = W.detach().float()
    work = W.detach().double().clone()
    hessian = H.detach().to(device=W.device, dtype=torch.float64).clone()
    grouped = spec.scale is not None and spec.granularity == "group"
    scale, zero_point, global_scale = compute_scale(original, spec)

    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    work[:, dead] = 0
    if act_order:
        order = torch.argsort(hessian.diagonal(), descending=True, stable=True)
        work = work[:, order]
        hessian = hessian[order][:, order]
    else:
        order = torch.arange(columns, device=W.device)
    hessian.diagonal().add_(damp * hessian.diagonal().mean())
    try:
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
        upper = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError as exc:
        raise ValueError("damped H must be positive definite; increase damp") from exc

    values = torch.empty_like(original, dtype=container_dtype(spec.format))
    for start in range(0, columns, block_size):
        end = min(start + block_size, columns)
        block = work[:, start:end].clone()
        errors = torch.zeros_like(block)
        for offset in range(end - start):
            column = start + offset
            original_column = int(order[column])
            if grouped:
                group = original_column // spec.group_size
                if not act_order and column % spec.group_size == 0:
                    group_end = min(column + spec.group_size, columns)
                    current = block[:, offset : min(group_end, end) - start]
                    if group_end > end:
                        # Pending lazy updates also belong to a group crossing a batch boundary.
                        tail = work[:, end:group_end].clone()
                        tail -= errors[:, :offset] @ upper[start:column, end:group_end]
                        current = torch.cat((current, tail), dim=1)
                    group_scale, group_zp = _group_scale(current, spec, global_scale)
                    scale[:, group : group + 1] = group_scale
                    if zero_point is not None:
                        zero_point[:, group : group + 1] = group_zp
                column_scale = scale[:, group : group + 1]
                column_zp = None if zero_point is None else zero_point[:, group : group + 1]
            else:
                column_scale, column_zp = scale, zero_point

            weight = block[:, offset : offset + 1]
            noise = (torch.randint(0, 2**32, weight.shape, dtype=torch.int64).to(weight.device)
                     if spec.rounding.value == "sr" else None)
            rounded = quantize_elements(weight, spec, column_scale, column_zp, global_scale, noise)
            values[:, original_column : original_column + 1] = rounded
            restored = _dequantize_column(rounded, column_scale, column_zp, global_scale,
                                          float_zero_point=spec.zero_point == "float")
            error = (weight - restored.double()) / upper[column, column]
            block[:, offset:] -= error * upper[column, column:end]
            errors[:, offset] = error[:, 0]
        work[:, end:] -= errors @ upper[start:end, end:]

    return QTensor(
        values=values,
        scale=scale,
        zero_point=zero_point,
        global_scale=global_scale,
        spec=spec,
        shape=tuple(W.shape),
    )
