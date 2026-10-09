"""Function-preserving transforms and their calibration statistics.

Sources: Hadamard rotation as in QuaRot (arXiv:2404.00456), SmoothQuant (arXiv:2211.10438) and
AWQ (arXiv:2306.00978).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .quant.spec import QuantSpec, TransformKind, TransformSpec


@dataclass
class CalibStats:
    n_tokens: int
    absmax: torch.Tensor
    absmean: torch.Tensor
    xtx: torch.Tensor | None = None
    samples: torch.Tensor | None = None


class StatsCollector:
    """Accumulate fp64 statistics and a uniform, seeded reservoir of 4096 rows."""

    def __init__(self, K: int, want_xtx: bool, want_samples: bool) -> None:
        if K < 1:
            raise ValueError("K must be positive")
        self.K = K
        self.n_tokens = 0
        self._absmax = torch.zeros(K, dtype=torch.float64)
        self._abs_sum = torch.zeros_like(self._absmax)
        self._xtx = torch.zeros(K, K, dtype=torch.float64) if want_xtx else None
        self._samples = torch.empty(0, K) if want_samples else None
        self._sample_keys = torch.empty(0, dtype=torch.float64)
        self._generator = torch.Generator().manual_seed(0)

    def update(self, x2d: torch.Tensor) -> None:
        if x2d.ndim != 2 or x2d.shape[1] != self.K:
            raise ValueError(f"expected a [tokens, {self.K}] activation tensor")
        if x2d.shape[0] == 0:
            return
        x = x2d.detach().to(torch.float64)
        if self.n_tokens == 0:
            self._absmax = self._absmax.to(x.device)
            self._abs_sum = self._abs_sum.to(x.device)
            if self._xtx is not None:
                self._xtx = self._xtx.to(x.device)
            if self._samples is not None:
                self._samples = self._samples.to(device=x.device, dtype=_working_dtype(x2d))
        if x.device != self._absmax.device:
            raise ValueError("calibration updates must use one device")
        self._absmax = torch.maximum(self._absmax, x.abs().amax(dim=0))
        self._abs_sum += x.abs().sum(dim=0)
        if self._xtx is not None:
            self._xtx += x.T @ x
        self.n_tokens += x.shape[0]
        if self._samples is not None:
            samples = torch.cat((self._samples, x2d.detach().to(self._samples.dtype)))
            keys = torch.cat((self._sample_keys, torch.rand(x.shape[0], generator=self._generator,
                                                         dtype=torch.float64)))
            # IID priorities give a uniform subset, independent of update boundaries.
            indices = keys.topk(min(4096, keys.numel()), largest=False).indices
            self._sample_keys = keys[indices]
            self._samples = samples[indices.to(samples.device)]

    def result(self) -> CalibStats:
        return CalibStats(
            self.n_tokens,
            self._absmax.clone(),
            self._abs_sum / max(self.n_tokens, 1),
            None if self._xtx is None else self._xtx.clone(),
            None if self._samples is None else self._samples.clone(),
        )


def _working_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _check_block(block: int) -> None:
    if block < 1 or block & (block - 1):
        raise ValueError("Hadamard block must be a positive power of two")


def hadamard(b: int) -> torch.Tensor:
    """Return the normalized Sylvester matrix in fp64 for reference calculations."""
    _check_block(b)
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < b:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(b)


def fwht(x: torch.Tensor, block: int) -> torch.Tensor:
    """Apply a blockwise fast Walsh-Hadamard transform in fp32."""
    return _fwht(x.float(), block)


def _fwht(x: torch.Tensor, block: int) -> torch.Tensor:
    _check_block(block)
    if x.ndim == 0 or x.shape[-1] % block:
        raise ValueError("the last dimension must be divisible by the Hadamard block")
    y = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    width = 1
    while width < block:
        pairs = y.reshape(*y.shape[:-1], block // (2 * width), 2, width)
        left, right = pairs.unbind(dim=-2)
        y = torch.stack((left + right, left - right), dim=-2).reshape(y.shape)
        width *= 2
    return (y / math.sqrt(block)).reshape(x.shape)


def random_sign_diag(K: int, seed: int) -> torch.Tensor:
    """Draw a reproducible ±1 diagonal without changing the global RNG state."""
    if K < 1:
        raise ValueError("K must be positive")
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (K,), generator=generator) * 2 - 1).float()


@dataclass
class LinearTransform:
    kind: TransformKind
    block: int = 0
    diag: torch.Tensor | None = None

    def _apply(self, x: torch.Tensor, *, weight: bool) -> torch.Tensor:
        x = x.to(_working_dtype(x))
        if self.kind == "none":
            return x
        if self.kind == "hadamard":
            return _fwht(x, self.block)
        if self.diag is None or x.ndim == 0 or self.diag.shape != (x.shape[-1],):
            raise ValueError("transform diagonal must match the last dimension")
        diag = self.diag.to(device=x.device, dtype=x.dtype)
        if self.kind == "random_hadamard":
            return _fwht(x * diag, self.block)
        if self.kind in ("smoothquant", "awq"):
            return x * diag if weight else x / diag
        raise ValueError(f"unknown transform {self.kind!r}")

    def apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Return x T."""
        return self._apply(x, weight=False)

    def apply_weight(self, W: torch.Tensor) -> torch.Tensor:
        """Return W T^{-T}."""
        return self._apply(W, weight=True)


def fit_transform(
    spec: TransformSpec,
    W: torch.Tensor,
    stats: CalibStats | None,
    *,
    weight_spec: QuantSpec | None = None,
    act_spec: QuantSpec | None = None,
) -> LinearTransform:
    """Fit a transform; zero AWQ channels use a 1e-5 pre-normalization floor."""
    if W.ndim != 2 or min(W.shape) < 1:
        raise ValueError("weights must be a nonempty [N, K] tensor")
    K = W.shape[1]
    if spec.kind == "none":
        return LinearTransform("none")
    if spec.kind in ("hadamard", "random_hadamard"):
        block = spec.block
        if block == 0:
            block = 128
            while K % block:
                block //= 2
        _check_block(block)
        if K % block:
            raise ValueError("K must be divisible by the Hadamard block")
        diag = random_sign_diag(K, spec.seed).to(W.device) if spec.kind == "random_hadamard" else None
        return LinearTransform(spec.kind, block, diag)
    if stats is None or stats.n_tokens < 1:
        raise ValueError(f"{spec.kind} requires calibration statistics")
    if stats.absmax.shape != (K,) or stats.absmean.shape != (K,):
        raise ValueError("calibration statistics must match K")
    if spec.kind == "smoothquant":
        amax = stats.absmax.to(device=W.device, dtype=torch.float64)
        wmax = W.detach().double().abs().amax(dim=0)
        scale = amax.pow(spec.alpha) / wmax.pow(1 - spec.alpha)
        # A channel absent from both operands needs no scaling.
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1e5).clamp(1e-5, 1e5)
        return LinearTransform(spec.kind, diag=scale)
    return _fit_awq(spec, [W], [stats], stats.absmean, [weight_spec], [act_spec])


def fit_transform_group(
    spec: TransformSpec,
    weights: Sequence[torch.Tensor],
    stats: Sequence[CalibStats],
    *,
    weight_specs: Sequence[QuantSpec | None] | None = None,
    act_specs: Sequence[QuantSpec | None] | None = None,
) -> LinearTransform:
    """Fit one diagonal for shared inputs; AWQ minimizes summed member errors."""
    if spec.kind not in ("smoothquant", "awq"):
        raise ValueError("shared-input groups require smoothquant or awq")
    if not weights or len(weights) != len(stats):
        raise ValueError("weights and calibration statistics must have equal nonzero lengths")
    weight_specs = [None] * len(weights) if weight_specs is None else weight_specs
    act_specs = [None] * len(weights) if act_specs is None else act_specs
    if len(weight_specs) != len(weights) or len(act_specs) != len(weights):
        raise ValueError("quantization specs must match the group size")
    first = weights[0]
    if first.ndim != 2 or min(first.shape) < 1:
        raise ValueError("weights must be nonempty [N, K] tensors")
    K = first.shape[1]
    for weight, stat in zip(weights, stats, strict=True):
        if weight.ndim != 2 or min(weight.shape) < 1 or weight.shape[1] != K:
            raise ValueError("group weights must share K and be nonempty [N, K] tensors")
        if weight.device != first.device:
            raise ValueError("shared-input weights must use one device")
        if stat.n_tokens < 1 or stat.absmax.shape != (K,) or stat.absmean.shape != (K,):
            raise ValueError("group calibration statistics must be nonempty and match K")
    if len(weights) == 1:
        return fit_transform(spec, first, stats[0], weight_spec=weight_specs[0], act_spec=act_specs[0])
    if spec.kind == "smoothquant":
        amax = torch.stack([s.absmax.to(first.device, torch.float64) for s in stats]).amax(dim=0)
        wmax = torch.stack([w.detach().double().abs().amax(dim=0) for w in weights]).amax(dim=0)
        scale = amax.pow(spec.alpha) / wmax.pow(1 - spec.alpha)
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1e5).clamp(1e-5, 1e5)
        return LinearTransform(spec.kind, diag=scale)
    abs_sum = torch.stack([s.absmean.to(first.device, torch.float64) * s.n_tokens for s in stats]).sum(0)
    absmean = abs_sum / sum(s.n_tokens for s in stats)
    return _fit_awq(spec, weights, stats, absmean, weight_specs, act_specs)


def _fit_awq(
    spec: TransformSpec,
    weights: Sequence[torch.Tensor],
    stats: Sequence[CalibStats],
    absmean: torch.Tensor,
    weight_specs: Sequence[QuantSpec | None],
    act_specs: Sequence[QuantSpec | None],
) -> LinearTransform:
    if spec.grid < 1:
        raise ValueError("AWQ grid must be positive")
    from .quant.api import fake_quant

    samples, targets = [], []
    for weight, stat in zip(weights, stats, strict=True):
        if stat.samples is None or stat.samples.ndim != 2 or stat.samples.shape[1] != weight.shape[1]:
            raise ValueError("AWQ requires calibration samples with shape [tokens, K]")
        if stat.samples.shape[0] == 0:
            raise ValueError("AWQ requires nonempty calibration samples")
        x = stat.samples.to(weight.device)
        samples.append(x)
        targets.append(weight.detach().double() @ x.double().T)
    absmean = absmean.to(device=weights[0].device, dtype=torch.float64)
    best_error = math.inf
    best_transform = None
    for alpha in torch.linspace(0, 1, spec.grid, dtype=torch.float64):
        scale = absmean.pow(float(alpha))
        scale = torch.where(scale == 0, 1e-5, scale)
        scale /= (scale.amax() * scale.amin()).sqrt()
        transform = LinearTransform("awq", diag=scale)
        error = 0.0
        for W, x, target, wspec, aspec in zip(
            weights, samples, targets, weight_specs, act_specs, strict=True,
        ):
            weight = transform.apply_weight(W.detach())
            activation = transform.apply_activation(x)
            if wspec is not None:
                weight = fake_quant(weight, wspec, backend="reference")
            if aspec is not None:
                activation = fake_quant(activation, aspec, backend="reference")
            error += float((weight.double() @ activation.double().T - target).square().sum())
        if best_transform is None or error < best_error:
            best_error, best_transform = error, transform
    assert best_transform is not None
    return best_transform
