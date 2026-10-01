"""Linear layers with explicit quantization and emulated accumulation."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from ..formats import FP32
from ..mma.api import as_operand, gemm
from ..mma.operand import Operand, tensor_state
from ..mma.spec import MMASpec
from ..quant.api import fake_quant, quantize
from ..quant.observer import ObserverState
from ..quant.spec import ObserverSpec, QuantSpec, TransformSpec
from ..quant.structure import outlier_mask, sparsity_mask
from ..recipe import LinearSpec
from ..transforms import LinearTransform, StatsCollector, fit_transform
from ..weight_quant import quantize_weight


def _operand_value(operand: Operand) -> torch.Tensor:
    value = operand.values.float()
    scale = operand.scale_per_element()
    if scale is not None:
        value = value * scale
    if operand.alpha is not None:
        value = value * operand.alpha
    return value


def _pack(operand: Operand) -> Operand:
    return replace(operand, values=operand.values.T.contiguous().T)


class _UseValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, value):
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _EmulatedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, weight, bias, result):
        ctx.save_for_backward(activation, weight)
        ctx.has_bias = bias is not None
        return result

    @staticmethod
    def backward(ctx, grad):
        activation, weight = ctx.saved_tensors
        grad = grad.float()
        return grad @ weight, grad.T @ activation, grad.sum(0) if ctx.has_bias else None, None


class HessianAccumulator:
    """One fp64 Gram matrix shared by layers consuming the same inputs."""

    def __init__(self) -> None:
        self.gram: torch.Tensor | None = None

    def update(self, activation: torch.Tensor) -> None:
        values = activation.detach().double()
        gram = values.T @ values
        if self.gram is None:
            self.gram = gram
        else:
            self.gram.add_(gram)


class _DiagonalObserverStats:
    """Retain channel maxima and an element reservoir, never calibration rows."""

    def __init__(self, spec: ObserverSpec) -> None:
        self.spec = spec
        self.maxima: list[torch.Tensor] = []
        self.samples = torch.empty(0, dtype=torch.float32)
        self.channels = torch.empty(0, dtype=torch.long)
        self.priorities = torch.empty(0, dtype=torch.float64)
        self.generator = torch.Generator().manual_seed(0)

    def update(self, rows: torch.Tensor) -> None:
        values = rows.detach().float()
        self.maxima.append(values.abs().amax(dim=0).cpu())
        if self.spec.kind not in ("percentile", "mse"):
            return
        flattened = values.cpu().flatten()
        if self.spec.kind == "percentile":
            flattened = flattened.abs()
        channels = torch.arange(values.shape[1]).repeat(values.shape[0])
        priorities = torch.rand(flattened.numel(), dtype=torch.float64, generator=self.generator)
        samples = torch.cat((self.samples, flattened))
        channels = torch.cat((self.channels, channels))
        priorities = torch.cat((self.priorities, priorities))
        if samples.numel() > self.spec.max_samples:
            keep = priorities.topk(self.spec.max_samples, largest=False).indices
            samples, channels, priorities = samples[keep], channels[keep], priorities[keep]
        self.samples, self.channels, self.priorities = samples, channels, priorities

    def restore(self, transform: LinearTransform, device: torch.device) -> ObserverState:
        reservoir = self.spec.kind in ("percentile", "mse")
        observer = ObserverState(replace(self.spec, kind="minmax") if reservoir else self.spec)
        for maximum in self.maxima:
            # Positive diagonal scaling is monotone: max_i |x_ij / s_j| equals
            # max_i |x_ij| / s_j, including the original fp32 division rounding.
            observer.observe(transform.apply_activation(maximum.to(device).unsqueeze(0)))
        if reservoir:
            observer.spec = self.spec
            diagonal = transform.diag.to(device=device, dtype=torch.float32)
            observer.samples = (self.samples.to(device) /
                                diagonal[self.channels.to(device)]).cpu()
            observer._priorities = self.priorities.clone()
            observer._generator.set_state(self.generator.get_state())
        return observer


class EmuLinear(nn.Module):
    """A cached weight operand, dynamic activations, and an STE backward."""

    def __init__(self, linear: nn.Linear, spec: LinearSpec, name: str = "", backend: str = "auto"):
        super().__init__()
        if backend not in ("auto", "reference", "triton"):
            raise ValueError(f"{name}.backend: expected auto, reference, or triton")
        if spec.weight is not None and spec.weight.observer is not None:
            raise ValueError(f"{name}.weight.observer: static observers are activation-only")
        if spec.weight is None and spec.weight_algo.kind == "gptq":
            raise ValueError(f"{name}.weight_algo: gptq requires a weight QuantSpec")
        self.spec, self.name, self.backend = spec, name, backend
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight
        self.bias = (nn.Parameter(linear.bias.detach().float(), requires_grad=linear.bias.requires_grad)
                     if linear.bias is not None else None)
        object.__setattr__(self, "_original", linear)
        self.mode = "frozen"
        self.observer = (ObserverState(spec.activation.observer)
                         if spec.activation is not None and spec.activation.observer is not None else None)
        self._stats = None
        self._hessian = None
        self._hessian_accumulator: HessianAccumulator | None = None
        self._hessian_owner = True
        self._hessian_replay = False
        self._hessian_rows = 0
        self._calibration_prepared = False
        self._diagonal_observer: _DiagonalObserverStats | None = None
        self._n_rows = 0
        self._calibration_inputs = None
        self._calibration_batches = 0
        self._calibration_passthrough = False
        self._fitted = spec.transform.kind not in ("smoothquant", "awq")
        initial = spec.transform if self._fitted else TransformSpec()
        self.transform = fit_transform(initial, self.weight.detach(), None)
        self._weight_state = None
        self._weight_noise = None
        self._weight_operand = None
        self._outlier_operand = None
        self._calibrated = not self.needs_calibration
        if self._calibrated:
            self.requantize()
        self.train(linear.training)

    @property
    def per_sequence(self) -> bool:
        """Whether a dynamic activation scale spans several tokens (tensor granularity, a two-level
        tensor scale, multi-row blocks, or an online history). Such layers quantize each sequence
        of a batch on its own, so a result does not depend on what else shares the batch."""
        spec = self.spec.activation
        if spec is None or spec.scale is None:
            return False
        if spec.observer is not None and spec.observer.kind != "history":
            return False  # a calibrated static scale is the same for every batch
        return (spec.granularity == "tensor" or spec.scale.two_level
                or (spec.granularity == "block" and spec.block[0] > 1))

    @classmethod
    def from_linear(cls, linear: nn.Linear, spec: LinearSpec, name: str = "", backend: str = "auto"):
        return cls(linear, spec, name, backend)

    @torch.no_grad()
    def requantize(self, hessian: torch.Tensor | None = None) -> None:
        weight = (self.weight.detach() if self.spec.transform.kind == "none"
                  else self.transform.apply_weight(self.weight.detach()))
        weight, outliers, zeros = self._structure(weight)
        self._weight_noise = self._noise(weight, self.spec.weight)
        if self.spec.weight is None:
            operand = as_operand(weight, compact=True)
        elif self.spec.weight_algo.kind == "gptq":
            if hessian is None:
                self._calibrated = False
                raise RuntimeError(f"{self.name}: GPTQ weights changed; call tricast.calibrate(...)")
            qweight = quantize_weight(weight, self.spec.weight, self.spec.weight_algo,
                                      hessian=hessian, backend=self.backend)
            operand = as_operand(qweight, compact=True)
        else:
            qweight = quantize(weight, self.spec.weight, backend=self.backend, noise=self._weight_noise)
            operand = as_operand(qweight, compact=True)
        if zeros is not None:
            # Exact zeros even where 0 does not dequantize to 0 (float zero points).
            operand = replace(operand, values=operand.values.masked_fill(zeros, 0))
        self._weight_operand = _pack(operand)
        self._outlier_operand = None if outliers is None else _pack(as_operand(
            quantize(outliers, QuantSpec(self.spec.outliers.format, scale=None), backend=self.backend),
            compact=True))
        self._weight_state = tensor_state(self.weight)

    def _structure(
        self, weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prune, then split off outliers: the weight to quantize, the outlier weight (or ``None``)
        and the positions that must be exact zeros in the quantized operand (or ``None``)."""
        keep = zeros = None
        if self.spec.sparsity.kind != "none":
            keep = sparsity_mask(weight, self.spec.sparsity)
            zeros = ~keep
            weight = weight.masked_fill(zeros, 0)
        if self.spec.outliers is None:
            return weight, None, zeros
        selected = outlier_mask(weight, self.spec.outliers, keep)
        zeros = selected if zeros is None else zeros | selected
        return weight.masked_fill(selected, 0), weight.masked_fill(~selected, 0), zeros

    @staticmethod
    def _noise(x, spec):
        if spec is not None and spec.rounding.value == "sr":
            return torch.randint(0, 2**32, x.shape, dtype=torch.int64, device=x.device)
        return None

    def _apply(self, fn, recurse=True):
        if self.spec.weight_algo.kind == "gptq" and not self._weight_current():
            self._calibrated = False
        result = super()._apply(fn, recurse=recurse)
        # Bias and all transform arithmetic remain fp32 even after model.half().
        if self.bias is not None:
            self.bias.data = self.bias.data.float()
        if self.spec.weight_algo.kind == "gptq" and self._weight_operand is not None:
            operand = self._weight_operand
            self._weight_operand = _pack(replace(
                operand, values=operand.values.to(self.weight.device),
                scale=None if operand.scale is None else operand.scale.to(self.weight.device),
                alpha=None if operand.alpha is None else operand.alpha.to(self.weight.device),
            ))
            self._weight_state = tensor_state(self.weight)
            if self._weight_noise is not None:
                self._weight_noise = self._weight_noise.to(self.weight.device)
        elif self._calibrated:
            self.requantize()
        return result

    def _weight_current(self) -> bool:
        return self._weight_state is not None and self._weight_state == tensor_state(self.weight)

    def refresh(self) -> None:
        """Rebuild the quantized weight now. Needed after weight edits PyTorch does not version
        (through ``.data`` or inside ``torch.inference_mode``); GPTQ layers need ``calibrate()`` again."""
        self._weight_state = None
        if self.spec.weight_algo.kind == "gptq":
            self._calibrated = False
        elif self._calibrated:
            self.requantize()

    def __getstate__(self) -> dict:
        # The saved state names this object's weight; a copy or pickle keeps only whether it was current.
        state = super().__getstate__()
        state["_weight_state"] = "current" if self._weight_current() else None
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        if self._weight_state == "current":
            self._weight_state = tensor_state(self.weight)

    @property
    def needs_calibration(self) -> bool:
        return (self.spec.transform.kind in ("smoothquant", "awq")
                or self.spec.weight_algo.kind == "gptq"
                or (self.observer is not None and self.observer.spec.kind != "history"))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() >= 3 and input.shape[0] > 1 and self.mode != "calibrate" and self.per_sequence:
            return torch.stack([self._forward(sequence) for sequence in input])
        return self._forward(input)

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if (self.training and torch.is_grad_enabled()
                and (self.spec.sparsity.kind != "none" or self.spec.outliers is not None)):
            raise NotImplementedError(f"{self.name}: sparsity and outliers are inference-only (no STE "
                                      "backward yet); use model.eval() or torch.no_grad()")
        if not self._calibration_passthrough and not self._calibrated:
            raise RuntimeError(f"{self.name}: calibration is required; call tricast.calibrate(...)")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"{self.name}: expected last dimension {self.in_features}, got {x.shape[-1]}")
        if self._calibration_passthrough and self.mode != "calibrate":
            return F.linear(x, self.weight, None if self.bias is None else self.bias.to(x.dtype))
        rows = x.reshape(-1, self.in_features)
        if self.mode == "calibrate" and self._stats is not None:
            diagonal = self.spec.transform.kind in ("smoothquant", "awq")
            if not self._calibration_prepared:
                self._stats.update(rows.detach())
                self._n_rows += rows.shape[0]
                if self._diagonal_observer is not None:
                    self._diagonal_observer.update(rows)
            if self.spec.weight_algo.kind == "gptq" and (
                self._hessian_replay or (not diagonal and not self._calibration_prepared)
            ):
                if self._hessian_owner:
                    # The former diagonal replay stored fp32 rows, including for fp64 inputs.
                    values = rows.detach().float() if diagonal else rows.detach()
                    self._hessian_accumulator.update(self.transform.apply_activation(values))
                self._hessian = self._hessian_accumulator.gram
                self._hessian_rows += rows.shape[0]
        activation = rows if self.spec.transform.kind == "none" else self.transform.apply_activation(rows)
        amax = None
        if self.observer is not None and not (
            self.mode == "calibrate" and
            (self._diagonal_observer is not None or self._calibration_prepared)
        ):
            self.observer.mode = self.mode
            amax = self.observer.observe(activation.detach())
        if self._calibration_passthrough:
            return F.linear(x, self.weight, None if self.bias is None else self.bias.to(x.dtype))
        noise = self._noise(activation, self.spec.activation)
        if self.spec.activation is None:
            operand = as_operand(activation)
        else:
            operand = as_operand(quantize(activation, self.spec.activation, backend=self.backend,
                                         amax=amax, noise=noise))
        if not self._weight_current():
            self.requantize()
        result = self._matmul(operand)
        if torch.is_grad_enabled() and self.training:
            activation_ste = activation
            if self.spec.activation is not None:
                activation_ste = fake_quant(activation, self.spec.activation, backend=self.backend,
                                            amax=amax, noise=noise)
            activation_ste = _UseValue.apply(activation_ste, _operand_value(operand))
            weight_ste = self.transform.apply_weight(self.weight)
            if self.spec.weight is not None:
                weight_ste = fake_quant(weight_ste, self.spec.weight, backend=self.backend,
                                        noise=self._weight_noise)
            weight_ste = _UseValue.apply(weight_ste, _operand_value(self._weight_operand))
            result = _EmulatedLinear.apply(activation_ste, weight_ste, self.bias, result.detach())
        return result.reshape(*x.shape[:-1], self.out_features).to(x.dtype)

    def _matmul(self, operand: Operand, mma: MMASpec | None = None, gemm_fn=None) -> torch.Tensor:
        """The layer's GEMM on an activation operand. ``mma`` replaces the main path's spec (the
        error report swaps in exact accumulation) while the outlier path stays as is; the report
        also passes its own ``gemm_fn`` so its recomputation is not taken for the forward GEMM."""
        mma = self.spec.mma if mma is None else mma
        gemm_fn = gemm if gemm_fn is None else gemm_fn
        if self._outlier_operand is None:
            return gemm_fn(operand, self._weight_operand, mma, bias=self.bias, backend=self.backend)
        return self._gemm_with_outliers(operand, mma, gemm_fn)

    def _gemm_with_outliers(self, operand: Operand, mma: MMASpec, gemm_fn) -> torch.Tensor:
        """Main and outlier GEMMs on one activation operand, each ending in fp32; their sum is
        one IEEE add, rounded once to the MMA output format."""
        main = gemm_fn(operand, self._weight_operand, replace(mma, out_format=FP32), bias=self.bias,
                       backend=self.backend)
        extra = gemm_fn(operand, self._outlier_operand, MMASpec("fp32_fma", out_format=FP32),
                        backend=self.backend)
        result = main + extra
        if mma.out_format == FP32:
            return result
        # saturate=False: the overflow rule of the GEMM epilogue's out_format rounding.
        cast = QuantSpec(mma.out_format, scale=None, saturate=False)
        return quantize(result, cast, backend=self.backend).values

    def begin_calibration(
        self, *, hessian_accumulator: HessianAccumulator | None = None, hessian_owner: bool = True,
    ) -> None:
        self.clear_calibration()
        self.mode = "calibrate"
        self._calibrated = False
        self._calibration_passthrough = True
        self._stats = StatsCollector(self.in_features, want_xtx=False,
                                     want_samples=self.spec.transform.kind == "awq")
        self._hessian, self._n_rows = None, 0
        self._hessian_accumulator = (hessian_accumulator if hessian_accumulator is not None
                                     else HessianAccumulator())
        self._hessian_owner = hessian_owner
        self._hessian_rows = 0
        self._hessian_replay = False
        self._calibration_prepared = False
        if self.observer is not None:
            self.observer = ObserverState(self.spec.activation.observer)
            if self.spec.transform.kind in ("smoothquant", "awq"):
                self._diagonal_observer = _DiagonalObserverStats(self.observer.spec)

    @torch.no_grad()
    def prepare_calibration(self, transform: LinearTransform | None = None) -> None:
        """Fit transforms before an optional exact, bounded-memory GPTQ input replay.

        For a fitted diagonal, ``D^-1 (X.T @ X) D^-1`` does not reproduce the
        fp32-rounded ``X / D`` in ENGINE §3.9. A second pass accumulates that
        exact operational Hessian without retaining or spooling original rows.
        """
        if self._calibration_prepared:
            return
        if not self._n_rows:
            raise ValueError(f"{self.name}: no calibration activations were collected")
        self.transform = transform or fit_transform(
            self.spec.transform, self.weight.detach(), self._stats.result(),
            weight_spec=self.spec.weight, act_spec=self.spec.activation,
        )
        self._fitted = True
        if self._diagonal_observer is not None:
            self.observer = self._diagonal_observer.restore(self.transform, self.weight.device)
        self._hessian_replay = (self.spec.transform.kind in ("smoothquant", "awq")
                                and self.spec.weight_algo.kind == "gptq")
        self._calibration_prepared = True

    @torch.no_grad()
    def finish_calibration(self, transform: LinearTransform | None = None) -> None:
        self.prepare_calibration(transform)
        if self.spec.weight_algo.kind == "gptq" and self._hessian_rows != self._n_rows:
            raise RuntimeError(
                f"{self.name}: fitted-diagonal GPTQ requires exact input replay; call "
                "prepare_calibration(), replay the same calibration batches, then "
                "finish_calibration(), or use tricast.calibrate(...)"
            )
        hessian = self._hessian_accumulator.gram
        hessian = None if hessian is None else hessian * (2.0 / self._n_rows)
        self.requantize(hessian)
        if self.observer is not None:
            self.observer.freeze(self.spec.activation)
            self.observer.clear_samples()
        self._calibrated = True
        self._calibration_passthrough = False
        self.mode = "frozen"
        self.clear_calibration()

    def clear_calibration(self) -> None:
        self._calibration_passthrough = False
        if self.mode == "calibrate":
            self.mode = "frozen"
        self._stats, self._hessian = None, None
        self._hessian_accumulator = None
        self._hessian_replay = False
        self._hessian_rows = 0
        self._calibration_prepared = False
        self._diagonal_observer = None
        self._calibration_inputs = None
        self._calibration_batches = 0

    def extra_repr(self) -> str:
        w = "none" if self.spec.weight is None else str(self.spec.weight.format)
        a = "none" if self.spec.activation is None else str(self.spec.activation.format)
        extra = ""
        if self.spec.sparsity.kind != "none":
            sparsity = self.spec.sparsity
            extra += (f", sparsity={sparsity.n}:{sparsity.m}" if sparsity.kind == "n:m"
                      else f", sparsity=unstructured {sparsity.ratio}")
        if self.spec.outliers is not None:
            extra += f", outliers={self.spec.outliers.fraction} in {self.spec.outliers.format}"
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"weight={w}, activation={a}, mma={self.spec.mma.algorithm}, "
                f"transform={self.spec.transform.kind}{extra}, backend={self.backend}, mode={self.mode}")
