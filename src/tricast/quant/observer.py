"""Calibration statistics and delayed per-tensor scaling."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import torch

from .spec import ObserverSpec, QuantSpec


class ObserverState:
    """Track calibration statistics without retaining the autograd graph."""

    def __init__(self, spec: ObserverSpec) -> None:
        if spec.max_samples < 1 or not 0 < spec.percentile <= 100:
            raise ValueError("observers need max_samples >= 1 and percentile in (0, 100]")
        self.spec = spec
        self.mode: Literal["calibrate", "frozen"] = "calibrate"
        self.amax: torch.Tensor | None = None
        self.count = 0
        self.history: list[torch.Tensor] = []
        self.samples = torch.empty(0, dtype=torch.float32)
        self._priorities = torch.empty(0, dtype=torch.float64)
        self._generator = torch.Generator().manual_seed(0)
        self._static_amax: torch.Tensor | None = None

    @property
    def static_amax(self) -> torch.Tensor | None:
        """The statistic fixed by the most recent ``freeze`` call."""
        return self._static_amax

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """Return this call's amax; history uses previous calls in both modes."""
        if self.mode not in ("calibrate", "frozen"):
            raise ValueError(f"unknown observer mode {self.mode!r}")
        if self.mode == "frozen" and self.spec.kind != "history":
            if self._static_amax is None:
                raise RuntimeError("freeze the observer before frozen evaluation")
            return self._static_amax.to(x.device).clone()
        if not x.numel():
            raise ValueError("cannot observe an empty tensor")
        current = x.detach().to(dtype=torch.float32).abs().amax().cpu()
        self.count += 1
        if self.spec.kind == "history":
            used = self._reduce_history() if self.history else current
            self.history.append(current)
            self.history = self.history[-self.spec.history_len :]
            self.amax = current
            return used.to(x.device).clone()
        if self.amax is None:
            self.amax = current
        elif self.spec.kind == "ema":
            self.amax = self.spec.decay * self.amax + (1.0 - self.spec.decay) * current
        else:
            self.amax = torch.maximum(self.amax, current)
        if self.spec.kind in ("percentile", "mse"):
            self._sample(x)
        return current.to(x.device).clone()

    def _reduce_history(self) -> torch.Tensor:
        if self.spec.reduce == "most_recent":
            return self.history[-1]
        return torch.stack(self.history).amax()

    def _sample(self, x: torch.Tensor) -> None:
        values = x.detach().to(device="cpu", dtype=torch.float32).flatten()
        if self.spec.kind == "percentile":
            values = values.abs()
        priorities = torch.rand(values.numel(), dtype=torch.float64, generator=self._generator)
        values = torch.cat((self.samples, values))
        priorities = torch.cat((self._priorities, priorities))
        if values.numel() > self.spec.max_samples:
            # Retaining the smallest independent random keys gives a uniform reservoir.
            keep = priorities.topk(self.spec.max_samples, largest=False).indices
            values, priorities = values[keep], priorities[keep]
        self.samples, self._priorities = values, priorities

    def clear_samples(self) -> None:
        """Release calibration reservoirs without changing frozen or online statistics."""
        self.samples = torch.empty(0, dtype=torch.float32)
        self._priorities = torch.empty(0, dtype=torch.float64)

    def freeze(self, quant_spec: QuantSpec) -> None:
        """Fix the calibration amax; delayed history remains live in frozen mode."""
        if quant_spec.granularity != "tensor" or quant_spec.scale is None:
            raise ValueError("observers need a scaled, per-tensor QuantSpec")
        if self.amax is None:
            raise RuntimeError("observe calibration data before freezing")
        if self.spec.kind == "history":
            chosen = self._reduce_history()
        elif self.spec.kind == "percentile":
            chosen = torch.quantile(self.samples.double(), self.spec.percentile / 100).float()
        elif self.spec.kind == "mse":
            chosen = self._mse_amax(quant_spec)
        else:
            chosen = self.amax
        self._static_amax = chosen.detach().float().clone()
        self.mode = "frozen"

    def _mse_amax(self, quant_spec: QuantSpec) -> torch.Tensor:
        from ..reference.quantize import quantize_reference

        scale = quant_spec.scale
        assert scale is not None
        ratios = scale.search or torch.linspace(1.0, 0.5, scale.mse_grid).tolist()
        amax = self.samples.abs().amax()
        chosen = amax * ratios[0]
        best_error = float("inf")
        for ratio in ratios:
            candidate = amax * ratio
            quantized = quantize_reference(self.samples, quant_spec, amax=candidate).dequantize()
            error = (self.samples.double() - quantized.double()).square().sum().item()
            if error < best_error:
                best_error, chosen = error, candidate
        return chosen

    def state_dict(self) -> dict:
        """Return independent tensors, including reservoir random-generator state."""
        return {
            "spec": asdict(self.spec),
            "mode": self.mode,
            "amax": None if self.amax is None else self.amax.clone(),
            "count": self.count,
            "history": [value.clone() for value in self.history],
            "samples": self.samples.clone(),
            "priorities": self._priorities.clone(),
            "generator_state": self._generator.get_state().clone(),
            "static_amax": None if self._static_amax is None else self._static_amax.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore statistics and subsequent deterministic reservoir sampling."""
        self.spec = ObserverSpec(**state["spec"])
        self.mode = state["mode"]
        self.amax = None if state["amax"] is None else state["amax"].detach().float().cpu().clone()
        self.count = state["count"]
        self.history = [value.detach().float().cpu().clone() for value in state["history"]]
        self.samples = state["samples"].detach().float().cpu().clone()
        self._priorities = state["priorities"].detach().double().cpu().clone()
        self._generator.set_state(state["generator_state"].cpu())
        static = state["static_amax"]
        self._static_amax = None if static is None else static.detach().float().cpu().clone()
