"""Inference-only Conv2d lowering onto the existing emulated linear arithmetic."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..mma.operand import tensor_state
from ..recipe import LinearSpec
from .linear import EmuLinear


class EmuConv2d(nn.Module):
    """Lower a Conv2d to one EmuLinear per convolution group.

    Each image is unfolded into spatial-patch rows. Within a group, K follows
    PyTorch's input-channel, kernel-height, kernel-width order. Weight rows are
    output channels; activation rows are patches, not original input channels.
    Tensor/block scale domains are local to each group, with activation batch
    isolation identical to EmuLinear. This is an arithmetic emulation, not a
    model of a vendor's implicit-GEMM convolution schedule.

    Only inference with dynamic quantization, RTN weights and no transform is
    supported. Parameters and state_dict names remain those of the Conv2d.
    """

    def __init__(
        self, conv: nn.Conv2d, spec: LinearSpec, name: str = "", backend: str = "auto",
    ) -> None:
        super().__init__()
        if type(conv) is not nn.Conv2d:
            raise NotImplementedError(f"{name}: only plain nn.Conv2d is supported, not custom subclasses")
        if any(getattr(conv, hook) for hook in ("_forward_pre_hooks", "_forward_hooks",
                                               "_backward_pre_hooks", "_backward_hooks")):
            raise NotImplementedError(f"{name}: Conv2d module hooks are not supported; "
                                      "remove hooks before patching")
        if backend not in ("auto", "reference", "triton"):
            raise ValueError(f"{name}.backend: expected auto, reference, or triton")
        if spec.transform.kind != "none":
            raise NotImplementedError(f"{name}.transform: Conv2d transforms are not supported")
        if spec.weight_algo.kind != "rtn":
            raise NotImplementedError(f"{name}.weight_algo: Conv2d supports RTN only, not GPTQ calibration")
        if any(q is not None and q.observer is not None for q in (spec.weight, spec.activation)):
            raise NotImplementedError(f"{name}.observer: Conv2d supports dynamic quantization only")
        if not isinstance(conv.weight, nn.Parameter):
            raise NotImplementedError(f"{name}: parametrized Conv2d weights are not supported")
        if not isinstance(conv.padding, str) and any(p < 0 for p in conv.padding):
            raise ValueError(f"{name}.padding: negative padding is not supported")
        self.spec, self.name, self.backend = spec, name, backend
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size, self.stride = conv.kernel_size, conv.stride
        self.padding, self.dilation, self.groups = conv.padding, conv.dilation, conv.groups
        self.padding_mode = conv.padding_mode
        self._padding = tuple(conv._reversed_padding_repeated_twice)
        self.weight, self.bias = conv.weight, conv.bias
        object.__setattr__(self, "_original", conv)
        # Engines are derived caches, deliberately not registered child modules:
        # Conv2d checkpoint names remain exactly ``weight`` and ``bias``.
        self._engines: list[EmuLinear] = []
        self._weight_state: tuple | None = None
        self._bias_state: tuple | None = None
        self.train(conv.training)
        self.refresh()

    @classmethod
    def from_conv2d(
        cls, conv: nn.Conv2d, spec: LinearSpec, name: str = "", backend: str = "auto",
    ) -> EmuConv2d:
        return cls(conv, spec, name, backend)

    @torch.no_grad()
    def refresh(self) -> None:
        """Rebuild weight caches after edits not tracked by PyTorch (.data/inference_mode)."""
        self._engines = []
        n = self.out_channels // self.groups
        k = self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
        for group in range(self.groups):
            # Do not initialize random weights: patching must not consume RNG state.
            linear = nn.Linear.__new__(nn.Linear)
            nn.Module.__init__(linear)
            linear.in_features, linear.out_features = k, n
            linear.weight = nn.Parameter(
                self.weight[group * n:(group + 1) * n].detach().reshape(n, k), requires_grad=False,
            )
            linear.bias = (None if self.bias is None else nn.Parameter(
                self.bias[group * n:(group + 1) * n].detach(), requires_grad=False,
            ))
            linear.eval()
            self._engines.append(EmuLinear.from_linear(
                linear, self.spec, f"{self.name}.group{group}", self.backend,
            ))
        self._weight_state = tensor_state(self.weight)
        self._bias_state = None if self.bias is None else tensor_state(self.bias)

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        # Retain quantized operands/noise: an unchanged stochastic weight must
        # not be rounded again just because the model is copied or serialized.
        state["_weight_state"] = ("current" if self._weight_state == tensor_state(self.weight) else None)
        state["_bias_state"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        if self._weight_state == "current":
            n = self.out_channels // self.groups
            k = self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
            for group, engine in enumerate(self._engines):
                # Deepcopy/pickle copies private group Parameters independently
                # of the public Conv2d Parameter; bind them back without recast.
                engine.weight = nn.Parameter(
                    self.weight[group * n:(group + 1) * n].detach().reshape(n, k), requires_grad=False,
                )
                engine.bias = (None if self.bias is None else nn.Parameter(
                    self.bias[group * n:(group + 1) * n].detach().float(), requires_grad=False,
                ))
                engine._original.weight, engine._original.bias = engine.weight, engine.bias
                engine._weight_state = tensor_state(engine.weight)
            self._weight_state = tensor_state(self.weight)
            self._bias_state = None if self.bias is None else tensor_state(self.bias)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> EmuConv2d:
        result = super()._apply(fn, recurse=recurse)
        # Derived, unregistered engines must not retain the old device's storage
        # after a dtype/device move. Rebuild lazily against the new Parameters.
        self._engines = []
        self._weight_state = self._bias_state = None
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            raise NotImplementedError(f"{self.name}: Conv2d training is not supported; "
                                      "use eval() or no_grad()")
        if input.ndim not in (3, 4):
            raise ValueError(f"{self.name}: expected CHW or NCHW input, got {input.ndim} dimensions")
        if input.shape[-3] != self.in_channels:
            raise ValueError(f"{self.name}: expected {self.in_channels} input channels, "
                             f"got {input.shape[-3]}")
        if input.device != self.weight.device or input.dtype != self.weight.dtype:
            raise ValueError(f"{self.name}: input and Conv2d weight must have the same device and dtype")
        if self._weight_state != tensor_state(self.weight):
            self.refresh()
        bias_state = None if self.bias is None else tensor_state(self.bias)
        if self._bias_state != bias_state:
            # A bias edit must not redraw stochastic weight rounding noise.
            n = self.out_channels // self.groups
            for group, engine in enumerate(self._engines):
                engine.bias = (None if self.bias is None else nn.Parameter(
                    self.bias[group * n:(group + 1) * n].detach().float(), requires_grad=False,
                ))
            self._bias_state = bias_state
        unbatched = input.ndim == 3
        x = input.unsqueeze(0) if unbatched else input
        if any(self._padding):
            mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            x = F.pad(x, self._padding, mode=mode)
        h = (x.shape[-2] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w = (x.shape[-1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        if h <= 0 or w <= 0:
            raise ValueError(f"{self.name}: kernel size exceeds the padded input size")
        if x.shape[0] == 0:
            return input.new_empty((0, self.out_channels, h, w))
        patches = F.unfold(x, self.kernel_size, dilation=self.dilation, stride=self.stride)
        patches = patches.reshape(x.shape[0], self.groups, -1, h * w)
        outputs = [engine(patches[:, group].transpose(1, 2))
                   for group, engine in enumerate(self._engines)]
        output = torch.cat(outputs, dim=-1).transpose(1, 2).reshape(x.shape[0], self.out_channels, h, w)
        return output.squeeze(0) if unbatched else output

    def extra_repr(self) -> str:
        w = "none" if self.spec.weight is None else str(self.spec.weight.format)
        a = "none" if self.spec.activation is None else str(self.spec.activation.format)
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"dilation={self.dilation}, groups={self.groups}, weight={w}, activation={a}, "
                f"mma={self.spec.mma.algorithm}, backend={self.backend}, inference_only=True")
