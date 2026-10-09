"""Emulated model arithmetic and reversible, opt-in model patching."""

from .attention import (
    AttentionPatchReport,
    AttentionSpec,
    emulated_attention,
    iter_emuattention,
    patch_attention,
    unpatch_attention,
)
from .conv import EmuConv2d
from .linear import EmuLinear
from .patch import PatchReport, iter_emuconv2d, iter_emulinear, patch_model, unpatch_model

__all__ = ["EmuConv2d", "EmuLinear", "PatchReport", "iter_emuconv2d", "iter_emulinear", "patch_model",
           "unpatch_model", "AttentionPatchReport", "AttentionSpec", "emulated_attention",
           "iter_emuattention", "patch_attention", "unpatch_attention"]
