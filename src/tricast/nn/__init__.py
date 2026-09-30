"""Emulated linear layers and reversible model patching."""

from .linear import EmuLinear
from .patch import PatchReport, iter_emulinear, patch_model, unpatch_model

__all__ = ["EmuLinear", "PatchReport", "iter_emulinear", "patch_model", "unpatch_model"]
