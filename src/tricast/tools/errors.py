"""Small, seeded reference-quantization experiments on synthetic tensors."""

from __future__ import annotations

import math
import platform
import subprocess
from pathlib import Path

import torch

from ..quant.api import quantize
from ..quant.spec import get_scheme

QUANTIZATION_ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "scheme": {"type": "string"},
        "source": {"type": "string", "enum": ["gaussian", "uniform"]},
        "n": {"type": "integer", "minimum": 1, "maximum": 65536},
        "seed": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807},
    },
    "required": ["scheme", "source", "n", "seed"], "additionalProperties": False,
}


def _env() -> dict:
    try:
        root = Path(__file__).resolve().parents[3]
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      cwd=root,
                                      stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=root,
                                            stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        sha = dirty = None
    return {"git_sha": sha, "git_dirty": dirty,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "device": "cpu", "dtype": "float32", "backend": "reference"}


def quantization_error(scheme: str, *, source: str = "gaussian", n: int = 4096, seed: int = 0) -> dict:
    """Measure SQNR of reconstructed values; storage includes all actual scale domains."""
    if type(n) is not int or not 1 <= n <= 65536:
        return {"is_error": True, "error": "n must be an integer in [1, 65536]"}
    if type(seed) is not int or not 0 <= seed <= 9223372036854775807:
        return {"is_error": True, "error": "seed must be an integer in [0, 9223372036854775807]"}
    if source not in ("gaussian", "uniform"):
        return {"is_error": True, "error": "source must be gaussian or uniform"}
    try:
        spec = get_scheme(scheme)
    except (ValueError, TypeError, AttributeError) as exc:
        return {"is_error": True, "error": str(exc)}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn((1, n), generator=generator) if source == "gaussian"
         else torch.rand((1, n), generator=generator) * 2 - 1)
    qt = quantize(x, spec, backend="reference")
    difference = x.double() - qt.dequantize().double()
    signal = x.double().square().sum().item()
    noise = difference.square().sum().item()
    sqnr = 10 * math.log10(signal / noise) if noise > 0 and signal > 0 else None
    scale_bits = 0
    if qt.scale is not None:
        fmt = spec.scale.format
        width = fmt.ebits if fmt.kind == "pow2" else fmt.bits
        scale_bits = qt.scale.numel() * ((width + 7) // 8 * 8)
    global_bits = qt.global_scale.numel() * 32 if qt.global_scale is not None else 0
    zero_point_bits = qt.zero_point.numel() * 32 if qt.zero_point is not None else 0
    return {"is_error": False, "scheme": scheme, "source": source, "n": n, "seed": seed,
            "sqnr_db": sqnr, "sqnr_status": "finite" if sqnr is not None else "infinite",
            "max_abs_err": difference.abs().max().item(),
            "bits_per_element": (n * spec.format.bits + scale_bits + global_bits + zero_point_bits) / n,
            "storage_assumptions": "Packed elements, byte-aligned scale encodings, "
                                   "fp32 global scales and zero points. "
                                   "Observer schemes use the current sample, not calibrated history.",
            "env": _env()}
