"""TriCast — emulate number formats, quantization schemes and MMA accumulation
arithmetic on CUDA cores (Triton), and evaluate Hugging Face models under them.

The spec layer (formats, rounding, quantization and MMA specs) imports eagerly;
everything that needs torch kernels, transformers or lm-eval loads on first use.
"""

from importlib import import_module

from .formats import (
    BF16,
    E8M0,
    FP4_E2M1,
    FP6_E2M3,
    FP6_E3M2,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    FP32,
    INT4,
    INT8,
    MXINT8,
    TF32,
    UE4M3,
    FloatFormat,
    Format,
    IntFormat,
    Pow2Format,
    get_format,
)
from .mma.spec import MMASpec, get_preset
from .quant.spec import (
    ObserverSpec,
    QuantSpec,
    ScaleSpec,
    TransformSpec,
    WeightAlgoSpec,
    bfp,
    get_scheme,
)
from .rounding import Rounding

__version__ = "0.1.0.dev0"

_LAZY = {
    "round_to_format": "tricast.reference.cast",
    "quantize": "tricast.quant.api",
    "fake_quant": "tricast.quant.api",
    "QTensor": "tricast.quant.qtensor",
    "gemm": "tricast.mma.api",
    "Recipe": "tricast.recipe",
    "load_recipe": "tricast.recipe",
    "EmuLinear": "tricast.nn.linear",
    "patch_model": "tricast.nn.patch",
    "calibrate": "tricast.calibration",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'tricast' has no attribute {name!r}")


__all__ = [
    "BF16", "E8M0", "FP4_E2M1", "FP6_E2M3", "FP6_E3M2", "FP8_E4M3", "FP8_E5M2", "FP16", "FP32",
    "INT4", "INT8", "MXINT8", "TF32", "UE4M3", "FloatFormat", "Format", "IntFormat", "Pow2Format",
    "get_format", "MMASpec", "get_preset", "ObserverSpec", "QuantSpec", "ScaleSpec", "TransformSpec",
    "WeightAlgoSpec", "bfp", "get_scheme", "Rounding", *_LAZY,
]
