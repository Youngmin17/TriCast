"""One FP8 dot product under four accumulators (the README's opening table).

Runs the exact reference on CPU:  python examples/one_dot_product.py
"""

import torch

from tricast import FP8_E4M3, FP32, MMASpec, get_preset
from tricast.mma.operand import Operand
from tricast.reference.cast import round_to_format
from tricast.reference.mma import gemm_reference

g = torch.Generator().manual_seed(196)
K = 32
a = round_to_format(torch.randn(1, K, generator=g) * torch.logspace(0, 2, K)[torch.randperm(K, generator=g)],
                    FP8_E4M3)
b = round_to_format(torch.randn(1, K, generator=g), FP8_E4M3)

hopper = get_preset("nvidia_hopper_fp8")
accumulators = {
    "fp64, one rounding": MMASpec("fp64", out_format=FP32),
    "Blackwell FP8   CoFDA F=25": get_preset("nvidia_blackwell_fp8").with_(out_format=FP32),
    "Hopper FP8      CoFDA F=13": hopper.with_(out_format=FP32),
    "narrow          CoFDA F=7": hopper.with_(f_bits=7, out_format=FP32),
}
for name, spec in accumulators.items():
    value = gemm_reference(Operand(a, FP8_E4M3), Operand(b, FP8_E4M3), spec)
    bits = value.view(torch.int32).item() & 0xFFFFFFFF
    print(f"{name:32s} {value.item()!r:<24} 0x{bits:08x}")
