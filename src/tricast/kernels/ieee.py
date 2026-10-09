"""IEEE binary32 arithmetic that never flushes subnormals.

Triton links libdevice with ``nvvm-reflect-ftz`` enabled, so ``tl.math.div_rn`` and
friends flush subnormal operands and results to zero. The reference follows IEEE
754 (tricast.reference.cast), so every fp32 operation whose operands or result can be
subnormal — scale arithmetic and the MMA epilogue — uses these PTX instructions,
whose non-``.ftz`` forms keep subnormals. Operands must already share a shape.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def div_rn(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def mul_rn(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fma_rn(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def f64_to_f32_rn(a):
    return tl.inline_asm_elementwise("cvt.rn.f32.f64 $0, $1;", "=f,d", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)
