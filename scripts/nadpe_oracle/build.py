#!/usr/bin/env python
"""JIT-build the NADPE MMA-Emu operators as a standalone torch extension.

No vLLM involved. Upstream sources are compiled unmodified from --src
(a copy of micro26-ae/csrc/quantization/mma_emu). CPU-only: nvcc does not need a
GPU, and passing an explicit -gencode keeps torch from probing the device.

  python build.py --target fp8   # mma_emu_scaled_fp8_mm      -> nadpe_mma_emu
  python build.py --target fp4   # nvfp4 + mxfp4 entry points -> nadpe_mma_emu_fp4
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
from torch.utils import cpp_extension

HERE = os.path.dirname(os.path.abspath(__file__))

CUDA_FLAGS = [
    "-O2",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_80,code=sm_80",
    # Match vLLM's build (cmake/utils.cmake get_torch_gpu_compiler_flags): for
    # CUDA >= 12.0 it drops these four torch COMMON_NVCC_FLAGS and adds
    # -DENABLE_FP8. torch's JIT always emits the -D forms first, so undefine them
    # here. Without this, scaled_fp8_mm.cuh (bias static_cast<float> of
    # __nv_bfloat16) does not compile.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-DENABLE_FP8",
]

TARGETS = {
    "fp8": ("nadpe_mma_emu", "binding_fp8.cpp", ["fp8_gemm_kernels.cu"]),
    "fp4": ("nadpe_mma_emu_fp4", "binding_fp4.cpp",
            ["nvfp4_gemm_kernels.cu", "mxfp4_gemm_kernels.cu"]),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), default="fp8")
    ap.add_argument("--src", default=os.path.join(HERE, "mma_emu"),
                    help="copy of micro26-ae/csrc/quantization/mma_emu")
    ap.add_argument("--build-dir", default=None)
    args = ap.parse_args()

    name, binding, cu_files = TARGETS[args.target]
    sources = [os.path.join(HERE, binding)] + [os.path.join(args.src, f) for f in cu_files]
    build_dir = args.build_dir or os.path.join(HERE, "build_" + args.target)
    os.makedirs(build_dir, exist_ok=True)

    nvcc = os.path.join(cpp_extension.CUDA_HOME or "", "bin", "nvcc")
    nvcc_ver = [ln for ln in subprocess.run([nvcc, "--version"], capture_output=True,
                                            text=True).stdout.splitlines() if "release" in ln]
    print("BUILD_CFG " + json.dumps({
        "target": args.target, "name": name, "sources": sources, "build_dir": build_dir,
        "cuda_home": cpp_extension.CUDA_HOME, "nvcc": nvcc_ver, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "extra_cuda_cflags": CUDA_FLAGS,
        "common_nvcc_flags": cpp_extension.COMMON_NVCC_FLAGS,
        "src_sha256": {os.path.basename(s): sha256(s) for s in sources},
    }), flush=True)

    t0 = time.time()
    mod = cpp_extension.load(
        name=name,
        sources=sources,
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=CUDA_FLAGS,
        build_directory=build_dir,
        verbose=True,
        is_python_module=True,
    )
    dt = time.time() - t0
    so = mod.__file__
    print("BUILD_OK " + json.dumps({
        "target": args.target, "module": name, "so": so, "so_sha256": sha256(so),
        "so_bytes": os.path.getsize(so), "build_seconds": round(dt, 1),
        "symbols": sorted(k for k in dir(mod) if not k.startswith("__")),
    }), flush=True)


if __name__ == "__main__":
    sys.exit(main())
