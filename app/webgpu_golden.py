"""Golden vectors for the browser (WebGPU) CoFDA path, computed by TriCast's reference backend on the CPU.

Each case is an FP8 E4M3 GEMM ``D = A·Bᵀ`` (A ``[M, K]``, B ``[N, K]`` as e4m3fn codes, per-tensor fp32
scales) under one CoFDA setting, with the expected fp32 output bits of ``tricast.gemm(..., backend=
"reference")`` and of ``MMASpec("fp64", out_format="fp32")`` on the same operands. The browser engine
(``app/web/js/webgpu/engine.js``) and the JS reference (``reference.js``) must reproduce ``expected`` and
``fp64`` bit for bit; ``app/web/webgpu_check.html`` runs that comparison.

    python -m app.webgpu_golden [--out app/web/demo/webgpu/golden.json]

Every case is a pure function of its index (``build_case``), so tests regenerate single cases.
Hex encodings: operand codes row-major, two hex digits per code; scales and outputs as 8-digit fp32 bit
patterns, outputs row-major ``[M, N]``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import tricast
from tricast.formats import FP8_E4M3, FP32
from tricast.mma.api import gemm
from tricast.mma.operand import Operand
from tricast.mma.spec import MMASpec

SEED = 20261007
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "web" / "demo" / "webgpu" / "golden.json"

F_BITS = (1, 3, 7, 13, 23, 25, 30, 48)
CHUNKS = (1, 2, 16, 32, 128)
F2_BITS = (7, 23, 48)
KS = (1, 2, 15, 32, 33, 128, 1024)
PATTERNS = ("uniform", "gauss", "sparse", "subnormal", "maxnormal", "cancel", "nan")  # cycled by the grid
SCALES = ("one", "mild", "tiny", "huge", "negative")


@dataclass(frozen=True)
class Plan:
    f_bits: int
    chunk_size: int
    c_mode: str
    f2_bits: int | None
    norm_rounding: str
    K: int
    M: int
    N: int
    pattern: str
    scales: str
    tag: str


def _grid() -> list[Plan]:
    """Every (F, CS, c_mode, norm_rounding) once; K, pattern, scales and shape cycle along."""
    plans = []
    for f_bits in F_BITS:
        for chunk in CHUNKS:
            for c_mode in ("fused", "decoupled"):
                for rounding in ("rtz", "rne"):
                    i = len(plans)
                    k = KS[i % len(KS)]
                    if chunk <= 2 and k == 1024:
                        k = 128  # keeps the CPU reference fast: at most 128 chunks per output
                    m, n = 1 + i % 3, 1 + (i // 3) % 4
                    if k == 1024:
                        m, n = 1, 1 + i % 3
                    f2_bits = F2_BITS[i % 3] if c_mode == "decoupled" else None
                    pattern = PATTERNS[(i // len(KS)) % len(PATTERNS)]
                    plans.append(Plan(f_bits, chunk, c_mode, f2_bits, rounding, k, m, n, pattern,
                                      SCALES[i % len(SCALES)], "grid"))
    return plans


def _targeted() -> list[Plan]:
    def p(f, cs, mode, f2, rnd, k, m, n, pattern, scales, tag):
        return Plan(f, cs, mode, f2 if mode == "decoupled" else None, rnd, k, m, n, pattern, scales, tag)

    plans = []
    # NaN codes: rows/columns that meet one are NaN, the rest stay finite.
    for i, (f, cs, mode) in enumerate([(13, 32, "fused"), (7, 32, "decoupled"), (1, 1, "fused"),
                                       (48, 128, "fused"), (25, 16, "decoupled"), (3, 2, "decoupled")]):
        for k in (33, 128):
            plans.append(p(f, cs, mode, F2_BITS[i % 3], "rtz" if k == 33 else "rne", k, 2, 3, "nan",
                           SCALES[i % 2], "nan"))
    # Exact cancellation inside and across chunks.
    for i, (f, cs, mode) in enumerate([(13, 32, "fused"), (13, 32, "decoupled"), (7, 1, "fused"),
                                       (7, 2, "decoupled"), (23, 16, "fused"), (48, 32, "decoupled"),
                                       (1, 128, "fused"), (30, 1, "decoupled")]):
        for k in (32, 1024 if cs >= 16 else 128):
            plans.append(p(f, cs, mode, F2_BITS[i % 3], ("rtz", "rne")[i % 2], k, 1, 2, "cancel", "one",
                           "cancel"))
    # Subnormal operands: products whose F-converted significand is 0 still set Emax.
    for i, f in enumerate((1, 2, 3, 5, 7, 13)):
        for mode in ("fused", "decoupled"):
            plans.append(p(f, (1, 2, 16, 32, 128)[i % 5], mode, F2_BITS[i % 3], ("rtz", "rne")[i % 2],
                           (33, 128, 1024)[i % 3], 1, 2, "subnormal", "one", "subnormal"))
    # The top-exponent product truncates to 0 at F <= 2 (Emax still counts it).
    for i, (f, cs) in enumerate(((1, 16), (2, 32), (1, 32), (2, 16))):
        for mode in ("fused", "decoupled"):
            plans.append(p(f, cs, mode, F2_BITS[i % 3], ("rtz", "rne")[i % 2], 128, 2, 3, "emax_mf0", "one",
                           "emax_mf0"))
    # Max-normal operands (±448) over long K.
    for i, (f, cs, mode) in enumerate([(13, 32, "fused"), (7, 32, "decoupled"), (25, 32, "fused"),
                                       (48, 128, "decoupled")]):
        for rnd in ("rtz", "rne"):
            plans.append(p(f, cs, mode, F2_BITS[i % 3], rnd, 1024, 1, 2, "maxnormal", "one", "maxnormal"))
    # Epilogue scales: subnormal and overflowing outputs, negative scales, quantizer-like scales.
    for i, scales in enumerate(("tiny", "huge", "negative", "mild")):
        for j, (f, cs, mode) in enumerate([(13, 32, "fused"), (7, 32, "decoupled"), (23, 16, "fused"),
                                           (25, 32, "fused"), (3, 2, "decoupled"), (48, 128, "fused")]):
            plans.append(p(f, cs, mode, F2_BITS[(i + j) % 3], ("rtz", "rne")[j % 2], (33, 128)[j % 2], 2, 3,
                           ("gauss", "uniform", "sparse")[j % 3], scales, f"scale_{scales}"))
    # Long K with one- and two-product chunks.
    plans.append(p(7, 1, "fused", None, "rtz", 1024, 1, 1, "gauss", "one", "long"))
    plans.append(p(13, 2, "decoupled", 23, "rne", 1024, 1, 1, "gauss", "mild", "long"))
    plans.append(p(48, 1, "fused", None, "rne", 1024, 1, 1, "uniform", "one", "long"))
    # Chunk sizes that are not powers of two (the last chunk is partial).
    for i, cs in enumerate((3, 5, 24, 48, 100, 127, 6, 12)):
        plans.append(p((13, 7, 25, 48)[i % 4], cs, ("fused", "decoupled")[i % 2], F2_BITS[i % 3],
                       ("rtz", "rne")[i % 2], (33, 128, 1024)[i % 3], 1, 2, "gauss", "one", "chunk_npot"))
    # Hardware-preset settings (out_format fp32) on Gaussian-like operands.
    for f, cs, mode, f2 in ((13, 32, "fused", None), (13, 16, "fused", None), (25, 32, "fused", None),
                            (7, 32, "fused", None), (7, 32, "decoupled", 23), (13, 32, "decoupled", 23)):
        for rnd in ("rtz", "rne"):
            plans.append(p(f, cs, mode, f2, rnd, 1024, 1, 2, "gauss", "mild", "preset"))
    # Larger output tiles (indexing).
    for i, (m, n) in enumerate(((5, 7), (4, 9), (7, 3), (3, 17))):
        for mode in ("fused", "decoupled"):
            plans.append(p((13, 23, 7, 30)[i], (16, 32, 2, 128)[i], mode, F2_BITS[i % 3],
                           ("rtz", "rne")[i % 2], (33, 128)[i % 2], m, n, "uniform", "mild", "shape"))
    # Narrow/wide accumulator boundaries of the WGSL kernel and the Number/BigInt boundary of reference.js.
    for i, (f, cs) in enumerate(((21, 128), (22, 128), (24, 32), (24, 16), (29, 1), (30, 2), (43, 128),
                                 (44, 32), (46, 32), (47, 32), (45, 16), (40, 128))):
        for mode in ("fused", "decoupled"):
            k = 128 if cs <= 2 else (128, 1024)[i % 2]
            plans.append(p(f, cs, mode, F2_BITS[i % 3], ("rtz", "rne")[i % 2], k, 1, 2,
                           ("gauss", "maxnormal", "uniform")[i % 3], "one", "boundary"))
    return plans


def plan() -> list[Plan]:
    return _grid() + _targeted()


def _f32(x: float) -> float:
    return torch.tensor(x, dtype=torch.float32).item()


def _bits(x: float) -> str:
    return f"{torch.tensor(x, dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF:08x}"


def _gauss(rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Log-normal-like magnitudes: exponent field around the bias, uniform mantissa, random sign."""
    field = (7 + 2.5 * torch.randn(rows, k, generator=g)).round().clamp(0, 15).to(torch.int64)
    mant = torch.randint(0, 8, (rows, k), generator=g)
    mant = torch.where((field == 15) & (mant == 7), 6, mant)  # 0x7F/0xFF are NaN
    sign = torch.randint(0, 2, (rows, k), generator=g)
    return (sign << 7) | (field << 3) | mant


def _uniform(rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    finite = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F])
    return finite[torch.randint(0, finite.numel(), (rows, k), generator=g)]


def _codes(pattern: str, rows: int, k: int, g: torch.Generator, side: str) -> torch.Tensor:
    if pattern == "uniform":
        codes = _uniform(rows, k, g)
    elif pattern in ("gauss", "cancel", "nan"):
        codes = _gauss(rows, k, g)
    elif pattern == "sparse":
        codes = _uniform(rows, k, g)
        zero = torch.rand(rows, k, generator=g) < 0.6
        codes = torch.where(zero, torch.randint(0, 2, (rows, k), generator=g) << 7, codes)
    elif pattern == "subnormal":
        sub = (torch.randint(0, 2, (rows, k), generator=g) << 7) | torch.randint(0, 8, (rows, k), generator=g)
        codes = torch.where(torch.rand(rows, k, generator=g) < 0.8, sub, _gauss(rows, k, g))
    elif pattern == "maxnormal":
        top = 0x7E | (torch.randint(0, 2, (rows, k), generator=g) << 7)
        codes = torch.where(torch.rand(rows, k, generator=g) < 0.7, top, _gauss(rows, k, g))
    elif pattern == "emax_mf0":
        # The same K positions on both sides (seeded by the case): A holds ±2^-9 (significand 1), B holds ±2^6
        # or ±2^7 (significand 8). Their product has the chunk's top exponent but m = 8, which F <= 2
        # truncates to 0; Emax must still include it.
        marked = torch.rand(k, generator=torch.Generator().manual_seed(g.initial_seed() + 1)) < 0.15
        mark = marked.expand(rows, k)
        bit = lambda: torch.randint(0, 2, (rows, k), generator=g)
        special = (bit() << 7) | (0x01 if side == "a" else 0x68 | (bit() << 3))
        small = (bit() << 7) | (torch.randint(4, 8, (rows, k), generator=g) << 3)  # exponents -3..0
        codes = torch.where(mark, special, small | torch.randint(0, 8, (rows, k), generator=g))
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    if pattern == "cancel" and k >= 2:
        pairs = k // 2 * 2
        if side == "a":  # equal neighbours in A ...
            codes[:, 1:pairs:2] = codes[:, 0:pairs:2]
        else:  # ... and negated neighbours in B: products cancel in pairs
            codes[:, 1:pairs:2] = codes[:, 0:pairs:2] ^ 0x80
    if pattern == "nan":
        row = 0 if side == "a" else rows - 1
        codes[row, int(torch.randint(0, k, (1,), generator=g))] = 0x7F if side == "a" else 0xFF
    return codes.to(torch.uint8)


def _scales(kind: str, g: torch.Generator) -> tuple[float, float]:
    u = torch.rand(2, generator=g).tolist()
    if kind == "one":
        return 1.0, 1.0
    if kind == "mild":  # absmax / 448 for activations and weights
        return _f32(2.0 ** -10 * (1 + 255 * u[0])), _f32(2.0 ** -12 * (1 + 63 * u[1]))
    if kind == "tiny":  # s_a·s_b ≈ 2^-140: subnormal and zero outputs
        return _f32(2.0 ** -70 * (1 + u[0])), _f32(2.0 ** -70 * (1 + u[1]))
    if kind == "huge":  # s_a·s_b ≈ 2^120: finite and overflowing (Inf) outputs
        return _f32(2.0 ** 60 * (1 + u[0])), _f32(2.0 ** 60 * (1 + u[1]))
    if kind == "negative":
        return _f32(-(0.25 + u[0])), _f32(0.5 + u[1])
    raise ValueError(f"unknown scale kind {kind!r}")


def canonical_mma(item: Plan) -> dict:
    """The setting as the browser passes it (the GUI's canonical mma: f2_bits only when decoupled)."""
    mma = {"algorithm": "cofda", "f_bits": item.f_bits, "chunk_size": item.chunk_size, "c_mode": item.c_mode,
           "promote_interval": 0, "norm_rounding": item.norm_rounding}
    if item.c_mode == "decoupled":
        mma["f2_bits"] = item.f2_bits
    return mma


def _operand(codes: torch.Tensor, scale: float) -> Operand:
    values = codes.view(torch.float8_e4m3fn).float()
    return Operand(values, FP8_E4M3, torch.tensor(scale, dtype=torch.float32), FP32, "tensor")


def _hex_bits(out: torch.Tensor) -> str:
    return "".join(f"{v & 0xFFFFFFFF:08x}" for v in out.contiguous().view(torch.int32).flatten().tolist())


def build_case(index: int) -> dict:
    """Case ``index`` of :func:`plan`, generated from a seed of its own."""
    item = plan()[index]
    g = torch.Generator().manual_seed(SEED * 1000 + index)
    a = _codes(item.pattern, item.M, item.K, g, "a")
    b = _codes(item.pattern, item.N, item.K, g, "b")
    scale_a, scale_b = _scales(item.scales, g)
    spec = MMASpec("cofda", f_bits=item.f_bits, chunk_size=item.chunk_size, c_mode=item.c_mode,
                   f2_bits=item.f2_bits or 23, norm_rounding=item.norm_rounding, out_format="fp32")
    a_op, b_op = _operand(a, scale_a), _operand(b, scale_b)
    expected = gemm(a_op, b_op, spec, backend="reference")
    exact = gemm(a_op, b_op, MMASpec("fp64", out_format="fp32"), backend="reference")
    return {"id": f"{index:03d}-{item.tag}", "tags": [item.tag, item.pattern, f"scale_{item.scales}"],
            "M": item.M, "N": item.N, "K": item.K,
            "a": bytes(a.flatten().tolist()).hex(), "b": bytes(b.flatten().tolist()).hex(),
            "scale_a": _bits(scale_a), "scale_b": _bits(scale_b), "mma": canonical_mma(item),
            "expected": _hex_bits(expected), "fp64": _hex_bits(exact)}


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build() -> dict:
    return {"generator": "app/webgpu_golden.py", "seed": SEED,
            "env": {"tricast": tricast.__version__, "git_sha": _git_sha(), "torch": torch.__version__},
            "operand_format": "fp8_e4m3", "out_format": "fp32",
            "layout": "a: [M, K] codes, b: [N, K] codes, row-major hex; scale_*, expected, fp64: fp32 bits",
            "cases": [build_case(i) for i in range(len(plan()))]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    golden = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, separators=(",", ":")) + "\n")
    print(f"WEBGPU_GOLDEN_DONE cases={len(golden['cases'])} out={args.out} "
          f"seconds={time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
