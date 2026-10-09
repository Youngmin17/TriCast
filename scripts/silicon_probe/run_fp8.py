"""Compare one Hopper FP8 instruction sequence to TriCast without hiding mismatches.

Requires an isolated UCL H100/H200, nvcc and cuobjdump. This is a numerical
silicon probe, not a GEMM throughput benchmark or a test of other chips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from tricast import gemm
from tricast.eval.envinfo import capture_env
from tricast.formats import get_format
from tricast.mma.operand import Operand
from tricast.mma.spec import get_preset
from tricast.reference.mma import _cofda, resolve_scale_apply

HERE = Path(__file__).resolve().parent
INSTRUCTION = "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
FP8 = get_format("fp8_e4m3")
# Only the output boundary changes: the PTX instruction exposes FP32, not BF16.
PRESET = replace(get_preset("nvidia_hopper_fp8"), out_format=get_format("fp32"))


@dataclass(frozen=True)
class Case:
    name: str
    a: torch.Tensor
    bt: torch.Tensor
    c: torch.Tensor
    reason: str
    c_prefix: torch.Tensor | None = None


def decode(raw: torch.Tensor) -> torch.Tensor:
    """Native E4M3FN decode; no calibration, scale or extra input rounding."""
    return raw.contiguous().view(torch.float8_e4m3fn).float()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    left, right = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    if left.shape != right.shape or left.dtype != torch.float32 or right.dtype != torch.float32:
        raise AssertionError("comparison requires same-shaped FP32 outputs")
    differing = left.view(torch.int32) != right.view(torch.int32)
    locations = differing.nonzero()
    return {
        "bit_equal": not bool(differing.any()), "mismatched_elements": int(differing.sum()),
        "outputs_finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "max_abs_diff": float((left.double() - right.double()).abs().max()),
        "first_mismatches": [{"index": i.tolist(),
                              "actual_bits": f"0x{int(left.view(torch.int32)[tuple(i)]) & 0xffffffff:08x}",
                              "expected_bits": f"0x{int(right.view(torch.int32)[tuple(i)]) & 0xffffffff:08x}"}
                             for i in locations[:8]],
    }


def native(module: ModuleType, a: torch.Tensor, bt: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return module.native_fp8_mma(a.cuda().contiguous(), bt.cuda().contiguous(), c.cuda().contiguous()).cpu()


def mapping_probe(module: ModuleType, out: Path) -> dict[str, Any]:
    # Every (row,column,K) pair independently probes the published fragment mapping.
    count = 16 * 8 * 32
    a, bt = torch.zeros(count + 2, 16, 32, dtype=torch.uint8), torch.zeros(count + 2, 8, 32,
                                                                       dtype=torch.uint8)
    c, expected = torch.zeros(count + 2, 16, 8), torch.zeros(count + 2, 16, 8)
    ids = torch.arange(count)
    row, col, k = ids // 256, (ids // 32) % 8, ids % 32
    a[ids, row, k], bt[ids, col, k], expected[ids, row, col] = 0x38, 0x38, 1.0
    a[-2], bt[-2], expected[-2] = 0x38, 0x38, 32.0
    c[-1] = torch.arange(1, 129, dtype=torch.float32).reshape(16, 8)
    expected[-1] = c[-1]
    actual = native(module, a, bt, c)
    report = compare(actual, expected)
    torch.save({"a_raw": a, "bt_raw": bt, "c": c, "native": actual, "expected": expected},
               out / "mapping.pt")
    return {"cases": count + 2, "single_term_cases": count, **report,
            "reason": ("4096 independent unit products locate every A/B lane coordinate; all-ones gives 32; "
                       "zero products preserve 128 distinct C coordinates")}


def cases(random_count: int, generator: torch.Generator) -> list[Case]:
    valid = torch.tensor([code for code in range(256) if (code & 127) != 127], dtype=torch.uint8)

    def raw(rows: int, k: int) -> torch.Tensor:
        return valid[torch.randint(len(valid), (rows, k), generator=generator)]

    result: list[Case] = []
    zero = torch.zeros(16, 8)
    for index in range(random_count):
        k = (32, 64, 96, 128)[index % 4]
        result.append(Case(f"random_{index:04d}_k{k}", raw(16, k), raw(8, k), zero,
                           "seeded finite E4M3 raw codes, including subnormals and signed zero"))
    patterns = {
        "cancellation": [0x7e, 0xfe, 0x01, 0x81, 0x38, 0xb8, 0x3f, 0xbf],
        "exponent_spread": [0x01, 0x08, 0x10, 0x28, 0x38, 0x50, 0x70, 0x7e],
        "subnormal_products": [0x01, 0x02, 0x03, 0x07, 0x81, 0x82, 0x83, 0x87],
        "signed_zero": [0x00, 0x80],
    }
    for name, pattern in patterns.items():
        for k in (32, 64):
            row = torch.tensor(pattern, dtype=torch.uint8).repeat(k // len(pattern))
            a = row.expand(16, k).clone()
            b = torch.full((8, k), 0x38, dtype=torch.uint8)
            result.append(Case(f"{name}_k{k}", a, b, zero,
                               "cancellation/sign/exponent boundaries in a fixed sequential K order"))
    # Initial FP32 C is not a bias. The public GEMM API starts at zero, so this
    # direct instruction contract uses the existing exact CoFDA transition only.
    c_values = torch.tensor([0.0, -0.0, 2.0**-149, -(2.0**-149), 2.0**-126, -(2.0**-126),
                             1 + 2.0**-23, -(1 + 2.0**-23), 2.0**20, -(2.0**20), 2.0**-18, -(2.0**-18)])
    c = c_values.repeat(11)[:128].reshape(16, 8)
    for k in (32, 64):
        result.append(Case(f"direct_fp32_c_k{k}", raw(16, k), raw(8, k), c,
                           "nonzero FP32 C, including payload bits not expressible as an FP8 input"))
    result.append(Case("direct_fp32_c_zero_products", torch.zeros(16, 32, dtype=torch.uint8),
                       torch.zeros(8, 32, dtype=torch.uint8), c,
                       "probe initial-C retention with zero products; compare the existing CoFDA transition"))
    # A 32-wide identity prefix produces this FP8-grid C exactly. This allows a
    # public reference/Triton GEMM check without adding an initial-C API or bias.
    # A sum of zero products does not construct a negative zero; choose actual
    # nonzero C here. Signed-zero C is covered by the direct FP32-C cases above.
    nonzero = valid[(valid & 127) != 0]
    prefix_codes = nonzero[torch.randint(len(nonzero), (16, 8), generator=generator)]
    for k in (32, 64):
        result.append(Case(f"prefix_nonzero_c_k{k}", raw(16, k), raw(8, k), decode(prefix_codes),
                           "identity prefix generates C exactly, then follows the same native MMA chunks",
                           prefix_codes))
    return result


def check_case(module: ModuleType, case: Case, out: Path) -> dict[str, Any]:
    av, bv = decode(case.a), decode(case.bt)
    a, b = Operand(av, FP8), Operand(bv, FP8)
    actual = native(module, case.a[None], case.bt[None], case.c[None])[0]
    if bool((case.c.view(torch.int32) != 0).any()):
        reference = _cofda(a, b, PRESET, resolve_scale_apply(PRESET, a, b), case.c.clone())
        scope = "direct_initial_c_reference_only"
    else:
        reference = gemm(a, b, PRESET, backend="reference")
        scope = "zero_c_public_gemm"
    reports = {"native_vs_reference": compare(actual, reference)}
    diagnostics = {}
    if case.name == "direct_fp32_c_zero_products":
        # PTX does not specify FP8 MMA rounding/subnormal handling. Report
        # passthrough empirically; do not turn it into a new acceptance rule.
        diagnostics = {"native_vs_initial_c": compare(actual, case.c),
                       "reference_vs_initial_c": compare(reference, case.c)}
    retained: dict[str, Any] = {"a_raw": case.a, "bt_raw": case.bt, "c": case.c,
                                "native": actual, "reference": reference}
    if scope == "zero_c_public_gemm":
        triton = gemm(Operand(av.cuda(), FP8), Operand(bv.cuda(), FP8), PRESET, backend="triton").cpu()
        reports.update(native_vs_triton=compare(actual, triton),
                       reference_vs_triton=compare(reference, triton))
        retained["triton"] = triton
    if case.c_prefix is not None:
        prefix_a, prefix_b = torch.zeros(16, 32, dtype=torch.uint8), torch.zeros(8, 32, dtype=torch.uint8)
        prefix_a[:, :8] = case.c_prefix
        prefix_b[torch.arange(8), torch.arange(8)] = 0x38
        full_a, full_b = torch.cat((prefix_a, case.a), 1), torch.cat((prefix_b, case.bt), 1)
        prefix_native = native(module, prefix_a[None], prefix_b[None], torch.zeros(1, 16, 8))[0]
        chained_native = native(module, full_a[None], full_b[None], torch.zeros(1, 16, 8))[0]
        ao, bo = Operand(decode(full_a), FP8), Operand(decode(full_b), FP8)
        ref = gemm(ao, bo, PRESET, backend="reference")
        triton = gemm(Operand(ao.values.cuda(), FP8), Operand(bo.values.cuda(), FP8), PRESET,
                      backend="triton").cpu()
        reports.update(prefix_generates_c=compare(prefix_native, case.c),
                       direct_c_vs_native_prefix=compare(actual, chained_native),
                       native_prefix_vs_reference=compare(chained_native, ref),
                       native_prefix_vs_triton=compare(chained_native, triton),
                       prefix_reference_vs_triton=compare(ref, triton))
        retained.update(prefix_codes=case.c_prefix, chained_native=chained_native, prefix_reference=ref,
                        prefix_triton=triton)
        scope = "direct_initial_c_and_exact_prefix_public_gemm"
    passed = all(r["bit_equal"] and r["outputs_finite"] for r in reports.values())
    if not passed:
        torch.save(retained, out / "mismatches" / f"{case.name}.pt")
    return {"name": case.name, "k": case.a.shape[1], "scope": scope, "passed": passed,
            "reason": case.reason, "comparisons": reports, "diagnostics": diagnostics}


def witness(module: ModuleType, out: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"extension": module.__file__,
                              "extension_sha256": digest(Path(module.__file__))}
    for kind in ("ptx", "sass"):
        command = ["cuobjdump", f"--dump-{kind}", module.__file__]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        (out / f"native.{kind}.log").write_text(run.stdout)
        (out / f"native.{kind}.stderr.log").write_text(run.stderr)
        pattern = re.escape(INSTRUCTION) if kind == "ptx" else r"(?:HMMA|MMA|QGMMA).*E4M3"
        lines = [line.strip() for line in run.stdout.splitlines() if re.search(pattern, line)]
        evidence[kind] = {"command": command, "returncode": run.returncode,
                          "instruction_present": bool(lines), "matching_lines": lines[:16]}
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=128)
    args = parser.parse_args()
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")
    if any((args.out / name).exists() for name in ("env.json", "result.json")):
        parser.error("output already contains a run; choose a fresh directory")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mismatches").mkdir()
    result: dict[str, Any] = {"status": "running", "instruction": INSTRUCTION, "cases": [],
                              "scope": "finite_raw_e4m3_hopper_instruction_sequence_not_universal_silicon",
                              "blackwell_fp4": "blocked: no Blackwell GPU; not emulated as silicon evidence"}
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expose exactly one isolated Hopper GPU with CUDA_VISIBLE_DEVICES")
        if torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("native FP8 probe requires Hopper sm_90 (H100/H200), not an A100/V100")
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        env = capture_env(extra={"command": [sys.executable, *sys.argv], "seed": 42,
                                 "probe_python_sha256": digest(Path(__file__)),
                                 "probe_cuda_sha256": digest(HERE / "fp8_mma.cu"),
                                 "preset": "nvidia_hopper_fp8", "out_format": "fp32",
                                 "initial_c": "direct transition or exact identity prefix, not Linear bias",
                                 "inputs": "raw finite E4M3FN bytes; no scales or calibration"})
        (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
        from torch.utils.cpp_extension import load

        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
        os.environ.setdefault("MAX_JOBS", "2")
        build = args.out.resolve() / "build"
        build.mkdir()
        module = load(name="tricast_native_hopper_fp8_probe", sources=[str(HERE / "fp8_mma.cu")],
                      extra_cuda_cflags=["-O2", "-lineinfo", "--fmad=false"],
                      build_directory=str(build), verbose=True)
        result["witness"] = witness(module, args.out)
        result["mapping"] = mapping_probe(module, args.out)
        selected = cases(args.random_cases, torch.Generator().manual_seed(42))
        torch.save([{ "name": c.name, "a_raw": c.a, "bt_raw": c.bt, "c": c.c, "c_prefix": c.c_prefix}
                    for c in selected], args.out / "inputs.pt")
        for case in selected:
            report = check_case(module, case, args.out)
            result["cases"].append(report)
            print(json.dumps({"case": case.name, "passed": report["passed"]}), flush=True)
        result["passed_cases"] = sum(c["passed"] for c in result["cases"])
        result["failed_cases"] = len(selected) - result["passed_cases"]
        observed = all(result["witness"][kind]["returncode"] == 0
                       and result["witness"][kind]["instruction_present"] for kind in ("ptx", "sass"))
        result["bypass_detected"] = not observed
        passed = (observed and result["mapping"]["bit_equal"] and result["mapping"]["outputs_finite"]
                  and result["failed_cases"] == 0)
        result["status"] = "passed" if passed else "failed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"SILICON_PROBE_DONE status={result['status']}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
