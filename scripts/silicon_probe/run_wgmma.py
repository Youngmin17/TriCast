"""Bitwise native Hopper WGMMA FP8 probe; no preset mutation or relaxed parity.

Requires one isolated H100/H200 and sm_90a nvcc/cuobjdump support. The old
mma.sync probe is preserved separately: its PTX alone is not a native witness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from run_fp8 import FP8, PRESET, Case, compare, decode, digest
from run_fp8 import cases as tile_cases

from tricast import gemm
from tricast.eval.envinfo import capture_env, source_identity
from tricast.mma.operand import Operand
from tricast.reference.mma import _cofda, resolve_scale_apply

HERE = Path(__file__).resolve().parent
INSTRUCTION = "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3"
SASS_PATTERN = r"\b(?:QGMMA|HGMMA)\.64x8x32\.F32\.E4M3\.E4M3\b"
CUDA_FLAGS = ["-O2", "-lineinfo", "--fmad=false",
              "-gencode=arch=compute_90a,code=sm_90a", "-gencode=arch=compute_90a,code=compute_90a"]


def native(module: ModuleType, a: torch.Tensor, bt: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return module.native_fp8_wgmma(a.cuda().contiguous(), bt.cuda().contiguous(), c.cuda().contiguous()).cpu()


def mapping_probe(module: ModuleType, out: Path) -> dict[str, Any]:
    count = 64 * 8 * 32
    a = torch.zeros(count + 2, 64, 32, dtype=torch.uint8)
    bt = torch.zeros(count + 2, 8, 32, dtype=torch.uint8)
    c, expected = torch.zeros(count + 2, 64, 8), torch.zeros(count + 2, 64, 8)
    ids = torch.arange(count)
    row, column, k = ids // 256, (ids // 32) % 8, ids % 32
    a[ids, row, k], bt[ids, column, k], expected[ids, row, column] = 0x38, 0x38, 1.0
    a[-2], bt[-2], expected[-2] = 0x38, 0x38, 32.0
    c[-1] = torch.arange(1, 513, dtype=torch.float32).reshape(64, 8)
    expected[-1] = c[-1]
    actual = native(module, a, bt, c)
    torch.save({"a_raw": a, "bt_raw": bt, "c": c, "native": actual, "expected": expected},
               out / "mapping.pt")
    return {"cases": count + 2, "single_term_cases": count, **compare(actual, expected),
            "reason": "16384 independent unit products, all-ones=32, and 512 unique C coordinates"}


def cases(random_count: int) -> list[Case]:
    # Same categories as the preserved MMA probe, but independent A/prefix rows
    # across all four warps. BT comes from the first seeded tile for every warp.
    generator = torch.Generator().manual_seed(42)
    tiles = [tile_cases(random_count, generator) for _ in range(4)]
    return [Case(first.name, torch.cat([tile[i].a for tile in tiles]), first.bt,
                 torch.cat([tile[i].c for tile in tiles]), first.reason,
                 torch.cat([tile[i].c_prefix for tile in tiles]) if first.c_prefix is not None else None)
            for i, first in enumerate(tiles[0])]


def check_case(module: ModuleType, case: Case, out: Path) -> dict[str, Any]:
    av, bv = decode(case.a), decode(case.bt)
    a, b = Operand(av, FP8), Operand(bv, FP8)
    actual = native(module, case.a[None], case.bt[None], case.c[None])[0]
    nonzero_c = bool((case.c.view(torch.int32) != 0).any())
    reference = (_cofda(a, b, PRESET, resolve_scale_apply(PRESET, a, b), case.c.clone())
                 if nonzero_c else gemm(a, b, PRESET, backend="reference"))
    scope = "direct_initial_c_reference_only" if nonzero_c else "zero_c_public_gemm"
    reports = {"native_vs_reference": compare(actual, reference)}
    retained: dict[str, Any] = {"a_raw": case.a, "bt_raw": case.bt, "c": case.c,
                                "native": actual, "reference": reference}
    diagnostics = {}
    if case.name == "direct_fp32_c_zero_products":
        diagnostics = {"native_vs_initial_c": compare(actual, case.c),
                       "reference_vs_initial_c": compare(reference, case.c)}
    if not nonzero_c:
        triton = gemm(Operand(av.cuda(), FP8), Operand(bv.cuda(), FP8), PRESET, backend="triton").cpu()
        reports.update(native_vs_triton=compare(actual, triton),
                       reference_vs_triton=compare(reference, triton))
        retained["triton"] = triton
    if case.c_prefix is not None:
        prefix_a, prefix_b = torch.zeros(64, 32, dtype=torch.uint8), torch.zeros(8, 32, dtype=torch.uint8)
        prefix_a[:, :8] = case.c_prefix
        prefix_b[torch.arange(8), torch.arange(8)] = 0x38
        full_a, full_b = torch.cat((prefix_a, case.a), 1), torch.cat((prefix_b, case.bt), 1)
        prefix = native(module, prefix_a[None], prefix_b[None], torch.zeros(1, 64, 8))[0]
        chained = native(module, full_a[None], full_b[None], torch.zeros(1, 64, 8))[0]
        ao, bo = Operand(decode(full_a), FP8), Operand(decode(full_b), FP8)
        ref = gemm(ao, bo, PRESET, backend="reference")
        triton = gemm(Operand(ao.values.cuda(), FP8), Operand(bo.values.cuda(), FP8), PRESET,
                      backend="triton").cpu()
        reports.update(prefix_generates_c=compare(prefix, case.c),
                       direct_c_vs_native_prefix=compare(actual, chained),
                       native_prefix_vs_reference=compare(chained, ref),
                       native_prefix_vs_triton=compare(chained, triton),
                       prefix_reference_vs_triton=compare(ref, triton))
        retained.update(prefix_codes=case.c_prefix, chained_native=chained, prefix_reference=ref,
                        prefix_triton=triton)
        scope = "direct_initial_c_and_exact_prefix_public_gemm"
    passed = all(report["bit_equal"] and report["outputs_finite"] for report in reports.values())
    if not passed:
        torch.save(retained, out / "mismatches" / f"{case.name}.pt")
    return {"name": case.name, "k": case.a.shape[1], "scope": scope, "passed": passed,
            "reason": case.reason, "comparisons": reports, "diagnostics": diagnostics}


def witness(binary: Path, out: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"extension": str(binary), "extension_sha256": digest(binary)}
    for kind in ("ptx", "sass"):
        command = ["cuobjdump", f"--dump-{kind}", str(binary)]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        path = out / f"native.{kind}.log"
        path.write_text(run.stdout)
        (out / f"native.{kind}.stderr.log").write_text(run.stderr)
        # FP16 HMMA after F2FP decoding must NOT count as native FP8 evidence.
        # NVIDIA's Hopper table names the native FP8 warpgroup opcode QGMMA.
        # Accept either disassembler spelling, with exact shape/dtype/operand tags.
        pattern = re.escape(INSTRUCTION) if kind == "ptx" else SASS_PATTERN
        lines = [line.strip() for line in run.stdout.splitlines() if re.search(pattern, line)]
        evidence[kind] = {"command": command, "returncode": run.returncode,
                          "dump_sha256": digest(path), "instruction_present": bool(lines),
                          "matching_lines": lines[:16]}
    return evidence


def command_version(command: list[str]) -> dict[str, Any]:
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    return {"command": command, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}


def witness_present(evidence: dict[str, Any]) -> bool:
    return all(evidence[kind]["returncode"] == 0 and evidence[kind]["instruction_present"]
               for kind in ("ptx", "sass"))


def reinspect(run: Path, binary: Path, out: Path) -> int:
    """Reinspect an unchanged built binary; never rerun or rewrite numerical evidence."""
    result: dict[str, Any] = {"status": "running", "instruction": INSTRUCTION,
                              "scope": "witness_reinspection_only_no_compile_or_gpu_vector_rerun"}
    env = {"utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
           "command": [sys.executable, *sys.argv], **source_identity(),
           "witness_script_sha256": digest(Path(__file__)), "sass_pattern": SASS_PATTERN,
           "original_run": str(run.resolve()), "extension": str(binary.resolve()),
           "gpu_vectors_rerun": False, "extension_recompiled": False}
    try:
        prior_path, prior_env_path = run / "result.json", run / "env.json"
        prior = json.loads(prior_path.read_text())
        prior_env = json.loads(prior_env_path.read_text())
        if prior.get("instruction") != INSTRUCTION:
            raise ValueError("original result is not this WGMMA instruction")
        binary_hash = digest(binary)
        if binary_hash != prior["witness"]["extension_sha256"]:
            raise ValueError("extension differs from the binary used for the original numerical run")
        selected = prior["cases"]
        if not selected or any(type(case.get("passed")) is not bool for case in selected):
            raise ValueError("original result lacks completed numerical case records")
        passed_count = sum(case["passed"] for case in selected)
        failed_count = len(selected) - passed_count
        if (passed_count, failed_count) != (prior["passed_cases"], prior["failed_cases"]):
            raise ValueError("original numerical counts disagree with retained case records")
        env.update(original_env_sha256=digest(prior_env_path), original_result_sha256=digest(prior_path),
                   original_env=prior_env, extension_sha256=binary_hash,
                   cuobjdump=command_version(["cuobjdump", "--version"]))
        evidence = witness(binary, out)
        observed = witness_present(evidence)
        mapping = prior["mapping"]
        passed = observed and mapping["bit_equal"] and mapping["outputs_finite"] and failed_count == 0
        result.update(status="passed" if passed else "failed",
                      witness_status="passed" if observed else "failed",
                      witness=evidence, bypass_detected=not observed, mapping=mapping,
                      passed_cases=passed_count, failed_cases=failed_count,
                      failed_case_names=[case["name"] for case in selected if not case["passed"]],
                      original_status_unchanged=prior["status"],
                      original_result_sha256=env["original_result_sha256"],
                      note="Only native witness reinspected; original numeric gates unchanged")
        # Reinspection must not mutate its source files, including via the tool invocation.
        if (digest(prior_path) != env["original_result_sha256"]
                or digest(prior_env_path) != env["original_env_sha256"]):
            raise RuntimeError("original evidence changed during witness reinspection")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        (out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"WGMMA_WITNESS_REINSPECTION_DONE status={result['status']} "
          f"witness_status={result.get('witness_status', 'failed')}", flush=True)
    return 0 if result["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=128)
    parser.add_argument("--reinspect-run", type=Path,
                        help="Retained completed run; inspect its exact extension without GPU vector rerun")
    parser.add_argument("--extension", type=Path,
                        help="Existing extension .so; required with --reinspect-run")
    args = parser.parse_args()
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")
    if (args.reinspect_run is None) != (args.extension is None):
        parser.error("--reinspect-run and --extension must be supplied together")
    if args.reinspect_run is not None and args.out.resolve() == args.reinspect_run.resolve():
        parser.error("reinspection must use a separate output directory")
    if any((args.out / name).exists() for name in ("env.json", "result.json")):
        parser.error("output already contains a run; choose a fresh directory")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.reinspect_run is not None:
        return reinspect(args.reinspect_run, args.extension, args.out)
    (args.out / "mismatches").mkdir()
    result: dict[str, Any] = {"status": "running", "instruction": INSTRUCTION, "cases": [],
                              "scope": "finite_raw_e4m3_hopper_wgmma_sequence_not_universal_silicon",
                              "blackwell_fp4": "blocked: no Blackwell GPU; not replaced by emulation"}
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("expose exactly one isolated Hopper GPU with CUDA_VISIBLE_DEVICES")
        if torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("WGMMA requires Hopper sm_90a (H100/H200), not A100/V100")
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a+PTX"
        os.environ.setdefault("MAX_JOBS", "2")
        env = capture_env(extra={"command": [sys.executable, *sys.argv], "seed": 42,
                                 "probe_python_sha256": digest(Path(__file__)),
                                 "helper_python_sha256": digest(HERE / "run_fp8.py"),
                                 "probe_cuda_sha256": digest(HERE / "fp8_wgmma.cu"),
                                 "preset": "nvidia_hopper_fp8", "out_format": "fp32",
                                 "mma_spec": repr(PRESET), "cuda_flags": CUDA_FLAGS,
                                 "cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
                                 "initial_c": "direct FP32 transition or exact FP8 identity prefix; not bias",
                                 "inputs": "finite raw E4M3FN bytes; no scales or calibration",
                                 "gpu_inventory": command_version(
                                     ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"]),
                                 "nvcc": command_version(["nvcc", "--version"]),
                                 "cuobjdump": command_version(["cuobjdump", "--version"])})
        (args.out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
        from torch.utils.cpp_extension import load

        build = args.out.resolve() / "build"
        build.mkdir()
        module = load(name="tricast_native_hopper_fp8_wgmma_probe", sources=[str(HERE / "fp8_wgmma.cu")],
                      extra_cuda_cflags=CUDA_FLAGS, build_directory=str(build), verbose=True)
        result["witness"] = witness(Path(module.__file__), args.out)
        result["mapping"] = mapping_probe(module, args.out)
        result["mapping"]["artifact_sha256"] = digest(args.out / "mapping.pt")
        selected = cases(args.random_cases)
        torch.save([{"name": case.name, "a_raw": case.a, "bt_raw": case.bt, "c": case.c,
                     "c_prefix": case.c_prefix} for case in selected], args.out / "inputs.pt")
        result["inputs_sha256"] = digest(args.out / "inputs.pt")
        for case in selected:
            report = check_case(module, case, args.out)
            result["cases"].append(report)
            print(json.dumps({"case": case.name, "passed": report["passed"]}), flush=True)
        result["passed_cases"] = sum(case["passed"] for case in result["cases"])
        result["failed_cases"] = len(selected) - result["passed_cases"]
        observed = witness_present(result["witness"])
        result["bypass_detected"] = not observed
        passed = (observed and result["mapping"]["bit_equal"] and result["mapping"]["outputs_finite"]
                  and result["failed_cases"] == 0)
        result["status"] = "passed" if passed else "failed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"WGMMA_SILICON_PROBE_DONE status={result['status']}", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
