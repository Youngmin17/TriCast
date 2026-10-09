#!/usr/bin/env python
"""NADPE MMA-Emu FP8 golden vectors for TriCast bit-exact checks.

GPU job (launch through gpu_cap.sh). Loads the prebuilt extension .so directly,
so the GPU job never triggers a JIT rebuild.

  gen    : generate vectors/fp8_<set>_<M>x<N>x<K>.pt, running every case twice
           (two full sweeps) and requiring bit-identical outputs.
  verify : reload the saved .pt files in a fresh process and re-run every case;
           outputs must match the stored bits exactly.

Vector file schema (torch.load(..., weights_only=True) safe):
  a_codes  uint8 [M,K]   E4M3 codes, row-major activations
  w_codes  uint8 [N,K]   E4M3 codes, weights; the op gets b = w.t() ([K,N], col-major)
  scale_a, scale_b       python floats, exactly representable in fp32
  cases    list of dict(algorithm, f_bits, g_bits, group_size, chunk_size,
                        out_bits int16 [M,N] = bf16 bit pattern)
           parameters an algorithm does not read are stored as 0.
  env      gpu / torch / cuda / nvcc / driver / sha256 of the NADPE FP8 sources
  meta     set, seed, shape, layout, epilogue and tiling notes
"""
import argparse
import glob
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# Dependency closure of fp8_gemm_kernels.cu inside mma_emu/.
FP8_SRC_FILES = [
    "fp8_gemm_kernels.cu",
    "gemm/scaled_fp8_mm.cuh",
    "formats/fp8_e4m3.cuh",
    "core/accumulator.cuh",
    "core/design_space.cuh",
    "core/fp32_utils.cuh",
    "core/gdfs_group.cuh",
    "core/tiling.cuh",
    "core/types.cuh",
]

SHAPES = [(33, 17, 96), (64, 40, 256), (5, 8, 77)]
INPUT_SETS = ["uniform", "realistic"]
SCALES = [("unit", 1.0, 1.0), ("scaled", 0.37, 2.5)]
COFDA_F = (3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25)
COFDA_CS = (16, 32)
GDFS_F = (7, 13, 25, 35)
GDFS_G = (3, 4, 5, 6, 8, 13, 32)
GDFS_GS = (8, 16)
SENTINEL = 0x7FAB  # bf16 NaN payload; the kernel's only NaN is 0x7FFF

LAYOUT = ("a: fp8 e4m3 [M,K] row-major; op b = w.t() with w [N,K] contiguous "
          "(b is [K,N] column-major, stride (1,K)); c: bf16 [M,N] row-major")
EPILOGUE = ("fp32 val = scale_a * (scale_b * acc) (two RN multiplies, no FMA, "
            "no bias) -> bf16 via __float2bfloat16_rn")
TILING = ("FP8EmuConfig BM=32 BN=8 BK=32. CoFDA chunks are CS-aligned from k=0. "
          "GDFS: groups of GS inside each BK=32 tile, then one F-bit FDA over the "
          "BK/GS group operands plus the running accumulator per tile.")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def load_ext(so_path):
    name = os.path.splitext(os.path.basename(so_path))[0]
    loader = importlib.machinery.ExtensionFileLoader(name, so_path)
    spec = importlib.util.spec_from_file_location(name, so_path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def case_grid():
    cases = []
    for alg in (2, 3):  # CoFDA C-fused, CoFDA C-decoupled
        for cs in COFDA_CS:
            for f in COFDA_F:
                cases.append(dict(algorithm=alg, f_bits=f, g_bits=0, group_size=0, chunk_size=cs))
    for gs in GDFS_GS:  # GDFS
        for f in GDFS_F:
            for g in GDFS_G:
                cases.append(dict(algorithm=1, f_bits=f, g_bits=g, group_size=gs, chunk_size=0))
    return cases


def make_codes(kind, M, N, K, seed):
    torch.manual_seed(seed)
    if kind == "uniform":
        # uniform over the 254 non-NaN E4M3 codes (0x00..0x7E, 0x80..0xFE),
        # which includes +0, -0 and all subnormals
        a = torch.randint(0, 254, (M, K), dtype=torch.int32)
        w = torch.randint(0, 254, (N, K), dtype=torch.int32)
        a = (a + (a >= 0x7F).to(torch.int32)).to(torch.uint8)
        w = (w + (w >= 0x7F).to(torch.int32)).to(torch.uint8)
    elif kind == "realistic":
        a = (torch.randn(M, K) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
        w = (torch.randn(N, K) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        raise ValueError(kind)
    for t in (a, w):
        assert not bool(((t & 0x7F) == 0x7F).any()), "NaN code generated"
    return a.contiguous(), w.contiguous()


def run_case(ext, a_dev, b_dev, sa_t, sb_t, case, M, N):
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a_dev.device)
    out.view(torch.int16).fill_(SENTINEL)
    ext.mma_emu_scaled_fp8_mm(out, a_dev, b_dev, sa_t, sb_t, None,
                              case["algorithm"], case["f_bits"], case["g_bits"],
                              case["group_size"], case["chunk_size"])
    err = ext.last_cuda_error()
    torch.cuda.synchronize()
    if err:
        raise RuntimeError(f"kernel launch failed for {case}: {err}")
    bits = out.view(torch.int16).cpu().clone()
    untouched = int((bits == SENTINEL).sum())
    if untouched:
        raise RuntimeError(f"bypass: {untouched} outputs never written for {case}")
    return bits


def to_device(a_codes, w_codes, dev):
    a_dev = a_codes.to(dev).view(torch.float8_e4m3fn)
    w_dev = w_codes.to(dev).view(torch.float8_e4m3fn)
    b_dev = w_dev.t()
    K = a_codes.shape[1]
    assert a_dev.is_contiguous() and b_dev.stride() == (1, K), b_dev.stride()
    return a_dev, b_dev


def smi(query):
    # gpu_cap.sh exports CUDA_DEVICE_ORDER=PCI_BUS_ID, so the visible index is the
    # nvidia-smi index of the GPU this job runs on
    idx = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader",
                            "-i", idx], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def collect_env(so_path, src_dir):
    src = {rel: sha256_file(os.path.join(src_dir, rel)) for rel in FP8_SRC_FILES}
    combined = hashlib.sha256("".join(f"{src[k]}  {k}\n" for k in sorted(src)).encode()).hexdigest()
    try:
        nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True,
                              text=True).stdout.strip().splitlines()[-2]
    except Exception as e:  # noqa: BLE001
        nvcc = f"n/a ({e})"
    props = torch.cuda.get_device_properties(0)
    return dict(
        gpu=props.name,
        gpu_capability=f"{props.major}.{props.minor}",
        gpu_uuid=smi("uuid"),
        driver=smi("driver_version"),
        clocks_sm_mem_mhz=smi("clocks.sm,clocks.mem"),
        host=socket.gethostname(),
        torch=str(torch.__version__),  # TorchVersion is not weights_only-loadable
        cuda=torch.version.cuda,
        nvcc=nvcc,
        ext_so_sha256=sha256_file(so_path),
        build_flags="-O2 -std=c++17 --expt-relaxed-constexpr "
                    "-gencode=arch=compute_80,code=sm_80 + torch COMMON_NVCC_FLAGS",
        nadpe_src_origin="micro26-ae-main (zip, no git metadata): "
                         "csrc/quantization/mma_emu, compiled unmodified",
        nadpe_src_sha256=src,
        nadpe_src_sha256_combined=combined,
    )


def cos_and_diff(bits, ref64):
    out = bits.view(torch.bfloat16).double()
    finite = bool(torch.isfinite(out).all())
    num = float((out * ref64).sum())
    den = float(out.norm() * ref64.norm())
    cos = num / den if den > 0 else float("nan")
    return finite, cos, float((out - ref64).abs().max())


def generate(args, ext, env, dev):
    os.makedirs(args.out, exist_ok=True)
    cases = case_grid()
    summary, files = [], []
    total_mismatch = 0
    n_runs = 0
    t0 = time.time()
    for si, kind in enumerate(INPUT_SETS):
        for hi, (M, N, K) in enumerate(SHAPES):
            seed = 1234 + si * len(SHAPES) + hi
            a_codes, w_codes = make_codes(kind, M, N, K, seed)
            a_dev, b_dev = to_device(a_codes, w_codes, dev)
            exact = (a_codes.view(torch.float8_e4m3fn).double()
                     @ w_codes.view(torch.float8_e4m3fn).double().t())
            for tag, sa, sb in SCALES:
                sa32, sb32 = f32(sa), f32(sb)
                sa_t = torch.tensor([sa32], dtype=torch.float32, device=dev)
                sb_t = torch.tensor([sb32], dtype=torch.float32, device=dev)
                ref64 = exact * sa32 * sb32
                # two full sweeps over the case grid; must be bit-identical
                sweep = [[run_case(ext, a_dev, b_dev, sa_t, sb_t, c, M, N) for c in cases]
                         for _ in range(2)]
                n_runs += 2 * len(cases)
                rec_cases, stats = [], []
                for c, b1, b2 in zip(cases, sweep[0], sweep[1], strict=False):
                    mism = int((b1 != b2).sum())
                    total_mismatch += mism
                    finite, cos, maxd = cos_and_diff(b1, ref64)
                    rec_cases.append(dict(c, out_bits=b1))
                    stats.append(dict(c, repeat_mismatch=mism, y_finite=finite,
                                      cos_vs_exact=cos, max_abs_diff_vs_exact=maxd))
                set_name = f"{kind}-{tag}"
                fname = f"fp8_{set_name}_{M}x{N}x{K}.pt"
                path = os.path.join(args.out, fname)
                torch.save(dict(
                    a_codes=a_codes, w_codes=w_codes,
                    scale_a=sa32, scale_b=sb32,
                    scale_a_bits=f32_bits(sa32), scale_b_bits=f32_bits(sb32),
                    cases=rec_cases, env=env,
                    meta=dict(set=set_name, input_kind=kind, scale_tag=tag, seed=seed,
                              M=M, N=N, K=K, out_dtype="bfloat16", layout=LAYOUT,
                              epilogue=EPILOGUE, tiling=TILING,
                              generator=("uniform: torch.randint over the 254 non-NaN "
                                         "codes; realistic: (torch.randn*0.5)"
                                         ".to(float8_e4m3fn); seed via torch.manual_seed")),
                ), path)
                # the file must round-trip through the torch>=2.6 default loader
                back = torch.load(path, weights_only=True)
                assert all(torch.equal(x["out_bits"], y["out_bits"])
                           for x, y in zip(back["cases"], rec_cases, strict=False))
                assert torch.equal(back["a_codes"], a_codes) and torch.equal(back["w_codes"], w_codes)
                cos_list = [s["cos_vs_exact"] for s in stats]
                files.append(dict(file=fname, bytes=os.path.getsize(path),
                                  sha256=sha256_file(path), cases=len(rec_cases),
                                  seed=seed, M=M, N=N, K=K, scale_a=sa32, scale_b=sb32))
                summary.append(dict(file=fname, min_cos=min(cos_list), max_cos=max(cos_list),
                                    all_finite=all(s["y_finite"] for s in stats),
                                    repeat_mismatch=sum(s["repeat_mismatch"] for s in stats),
                                    per_case=stats))
                print(f"VEC {fname} cases={len(rec_cases)} bytes={os.path.getsize(path)} "
                      f"cos[min,max]=[{min(cos_list):.6f},{max(cos_list):.6f}] "
                      f"repeat_mismatch={summary[-1]['repeat_mismatch']}", flush=True)
    manifest = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    env=env, files=files, cases_per_file=len(cases),
                    total_cases=len(cases) * len(files),
                    layout=LAYOUT, epilogue=EPILOGUE, tiling=TILING,
                    grid=dict(cofda_f=COFDA_F, cofda_cs=COFDA_CS, gdfs_f=GDFS_F,
                              gdfs_g=GDFS_G, gdfs_gs=GDFS_GS, shapes=SHAPES,
                              scales=SCALES, input_sets=INPUT_SETS))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    result = dict(mode="gen", kernel_runs=n_runs, repeat_bit_mismatch_total=total_mismatch,
                  seconds=round(time.time() - t0, 1), files=files, summary=summary)
    return result, total_mismatch == 0


def verify(args, ext, env, dev):
    paths = sorted(glob.glob(os.path.join(args.out, "fp8_*.pt")))
    assert paths, "no vector files"
    total_cases, total_mismatch, rows = 0, 0, []
    t0 = time.time()
    for path in paths:
        d = torch.load(path, weights_only=True)
        a_codes, w_codes = d["a_codes"], d["w_codes"]
        M, K = a_codes.shape
        N = w_codes.shape[0]
        assert a_codes.dtype == torch.uint8 and w_codes.dtype == torch.uint8
        assert d["env"]["ext_so_sha256"] == env["ext_so_sha256"], "extension .so changed"
        a_dev, b_dev = to_device(a_codes, w_codes, dev)
        sa_t = torch.tensor([d["scale_a"]], dtype=torch.float32, device=dev)
        sb_t = torch.tensor([d["scale_b"]], dtype=torch.float32, device=dev)
        mism_cases = 0
        for c in d["cases"]:
            ob = c["out_bits"]
            assert ob.dtype == torch.int16 and tuple(ob.shape) == (M, N)
            bits = run_case(ext, a_dev, b_dev, sa_t, sb_t, c, M, N)
            m = int((bits != ob).sum())
            total_mismatch += m
            mism_cases += int(m > 0)
            total_cases += 1
        rows.append(dict(file=os.path.basename(path), cases=len(d["cases"]),
                         mismatched_cases=mism_cases))
        print(f"VERIFY {os.path.basename(path)} cases={len(d['cases'])} "
              f"mismatched_cases={mism_cases}", flush=True)
    result = dict(mode="verify", files=len(paths), cases=total_cases,
                  bit_mismatch_total=total_mismatch, seconds=round(time.time() - t0, 1),
                  rows=rows)
    return result, total_mismatch == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "verify"])
    ap.add_argument("--so", default=os.path.join(HERE, "build_fp8", "nadpe_mma_emu.so"))
    ap.add_argument("--src", default=os.path.join(HERE, "mma_emu"))
    ap.add_argument("--out", default=os.path.join(HERE, "vectors"))
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    os.makedirs(args.run_dir, exist_ok=True)
    assert torch.cuda.is_available(), "needs a GPU (launch via gpu_cap.sh)"
    dev = torch.device("cuda", 0)

    ext = load_ext(args.so)
    env = collect_env(args.so, args.src)
    assert all(type(v) in (str, dict) for v in env.values()), "env must hold plain types"
    env_run = dict(env, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   python=sys.version.split()[0], argv=sys.argv,
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   git_sha="n/a (oracle scripts not in a git repo; NADPE src is a zip)",
                   model_sha="n/a (synthetic inputs)", dataset_sha="n/a (seeded synthetic)")
    with open(os.path.join(args.run_dir, f"env_{args.mode}.json"), "w") as f:
        json.dump(env_run, f, indent=1)
    print("ENV " + json.dumps({k: env[k] for k in ("gpu", "gpu_capability", "driver", "torch",
                                                   "cuda", "nvcc", "ext_so_sha256")}), flush=True)

    fn = generate if args.mode == "gen" else verify
    try:
        result, ok = fn(args, ext, env, dev)
        rc = 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result, rc = dict(mode=args.mode, error=repr(e)), 2
    with open(os.path.join(args.run_dir, f"result_{args.mode}.json"), "w") as f:
        json.dump(result, f, indent=1)
    head = {k: v for k, v in result.items() if k not in ("summary", "files", "rows")}
    print("RESULT " + json.dumps(head), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
