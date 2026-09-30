#!/usr/bin/env python
"""NADPE MMA-Emu NVFP4 / MXFP4 golden vectors for TriCast bit-exact checks.

GPU job (launch through gpu_cap.sh). Uses the prebuilt build_fp4/nadpe_mma_emu_fp4.so.

  gen    : write vectors/fp4_<nvfp4|mxfp4>_<set>_<M>x<N>x<K>.pt, every case run in
           two full sweeps that must agree bit for bit; merges an "fp4" section
           into vectors/manifest.json (FP8 keys untouched).
  verify : fresh process; rebuilds the kernel inputs (packing, N padding, scale
           swizzle) from the stored LOGICAL codes and re-runs every case.

Vector file schema (torch.load(..., weights_only=True) safe):
  a_codes        uint8 [M,K]    E2M1 4-bit codes (unpacked, sign = bit 3)
  w_codes        uint8 [N,K]    E2M1 codes of the weights (out = a @ w^T)
  a_scale_codes  uint8 [M,K/B]  block-scale codes, logical row/block order
  w_scale_codes  uint8 [N,K/B]
  block_size     B = 16 (nvfp4) | 32 (mxfp4)
  scale_format   "ue4m3" | "e8m0"
  alpha          fp32 global scale (nvfp4); 1.0 for mxfp4 (kernel has no alpha)
  cases          list of dict(algorithm, f_bits, g_bits, group_size, chunk_size,
                              out_bits int16 [M,N] = bf16 bits)
  env, meta
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import torch
from gen_vectors import SENTINEL, f32, f32_bits, load_ext, sha256_file, smi

HERE = os.path.dirname(os.path.abspath(__file__))

FP4_SRC_FILES = [
    "nvfp4_gemm_kernels.cu", "mxfp4_gemm_kernels.cu",
    "gemm/scaled_nvfp4_mm.cuh", "gemm/scaled_mxfp4_mm.cuh",
    "formats/fp4_e2m1.cuh", "formats/nvfp4_ue4m3.cuh", "formats/mxfp4_e8m0.cuh",
    "formats/scale_swizzle.cuh",
    "core/accumulator.cuh", "core/design_space.cuh", "core/fp32_utils.cuh",
    "core/gdfs_group.cuh", "core/tiling.cuh", "core/types.cuh",
]

GDFS_F = (7, 9, 10, 11, 13, 15, 25, 35)
COFDA_F = (3, 5, 7, 9, 10, 11, 12, 13, 17, 21, 25)
FP4_G = (3, 4, 5, 6)
REQ_SHAPES = [(33, 17, 128), (5, 8, 96)]
INPUT_SETS = ["uniform", "realistic"]
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

FMT = {
    "nvfp4": dict(block=16, scale_format="ue4m3", k_align=32, pad_scale=0x38, alpha=True),
    "mxfp4": dict(block=32, scale_format="e8m0", k_align=64, pad_scale=127, alpha=False),
}
N_ALIGN = 32  # both entry points require N % 32 == 0

SWIZZLE_DOC = ("kernel A_sf/B_sf = logical [R,K/B] codes padded to [round_up(R,128), "
               "round_up(K/B,4)] (pad = 1.0 code), then view [Rp/128,4,32,KBp/4,4] -> "
               "permute(0,3,2,1,4) -> contiguous (vLLM swizzle_blockscale); the kernel reads "
               "(r,kb) at kb%4 + 4*((r%128)//32 + 4*(r%32 + 32*(kb//4 + (KBp/4)*(r//128))))")
PACK_DOC = "A/B passed packed [R,K/2] uint8: low nibble = even k, high nibble = odd k"
NPAD_DOC = ("op requires N%32==0: w rows N..N_pad-1 are zero codes with 1.0 scale codes; "
            "outputs are per-(m,n) independent, stored out_bits = first N columns")
EPILOGUE = {"nvfp4": "fp32 acc * alpha (one RN multiply) -> bf16 __float2bfloat16_rn",
            "mxfp4": "no alpha: fp32 acc -> bf16 __float2bfloat16_rn"}
TILING = {
    "nvfp4": ("BM=32 BN=8 BK=64. GDFS: 4 groups of GS=16 per BK tile (one UE4M3 block each) "
              "-> apply_ue4m3_scales -> one F-bit FDA over the 4 group operands + acc per "
              "tile. CoFDA: CS=16 chunks from k=0 (one block each), scales applied per "
              "product, chunked F-bit FDA with acc. GS/CS are fixed by the kernel."),
    "mxfp4": ("BM=32 BN=8 BK=64. GDFS: 64/GS groups per BK tile, group g uses block "
              "(64*kt+g*GS)//32 -> apply_e8m0_scales -> one F-bit FDA over the 64/GS group "
              "operands + acc per tile. CoFDA: CS chunks from k=0, E8M0 applied per product."),
}
SCALE_RULES = {
    "nvfp4": ("UE4M3: MSB masked off (code & 0x7F); (code&0x7F)==0x7F is NaN; masked 0 is a "
              "zero scale; e=0 subnormal sig=m exp=-6, else sig=8+m exp=e-7 (value sig/8*2^exp)"),
    "mxfp4": ("E8M0: 0xFF NaN; code 0 is a ZERO scale in NADPE (not 2^-127); "
              "else 2^(code-127)"),
}


def round_up(x, y):
    return (x + y - 1) // y * y


def fp4_cases(fmt):
    cases = []
    if fmt == "nvfp4":
        for f in GDFS_F:
            for g in FP4_G:
                cases.append(dict(algorithm=1, f_bits=f, g_bits=g, group_size=16, chunk_size=0))
        for f in COFDA_F:
            cases.append(dict(algorithm=2, f_bits=f, g_bits=0, group_size=0, chunk_size=16))
    else:
        for gs in (16, 32):
            for f in GDFS_F:
                for g in FP4_G:
                    cases.append(dict(algorithm=1, f_bits=f, g_bits=g, group_size=gs,
                                      chunk_size=0))
        for cs in (16, 32):
            for f in COFDA_F:
                cases.append(dict(algorithm=2, f_bits=f, g_bits=0, group_size=0, chunk_size=cs))
    return cases


def e2m1_quantize(q):
    """nearest E2M1 value (ties toward the lower magnitude), saturating at 6."""
    mag = q.abs().clamp(max=6.0)
    idx = (mag.unsqueeze(-1) - E2M1_VALUES).abs().argmin(dim=-1).to(torch.uint8)
    return idx | ((q < 0).to(torch.uint8) << 3)


def quant_nvfp4(x):
    R, K = x.shape
    gs = f32(float(x.abs().max()) / (448.0 * 6.0))
    blocks = x.reshape(R, K // 16, 16)
    s = (blocks.abs().amax(dim=-1) / 6.0 / gs).clamp(max=448.0)
    s_code = s.to(torch.float8_e4m3fn).view(torch.uint8)
    denom = (s_code.view(torch.float8_e4m3fn).float() * gs).unsqueeze(-1)
    q = torch.where(denom > 0, blocks / torch.where(denom > 0, denom, 1.0), 0.0)
    return e2m1_quantize(q).reshape(R, K), s_code, gs


def quant_mxfp4(x):
    R, K = x.shape
    blocks = x.reshape(R, K // 32, 32)
    amax = blocks.abs().amax(dim=-1)
    e = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126))) - 2  # E2M1 emax = 2
    code = (e + 127).clamp(0, 254).to(torch.uint8)
    q = blocks / torch.pow(2.0, code.float() - 127).unsqueeze(-1)
    return e2m1_quantize(q).reshape(R, K), code


def make_inputs(fmt, kind, M, N, K, seed):
    torch.manual_seed(seed)
    B = FMT[fmt]["block"]
    KB = K // B
    if kind == "uniform":
        a = torch.randint(0, 16, (M, K), dtype=torch.uint8)
        w = torch.randint(0, 16, (N, K), dtype=torch.uint8)
        if fmt == "nvfp4":
            # all non-NaN UE4M3 codes, MSB (ignored by the kernel) included
            def sc(r):
                c = torch.randint(0, 254, (r, KB), dtype=torch.int32)
                return (c + (c >= 0x7F).to(torch.int32)).to(torch.uint8)
            a_sc, w_sc = sc(M), sc(N)
            alpha = f32(0.37)
        else:
            # E8M0 in [117,137] keeps every partial sum inside the fp32 normal range;
            # code 0 sprinkled (~10%) plus whole zero rows (a row 1, w row 2) that
            # separate NADPE's zero-scale rule from a 2^-127 reading
            def sc(r, zero_row):
                c = torch.randint(117, 138, (r, KB), dtype=torch.int32)
                c[torch.rand(r, KB) < 0.10] = 0
                if zero_row < r:
                    c[zero_row] = 0
                return c.to(torch.uint8)
            a_sc, w_sc = sc(M, 1), sc(N, 2)
            alpha = 1.0
    elif kind == "realistic":
        xa, xw = torch.randn(M, K), torch.randn(N, K)
        if fmt == "nvfp4":
            a, a_sc, ga = quant_nvfp4(xa)
            w, w_sc, gw = quant_nvfp4(xw)
            alpha = f32(f32(ga) * f32(gw))
        else:
            a, a_sc = quant_mxfp4(xa)
            w, w_sc = quant_mxfp4(xw)
            alpha = 1.0
    else:
        raise ValueError(kind)
    if fmt == "nvfp4":
        assert not bool(((a_sc & 0x7F) == 0x7F).any() or ((w_sc & 0x7F) == 0x7F).any())
    else:
        assert not bool((a_sc == 255).any() or (w_sc == 255).any())
    return a.contiguous(), w.contiguous(), a_sc.contiguous(), w_sc.contiguous(), alpha


def pack(codes):
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def swizzle(logical, rows_pad, pad_code):
    R, KB = logical.shape
    Rp, KBp = round_up(rows_pad, 128), round_up(KB, 4)
    P = torch.full((Rp, KBp), pad_code, dtype=torch.uint8)
    P[:R, :KB] = logical
    S = P.reshape(Rp // 128, 4, 32, KBp // 4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(Rp, KBp)
    # self-check against the kernel's swizzle::read_scale index formula
    flat = S.reshape(-1)
    r = torch.arange(R).unsqueeze(1).expand(R, KB)
    kb = torch.arange(KB).unsqueeze(0).expand(R, KB)
    idx = kb % 4 + 4 * ((r % 128) // 32 + 4 * (r % 32 + 32 * (kb // 4 + (KBp // 4) * (r // 128))))
    assert torch.equal(flat[idx], logical), "swizzle mismatch"
    return S


def kernel_inputs(fmt, a, w, a_sc, w_sc, alpha, dev):
    M, K = a.shape
    N = w.shape[0]
    Np = round_up(N, N_ALIGN)
    pad = FMT[fmt]["pad_scale"]
    w_pad = torch.zeros((Np, K), dtype=torch.uint8)
    w_pad[:N] = w
    w_sc_pad = torch.full((Np, w_sc.shape[1]), pad, dtype=torch.uint8)
    w_sc_pad[:N] = w_sc
    A = pack(a).to(dev)
    Bm = pack(w_pad).to(dev)
    A_sf = swizzle(a_sc, M, pad).to(dev)
    B_sf = swizzle(w_sc_pad, Np, pad).to(dev)
    if fmt == "nvfp4":
        A_sf, B_sf = A_sf.view(torch.float8_e4m3fn), B_sf.view(torch.float8_e4m3fn)
    al = torch.tensor([alpha], dtype=torch.float32, device=dev)
    return dict(A=A, B=Bm, A_sf=A_sf, B_sf=B_sf, alpha=al, M=M, N=N, Np=Np)


def run_case(ext, fmt, ki, case):
    out = torch.empty((ki["M"], ki["Np"]), dtype=torch.bfloat16, device=ki["A"].device)
    out.view(torch.int16).fill_(SENTINEL)
    if fmt == "nvfp4":
        ext.mma_emu_scaled_nvfp4_mm(out, ki["A"], ki["B"], ki["A_sf"], ki["B_sf"], ki["alpha"],
                                    case["algorithm"], case["f_bits"], case["g_bits"])
    else:
        ext.mma_emu_scaled_mxfp4_mm(out, ki["A"], ki["B"], ki["A_sf"], ki["B_sf"],
                                    case["algorithm"], case["f_bits"], case["g_bits"],
                                    case["group_size"], case["chunk_size"])
    err = ext.last_cuda_error()
    torch.cuda.synchronize()
    if err:
        raise RuntimeError(f"kernel launch failed for {fmt} {case}: {err}")
    full = out.view(torch.int16).cpu()
    untouched = int((full == SENTINEL).sum())
    if untouched:
        raise RuntimeError(f"bypass: {untouched} outputs never written for {fmt} {case}")
    return full[:, :ki["N"]].contiguous().clone()


def exact_ref(fmt, a, w, a_sc, w_sc, alpha):
    """float64 value of alpha * sum_k a*w*sa*sw under NADPE's scale rules."""
    def dec(c):
        mag = E2M1_VALUES.double()[(c & 7).long()]
        return torch.where((c & 8) != 0, -mag, mag)

    def sval(c):
        if fmt == "nvfp4":
            cm = (c & 0x7F).long()
            e, m = cm >> 3, cm & 7
            normal = (8 + m).double() / 8 * torch.pow(2.0, (e - 7).double())
            return torch.where(e == 0, m.double() / 8 * 2.0 ** -6, normal)
        return torch.where(c == 0, torch.zeros_like(c, dtype=torch.float64),
                           torch.pow(2.0, c.double() - 127))
    B = FMT[fmt]["block"]
    xa = dec(a) * sval(a_sc).repeat_interleave(B, dim=1)
    xw = dec(w) * sval(w_sc).repeat_interleave(B, dim=1)
    return (xa @ xw.t()) * alpha


def cos_of(bits, ref):
    y = bits.view(torch.bfloat16).double()
    den = float(y.norm() * ref.norm())
    return (float((y * ref).sum()) / den if den > 0 else float("nan"),
            bool(torch.isfinite(y).all()), float((y - ref).abs().max()))


def collect_env(so_path, src_dir):
    src = {rel: sha256_file(os.path.join(src_dir, rel)) for rel in FP4_SRC_FILES}
    combined = hashlib.sha256("".join(f"{src[k]}  {k}\n" for k in sorted(src)).encode()).hexdigest()
    import subprocess
    nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True,
                          text=True).stdout.strip().splitlines()[-2]
    props = torch.cuda.get_device_properties(0)
    env = dict(gpu=props.name, gpu_capability=f"{props.major}.{props.minor}",
               gpu_uuid=smi("uuid"), driver=smi("driver_version"),
               clocks_sm_mem_mhz=smi("clocks.sm,clocks.mem"),
               torch=str(torch.__version__), cuda=torch.version.cuda, nvcc=nvcc,
               ext_so_sha256=sha256_file(so_path),
               build_flags="-O2 -std=c++17 --expt-relaxed-constexpr "
                           "-gencode=arch=compute_80,code=sm_80 -U__CUDA_NO_* x4 -DENABLE_FP8 "
                           "+ torch COMMON_NVCC_FLAGS",
               nadpe_src_origin="micro26-ae-main (zip, no git metadata): "
                                "csrc/quantization/mma_emu, compiled unmodified",
               nadpe_src_sha256=src, nadpe_src_sha256_combined=combined)
    assert all(type(v) in (str, dict) for v in env.values())
    return env


def generate(args, ext, env, dev):
    os.makedirs(args.out, exist_ok=True)
    files, summary = [], []
    total_mismatch = n_runs = 0
    t0 = time.time()
    i = 0
    for fmt in ("nvfp4", "mxfp4"):
        cases = fp4_cases(fmt)
        for kind in INPUT_SETS:
            for (M, N, Kreq) in REQ_SHAPES:
                K = round_up(Kreq, FMT[fmt]["k_align"])
                seed = 2234 + i
                i += 1
                a, w, a_sc, w_sc, alpha = make_inputs(fmt, kind, M, N, K, seed)
                ki = kernel_inputs(fmt, a, w, a_sc, w_sc, alpha, dev)
                ref = exact_ref(fmt, a, w, a_sc, w_sc, alpha)
                sweeps = [[run_case(ext, fmt, ki, c) for c in cases] for _ in range(2)]
                n_runs += 2 * len(cases)
                rec, stats = [], []
                for c, b1, b2 in zip(cases, *sweeps, strict=False):
                    mism = int((b1 != b2).sum())
                    total_mismatch += mism
                    cos, finite, maxd = cos_of(b1, ref)
                    rec.append(dict(c, out_bits=b1))
                    stats.append(dict(c, repeat_mismatch=mism, y_finite=finite,
                                      cos_vs_exact=cos, max_abs_diff_vs_exact=maxd))
                fname = f"fp4_{fmt}_{kind}_{M}x{N}x{K}.pt"
                path = os.path.join(args.out, fname)
                blk = FMT[fmt]["block"]
                torch.save(dict(
                    a_codes=a, w_codes=w, a_scale_codes=a_sc, w_scale_codes=w_sc,
                    block_size=blk, scale_format=FMT[fmt]["scale_format"],
                    alpha=alpha, alpha_bits=f32_bits(alpha), cases=rec, env=env,
                    meta=dict(format=fmt, set=kind, seed=seed, M=M, N=N, K=K, K_requested=Kreq,
                              N_pad=ki["Np"], out_dtype="bfloat16", packing=PACK_DOC,
                              scale_swizzle=SWIZZLE_DOC, n_padding=NPAD_DOC,
                              epilogue=EPILOGUE[fmt], tiling=TILING[fmt],
                              scale_rules=SCALE_RULES[fmt],
                              unused_params="parameters the algorithm does not read are 0",
                              generator=_gen_doc(fmt, kind)),
                ), path)
                back = torch.load(path, weights_only=True)
                assert all(torch.equal(x["out_bits"], y["out_bits"])
                           for x, y in zip(back["cases"], rec, strict=False))
                cl = [s["cos_vs_exact"] for s in stats]
                files.append(dict(file=fname, bytes=os.path.getsize(path),
                                  sha256=sha256_file(path), cases=len(rec), format=fmt,
                                  set=kind, seed=seed, M=M, N=N, K=K, K_requested=Kreq,
                                  N_pad=ki["Np"], block_size=blk, alpha=alpha))
                summary.append(dict(file=fname, min_cos=min(cl), max_cos=max(cl),
                                    all_finite=all(s["y_finite"] for s in stats),
                                    zero_outputs=sum(int((r["out_bits"] == 0).sum()) for r in rec),
                                    repeat_mismatch=sum(s["repeat_mismatch"] for s in stats),
                                    per_case=stats))
                print(f"VEC {fname} cases={len(rec)} bytes={os.path.getsize(path)} "
                      f"cos[min,max]=[{min(cl):.6f},{max(cl):.6f}] "
                      f"repeat_mismatch={summary[-1]['repeat_mismatch']}", flush=True)
    _merge_manifest(args.out, env, files)
    result = dict(mode="gen", kernel_runs=n_runs, repeat_bit_mismatch_total=total_mismatch,
                  seconds=round(time.time() - t0, 1), files=files, summary=summary)
    return result, total_mismatch == 0


def _gen_doc(fmt, kind):
    if kind == "uniform":
        if fmt == "nvfp4":
            return ("E2M1 codes uniform 0..15; UE4M3 scale codes uniform over the 254 non-NaN "
                    "codes (MSB set included); alpha = fp32(0.37)")
        return ("E2M1 codes uniform 0..15; E8M0 codes uniform in [117,137], ~10% set to 0, "
                "a row 1 and w row 2 all 0")
    if fmt == "nvfp4":
        return ("randn -> global scale amax/(448*6) -> per-16 block scale amax/6/gs as "
                "float8_e4m3fn (RNE) -> x/(s*gs) to nearest E2M1 (ties low, sat 6); "
                "alpha = fp32(gs_a*gs_w)")
    return ("randn -> per-32 block E8M0 = floor(log2(amax)) - 2 + 127 -> x/2^(code-127) to "
            "nearest E2M1 (ties low, sat 6)")


def _merge_manifest(out_dir, env, files):
    path = os.path.join(out_dir, "manifest.json")
    man = json.load(open(path)) if os.path.exists(path) else {}
    man["fp4"] = dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), env=env, files=files,
        cases_per_file={f: len(fp4_cases(f)) for f in FMT},
        total_cases=sum(f["cases"] for f in files),
        grid=dict(nvfp4="GDFS F in GDFS_F x G in {3,4,5,6} (GS=16 fixed) + CoFDA F in COFDA_F "
                        "(CS=16 fixed); no C-decoupled (entry rejects algorithm 3)",
                  mxfp4="GDFS GS in {16,32} x F in GDFS_F x G in {3,4,5,6} + CoFDA CS in "
                        "{16,32} x F in COFDA_F; no C-decoupled (entry rejects algorithm 3)",
                  gdfs_f=GDFS_F, cofda_f=COFDA_F, fp4_g=FP4_G,
                  requested_shapes=REQ_SHAPES,
                  k_rule="K rounded up to the entry's multiple: nvfp4 32, mxfp4 64"),
        packing=PACK_DOC, scale_swizzle=SWIZZLE_DOC, n_padding=NPAD_DOC,
        epilogue=EPILOGUE, tiling=TILING, scale_rules=SCALE_RULES)
    with open(path, "w") as f:
        json.dump(man, f, indent=1)


def verify(args, ext, env, dev):
    paths = sorted(glob.glob(os.path.join(args.out, "fp4_*.pt")))
    assert paths, "no fp4 vector files"
    total_cases = total_mismatch = 0
    rows = []
    t0 = time.time()
    for path in paths:
        d = torch.load(path, weights_only=True)
        fmt = d["meta"]["format"]
        assert d["env"]["ext_so_sha256"] == env["ext_so_sha256"], "extension .so changed"
        assert d["a_scale_codes"].shape[1] * d["block_size"] == d["a_codes"].shape[1]
        ki = kernel_inputs(fmt, d["a_codes"], d["w_codes"], d["a_scale_codes"],
                           d["w_scale_codes"], d["alpha"], dev)
        bad = 0
        for c in d["cases"]:
            bits = run_case(ext, fmt, ki, c)
            m = int((bits != c["out_bits"]).sum())
            total_mismatch += m
            bad += int(m > 0)
            total_cases += 1
        rows.append(dict(file=os.path.basename(path), cases=len(d["cases"]), mismatched_cases=bad))
        print(f"VERIFY {os.path.basename(path)} cases={len(d['cases'])} mismatched_cases={bad}",
              flush=True)
    result = dict(mode="verify", files=len(paths), cases=total_cases,
                  bit_mismatch_total=total_mismatch, seconds=round(time.time() - t0, 1), rows=rows)
    return result, total_mismatch == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "verify"])
    ap.add_argument("--so", default=os.path.join(HERE, "build_fp4", "nadpe_mma_emu_fp4.so"))
    ap.add_argument("--src", default=os.path.join(HERE, "mma_emu"))
    ap.add_argument("--out", default=os.path.join(HERE, "vectors"))
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    os.makedirs(args.run_dir, exist_ok=True)
    assert torch.cuda.is_available(), "needs a GPU (launch via gpu_cap.sh)"
    dev = torch.device("cuda", 0)
    ext = load_ext(args.so)
    env = collect_env(args.so, args.src)
    with open(os.path.join(args.run_dir, f"env_fp4_{args.mode}.json"), "w") as f:
        json.dump(dict(env, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                  f, indent=1)
    print("ENV " + json.dumps({k: env[k] for k in ("gpu", "driver", "torch", "cuda", "nvcc",
                                                   "ext_so_sha256")}), flush=True)
    fn = generate if args.mode == "gen" else verify
    try:
        result, ok = fn(args, ext, env, dev)
        rc = 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result, rc = dict(mode=args.mode, error=repr(e)), 2
    with open(os.path.join(args.run_dir, f"result_fp4_{args.mode}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("RESULT " + json.dumps({k: v for k, v in result.items()
                                  if k not in ("summary", "files", "rows")}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
