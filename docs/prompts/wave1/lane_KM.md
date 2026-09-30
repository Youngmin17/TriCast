
[Lane KM — Triton MMA 커널]
소유 파일 (이것만 생성/수정):
  src/tricast/kernels/mma_core.py, src/tricast/kernels/mma.py, tests/gpu/test_triton_mma.py
(src/tricast/kernels/__init__.py 는 Lane KQ 소유 — 건드리지 말 것. 네 모듈은 `import triton` 을 직접 하고
 실패 시 ImportError 를 그대로 올린다.)

로컬에 triton/CUDA 없음 → 커널은 Claude 가 GPU 에서 검증·수정한다. Triton 3.4 기준, 매우 신중히.
레퍼런스 = tricast.reference.mma.gemm_reference (Lane M 동시 구현) 와 bit-exact 가 목표.

구현 (ENGINE §4, §5):
1) mma_core.py — @triton.jit 헬퍼:
   decode_f32(v, MBITS, EMIN, IS_INT, FRAC) -> (neg, e, m) (format-native 비정규화 규약, §4.1);
   c_operand(c_f32, F) (NADPE fp32_to_operand<F>; 서브노멀 정규화 — clz 필요 시
   tl.extra.cuda.libdevice.clz / 또는 정수 비트 트릭); fixed_to_f32(S_int64, Emax, F, NORM_RNE)
   (NADPE fixed_to_fp32<F> + rne 확장, fp32 비트는 정수 조립); 모든 가변 시프트는
   tl.where(d > 63, 0, x >> tl.minimum(d, 63)) 식 가드 (LLVM poison 방지).
2) mma.py — gemm_triton(a: Operand, b: Operand, spec: MMASpec, bias=None) -> Tensor [M, N]
   (Operand = tricast.mma.operand.Operand; scale 레이아웃은 그 docstring).
   - Python: K-major 로 전치·연속화 (a_t [K, M], b_t [K, N]); scale 종류(none/tensor/row/k)·k_domain 을
     커널 인자로; 형식 constexpr; scale_apply 는 tricast.reference.mma.resolve_scale_apply 를 import 해
     레퍼런스와 같은 규칙으로 해석.
   - 커널 cofda (fused / decoupled(F2) / promote_interval): 청크마다 스트리밍 2-pass —
     pass1 = tl.static_range(CS) 로 a_k[BM], b_k[BN] 로드 → decode → e_p·nz·specials → Emax,
     pass2 = 재로드 → m_p = m_a*m_b (int64) → radix F 시프트 → 정렬 시프트(가드) → ± 누적 → c 규칙(§4.3) →
     fixed_to_f32. K-varying scale: product 레벨 (scale significand 곱·지수 합) 또는 promote (구간 P 를
     fp32 fma(P, s_a*s_b, acc)). 마지막 청크 zero padding 은 mask 로.
   - 커널 gdfs: 타일(KT)마다 그룹(GS) 결과를 최대 8개 constexpr-unroll 레지스터(S0..S7, E0..E7)에 보관 →
     그룹 operand (scale 곱, E8M0 field-0 규칙) → 타일 FDA.
   - 커널 fp32_fma (tl.fma 순차), fp64 (fp64 fma), int_exact (int64 합).
   - epilogue: s_a·(s_b·acc) CUTLASS 순서, alpha, bias(fp32), 출력 bf16/fp16 저장은 .to(dtype) (RNE, 실제
     저장이므로 허용), NaN/Inf 규칙.
   - 타일 BM/BN ∈ {16, 32, 64}, num_warps ∈ {2, 4, 8} — 소수 config autotune (key: M, N, K, 모드).
     결과는 타일과 무관해야 한다 (각 출력 원소 독립).
3) tests/gpu/test_triton_mma.py (pytestmark = pytest.mark.gpu) — gemm_reference 와 bit-exact:
   알고리즘 5종 × 형식 (fp8 e4m3/e5m2, fp6 두 종, fp4, bf16, fp16, int8, mxint8) × scale 종류
   (tensor, row, group MX/NV, block) × F ∈ {3,7,13,23,25,35} × CS ∈ {8,16,32} × c_mode × norm_rounding ×
   promote; 무작위 shape (M, N ≤ 96, K ≤ 512, K 가 CS 배수 아님 포함); 서브노멀·NaN·Inf 주입 사례;
   1줄 성능 스모크 (M=N=K=1024 hopper preset → emulated TMAC/s 출력, assert 없음).
   파라미터 조합이 폭발하지 않게 pytest.mark.parametrize 를 대표 조합 ~150개 이내로 구성.

완료 기준 (로컬): 3개 파일 `python -m py_compile` 통과, ruff 0. 커널 설계 의도·가정을 docstring 에 짧게.
