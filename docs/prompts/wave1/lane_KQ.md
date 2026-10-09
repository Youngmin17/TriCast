<!-- 2026-09-30 실제 사용 원본. 경로·번호는 당시 기준 -->

[Lane KQ — Triton 양자화 커널]
소유 파일 (이것만 생성/수정):
  src/tricast/kernels/__init__.py, src/tricast/kernels/cast_core.py, src/tricast/kernels/quantize.py,
  tests/gpu/test_triton_quantize.py

로컬에는 triton/CUDA 가 없다. 커널은 Claude 가 A100/H100 에서 실행·검증한다. 그러므로:
  (1) Triton 3.4 API 기준으로 문법·타입을 매우 신중히 쓴다 (x.to(tl.int32, bitcast=True) 비트캐스트,
      tl.where, tl.math.div_rn, tl.randint, tl.constexpr 분기; 정수 시프트 폭은 항상 가드).
  (2) kernels/__init__.py 는 `import triton` 을 하고 실패하면 ImportError 를 그대로 올린다
      (tricast.quant.api.resolve_backend 가 이를 잡아 reference 로 폴백).
  (3) 레퍼런스(tricast.reference.cast.round_to_format, tricast.reference.quantize — Lane Q 동시 구현)와
      bit-exact 해야 한다.

구현:
1) cast_core.py — @triton.jit round_to_format(x (fp32 블록), 형식·모드 constexpr 들, noise, …) -> fp32.
   ENGINE §2 "Triton equivalent" 정수 비트 알고리즘: fp32 비트 → 지수·24bit significand,
   shift = 23 - mbits + max(0, emin - e) (25 로 clamp), kept/rem/half, 모드별 inc (SR 은 64bit 비교),
   kept+inc 와 q 로 fp32 비트를 정수 연산으로 재조립 (float 곱 2^q 금지 — FTZ), 오버플로/포화/특수값,
   subnormals=False 플러시, int 형식(frac, qmin/qmax, unsigned), pow2 형식. `.to(bf16).to(f32)` 같은
   왕복 반올림 흉내 금지. Python 헬퍼 fmt_constexprs(fmt, rounding, saturate) -> dict.
2) quantize.py
   - round_to_format_triton(x, fmt, rounding, *, saturate=True, noise=None, sr_bits=32) -> fp32 텐서.
   - quantize_triton(x, spec, *, amax=None, noise=None, seed=0) -> QTensor
     (from tricast.quant.qtensor import QTensor — §3.7 필드). [rows, K] 를 scale 도메인 단위로 처리:
     group/row 는 프로그램당 여러 row·그룹, tl.max 로 amax, 스케일 계산 (absmax: tl.math.div_rn(amax, M)
     → scale 형식 반올림은 cast_core 재사용; pow2_floor/ceil: 지수 비트로 정확히, microxcaling clamp 규칙),
     원소 = round(div_rn(x, s_eff)); two-level 은 텐서 amax·d2 를 torch 로 (fp32 RN) 먼저 구한 뒤 블록 커널;
     tensor/block granularity 도 지원. method ∈ {mse, percentile} 또는 zero_point != none 이면 스케일은
     tricast.reference.quantize.compute_scale 을 GPU 텐서로 호출해 구하고 원소 반올림만 Triton (docstring 에
     명시). noise 가 있으면 사용, 없으면 tl.randint(seed, offset).
   - values 는 tricast.formats.container_dtype(spec.format) 로 저장.
3) tests/gpu/test_triton_quantize.py (pytestmark = pytest.mark.gpu) — 레퍼런스와 bit-exact:
   레지스트리 전 형식 × 6 모드 (SR 은 같은 noise) × saturate on/off, conftest.wide_fp32 1e5 + 경계값;
   SCHEMES 전부 × 여러 shape (K 가 그룹 배수가 아닌 경우 포함) 에서 QTensor 의 values/scale/zero_point/
   global_scale 이 모두 같음; 1줄 성능 스모크 (GB/s 출력, assert 없음).

완료 기준 (로컬에서 가능한 것): 3개 파일이 `python -m py_compile` 통과, ruff 0, CPU 에서
`import tricast.kernels` 가 triton 부재로 ImportError. (GPU 실행·수정은 Claude 가 이어서 한다 —
커널 설계 의도와 가정을 docstring 에 짧게 남겨라.)
