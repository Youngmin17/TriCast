<!-- 2026-09-30 실제 사용 원본. 경로·번호는 당시 기준 -->

[Lane Q — 양자화 레퍼런스 + QTensor + API + Observer]
소유 파일 (이것만 생성/수정):
  src/tricast/quant/qtensor.py, src/tricast/quant/api.py, src/tricast/quant/observer.py,
  src/tricast/reference/quantize.py,
  tests/test_cast_reference.py, tests/test_quantize_reference.py, tests/test_observer.py,
  tests/test_microxcaling_parity.py

구현 (ENGINE.md §2, §3 을 문장 그대로):
1) src/tricast/reference/quantize.py
   - view_2d(x) -> fp32 [rows, K]; scale 레이아웃은 §3.7 (tensor [], row [rows,1], group [rows, ceil(K/G)],
     block [ceil(rows/br), ceil(K/bc)]). 짧은 가장자리 도메인은 유효 원소만 사용. NaN 전파.
   - compute_amax(x2d, spec) -> amax (scale 레이아웃).
   - compute_scale(x2d, spec, amax=None) -> tuple[scale, zero_point | None, global_scale | None]:
     §3.3 (absmax, pow2_floor — microxcaling 과 동일한 clamp: e > emax_SF → NaN scale, e < emin_SF → emin_SF,
     amax=0 이면 2^-126 으로 간주; pow2_ceil; percentile; mse — search 비었으면 linspace(1.0, 0.5, mse_grid),
     동률은 가장 앞 r; search=(1.0,1.5) = Four-over-Six), §3.4 two-level (TensorRT-ModelOpt 순서),
     §3.5 zero point. 나눗셈·곱셈은 fp32 RN (torch fp32 연산은 CPU/CUDA 모두 IEEE RN).
   - expand_scale(t, spec, rows, K) -> [rows, K] 원소별 전개 (scale 과 zero_point 모두에 사용).
   - quantize_elements(x2d, spec, scale_pe, zp_pe=None, global_scale=None, noise=None) -> fp32 values:
     §3.5/§3.6 원소 반올림 (scale 이 주어졌을 때). **GPTQ 레인이 이 시그니처 그대로 사용 — 고정.**
   - quantize_reference(x, spec, *, amax=None, noise=None) -> QTensor. amax override 는 tensor granularity 전용
     (observer 용). spec.scale is None → 직접 캐스트.
2) src/tricast/quant/qtensor.py — QTensor dataclass, §3.7 그대로: values, scale, zero_point, global_scale,
   spec, shape. 속성 rows, K; scale_per_element(); dequantize(dtype=torch.float32) → 원래 shape;
   mma_operand() → scaled 모드: values fp32 / dequant 모드: round_to_format(x̂, dequant_format, RNE,
   saturate=False); to(device). values 는 tricast.formats.container_dtype(spec.format) dtype 으로 저장
   (값이 격자 위라 정확).
3) src/tricast/quant/observer.py — ObserverState(spec: ObserverSpec), §3.11:
   mode "calibrate" | "frozen"; observe(x) -> 이번 호출에 쓸 amax (fp32 scalar tensor) — 모드·kind 에 따라
   상태 갱신 포함 (minmax 누적 max; ema: 첫 호출 초기화 후 decay 식; history: delayed — 이전 기록으로
   reduce(max|most_recent), 첫 호출은 자기 amax, 사용 후 append, 길이 ≤ history_len, frozen 에서도 계속 갱신;
   percentile/mse: |x| 저장소(≤ max_samples, torch.Generator seed 0 균일 부분표본) → freeze 때 계산).
   freeze(quant_spec) — static amax 확정 (mse 는 compute_scale 탐색을 저장소에 적용해 최적 r·amax).
   static_amax 속성; state_dict()/load_state_dict().
4) src/tricast/quant/api.py
   - resolve_backend(backend: str, *tensors) -> "triton" | "reference": "auto" → 모든 텐서가 CUDA 이고
     `import tricast.kernels` 성공 시 triton, 아니면 reference. CPU 텐서에 "triton" 명시 → ValueError.
   - quantize(x, spec, *, backend="auto", amax=None, noise=None) -> QTensor
     (triton: `from tricast.kernels.quantize import quantize_triton` 지연 import).
   - fake_quant(x, spec, *, backend="auto", amax=None, noise=None) -> Tensor (x.dtype): mma_operand 와
     같은 값을 dequantize 해 반환 (dequant 모드는 dequant_format 반올림 포함). torch.autograd.Function 으로
     clipped STE (§3.8): |x / s_eff| ≤ max_elem (포화 안 됨) 인 곳만 grad 통과, scale 로 grad 없음.
   spec 은 QuantSpec 또는 스킴 이름 문자열(get_scheme) 허용.

테스트 (작은 크기, seed 고정):
 - tests/test_cast_reference.py — round_to_format / decode L0:
   torch parity (float8_e4m3fn, float8_e5m2, float8_e4m3fnuz, float8_e5m2fnuz, bfloat16, float16;
   RNE, saturate=False) on conftest.wide_fp32 (≥1e5) + 경계값; 작은 형식(fp4, fp6 두 종, e4m3, int4,
   mxint8, uint4, 사용자 정의 e3m4:none)은 격자를 전부 열거해 만든 brute-force 최근접 오라클
   (fractions.Fraction) 과 6개 모드(RNE/RNA/RTZ/RUP/RDN, SR 은 명시 noise) 결정이 전부 일치;
   오버플로 정책 (ieee: 모드별 inf/max, fn/fnuz: NaN, none: max; saturate=True: max);
   subnormals=False 플러시; pow2(E8M0) 반올림; SR 통계 (20000회 평균이 x 의 4σ 이내) 와 sr_bits 효과;
   decode() 가 fp8/fp6/fp4/int/pow2 격자에서 정확히 복원.
 - tests/test_quantize_reference.py — 모든 granularity × scale method × two-level × zero point ×
   mse/percentile/Four-over-Six 를 손계산 기대값으로 (예: MXFP4 블록 [값들] → scale 2^k 와 원소를 명시);
   shape 왕복, NaN 도메인(pow2 overflow), 짧은 그룹, container dtype 정확성, dequantize, dequant 모드
   mma_operand, 스킴 전부(SCHEMES)에 대해 quantize→dequantize 동작, fake_quant 의 STE 기울기.
 - tests/test_observer.py — kind 별 의미 (ema 수식, history delayed / most_recent, minmax, percentile, mse),
   freeze, state_dict 왕복.
 - tests/test_microxcaling_parity.py — pytestmark = pytest.mark.oracle; mx = pytest.importorskip("mx").
   elem ∈ {fp8_e4m3, fp8_e5m2, fp6_e3m2, fp6_e2m3, fp4_e2m1, int8→mxint8, int4→mxint4},
   round ∈ {"even"→RNE, "nearest"→RNA, "floor"→RTZ} (tricast.rounding.from_microxcaling):
   mx.mx_ops._quantize_mx(x, 8, elem, shared_exp_method="max", axes=[-1], block_size=32, round=r)
   vs QuantSpec(elem, "group", group_size=32, scale=ScaleSpec(E8M0, "pow2_floor"), rounding=...)
   의 dequantize(). + mx.elemwise_ops._quantize_elemwise_core(A, mbits, ebits, max_norm, round,
   saturate_normals=True) vs round_to_format(saturate=True).
   입력: randn * 10**U(-3,3), 0 포함, 음수, 짧은 블록. 기대: 완전 일치. 단, microxcaling 이 fp32
   torch.log2 로 지수를 구해 2의 거듭제곱 바로 아래 값을 잘못 분류하는 경우는 (frexp 기반 정확 지수 vs
   torch.floor(torch.log2(x)) 비교로) 찾아 제외하고, 제외 비율 < 1e-3 을 assert 하며 개수를 출력.

완료 기준: 위 4개 테스트 파일 전부 PASS (oracle 포함 — PYTHONPATH 에 microxcaling), ruff 경고 0.
