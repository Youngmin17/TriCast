
[Lane M — MMA 레퍼런스 + gemm API]
소유 파일 (이것만 생성/수정):
  src/tricast/reference/mma.py, src/tricast/mma/api.py,
  tests/test_mma_reference.py, tests/test_mma_invariants.py

구현 (ENGINE.md §4 를 문장 그대로; NADPE CUDA 코드를 줄 단위로 대조):
1) src/tricast/reference/mma.py
   - gemm_reference(a: Operand, b: Operand, spec: MMASpec, bias: Tensor | None = None) -> Tensor [M, N]
     (dtype = spec.out_format 의 torch dtype: bf16/fp16/fp32). Operand = tricast.mma.operand.Operand.
   - resolve_scale_apply(spec, a, b) -> str: §4.6 의 auto 해석 (불가능 조합은 ValueError 에 이유 명시).
     Triton 레인이 이 함수를 import 해 같은 규칙을 쓰므로 public 으로 둔다.
   - decode 는 tricast.reference.cast.decode 사용 (NaN/Inf 는 decode 전에 분리).
   - 정확 int64 벡터화: 청크마다 [M, N, n_terms] 텐서. 테스트 크기(M,N ≤ 64, K ≤ 1024)에서 CPU 수 초.
   - fda() §4.3 = NADPE chunked_accumulate<F,N> + fixed_to_fp32<F>: fp32 누산기 c 의 operand 변환
     (fp32_to_operand<F>: 서브노멀 정규화, radix 23→F), NaN/Inf 규칙, Emax, shift ≥ 64 → 0, to_fp32
     (rtz 기본 / rne 확장 — §4.3 5단계), 서브노멀·오버플로 경로. fp32 비트는 int 로 조립(torch int32 view).
   - cofda: fused / decoupled(F2) / promote_interval (fp32 fma 는 정확히 한 번 반올림: fp64 곱(정확) +
     TwoSum 보정 반올림으로 구현; "fp64 덧셈 후 fp32 반올림"은 이중 반올림이라 금지).
   - gdfs §4.5: 그룹 합(G radix), 그룹 operand 에 scale significand 곱·지수 합 (UE4M3, E8M0, 그 외 float
     scale 일반화), E8M0 field-0(2^-127) = 0 기여 규칙, zero/NaN 규칙, 타일당 KT/GS 그룹을 FDA 1회.
   - fp32_fma (정확 fma 체인), fp64 (fp64 FMA 체인 → fp32 RNE), int_exact.
   - scale 적용 §4.6 (product / group / promote / operand / epilogue), epilogue §4.9 (s_a·(s_b·acc) CUTLASS
     순서, alpha, fp32 bias, 출력 round_to_format(out_format, RNE, saturate=False)).
   - int64 headroom 검증 §4.5 (초과 시 ValueError).
   - 곱 §4.2 (F < R 이면 right shift 절단; zero 판정은 operand zero 로만).
2) src/tricast/mma/api.py
   - as_operand(x: QTensor | Tensor) -> Operand. QTensor 는 Lane Q 가 동시에 구현 중 (§3.7 필드: values,
     scale, zero_point, global_scale, spec, shape). scaled 모드: values fp32 + scale 레이아웃
     (tensor → "tensor", row → "row", group → "k" with k_domain=group_size and scale [rows, ceil(K/G)],
     block → "k" with k_domain=bc and scale 을 행 방향으로 전개한 [rows, ceil(K/bc)]); two-level → alpha =
     global_scale; dequant 모드 → values = qt.mma_operand(), fmt = spec.dequant_format, scale 없음;
     plain tensor → values fp32, fmt = tricast.formats.format_of_dtype(x.dtype).
   - gemm(a, b, spec: MMASpec | str | dict, *, bias=None, backend="auto") -> Tensor: a [..., K] 활성,
     b [N, K] 가중치 (nn.Linear 레이아웃) → [..., N]. spec 문자열 = preset 이름, dict = MMASpec.from_dict.
     backend auto → CUDA 텐서이고 `tricast.kernels.mma` import 성공 시 gemm_triton(a_op, b_op, spec, bias),
     아니면 gemm_reference. Operand 를 직접 받아도 동작.

테스트:
 - tests/test_mma_reference.py — (a) 손으로 유도한 NADPE 사례를 비트 단위 기대값으로 (계산 과정을 주석
   1~2줄로): fp8 e4m3 곱 몇 개 + c, F=3/13/25 절단, c-fused vs decoupled 가 다른 사례, 서브노멀 입력,
   NaN / Inf / +Inf·-Inf, 전부 0 인 청크 (c 그대로), shift ≥ 64, 오버플로 → inf, 결과가 서브노멀,
   norm_rounding rne; (b) GDFS: NVFP4 (e2m1 + ue4m3 + alpha) 와 MXFP4 (e8m0, field-0 규칙) 그룹 사례;
   (c) promote_interval (DeepSeek 방식) 사례; (d) epilogue 순서·bias·출력 반올림; (e) 잘못된 조합의 ValueError.
 - tests/test_mma_invariants.py — F 가 충분히 클 때 (fp8 곱, F=40, CS ≥ K) cofda == 정확 합(Fraction)을 fp32 로
   RZ 한 값; F 감소 시 평균 오차 비감소 경향; fp32_fma == Fraction 기반 정확-한번-반올림 fma 순차 체인;
   fp64 == fp64 순차 체인; int_exact == 정수 합; 출력 원소 독립성 (행/열 순열 등가); as_operand 레이아웃.

완료 기준: 두 테스트 파일 PASS, ruff 0.
