<!-- 2026-09-30 실제 사용 원본. 경로·번호는 당시 기준 -->

[Lane A — 변환 (Hadamard / RHT / SmoothQuant / AWQ) + 가중치 알고리즘 (RTN / GPTQ)]
소유 파일 (이것만 생성/수정):
  src/tricast/transforms.py, src/tricast/weight_quant/__init__.py, src/tricast/weight_quant/gptq.py,
  tests/test_transforms.py, tests/test_gptq.py

구현 (ENGINE §3.9, §3.10):
1) src/tricast/transforms.py
   - @dataclass CalibStats: n_tokens, absmax [K] (max|X_j|), absmean [K] (mean|X_j|), xtx [K,K] fp64 | None,
     samples [n,K] | None (AWQ 탐색용 저장소, ≤ 4096 행, 균일 부분표본 seed 0).
     StatsCollector(K, want_xtx: bool, want_samples: bool): update(x2d), result() -> CalibStats.
   - hadamard(b) (Sylvester, 1/sqrt(b) 정규화), fwht(x, block) (마지막 축을 block 단위로 빠른 WHT, fp32),
     random_sign_diag(K, seed) (torch.Generator().manual_seed(seed)).
   - @dataclass LinearTransform(kind, block, diag): apply_activation(x) = x T; apply_weight(W) = W T^{-T};
     kind "none" 은 항등. hadamard: T=blockdiag(H_b) (직교, T^{-T}=T); random_hadamard: T = D·H;
     smoothquant/awq: T = diag(1/s) → W T^{-T} = W diag(s).
   - fit_transform(spec: TransformSpec, W [N,K], stats: CalibStats | None, *, weight_spec=None,
     act_spec=None) -> LinearTransform. block=0 → K 를 나누는 2의 거듭제곱 ≤ 128 중 최대.
     smoothquant: s_j = absmax_j^α / max|W_j|^(1-α), [1e-5, 1e5] clamp. awq: s_j = absmean_j^α 를
     sqrt(max·min) 으로 정규화, α ∈ linspace(0,1,grid) 중 ‖Q(W diag(s))(X/s)ᵀ − W Xᵀ‖² 최소 (X = stats.samples,
     Q = tricast.quant.api.fake_quant — Lane Q 동시 구현; weight_spec/act_spec 이 None 이면 그 쪽은 비양자화).
2) src/tricast/weight_quant/
   - __init__.py: quantize_weight(W, spec: QuantSpec, algo: WeightAlgoSpec, *, hessian=None,
     backend="reference") -> QTensor: rtn → tricast.quant.api.quantize; gptq → gptq().
   - gptq.py: gptq(W [N,K], H [K,K], spec, *, block_size=128, damp=0.01, act_order=False) -> QTensor.
     Frantar et al. (H 댐핑 damp·mean(diag H), H^{-1} 의 upper Cholesky, lazy batch 갱신, dead column(H_jj=0)
     처리). 열 반올림은 tricast.reference.quantize.quantize_elements(x2d, spec, scale_pe, zp_pe,
     global_scale) + compute_scale(...) (Lane Q, 시그니처 고정): tensor/row 스케일은 원본 W 로 1회,
     group 스케일은 그룹 첫 열에 도달할 때 현재 갱신된 W 의 그 그룹 열들로 계산, two-level global 은 원본
     W 로 1회, block(2-D) granularity 는 ValueError. act_order 는 static groups 방식 (그룹 스케일을 원래 열
     순서 기준으로 미리 계산). 결과 QTensor 는 quantize() 결과와 같은 레이아웃.

테스트:
 - tests/test_transforms.py: apply_activation(x) @ apply_weight(W).T ≈ x @ W.T (fp64 에서 상대오차 1e-10 —
   허용 오차 사용 이유: 변환은 실수 연산), hadamard 직교성, rht seed 결정론, smoothquant 공식, awq 가 grid 에서
   α 를 골라 α=0 대비 오차 비증가, block 자동 선택, StatsCollector 통계.
 - tests/test_gptq.py: 합성 선형층 (N=64, K=256, X 는 채널별 스케일이 다른 가우시안) 에서 GPTQ 의
   ‖(W−Ŵ)Xᵀ‖ 가 RTN 보다 작음 — int4 g32, mxfp4, nvfp4, fp8 row 각각; 결과 값이 격자 위(decode 성공);
   act_order, damp, 결정론.
Lane Q 모듈이 아직 없으면 pytest.importorskip 로 skip 되게 하되 올바르게 작성하고, 작업 막바지에 Lane Q 파일이
생겼으면 실제로 돌려 통과시킨다.

완료 기준: 두 테스트 PASS (의존 모듈 존재 시), ruff 0.
