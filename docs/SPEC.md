# 제품 스펙 (SDD, 강의 4)

> 이 문서가 "무엇을 만드는가"의 단일 진실 공급원이다. 수치 의미론의 정본은 `docs/design/ENGINE.md`,
> 도메인 구조의 정본은 `docs/ontology.yaml`, 근거의 원천은 `docs/research/interviews.md` 의 로그 번호다.
> 상태 (2026-10-01): 팀 인터뷰 10건을 반영했다. 구현 상태는 `support_matrix.yaml` 이 원천이다.

## 1. 문제
저정밀 모델과 NPU 연산기를 설계·검증하는 사람이 정밀도·누산·희소성·이상치 처리 조합이나 GPU 세대를 바꿀
때마다 그 효과를 다시 구현하지 않고는 확인하지 못해, 조합마다 코드와 CUDA 커널을 새로 쓰고 검증한다.
(상세: `docs/PROBLEM.md`, 로그 1, 3, 9, 10)

## 2. 타깃 사용자
양자화·저정밀도 학습 연구자 (로그 1, 3), NPU 연산기 개발자 (로그 9), LLM 경량화 연구자 (로그 10).
비대상: ONNX 변환·기기 배포 엔지니어 (로그 2), 경량화 혜택을 기대하는 일반 사용자 (로그 5–8), 서빙 처리량
최적화 엔지니어.

## 3. 핵심 기능 (한 문장)
정밀도·양자화·누산 알고리즘·희소성·이상치 보존을 **레시피 값으로** 바꾸면, 커널을 다시 쓰지 않고 실제
연산기와 **비트 단위로 같은 산술**로 Hugging Face 모델을 돌려 **모델 품질(PPL·벤치마크)과 누산기 ULP 오차를
근거(환경·출처)와 함께** 돌려준다.

## 4. 범위

- 포함: 임의 수 형식(float ExMy / int / E8M0)과 반올림 6종; 양자화 (tensor/row/group/block, BFP·MX·NVFP4,
  zero point, 스케일 방법 5종, two-level, observer: EMA·delayed history 등); MMA 누산 (CoFDA fused/decoupled,
  GDFS, DeepSeek 방식 승격, fp32 FMA, fp64, 정수); 가중치 희소성 (N:M, 비율) 과 이상치 고정밀 보존 (로그 10);
  변환 (Hadamard, RHT, SmoothQuant, AWQ); 가중치 알고리즘 (RTN, GPTQ); KV 캐시 양자화 (KIVI); HF 모델 linear
  레이어 패치; WikiText-2 PPL; lm-eval 과제; 레이어별 오차 리포트 (MSE·SQNR·코사인·logits KL·누산기 ULP —
  로그 9); 레시피 검증과 스윕; 결과·환경 기록; 자연어 → 레시피 파싱 에이전트
- 비포함: ONNX 등 배포 형식 변환과 실제 기기 실행 (로그 2), 칩 면적·전력·타이밍 추정 (로그 9·10 의 목표지만
  출처 없는 하드웨어 수치는 만들지 않는다), 클라우드·데이터센터 운영, 모델 서비스 배포, CUDA 커널 자동 생성,
  RTL 생성, 대규모 학습 (QAT 는 STE 수준), attention `QKᵀ`·`PV` matmul 에뮬레이션 (현재 — 로드맵)

## 5. 인터페이스

| 인터페이스 | 입력 | 출력 | 상태 |
|---|---|---|---|
| `tricast ppl --model M --recipe R` | 모델 id, 레시피 이름·경로 | `{ppl, nll, n_tokens, env}` | 구현됨 — Qwen3-0.6B 24개 레시피 (`docs/results/qwen3_0.6b.md`) |
| `tricast eval --model M --recipe R --tasks a,b` | + lm-eval 과제 | lm-eval 결과 + env | 구현됨 — HellaSwag·CoQA |
| `python -m tricast.eval.lmeval --model tricast --model_args pretrained=M,recipe=R --tasks …` | lm-eval 표준 인자 | lm-eval 표준 | 구현됨 |
| `tricast report --model M --recipe R` | 모델, 레시피 | 레이어별 MSE·SQNR·코사인·누산기 ULP + 모델 logits KL·top-1 일치 (JSON + md) | 구현됨 (ULP: 2026-10-01) |
| `tricast run CONFIG.yaml` | 모델·레시피·과제 목록·스윕 | 레시피별 JSON + summary.md | 구현됨 |
| Python `tricast.quantize / gemm / patch_model` | 텐서·스펙 | QTensor·텐서·패치 보고 | 구현됨 |
| `tricast agent "<자연어>"` | 자연어 요청 | `EmulationRequest` JSON → 실행 계획 → 결과 보고 | 구현됨 — 가짜 모델 client 로 테스트, 실제 Claude API 경로는 미검증 |

레시피의 희소성·이상치 필드 (가중치에 적용):

```yaml
defaults:
  sparsity: {kind: "n:m", n: 2, m: 4}        # 또는 {kind: unstructured, ratio: 0.5}
  outliers: {fraction: 0.005, format: bf16}  # 크기 상위 0.5% 를 bf16 으로 따로 보존
```

## 6. 수용 기준 (AC, EARS — "[조건]일 때, TriCast 는 [동작]한다")

- AC1 [상시 적용]: TriCast 는 항상 Triton 커널 결과를 레퍼런스 구현과 비트 단위로 일치시킨다 — 한 원소라도
  다르면 실패. (`tests/gpu/`; 근거: 로그 1·9 의 "구현이 맞는지 확인")
- AC2 [상시 적용]: TriCast 는 항상 출처(provenance)가 기록된 하드웨어 프리셋만 제공한다. (로그 3·9 — 세대별
  하드웨어 지원 차이)
- AC3 [이벤트 기반]: 사용자가 평가를 실행하면, TriCast 는 지표와 함께 git SHA·모델 revision·데이터셋
  fingerprint·레시피 해시를 저장한다.
- AC4 [이벤트 기반]: bf16 passthrough 레시피로 평가하면, TriCast 는 패치하지 않은 모델과 같은 PPL (상대차
  ≤ 1e-3)을 보고한다. (측정: +0.004%)
- AC5 [예외 대응]: 레시피가 잘못되면, TriCast 는 실행 전에 거부하고 오류가 난 필드 경로를 알려 준다.
- AC6 [예외 대응]: 자연어 요청에 없는 정밀도·누산 파라미터가 필요하면, TriCast 에이전트는 값을 지어내지 않고
  기본값을 `assumptions` 에 적거나 되묻는다.
- AC7 [이벤트 기반]: 독립 구현(NADPE CUDA 커널)의 골든 벡터로 검증하면, TriCast 는 모든 케이스에서 비트
  일치한다. (`tests/data/nadpe/` — 레퍼런스는 `tests/test_golden.py`, Triton 은 GPU 에서
  `scripts/nadpe_oracle/check_triton.py`)
- AC8 [이벤트 기반]: 레시피에 `sparsity: {kind: "n:m", n: N, m: M}` 가 있으면, TriCast 는 각 출력 행의 K 축
  연속 M 개 가중치 가운데 크기가 큰 N 개만 남기고 (동률은 앞 인덱스) 나머지를 정확히 0 으로 만든 뒤
  양자화한다. (로그 10; `tests/test_sparsity_outliers.py`)
- AC9 [이벤트 기반]: 레시피에 `outliers: {fraction: f, format: F}` 가 있으면, TriCast 는 크기 상위 f 비율의
  가중치를 형식 F 로 따로 보존하고 나머지만 양자화하며, 두 경로의 합을 출력 형식으로 한 번만 반올림한다.
  (로그 10; `tests/test_sparsity_outliers.py`, `tests/gpu/test_triton_structure.py`)
- AC10 [이벤트 기반]: 사용자가 오차 리포트를 실행하면, TriCast 는 레이어마다 같은 양자화 피연산자를 fp64 로
  누산한 결과(fp64 합을 fp32 로 한 번 반올림한 뒤 같은 epilogue)와 비교한 누산기 ULP 오차를 출력 형식의 ULP
  단위로 보고하고, fp64 누산 레이어에서는 0 을 보고한다. (로그 9; `tests/test_analysis_ulp.py`)

> AC ↔ 골든 케이스: `tests/harness/golden_cases.yaml` (강의 5).
