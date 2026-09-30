# 제품 스펙 (SDD, 강의 4) — 초안 ✍️

> 이 문서가 "무엇을 만드는가"의 단일 진실 공급원이다. 수치 의미론의 정본은 `docs/design/ENGINE.md`,
> 도메인 구조의 정본은 `docs/ontology.yaml`. 수용 기준(AC)은 사람이 확정한다 — 아래 AC 는 초안이다.

## 1. 문제
저정밀 연산기 설계자가 수 형식·양자화·누산기 구조가 LLM 품질에 주는 영향을 설계 단계에서 믿을 수 있게,
빠르게 확인하지 못한다. (상세: `docs/PROBLEM.md`)

## 2. 타깃 사용자
연산기 RTL·마이크로아키텍처 설계자, 수 형식·양자화 방식을 설계하는 연구자.
비대상: 서빙 처리량 최적화 엔지니어, 대규모 학습 실무자, 일반 LLM 사용자.

## 3. 핵심 기능 (한 문장)
정밀도·양자화·누산기 구성을 **레시피(또는 자연어)**로 정의하면, 실제 연산기와 **비트 단위로 같은 산술**로
Hugging Face 모델을 돌려 **품질 수치(PPL·벤치마크)를 근거(환경·출처)와 함께** 돌려준다.

## 4. 범위

- 포함: 임의 수 형식(float ExMy / int / E8M0)과 반올림 6종; 양자화 (tensor/row/group/block, BFP·MX·NVFP4,
  zero point, 스케일 방법 5종, two-level, observer: EMA·delayed history 등); MMA 누산 (CoFDA fused/decoupled,
  GDFS, DeepSeek 방식 승격, fp32 FMA, fp64, 정수); 변환 (Hadamard, RHT, SmoothQuant, AWQ); 가중치 알고리즘
  (RTN, GPTQ); HF 모델 linear 레이어 패치; WikiText-2 PPL; lm-eval 과제; 레시피 검증과 스윕; 결과·환경 기록;
  자연어 → 레시피 파싱 에이전트 (강의 3·6에서 구현)
- 비포함: RTL 생성, 면적·전력·타이밍 추정, 서빙 처리량 측정, attention·KV 캐시 matmul 에뮬레이션 (현재 —
  로드맵), 대규모 학습

## 5. 인터페이스

| 인터페이스 | 입력 | 출력 | 상태 |
|---|---|---|---|
| `tricast ppl --model M --recipe R` | 모델 id, 레시피 이름·경로 | `{ppl, nll, n_tokens, env}` | 구현 중 |
| `tricast eval --model M --recipe R --tasks a,b` | + lm-eval 과제 | lm-eval 결과 + env | 구현 중 |
| `python -m tricast.eval.lmeval --model tricast --model_args pretrained=M,recipe=R --tasks …` | lm-eval 표준 인자 (TriCast 어댑터 등록 후 lm-eval CLI 로 넘김) | lm-eval 표준 | 구현 중 |
| `tricast report --model M --recipe R` | 모델, 레시피 | 레이어별 MSE·SQNR·코사인 + 모델 logits KL·top-1 일치 (JSON + md) | 구현 중 |
| `tricast run CONFIG.yaml` | 모델·레시피·과제 목록·스윕 | 레시피별 JSON + summary.md | 구현 중 |
| Python `tricast.quantize / gemm / patch_model` | 텐서·스펙 | QTensor·텐서·패치 보고 | 구현 중 |
| `tricast agent "<자연어>"` | 자연어 요청 | `EmulationRequest` JSON → 실행 계획 → 결과 보고 | 강의 3·6 |

## 6. 수용 기준 (AC, EARS — "[조건]일 때, TriCast 는 [동작]한다") ✍️

- AC1 [상시 적용]: TriCast 는 항상 Triton 커널 결과를 레퍼런스 구현과 비트 단위로 일치시킨다 — 한 원소라도
  다르면 실패. (`tests/gpu/`)
- AC2 [상시 적용]: TriCast 는 항상 출처(provenance)가 기록된 하드웨어 프리셋만 제공한다.
- AC3 [이벤트 기반]: 사용자가 평가를 실행하면, TriCast 는 지표와 함께 git SHA·모델 revision·데이터셋
  fingerprint·레시피 해시를 저장한다.
- AC4 [이벤트 기반]: bf16 passthrough 레시피로 평가하면, TriCast 는 패치하지 않은 모델과 같은 PPL (상대차
  ≤ 1e-3)을 보고한다.
- AC5 [예외 대응]: 레시피가 잘못되면, TriCast 는 실행 전에 거부하고 오류가 난 필드 경로를 알려 준다.
- AC6 [예외 대응]: 자연어 요청에 없는 정밀도·누산 파라미터가 필요하면, TriCast 에이전트는 값을 지어내지 않고
  기본값을 `assumptions` 에 적거나 되묻는다. (강의 3 구현 후 판정)
- AC7 [이벤트 기반]: 독립 구현(NADPE CUDA 커널)의 골든 벡터로 검증하면, TriCast 는 모든 케이스에서 비트
  일치한다. (`tests/data/nadpe/` — 레퍼런스는 `tests/test_golden.py`, Triton 은 GPU 에서
  `scripts/nadpe_oracle/check_triton.py`)

> AC ↔ 골든 케이스: `tests/harness/golden_cases.yaml` (강의 5).
