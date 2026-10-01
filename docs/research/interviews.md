# 인터뷰 프로토콜과 로그 (강의 2)

> 상태 (2026-10-01): A·B·D·E절과 자가 점검은 팀이 수행·기록한 인터뷰 문서(`interview.md`)를 그대로 옮겼다.
> C절은 그 문서의 기록(응답자 화면 관찰 없음)에 공개 작업물 역추적 2건(관찰 A1·A2)을 덧붙였다.
> 여기서 매기는 **로그 번호·관찰 번호**가 `docs/ontology.yaml` 의 `evidence:` 와 `docs/PROBLEM.md` 증거의
> 원천이다. E절은 인터뷰 어휘로 쓴 온톨로지이고, 제품 대표어로 확정한 판은 `docs/ontology.yaml` 이다
> (대응: NumericFormat → Format, EmulationConfig → Recipe, Experiment·Measurement → EvalRun,
> HardwareEnvironment → Preset 과 EvalRun.env).

## A. 인터뷰 프로토콜

- **대상(누구를, 왜)**: AI 모델의 양자화·저정밀도 학습을 실제로 수행해 본 연구자와 ML 엔지니어. 정밀도 설정 변경, 모델 변환, 실제 하드웨어 실행을 직접 경험한 사람이어야 사용한 도구와 반복 작업, 성능·호환성 문제 및 대응 과정을 과거 행동에 근거해 재구성할 수 있다. 또한 비전문가는 AI 경량화에 따른 접근성 향상, 전력 절감, 저비용 AI 이용 가능성에 대한 일반 사용자의 기대를 확인하기 위해 대상으로 삼는다.
- **가설**: 설정 변경 때마다 일어나는 코드 수정·구현 검증과 프레임워크/디바이스 호환성 확인이다. 기존 도구만으로 이 과정이 빠르고 매끄럽다면 반증된다.
- **질문 5개**
  1. 가장 최근 양자화 또는 저정밀도 실험을 처음부터 끝까지 들려주세요. — 실제 사건 확보
  2. 모델·정밀도·프레임워크·하드웨어는 어떻게 정했나요? — 설정과 환경 확인
  3. 설정을 바꿀 때 코드나 도구를 무엇을 얼마나 바꾸었나요? — 재작업 비용 확인
  4. 속도·메모리·정확도·호환성에서 기대와 달랐던 점과 대응은 무엇이었나요? — 페인과 우회로 확인
  5. 어떤 설정을 채택하거나 포기했으며 기준은 무엇이었나요? — 최종 행동 확인
- **유도 질문 점검**: ❌ “설정만 하면 되는 도구를 쓰시겠죠?” → ✅ “지난 실험에서 정밀도나 실행 환경을 바꾸기 위해 실제로 무엇을 하셨나요?”

## B. 인터뷰 로그 (10건)

| # | 대상(역할 포함) | 핵심 인용 | 태그(오픈 코딩) | 도출된 페인 |
|---|---|---|---|---|
| 1 | 대학원생 A (Transformer 언어 모델) | “bit 수나 scaling 방법을 변경하면 관련 코드를 다시 수정해야 했고, 제대로 구현됐는지 확인하는 데도 시간이 많이 걸렸습니다.” | `설정변경_코드수정` `구현검증` `실험반복` | 설정 변경마다 구현·검증 반복 |
| 2 | 대학원생 B (이미지 분류 모델) | “PyTorch에서는 정상적으로 실행되는데 ONNX로 변환하면서 지원되지 않는 연산이 생기기도 했고, 실제 디바이스에서 돌렸을 때 예상한 만큼 속도가 나오지 않는 경우도 있었습니다.” | `PyTorch_ONNX전환` `미지원연산` `디바이스검증` | 변환·디바이스 단계에서 호환성과 성능 불일치 |
| 3 | Research Engineer (FP8 학습 경험) | “hopper architecture에서 fp8 연산에 하드웨어 support가 되지 않는 부분이 있어 algorithm적으로 처리해야하는 부분이… 속도도 생각보다 빠르지 않고 메모리 사용량도 많았습니다.” | `하드웨어지원공백` `알고리즘우회` `속도저하` `메모리증가` | 지원 공백을 우회 구현해야 하며 기대 효율 미달 |
| 4 | 석사 졸업 후 박사과정 준비자 | 경험·병목·가속 라이브러리 필요 여부에 “Y / Y / Y”로 답함. | `병목경험` `도구필요` | 필요성 신호만 있고 사건 세부 없음 |
| 5 | 전자공학부 졸업·반도체 취업 준비자 | “데이터센터 전력소모 감소에 도움이 될거같음” | `전력절감` | 전력 절감 기대 |
| 6 | 소설 작가 | “하드웨어 비용 및 인프라 절감이 가장 유용할 것 같다.” | `인프라절감` | 도메인 외의 비용 절감 기대 |
| 7 | PC·네트워크 유지보수 | “Enterprise급 AI 가속기 없이도 구축할 수 있는 고효율 모델들을 양산할 수 있는 가능성” | `가속기의존감소` `구축접근성` | 고가 가속기 의존 감소 기대 |
| 8 | 디자인 분야 학생 | “적은 메모리와 연산량만으로 AI를 학습시킬 수 있다는 점은 기술의 접근성과 활용성을 높일 수 있다.” | `경량학습` `접근성` | 저자원 학습의 접근성 기대 |
| 9 | 이윤태 (NPU 개발 실무) | “GPU 세대에 따라서 커널 작업을 반복적으로 해야하고, 알고리즘 묘사와 정밀도 조합에 따라서도 마찬가지다.” | `NPU개발` `ULP오차` `CUDA커널구축` `GPU세대별재작업` `반복검증` | 연산 알고리즘·정밀도 조합과 GPU 세대가 바뀔 때마다 에뮬레이션 커널과 검증을 반복 |
| 10 | 조서연 (LLM 최적화 연구) | “비트 폭 하나 바꾸고, 희소 비율 하나 틀고, 이상치 보존 방식 하나 수정할 때마다 CUDA 커널을 다시 깎아야 합니다.” | `비트폭변경` `희소성변경` `이상치처리` `커널유지보수` `아키텍처종속` | 연구 본체보다 시뮬레이션용 CUDA 커널 유지보수에 더 많은 시간이 들 수 있음 |

**도출된 페인 (누적)**

1. **설정 변경마다 반복되는 구현·검증** [문제, 우회로] — bit 수·scaling 방식을 바꿀 때 코드를 다시 수정하고, 작은 모델·데이터셋으로 후보를 먼저 줄인다. (로그 1)
2. **프레임워크·변환·디바이스 호환성 단절** [문제, 실패 유형] — PyTorch에서 되던 모델도 ONNX 변환 시 미지원 연산이 생기며, 실제 기기 성능도 다를 수 있다. (로그 2)
3. **하드웨어·정밀도 호환성의 구현 부담** [문제, 제약] — FP8 하드웨어 지원 공백을 알고리즘 처리로 메워야 한다. (로그 3)
4. **예상보다 낮은 실행 효율** [실패 유형] — QAT/FP8 학습에서 속도가 느리거나 메모리를 많이 써 실험 범위를 줄인다. (로그 1, 3, 4)
5. **고가 가속기와 인프라 비용 의존** [문제] — 제한된 자원에서도 고효율 모델을 구축하고 싶다는 기대가 있다. (로그 5, 6, 7, 8)
6. **연산 설계 조합별 반복 검증** [문제, 제약] — 누적·라운딩 등 연산 알고리즘과 ULP 오차 허용치가 모델 정확도, 칩 면적, 전력에 영향을 주므로 조합마다 검증해야 한다. (로그 9, 10)
7. **시뮬레이션 가속용 CUDA 커널 유지보수** [문제, 우회로] — 비트 폭·희소 비율·이상치 처리·GPU 세대가 바뀔 때마다 커널을 수정하고 최적화해야 한다. (로그 9, 10)

## C. 워크플로 관찰 기록

제품 또는 실험 화면 공유가 어려워 작성에서 제외하였다.

응답자 관찰 대신 공개 작업물의 절차와 수치를 그대로 옮긴 기록 2건을 둔다 (방법: 기존 작업물 역추적).
사용자 페인의 증거는 B절 로그이며, 이 관찰은 스펙 후보를 좁히는 데만 쓴다.

**관찰 A1** — 대상: NADPE (MICRO'26 *Not All Dot Products Are Equal*) artifact, `micro26-ae` /
방법: README·설정·커널 소스 역추적 (2026-09-30)

| 관찰 항목 | 관찰 내용 (본 것만 적는다) | 스펙 후보 (무엇으로 변환되는가) |
|---|---|---|
| 단계와 도구 | ① vLLM 포크 + MMA-Emu 커널 빌드 (`build.sh`, 약 30분) 또는 Docker 이미지 → ② 실험별 `config.yaml` 에 `algorithm`/`f_bits`/`chunk_size` → ③ `run_experiment.py` 가 lm-eval 구동 → ④ 그림 스크립트 | 레시피 파일 하나로 구성 → 평가 → 결과 기록 |
| 전환(Handoff) | 허용 파라미터 값이 `core/design_space.cuh` 의 고정 집합이다 ("bounding these sets bounds the build") — 집합 밖의 F·G·CS 값은 C++ 수정과 재빌드가 필요 | 파라미터를 런타임 값으로 받는 커널 (재빌드 없음) |
| 우회로(Workaround) | Hopper 누산을 Blackwell 에서 에뮬레이트해 H100 결과와 비교 (Table 6) — 두 종류의 GPU 가 필요 | 에뮬레이션과 실리콘을 비교하는 도구 |
| 멈칫(Hesitation) | 전체 재현 약 65시간, FP8 CoFDA 36개 설정 스윕 약 30시간 (RTX PRO 6000 1장, README 표기) | 설정당 처리 시간이 탐색 루프의 병목 → Evals 지표 후보 |

**관찰 A2** — 대상: microsoft/microxcaling (MX 양자화 라이브러리) / 방법: 소스 역추적

| 관찰 항목 | 관찰 내용 (본 것만 적는다) | 스펙 후보 |
|---|---|---|
| 단계와 도구 | `mx.Linear` 가 입력·가중치를 MX 형식으로 양자화한 뒤 `F.linear` 로 곱한다 (`mx/linear.py:86`) | 형식 양자화와 누산 산술을 한 레시피에서 함께 정의 |
| 전환 | 누산은 PyTorch matmul 정밀도 설정(`set_matmul_precision`)에 맡긴다 — 누산기 비트 폭·정렬·절단은 모델링 대상이 아니다 | MMA 누산 알고리즘을 명시적 파라미터로 |
| 우회로 | — | — |
| 멈칫 | — | — |

- 진술과 관찰의 차이: 로그 9·10은 "조합이나 GPU 세대가 바뀔 때마다 커널을 다시 작성한다"고 말했다. 관찰 A1의
  "허용 값 밖 파라미터는 C++ 수정과 재빌드"는 같은 방향의 사실이지만, 응답자의 작업을 직접 본 것은 아니다.

## D. 핵심 Job Story

> 저정밀도 모델이나 NPU 연산 설계를 비교·검증할 때, 나는 정밀도·연산 알고리즘·희소성·이상치 처리·하드웨어 조건을 바꾼 결과를 빠르게 에뮬레이션하고 싶다. 그래서 조합이나 GPU 세대가 바뀔 때마다 CUDA 커널을 다시 작성하지 않고, 모델 정확도와 실행 성능, 칩 면적·전력 목표를 만족하는 설정을 선택할 수 있도록. (로그 1, 2, 3, 9, 10)

| 요소 | 내용 | 근거 |
|---|---|---|
| 상황(트리거) | bit 수·scaling·연산 알고리즘·희소성·이상치 처리 방식을 바꾸거나, 모델을 실제 하드웨어용 형식으로 변환할 때 | 로그 1, 2, 9, 10 |
| 기능적 Job | 여러 설계 조합을 GPU에서 고속 에뮬레이션하고 정확도·속도·메모리·ULP 오차 및 하드웨어 비용 영향을 비교 | 로그 1, 2, 3, 9, 10 |
| 감정적 Job | 커널 구현과 변환 결과가 실제 하드웨어 동작을 충분히 재현하는지에 대한 불확실성을 줄이기 | 로그 1, 2, 9, 10 |
| 사회적 Job | 팀 동료가 재현·검증할 수 있는 실험 결과를 빠르게 공유해 모델·하드웨어 설계 의사결정에 기여하는 연구자로 인정받기 | 로그 1, 2 |
| 현재 고용된 대안 | PyTorch 직접 구현, 오픈소스·공식 문서·GitHub issue 탐색, ONNX 변환·모델 구조 변경, 설계 조합별 CUDA 커널 직접 작성 | 로그 1, 2, 9, 10 |

| 힘 | 내용 | 근거 |
|---|---|---|
| Push | 설정·연산 조합·GPU 세대가 바뀔 때마다 코드와 CUDA 커널을 수정·검증하고, 변환 후 호환성 문제도 해결해야 함 | 로그 1, 2, 9, 10 |
| Pull | 목표 정밀도와 연산·희소성·이상치 처리 조건을 입력하면 고속 에뮬레이션과 비교 결과를 제공하는 통합 프레임워크 — **가설** | (가설) |
| Anxiety | 에뮬레이션이 실제 하드웨어 동작과 오차를 충분히 재현하지 못하거나, 저정밀도 적용 뒤 정확도·성능이 저하될 가능성 | 로그 2, 3, 9, 10 |
| Habit | PyTorch 직접 구현, 기존 자료 탐색, 설계 조합과 GPU 세대별 CUDA 커널 수작업 유지보수 | 로그 1, 2, 9, 10 |

- 온톨로지로 되쓰기: `Experiment`가 `EmulationConfig`·`HardwareEnvironment`를 설정 → `Model`을 `ModelFormat`으로 변환·실행 → `Measurement`로 모델·하드웨어 영향을 비교

## E. 미니 온톨로지

- 반복 명사 → 클래스 후보: 모델, 수치 형식, 연산 알고리즘, 에뮬레이션 설정, PyTorch/ONNX 형식, 하드웨어, 실험, 측정 결과
- 동의어 통합: `FP8`, `INT8`, `bit 수` → `NumericFormat`
- 범위 밖: 클라우드·데이터센터 운영, 모델 서비스 배포, CUDA 커널 자동 생성

```yaml
version: 1
classes:
  User:
    description: 저정밀도 실험을 수행하는 연구자·개발자
    evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
    attributes:
      role: {type: string, examples: [quantization_researcher, edge_deployment_engineer, NPU_engineer], evidence: [로그 1, 로그 2, 로그 9, 로그 10]}
    relations:
      - runs: Experiment
        evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
  Experiment:
    description: 모델·정밀도·하드웨어 조건에서 수행하는 학습 또는 배포 검증
    evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
    attributes:
      objective: {type: enum, values: [training, deployment_validation, hardware_emulation], evidence: [로그 1, 로그 2, 로그 9, 로그 10]}
    relations:
      - configures: NumericFormat
        evidence: [로그 1, 로그 2, 로그 3]
      - uses: EmulationConfig
        evidence: [로그 9, 로그 10]
      - runs_on: HardwareEnvironment
        evidence: [로그 2, 로그 3, 로그 9, 로그 10]
      - evaluates: Model
        evidence: [로그 1, 로그 2]
      - produces: Measurement
        evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
  NumericFormat:
    description: 모델 학습·실행의 수치 정밀도 형식
    evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
    attributes:
      name: {type: string, examples: [FP8, INT8], evidence: [로그 2, 로그 3]}
      bit_width: {type: int, examples: [8], evidence: [로그 2, 로그 3, 로그 10]}
    relations:
      - applied_in: Experiment
        evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
  EmulationConfig:
    description: 실제 하드웨어 연산을 GPU에서 모사하기 위한 정밀도·연산·희소성·이상치 처리 조건 묶음
    evidence: [로그 9, 로그 10]
    attributes:
      bit_width: {type: int, evidence: [로그 10]}
      arithmetic_algorithm: {type: string, examples: [accumulation, rounding], evidence: [로그 9, 로그 10]}
      sparsity_pattern: {type: string, examples: ["N:M"], evidence: [로그 10]}
      sparsity_ratio: {type: float, evidence: [로그 10]}
      outlier_policy: {type: string, examples: [preserve_high_precision], evidence: [로그 10]}
      ulp_tolerance: {type: float, evidence: [로그 9, 로그 10]}
    relations:
      - applied_in: Experiment
        evidence: [로그 9, 로그 10]
  HardwareEnvironment:
    description: 실험이 실행되는 GPU 아키텍처 또는 엣지 AI 가속기 환경
    evidence: [로그 2, 로그 3, 로그 7, 로그 9, 로그 10]
    attributes:
      architecture: {type: string, examples: [Hopper], evidence: [로그 3]}
      gpu_generation: {type: string, note: "세대 변경 시 커널 재최적화 여부를 기록", evidence: [로그 9, 로그 10]}
      precision_support: {type: list, examples: [FP8], note: "지원 공백도 기록한다", evidence: [로그 3]}
    relations:
      - hosts: Experiment
        evidence: [로그 2, 로그 3, 로그 9, 로그 10]
  Model:
    description: 저정밀도 설정으로 학습 또는 실행하는 AI 모델
    evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
    attributes:
      training_mode: {type: string, examples: [QAT, FP8_training], evidence: [로그 1, 로그 3]}
    relations:
      - converted_to: ModelFormat
        evidence: [로그 2]
      - evaluated_by: Experiment
        evidence: [로그 1, 로그 2]
  ModelFormat:
    description: 모델을 프레임워크에서 실제 배포 환경으로 옮기는 표현 형식
    evidence: [로그 2]
    attributes:
      name: {type: string, examples: [PyTorch, ONNX], evidence: [로그 2]}
    relations:
      - represents: Model
        evidence: [로그 2]
  Measurement:
    description: 실험 결과의 속도·메모리·정확도·ULP 오차와 하드웨어 비용 추정 지표
    evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
    attributes:
      throughput: {type: float, evidence: [로그 1, 로그 2, 로그 3]}
      memory_usage: {type: float, evidence: [로그 3]}
      accuracy: {type: float, evidence: [로그 2]}
      ulp_error: {type: float, evidence: [로그 9, 로그 10]}
      estimated_chip_area: {type: float, evidence: [로그 9, 로그 10]}
      estimated_power: {type: float, evidence: [로그 9, 로그 10]}
    relations:
      - generated_by: Experiment
        evidence: [로그 1, 로그 2, 로그 3, 로그 9, 로그 10]
# 범위 밖: 클라우드·데이터센터 운영, 모델 서비스 배포, CUDA 커널 자동 생성
# 핵심 흐름: User --(runs)--> Experiment --(uses)--> EmulationConfig
#            Model --(converted_to)--> ModelFormat; Experiment --(produces)--> Measurement
```

## 자가 점검

- [x] 로그 10건: `interview.txt` 4건, 경험 설문 1건, 수요 설문·메신저 5건
- [x] 페인과 Job Story를 실제 경험·작업 맥락 로그 1~3, 9~10에 연결
- [x] 클래스·속성·관계마다 `evidence`를 연결
- [ ] 실제 화면 공유 기반 워크플로 관찰: 아직 없음
- [ ] 로그 4~8의 수요 설문을 심층 인터뷰로 보강
