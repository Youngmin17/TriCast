# 레인 R2 노트: LLM-as-optimizer · AutoML/압축/시스템 튜닝 에이전트

> 2026-11-03 · 목적 `novelty-audit` · 레인 R2 (워커 1명, 하위 워커 없음) · survey.yaml 은 오케스트레이터가 쓴다.
> 카드는 `~/research-kb/papers/` 에 있다. `kb.py lint` 결과는 카드 1386, 오류 0, 경고 453 이다 (경고는 venues.md 미등재 venue, 이 레인 신규분은 OpenReview·Nature·VLDB·EACL 4건).
> 판정 어휘는 `preempts` / `partial` / `adjacent` / `none` / `unknown` 이다. 판정 근거 위치는 카드의 절·표다.
> 같은 날 R1 레인이 에이전트형 양자화 카드 6장(apqf · haqa · tap · explorerreviewer · quantforge · fca)을 먼저 썼다. 이 레인은 그 카드를 중복 작성하지 않고 인용만 한다.

## 0. 요약

| 주장 | 레인 판정 | 한 줄 근거 |
|---|---|---|
| C1 전체 (원장 검색 + 타입 검사 + LLM 이 층별 혼합 방법 할당 제안 + 같은 측정 수에서 해석적·블랙박스 탐색보다 우수) | **partial** | 전체 조합을 선점한 문헌은 찾지 못했다. 가장 가까운 것은 **동시기 ICLR'27 심사 중 제출물 2편**이다: autoqs-openreview26 (LLM 제안 + 평가 이력 + 결정적 검증 + 예산, 다만 group size 만 탐색), explorerreviewer-openreview26 (공유 research log + 리뷰어 에이전트). 둘 다 본문을 받지 못해 세부 차별점은 `unknown` 이다. |
| C1-a LLM 이 측정 민감도로 층별 압축 수준을 제안 | **partial (arXiv 수준 선점)** | apqf-arxiv26 (CNN/ViT 층별 비트, 1회 제안), kodathala-arxiv26 (Qwen3 층별 희소 증분 + PPL 롤백), autopruner-arxiv26 (VLA 층 삭제 + 평가 이력). tier 1 게재작은 없다. |
| C1-b 측정 레코드 원장 검색으로 제안을 접지 | **partial** | mlcopilot-eacl24 (과거 실험 레코드 정규화 + 검색, 검색 제거 시 붕괴), accelopt-mlsys26 (경험 메모리), prompts-mlsys26 (지식 베이스 RAG). 같은 실행 안의 측정 원장을 id 로 인용하게 하는 설계는 찾지 못했다. |
| C1-c 타입 검사 (크기 예산 · 증거 id 인용 · 수치-레코드 지지) | **partial (구성요소) / unknown (조합)** | 수치 검증 프로토콜은 pcn-arxiv25 (tier 0, 실험 없음), 제안 검증은 autopruner · apqf · helionllm-blog26 의 형식·제약 검사뿐이다. 최적화 루프 안에서 "인용 id 가 있고 그 레코드가 수치를 지지하는지"를 결정적으로 거르는 사례는 찾지 못했다. |
| C1-d 같은 측정 수에서 LLM 제안 > 블랙박스 | **adjacent (HPO·시스템 튜닝에서는 정착된 주장 유형)** | zhang-arxiv23 (10–60 평가에서 LLM > SMAC, 100 평가에서 역전), llambo-iclr24 (25 trials 등예산, 초기 이득), haqa-arxiv26 (10 라운드, BO·NSGA2·랜덤 대비), helionllm-blog26 (LFBO 와 같은 성능에 측정 ~10× 적음). 층별 LLM 양자화 할당에서 해석적 knapsack 과 블랙박스를 등측정으로 함께 비교한 tier-1 문헌은 찾지 못했다. |
| C3 이질적 방법 하나의 층별 공간 | **partial (R1 소관)** | 이 레인에서 본 LLM 에이전트 문헌 중 방법 × 포맷 × 방법 옵션을 층마다 고르는 것은 없다 (autoqs = group size, haqa = 전역 하이퍼파라미터, apqf = 비트만). 방법 혼합 자체는 R1 의 calm-arxiv25 · modeloptaq-github26 참조. |

프로토콜(`protocol.md`)에 바로 반영할 사실 네 가지:

1. **비교 축을 예산 곡선으로 그려야 한다.** zhang-arxiv23 은 LLM 이 10·30·60 평가에서 BO 를 이기다가 100 평가에서 진다(Tab 5). rebench-icml25 은 2h 에서 에이전트 4×, 32h 에서 인간 2× 로 뒤집힌다. 단일 N 비교는 리뷰어가 받아들이지 않을 가능성이 크다. §7 의 any-time 곡선을 주 그림으로 올리기를 권고한다.
2. **BO/TPE 와 "LLM-seeded BO" 하이브리드는 필수 비교군이다.** LLM 최적화기 문헌의 표준 블랙박스는 SMAC·TPE·GP-BO 다 (llambo · zhang-arxiv23 · gptuner · haqa). §6 의 TPE 는 "추가 예정"이다. 하이브리드는 zhang-arxiv23 (LLM 10 제안으로 BO 시드, 32 과제 중 21 과제 동등 이상)과 helionllm-blog26 (LLM seed → LFBO)에서 이미 강한 기준선으로 쓰인다.
3. **검증 KL 로 고르고 test 로 보고하는 선택 편향을 따로 재야 한다.** airadojo-neurips25 는 검증 점수로 최종 선택을 하면 test 오라클 대비 9–13 %p 를 잃는다고 보고했다 (§5.3). 에이전트는 블랙박스보다 많은 후보를 "본다". §4 의 "최종 평가는 최선 레시피 1회" 규칙은 맞다. 여기에 검증–test 순위 상관(또는 top-k 제출)을 보고에 추가하기를 권고한다.
4. **"무원장·무피드백 LLM arm" 은 문헌의 표준 절제다.** evoprompting (같은 표본 수의 naive prompting), reevo ("LLM only"), alphaevolve ("No evolution", "No context"), kernelbench (같은 호출 수 반복 샘플링), opro (1단계 50개 생성), mlcopilot (경험 검색 제거 → 붕괴)이 모두 넣었다. §6 절제 행에 "LLM, 측정 이력 없음, 같은 N" 을 명시하기를 권고한다.

## 1. 검색 로그 (2026-11-03)

### 1.1 출처와 질의

- DBLP 는 봇 검사(Anubis)에 막혔다. venue 확인은 OpenReview API(DBLP 미러 레코드 포함) · OpenAlex(mailto 없음) · Crossref(공개 pool) · 학회 프로그램 JSON 으로 했다.
- 웹 요청에 사용자 식별자를 넣지 않았다. 이메일을 요구하는 API 는 쓰지 않았다.

| # | 도구 | 질의 / 대상 | 결과 요지 |
|:-:|---|---|---|
| W1 | WebSearch | LLM agent mixed-precision quantization search per-layer bit allocation arXiv 2025 | 비-LLM 할당기만 (CoopQ, Q-Strata, MixQuant — R1 소관) |
| W2 | WebSearch | agent LLM-driven model compression quantization pruning automated | APQF, ProfilingAgent, "LLMs can Compress LLMs" 발견 |
| W3 | WebSearch | LLM agent quantization configuration GPTQ AWQ agentic bit-width 2026 | arXiv 2609.09867 (층별 교체 move 비가산성 — R1/C2 로 전달) |
| W4 | WebSearch | LLM as optimizer for mixed-precision bit-width policy | AMQ, ScaleBITS (R1 소관) |
| W5 | WebSearch | LLM-driven evolutionary search discovers pruning/quantization algorithm | arXiv 2606.25532 (지식 그래프 진화, 초록만) |
| W6 | WebSearch | AI research agents fabricate experimental results pitfalls | MLR-Bench, Luo et al. 워크숍, Jr. AI Scientist |
| W7 | WebSearch | LLM in-context optimizer mixed-precision vs evolutionary/BO ICML/NeurIPS/ICLR 2026 | TAP, HAQA (R1 이 카드화), AutoQRA (ICML'26, 비-LLM) |
| W8 | WebSearch | LLM agent quantization recipe GPTQ act_order damp group size | 학술 결과 없음 (튜토리얼만) |
| W9 | WebSearch | LLM-guided autotuning compiler/GPU kernel vs BO equal trials | PyTorch Helion LLM autotuner 블로그, Record-Remix-Replay |
| W10 | WebSearch | LLM agent provenance check reported numbers traceable to tool output | Proof-Carrying Numbers, NabaOS |
| W11 | WebSearch | benchmark for LLM agents performing compression | ACBench (반대 방향: 압축이 에이전트에 미치는 영향) |
| A1–A4 | arXiv API | `abs:(LLM AND agent AND mixed-precision)`, `abs:((LLM) AND (agent OR agentic) AND (quantization OR pruning) AND (search OR optimizer OR planner))` (156건 중 160건 범위 열람), `abs:(LLM-guided OR LLM-driven OR LLM agent ...) AND (quantization OR pruning OR compression) AND (layer ...)` (39건 전부), `abs:group-size AND LLM AND quantization` | APQF, AutoPruner, Kodathala, AIPC(불명확) |
| O1 | OpenReview API (제목) | OPRO, LLAMBO, MLAgentBench, MLE-bench, AutoML-Agent, AIDE, ReEvo, EoH, Zhang HPO, AgentHPO, PPSN'24, GPT-NAS/GENIUS, KernelBench, AlphaEvolve, RE-Bench, MLGym, MLCopilot, GPTuner, LLAMBO 재현, LLM-ES, AI Scientist, BOPRO, AIRA, MLR-Bench, Luo et al., LMABO, BOLT, PROMPTS 등 | venue 확인 |
| O2 | OpenReview API (전문) | agentic quantization · LLM agent mixed-precision · LLM-guided quantization · agent compression layer-wise · LLM-driven quantization strategy · automated quantization LLM proposal evaluation history · LLM evolutionary search quantization · LLM proposes layer-wise sparsity 등 16개 | **AutoQS (ICLR'27 제출)** 발견, Error-Guided Composition (ICLR'27 제출, C3 → R1 전달) |
| P1 | MLSys 가상 프로그램 JSON | 2024 (37건, 일부), 2025 (61건), 2026 (308건, 제목만) 제목·초록 키워드 필터 | AccelOpt, PROMPTS, Automated Algorithm Design for Auto-Tuning Optimizers, Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems, Agentic Operator Generation |
| G1 | gh api | 18개 repo (stars · 마지막 push · license) + `gh search code llm_guided --repo pytorch/helion` | Helion `test/test_llm_autotuner.py` 존재 확인 |
| X | 기타 | OpenAlex 7건, Crossref 1건(FunSearch), EuropePMC/PMC 전문 1건 | — |

WebSearch 사용: **11회** (상한 45).

### 1.2 venue × 연도 확인 범위

방법은 키워드 검색(OpenReview 제목·전문, arXiv API, OpenAlex, WebSearch)이다. venue 목록 전수 열람은 MLSys 만 했다. 나머지 칸의 "없음"은 "이 질의로 찾지 못함"이다.

| venue | 2023 | 2024 | 2025 | 2026 | 방법 |
|---|---|---|---|---|---|
| ICLR | — | ✔ (OPRO, LLAMBO) | ✔ (MLE-bench, BOPRO) | ✔ (LMABO, BOLT 확인, 카드 미작성) + 2027 제출 검색 | OpenReview |
| ICML | — | ✔ (MLAgentBench, EoH) | ✔ (AutoML-Agent, KernelBench, RE-Bench) | 키워드 검색만 — 해당 주제 tier-1 없음 (AutoQRA 는 비-LLM) | OpenReview |
| NeurIPS | ✔ (EvoPrompting) | ✔ (ReEvo) | ✔ (AIRA-dojo, MLR-Bench D&B) | **unknown** — 2026 accepted list 미확인 | OpenReview |
| MLSys | — | 부분 (JSON 37건) | ✔ (61건 전수, 해당 없음) | ✔ (308건 제목 전수, AccelOpt · PROMPTS 카드) | 프로그램 JSON |
| Nature/Science | FunSearch (온라인 2023-12) | — | — | AI Scientist (Nature 2026, 제외) | OpenAlex · Crossref |
| arXiv | — | — | ✔ | ✔ (2026-10-30 까지) | arXiv API · WebSearch |

## 2. 주장 × 가장 가까운 선행 (1–3 편) × 차이 × 판정

| 주장 | 가장 가까운 선행 | 차이 (한 문장) | 판정 · 근거 위치 |
|---|---|---|---|
| **C1 전체** | autoqs-openreview26 | AutoQS 는 고정 비트·고정 방법에서 층×연산자 group size 만 고르고 비교군이 균일 GPTQ/AWQ 뿐이다 (초록). C1 은 방법 × 포맷 × 옵션을 섞고 증거 인용 검사를 하며 knapsack·블랙박스와 등측정 비교를 한다. | partial · 카드 mechanism/results (초록 기준, 본문 `unknown`) |
| | explorerreviewer-openreview26 | Explorer–Reviewer 는 LLM 리뷰어가 4개 체크포인트에서 결론을 심사한다. C1 은 결정적 타입 검사로 수치-레코드 지지를 거른다. 층별 혼합 할당 여부와 등측정 비교 여부는 초록에 없다. | partial · 카드 limits (본문 `unknown`) |
| | haqa-arxiv26 | HAQA 는 LLM 에이전트를 Local · BO · Random · NSGA2 와 10 라운드로 비교했다. 그러나 대상은 전역 미세조정·배포 하이퍼파라미터이고, 원장 검색·수치 검사·층별 할당이 없다. | adjacent · Tab 1–2 |
| **C1-a** 층별 할당 제안 | apqf-arxiv26 | APQF 는 프로파일 근거로 층별 비트를 **1회** 제안(피드백 루프 없음)하고 대상이 CNN/ViT 이며 비트만 고른다. | partial · §3.3, Tab 1–4 |
| | kodathala-arxiv26 | 반복 루프·PPL 롤백은 C1 과 같지만, 층별 희소 증분 하나의 축이고 기준선은 N:M magnitude 뿐이다. | partial · §3, Alg 1, Tab 2–4 |
| | autopruner-arxiv26 | 증거 저장소(하드 제약/수정 가능 증거 분리) + 평가 이력 + 프로그램 검사가 C1 구조와 가장 비슷하다. 대상은 VLA 층 삭제이고 등측정 블랙박스 비교가 없다. | partial · §III, Tab I, §V-B |
| **C1-b** 원장 검색 접지 | mlcopilot-eacl24 | 과거 실험 레코드를 정규화(수치 이산화)해 검색하고 지식을 사후 검증한다. 검색 단위가 **과제 간** 메타 학습이고, 같은 실행 안의 측정 원장·증거 id 인용은 없다. | partial · §4.1–4.5, Tab 2–5 |
| | accelopt-mlsys26 | 느린-빠른 커널 쌍의 경험 메모리를 다음 생성에 넣는다. 대상은 커널 코드이고 수치 인용 검사는 없다 (초록). | adjacent · abstract |
| | bopro-iclr25 | 어떤 과거 생성물을 프롬프트에 넣을지를 BO 로 고른다. 이 문제가 성능을 좌우함을 보였다(1D-ARC 에서는 best-k 에 패배). 측정 원장의 의미 검색은 아니다. | adjacent · abstract |
| **C1-c** 타입 검사 | pcn-arxiv25 | 수치 구간을 claim 에 묶고 렌더러에서 정책 대조(fail-closed)한다. 실험이 없고 최적화 루프에 적용한 예도 없다. | partial · abstract |
| | autopruner-arxiv26 | 프로그램이 기하·중복 제약을 검사하고 거부 사유를 돌려준다. 수치·증거 인용 검사는 없다. | partial · §III |
| | mlagentbench-icml24 | Fact Check 필드로 자기 수치를 관측과 대조하게 하지만 LLM 자기 점검일 뿐이다. 그래도 Claude v1.0 실행의 ~20% 가 근거 없는 10% 개선을 주장했다. 이것이 C1-c 의 동기 근거다. | adjacent · §3.1, App D.2 |
| **C1-d** 등측정 LLM > 블랙박스 | zhang-arxiv23 | HPO(2–5 차원)에서 같은 평가 수 10/30/60/100 으로 SMAC 와 비교했다. 소예산에서는 우위였고 100 평가에서 BO 가 역전했다. 고차원 층별 할당(수백 칸)과 해석적 prior 는 없다. | adjacent · Tab 1, 3, 5 |
| | llambo-iclr24 | 같은 5 초기점 + 25 trials 에서 GP·TPE·SMAC 를 이겼다(초기 trial 에 집중). 저차원 HPO 이고 단일 LLM 이다. | adjacent · Fig 7 |
| | helionllm-blog26 | 측정 이력을 넣은 LLM 이 LFBO 와 같은 커널 성능(1.009×)에 측정 구성 수 ~10× 적게 도달했다. 고정 예산 비교가 아니라 "같은 성능까지의 측정 수" 비교다. | adjacent · Result 1–3 |
| **C3** (참고) | calm-arxiv25 (R1 카드) | 층별 GPTQ/AWQ/SmoothQuant 선택(CKA greedy)이다. LLM 에이전트도 원장도 없다. | partial (R1 판정 따름) |

C2 에 대해 이 레인에서 나온 반증 근거 2건 (R1·오케스트레이터 전달용, 카드는 autopruner 만):
- autopruner-arxiv26 Fig 4: VLA 층 쌍의 93.8–98.8% 가 sub-additive 였다.
- arXiv 2609.09867 "Contextual Utility of Quantization Moves in Extreme Low-Bit LLMs" (Xiao, Cao; v2 2026-10-16, cs.DB): 단일 교체 측정의 조합 잔차가 고립 가격 오차 합의 26× · 42× 였다 (Llama-3.2-1B, 3B GPTQ). Qwen3-0.6B 전수 격자에서 8개 move 중 6개가 ARC-KL 부호 역전을 보였다 (초록). **카드 미작성** — R1 소관(가산성·민감도). C2 의 "가산 표 + 정확 knapsack" 정당화에 직접 걸리므로 R1 카드화를 권고한다.

## 3. 카드별 겹침 판정 (이 레인이 쓴 카드 + 인용한 R1 카드)

| 카드 | tier | C1 | C3 | 근거 위치 · 메모 |
|---|:-:|---|---|---|
| autoqs-openreview26 | 0 (ICLR'27 제출) | partial | adjacent | 초록: LLM 제안 · 평가 이력 · 결정적 검증 · 프록시 + 주기 실측 · 예산. 본문 PDF 는 OpenReview 봇 검사로 받지 못함 |
| explorerreviewer-openreview26 (R1) | 0 | partial | unknown | R1 카드 메모의 차별 축 (a)–(d) 동의 |
| haqa-arxiv26 (R1) | 0 | adjacent | none | 등라운드 BO·NSGA2·랜덤 비교가 있는 유일한 LLM 양자화 에이전트. 전역 하이퍼파라미터만. Tab 2 평균 불일치 |
| apqf-arxiv26 (R1) | 0 | partial | none | 층별 비트 1회 제안, CNN/ViT, 블랙박스 비교 없음 |
| quantforge-arxiv26 (R1) | 0 | adjacent | none | 같은 평가기 호출 240 회에서 메모리 방식 비교. 대상은 양자화기 프로그램 발견 |
| fca-arxiv26 (R1) | 0 | adjacent | adjacent | 40 구성 카탈로그에서 랜덤·TPE·one-shot 과 평가 수 비교, 비전 소형 모델 |
| tap-arxiv25 (R1) | 0 | adjacent | none | LLM 이 MPQ 프록시 코드를 진화. "직접 프롬프팅은 층별 비트 할당이 불안정" (§1) — C1 에 대한 반대 근거 |
| kodathala-arxiv26 | 0 | partial | none | 위 표 |
| autopruner-arxiv26 | 0 | partial | none | 위 표, C2 반례(Fig 4) |
| profilingagent-arxiv25 | 0 | adjacent | none | APQF 전신, 초록만 |
| opro-iclr24 | 1 | adjacent | none | LLM-as-optimizer 원형. 등측정 블랙박스 비교 없음, n≥20 TSP 에서 휴리스틱에 패배 (Tab 3) |
| llambo-iclr24 | 1 | adjacent | none | 등예산 BO 비교 선례 (Fig 7) |
| bopro-iclr25 | 1 | adjacent | none | 이력 선택이 핵심 (초록) |
| zhang-arxiv23 | 0 | adjacent | none | 예산 곡선 역전 (Tab 5) |
| evoprompting-neurips23 | 1 | adjacent | none | 표본 수를 맞춘 naive prompting arm (§4.1) |
| funsearch-nature24 | 1 | adjacent | none | 평가기 + 진화 원형, 실행 성공률 보고 (4/140) |
| alphaevolve-arxiv25 | 0 | adjacent | none | No evolution / No context 절제 (Fig 8), 커널 타일링 23% |
| eoh-icml24 | 1 | adjacent | none | 질의 예산 2,000 vs ~10^6 (§4.2) |
| reevo-neurips24 | 1 | adjacent | none | 평가 ≤100, LLM only arm, 문제 은닉 (Tab 5) |
| mlagentbench-icml24 | 1 | adjacent | none | 근거 없는 성공 주장 ~20% (App D.2) |
| mlebench-iclr25 | 1 | none | none | pass@k · SEM · 시간 스케일링 보고 형식 |
| aide-arxiv25 | 0 | adjacent | none | 이력 요약 트리 탐색, 기준선 자원 불일치 예 |
| airadojo-neurips25 | 1 | adjacent | none | 정책 × 연산자 분리, 검증 과적합 9–13 %p (§5.3) |
| automlagent-icml25 | 1 | adjacent | none | 탐색 중 **추정** 지표 사용 (§3.5) — C1-c 의 대조 사례 |
| rebench-icml25 | 1 | none | none | 예산 축 역전, best-of-k, 비용 |
| mlrbench-neurips25 | 1 | adjacent | none | 조작·무효 결과 ~80% (초록) |
| kernelbench-icml25 | 1 | adjacent | none | 같은 10 호출에서 샘플링 vs 피드백 (Tab 1) |
| gptuner-vldb24 | 1 | adjacent | none | LLM 지식 + BO, 같은 100 반복 (§8.1) |
| mlcopilot-eacl24 | 2 | partial | none | 실험 레코드 검색 (Tab 3–4) |
| accelopt-mlsys26 | 1 | adjacent | none | 경험 메모리 (초록) |
| prompts-mlsys26 | 1 | adjacent | none | RAG 제안 + 적은 시도 (초록) |
| pcn-arxiv25 | 0 | partial | none | 수치 claim 검증 (초록) |
| helionllm-blog26 | 0 | adjacent | none | 배포 코드의 LLM 구성 제안 + 출력 계약 |

## 4. 제외 · 카드 미작성 항목과 이유

| 항목 | 이유 |
|---|---|
| GPT-NAS (Yu et al.), GENIUS "Can GPT-4 Perform NAS?" (arXiv 2023) | NAS 이고 tier 0 이다. EvoPrompting(NeurIPS'23)이 대표한다. GPT-NAS 의 게재지는 이번 조사에서 확인하지 못했다 |
| LLMatic (GECCO 2024) | GECCO 는 venues.md 에 없다. LLM+QD NAS |
| LLMs as Evolution Strategies (GECCO Companion 2024), LLMs as Evolutionary Optimizers (CEC 2024) | tier 외 venue |
| AgentHPO (CPAL 2025) | CPAL 은 tier 외 venue. 메커니즘(이력 기반 LLM HPO)은 zhang-arxiv23 · llambo 가 대표한다 |
| Zhang et al. PPSN 2024 "Understanding the Importance of Evolutionary Search in Automated Heuristic Design with LLMs" | tier 외 venue이고 본문을 확인하지 않았다. 무진화 arm 의 중요성은 reevo · alphaevolve 카드로 대신한다 |
| LMABO "Adaptive Acquisition Selection for BO with LLMs", BOLT "Scaling Multi-Task BO with LLMs" (둘 다 ICLR 2026) | LLM 이 BO 내부 구성요소(획득함수 선택·초기화 전이)를 맡는다. 메커니즘은 llambo 가 대표해 카드를 쓰지 않았다. 2026 tier-1 LLM-BO 인용이 필요하면 카드화한다 |
| The AI Scientist (arXiv 2024) / "Towards end-to-end automation of AI research" (Nature 2026) | 연구 전체 자동화이고 구성 튜닝 루프가 아니다. 결과 조작 근거는 mlrbench · mlagentbench 로 대신한다 |
| Luo et al. "The More You Automate, the Less You See" (NeurIPS 2025 AI4Science 워크숍) | 워크숍(tier 0)이다. mlrbench 카드 메모에 인용했다 (post-hoc 선택 편향, 벤치 선택) |
| "Beyond Execution: Auditing Experimental Fidelity" (ICLR'27 제출) | 초록을 열람하지 못했다 (`unknown`) |
| LLAMBO 재현 연구 (arXiv 2511.18891) | llambo 카드 메모에 흡수했다 |
| "Agentic evolution of physically constrained foundation models" (arXiv 2606.25532) | 압축 **방법** 발견(지식 그래프 진화)이고 층별 할당 탐색이 아니다. 초록만 봤고 피드백·예산이 미기재다 |
| AIPC (arXiv 2610.34890) | 에이전트가 압축 설정을 고르는지 초록으로 판단할 수 없다 (`unknown`) |
| AMQ, ScaleBITS, CoopQ/IMPQ, MixQuant, Q-Strata, GAMMA, WINDQuant, AutoQRA, MxMoE, DP-LLM | 비-LLM 혼합 정밀도 탐색이라 R1 소관이다 |
| arXiv 2609.09867 (Contextual Utility of Quantization Moves), "Error-Guided Refinement and Composition for LLM Quantization" (ICLR'27 제출, OpenReview o9aMmEdpyD) | 각각 C2 가산성 반증과 C3 방법 결합(코드북 합성)이라 R1 소관이다. §2 에 전달했다 |
| Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems, Automated Algorithm Design for Auto-Tuning Optimizers, Agentic Operator Generation for ML ASICs (모두 MLSys 2026), Record-Remix-Replay (arXiv 2604.11109), KForge, Xe-Forge | 커널 코드 생성·자동 튜닝 알고리즘 설계라 인접이다. kernelbench · accelopt · alphaevolve · helion 이 대표한다 |
| ACBench "Can Compressed LLMs Truly Act?" (ICML 2025) | 반대 방향이다 (압축이 에이전트 능력에 주는 영향) |
| NabaOS 서명 tool receipt (arXiv 2603.10060) | pcn 카드 메모에 인용했다. 자체 벤치 수치는 검증하지 못했다 |
| MLGym (COLM 2025) | 연구 에이전트 벤치이고 COLM 은 tier 외 venue 다 |

## 5. 작성한 카드 (27 편, 전부 신규)

- 에이전트형 압축 4: `kodathala-arxiv26` · `autopruner-arxiv26` · `profilingagent-arxiv25` · `autoqs-openreview26`
- LLM-as-optimizer / LLM-BO 5: `opro-iclr24` · `llambo-iclr24` · `bopro-iclr25` · `zhang-arxiv23` · `evoprompting-neurips23`
- LLM 진화 탐색 4: `funsearch-nature24` · `alphaevolve-arxiv25` · `eoh-icml24` · `reevo-neurips24`
- ML 엔지니어링 에이전트 · 벤치 8: `mlagentbench-icml24` · `mlebench-iclr25` · `aide-arxiv25` · `airadojo-neurips25` · `automlagent-icml25` · `rebench-icml25` · `mlrbench-neurips25` · `mlcopilot-eacl24`
- 시스템·커널 튜닝 (인접) 5: `kernelbench-icml25` · `gptuner-vldb24` · `accelopt-mlsys26` · `prompts-mlsys26` · `helionllm-blog26`
- 수치 검증 1: `pcn-arxiv25`

내용 확인 수준:
- 원문(HTML/PDF/PMC)을 읽은 카드: opro · llambo · evoprompting · funsearch · alphaevolve · eoh · reevo · mlagentbench · mlebench · aide · airadojo · automlagent · rebench · kernelbench · gptuner · mlcopilot · zhang-arxiv23 · kodathala · autopruner · helion
- 초록만 읽은 카드 (카드 limits 에 명시): bopro · mlrbench · accelopt · prompts · pcn · profilingagent · autoqs
- 코드 상태는 `gh api` 로 2026-11-03 에 확인했다. 공식 코드가 없는 것: evoprompting · alphaevolve · kodathala · autopruner · profilingagent · pcn. 미확인(unknown): bopro · zhang-arxiv23 · autoqs · prompts

## 6. "LLM 에이전트가 블랙박스 탐색을 이긴다" 주장에 리뷰어가 기대할 평가 프로토콜

### 6.1 카드별 공정성 장치

| 카드 | 예산 단위 · 일치 여부 | 블랙박스 기준선 | 무피드백/무검색 arm | 통계 | 비용 보고 |
|---|---|---|---|---|---|
| llambo-iclr24 | 목적함수 평가 25, 같은 5 초기점 (일치) | GP-DKL · SKOpt · TPE · SMAC3 | UnInf(설명 제거) | 5 seed | 없음 (§8 정성) |
| zhang-arxiv23 | 평가 10/30/60/100 곡선 (일치) | 랜덤(bootstrap 500) · SMAC RF/GP | — (LLM-seeded BO 하이브리드) | HPOBench 1 seed (약점) | $8 / 32과제×10 |
| haqa-arxiv26 | 10 라운드 (기준선 일치 명시 안 됨) | Local · BO · Random · NSGA2 | 없음 | 일부 std | ~150K 토큰 ~$5 |
| fca-arxiv26 | 사전 평가 카탈로그 질의 수 | 랜덤 · TPE · one-shot | one-shot 계획 | 단일 데이터셋 | — |
| quantforge-arxiv26 | 평가기 호출 240 (일치) | LLM 진화 변형만 | score-only | 8회 중 성공 수 | — |
| gptuner-vldb24 | DBMS 반복 100, 3 세션 (일치) | DDPG++ · GP · SMAC · DB-BERT | 지식 품질 변형 | median + 사분위 | 2.1M 토큰 $23.7 |
| helionllm-blog26 | 측정 구성 수 + 벽시계 (같은 성능 도달 기준) | LFBO | 하이브리드 | 33 사례 geomean | 벽시계만 |
| reevo-neurips24 | 휴리스틱 평가 ≤100 | EoH | LLM only · 반성 제거 | mean±std, 3 run | 호출당 $0.0003 |
| eoh-icml24 | LLM 질의 2,000 (FunSearch 10^6 과 불일치) | FunSearch · Best Fit | EoC · T2T2C | 3 run, 분산 없음 | — |
| kernelbench-icml25 | LLM 호출 10 (일치) | 반복 샘플링 @10 | 단일 시도 | — | — |
| evoprompting-neurips23 | 후보 1,600 | 비-LLM 진화 없음 (공백) | 같은 표본 수 naive prompting | 그림만 | — |
| opro-iclr24 | 단계 × 8 후보 | 없음 (공백) | 1단계 50개 생성 | 표에 std 없음 | — |
| alphaevolve-arxiv25 | 계산 예산 비율 곡선 | — | No evolution · No context · Small LLM | 3 seed, std 음영 | 절대량 없음 |
| airadojo-neurips25 | 24h · H200 1장 (일치) | 정책 Greedy/MCTS/Evo | 연산자 교체 arm | 20 seed, 95% stratified bootstrap | 완료 토큰 ~2× (정성) |
| mlebench-iclr25 | 24h · A10 1장 (일치) | 스캐폴드 3종 | — | 3–36 seed, mean±SEM, pass@k | — |
| rebench-icml25 | 총 시간 2/8/32h, best-of-k | 인간 전문가 | — | percentile bootstrap 95% CI | 29M 토큰 $123 / 8h |

### 6.2 권고 체크리스트 (C1 · protocol.md 대비)

1. **측정 단위를 하나로 정의하고 표 비용까지 센다.** C1 의 해석적 방법은 층 × 방법 단일 층 KL 표를 쓴다. 이 표 비용을 모든 방법이 공유하거나, 블랙박스에 같은 측정 수(또는 같은 GPU 시간)를 준다. protocol §4 의 "표 없는 같은 GPU 시간" 변형이 이 요구를 맞춘다. 선례: gptuner (반복 수 단위, 최적화기 오버헤드 무시 근거), helion (측정 구성 수 = 기계 독립 지표 + 벽시계 병기).
2. **단일 N 이 아니라 any-time 곡선을 그린다.** 예산점은 최소 3개다. 역전 지점을 숨기지 않는다. 근거: zhang-arxiv23 Tab 5 (100 평가에서 BO 역전), rebench Fig 2, mlebench 24h→100h.
3. **블랙박스 기준선 세트:** 랜덤 · 탐욕(표 예측 순) · 진화(EvoPress 식 (1+λ), μ>1) · **TPE/SMAC BO** · **LLM-seeded BO 하이브리드** · 해석적 knapsack (측정 0). TPE 와 하이브리드가 빠지면 LLM-BO 문헌(llambo · zhang-arxiv23 · gptuner · haqa · helion)의 표준선에 못 미친다.
4. **같은 제안 연산자에서 탐색기만 바꾸는 arm 을 둔다.** airadojo 가 보였듯 "정책 vs 연산자" 를 섞으면 이득의 귀속이 불가능하다. C1 에서는 같은 LLM 제안을 무작위로 받아들이는 arm, 또는 LLM 이 제안하고 탐욕이 선택하는 arm 이 해당한다.
5. **필수 절제 (같은 N):**
   - LLM, 측정 이력 없음 (evoprompting naive · reevo LLM only · alphaevolve No evolution · kernelbench 반복 샘플링)
   - LLM, 원장 검색 없음 (mlcopilot w/o experience · alphaevolve No context · llambo UnInf)
   - LLM, 타입 검사 없음: 위반률과 그 결과로 바뀐 최종 KL 을 보고한다. 선례가 없는 C1 고유 절제다
   - LLM 크기·종류 교체 (Opus/Sonnet/Haiku + 오픈 모델 1종): zhang-arxiv23 에서 GPT-3.5 는 랜덤보다 못했다(43.75%). mlebench 에서 Llama 405B 3.0% vs o1 16.9%. helion 에서는 3 모델이 ±3% 이내였다
6. **통계:** 셀(모델 × 예산)마다 블랙박스 seed ≥5, 에이전트 독립 실행 ≥3 을 둔다. 평균 ± SEM 또는 bootstrap 95% CI, 셀 단위 짝 검정 + Holm 보정을 쓴다 (protocol §7 과 일치). 에이전트 실행 간 성공률도 보고한다 (funsearch 512 cap set 4/140 처럼). reevo 의 black-box 설정 std ±0.82 는 작은 n 이 위험함을 보여 준다.
7. **선택 편향 분리:** 탐색은 검증 KL 로, 보고는 held-out(WikiText-2 test · C4 · lm-eval)으로 한다. 각 방법이 본 후보 수와 검증–test 순위 상관을 함께 낸다. 근거: airadojo §5.3 (오라클 격차 9–13 %p), alphaevolve 커널 shape 학습/평가 분리.
8. **비용 표:** LLM 입력/출력 토큰 · $ · 벽시계 · GPU 시간(측정·표·보정 분리)을 방법별 한 표로 낸다. 근거: rebench ($123/8h), mlagentbench (6M 토큰/$60), automlagent ($0.30/525 s), gptuner ($23.7), accelopt (오픈 모델 26× 저렴).
9. **무효 제안 회계:** 타입 검사로 거부된 제안을 예산에 넣는지 명시하고 거부율을 보고한다. 근거: autopruner (무효 라운드도 예산 소진), helion (형식 오류·중복 제거), apqf (수리 1회 후 실패).
10. **수치 충실도 감사:** 논문·로그에 보고하는 모든 수치가 원장 레코드 id 로 역추적되는지 자동 검사 결과(통과율)를 낸다. 근거: mlagentbench ~20% 근거 없는 주장, mlrbench ~80% 조작·무효, automlagent 의 추정 지표 사용. 이 항목은 C1-c 의 차별점을 실증하는 실험이기도 하다.
11. **암기·사전지식 통제:** 방법 이름을 익명화한 arm(reevo black-box 명명, mlebench 설명 난독화), 학습 컷오프 이후 모델 사용(llambo 사설 데이터, mlcopilot HyperFD)을 둔다. 양자화에서는 "GPTQ act_order 가 좋다" 같은 사전지식 효과를 분리하는 장치다.

## 7. unknown 과 사용자·오케스트레이터 결정 사항

- **autoqs-openreview26 · explorerreviewer-openreview26 본문:** OpenReview PDF 가 봇 검사에 막혀 초록만 확인했다. C1 차별점(혼합 방법 공간, 증거 인용 검사, knapsack · 블랙박스 등측정 비교)의 최종 판정은 본문을 확인해야 한다. 결정 항목: 사용자 브라우저로 PDF 2편을 받아 `source_pdf` 에 연결.
- **NeurIPS 2026 accepted papers:** 확인하지 못했다 (`unknown`). 12월 학회 전후로 재검색이 필요하다.
- **ICML 2026 · ICLR 2026:** 키워드 검색만 했다. 전수 목록 대조는 하지 않았다.
- **MLSys 2024:** 프로그램 JSON 이 37건만 반환했다 (부분). MLSys 2026 은 제목만 필터했다(초록 없음).
- **tap-arxiv25 의 반대 근거:** "직접 프롬프팅은 층별 비트 할당이 불안정" (§1). C1 은 반복·원장·검사로 이를 다룬다는 논리를 실험(무이력 · 무검사 arm)으로 보여야 한다.
- **C2 반례 2건 (autopruner Fig 4, arXiv 2609.09867):** R1 카드화와 C2 문구(가산성 가정) 재검토를 권고한다.
- 코드 상태 unknown: bopro · zhang-arxiv23 · autoqs · prompts.
- 카드 커밋: 이 레인은 커밋하지 않았다 (지시). `~/research-kb` 커밋은 오케스트레이터가 대주제 1커밋으로 한다.
