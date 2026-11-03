# 레인 R1 노트: 혼합 정밀도 · 혼합 방법 할당 탐색과 민감도 대리지표

> 2026-11-03 · 목적 `novelty-audit` · 레인 R1 (워커 1명, 하위 워커 없음) · survey.yaml 은 오케스트레이터가 쓴다.
> 카드는 `~/research-kb/papers/` 에 있고 `kb.py lint` 결과는 오류 0 이다.
> 판정 어휘는 `preempts` / `partial` / `adjacent` / `none` / `unknown` 이다. 판정마다 근거 위치는 카드의 절·표다.

## 0. 요약

| 주장 | 레인 판정 | 한 줄 근거 |
|---|---|---|
| C1 에이전트 혼합 방법 탐색 | **partial** — 전체 조합을 선점한 것은 찾지 못했다. 다만 구성 요소마다 선행이 있다. | 공유 로그와 리뷰어를 둔 양자화 에이전트는 explorerreviewer-openreview26 (ICLR'27 심사 중, 초록만 확인)에 있다. 측정 근거로 층별 비트를 고르고 실행 전에 검증하는 LLM 플래너는 apqf-arxiv26 (비전 모델)에 있다. BO·NSGA-II 를 비교군으로 둔 LLM 양자화 에이전트는 haqa-arxiv26 (전역 하이퍼파라미터만 다룸)이다. |
| C2 단일 층 KL 표 → 정확 MCKP | **partial (핵심 메커니즘은 선점됨)** | 단일 층 KL 표 + 가산 + 크기 예산 DP 는 zeroq-cvpr20 (2020, CNN, 비트폭)에 있다. 측정 손실 표 + IP 는 adaquant-icml21 에 있다. LLM 에서 단일 층 섭동 + 가산 + 정확 DP 는 higgs-naacl25 이다. (층, 포맷 레시피)별 단일 층 KL 표는 코드 modeloptaq-github26 의 `kl_div` 에 있다. 가산 합이 구성 순위를 보존한다는 결과(쌍 오순서 2% 이하)는 hill-arxiv26 이 2026-07 에 먼저 보고했다. |
| C3 이질적 방법 하나의 층별 공간 | **partial** | GPTQ·AWQ·SmoothQuant 를 층별로 고르는 것은 calm-arxiv25 에 있다. 포맷 + AWQ/SQ config + KV 층별 탐색은 modeloptaq-github26 에 있다. 블록마다 W-only 와 W-A scheme 을 고르는 것은 mxmoe-icml25 에 있다. 데이터 타입 혼합은 autoscheme-github25, KV 만 층별로 다루는 것은 kvtuner-icml25 다. GPTQ+AWQ+SQ+Hadamard+NVFP4/MX/INT8/BFP+N:M+KV 를 한 공간에 넣은 사례는 찾지 못했다. 다만 Unsloth 같은 비공개 도구는 확인하지 못해 `unknown` 이다. |

프로토콜(`protocol.md`)에 직접 영향이 있는 사실은 세 가지다.

1. **H1 은 일부 선점됐다.** calm-arxiv25 의 Tab 4 는 같은 평균 4-bit 에서 비트만 섞는 경우와 방법을 섞는 경우를 이미 비교했고, 방법을 섞는 쪽이 낫다고 보고했다. 그래서 H1 의 신규성은 다음 넷에 기대야 한다.
   - 더 넓은 공간: 포맷, 회전, 희소, KV, 방법 옵션
   - Linear 단위
   - stored-bits 예산 아래의 정확한 최적화
   - end-to-end KL 목적 (CALM 은 국소 CKA 를 쓰고 greedy 로 고른다)
2. **H2 비교군의 표준선은 이미 정해져 있다.** hill-arxiv26 은 N=300 측정 구성으로 GP·RF·AMQ-RBF·CoopQ 를 같은 knapsack 에 넣어 비교했다. 리뷰어는 이 프로토콜을 기준으로 볼 것이다.
3. **ModelOpt AutoQuantize 는 반드시 비교해야 한다.** `kl_div`·`gradient`·`aumann_shapley` 세 모드를 모두 넣는다. §6 외부 비교군에 이미 후보로 올라 있고, 레인 판정으로 확정을 권고한다.

## 1. 검색 로그 (2026-11-03)

### 1.1 출처와 질의

| 출처 | 질의·대상 | 결과·비고 |
|---|---|---|
| KB RAG `kb.py search` | mixed precision search / layer sensitivity Hessian / bit allocation knapsack / per-layer quantization assignment / mixed-precision LLM quantization / evolutionary search quantization / AutoQuantize / HAWQ / HAQ reinforcement / agent LLM quantization / salience mixed precision / importance matrix | 기존 카드 재사용: evopress-icml25, mixllm-mlsys26, owq-aaai24, squeezellm-icml24, spqr-iclr24, resq-icml25, fgmp-arxiv25, blockdialect-icml25, jsq-icml24, intvsfp-icml26 |
| DBLP API | `mixed precision quantization large language models` | **실패.** Anubis 봇 차단에 걸렸다. 서지 확인은 OpenReview·OpenAlex·PMLR·NeurIPS proceedings·arXiv comment 로 대체했다. |
| OpenAlex search (mailto 없음) | mixed-precision quantization LLM / layer-wise mixed precision bit allocation / sensitivity knapsack bit allocation / ILP bit-width allocation / evolutionary search quantization LLM / LLM agent quantization configuration search / KL layer sensitivity mixed precision / LLM agent mixed-precision / LLM-guided search quantization policy / Hessian aware quantization / differentiable mixed precision search / RL hardware-aware quantization / linearity theorem DP / multiple-choice knapsack MPQ / joint pruning quantization search | 15 질의, 질의당 25–50 건 |
| OpenAlex title filter | 12 개 제목의 venue 확인 | 1 회 429. 나머지는 OpenReview·PMLR 로 확인했다. |
| arXiv API | 제목 조회 약 35 편, id_list 3 회, 키워드 sweep 7 회 (`agent AND quantization AND "mixed-precision"`, `agentic AND quantization`, `"LLM-driven" AND quantization AND search`, `"coding agent" AND quantization`, `"large language model" AND "per-layer" AND "quantization method"`, `"mixed-method" AND quantization`, `"single-layer" AND "KL divergence" AND …`) | sweep 에서 새로 찾은 것: calm-arxiv25, quantforge-arxiv26, fca-arxiv26, RAMP, Joint SP+MPQ |
| OpenReview API v2 `notes/search` (group 지정) | term `mixed-precision quantization`: ICLR 2023·2024·2025·2026·2027, ICML 2024·2025·2026, NeurIPS 2023·2024·2025·2026. term `agent quantization`: ICLR 2027 | 질의당 상위 40 건. NeurIPS 2026 은 0 건이 반환됐다 (색인 안 됨으로 보며, 미확인). ICLR 2023·NeurIPS 2023 은 출력이 잘려 일부만 봤다. `notes?id=`·PDF 는 403 또는 봇 검증에 막혔다. |
| MLSys proceedings | 2024·2025·2026 목록에서 제목 정규식 `mixed\|precision\|bit\|quantiz` | 2024 페이지는 37 편만 파싱됐다 (부분). |
| WebSearch (3/45 회 사용) | ① `LLM agent proposes per-layer quantization configuration mixed-precision large language model 2026 arXiv` ② `"agentic" OR "LLM-driven" quantization search mixed-precision recipe … Bayesian optimization` ③ `Unsloth Dynamic 2.0 GGUF quants per-layer quantization selection KL divergence method` | ① 에서 AdaqL (NeurIPS 2026), RAMP, APreQEL, GEMQ 를 찾았다. ② 에서 TAP, AutoQRA 를 찾았다. ③ Unsloth 의 선택 알고리즘은 공개되지 않았다. |
| WebFetch | arXiv HTML 5 편 (HAQA, APQF, AMQ, HIGGS, TAP), PDF 3 편 (ZeroQ, AdaQuant, Chen'21; 로컬 pdftotext), NeurIPS virtual (AdaqL), ACL Anthology (2026.emnlp-main.142), OpenReview PDF (봇 검증 실패) | 그 밖의 PDF 25 편은 arXiv 에서 직접 받아 pdftotext 로 정독했다. |
| GitHub `gh api` / raw | ModelOpt (`algorithms.py`, `_auto_quantize_shapley.py`, `model_quant.py`, recipes, commit 이력), AutoRound (`step_by_step.md`, `auto_scheme/delta_loss.py`, commits), llama.cpp (`src/llama-quant.cpp`, PR #15550), AIMET (`amp.rst`), INC (`autotune.md`), NNCF (`mixed_precision.py`), HQQ README, llm-compressor `examples/quantization_non_uniform`, torchao 검색 | 별·라이선스·last push 는 약 25 개 repo 를 확인했다. |

### 1.2 venue × 연도 확인 범위

LLM 관련 주제는 2023–2026 을 다 보고, 2018–2022 는 대표작만 넣었다.
표기: ✓ 확인 / △ 부분 확인 (키워드 검색만 했거나 출력이 잘림) / — 범위 밖

| venue | 2018–22 대표작 | 2023 | 2024 | 2025 | 2026 | 2027 심사 중 |
|---|---|---|---|---|---|---|
| ICLR | ✓ AutoQ'20, BRECQ'21 | △ | ✓ (group search) | ✓ | ✓ | ✓ (2 term) |
| ICML | ✓ HAWQ-V3'21, AdaQuant'21 | △ (OpenAlex) | ✓ | ✓ | ✓ | — |
| NeurIPS | ✓ HAWQ-V2'20, Bayesian Bits'20 | △ | ✓ | ✓ | △ (search 0 건. AdaqL 은 WebSearch 로만 찾음) | — |
| MLSys | — | — | △ (37 편만 파싱) | ✓ | ✓ | — |
| CVPR/ICCV/ECCV | ✓ HAQ'19, HAWQ'19, ZeroQ'20, EdMIPS'20, Chen'21, LIMPQ'22 | — | — | — | — | — |
| ACL/EMNLP/NAACL (LLM 양자화 main) | — | △ | △ | △ AMQ, HIGGS | △ CoopQ (Findings), Scaling Laws for Mixed Quant (EMNLP'26), Q-Strata·FAMPWQ (EMNLP'26 제목만) | — |
| DAC (OpenAlex 로 확인) | — | — | △ APTQ | △ CLADO | △ ExQuant (제목만) | — |
| arXiv (핵심 겹침만) | — | — | △ | △ | △ | — |
| 코드 레인 | ModelOpt · AutoRound · llama.cpp · AIMET · INC · NNCF · HQQ · llm-compressor · torchao · Unsloth (비공개) | | | | | |

## 2. 주장 × 가장 가까운 선행 (1–3 편) × 차이 × 판정

| 주장 | 가장 가까운 선행 | 차이 (한 문장) | 판정 | 근거 위치 |
|---|---|---|---|---|
| C1 | explorerreviewer-openreview26 (ICLR'27 심사 중) | 코딩 에이전트가 방법을 고르고, 리뷰어가 공유 research log 위에서 근거 없는 결론을 막는다. 대상은 2-bit weight-only 전역 레시피이고, 비교군은 단독 에이전트와 공개 기준선이다. 층별 혼합 방법, 결정적 타입 검사, 블랙박스 등측정 비교는 초록에 없다. | partial (본문 미확인 부분은 unknown) | 카드 '메커니즘'·'수치' (초록) |
| C1 | apqf-arxiv26 | 측정 프로파일로 LLM 플래너가 층별 비트폭을 제안하고 실행 전에 측정 데이터·모듈과 대조 검증한다. 대상은 CNN/ViT 이고, 비트폭만 다루며, 1회 제안이다. BO·진화·knapsack 과의 등측정 비교가 없다. | partial | 카드 '메커니즘' (§3 검증), Tab 1–3 |
| C1 | haqa-arxiv26 | LLM 에이전트(ReAct)가 측정 피드백으로 반복 제안하고 BO·random·NSGA-II 와 10 라운드 안에서 비교한다. 결정 대상은 전역 미세조정 하이퍼파라미터와 배포 비트폭이다. 원장 검색과 타입 검사는 없다. | partial | Tab 1–2, §3.2 |
| C2 | zeroq-cvpr20 | 층 i 만 k-bit 로 둔 출력 KL 표 Ω_i(k) (Eq 2)를 가산하고, 크기 예산 DP Pareto (Eq 4)로 푼다. 이것이 C2 메커니즘의 핵심이다. 다만 대상은 CNN 비트폭이고, patch floor 차감과 예측합-실측 순위 상관은 없다. | partial (핵심 선점) | Eq 2·4, §III-B, 부록 A, Tab Ia |
| C2 | modeloptaq-github26 (`kl_div` 는 2025-11-26 이전부터 있음) | (층, 포맷+알고리즘 config)별로 단일 층 KL 을 측정한 표가 제품 코드에 있다. 선택 방식은 가산 MCKP 가 아니라 max 점수 minimax 이진 탐색이다. `gradient` 모드는 Fisher 가산 점수를 LP 로 푼다. | partial | `AutoQuantizeKLDivSearcher.estimate_sensitivity_scores`, `run_search_with_stats` |
| C2 | higgs-naacl25 · hill-arxiv26 | HIGGS 는 LLM 층별 섭동의 end-to-end PPL/KL → 가산 α_l·t² → 평균 비트 예산 아래 정확 해(CP-SAT)다. 후보는 HIGGS 격자뿐이고 예측-실측 일치는 그림으로만 보였다. Hill 은 가산 합의 단조 변환이 순위를 보존하고(쌍 오순서 2% 이하) 고립 민감도가 구성 손실을 71–376% 잘못 가격 매긴다는 것을 비트 등급 수준에서 보였다. | partial | HIGGS Thm 1·Alg 3·§5·Fig 1·Tab 3–4 / Hill abstract·§7·App C |
| C3 | calm-arxiv25 | 디코더 층마다 {GPTQ, AWQ, SmoothQuant(+α 변형)} 중 하나를 CKA greedy 로 고른다. W4A8 로 고정하고 예산 최적화는 없다. 포맷·회전·희소·KV 는 없다. | partial | §3.2–3.3, Tab 1·4·6 |
| C3 | modeloptaq-github26 | 층별로 FP8·NVFP4·INT4-AWQ·W4A8-AWQ·INT8-SQ·사용자 config 와 KV 포맷 쌍을 effective_bits 예산 아래에서 고른다. Hadamard, N:M, GPTQ 옵션은 사용자가 config 로 나열해야만 들어간다. | partial | `model_quant.auto_quantize` docstring, recipes yaml |
| C3 | mxmoe-icml25 | MoE linear 블록마다 W-only 와 W-A scheme(비트·그룹 변형)을 ILP 로 고른다. GPTQ 하나만 쓰고 회전·스무딩·희소는 없다. | partial | Eq 5–7, App A |

## 3. 카드별 겹침 판정 (C1 / C2 / C3)

| 카드 | C1 | C2 | C3 | 근거 |
|---|---|---|---|---|
| haq-cvpr19 | adjacent (RL 블랙박스, 층별 비트) | none | none | §3, Tab 3 |
| hawq-iccv19 | none | adjacent (Hessian 고유값 대리지표) | none | §3, Tab 2 |
| hawqv2-neurips20 | none | adjacent (가산 Hessian-trace Ω + Pareto. C2 가 대조군으로 삼는 바로 그 대리지표) | none | Eq 2.11, Tab 2 |
| hawqv3-icml21 | none | adjacent (가산 Hessian + 정확 ILP, 지연 제약) | none | Eq 8–11, Tab 2 |
| edmips-cvpr20 | adjacent (미분 탐색) | none | none | Tab 1 |
| bayesianbits-neurips20 | adjacent | none | adjacent (양자화 + 0-bit 가지치기 통합) | §2, Tab 4 |
| autoq-iclr20 | adjacent (계층 RL) | none | none | abstract |
| zeroq-cvpr20 | none | **partial (핵심)** | none | Eq 2·4, Tab Ia |
| adaquant-icml21 | none | **partial** (측정한 단일 층 손실 + 가산 IP) | none | Eq 5, §4 |
| chen-iccv21 | none | adjacent (MCKP 정식화이나 Hessian 을 쓰고 greedy 로 푼다) | none | §3.4, Tab 1 |
| brecq-iclr21 | none | adjacent (가산 가정 비판, 블록 내 교차 항 + GA) | none | §3.4, Fig 2 |
| limpq-eccv22 | none | adjacent (학습한 지표 + ILP) | none | Eq 4 |
| clado-dac25 | none | adjacent (가산 가정에 대한 반례: 교차 항을 빼면 −6.9%p) | none | Tab 1 |
| higgs-naacl25 | none | **partial** | adjacent (단일 방법의 격자 변형만) | Thm 1, §5, Tab 3–4 |
| hill-arxiv26 | adjacent (N=300 같은 측정에서 GP·RF 대리모델 비교) | **partial** (가산 순위 보존, knapsack, ModelOpt 대비) | none | abstract, §6–7 |
| amq-emnlp25 | adjacent (NSGA-II + RBF 블랙박스, 실측 약 1 만 회) | adjacent (단일 층 JSD 는 가지치기용으로만 씀) | none | §3, Tab 1·4·6 |
| evopress-icml25 (기존 카드) | adjacent (KL 적합도 진화 탐색, 동적 비트폭 GPTQ 2–6) | adjacent (가산 가정이 깨짐을 보임) | none | 카드 Tab 2·14–15 |
| slimllm-icml25 | none | adjacent (층 안 그룹 배분. ILP < SBA, Tab 4) | none | Tab 1·4 |
| pbllm-iclr24 | none | none | adjacent | Tab 2 |
| kvtuner-icml25 | none | adjacent | partial (KV 정밀도 쌍 층별 탐색) | abstract, §4 |
| mxmoe-icml25 | none | adjacent (블록 하나만 바꾼 국소 출력 거리 + ILP) | **partial** | Eq 5–7 |
| aptq-dac24 | none | adjacent | none | Tab 1 |
| coopq-acl26 | none | adjacent (Shapley 쌍 상호작용 + QP) | none | abstract |
| autoqra-icml26 | adjacent (EA + trust-region BO) | none | none | Tab 1 |
| haqa-arxiv26 | **partial** | none | none | Tab 1–2 |
| apqf-arxiv26 | **partial** | adjacent (가지치기 민감도 sweep) | adjacent (가지치기 + 비트) | §3, Tab 1–3 |
| tap-arxiv25 | adjacent (LLM 이 proxy 프로그램을 진화시킴) | adjacent | none | Tab 3 |
| explorerreviewer-openreview26 | **partial / unknown** | unknown | unknown | 초록 |
| bytemckp-openreview26 | none | **partial** (바이트 MCKP + 측정 민감도 vs llama.cpp 프리셋) | adjacent (GGUF 타입) | 초록 |
| calm-arxiv25 | none | adjacent (국소 CKA greedy) | **partial** | Tab 1·4·6 |
| quantforge-arxiv26 | adjacent (증거 기반 LLM 진화, 240 호출 등예산) | none | none | abstract |
| fca-arxiv26 | adjacent (LM 플래너 vs TPE·random, 40 구성) | none | adjacent | abstract |
| fliqs-automl24 | adjacent | none | adjacent (층별 INT/FP 포맷) | abstract |
| micromix-iclr26 | none | none | adjacent (층 안 채널 MXFP4/6/8) | §3 |
| modeloptaq-github26 | none | **partial** | **partial** | 소스 |
| autoscheme-github25 | none | adjacent (기울기 대리 + DP) | partial (MXFP4/8·NVFP4·GGUF 혼합) | delta_loss.py, docs |
| llamacppquant-github26 | none | adjacent (프리셋은 휴리스틱. PR #15550 은 국소 오차 + Lagrangian) | adjacent | llama-quant.cpp, PR 본문 |
| aimetamp-github26 | none | partial (그룹 하나만 바꾼 측정 정확도 목록 + greedy Pareto, 정수 비트) | none | amp.rst Phase 1–2 |
| incautotune-github26 | adjacent (전역 방법×옵션 순차 시도) | none | adjacent | autotune.md |
| nncfmp-github26 | none | adjacent (비율 + 대리지표 순위) | none | mixed_precision.py |

## 4. 제외 · 카드 미작성 항목과 이유

| 항목 | 이유 |
|---|---|
| DNAS (Wu et al., arXiv 2018) | arXiv 만 있다. 미분 탐색 축은 edmips-cvpr20 으로 대표했다. |
| Uhlich et al. "Mixed Precision DNNs: all you need is a good parametrization" (ICLR 2020) | tier 1 이지만 미분 탐색 축이 EdMIPS·Bayesian Bits 와 같고, C1–C3 와의 겹침은 adjacent 이하다. 시간 예산 때문에 카드를 쓰지 않았다. 필요하면 추가한다. |
| OMPQ (AAAI 2023), Q-BERT (AAAI 2020), FracBits (AAAI 2021) | AAAI 는 venues.md tier 에 없다. 대리지표 계열이라 adjacent 다. |
| LLM-MQ (NeurIPS 2023 ENLSP workshop) | workshop 은 tier 0 이다. 메커니즘(1차 Taylor + ILP)이 modeloptaq `gradient` 와 같다. |
| CMPQ, CPTQuant, MixQuant, SFMP, GAMMA, DMQ, APreQEL, RAMP (SAC RL), Joint SP+MPQ (arXiv 2606.07819) | arXiv 또는 tier 외다. 비트폭 할당 축이고 C1–C3 를 넘어서는 선점은 없다. RAMP 의 "정책 zero-shot 전이"는 H3 와 관련이 있으니 R2 레인에서 확인하길 권한다. |
| GEMQ (ICML 2026), Q-Strata·FAMPWQ (EMNLP 2026), ExQuant (DAC 2026) | 제목과 초록 수준만 봤다. MoE 전문가 단위·Fisher 비트 할당이라 adjacent 다. 시간 때문에 카드를 쓰지 않았다. |
| Scaling Laws for Mixed Quantization (EMNLP 2026 main) | 혼합 비율 스케일링 법칙이고 할당 탐색이 아니다. adjacent. |
| A KL Lens on Quantization (EDGE workshop / arXiv 2026) | 단일 층 KL forward-only 민감도라 C2 와 관련이 있다. workshop 이라 tier 0 이다. **후속 레인에서 본문 확인을 권한다.** |
| AdaqL (NeurIPS 2026, OpenReview 8CHf5OB1Zz) | NeurIPS virtual 초록만 봤다 ("할당을 end-to-end 최적화로"). OpenReview 가 403 이라 메커니즘을 확인하지 못했다 → **unknown**. 근거가 약해 카드를 쓰지 않았다. |
| ICLR 2027 MPQ 제출물: BRACE, PACE, DepthQuant, LSMXQ, RampQ, MixQuant, HiCAP-Q, Task-Aware QUBO, Attention Sensitivity→Layer Role, Function-Space Sensitivity 등 | 심사 중이다. BRACE·PACE 초록을 읽었고, 국소 2차·곡률 대리지표 계열이라 adjacent 다. 카드는 C1·C2 와 직접 겹치는 두 편(explorerreviewer, bytemckp)만 썼다. |
| Progressive Mixed-Precision Decoding (ICLR 2025), DP-LLM (NeurIPS 2025), MorphServe (MLSys 2026) | 런타임·단계별 정밀도 전환이라 정적 레시피 할당과 다르다. |
| Any-Precision LLM (ICML 2024), Matryoshka Quantization (ICML 2025), BitStack (ICLR 2025) | 크기를 바꿀 수 있는 모델 표현이고 할당 탐색이 아니다. AMQ 의 비교군이다. |
| GuidedQuant (ICML 2025), QEP | end-loss guided 층 양자화다. 층 간 할당은 하지 않는다. |
| HQQ (dynamic_config), llm-compressor (multi-modifier recipe), torchao autoquant | HQQ·llm-compressor 는 층별 이질 방법 레시피를 **수동으로** 표현만 한다. 예: AWQ W4A16 attn + GPTQ W8A8 mlp. 문서는 "best practices … after further research and sensitivity analysis" 라고 쓴다. 정확도 기반 자동 탐색은 없다. torchao autoquant 는 측정 **속도**로 고른다. C3 에는 adjacent 다 (표현은 있고 탐색은 없음). |
| Unsloth Dynamic 2.0 | "모든 층의 타입을 동적으로 조정해 평균 KLD 를 최소화"한다고만 밝힌다. 선택 알고리즘은 공개되지 않았다 → **unknown**. |
| MicroMix 이외의 ICLR 2026 MPQ (IMPQ desk reject, AutoMixQ withdrawn, CMPS rejected) | 미게재다. IMPQ 는 coopq-acl26 과 같은 논문이다. |

## 5. 작성한 카드 (39 편, 전부 신규)

- 기초 (2018–2022): haq-cvpr19, hawq-iccv19, hawqv2-neurips20, hawqv3-icml21, edmips-cvpr20, bayesianbits-neurips20, autoq-iclr20, zeroq-cvpr20, adaquant-icml21, chen-iccv21, brecq-iclr21, limpq-eccv22
- 가산성·교차 항: clado-dac25, coopq-acl26, hill-arxiv26
- LLM 할당 (2023–2026): higgs-naacl25, amq-emnlp25, slimllm-icml25, pbllm-iclr24, kvtuner-icml25, mxmoe-icml25, aptq-dac24, autoqra-icml26, micromix-iclr26, calm-arxiv25, fliqs-automl24
- 에이전트·LLM 주도: haqa-arxiv26, apqf-arxiv26, tap-arxiv25, quantforge-arxiv26, fca-arxiv26, explorerreviewer-openreview26, bytemckp-openreview26
- 코드 레인: modeloptaq-github26, autoscheme-github25, llamacppquant-github26, aimetamp-github26, incautotune-github26, nncfmp-github26
- 참조한 기존 카드: evopress-icml25, mixllm-mlsys26, blockdialect-icml25, owq-aaai24, squeezellm-icml24, spqr-iclr24, resq-icml25, fgmp-arxiv25, jsq-icml24

`kb.py lint` 결과는 카드 1359, 오류 0 이다. 새 카드의 경고 6 건은 모두 의도한 것이다.
- amq·higgs: EMNLP·NAACL 을 LLM 양자화 main 이라 tier 1 로 뒀다.
- coopq: venue 가 'ACL Findings' 다.
- openreview 2 건: 심사 중이다.
- fliqs: venue 가 'AutoML' 이다.

## 6. 리뷰어가 요구할 C1/C2 비교군 5 (+ 필수 절제 2)

| # | 비교군 | 왜 요구되나 | 공식 코드 상태 (2026-11-03) |
|:-:|---|---|---|
| 1 | **NVIDIA ModelOpt AutoQuantize**: `gradient`(Fisher-Taylor + LP), `kl_div`(단일 층 KL 표 + minimax), `aumann_shapley`(Hill 추정기) | 같은 '(층, 포맷/방법 config) 단일 층 KL 표 + 예산' 을 제공하는 산업 표준이다. C2 의 정확 MCKP·patch floor 가 이것보다 낫다는 것을 보여야 한다. 후보 config 를 TriCast 후보와 같게 맞춘다. | official, Apache-2.0, 5.5k★, 2026-11-03 push |
| 2 | **HIGGS 선형성 정리 DP** (higgs-naacl25) | LLM 에서 "단일 층 섭동 → 가산 → 정확 DP" 의 가장 가까운 tier-1 선행이다. 잡음 주입 α 적합이 측정 KL 표보다 싸다는 반론을 받게 된다. | official (FLUTE: 양자화기·커널). 동적 할당 스크립트 공개 여부는 미확인. 공식이 단순해 재구현할 수 있다. |
| 3 | **EvoPress** (evopress-icml25) | KL 적합도를 쓰는 예산 보존 진화 탐색이다. GPTQ 2–6 bit DB 로 LLM 동적 비트폭을 이미 다뤘다. 같은 측정 수 블랙박스 비교의 표준이다. | official, IST-DASLab/EvoPress, Apache-2.0, 2026-06-14 |
| 4 | **AMQ** (amq-emnlp25) | NSGA-II + RBF 대리모델로 LLM 층별 비트 탐색을 한다. 대리모델 기반 블랙박스 비교군이다. 원래 예산은 실측 약 1 만 회이므로 같은 N 곡선으로 맞춰 비교해야 한다. | official, dlwns147/amq, Apache-2.0, 2026-04-29 |
| 5 | **Hill 가산·coverage knapsack (AS-add)** 와 **CoopQ** | 가산 가정의 편향을 교정하는 해석적 할당기다. C2 의 "Spearman 0.99" 가 Hill 의 순위 보존 결과와 무엇이 다른지, CLADO·BRECQ·EvoPress 의 반례와 어떻게 맞는지 리뷰어가 물을 것이다. | Hill: 논문 코드 unknown, ModelOpt `aumann_shapley` 가 구현함(official). CoopQ: unknown. |
| 절제 A | 에이전트에서 원장 검색과 타입 검사를 뺀 판 (Explorer–Reviewer 식 LLM 리뷰 대조 포함) | C1 의 기여가 "LLM" 이 아니라 "근거 + 검사" 라는 것을 보이려면 필요하다. explorerreviewer-openreview26 이 직접 경쟁작이다. | ER 코드 unknown (익명 심사 중). 프로토콜 §6 의 절제 행과 같다. |
| 절제 B | TPE/BO (Optuna, TuRBO) 와 random 을 같은 N 으로 | HAQA·FCA·AutoQRA·Hill 이 모두 이 비교를 했다. 빠지면 바로 지적된다. | Optuna (공개). AutoQRA 코드 unknown. |

C3 전용으로 하나 더: **CALM** (층별 GPTQ/AWQ/SQ CKA 선택). 같은 W4A8 조건에서 재현해 비교해야 한다. 공식 코드는 미확인(unknown)이라 재구현이 필요하다.

## 7. unknown 과 사용자 결정 사항

- **unknown**:
  - explorerreviewer-openreview26·bytemckp-openreview26 은 본문을 보지 못했다. OpenReview PDF 는 봇 검증에 막혔다. 확인하지 못한 것은 다음과 같다.
    - ER: 층별 할당을 하는지, 로그 내용을 검사하는 방식, 등측정 비교를 했는지
    - byte-MCKP: 민감도 측정 방식과 후보 포맷
  - AdaqL (NeurIPS 2026) 의 메커니즘
  - Unsloth Dynamic 2.0 의 알고리즘
  - ModelOpt `kl_div` 의 도입 시점: 2025-11-26 이전이라는 것만 안다. GitHub 공개 이력이 2025-07-31 부터라 그 전은 확인할 수 없다.
  - 코드 공개 여부: Bayesian Bits, CoopQ, Hill, AutoQRA, APQF, TAP, CALM
  - NeurIPS 2026 accepted 전수 목록: 검색 API 가 0 건을 돌려줬다.
  - ICLR 2023·NeurIPS 2023 group search 는 출력이 잘려 일부만 봤다.
- **수치 주의**:
  - aptq-dac24 Tab 1 의 FP16 LLaMA-7B C4 PPL 5.22 는 통상값(약 7.1)과 다르다.
  - haqa-arxiv26 Tab 2 의 평균은 행별 점수의 평균과 4–5 점 어긋난다 (WebFetch 추출 기준).
  - 두 수치는 인용하기 전에 PDF 로 다시 확인한다.
- **결정 필요**:
  1. ICLR 2027 경쟁 제출물 2 편의 PDF 를 사용자 기관 접근이나 로그인 브라우저로 받을지 정한다. C1 의 차별 문구가 이 본문에 달려 있다.
  2. 프로토콜 §6 의 외부 비교군 확정을 권고한다. 후보는 SliM-LLM · ModelOpt AutoQuantize · EvoPress 이고, 여기에 HIGGS-DP, AMQ, CALM(재구현)을 더할지 정해야 한다. SliM-LLM 은 층 안 그룹 할당이라 트랙 M 의 "혼합 정밀도 PTQ" 대조로는 쓸 수 있다. 하지만 층별 할당 비교군으로는 약하다.
  3. H1 문구를 고친다. CALM Tab 4 가 "같은 4-bit 에서 방법 혼합 > 비트 혼합" 을 이미 보고했으므로, H1 을 "Linear 단위 · 넓은 공간 · 정확 예산 최적화 · end-to-end KL" 조건으로 좁혀 쓰길 권한다.
