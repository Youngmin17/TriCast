<p align="center"><img src="assets/readme/slide-01.png" alt="Tri-Cast: 양자화된 AI 서비스를 온디바이스에 배포하기 전에 성능을 미리 확인하는 프레임워크" width="100%"></p>

<p align="center"><img src="assets/readme/slide-02.png" alt="온디바이스 AI 서비스는 이렇게 만들어집니다" width="100%"></p>

<p align="center"><img src="assets/readme/slide-03.png" alt="양자화란? 숫자 하나에 쓰는 칸(비트)을 줄이는 일입니다" width="100%"></p>

<p align="center"><img src="assets/readme/slide-04.png" alt="같은 모델도 4비트로 줄이면 결과가 달라집니다" width="100%"></p>

<p align="center"><img src="assets/readme/slide-05.png" alt="성능을 지키는 방법은 많지만, 내 모델엔 직접 구현·튜닝해야 합니다" width="100%"></p>

<p align="center"><img src="assets/readme/slide-06.gif" alt="Tri-Cast: 버튼 하나로 양자화하고 옵션을 조정하며, 결과를 시각화" width="100%"></p>

<p align="center"><img src="assets/readme/slide-07.png" alt="Auto-tune V1: 같은 크기에서 PPL 최대 20% 개선" width="100%"></p>

<p align="center"><img src="assets/readme/slide-08.png" alt="Auto-tune V2: Agent가 후보를 고르고, 사용자는 대화로 조정합니다" width="100%"></p>

<p align="center"><img src="assets/readme/slide-09.png" alt="Tri-Cast 소프트웨어: 양자화부터 최적화 · 평가 · 배포까지 한 번에" width="100%"></p>

<p align="center"><img src="assets/readme/slide-10.png" alt="STM 멘토와 PoC 를 진행 중이고, 오픈소스로 공개할 예정입니다" width="100%"></p>

---

## TriCast 한눈에

**v2.0 — Quantizer.** 레이어마다 양자화 방법(GPTQ · AWQ · SmoothQuant · MXFP4/8 · NVFP4 · NVINT4 · INT8 · BFP)을
웹 격자에서 클릭하거나 YAML 로 적으면, TriCast 가 레시피 한 장으로 모델에 적용하고 레이어별 오차(MSE · SQNR · 코사인)와
모델 품질(PPL · lm-eval 과제별)을 실행 환경 기록(git SHA · 모델 revision · 레시피 해시)과 함께 돌려준다.

**Auto-tune — 휴리스틱과 AI 를 함께 쓴다.** 크기 예산(가중치 평균 비트)을 주면 ① 층 하나만 양자화한 KL 표 위에서 정확
배낭(휴리스틱)이 층마다 방법을 즉시 고르고, ② Claude 에이전트가 ledger 에서 RAG 로 불러온 레이어 카드(방법별 KL · 레이어
오차 · 가중치/활성 분포 · 희소화 민감도, 숫자마다 레코드 id)를 인용해 그 배정을 고칠 제안을 내며, ③ 코드 검사(예산 · 인용
실존 · 수치 근거)를 통과한 제안만 GPU 로 측정한다. 요청은 문장으로도 받는다 ("Qwen3-0.6B 를 3.2비트 안에서, 첫 블록은
BF16 으로"). 저비트 방법 16종(GPTQ · RHT+GPTQ · AWQ(+clip) · HQQ 2–4비트 · NF4 · INT8)을 층마다 섞는다.

칩(NPU·GPU 세대)의 수 형식 · 반올림 · MMA 누산 규칙은 같은 레시피의 선택 축이다 — TriCast 가 그 산술을 GPU 의 CUDA
core 에서 **비트 단위로 똑같이** 에뮬레이트한다. 누산 모델(CoFDA · GDFS)은 NADPE 논문(MICRO'26)의 정의를 따르고,
레퍼런스는 그 독립 구현의 골든 벡터와 대조한다. 위 슬라이드는 v2(Quantizer · Auto-tune) 발표 자료다 (v7 덱, 6장은 Quantizer 화면 녹화).

## 설치와 빠른 시작

```bash
pip install -e ".[eval,dev,app]"        # Linux + CUDA GPU 에서는 ".[triton,eval,dev,app]" — 그 외에는 레퍼런스 백엔드
tricast formats && tricast presets      # 등록된 수 형식 20종 · 하드웨어 프리셋 8종 (출처 포함)
tricast cast --format fp8_e4m3 --rounding rne 464 465     # → 448 448 (포화)
tricast recipe-check hopper_fp8_w8a8                      # 스키마 검증 + 레시피 SHA-256
tricast ppl --model Qwen/Qwen3-0.6B --recipe hopper_fp8_w8a8
python -m app.server --demo             # TriCast Studio — GPU 없이 기록된 실행을 탐색 (fastapi, uvicorn)
                                        #   Quantizer 격자: /quantizer.html · 레이어 지도: /layers.html
python -m app.quantizer_recipe --preset qd_mixed           # 화면과 바이트 동일한 레시피 YAML
python -m app.autotune solve --table table.json --budget-bits 9/2 --name auto_4.50 --out out/   # Auto-tune
make check                              # ruff + CPU 테스트
```

## 사용 방법 (처음 실행하는 분께)

### 1. 화면만 보기 — GPU · API 키 없이 5분

```bash
git clone https://github.com/Youngmin17/TriCast.git && cd TriCast
python3 --version                                         # 3.10 이상이어야 한다 (macOS 기본 python3 는 3.9)
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pillow numpy jsonschema pyyaml   # 녹화 모드에 필요한 것만
PYTHONPATH=src python -m app.server --demo               # http://127.0.0.1:8765 (src: 레시피 파일 생성용, torch 불필요)
```

브라우저에서 **http://127.0.0.1:8765/autotune.html** 을 연다 (`file://` 로 직접 열면 동작하지 않는다).

| 단계 | 해 볼 것 | 보이는 것 |
|---|---|---|
| 1 크기 예산 | 슬라이더를 움직이거나 `3.15`, `403/128` 을 입력 | 브라우저가 정확 배낭을 다시 풀고 걸린 시간을 표시 |
| 1 요청 문장 | 저장된 예시 칩을 누름 | 해석 결과(모델 · 예산 · 제약 · 가정 · 코드 판정)와 제약이 걸린 배정 |
| 2 휴리스틱 배정 | 격자 칸에 마우스 · 방향키 | 칸마다 방법, 단층 KL, 가중치 MSE · 층 출력 SQNR (그 방법의 균일 리포트), 분포 · 희소화 정보, 같은 크기 균일 레시피와 비교 |
| 3 AI 에이전트 | 제안 카드에 마우스 | 바뀌는 칸이 격자에 점선으로 표시, 인용 레코드, 검사 결과, 측정 KL |
| 2 레시피 파일 | "이 배정의 레시피" → 레시피 파일 받기 | `app.autotune solve` 가 쓰는 것과 같은 YAML 한 장. 측정된 배정이면 SHA-256 이 GPU 측정 기록의 파일과 같다고 표시, 복사 · 내려받기 |
| 4 측정 결과 | 차트 · "표로 보기" | 균일 · 휴리스틱 · AI 최선 · AI 옵션 적용의 비트 대비 검증 KL / test PPL (GPU 기록값, GPU · 트리 SHA 표시) |

화면의 AI 응답과 측정값은 **저장된 기록**이다. 배지("저장된 응답 · 날짜", "GPU 기록값")가 그 사실을 표시한다.
Quantizer 격자(`/quantizer.html`), 레이어 지도(`/layers.html`), Studio(`/`)도 같은 서버에서 열린다.

### 2. AI 를 실시간으로 부르기 (선택)

[Claude Code](https://docs.claude.com/claude-code) CLI 에 로그인한 PC 에서, 전체 설치 뒤 세션 설정 파일을 주고 띄운다.

```bash
pip install -e ".[eval,dev,app]"
python -m app.server --demo --agent evals/fixtures/tune_inputs/s3150/loop.json   # 저장소에 든 데모 세션
```

저장소에 든 세션은 ledger 를 그 세션이 닿는 280 레코드로 줄인 것이라, 새 제안의 근거 검색이 데모(전체 ledger)보다 좁다.
직접 돌린 세션이 있으면 `runs/<세션>/loop.json` 을 준다.

요청 상자의 "요청 해석"과 "Claude 에게 새 제안 받기"가 실제로 호출된다 (도구 없음 · JSON 스키마 강제 · 비용 상한).
실시간 제안은 검사까지만 하고 측정하지 않는다. `--agent` 없이 띄우면 두 버튼은 이유와 띄우는 방법을 알려 주고 저장된
기록을 계속 보여 준다.

### 3. 명령줄로 같은 흐름 재현 (GPU 노드)

```bash
python -m app.autotune table --model Qwen/Qwen3-0.6B --revision <sha> --device cuda --split validation \
    --calibrate --methods bf16,gptq_int3,rht_gptq_int3,awq_clip_int3,hqq_int3,nf4,int8_w --mma fp32_fma --out t/table.json
python -m app.autotune solve --table t/table.json --budget-bits 403/128 --out t/auto      # 휴리스틱 (CPU, 초 단위)
python -m app.layer_profile --model Qwen/Qwen3-0.6B --device cuda --out t/profile.json   # 분포 · 희소화
python -m app.ledger ingest t --ledger runs/ledger.jsonl                                  # 수치 → 레코드 + RAG 문서
python -m app.layer_cards --table t/table.json --profile t/profile.json --out t/cards.json  # 레이어 카드 JSON
python -m app.tune_loop run --config loop.json --rounds 2 --per-round 4 --proposer claude-code   # AI 라운드
python scripts/quantizer/autotune_xfer.py --solutions t --ids auto_2.50 auto_3.00 \
    --options-from runs/<세션>/proposals/<최선> --out t/xfer                                 # AI 옵션 → 다른 예산 (측정은 GPU)
python -m app.export_autotune_demo --table t/table.json --measured t --profile t/profile.json \
    --session runs/<세션> [--transfer t/xfer] --out app/web/demo/autotune                    # 화면 데이터
```

### 4. 검사

```bash
make check                                                        # ruff + CPU 테스트
make test-app                                                     # 웹 앱 · Auto-tune · 튜닝 루프 테스트
node scripts/quantizer/autotune_check.mjs out python <table.json>  # 브라우저 배낭 = Python 배낭 (예산별)
python evals/run_tune_evals.py --table evals/fixtures/tune_inputs/table.json \
    --ledger evals/fixtures/tune_inputs/ledger.jsonl --fixture s3150=evals/fixtures/tune_inputs/s3150 \
    --mode stored --from-run 2026-11-09T0054 --change re-grade --golden n/a --out /tmp/tricast_evals   # 튜닝 evals 재채점 (LLM 호출 없음, 저장소를 바꾸지 않음, evals/EVAL_DESIGN.md §9)
```

피드백은 GitHub 이슈로 남겨 주면 된다 (무엇을 하려 했는지 · 본 화면 · 기대한 것).

## 문서 지도

| 무엇 | 경로 |
|---|---|
| 왜 만드는가 — 문제 정의서 | [`docs/PROBLEM.md`](docs/PROBLEM.md) |
| 무엇을 만드는가 — 제품 스펙과 수용 기준 (AC1–AC28) | [`docs/SPEC.md`](docs/SPEC.md) |
| 기술 타당성 스파이크 — 비트 정확 에뮬레이션과 속도 · 레이어 국소 지표 · Auto-tune | [`cuda_core_bit_exact_emulation.md`](docs/spikes/cuda_core_bit_exact_emulation.md) · [`2026-10-13_layer_sqnr_proxy.md`](docs/spikes/2026-10-13_layer_sqnr_proxy.md) · [`2026-10-13_autotune_mckp.md`](docs/spikes/2026-10-13_autotune_mckp.md) |
| Quantizer 측정 기록 (실행 · 해시) | [`docs/evidence/2026-10-13_quantizer_runs.md`](docs/evidence/2026-10-13_quantizer_runs.md) |
| 인터뷰 · 관찰 · Job Story | [`docs/research/interviews.md`](docs/research/interviews.md) |
| 도메인 온톨로지 (정본 YAML / 분석) | [`docs/ontology.yaml`](docs/ontology.yaml) · [`docs/ontology.md`](docs/ontology.md) |
| 코딩 에이전트 규칙 · 용어집 | [`AGENTS.md`](AGENTS.md) (Claude Code 는 [`CLAUDE.md`](CLAUDE.md) 가 import) |
| 구조화 출력 스키마 · 파싱 프롬프트 | [`src/tricast/schemas/`](src/tricast/schemas/) · [`src/tricast/prompts/parse_query.md`](src/tricast/prompts/parse_query.md) |
| 골든 케이스 · 골든 테스트 | [`tests/harness/golden_cases.yaml`](tests/harness/golden_cases.yaml) · [`tests/test_golden.py`](tests/test_golden.py) · [`tests/harness/golden_compare.yaml`](tests/harness/golden_compare.yaml) · [`tests/test_compare_golden.py`](tests/test_compare_golden.py) |
| 위임 프롬프트 원본 · 검증 루프 기록 | [`docs/prompts/delegation_compare.md`](docs/prompts/delegation_compare.md) (AC13, 지침 순서 한 사이클) · [`docs/prompts/delegation_examples.md`](docs/prompts/delegation_examples.md) · [`docs/prompts/wave1/`](docs/prompts/wave1/) |
| 골든 하니스 결함 주입 검사 · 용어집 작동 확인 | [`docs/prompts/delegation_golden_harness.md`](docs/prompts/delegation_golden_harness.md) · [`docs/prompts/glossary_check.md`](docs/prompts/glossary_check.md) |
| RAG 설정 · evals 안내 | [`config/rag.yaml`](config/rag.yaml) · [`evals/README.md`](evals/README.md) (v2 = 지금 제품, v1 = 10월 요청 파싱) |
| 웹 앱 (TriCast Studio · Quantizer · 레이어 지도 · Auto-tune) | [`app/README.md`](app/README.md) · [`app/web/autotune.html`](app/web/autotune.html) |
| 아키텍처 — 모듈 경계 · 데이터 흐름 · 불변식 | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| 중간평가 2 시연 시나리오 · QA 기록 | [`docs/DEMO.md`](docs/DEMO.md) · [`docs/QA.md`](docs/QA.md) |
| 튜닝 evals — 지표 · 데이터셋 · judge · 러너 위임 | [`evals/EVAL_DESIGN.md`](evals/EVAL_DESIGN.md) · [`evals/tune_evalset.jsonl`](evals/tune_evalset.jsonl) · [`evals/tune_judge_prompt.md`](evals/tune_judge_prompt.md) · [`docs/prompts/delegation_evals.md`](docs/prompts/delegation_evals.md) |
| 구현 · 검증 상태의 원천 | [`support_matrix.yaml`](support_matrix.yaml) |

## 검증 상태 (`support_matrix.yaml` 기록)

- 독립 구현 대조: NADPE MMA-Emu CUDA 골든 벡터 **1,716건 전부 비트 일치** (FP8 1,200 · NVFP4 172 · MXFP4 344; A100 · V100)
- Triton 커널 = 정확 레퍼런스 (fp64 · int64 · Fraction) 비트 일치 — GPU 스위트 896 passed (H200)
- 실제 칩 대조: H200 WGMMA **141건 중 140건 일치**, 1건(FP32-C 직접 입력) 불일치 — native 비트 일치는 미검증
- 프리셋 8종 모두 출처(`provenance`) 보유. 실제 칩 확인은 Hopper 일부뿐이며 나머지는 출처 기반
- 전체 재실행 (2026-10-09, geneva A100): lint 통과 · CPU 2,274 passed (2 skipped) · GPU 스위트 + 골든 945 passed
  (1 skipped) · 앱 161 passed — [실행 기록](docs/evidence/2026-10-09_full_check.md)
- 슬라이드 6 의 코드 줄 수: 빈 줄 · 주석 · 괄호만 있는 줄을 빼고 센 값이다. Tri-Cast 레시피 `fp8_f7_lowacc.yaml` 18줄,
  NADPE 공개 에뮬레이터의 FP8 경로 (core 6개 · `fp8_e4m3.cuh` · `scaled_fp8_mm.cuh` · `fp8_gemm_kernels.cu`) 는
  [`scripts/bench/count_code_lines.py`](scripts/bench/count_code_lines.py) 로 다시 세면 1,316줄이다 (슬라이드 1,296줄, 감소율은 둘 다 98.6%)
- 슬라이드 4 의 "로그 9·10" 은 2026-10-01 인터뷰 기록의 번호다 — 현재 [인터뷰 기록](docs/research/interviews.md)의 로그 01·02

## 참고 자료

누산 모델과 하드웨어 프리셋은 NADPE 논문의 정의와 값을 따른다 — J. Kim, C. Kim, J. Park, "Not All Dot Products
Are Equal: The Hidden MMA Arithmetic Design Space Drives Cross-Architecture LLM Inference Gaps", MICRO'26, artifact
[doi:10.5281/zenodo.21505180](https://doi.org/10.5281/zenodo.21505180). 수 형식(OCP MX v1.0, FP8, NVFP4), 양자화
방식(GPTQ, AWQ, SmoothQuant, QuaRot, KIVI), 평가 도구·데이터·모델의 1차 자료와 각 자료를 따르는 코드 위치는
[`docs/SPEC.md` 참고 자료](docs/SPEC.md#참고-자료)에 있다.

## 라이선스

[MIT](LICENSE). 모델 가중치와 데이터셋은 각 배포처의 라이선스를 따른다 (Llama 3.2 Community License 등).
