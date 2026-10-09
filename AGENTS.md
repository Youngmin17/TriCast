# AGENTS.md — TriCast

> 코딩 에이전트가 **작업 중 상시 준수할 규칙**. 도구와 무관하게 이 파일 한 벌이 원천이다 (Claude Code 는
> `CLAUDE.md` 의 `@AGENTS.md` 로 읽는다). 왜 만드는가는 `docs/PROBLEM.md`, 무엇을 만드는가는 `docs/SPEC.md`,
> 도메인 구조는 `docs/ontology.yaml`, 수치 의미론은 실행 가능한 정의인 `src/tricast/reference/` 가 정본이다
> — 여기서는 복제하지 않고 가리킨다.
> 절대 규칙과 완료의 정의는 2026-10-01 팀이 확정했다.

## 1. 제품 맥락

양자화 연구자와 NPU 연산기 개발자가 수 형식·양자화·MMA 누산 알고리즘·희소성·이상치 보존을 레시피로 정의하면,
TriCast 가 그 산술을 CUDA core 에서 **비트 단위로 정확하게** 에뮬레이트해 Hugging Face 모델 품질(PPL·lm-eval)과
누산기 오차를 근거와 함께 돌려준다 — 조합이 바뀔 때마다 커널을 다시 쓰지 않도록 (docs/research/interviews.md
페인 P1·P2, 로그 01, 02, 03, 07). 핵심 가치는 "믿을 수 있는 산술 + 빠른 LLM 수준 평가". 이 저장소는
AI캡스톤디자인 과제 저장소이기도 하다.

## 2. 도메인 용어집 (어휘 발췌 — 인터뷰로 역추적되는 구조는 `docs/ontology.yaml`, 나머지는 코드 어휘)

- Format: `name`, `kind` (float / int / pow2), `max_normal`, `special` (ieee / fn / fnuz / none)
- QuantSpec: `format`, `granularity` (tensor / row / group / block), `group_size`, `scale.method`
  (absmax / pow2_floor / pow2_ceil / mse / percentile), `rounding` (rne / rna / rtz / rup / rdn / sr),
  `zero_point`, `observer` (minmax / ema / history / percentile / mse), `mma_input` (scaled / dequant)
- MMASpec: `algorithm` (cofda / gdfs / fp32_fma / fp64 / int_exact), `f_bits`, `chunk_size`, `c_mode`
  (fused / decoupled), `g_bits`, `group_size`, `k_tile`, `promote_interval`
- HardwarePreset (코드 `tricast.mma.spec.PRESETS`): MMASpec + `provenance` (출처·검증 상태 필수)
- Recipe: `defaults` / `overrides` 의 `weight`·`activation` (QuantSpec), `mma` (MMASpec), `transform`,
  `weight_algo`, `sparsity` (`kind`: none / n:m / unstructured, `n`·`m`·`ratio`), `outliers` (`fraction`,
  `format`) — 온톨로지 `WeightStructure`; `include` / `exclude`; override 선택자 `match` (이름 패턴) · `layers`
  (decoder 블록 번호 **문자열**, 예 `"0,-1"` · `"0-7"`, `-1` = 마지막) · `modules` (leaf 이름) 와 `skip` (그 레이어는
  양자화하지 않음) — 온톨로지 `LayerRule`; `kv` (`preset`, `mode`, `layers` — 절대 번호 문자열 `"0-7"`, `-1` 은
  쓰지 않음) — 온톨로지 `KVSpec`
- EvalRun: `metrics` + `env` (git SHA, 모델 revision, 데이터셋 fingerprint, 레시피 해시). 리포트의 `mma_ulp` =
  같은 양자화 피연산자를 fp64 로 누산한 결과와의 거리 (출력 형식의 ULP 단위)
- EmulationRequest: 자연어 요청의 구조화 결과, `assumptions` (기본값으로 채운 항목)
- Reference (코드 `tricast.reference`, 비교 기준 `native` / `same_quant_fp64`), ErrorReport (코드
  `tricast.analysis.layer_report`: `mma_ulp`, 레이어별 오차)
- 코드 객체가 없는 도메인 어휘: `Engineer`, `Chip`, `Symptom`, `RootCause` (`category`: accumulator_width /
  rounding / special_values / scale_granularity / …) — 정의는 `docs/ontology.yaml`

신뢰 수준: 자연어 요청·`EmulationRequest`·LLM 응답·RAG 문서는 외부 입력 (스키마 검증 전에는 값으로 쓰지 않는다).
HardwarePreset 수치는 `provenance` 에 적힌 출처까지만 믿는다. EvalRun 수치는 `env` 가 함께 있을 때만 근거가 된다.

혼동 주의: `f_bits` (F, 누산 데이터패스) ≠ `g_bits` (G, GDFS 그룹 내부). `QuantSpec.rounding` 은 원소,
`ScaleSpec.rounding` 은 스케일. microxcaling 의 `"nearest"` = `rna`, `"floor"` = `rtz`
(`tricast.rounding.from_microxcaling`). granularity `channel`·`token` 은 `row` 의 동의어.

## 3. 절대 규칙 (위반한 결과물은 수용하지 않는다)

1. Triton 결과는 레퍼런스와, 레퍼런스는 독립 구현(NADPE) 골든 벡터와 비트 단위로 같아야 한다 — 차이를 허용 오차로
   덮지 않는다. (↔ AC1)
2. 출처 없는 하드웨어 프리셋·파라미터를 만들지 않는다 — 모르면 "미검증"으로 표시한다. (↔ AC2)
3. 평가 수치는 환경 캡처와 함께만 보고한다. (↔ AC3)
4. 요청에 없는 정밀도·누산 파라미터를 지어내지 않는다 — 기본값은 `assumptions` 에 적는다 (에이전트 파싱은
   `EmulationRequest.assumptions`, 손으로 쓰는 레시피 파일은 주석). (↔ AC6)

## 4. 금지 사항 (위임 작업에 항상 적용)

1. **테스트와 골든 데이터를 고치지 않는다.** `tests/`, `tests/harness/golden_cases.yaml`, `tests/data/`, `app/tests/`,
   `app/web/demo/webgpu/golden.json` 의 변경은 사람이 승인한다. 실패하면 구현을 고친다. 사람이 지시한 변경과 포매팅은 예외지만, 판정에 쓰이는 것 (단언문,
   케이스, 기대값, skip 조건) 은 건드리지 않는다.
2. **완료 조건을 좁히지 않는다.** skip·xfail·케이스 삭제·허용 오차 추가로 통과시키지 않는다.
3. **근거 없는 결과를 내놓지 않는다.** 통과했다면 각 케이스가 왜 통과하는지 한 줄씩 설명한다.
4. 공유 명세 (`src/tricast/formats.py`, `rounding.py`, `quant/spec.py`, `mma/spec.py`) 와 수치 의미론의 정본
   `src/tricast/reference/` 변경은 사람이 승인한다.
5. **멈추고 사람에게 보고한다** — 같은 실패가 3회 반복될 때, 테스트·골든 데이터·공유 명세를 바꿔야만 통과할 것
   같을 때, SPEC·AGENTS·골든 케이스가 서로 충돌하거나 정해지지 않은 판단 (동률 규칙, 기본값, 허용 범위) 이 필요할 때.
   보고에는 시도한 것, 실패 출력, 막힌 결정을 적는다.

## 5. 코딩 컨벤션

- 타입 힌트 필수. 명세는 frozen dataclass 로, 경계에서 검증한다.
- 레퍼런스(`tricast.reference`)는 정확 연산만 쓴다 (fp64 / int64 / Fraction). 커널은 레퍼런스와 비트 일치.
- Triton: 서브노멀이 닿을 수 있는 fp32 나눗셈·곱셈·덧셈은 `tricast.kernels.ieee` 의 IEEE PTX 헬퍼로 (Triton 의
  libdevice 는 flush-to-zero 로 링크된다), 가변 시프트는 폭 가드, dtype 왕복으로 반올림을 흉내 내지 않는다.
- 결정론: seed 고정, 순서를 내놓는 함수는 동률 규칙을 명시한다. 같은 입력에 결과가 흔들리면 완료가 아니다.
- 새 기능은 레퍼런스 + 테스트부터, 그다음 커널.

## 6. 완료의 정의

`pytest -q` 통과 (CPU) + 커널·GPU 경로를 바꿨다면 `pytest tests/gpu -q` 통과 (A100 · H200 · V100 에서 확인) + `ruff check .` 무경고
+ 변경을 근거(테스트·측정값)로 설명할 수 있음.

## 7. 운영 정보

개발 환경
- Python ≥ 3.10, torch ≥ 2.4. Triton 커널은 Linux + CUDA GPU 에서만 (A100 · H200 · V100 에서 레퍼런스와 비트 일치
  확인) — 그 외에는 레퍼런스 백엔드로 동작.
- 설치: `pip install -e ".[triton,eval,dev]"`. microxcaling 오라클 테스트는 클론을 `PYTHONPATH` 에 추가.

자주 쓰는 명령
- 테스트: `pytest -q` / GPU: `pytest tests/gpu -q` / 린트: `ruff check .` — 한 번에: `make check` (린트 + CPU 테스트)
- 골든 케이스: `pytest tests/test_golden.py tests/test_compare_golden.py -q -rs` (CPU 에서 GPU 전용 1건만 skip 이 정상) ·
  웹 앱: `make test-app`
  (`pip install -e ".[app]"`)
- 둘러보기: `tricast formats`, `tricast schemes`, `tricast presets`, `tricast recipe-check <이름|경로>`
- 평가: `tricast ppl --model Qwen/Qwen3-0.6B --recipe hopper_fp8_w8a8`,
  `tricast eval --model … --recipe … --tasks hellaswag,coqa`, `tricast run configs/sweeps/<sweep>.yaml`,
  lm-eval CLI 그대로: `python -m tricast.eval.lmeval --model tricast --model_args pretrained=…,recipe=… --tasks …`
- 보정이 필요한 레시피 (GPTQ·AWQ·SmoothQuant·정적 observer): `--calib-dataset/--calib-samples/--calib-seqlen/--calib-seed`
- 오차 분석: `tricast report --model … --recipe …` (레이어별 MSE·SQNR·코사인, 모델 logits KL)
- 데모: `python examples/demo_qwen3.py --quick`

디렉터리
- `src/tricast/` — 명세(`formats`, `rounding`, `quant/spec`, `mma/spec`), 레퍼런스(`reference/`), 커널
  (`kernels/`), API(`quant/api`, `mma/api`), 통합(`recipe`, `nn/`, `calibrate`, `eval/`, `cli`)
- `src/tricast/recipes/` — 이름으로 부르는 기본 레시피 (패키지에 포함, `tricast.recipe.list_recipes()`),
  `configs/sweeps/` — 스윕
- `tests/` (CPU), `tests/gpu/` (Triton), `tests/data/nadpe/` (독립 구현 골든 벡터), `tests/harness/` (골든 케이스)
- `docs/` — `PROBLEM.md` (왜), `SPEC.md` (무엇 · AC), `ontology.yaml` (도메인 구조, 분석은 `ontology.md`),
  `research/interviews.md` (인터뷰 로그), `prompts/` (위임·검증 기록)

진행 상태
- 구현 진행 상태는 여기에 적지 않는다 — `pytest` 결과와 `support_matrix.yaml` 이 원천이다.
