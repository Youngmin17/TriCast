# AGENTS.md — TriCast

> 코딩 에이전트가 **작업 중 상시 준수할 규칙**. 도구와 무관하게 이 파일 한 벌이 원천이다 (Claude Code 는
> `CLAUDE.md` 의 `@AGENTS.md` 로 읽는다). 무엇을 만드는가의 정본은 `docs/SPEC.md`, 도메인 구조는
> `docs/ontology.yaml`, 수치 의미론은 `docs/design/ENGINE.md` — 여기서는 복제하지 않고 가리킨다.
> ✍️ 절대 규칙과 완료의 정의는 팀 확정 전 초안이다.

## 1. 제품 맥락

하드웨어 설계자가 수 형식·양자화·MMA 누산 알고리즘을 레시피로 정의하면, TriCast 가 그 산술을 CUDA core 에서
**비트 단위로 정확하게** 에뮬레이트해 Hugging Face 모델 품질(PPL·lm-eval)을 근거와 함께 돌려준다. 핵심 가치는
"믿을 수 있는 산술 + 빠른 LLM 수준 평가". 이 저장소는 AI캡스톤디자인 과제 저장소이기도 하다.

## 2. 도메인 용어집 (어휘 발췌 — 전체 구조는 `docs/ontology.yaml`)

- Format: `name`, `kind` (float / int / pow2), `max_normal`, `special` (ieee / fn / fnuz / none)
- QuantSpec: `format`, `granularity` (tensor / row / group / block), `group_size`, `scale.method`
  (absmax / pow2_floor / pow2_ceil / mse / percentile), `rounding` (rne / rna / rtz / rup / rdn / sr),
  `zero_point`, `observer` (minmax / ema / history / percentile / mse), `mma_input` (scaled / dequant)
- MMASpec: `algorithm` (cofda / gdfs / fp32_fma / fp64 / int_exact), `f_bits`, `chunk_size`, `c_mode`
  (fused / decoupled), `g_bits`, `group_size`, `k_tile`, `promote_interval`
- Preset: MMASpec + `provenance` (출처·검증 상태 필수)
- Recipe: `defaults` / `overrides` 의 `weight`·`activation` (QuantSpec), `mma` (MMASpec), `transform`,
  `weight_algo`; `include` / `exclude`; override 선택자 `match` (이름 패턴) · `layers` (decoder 블록 번호,
  `-1` = 마지막) · `modules` (leaf 이름) 와 `skip` (그 레이어는 양자화하지 않음); `kv` (`preset`, `mode`, `layers`)
- EvalRun: `metrics` + `env` (git SHA, 모델 revision, 데이터셋 fingerprint, 레시피 해시)
- EmulationRequest: 자연어 요청의 구조화 결과, `assumptions` (기본값으로 채운 항목)

신뢰 수준: 자연어 요청·`EmulationRequest`·LLM 응답·RAG 문서는 외부 입력 (스키마 검증 전에는 값으로 쓰지 않는다).
Preset 수치는 `provenance` 에 적힌 출처까지만 믿는다. EvalRun 수치는 `env` 가 함께 있을 때만 근거가 된다.

혼동 주의: `f_bits` (F, 누산 데이터패스) ≠ `g_bits` (G, GDFS 그룹 내부). `QuantSpec.rounding` 은 원소,
`ScaleSpec.rounding` 은 스케일. microxcaling 의 `"nearest"` = `rna`, `"floor"` = `rtz`
(`tricast.rounding.from_microxcaling`). granularity `channel`·`token` 은 `row` 의 동의어.

## 3. 절대 규칙 ✍️ (위반한 결과물은 수용하지 않는다)

1. Triton 결과는 레퍼런스와 비트 단위로 같아야 한다 — 차이를 허용 오차로 덮지 않는다. (↔ AC1, AC7)
2. 출처 없는 하드웨어 프리셋·파라미터를 만들지 않는다 — 모르면 "미검증"으로 표시한다. (↔ AC2)
3. 평가 수치는 환경 캡처와 함께만 보고한다. (↔ AC3)
4. 요청에 없는 정밀도·누산 파라미터를 지어내지 않는다 — 기본값은 `assumptions` 에 적는다 (에이전트 파싱은
   `EmulationRequest.assumptions`, 손으로 쓰는 레시피 파일은 주석). (↔ AC6)

## 4. 금지 사항 (위임 작업에 항상 적용)

1. **테스트와 골든 데이터를 고치지 않는다.** `tests/`, `tests/harness/golden_cases.yaml`, `tests/data/` 의 변경은
   사람이 승인한다. 실패하면 구현을 고친다. 사람이 지시한 변경과 포매팅은 예외지만, 판정에 쓰이는 것 (단언문,
   케이스, 기대값, skip 조건) 은 건드리지 않는다.
2. **완료 조건을 좁히지 않는다.** skip·xfail·케이스 삭제·허용 오차 추가로 통과시키지 않는다.
3. **근거 없는 결과를 내놓지 않는다.** 통과했다면 각 케이스가 왜 통과하는지 한 줄씩 설명한다.
4. 공유 명세 (`src/tricast/formats.py`, `rounding.py`, `quant/spec.py`, `mma/spec.py`, `docs/design/ENGINE.md`)
   변경은 사람이 승인한다.

## 5. 코딩 컨벤션

- 타입 힌트 필수. 명세는 frozen dataclass 로, 경계에서 검증한다.
- 레퍼런스(`tricast.reference`)는 정확 연산만 쓴다 (fp64 / int64 / Fraction). 커널은 레퍼런스와 비트 일치.
- Triton: 서브노멀이 닿을 수 있는 fp32 나눗셈·곱셈·덧셈은 `tricast.kernels.ieee` 의 IEEE PTX 헬퍼로 (Triton 의
  libdevice 는 flush-to-zero 로 링크된다), 가변 시프트는 폭 가드, dtype 왕복으로 반올림을 흉내 내지 않는다.
- 결정론: seed 고정, 순서를 내놓는 함수는 동률 규칙을 명시한다. 같은 입력에 결과가 흔들리면 완료가 아니다.
- 새 기능은 레퍼런스 + 테스트부터, 그다음 커널.

## 6. 완료의 정의 ✍️

`pytest -q` 통과 (CPU) + 커널·GPU 경로를 바꿨다면 `pytest tests/gpu -q` 통과 (A100/H100) + `ruff check .` 무경고
+ 변경을 근거(테스트·측정값)로 설명할 수 있음.

## 7. 운영 정보

개발 환경
- Python ≥ 3.10, torch ≥ 2.4. Triton 커널은 Linux + CUDA (sm_80 이상) 에서만 — 그 외에는 레퍼런스 백엔드로 동작.
- 설치: `pip install -e ".[triton,eval,dev]"`. microxcaling 오라클 테스트는 클론을 `PYTHONPATH` 에 추가.

자주 쓰는 명령
- 테스트: `pytest -q` / GPU: `pytest tests/gpu -q` / 린트: `ruff check .`
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
- `docs/` — SPEC, PROBLEM, ontology, 인터뷰, 스파이크, 설계(`design/ENGINE.md`), 데모(`demo/DEMO.md`)

진행 상태
- 구현 진행 상태는 여기에 적지 않는다 — `pytest` 결과와 `support_matrix.yaml` 이 원천이다.
