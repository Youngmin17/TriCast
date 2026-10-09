# 위임 프롬프트와 검증 루프 기록 (강의 5)

> Wave 1 (2026-09-30) 은 지휘 쪽이 먼저 고정한 계약(당시 계약서와 공유 명세 파일)을 기준으로 7개 레인을 하위
> 에이전트에게 위임했다. 원본 프롬프트: [`wave1/`](wave1/) (공통 규칙 [`00_common.md`](wave1/00_common.md) + 레인별 `lane_*.md`).
>
> 수치와 사건은 2026-10-01 기록(squash 이전 이력의 같은 경로 파일) 그대로다. 회차 번호와 원인 분류는 이 정리에서
> 강의 5 형식에 맞춰 붙였다. §3 의 "이후 판정" · "현재 판정" 행과 §4 사후 확인 표는 squash 이전 이력의 커밋 메시지 ·
> 문서, 현재 저장소의 README · CHANGELOG 에서 찾은 근거로 덧붙였다. 참조는 현재 저장소 기준으로 바꿨다: AC 번호는 [`docs/SPEC.md`](../SPEC.md) §6 기준이고
> (원본의 "NADPE 독립 구현과 비트 일치" = AC7 → 현재 AC1), 원본의 계약서 대신 수치 의미론의 현재 정본인
> [`src/tricast/reference/`](../../src/tricast/reference/) 를 가리킨다. 커밋 해시 (`64c10e3`, `82db449` 등) 는 squash 이전 이력의 것이고,
> 시각은 UTC 다.

| 절 | 강의 5 지침 |
|---|---|
| [역할](#역할) | — |
| [§1 위임 대상 선정](#1-위임-대상-선정--wave-1-2026-09-30) | A |
| [§2 프롬프트 원본](#2-프롬프트-원본) | D |
| [§3 검증 루프 기록](#3-검증-루프-기록) | E |
| [§4 수용 판단](#4-수용-판단) | F |
| [§5 테스트 하니스 결함 주입 검사](#5-테스트-하니스-결함-주입-검사) | C |
| [§6 에이전트 하니스](#6-에이전트-하니스) | G |
| [§7 용어집 작동 확인](#7-용어집-작동-확인-강의-4-지침--2026-09-30) | 강의 4 |

## 역할

| 역할 | 누구 | 한 일 |
|---|---|---|
| 지휘 | 팀원 Youngmin17 + 메인 Claude Code 세션 | 레인 분할, 계약 (공유 명세) 초안, 판정 실행 (테스트 재실행), GPU 실행 · 수정 |
| 위임 | 레인 하위 에이전트 (Codex) | 레인별 소유 파일의 구현과 테스트 |
| 기준 추인 | 팀 (2026-10-01) | 골든 케이스, AGENTS.md 절대 규칙과 완료의 정의, 스파이크 성공 조건 등 판정 기준 추인 — 커밋 `82db449` "docs: record team approval of course work" (2026-10-01 08:47 UTC), [`CHANGELOG.md`](../../CHANGELOG.md) |

프롬프트 원본의 "Claude" 는 메인 세션, "Codex" 는 하위 에이전트를 가리킨다. 본문의 "지휘 쪽" 은 첫 행이다.

## 1. 위임 대상 선정 — Wave 1 (2026-09-30)

| 레인 | 위임한 것 | 담당 | 판정 기준 (위임 전에 정함) |
|---|---|---|---|
| Q | 양자화 레퍼런스 · QTensor · API · observer | 하위 에이전트 | 레퍼런스 테스트 + microxcaling 오라클과 비트 일치 |
| M | MMA 누산 레퍼런스 (FDA/CoFDA/GDFS …) | 하위 에이전트 | 손으로 유도한 NADPE 사례 · 불변식 테스트 |
| KQ | Triton 양자화 커널 | 하위 에이전트 | GPU 에서 레퍼런스와 비트 일치 (`tests/gpu`) |
| KM | Triton MMA 커널 | 하위 에이전트 | GPU 에서 레퍼런스와 비트 일치 (`tests/gpu`) |
| A | 변환 (Hadamard · SmoothQuant · AWQ) · GPTQ | 하위 에이전트 | 불변식 · GPTQ 오차 < RTN 오차 |
| I | Recipe · HF 패치 · calibrate · PPL · lm-eval · CLI | 하위 에이전트 | tiny 모델 CPU 테스트 · 레시피 스키마 검증 |
| D | Qwen3-0.6B 데모 | 하위 에이전트 | 문법·`--help` (실행은 지휘 쪽이 GPU 에서 확인) |
| — | 형식·반올림·명세·레퍼런스 캐스트·계약서 | 지휘 | torch 네이티브 캐스트와 비트 일치 |

### 위임한 것과 이유

- **위임한 것**: 7개 레인 (Q · M · KQ · KM · A · I · D).
- **이유**: 레인마다 판정 기준을 위임 전에 실행 가능한 형태로 정할 수 있었다 (위 표 마지막 열). 출력이 레퍼런스 ·
  오라클 · 손으로 유도한 사례와 비트 단위로 비교되거나, 불변식 · 스키마 검증으로 판정된다.
- **범위**: 레인마다 **소유 파일**을 지정해 변경 범위를 파일 단위로 한정했다 (§2 표의 대상 파일).

### 지휘 쪽이 직접 담당한 것과 이유

| 직접 담당한 것 | 이유 (원본 근거) |
|---|---|
| 명세 계층 (형식 · 반올림 · QuantSpec · MMASpec) · 레퍼런스 캐스트 · 계약서. 위임 전에 먼저 고정하고 커밋했다 (`64c10e3`, 2026-09-30 09:08 UTC) | 모든 레인이 같은 어휘와 시그니처로 병렬 작업하려면 경계가 먼저 있어야 한다 |
| 공유 명세 파일의 수정. 하위 에이전트에게는 금지하고, 명세에 문제가 있으면 고치지 말고 `CONCERNS` 로 보고하게 했다 | 기준을 에이전트가 바꾸면 기준으로 에이전트를 판정할 수 없다 |
| 판정. 지휘 쪽에서 테스트를 다시 실행해 하위 에이전트의 보고와 대조했다 | 모든 레인에 같은 완료 형식 (`CODEX_DONE / FILES / TESTS / CONCERNS`) 을 요구하되, 그 보고를 그대로 믿지 않기로 했다 |
| KQ · KM 커널의 GPU 실행 · 검증 · 수정, D 데모의 GPU 실행 | 위임 환경은 로컬 macOS 로 triton/CUDA 가 없고, 하위 에이전트에게 SSH · 원격 GPU 사용을 금지했다 ([`00_common.md`](wave1/00_common.md), [`lane_KQ.md`](wave1/lane_KQ.md), [`lane_KM.md`](wave1/lane_KM.md), [`lane_D.md`](wave1/lane_D.md)) |

## 2. 프롬프트 원본

각 레인에는 공통 규칙 `00_common.md` 와 레인 파일을 함께 붙여넣었다. 아래 파일은 실제로 붙여넣은 텍스트 그대로이며,
맨 위에 날짜 주석 한 줄만 더했다. 프롬프트 안의 경로 · 번호는 당시 기준이다. 원본이 "먼저 정독" 첫 줄로 지정한
계약서의 역할은 현재 [`src/tricast/reference/`](../../src/tricast/reference/) 가 맡는다.

| 레인 | 원본 | 대상 파일 (소유 파일, 경로는 `src/tricast/` 기준) | 목표 | 완료 기준 (원문 요약) |
|---|---|---|---|---|
| 공통 | [`00_common.md`](wave1/00_common.md) | 레인별 소유 파일만 생성 · 수정. 공유 명세 파일 (`formats.py`, `rounding.py`, `quant/spec.py`, `mma/spec.py`, `mma/operand.py`, `reference/cast.py`, 계약서, `tests/conftest.py`, `pyproject.toml`, `__init__.py`) 은 수정 금지 | 제품 맥락, 먼저 읽을 계약 문서와 참고 코드 (NADPE MMA-Emu CUDA, microxcaling), 로컬 실행 환경 (macOS, CUDA 없음), 스타일, bit-exact 규약 (레퍼런스는 정확 연산, 테스트 비교는 `==`) | 마지막에 `CODEX_DONE` / `FILES` / `TESTS` (pytest 최종 요약 줄 그대로) / `CONCERNS` 4줄 출력. 커밋 · 푸시 · 브랜치 생성 · SSH · 원격 GPU 금지 |
| Q | [`lane_Q.md`](wave1/lane_Q.md) | `reference/quantize.py`, `quant/qtensor.py`, `quant/api.py`, `quant/observer.py` + `tests/test_cast_reference.py`, `test_quantize_reference.py`, `test_observer.py`, `test_microxcaling_parity.py` | 양자화 레퍼런스 (scale 계산 · 원소 반올림), QTensor, `quantize` · `fake_quant` (clipped STE), observer | 4개 테스트 파일 전부 PASS (microxcaling 오라클 포함), ruff 경고 0 |
| M | [`lane_M.md`](wave1/lane_M.md) | `reference/mma.py`, `mma/api.py` + `tests/test_mma_reference.py`, `test_mma_invariants.py` | MMA 누산 레퍼런스 (FDA · CoFDA · GDFS · fp32_fma · fp64 · int_exact) 와 `gemm` API. NADPE CUDA 코드와 줄 단위 대조 | 두 테스트 파일 PASS, ruff 0 |
| KQ | [`lane_KQ.md`](wave1/lane_KQ.md) | `kernels/__init__.py`, `kernels/cast_core.py`, `kernels/quantize.py` + `tests/gpu/test_triton_quantize.py` | Triton 양자화 커널. 레퍼런스와 bit-exact | 로컬: 3개 파일 `py_compile` 통과, ruff 0, CPU 에서 `import tricast.kernels` 가 ImportError. GPU 실행 · 수정은 지휘 쪽이 이어서 함 |
| KM | [`lane_KM.md`](wave1/lane_KM.md) | `kernels/mma_core.py`, `kernels/mma.py` + `tests/gpu/test_triton_mma.py` | Triton MMA 커널 (cofda · gdfs · fp32_fma · fp64 · int_exact). `gemm_reference` 와 bit-exact | 로컬: 3개 파일 `py_compile` 통과, ruff 0. GPU 검증 · 수정은 지휘 쪽 |
| A | [`lane_A.md`](wave1/lane_A.md) | `transforms.py`, `weight_quant/__init__.py`, `weight_quant/gptq.py` + `tests/test_transforms.py`, `test_gptq.py` | 변환 (Hadamard · RHT · SmoothQuant · AWQ) 과 가중치 알고리즘 (RTN · GPTQ) | 두 테스트 PASS (의존 모듈 존재 시), ruff 0 |
| I | [`lane_I.md`](wave1/lane_I.md) | `recipe.py`, `schemas/recipe.schema.json`, `nn/`, `calibrate.py` (현재 [`calibration.py`](../../src/tricast/calibration.py)), `eval/` (ppl · lmeval · runner · envinfo), `cli.py`, `configs/recipes/*.yaml` (현재 [`src/tricast/recipes/`](../../src/tricast/recipes/)), `configs/sweeps/*.yaml` + 테스트 4개 | Recipe · EmuLinear · patch · calibrate · PPL · lm-eval 어댑터 · runner · CLI · 기본 레시피 | 테스트 PASS (의존 모듈 존재 시), ruff 0 |
| D | [`lane_D.md`](wave1/lane_D.md) | `examples/demo_qwen3.py`, `examples/README.md`, `docs/demo/DEMO.md` (현재 저장소에는 앞의 두 파일만 있음) | Qwen3-0.6B 발표용 데모: 형식 비교 · 누산 알고리즘 비교 · 레시피별 PPL · 생성 비교. 수치는 "실측 후 채움" 으로 두고 지어내지 않음 | `py_compile`, ruff 0, `--help` 동작. 실제 실행은 지휘 쪽이 GPU 에서 |

레인 파일은 모두 같은 순서로 되어 있다: 소유 파일 → 구현 (계약서 절 번호와 함수 시그니처 지정) → 테스트 → 완료 기준.
변경 범위 · 보고 형식 · 실행 명령 · bit-exact 규약은 공통 파일에 한 번만 적었다.

### 강의 5 D절 요소 대응

| D절 요소 | Wave 1 원본 위치 | 원본에 없을 때 현재 적용되는 규칙 |
|---|---|---|
| 대상 (파일 · 함수) | `lane_*.md` "소유 파일", 구현 절의 함수 시그니처 | — |
| 목표 | `lane_*.md` 머리 줄과 구현 절, `00_common.md` 제품 맥락 | — |
| 절대 규칙 | `00_common.md` bit-exact 계약 (레퍼런스는 정확 연산, 테스트 비교는 `==`) | [`AGENTS.md`](../../AGENTS.md) §3 도 함께 적용 |
| 판단 규칙과 참조 위치 | `00_common.md` "먼저 정독" (계약서 · 공유 명세 파일), `lane_*.md` 의 계약서 절 번호 | — |
| 완료 기준 — 실행 명령 + 통과 조건 | `00_common.md` 테스트 · 린트 명령, `lane_*.md` "완료 기준" (테스트 PASS, ruff 0) | AGENTS.md §6 완료의 정의 |
| 완료 기준 — skip · xfail 0 | `00_common.md` 는 의존 레인 모듈이 없을 때 `pytest.importorskip` skip 을 허용하고, 작업 끝무렵 모듈이 생겼으면 실제로 돌려 통과시키게 했다 | AGENTS.md §4.2 (skip · xfail · 케이스 삭제 · 허용 오차로 통과시키지 않음) |
| 변경 범위 | `lane_*.md` 소유 파일, `00_common.md` 공유 명세 수정 금지 | — |
| 중단 조건 | `00_common.md`: 명세 문제는 고치지 말고 `CONCERNS` 로 보고 | AGENTS.md §4.5 (같은 실패 3회, 판정 파일을 바꿔야만 통과할 것 같을 때, 충돌 · 미정 판단) |
| 통과 이유 설명 요구 | — (완료 보고는 `TESTS:` 에 pytest 요약 줄) | AGENTS.md §4.3 (케이스마다 통과 이유 한 줄) |

### 보완한 프롬프트 예 — 실행 기록과 구분, 다음 위임부터 사용

아래는 `lane_Q.md` 의 구조에 중단 조건, skip · xfail 0 완료 기준, 케이스별 통과 이유 요구를 더한 예다. 이 블록으로
실행한 기록은 아직 없다. §3 의 회차 기록은 모두 위 원본 프롬프트로 실행한 결과다.

```text
[Lane Q — 양자화 레퍼런스 + QTensor + API + Observer]  (보완 예)
대상 (이것만 생성/수정):
  src/tricast/reference/quantize.py, src/tricast/quant/qtensor.py, src/tricast/quant/api.py,
  src/tricast/quant/observer.py,
  tests/test_cast_reference.py, tests/test_quantize_reference.py, tests/test_observer.py,
  tests/test_microxcaling_parity.py
- 목표: 양자화 레퍼런스 (scale 계산 · 원소 반올림), QTensor, quantize / fake_quant (clipped STE), observer.
  구현 · 테스트 세부는 wave1/lane_Q.md 의 "구현" 1)~4) 와 "테스트" 절을 그대로 따른다.
- 절대 규칙 (AGENTS.md §3-1): 레퍼런스는 정확 연산 (fp64 / int64 / Fraction) 만 쓴다. 테스트 비교는 == (NaN==NaN).
  차이를 허용 오차로 덮지 않는다.
- 판단 규칙과 참조 위치: 수치 의미론은 src/tricast/reference/ 와 공유 명세 (formats.py, rounding.py, quant/spec.py).
  quantize_elements(x2d, spec, scale_pe, zp_pe=None, global_scale=None, noise=None) 시그니처는 고정 (GPTQ 레인이 사용).
- 완료 기준:
  PYTHONPATH=src:<microxcaling 클론> python -m pytest tests/test_cast_reference.py tests/test_quantize_reference.py \
    tests/test_observer.py tests/test_microxcaling_parity.py -q -rs
  → failed 0, skipped 0, xfailed 0 (microxcaling 오라클 포함), ruff check 경고 0.
- 변경 범위: 위 대상 파일만. 공유 명세 · src/tricast/reference/cast.py · 다른 레인 파일 · 다른 테스트는 수정하지 않는다
  (AGENTS.md §4.1 · §4.4). 문제를 찾으면 CONCERNS 에 적는다.
- 중단 조건 (AGENTS.md §4.5): 같은 실패가 3회 반복되거나, 공유 명세 · 테스트를 바꿔야만 통과할 것 같거나, 계약과
  명세가 충돌하면 멈추고 시도한 것 · 실패 출력 · 막힌 결정을 보고한다.
끝나면 pytest 요약 줄 그대로와, 테스트 함수마다 무엇을 검사하고 왜 통과하는지 한 줄씩 설명해줘 (AGENTS.md §4.3).
보고 형식: CODEX_DONE / FILES / TESTS / CONCERNS + 통과 이유 목록
```

## 3. 검증 루프 기록

형식: 회차 결과 → 원인 분류 → 보강 → 재검증. 원인 분류는 `[환경]`, `[컨텍스트]` (프롬프트 · 계약서의 지시나 전제),
`[명세 불일치]`, `[지휘 쪽 파일 결함]`, `[구현]` 으로 적는다. 통과 개수는 원본의 pytest 요약 그대로다 (단위: pytest 노드).
회차 번호는 레인별이며, 1회차는 7개 레인을 동시에 투입한 첫 실행이다. 표 끝의 "이후 판정" 행은 해당 커밋 메시지 · 문서의
수치, "현재 판정" 행은 2026-10-08 전체 재실행 결과다.

| 회차 · 레인 | 결과 | 무엇이 문제였나 | 무엇을 고쳤나 (프롬프트/컨텍스트/코드) |
|---|---|---|---|
| 1 · 7개 레인 동시 | 백그라운드 작업 7개가 수 초 안에 모두 종료. 종료 코드만 보면 성공처럼 보였으나 로그는 전부 `EXIT rc=1` — 실행된 레인 0/7 | `[환경]` 로컬 설정의 기본 모델이 계정에서 막혀 있었다: `The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account` | 실행 명령: 최근 세션 기록에서 실제로 쓰인 모델을 찾아 1줄 프로브 (`PROBE_OK`) 로 확인한 뒤 `-m gpt-6-astra` 로 재투입. 프롬프트는 그대로 |
| 2 · Q | `476 passed, 9 failed` | `[지휘 쪽 파일 결함]` 실패 9건 모두 지휘 쪽이 작성한 공유 파일 `reference/cast.py` 의 결함 3종 — unsigned 정수 격자의 −0, E8M0 에 +Inf 입력, `decode()` 의 범위 검사 누락. 하위 에이전트는 소유권 규칙대로 공유 파일을 고치지 않고 `CONCERNS` 로 보고했고, 테스트를 결함에 맞게 느슨하게 바꾸지 않았다 | 코드 (지휘 쪽): [`src/tricast/reference/cast.py`](../../src/tricast/reference/cast.py) 의 결함 3종 수정. 레인 Q 의 테스트는 그대로 |
| 3 · Q (재검증) | 전체 CPU 스위트 `812 passed` | — | — |
| 2 · A | `111 passed`, ruff 0 | `[명세 불일치]` 명세 docstring 은 RHT 를 `H·D`, 계약서는 `D·H` 로 적고 있었다. 하위 에이전트는 계약서를 따랐다고 보고 | 컨텍스트: 명세 docstring 을 계약서에 맞춰 수정 |
| 2 · M | `129 passed`, ruff 0 | `[명세 불일치]` (a) NADPE 원본의 fp32 서브노멀 누산기 정규화가 한 비트 더 시프트한다 (도달 불가능한 경로). 하위 에이전트는 계약서의 정확한 정규화를 따르고 차이를 보고. (b) E8M0 field-0 규칙의 적용 범위가 계약서에 group 레벨로만 적혀 있었다 | 컨텍스트: (b) NADPE 소스 확인 후 계약서에 product 레벨까지 명확히 적음. (a) 는 계약서의 정규화를 따른 구현 그대로 |
| 검증 · 레퍼런스 | 계약서만 보고 만든 레퍼런스를 NADPE 원본 CUDA 커널 (수정 없이 단독 빌드) 과 비교: FP8 1200 + FP4 516 = 1716 케이스 전부 비트 일치 (현재 AC1) | — (오라클 벡터는 결정론 2400회 재실행과 독립 numpy 모델로 먼저 검증) | — |
| 2 · KQ (로컬) | 로컬 skip (GPU 전용) | — | — (GPU 판정으로 넘김) |
| 3 · KQ (GPU 첫 실행) | 579 중 551 통과, 28 실패 | 실패 28건은 두 원인으로 갈렸다. `[컨텍스트]` 프롬프트 [`lane_KQ.md`](wave1/lane_KQ.md) 는 스케일 · 원소 나눗셈에 `tl.math.div_rn` 을 쓰도록 적었는데, Triton 이 libdevice 를 flush-to-zero 로 링크한다 (`set_nvvm_reflect_ftz`) → 스케일이 fp32 서브노멀 (E8M0 2^-127) 이면 0/0 = NaN. 1M 개 나눗셈 프로브로 `tl.math.div_rn` (FTZ) 과 inline PTX `div.rn.f32` (IEEE, 불일치 0) 를 비교해 확정. `[컨텍스트]` 프롬프트 [`lane_Q.md`](wave1/lane_Q.md) 는 "torch fp32 연산은 CPU/CUDA 모두 IEEE RN" 을 전제로 적었는데, PyTorch CUDA 의 `tensor / 스칼라` 는 역수 곱셈이라 정확히 반올림되지 않는다 (CPU 는 정확) → 레퍼런스가 장치마다 다른 값을 내 스케일이 1 ulp 달랐다 (레퍼런스 쪽 문제) | 코드: 스케일 · epilogue 의 fp32 연산을 IEEE PTX 헬퍼 ([`src/tricast/kernels/ieee.py`](../../src/tricast/kernels/ieee.py)) 로 교체. 레퍼런스는 fp64 계산 후 fp32 1회 반올림 (53 ≥ 2·24+2 라 이중 반올림 무해) 으로 바꿔 장치 무관하게. 컨텍스트: 같은 규칙이 현재 [`AGENTS.md`](../../AGENTS.md) §5 (Triton 컨벤션) 에 있다 |
| 4 · KQ (GPU 재실행) | 579/579 비트 일치 | — | — |
| 2 · KM (로컬) | 작성 시점 진행 중 | — | — |
| 3 · KM (GPU 첫 실행) | 컴파일 오류 `'int' object has no attribute 'type'` | `[구현]` constexpr 튜플을 인덱싱한 값이 포장이 풀린 파이썬 튜플로 중첩 함수에 전달됐다. 구성 요소를 단독 커널로 이분 탐색해도 재현되지 않아, Triton 코드 생성기의 `call_JitFunction` 을 감싸 인자 타입을 출력해서 원인을 찾았다 | 코드: `tl.constexpr(...)` 로 재포장 |
| 4 · KM (GPU 재실행) | 149/149 비트 일치 | — | — |
| 5 · KM (D 데모 실행 중) | Qwen3 데모가 47분 동안 CPU 100% · GPU 0%. 정확성 테스트 149개는 모두 통과한 상태 | `[컨텍스트]` 프롬프트 [`lane_KM.md`](wave1/lane_KM.md) 는 autotune key 를 `M, N, K, 모드` 로 적었다. `[구현]` Triton MMA 커널이 M · N · K 를 `tl.constexpr` 로 선언해 행렬 크기마다 재컴파일하고 있었다 (LLM 평가는 배치마다 시퀀스 길이가 다르다) | 코드: 크기를 런타임 인자로, autotune 키는 M 을 2의 거듭제곱 버킷으로, 가중치 K-major 패킹을 한 번만 |
| 2 · I | 작성 시점 진행 중 | — | — |
| 2 · D | 작성 시점 진행 중. 실행 중 KM 의 성능 결함이 드러남 (위 5 · KM) | — | — |
| Wave 3 · Codex 적대적 검토 (테스트가 덮지 않는 경로를 찾게 지시) | CRITICAL 0 / MAJOR 15 / MINOR 1, 모두 file:line 과 재현 입력 포함. 이 시점 테스트는 모두 통과 상태 | `[구현]` 테스트가 덮지 않는 경로. 예: 구조화 출력이 지원하지 않는 `minimum` · `maxItems` 를 보내 실제 API 에서 400, 새 프로세스의 `lm_eval --model tricast` 미등록, 보정 없이 forward 하면 SmoothQuant/GPTQ 가 조용히 생략 | — |
| Wave 3 · Claude 4관점 검토 | 에이전트 4개 모두 실패 | `[환경]` 네트워크 단절 (`ENOTFOUND`) | 코드가 바뀐 Wave 4 뒤에 재실행했다 |
| Wave 3 · 원격 GPU 대기 | 원격 GPU 작업을 기다리던 ssh 가 끊겼는데 성공 (exit 0) 으로 보였다 | `[환경]` 대기 명령 끝의 `; true` 때문에 연결이 끊겨도 exit 0 | 판정 방법: 원격 로그의 완료 마커로 판정해 작업이 살아 있음을 확인 |
| 이후 판정 · Wave 3 예시 3건 | 세 예시에 대응하는 검사가 수용 커밋에 들어 있다: API 로 보내는 스키마에 `minimum` · `maxItems` 등이 없는지 — `tests/test_agent_parser.py::test_sdk_schema_is_projected_but_local_constraints_remain`, `tests/test_agent_loop.py::test_api_tool_schemas_remove_constraints_but_local_validation_keeps_them` (`6b9894e`); 새 프로세스의 lm-eval 등록 — `tests/test_eval.py::test_lmeval_entrypoint_registers_in_fresh_process` (`28cad6f`); 보정 전 forward 거부 — `tests/test_calibrate.py::test_calibration_required_before_forward` (`28cad6f`, 현재 AC9) | — | 코드 + 테스트 (수용 커밋 `6b9894e` · `28cad6f`, 2026-09-30 13:00 UTC) |
| 이후 판정 · I | 수용 커밋 `28cad6f` (2026-09-30 13:00 UTC) "feat: add HF patching, calibration and evaluation". 같은 시각 `3b32b07` 의 `support_matrix.yaml`: CPU 스위트 (`pytest --ignore=tests/gpu`, macOS arm64, torch 2.11) 전체 통과. Linux x86_64 (torch 2.8.0): `b4d7225` 기록 2254 passed · 6 failed → `82db449` 에서 2337 passed (`cac62e4` 기록) | `[환경]` 실패 6건은 네이티브 (에뮬레이션하지 않은) CPU 결과를 비트 단위로 비교하던 테스트로, Linux 의 BLAS · SDPA 가 shape 에 따라 다르게 반올림했다. 레인 I 의 `test_eval` (nvfp4 배치 크기) · `test_nn_patch` (STE) 가 포함된다 | 테스트 (지휘 쪽, `9a2680b` — 팀 추인 커밋 `82db449` 의 직전 커밋): 네이티브 연산 부분에만 허용 오차 (배치 간 perplexity 상대 1e-12, STE 기울기 상대 1e-6, SDPA 단일 청크 캐시 logits 1e-6, [`CHANGELOG.md`](../../CHANGELOG.md)). 에뮬레이션 결과의 비트 비교는 그대로 |
| 이후 판정 · D | 수용 커밋 `28cad6f` (`examples/demo_qwen3.py` 포함). `b4d7225` 의 데모 문서 (`docs/demo/DEMO.md`): `b68addd` (clean) 에서 `--quick` 한 번 실행 — A100-SXM4-80GB, Qwen3-0.6B `c1899de`, bf16, seed 42, Triton 캐시가 준비된 상태로 약 22분 | — | — |
| 이후 판정 · KM 성능 | 스파이크 기록 (`8dce76f`, A100, 2026-09-30): 튜닝 전 커널 (`3b32b07` 에 들어간 그대로) 로 `hopper_fp8_w8a8` 2048 토큰 창 하나 83.4 s, GEMM 처리량 0.011 TMAC/s (NADPE 0.15 의 7%). `bb44ccd` (2026-09-30 14:53 UTC) 이후: 0.15 TMAC/s, 창 하나 83 s → 6 s, WikiText-2 전체 14.8 분 (PPL 21.2035), GPU 스위트 824 passed (A100 · V100), NADPE 1716 케이스 비트 일치 | `[구현]` 산술량이 아니라 레지스터 스필 (모든 타일 설정에서 스필 1000~6000) | 코드 (`bb44ccd`): 유한 입력 전용 컴파일, 런타임 루프, int32 디코드 |
| 현재 판정 (2026-10-08, geneva A100 · x86_64) | lint 통과 · CPU 2,242 passed · 2 skipped (GPU 숨김) · GPU 스위트 + 골든 945 passed · 1 skipped · 앱 161 passed ([`README.md`](../../README.md) 검증 상태) | — | 레인 I · D 산출물의 현재 판정 파일은 아래 표 |

현재 판정에서 레인 I · D 산출물을 판정하는 파일 (현재 `tests/` 를 [`lane_I.md`](wave1/lane_I.md) · [`lane_D.md`](wave1/lane_D.md) 의 대상 파일과 대조):

| 레인 | 산출물 | 판정 파일 |
|---|---|---|
| I | 레시피 · 스키마 | [`tests/test_recipe.py`](../../tests/test_recipe.py) |
| I | 모델 패치 (`nn/`) | [`tests/test_nn_patch.py`](../../tests/test_nn_patch.py) |
| I | 평가 (`eval/` — PPL · lm-eval 어댑터 · runner · envinfo) | [`tests/test_eval.py`](../../tests/test_eval.py) |
| I | CLI (`cli.py`) | [`tests/test_cli.py`](../../tests/test_cli.py) |
| I | 보정 (`calibrate.py` → 현재 `calibration.py`) | [`tests/test_calibrate.py`](../../tests/test_calibrate.py) (레인 I 의 테스트 4개 밖, `28cad6f` 부터 있음) |
| D | `examples/demo_qwen3.py` | pytest 대상 밖 (`testpaths = ["tests"]`). 판정은 `ruff check .` (lint) 과 [`examples/README.md`](../../examples/README.md) 의 검사 명령 (`py_compile`, `ruff check`, `--help`) — 레인 D 완료 기준과 같은 세 가지 |

### 원본에 남긴 판정 규칙

- 완료는 종료 코드가 아니라 로그의 완료 마커로 판정한다. 대량 위임 전에 1줄 프로브를 한다. (1회차, Wave 3 원격 대기)
- 성능 결함은 정확성 테스트가 잡지 못한다. 정확성 테스트 149개가 모두 통과한 상태에서 재컴파일 결함이 데모 실행으로
  드러났다. (5 · KM)
- 적대적 검토는 테스트가 덮지 않는 경로를 찾게 지시한다. 테스트가 모두 통과한 상태에서 MAJOR 15건이 나왔다. (Wave 3)

### 새로 드러난 미명세 · 불일치 항목과 반영 위치

| 항목 | 드러난 곳 | 반영 위치 |
|---|---|---|
| RHT 곱 순서 (`H·D` vs `D·H`) | A | 명세 docstring 을 계약서 (`D·H`) 에 맞춤 |
| E8M0 field-0 (2^-127) 규칙의 적용 범위 | M | 계약서에 product 레벨까지 명시 (NADPE 소스 확인 후) |
| NADPE 서브노멀 누산기 정규화의 1비트 차이 (도달 불가능한 경로) | M | 계약서의 정확한 정규화를 따름, 차이는 보고로 남김 |
| CUDA 에서 `tensor / 스칼라` 의 반올림 | KQ (GPU 단계) | 레퍼런스를 fp64 계산 후 fp32 1회 반올림으로 |
| Triton libdevice 의 flush-to-zero | KQ (GPU 단계) | IEEE PTX 헬퍼 [`kernels/ieee.py`](../../src/tricast/kernels/ieee.py), 현재 [`AGENTS.md`](../../AGENTS.md) §5 |
| 레이어 건너뛰기 선택자 · `assumptions` 위치 | 용어집 확인 (§7) | [`AGENTS.md`](../../AGENTS.md) §2 Recipe 줄, §3 절대 규칙 4 |

### 원본 작성 시점 레인별 판정 (2026-09-30 스냅샷)

원본 문서 끝 표의 수치를 옮긴다. KQ · KM 의 GPU 결과는 위 회차 표 (3 · 4회차) 에 있다.

| 레인 | 로컬 판정 (지휘 쪽 재실행) | 비고 |
|---|---|---|
| M | 129 passed, ruff 0 | |
| A | 111 passed, ruff 0 | |
| Q | 476 passed / 9 failed → 공유 파일 수정 후 통과 | microxcaling 오라클 포함 |
| KQ | 로컬 skip (GPU 전용) | GPU 판정: 진행 중 |
| KM, I, D | 진행 중 | |

## 4. 수용 판단

| 대상 | 최종 판정 | 확인한 근거 |
|---|---|---|
| Q | 공유 파일 수정 후 전체 CPU 스위트 `812 passed` | microxcaling 오라클 포함. 실패 9건은 공유 파일 결함으로 분류됐고, 레인 Q 의 테스트는 바뀌지 않았다 |
| M | `129 passed`, ruff 0 | 계약서와 NADPE 원본의 차이 보고를 처리 (§3 2 · M) |
| A | `111 passed`, ruff 0 | 명세 docstring 불일치 보고를 처리 (§3 2 · A) |
| KQ | GPU 579/579 비트 일치 | 실패 28건의 원인 2종을 커널 · 레퍼런스에서 각각 고친 뒤 |
| KM | GPU 149/149 비트 일치 | 이후 데모 실행에서 재컴파일 성능 결함을 찾아 고침. 레지스터 스필 해소 후 (`bb44ccd`) GPU 스위트 824 passed, NADPE 1716 비트 일치, 0.15 TMAC/s (§3 이후 판정) |
| I | 수용 커밋 `28cad6f`. CPU 스위트 통과 (macOS, `3b32b07` 기록), Linux 2337 passed (`82db449`) | §3 이후 판정 · I. 현재 판정 파일은 §3 끝 표 |
| D | 수용 커밋 `28cad6f`. `b68addd` 에서 A100 `--quick` 실행 (`b4d7225` 데모 문서) | §3 이후 판정 · D. 현재 판정은 lint 와 `examples/README.md` 검사 명령 |
| 레퍼런스 | NADPE 1716/1716 비트 일치 (현재 AC1, [`tests/data/nadpe/`](../../tests/data/nadpe/)) | 오라클 벡터를 결정론 2400회 재실행과 독립 numpy 모델로 먼저 검증 |

- **판정 방법**: 하위 에이전트의 완료 보고 (`CODEX_DONE / FILES / TESTS / CONCERNS`) 를 그대로 믿지 않고, 지휘 쪽에서
  테스트를 다시 실행해 판정했다.
- **테스트 무변경**: 레인 Q 는 실패 9건을 공유 파일 결함으로 보고했고, 테스트를 결함에 맞게 느슨하게 바꾸지 않았으며
  공유 파일도 고치지 않았다. 현재 [`AGENTS.md`](../../AGENTS.md) §4 금지 사항 1·2 와 같은 내용이며, Wave 1 당시에는
  [`00_common.md`](wave1/00_common.md) 의 공유 명세 수정 금지 · `CONCERNS` 보고 규칙으로 전달했다. 판정 기준 추인 이후의
  판정 파일 변경은 아래 사후 확인 표로 대조했다.
- **공유 파일 수정 경위**: 공유 명세 파일은 지휘 쪽이 작성했고, 하위 에이전트에게는 수정을 금지했다. 수정은 모두
  하위 에이전트의 `CONCERNS` 보고 → 지휘 쪽 확인 → 수정 순서였다: `reference/cast.py` 결함 3종 (Q 보고), 명세 docstring 의
  RHT 곱 순서 (A 보고), 계약서의 E8M0 field-0 적용 범위 (M 보고, NADPE 소스 확인 후). GPU 단계의 커널 · 레퍼런스 수정
  (KQ · KM) 은 프롬프트에 적은 대로 지휘 쪽이 이어서 했다.
- **독립 근거의 두 층**: Triton 커널은 레퍼런스와 (KQ 579/579, KM 149/149), 레퍼런스는 독립 구현 NADPE 와 (1716/1716)
  비트 단위로 비교했다. 현재 SPEC 에서는 둘 다 AC1 의 판정 대상이다.

### 사후 확인 — 판정 파일 변경 이력 (2026-10-09 실행)

Wave 1 (2026-09-30) 은 판정 기준 추인 (2026-10-01, `82db449`) 보다 앞선다. 아래는 추인 커밋을 기준으로 이후의 판정 파일
(`tests/harness`, `tests/test_golden.py`, `tests/data`, `app/tests/`, `app/web/demo/webgpu/golden.json`) 변경과, 저장소
정리 중 바뀐 `tests/test_cli.py` 를 대조한 것이다. 확인 명령의 경로 인자는 이 대상 목록이다.

| 구간 | 확인 명령 | 판정 파일 변경 | 승인 근거 |
|---|---|---|---|
| `82db449` (2026-10-01 08:47 UTC, 팀 추인) → squash 이전 develop 끝 `8086996` (2026-10-08 04:35 UTC) | `git diff --stat 82db449 refs/remotes/b/heads/develop -- <대상>` (이력 클론) | `tests/harness` · `tests/test_golden.py` · `tests/data`: 변경 없음 (두 시점 모두 23개 파일). `app/tests/` · `golden.json`: 추인 뒤 새로 생김 — 7개 파일 (+1,765줄), `625d509` "feat: add TriCast Studio web app" (2026-10-07 09:43 UTC, "app/tests: 159 passed") 와 `5303f99` "fix: harden Studio live mode and GUI state" (10:16 UTC, `test_server.py` +32/−2, "app/tests: 161") | Youngmin17 커밋 — 웹 앱과 함께 새로 만든 판정 파일 (기존 판정 파일 변경 없음) |
| squash 이전 develop 끝 → 현재 루트 `c0998d8` (2026-10-08 09:04 UTC) | 경로별 tree 해시 비교 | 변경 없음 (`tests/harness` `b59544b`, `tests/test_golden.py` `325925a`, `tests/data` `00ed9e0`, `app/tests` `fceeeec`, `golden.json` `55cab21` 동일) | — |
| `c0998d8` → `16a22bd` (2026-10-08 11:10 UTC) | `git diff --numstat c0998d8 16a22bd -- <대상>` | `golden_cases.yaml` +7/−7: AC 번호 재태깅 (NADPE 케이스 2개 `AC7` → `AC1`) 과 note 문구 5줄. `test_golden.py` +18/−3: AC 집합 단언을 AC1–AC6 으로, 허용 키 검사, 새 보조 검사 `test_golden_vectors_present` (NADPE 벡터 sha256 대조). `tests/data` · `app/tests/` · `golden.json` 변경 없음 | 2026-10-08 사용자 승인 (저장소 정리) |
| `c0998d8` → `16a22bd` | `git diff --numstat c0998d8 16a22bd -- tests/test_cli.py` | `tests/test_cli.py` +4/−3: CLI 호출 3곳에 `--device cpu` 추가, 단언은 그대로. 가짜 모델 로더가 device 를 무시해 GPU 가 보이는 머신에서 CLI 테스트 3개가 실패했다 (`3820b5b`, A100 에서 재현 — [`CHANGELOG.md`](../../CHANGELOG.md)) | 2026-10-08~09 사용자 승인 (저장소 정리, Youngmin17) |
| `16a22bd` → 작업 트리 (2026-10-09 확인) | `git diff --numstat HEAD -- <대상>` | `app/tests/test_webgpu_golden.py` +10/−1: `fp64` 필드의 NaN 을 class 로 비교 (모든 NaN 을 `7fc00000` 으로 맞춤) — `webgpu_check.html` 계약과 같은 범위, `expected` 필드는 비트 비교 그대로. x86_64 Linux (geneva) 에서 `nan` 케이스가 `fff00000` 으로 재생성되어 macOS 에서 기록한 골든 (`7ff00000`) 과 달라 `3820b5b` 에서 실패했고, 변경 후 앱 테스트 161개가 통과 ([`CHANGELOG.md`](../../CHANGELOG.md)). 나머지 대상은 변경 없음 | 2026-10-08~09 사용자 승인 (저장소 정리, Youngmin17) |

## 5. 테스트 하니스 결함 주입 검사

골든 테스트 하니스의 결함 주입 검사 (지침 C절) 는 [`delegation_golden_harness.md`](delegation_golden_harness.md) 에 있다.

## 6. 에이전트 하니스

현재 저장소 파일 기준이다.

| 파일 | 역할 | 내용 |
|---|---|---|
| [`CLAUDE.md`](../../CLAUDE.md) | Claude Code 진입점 | `@AGENTS.md` 한 줄로 공통 지침을 불러오고 규칙은 복제하지 않는다. 로드 확인 방법 (새 세션 `/context` 에 `CLAUDE.md` 와 `AGENTS.md` 가 함께 있어야 정상, `docs/SPEC.md` · `docs/ontology.yaml` 은 요청 시 읽는 파일이라 없는 것이 정상), 슬래시 명령과 훅 안내 |
| [`AGENTS.md`](../../AGENTS.md) | 도구와 무관한 상시 지침 (원천) | §2 용어집, §3 절대 규칙 4개 (↔ AC1 · AC2 · AC3 · AC6), §4 금지 사항 — 1 테스트 · 골든 데이터 수정 금지, 2 skip · xfail · 케이스 삭제 · 허용 오차로 완료 조건 축소 금지, 3 케이스별 통과 이유 설명, 4 공유 명세와 `src/tricast/reference/` 변경은 사람 승인, 5 중단 조건 (같은 실패 3회 반복, 테스트 · 골든 데이터 · 공유 명세를 바꿔야만 통과할 것 같을 때, SPEC · AGENTS · 골든 케이스 충돌이나 정해지지 않은 판단이 필요할 때 — 시도한 것 · 실패 출력 · 막힌 결정을 보고), §6 완료의 정의, §7 실행 명령 |
| [`.claude/settings.json`](../../.claude/settings.json) | 도구 권한과 훅 연결 | `permissions.allow`: `make lint`, `make test`, `make check`, `ruff check .`, `pytest *`, `python -m pytest *` (`make check` = lint + CPU 테스트, [`Makefile`](../../Makefile)). `permissions.ask`: `git push *`. PreToolUse (`Edit\|Write\|MultiEdit\|NotebookEdit`) → `guard.py pre`, PostToolUse (`Edit\|Write\|MultiEdit`) → `guard.py post` |
| [`.claude/commands/check.md`](../../.claude/commands/check.md) | 재사용 명령 `/check` — 검사와 보고 (사람이 호출, `disable-model-invocation: true`) | `make lint`, `make test` (CPU), CUDA 와 triton 이 있으면 `make gpu`, `pytest tests/test_golden.py tests/test_compare_golden.py -q -rfs`. 명령별 passed / failed / skipped / xfailed, 골든 노드와 보조 검사를 나눠 세고 skip 사유를 적는다. 필수 케이스의 skip · xfail 은 완료로 치지 않는다. 코드를 고치지 않는다 |
| [`.claude/commands/golden.md`](../../.claude/commands/golden.md) | 재사용 명령 `/golden` — 진단 전용 (사람이 호출, `disable-model-invocation: true`) | `pytest tests/test_golden.py tests/test_compare_golden.py -v -rfs` — 골든 케이스 두 묶음과 `tests/data/nadpe/` 벡터. 실행하지 못했으면 통과로 추정하지 않는다. 실패 케이스마다 AC 번호 (SPEC §6) · 기대값 · 실제값 · 의심 원인을 표로. 테스트 · 골든 데이터는 수정하지 않고 구현 쪽 후보만 제시 |
| [`.claude/hooks/guard.py`](../../.claude/hooks/guard.py) | 판정 기준 변경 통제와 린트 | `pre`: Edit/Write 대상이 보호 목록 (`tests/*`, 공유 명세 4개, `Makefile`, `pyproject.toml`, `.github/workflows/*`, `.claude/*`, 수치 의미론의 정본 `src/tricast/reference/*`) 에 걸리면 `permissionDecision: "ask"` 로 사람에게 묻는다. `post`: 편집한 `.py` 에 `ruff check` — 경고가 있으면 exit 2 로 돌려준다. `TRICAST_HOOKS=off` 로 끈다 |

통제 범위와 다음 정리 대상:

- 훅은 Edit/Write 도구 호출을 대상으로 한다 (`CLAUDE.md`, `guard.py` docstring). Bash 로 바꾼 변경은 수용 전
  `git diff` 대조 (§4 사후 확인 표와 같은 방식) 로 확인한다.
- 보호 목록에 걸린 편집은 사람에게 묻는 방식 (`ask`) 으로 처리한다.
- 보호 목록은 `AGENTS.md` §4.1 · §4.4 와 같은 범위다 (2026-10-09 정리).
- `/check` · `/golden` 은 사람이 호출 시점을 정하는 명령이라 자동 호출을 끈다 (`disable-model-invocation: true`).
- 로드 확인: 새 세션에서 `/context` 로 `CLAUDE.md` 와 `AGENTS.md` 가 함께 로드되는지 본다 (`CLAUDE.md` 안내).
  지침을 넣고 뺀 상태에서 같은 요청을 비교한 기록은 [glossary_check.md](./glossary_check.md) 에 있다.

Wave 1 과의 관계: Wave 1 프롬프트는 기초 커밋 `64c10e3` (2026-09-30 09:08 UTC) 위에서 쓰였고, 규칙을 `AGENTS.md` 대신
[`00_common.md`](wave1/00_common.md) 에 직접 적어 전달했다 — 변경 범위 (소유 파일만, 공유 명세 수정 금지), 보고 경로
(`CONCERNS`), 커밋 · 푸시 · 브랜치 생성 금지, 테스트 · 린트 명령, 완료 보고 형식. `CLAUDE.md` · `AGENTS.md` ·
`.claude/commands/check.md` · `golden.md` 는 Wave 1 레인 산출물이 커밋된 것과 같은 2026-09-30 13:00 UTC 의 커밋 `6b9894e` 에서,
`.claude/settings.json` · `.claude/hooks/guard.py` 는 2026-10-07 07:20 UTC 의 커밋 `7b3f075` 에서 처음 추가됐다.

## 7. 용어집 작동 확인 (강의 4 지침) — 2026-09-30

대화 맥락이 없는 새 에이전트에게 AGENTS.md 만 읽게 하고 "누산기 소수 비트를 13으로, 한 번에 32개씩 묶어서
누산하는 FP8 W8A8 레시피를 만들어줘. 첫 번째와 마지막 레이어는 양자화하지 말고." 를 시켰다.

- 작동: `f_bits: 13`, `chunk_size: 32` 를 용어집 줄에서 그대로 가져왔고 `g_bits` 와 헷갈리지 않았다 (혼동 주의 줄 인용).
  새로 만든 필드 이름은 없었고 `tricast recipe-check` 를 통과했다.
- 부족: 레이어 건너뛰기에 쓰는 `layers` / `skip` 은 용어집에 없어서 스키마와 번들 레시피를 뒤져 찾았다
  (용어집만 보면 `exclude` 를 쓰기 쉬운데, `exclude` 는 "마지막 레이어"를 표현하지 못한다). 절대 규칙 4 의
  `assumptions` 는 레시피 파일에 해당 키가 없어 모호했다.
- 조치: 용어집의 Recipe 줄에 override 선택자 (`match` / `layers` / `modules`, `skip`) 와 `kv` 를 넣고, 규칙 4 에
  `assumptions` 가 어디에 적히는지 (에이전트 파싱 vs 레시피 주석) 를 명시했다. 두 내용은 현재 [`AGENTS.md`](../../AGENTS.md)
  §2 Recipe 줄과 §3 절대 규칙 4 에 있다.

2026-10-08 재확인 결과는 [`glossary_check.md`](glossary_check.md) 에 있다.

---

**원본 기록 범위** — 회차 표의 원본 행은 2026-10-01 기록, "이후 판정" 행과 사후 확인 표는 squash 이전 이력의 커밋
메시지 · 문서와 현재 CHANGELOG · README 에서 찾은 근거만 옮겼다. 그 범위에서 근거를 찾지 못한 항목: KM 의 로컬 판정
(`py_compile` · ruff), M (a) 항목의 후속 조치, Wave 3 MAJOR 15건 중 예시 3건을 뺀 12건의 건별 처리, Wave 1 당시의 기준
커밋 대비 `git diff` (추인 이후 구간은 §4 사후 확인 표), 추인 뒤 새로 생긴 `app/tests/` · `golden.json` (`625d509` ·
`5303f99`) 의 승인 기록.
