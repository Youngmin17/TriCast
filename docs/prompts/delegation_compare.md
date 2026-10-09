# 위임 기록 — PPL 비교 판정 `compare_ppl` (강의 5, 2026-10-09)

강의 5 지침의 순서 그대로 돈 위임 한 사이클이다: 사람이 기준(AC13, 골든 케이스)을 확정 → 테스트 하니스 위임 →
사람의 하니스 검토(결함 주입) → 기능 위임 → 골든 케이스 단위 회차 기록 → 사람의 수용 판단.

| 역할 | 누구 |
|---|---|
| 기준 확정 · 수용 판단 | Youngmin17 |
| 지휘 · 검증 실행 | 메인 Claude Code 세션 (Youngmin17 지휘) |
| 위임 받은 쪽 | Claude Code 하위 에이전트 (general-purpose), 위임마다 새 세션 |

## 1. 위임 대상 선정

- **위임한 기능**: `src/tricast/eval/compare.py` 의 `compare_ppl(base, emu, limit=1e-3)` — 두 평가 기록(EvalRun)의 PPL 을
  비교해 상대 차이와 합격 여부를 낸다. 파일 하나, 함수 하나.
- **직접 담당한 것**: 비교해도 되는 조건(모델 revision · 데이터셋 지문 · 평가 토큰 수가 같을 것), 상대 차이의 정의(절댓값,
  기준 PPL 로 나눔), 경계 포함 여부(한계와 같으면 합격), 기준선이 무효일 때와 에뮬레이션이 NaN 일 때의 처리.
- **위임한 이유**: 입력과 기대 결과를 데이터(골든 케이스)로 적을 수 있고, 완료 여부를 테스트로 판정할 수 있다.
- **직접 담당한 이유**: "무엇을 같은 조건으로 볼 것인가"는 제품의 판단이다. 근거는 페인 P7 "누가 맞는지 가릴 공통 비교
  기준이 없어 책임 공방·일정 지연이 생긴다"(로그 06 · 16 · 18)와 로그 16 의 "누구 책임인지 가릴 기준이 없다는 거죠"다.

## 2. 수용 기준과 골든 케이스 (사람이 확정)

- 수용 기준: [SPEC](../SPEC.md#6-수용-기준) **AC13** [이벤트 기반] — 2026-10-09 Youngmin17 확정.
- 골든 케이스: [`tests/harness/golden_compare.yaml`](../../tests/harness/golden_compare.yaml) — 처음 11건을 확정하고, 회차 2 에서 3건,
  회차 3 에서 10건, 회차 4 에서 3건을 더해 27건이다 (모두 같은 날 Youngmin17 확정). 회차 3 에서 AC 를 AC13 (판정) · AC14 (거부) 로 나눴다.
  아래 표는 처음 확정한 11건이다 (회차 3 이후의 분류는 §8).

| 분류 | 케이스 | 판정 키 | 검사 범위 |
|---|---|---|---|
| 정상 | `compare_passthrough_within_limit`, `compare_lowacc_fails` | `pass`, `relative` | 반환값 전체 |
| 경계 | `compare_exact_limit_passes`, `compare_just_above_limit_fails`, `compare_lower_ppl_is_still_a_change`, `compare_nan_emulation_fails` | `pass` (+ `relative`) | 한계 경계, 부호, NaN |
| 금지 | `compare_rejects_model_revision_mismatch`, `compare_rejects_dataset_mismatch`, `compare_rejects_token_count_mismatch`, `compare_rejects_invalid_baseline`, `compare_rejects_missing_metric` | `error` (필드 경로) | 예외 종류와 메시지 접두어 |

- 정상 케이스의 수치는 실제 기록이다: Qwen3-0.6B native 20.9662 · `bf16_passthrough` 20.9669 (`support_matrix.yaml`),
  Llama-3.2-1B native 9.7564 ([SPEC AC4](../SPEC.md#6-수용-기준)) · `fp8_f7_lowacc` 141.25 ([PROBLEM §3-5](../PROBLEM.md)).
- 경계 케이스 1000 → 1001 은 (1001 − 1000) / 1000 의 반올림 결과가 리터럴 `1e-3` 과 같은 double 이 되도록 고른 값이다. 이 등식은
  뺄셈을 먼저 하는 계산식에서 성립하므로, 골든의 상대 오차 1e-12 는 계산식 자체를 고정하는 역할도 한다 (YAML 머리 주석).

## 3. 테스트 하니스 위임

### 3-1. 프롬프트 원본 (2026-10-09, 실제로 보낸 텍스트)

```text
tests/test_compare_golden.py 를 만들어줘.
- 목표: tests/harness/golden_compare.yaml 의 `cases` 를 읽어 케이스마다 하나씩 도는 골든 테스트. `_records` 는 YAML 앵커 저장소라 케이스로 읽지 않는다.
- 케이스를 코드에 복사하지 말 것. YAML 에서 읽고, 케이스 `id` 를 테스트 id 로 쓴다.
- 대상 함수: `tricast.eval.compare.compare_ppl(base, emu, limit=1e-3)` → `{"relative": float, "pass": bool, "limit": float}`. 케이스 `input` 에 `limit` 키가 있으면 넘기고, 없으면 기본값을 쓴다.
- 검사 항목 (케이스의 expected 에 있는 키만 검사): `pass` → 반환 `pass` 와 같음 / `relative` → 반환 `relative` 와 상대 오차 1e-12 이내 (기대값이 inf 면 정확히 inf) / `error` → `ValueError` 가 나고 메시지가 `"<경로>:"` 로 시작.
- 로더 검사 (항상 실행되는 별도 테스트): id 중복 없음, 모든 케이스에 id·ac·kind·input·expected·note, 케이스 키는 이 여섯 개 밖의 키를 허용하지 않음, expected 에 pass·relative·error 중 하나 이상이고 그 밖의 키는 없음, ac 는 AC13, kind 는 normal / boundary / forbidden, note 에 ac 문자열이 들어 있음.
- `tricast.eval.compare` 가 아직 없으면 케이스 테스트는 skip 하되 로더 검사는 항상 실행한다.
- src/ 아래는 건드리지 말 것. 테스트 파일만 만든다. 기존 tests/test_golden.py 의 스타일(모듈 상수로 YAML 로드, 짧은 함수)을 따르고 ruff (line-length 110) 를 통과한다.
- 중단 조건: 같은 실패가 3회 반복되거나, YAML·docs/SPEC.md AC13 과 충돌하는 판단이 필요하면 멈추고 보고한다.
끝나면 `--collect-only -q` 결과와 `-q -rs` 실행 결과를 원문 그대로 보고해줘.
```

(앞에 붙인 공통 머리: 작업 레포 경로, 커밋·push 금지, 실행 명령, "레포 규칙은 AGENTS.md 를 먼저 읽어라".)

### 3-2. 하위 에이전트가 정한 것 (프롬프트 밖, 지휘 쪽 검토 후 유지)

| 결정 | 판단 |
|---|---|
| 모듈이 **없을 때만** skip (`find_spec`), 있는데 import 가 실패하면 실패 | 유지 — 구현 후 skip 으로 통과하는 길을 막는다 (AGENTS §4.2) |
| `pass` 는 `is` 로 비교, `relative` 는 `float` 인지 먼저 확인 | 유지 — 반환 계약이 `bool` · `float` |
| 입력을 `deepcopy` 해서 넘김 | 유지 — YAML 앵커가 dict 하나를 여러 케이스에서 같이 쓴다 |
| 로더 검사에 "케이스 0건 금지" | 유지 |

### 3-3. 사람 쪽 하니스 검토 — 결함 주입 (2026-10-09)

작업 트리를 건드리지 않도록 `src/` · `tests/` 복사본에서 `src/tricast/eval/compare.py` 자리에 임시 구현을 넣어 돌렸다.
명령: `pytest tests/test_compare_golden.py -q -rf`. 하니스와 YAML 은 고정.

| 단계 | 넣은 구현 | 실패해야 할 케이스 (사전 예측) | 실제 결과 |
|---|---|---|---|
| (1) 수집 | — | 로더 1 + 골든 11 = 12 노드 | `12 tests collected`, 미구현 상태 `1 passed, 11 skipped` (skip 사유: 모듈 없음) |
| (2) 통과 가능 확인 | 케이스별 기대값을 YAML 에서 찾아 그대로 돌려줌 | 없음 | `12 passed`, skip 0 |
| (2) F1 판정 로직 생략 | 항상 `{relative: 0.0, pass: True}` | 골든 11건 전부 | `11 failed, 1 passed` — 골든 11건 |
| (2) F2 금지된 결과 | 조건 검사 없이 비교해 결과를 냄 | 금지 5건 | `5 failed, 7 passed` — `compare_rejects_*` 5건 |
| (2) F3 경계 | `relative < limit` (경계 제외) | `compare_exact_limit_passes` | `1 failed, 11 passed` — 그 1건 |
| (3) 원복 | 임시 구현 삭제 | — | `1 passed, 11 skipped`, 작업 트리에 `compare.py` 없음 |

검토 결과 하니스를 고정했다 (파일 머리 표기, AGENTS §4.1).

### 3-4. 하니스 회차 기록

| 회차 | 결과 | 무엇이 문제였나 | 무엇을 고쳤나 |
|---|---|---|---|
| 1 | 로더 1 passed, 골든 11 skipped (미구현). 하위 에이전트 보고: 자체 확인에서 구현 결함 변형 10종 · YAML 결함 10종 검출 | 하위 에이전트 보고: ① YAML 머리 주석의 판정 키 이름(`expect_pass` …)이 데이터(`pass` …)와 다름 ② SPEC AC13 행의 `\|` 이스케이프 때문에 `tests/test_spec_traceability.py` 의 표 파서가 셀 수를 잘못 셈 | [컨텍스트] 두 문서 오류는 지휘 쪽이 고침 — YAML 주석의 키 이름(판정 데이터 무변경), SPEC 문장에서 `\|` 제거. 하니스 코드는 무변경 |

## 4. 기능 위임

### 4-1. 프롬프트 원본 (2026-10-09, 실제로 보낸 텍스트)

```text
src/tricast/eval/compare.py 의 compare_ppl(base, emu, limit=1e-3) 를 구현해줘.
- 목표: 두 평가 기록(EvalRun)의 PPL 을 비교해 {"relative": float, "pass": bool, "limit": float} 를 돌려준다.
- 절대 규칙 (AGENTS.md §4): 테스트와 골든 케이스(tests/test_compare_golden.py, tests/harness/golden_compare.yaml)를 수정하지 말 것. skip·xfail·허용 오차 추가로 통과시키지 말 것.
- 판단 규칙 (docs/SPEC.md §6 AC13, 골든 YAML 머리 주석):
  1. 두 기록 모두 metrics.ppl 이 있어야 한다. 없으면 ValueError("<base|emu>.metrics.ppl: ...").
  2. env.model_sha, metrics.ppl.dataset_fingerprint, metrics.ppl.n_tokens 가 두 기록에서 같아야 한다. 다르면 ValueError("<그 경로>: ...") — 이 경로에는 base/emu 접두어를 붙이지 않는다.
  3. 기준 PPL(base.metrics.ppl.ppl)이 유한한 양수가 아니면 ValueError("base.metrics.ppl.ppl: ...").
  4. relative = |emu − base| / base. emu PPL 이 유한하지 않으면 relative = inf.
  5. pass = relative <= limit (경계 포함). pass 는 Python bool, relative 는 float.
  6. 입력 dict 를 바꾸지 않는다. 검사 순서는 1 → 2 → 3.
- 완료 기준: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_compare_golden.py -q -rs` 에서 골든 케이스 11건 + 로더 1건 = 12 passed, skipped 0, xfailed 0. 그리고 `.venv/bin/ruff check .` 무경고.
- 변경 범위: src/tricast/eval/compare.py 하나만 새로 만든다. 타입 힌트 필수, 모듈 docstring 한 줄, 레포의 기존 eval 모듈 스타일을 따른다.
- 중단 조건: 같은 실패가 3회 반복되거나, 판단 규칙과 골든 케이스가 충돌한다고 판단되면 멈추고 시도한 것·실패 출력·막힌 결정을 보고한다.
끝나면 pytest 출력 원문과, 각 골든 케이스(11건)를 왜 통과하는지 한 줄씩 설명해줘.
```

| 지침 D절 요소 | 이 프롬프트의 줄 | 출처 |
|---|---|---|
| 대상 · 목표 | 첫 줄, "목표" | 위 §1 |
| 절대 규칙 | "절대 규칙" | AGENTS.md §4.1 · §4.2 |
| 판단 규칙과 참조 위치 | "판단 규칙" 1~6 | SPEC AC13, 골든 YAML 머리 주석 |
| 완료 기준 (명령 + 통과 조건) | "완료 기준" | 12 passed, skip · xfail 0 |
| 변경 범위 | "변경 범위" | 파일 하나 |
| 중단 조건 | "중단 조건" | AGENTS.md §4.5 |
| 통과 이유 요구 | 마지막 줄 | AGENTS.md §4.3 |

## 5. 검증 루프 기록 (골든 케이스 단위)

| 회차 | 결과 (골든 11건) | 무엇이 문제였나 | 무엇을 고쳤나 |
|---|---|---|---|
| 1 | **11 / 11 통과** — `pytest` 출력 `12 passed` (골든 11 + 로더 1), skipped 0, xfailed 0. `ruff check .` 무경고 | — | 없음. 케이스별 통과 이유를 받아 아래 표로 검토 |
| 2 | **14 / 14 통과** — `15 passed` (골든 14 + 로더 1), skipped 0, xfailed 0 | 회차 1 에서 드러난 미명세 3건 (§6) 이 골든 케이스로 고정되지 않았음 | [케이스] Youngmin17 이 §6 의 처리를 승인하고 금지 케이스 3건을 골든에 추가 (`compare_rejects_missing_model_revision`, `compare_rejects_non_numeric_emulation`, `compare_rejects_invalid_limit`). 구현 무변경 |

- 실행한 상태와 명령: 회차 1 · 2 · 3 모두 커밋 `16a22bd` 위 작업 트리 (macOS 로컬 `.venv`),
  `PYTHONPATH=src .venv/bin/python -m pytest tests/test_compare_golden.py -q -rs`. 출력은 이 기록에 원문으로 옮겼다.
  최종 판정 파일과 이 기록은 커밋 `a249030`, 구현은 그 다음 커밋 `e819485` 에 들어갔다.
  (지휘 쪽 재실행 결과도 `12 passed in 0.11s`).
- 하위 에이전트는 CPU 전체 `pytest -q` 도 돌려 `2256 passed, 899 skipped` (skip 은 CUDA 전용) 를 보고했다.
- 판정 파일 대조 (구현 위임 전후): 결함 주입 직전 복사본과 비교해 `golden_compare.yaml` 은 같고, `test_compare_golden.py` 는
  지휘 쪽이 고친 머리 주석 (1줄 → 2줄) 만 다르다. 구현 위임에서 바뀐 파일은 새 파일 `src/tricast/eval/compare.py` 하나다.
  회차 2 · 3 의 케이스 추가는 사람이 내린 결정이며 각 회차 행에 적었다.

### 5-1. 케이스별 통과 이유 (하위 에이전트 설명 → 지휘 쪽 검토)

| 케이스 | 통과 이유 | 판정 기준과 맞는가 |
|---|---|---|
| `compare_passthrough_within_limit` | \|20.9669 − 20.9662\| / 20.9662 = 3.338707061834613e-05 ≤ 1e-3 | 맞음 — 같은 조건에서 비교 |
| `compare_lowacc_fails` | (141.25 − 9.7564) / 9.7564 = 13.47767619203805 > 1e-3 | 맞음 |
| `compare_exact_limit_passes` | 1 / 1000 = 0.001, `<=` 라 경계 합격 | 맞음 — 경계 포함 규칙 |
| `compare_just_above_limit_fails` | 0.0010000000999999656 > 1e-3 | 맞음 |
| `compare_lower_ppl_is_still_a_change` | 절댓값이라 995 도 0.005 → 불합격 | 맞음 |
| `compare_nan_emulation_fails` | emu NaN → 오류 없이 relative = inf, pass = False | 맞음 — 비교 오류가 아니라 불합격 결과 |
| `compare_rejects_model_revision_mismatch` | `env.model_sha: base 'aaa' differs from emu 'bbb'` | 맞음 — 점수가 아니라 거부 |
| `compare_rejects_dataset_mismatch` | `metrics.ppl.dataset_fingerprint: base 'fp' differs from emu 'other'` | 맞음 |
| `compare_rejects_token_count_mismatch` | `metrics.ppl.n_tokens: base 64 differs from emu 63` | 맞음 |
| `compare_rejects_invalid_baseline` | 조건 검사 통과 후 기준 0.0 이 3번 규칙에서 거부 | 맞음 — 검사 순서 1 → 2 → 3 |
| `compare_rejects_missing_metric` | emu `metrics` 가 `{}` → 1번 규칙에서 거부 | 맞음 |

코드 확인: 같은 조건 검사는 `SAME_CONDITIONS` 세 경로를 비교해 다르면 `ValueError` 를 내는 판정 조건이고, 상대 차이에 섞는
가중치가 아니다. 입력 dict 는 읽기만 한다.

회차 2 에서 더한 3건의 통과 이유 (현재 구현으로 다시 실행한 메시지):

| 케이스 | 통과 이유 |
|---|---|
| `compare_rejects_missing_model_revision` | 두 기록 모두 revision 이 없어 `env.model_sha: base has None, expected a non-empty string` |
| `compare_rejects_non_numeric_emulation` | `emu.metrics.ppl.ppl: 'n/a' is not a number` |
| `compare_rejects_invalid_limit` | `limit: -0.001 is not a finite non-negative number` |

## 6. 새로 드러난 미명세 항목

프롬프트와 골든 케이스에 없던 경우를 하위 에이전트가 다음처럼 정했다 (어느 것도 골든 케이스로 고정되지 않음).

| 경우 | 구현의 처리 | 근거 (하위 에이전트) |
|---|---|---|
| 비교 필드가 없거나 `None` (양쪽 모두 없어도) | `"<경로>: missing in base\|emu"` 로 거부 | `runner` 가 `model_sha` 로 비어 있지 않은 문자열을 요구, AGENTS §2 "EvalRun 수치는 env 가 함께 있을 때만 근거" |
| emu PPL 이 숫자가 아님 (문자열 · `None` · bool) | `emu.metrics.ppl.ppl: … is not a number` 로 거부 | NaN · inf 는 측정 결과, 숫자가 아닌 값은 기록 오류 |
| `limit` 이 NaN · 음수 · inf | `limit: … is not a finite non-negative number` 로 거부 | 한계가 정의되지 않으면 판정할 수 없음 |

결정 (2026-10-09, Youngmin17): 세 처리를 모두 승인하고 금지 케이스로 고정했다. SPEC AC13 문장에도 같은 조건을 적었다.

## 7. 수용 판단

| 항목 | 내용 |
|---|---|
| 확인한 것 (회차 2 시점) | 골든 14 / 14 통과 (skip · xfail 0), `ruff check .` 무경고, 판정 파일 무변경 (위임 전 복사본 대조), 케이스별 통과 이유 검토, 같은 조건 검사가 거부 조건으로 구현됨, 미명세 3건 결정 |
| 수용 | **수용 — Youngmin17, 2026-10-09** |
| 이후 | 이 하니스와 골든 케이스의 수정은 사람이 승인한다 (AGENTS.md §4.1, `.claude/hooks/guard.py` 보호 목록 `tests/*`) |

## 8. 회차 3 — 적대적 검토가 찾은 공백

수용 뒤 적대적 검토가 두 가지를 찾았다. (1) AC13 문장의 조건 일부 — 사용자 지정 한계, emu `inf`, 유한하지 않은 기준 PPL, NaN 한계,
bool, base 쪽 지표 누락, 반환 `limit` — 를 검사하는 골든이 없어, 구현에 넣은 결함 7종이 14건을 모두 통과했다. (2) 구현이 빈 revision ·
`n_tokens: 0` 을 받아들여 러너의 유효성 규칙과 어긋났고, `compare_ppl` 을 부르는 운영 경로가 없었다.

### 8-1. 기준 변경 (사람이 확정, 2026-10-09 Youngmin17)

- AC 를 강의 4 의 EARS 유형대로 나눴다: **AC13** [이벤트 기반] 판정 · **AC14** [예외 대응] 거부 ([SPEC §6](../SPEC.md#6-수용-기준)).
- 골든 10건 추가 (AC13 3 · AC14 7), 기존 금지 케이스를 AC14 로 다시 태깅 → 24건 (AC13 정상 3 · 경계 6, AC14 금지 15).
- YAML 머리 주석에 상대 오차 1e-12 가 계산식을 고정한다는 뜻을 적었다.

### 8-2. 하니스 재위임 (같은 하위 에이전트 세션) — 지시 원문

```text
회차 3 — tests/test_compare_golden.py 를 고쳐줘. 골든 YAML 은 사람이 이미 고쳤다 (AC13 판정 · AC14 거부로 나눔, 24건, docs/SPEC.md §6 AC13·AC14).
- 로더 검사 보강: ac 는 AC13 또는 AC14. input 키는 base · emu · limit 밖을 허용하지 않고 base · emu 는 필수. expected 에 error 가 있으면 pass · relative 와 함께 둘 수 없다.
- 케이스 검사 보강: input 에 limit 이 있고 error 케이스가 아니면 반환 "limit" 이 입력 limit 과 같아야 한다.
- 그 밖의 판정 (키별 검사, 모듈이 없을 때만 skip, deepcopy, 상대 오차 1e-12) 은 그대로 둔다. 파일 머리 표기 두 줄도 그대로 둔다.
- src/ 와 YAML 은 건드리지 말 것.
- 중단 조건: 같은 실패가 3회 반복되거나, YAML · SPEC AC13/AC14 와 충돌하는 판단이 필요하면 멈추고 보고한다.
끝나면 `--collect-only -q` 개수와 `-q -rs` 결과 원문, 그리고 현재 구현 (src/tricast/eval/compare.py) 에서 실패하는 케이스 목록과 각 실패 메시지 첫 줄을 보고해줘.
```

결과: `25 tests collected` (로더 1 + 골든 24). 회차 2 구현으로 `2 failed, 23 passed` — 실패는 `compare_rejects_empty_model_revision`,
`compare_rejects_zero_token_count` (둘 다 `DID NOT RAISE <class 'ValueError'>`) 로, 구현의 공백을 하니스가 잡았다. 하위 에이전트
보고: 로더 결함 6종 (ac `AC12`, input 오타 키, base · emu 누락, error 와 pass · relative 동시) 모두 검출.
판정 파일 고정 시점의 sha256: `test_compare_golden.py` `b0e27e4f22bc1b5a…`, `golden_compare.yaml` `641003e39fbfeecb…`.

### 8-3. 기능 재위임 (같은 하위 에이전트 세션) — 지시 원문

범위를 파일 하나에서 세 파일로 넓힌 이유: §8 의 (2) — 판정을 부르는 운영 경로가 없었다. `summarize.py` 는 native 와 레시피의
PPL 을 조건 확인 없이 비교하던 곳이라 AC14 의 판정을 쓸 첫 자리다.

```text
회차 3 — src/tricast/eval/compare.py 를 고쳐줘. 기준이 바뀌었다: docs/SPEC.md §6 AC13 (판정) · AC14 (거부), 골든 24건 (tests/harness/golden_compare.yaml), 하니스도 보강됐다 (tests/test_compare_golden.py). 둘 다 사람이 고정한 판정 파일이니 수정하지 말 것.
- 판단 규칙 추가 (AC14): 비교 조건 값은 러너의 유효성 규칙과 같게 본다 — `env.model_sha` · `metrics.ppl.dataset_fingerprint` 는 비어 있지 않은 문자열, `metrics.ppl.n_tokens` 는 bool 이 아닌 0 보다 큰 정수. 한쪽이라도 어기면 `"<경로>: ..."` 로 거부한다 (경로에 base/emu 접두어 없음, 기존 같은 조건 검사와 같은 형식). 숫자 판정은 src/tricast/eval/runner.py 의 `_finite_number` 와 같은 규칙을 쓰고, 아주 큰 정수에서 OverflowError 가 새지 않게 한다 (ValueError 로 거부).
- `tricast.eval` 의 `_EXPORTS` (src/tricast/eval/__init__.py) 에 `compare_ppl` 을 추가한다.
- scripts/e2e/summarize.py 의 `ppl_table` 이 native 와 각 레시피 기록을 비교할 때 `compare_ppl` 로 비교 가능 여부를 먼저 판정하고, 거부되면 그 행의 "vs native" 칸에 `not comparable: <ValueError 메시지의 경로>` 를 쓴다. 비교 가능하면 지금처럼 부호 있는 % 를 쓴다. native 값 하나를 고르는 지금의 검사는 유지한다.
- 완료 기준: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_compare_golden.py -q -rs` 에서 골든 24 + 로더 1 = 25 passed, skipped 0, xfailed 0. summarize.py 를 다루는 테스트가 tests/ 에 있으면 그것도 통과. `.venv/bin/ruff check .` 무경고.
- 변경 범위: src/tricast/eval/compare.py, src/tricast/eval/__init__.py, scripts/e2e/summarize.py 세 파일.
- 중단 조건: 같은 실패가 3회 반복되거나, 판단 규칙과 골든 케이스가 충돌한다고 판단되면 멈추고 시도한 것·실패 출력·막힌 결정을 보고한다.
끝나면 pytest 출력 원문, 회차 3 에서 새로 추가된 골든 10건 각각이 왜 통과하는지 한 줄씩, summarize.py 를 실제 기록 형태의 dict 두 개로 호출해 본 출력 (비교 가능 / 거부 각 1) 을 보고해줘.
```

### 8-4. 회차 3 결과

| 회차 | 결과 (골든 24건) | 무엇이 문제였나 | 무엇을 고쳤나 |
|---|---|---|---|
| 3 | **24 / 24 통과** — `25 passed` (골든 24 + 로더 1), skipped 0, xfailed 0. `ruff check .` 무경고 | §8 첫 단락의 두 공백 | [케이스] 10건 추가와 AC 분리 (사람), [하니스] 로더 · 반환 limit 검사 (재위임), [구현] 러너 유효성 규칙 · export · `summarize.py` 연결 (재위임) |

- 판정 파일 대조: 기능 재위임 전후 sha256 이 같다 (`shasum -a 256 -c` OK).
- 운영 경로: `scripts/e2e/summarize.py` 의 `ppl_table` 이 `compare_ppl` 로 native 와 각 레시피의 비교 가능 여부를 먼저 판정한다. 실제 러너
  기록(`runs/20261001_cluster/ppl_d_f3d6fe6/`)으로 호출한 출력 — 같은 조건이면 `+310821.85%`, 레시피 기록의 데이터셋 지문만 바꾸면
  `not comparable: metrics.ppl.dataset_fingerprint`.

새 골든 10건의 통과 이유 (하위 에이전트 설명 → 지휘 쪽 검토):

| 케이스 | 통과 이유 |
|---|---|
| `compare_custom_limit_passes` | limit 0.01 유효, 0.005 ≤ 0.01 → pass, 반환 limit 0.01 |
| `compare_zero_limit_accepts_identical` | limit 0 유효, relative 0.0 ≤ 0.0 → pass (경계 포함) |
| `compare_infinite_emulation_fails` | emu inf 는 숫자라 거부하지 않고 relative inf, pass False |
| `compare_rejects_infinite_baseline` | 기준 inf 가 유한하지 않아 `base.metrics.ppl.ppl:` |
| `compare_rejects_negative_baseline` | 기준 −1.0 이 양수가 아니어서 `base.metrics.ppl.ppl:` |
| `compare_rejects_nan_limit` | limit NaN 이 유한하지 않아 `limit:` |
| `compare_rejects_empty_model_revision` | 값이 같아도 유효성 검사가 먼저 — `env.model_sha: base has ''` |
| `compare_rejects_zero_token_count` | 0 이 양의 정수가 아니어서 `metrics.ppl.n_tokens:` |
| `compare_rejects_missing_baseline_metric` | base `metrics` 가 `{}` → `base.metrics.ppl:` |
| `compare_rejects_boolean_emulation` | bool 은 숫자로 보지 않아 `emu.metrics.ppl.ppl:` |

### 8-5. 결함 재주입 — 검토가 찾은 7종이 이제 걸리는가 (지휘 쪽, 복사본)

최종 구현에 결함을 하나씩 넣고 `pytest tests/test_compare_golden.py -q -rf` 를 돌렸다. 회차 1 골든 (14건) 에서는 7종 모두 통과했다.
재현: `python scripts/harness/compare_fault_injection.py` — 이 7종에 §3-3 의 세 결함을 더한 10종을 복사본에서 돌려 모두
검출되는지 확인한다 (검출되지 않는 결함이 있으면 종료 코드 1).

| 결함 | 결과 | 실패한 케이스 |
|---|---|---|
| a 한계 인자 무시 (1e-3 고정) | `1 failed, 24 passed` | `compare_custom_limit_passes` |
| b emu inf 를 오류로 처리 | `1 failed, 24 passed` | `compare_infinite_emulation_fails` |
| c 기준 inf · 음수 허용 | `2 failed, 23 passed` | `compare_rejects_infinite_baseline`, `compare_rejects_negative_baseline` |
| d 한계 NaN 허용 | `1 failed, 24 passed` | `compare_rejects_nan_limit` |
| e bool 을 PPL 로 받음 | `1 failed, 24 passed` | `compare_rejects_boolean_emulation` |
| f base 지표 누락 검사 생략 | `1 failed, 24 passed` | `compare_rejects_missing_baseline_metric` |
| g 반환에서 limit 키 제거 | `2 failed, 23 passed` | `compare_custom_limit_passes`, `compare_zero_limit_accepts_identical` |
| 원복 | `25 passed` | — |

### 8-6. 회차 3 에서 드러난 미명세 항목

| 경우 | 구현의 처리 |
|---|---|
| 숫자 판정 규칙 | 러너의 `_finite_number` 와 같은 규칙을 `compare.py` 에 따로 둔다 (러너는 torch · numpy 를 불러와 import 가 무겁다) |
| float 로 표현할 수 없는 큰 정수 emu PPL (예 10**400) | `emu.metrics.ppl.ppl: … is too large for a float` 로 거부 |
| `summarize.py` 표 아래 요약 줄 | 첫 complete 기록의 창 수 · 토큰 수 · 지문을 그대로 쓴다 (비교 거부 행이 있어도) |

결정 (2026-10-09, Youngmin17): 세 처리를 그대로 승인했다 (골든은 추가하지 않음).

### 8-7. 회차 3 수용 판단

| 항목 | 내용 |
|---|---|
| 확인한 것 | 골든 24 / 24 (skip · xfail 0), `ruff check .` 무경고, 재위임 전후 판정 파일 sha256 동일, 검토가 찾은 결함 7종 모두 검출, `summarize.py` 연결을 실제 기록으로 확인, 미명세 3건 결정 |
| 수용 | **수용 — Youngmin17, 2026-10-09** |
| 커밋 | 판정 파일 (골든 · 하니스) 과 이 기록은 구현보다 앞선 커밋에, 구현 · export · `summarize.py` 는 그 다음 커밋에 들어간다 |

## 9. 회차 4 — 재채점이 찾은 미고정 경계

수용 뒤 독립 재채점이 골든으로 고정되지 않은 동작 3가지를 실행으로 찾았다. Youngmin17 이 세 가지 모두 **현재 구현의 동작을
기준으로 확정**했고 (2026-10-09), 골든 3건을 더했다. 구현은 바꾸지 않았다.

| 케이스 | AC | 확정한 동작 |
|---|---|---|
| `compare_model_id_alone_is_not_a_condition` | AC13 경계 | revision 이 모델을 특정하므로 이름만 다른 기록은 비교한다 → pass, relative 0.0 |
| `compare_negative_emulation_fails` | AC13 경계 | 숫자인 에뮬레이션 PPL 은 음수라도 판정한다 → relative 1.005, 불합격 |
| `compare_rejects_boolean_token_count` | AC14 금지 | bool 은 토큰 수가 아니다 → `metrics.ppl.n_tokens:` 로 거부 |

| 회차 | 결과 (골든 27건) | 무엇이 문제였나 | 무엇을 고쳤나 |
|---|---|---|---|
| 4 | **27 / 27 통과** — `28 passed` (골든 27 + 로더 1), skipped 0, xfailed 0 | 세 동작이 골든에 없었다 | [케이스] 3건 추가 (사람 확정). 하니스 · 구현 무변경 |

최종 구성: AC13 정상 3 · 경계 8, AC14 금지 16.

