> 튜닝 골든 하니스(AC24 · AC28)의 결함 주입 기록이다. 하니스는 2026-11-09 에 AI 가 초안을 쓰고 사용자가 승인했다
> (`tests/test_tune_golden.py` 머리 주석). 이 기록의 실행은 2026-11-17 Claude Code 가 Youngmin17 의 지시로 했다.

# 튜닝 골든 하니스 결함 주입 검사 (강의 5 지침 C절)

## 1. 대상과 범위

| 항목 | 값 |
|---|---|
| 하니스 파일 | `tests/test_tune_golden.py`, `tests/harness/golden_tune.yaml` (케이스 29개 — 정상 5 · 경계 8 · 금지 16, AC24 14 · AC28 15) |
| 결함을 넣은 구현 파일 | `app/tune_agent.py` (제안 검사, AC24) · `app/tune_request.py` (요청 해석의 코드 검사, AC28) |
| 기준 트리 | `7270ac4` — `app/` 은 이 커밋 그대로 (`git diff --quiet app/` 통과) |
| 재현 명령 | `PYTHONPATH=src python scripts/harness/tune_fault_injection.py [out.json]` |
| 환경 | Python 3.14.5, macOS (arm64), CPU — LLM 호출 · GPU 없음. 입력은 저장소의 데모 자료 `evals/fixtures/tune_inputs/` |

절차: 결함마다 **29개 케이스 전부**를 돌려 실패한 케이스와 메시지 첫 줄을 적는다 (테스트 안의
`test_the_goldens_catch_a_broken_check` 는 기대한 케이스만 돌리므로, 다른 케이스가 같이 깨지는지는 여기서 본다).
결함은 두 가지로 넣었다.

- **A** — 테스트 파일의 `MUTATIONS` 5종: 구현 함수 하나를 망가진 함수로 바꾼다.
- **B** — 조건문 하나를 `if False:` 로 바꿔 검사 하나를 끈다 (8종). 디스크의 파일은 쓰지 않고 메모리의 모듈만 바꾸며,
  결함마다 모듈 상태를 스냅숏에서 되돌린다.

결함 없는 실행은 처음과 끝에 한 번씩 했다: 두 번 모두 **29/29 통과**.

## 2. 결과 요약

| # | 바꾼 파일 | 바꾼 내용 | 실패해야 할 케이스 | 실제로 실패한 케이스 | 판정 |
|---|---|---|---|---|---|
| A1 | `app/tune_agent.py` | `unsupported_numbers` 가 늘 빈 목록 (근거 글 숫자 검사 끔) | `number_not_in_record` · `predicted_value_as_measured` · `floor_subtracted_value` · `other_proposals_parameter` | 같은 4개 | 검출 |
| A2 | `app/tune_request.py` | `cheapest_bits` 가 늘 0 (크기 충돌 계산 끔) | `budget_below_any_assignment` · `only_int8_conflict` · `half_the_blocks_bf16_conflict` | 같은 3개 | 검출 |
| A3 | `app/tune_request.py` | `cheapest_compute` 가 늘 0 (연산 충돌 계산 끔) | `compute_below_any_assignment` | 같은 1개 | 검출 |
| A4 | `app/tune_agent.py` | `evaluate` 에 예산 99 비트를 넘김 (예산 검사 끔) | `over_budget_by_cells` · `over_budget_by_session_budget` | 같은 2개 | 검출 |
| A5 | `app/tune_request.py` | `constrained` 가 제약을 무시하고 표를 그대로 반환 | `only_int8_conflict` · `half_the_blocks_bf16_conflict` · `solved_within_constraints` · `solved_fraction_budget` | 같은 4개 | 검출 |
| B1 | `app/tune_agent.py` | `if not cites:` → `if False:` (인용 없는 제안 허용) | `no_citation` | 같은 1개 | 검출 ※1 |
| B2 | `app/tune_agent.py` | `if unknown:` → `if False:` (ledger 에 없는 인용 허용) | `cite_outside_ledger` | 같은 1개 | 검출 |
| B3 | `app/tune_agent.py` | `if mentioned:` → `if False:` (근거 글 속 없는 id 허용) | `rationale_names_unknown_id` | 같은 1개 | 검출 |
| B4 | `app/tune_request.py` | `if errors:` → `if False:` (응답 스키마 검사 끔) | `budget_not_a_number` | 같은 1개 | 검출 ※1 |
| B5 | `app/tune_request.py` | `if result["model"] is not None and result["model"] not in models:` → `if False:` (표 없는 모델 허용) | `model_without_table` | 같은 1개 | 검출 |
| B6 | `app/tune_request.py` | 방법 이름 검사의 `if unknown:` → `if False:` (표에 없는 방법 허용) | `unknown_excluded_method` · `unknown_allowed_method` | 같은 2개 | 검출 |
| B7 | `app/tune_request.py` | `if outside:` → `if False:` (모델 밖 블록 허용) | `block_outside_model` | 같은 1개 | 검출 |
| B8 | `app/tune_request.py` | 모델 · 예산이 빠진 `tune` 을 되묻기로 바꾸는 조건 → `if False:` | `no_model_asks` · `no_size_budget_asks` | 같은 2개 | 검출 ※2 |

케이스 이름은 앞의 `tune_proposal_` · `tune_reply_` · `tune_request_` 를 줄였다. **13종 모두 실패해야 할 케이스가 빠짐없이
실패했고, 그 밖의 케이스는 하나도 실패하지 않았다** (`TUNE_FAULTS_DONE 13/13`).

※1 다른 검사가 대신 거부했지만, 골든이 거부 메시지까지 확인하므로 실패로 잡혔다. B1 은 숫자 근거 검사가
(`rationale numbers not supported … ['0.43614466651129274']` — 인용이 없으니 근거 숫자도 없다), B4 는 예산 숫자 검사가
(`budget_bits: '3 bits' is not a number`) 거부했다. 사용자 쪽에서는 두 경우 모두 여전히 거부되지만(이중 방어),
어느 검사가 막았는지는 골든이 구분한다.
※2 `no_model_asks` 는 `outcome: tune` 이 그대로 남아 단언에서, `no_size_budget_asks` 는 예산 없이 배낭을 풀려다
`TypeError` 로 실패했다 — 둘 다 "값을 지어내지 않고 되묻는다"가 깨진 것을 잡는다.

## 3. 실패 메시지 (첫 줄 원문)

| # | 케이스 | 메시지 |
|---|---|---|
| A1 | 4개 모두 | `Failed: DID NOT RAISE <class 'app.tune_agent.TuneError'>` |
| A2 | 3개 모두 | `AssertionError: {'outcome': 'tune', …}` (기대 `conflict`) |
| A3 | `compute_below_any_assignment` | `AssertionError: {'outcome': 'tune', … 'compute_budget': '5/16', …}` |
| A4 | 2개 모두 | `Failed: DID NOT RAISE <class 'app.tune_agent.TuneError'>` |
| A5 | 충돌 2개 | `AssertionError: {'outcome': 'tune', …}` |
| A5 | `solved_*` 2개 | `AssertionError` (배낭의 해가 제약을 어김) |
| B1 | `no_citation` | `AssertionError: rationale numbers not supported by the cited records or the proposal: ['0.43614466651129274']` |
| B2 · B3 | 각 1개 | `Failed: DID NOT RAISE <class 'app.tune_agent.TuneError'>` |
| B4 | `budget_not_a_number` | `AssertionError: budget_bits: '3 bits' is not a number (write "3.2" or "403/128")` |
| B5 · B6 · B7 | 각 케이스 | `Failed: DID NOT RAISE <class 'app.tune_request.RequestError'>` |
| B8 | `no_model_asks` | `AssertionError: {'outcome': 'tune', 'model': None, 'budget_bits': '3', …}` |
| B8 | `no_size_budget_asks` | `TypeError: argument should be a string or a Rational instance or have the as_integer_ratio() method` |

## 4. 결함을 넣지 않은 케이스

29개 중 아래 7개는 이번 결함 목록이 겨냥하지 않았다. 결함 없는 실행에서는 모두 통과한다.

| 케이스 | 분류 | 이유 |
|---|---|---|
| `tune_proposal_real_accepted_same_recipe` · `tune_reply_real_accepted` | 정상 | 수락돼야 하는 케이스 — 검사를 끄면 오히려 통과한다. 레시피 SHA-256 이 세션 기록과 같은지를 본다 |
| `tune_session_measurement_protocol` | 정상 | 세션 파일의 고정값(검증 창 · seed 43)을 읽는 케이스 |
| `tune_proposal_method_without_table_entry` · `tune_proposal_unknown_method` | 금지 | 검사가 격자 생성(`app/tune_agent.py` 의 `_recipe` → 생성기 build)과 KL 표 대조(`app/autotune.py:513`) 안에 있어 이번 목록에 넣지 않았다 — 다음 보강 후보 |
| `tune_request_out_of_scope_kept` | 금지 | 코드가 해석의 `out_of_scope` 를 그대로 두는 것을 보는 케이스라 끌 검사가 없다 |
| `tune_request_compute_budget_met` | 경계 | 연산 예산이 표의 최소와 같아 통과해야 하는 케이스 |

## 5. 판단

- 하니스는 AC24 의 제안 검사(근거 숫자 · 인용 · 예산)와 AC28 의 코드 검사(충돌 계산 · 되묻기 · 표에 없는 값 거부 · 스키마)를
  하나씩 꺼도 해당 골든 케이스로 잡는다. 결함마다 다른 케이스가 같이 깨지지 않아, 실패한 케이스 이름이 곧 꺼진 검사를 가리킨다.
- 테스트와 YAML 은 바꾸지 않았다. 판정 데이터를 고치지 않고 구현 쪽에만 결함을 넣었다.
- 남은 빈틈: §4 의 격자 생성 · KL 표 대조 검사 2개는 결함을 넣지 않았다.
