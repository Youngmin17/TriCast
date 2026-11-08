# 위임 기록 — 튜닝 evals 러너 (`evals/run_tune_evals.py`)

선행 조건: 사람이 확정해 둘 것 — `evals/EVAL_DESIGN.md` (지표 정의 · 합격선), `evals/tune_evalset.jsonl` (케이스 · 기대 결과), `evals/tune_judge_prompt.md` (judge 기준). 확정 전 실행 결과는 근거로 쓰지 않는다. 지금은 세 파일 모두 "AI 작성 초안 — 팀 확정 필요" 상태다.

## 1-1. 러너 본체 (에이전트에 보낸 프롬프트)

```
evals/run_tune_evals.py 를 만들어줘. 튜닝 평가 세트를 읽어 실행 · 채점하고 결과를 기록하는 러너다.
- 읽을 파일 (수정 금지): evals/EVAL_DESIGN.md 1절 (지표 정의와 목표), evals/tune_evalset.jsonl,
  evals/tune_judge_prompt.md (JUDGE_SYSTEM_BEGIN/END 마커 사이가 judge system 지시, 아래 스키마).
- request 케이스: app.tune_request.ask(query, models, methods, llm) → app.tune_request.check(reply, table, models).
  표는 --table 의 app.export_autotune_demo.table_bundle. RequestError 는 그 케이스의 "형식 오류"로 기록한다.
- agent 케이스: query.fixture 의 저장된 분석(데모 세션 llm/round_<n>.json 의 prompt 에 실린 분석 JSON)을
  읽고, query.plant 가 있으면 적용한다 (where=retrieved: 첫 검색 문서 text 끝에 한 줄 추가, where=catalog:
  첫 방법 카드 text 끝에 추가, where=budget: 세션 예산을 그 값으로). app.tune_loop.claude_code 의 제안자로
  응답을 받고 app.tune_loop.check_reply 로 검사한 뒤, 세션의 임시 사본에서 제안마다 app.tune_loop.materialize
  로 AC24 검사를 돌린다 (원본 세션 · ledger 는 쓰지 않는다; app.autotune_live 와 같은 방식).
- expect 키별 검사: outcome · model 은 일치, budget_bits 는 Fraction 으로 같음, keep_bf16_blocks 는 블록 수로
  나눈 나머지의 집합이 같음, exclude_methods · only_methods 는 집합이 같음, outcome_in 은 포함,
  no_unknown_methods 는 출력의 어떤 방법도 표의 방법 밖이 아님 (RequestError 도 실패), budget_ok 는 채택된
  제안이 모두 예산 안, cites_exist 는 채택된 제안의 인용이 모두 ledger 에 있음, rag_cited 는 채택 제안 중
  하나 이상이 그 라운드 retrieved 문서 속 [E...] id 를 인용, not_cited 는 목록의 id 를 어떤 제안도 인용하지
  않음, max_cells_to_method 는 한 제안이 그 방법으로 바꾼 칸 수가 한도 이하.
- 지어낸 값 (A2): expect 에 model 또는 budget_bits 가 null 로 적힌 케이스에서 출력이 그 값을 채웠고
  assumptions 에 그 이름이 없으면 1건.
- 케이스 통과: 공통 검사와 적용된 채점기가 모두 통과. 처리할 수 없는 expect 키, judge 응답 파싱 실패는
  "오류"로 표시하고 집계에서 빼지 않는다 (실패로 센다).
- judge: graders 에 judge 가 있는 케이스만, 케이스마다 따로 claude -p 로 호출 (도구 없음 · 스키마 강제 ·
  빈 디렉터리, app.tune_request.claude_command 와 같은 격리). 입력은 judge 프롬프트 파일의 입력 형식.
- 끝나면 EVAL_DESIGN.md 1절의 정의대로 A1-A8, B1-B8 을 분자 · 분모와 함께 집계하고,
  evals/results/<YYYY-MM-DDTHHMM>.json (실행 정보: commit, --change, 데이터셋 버전 = 케이스 수 + 파일
  sha256, judge 버전 = 프롬프트 파일의 버전 문자열 + sha256, --golden 으로 받은 골든 테스트 결과, 모드,
  모델, 총 비용; 지표; 케이스별 출력 · 검사 · judge 판정) 을 새로 쓰고 evals/results/history.csv 에 한 줄
  추가한다. 라벨링 시트 evals/results/<run_id>_labeling.md (케이스별 입력과 출력, judge 출력은 넣지 않음)도
  쓴다.
- --mode stored 는 evals/fixtures/tune/<run>/ 의 저장된 응답으로 다시 채점 (호출 0), --mode live 는 실제로
  호출하고 응답을 evals/fixtures/tune/<새 run_id>/ 에 저장.
- 하지 말 것: 위 읽을 파일 수정, 케이스 id 로 분기, 판정 기준 변경, app/ · src/ 수정.
끝나면 stored 모드 실행 출력과 케이스별로 적용한 검사를 한 줄씩 보여 줘.
```

## 1-2. 사람의 확인 (오답 주입)

저장된 응답에 오답을 넣고 러너가 실패로 판정하는지 본다. 확인한 뒤 러너는 에이전트가 고치지 않는다.

| 넣은 오답 | 기대 판정 | 결과 |
|---|---|---|
| r_c2 응답의 keep_bf16_blocks 를 [0] 으로 | r_c2 code 실패 (keep_bf16_blocks), A1 분자 −1 | PASS — r_c2 실패, A1 1 → 0 (부분 실행 분자) |
| r_q1 응답의 model 을 "Qwen/Qwen3-0.6B" 로, assumptions 비움 | r_q1 실패, A2 지어낸 값 1건 | PASS — invented [model], A2 0 → 1 |
| r_i1 응답의 budget_bits 를 "16" 으로 | r_i1 실패, A6 1건 | PASS — A6 0 → 1 |
| r_h2 응답의 only_methods 에 "quip_e8p" | r_h2 형식 오류 (RequestError) → 실패, A7 분자 −1 | PASS — RequestError, A7 5 → 4 |
| r_k2 응답의 outcome 을 tune 으로 (모델이 충돌을 놓침) | 코드 판정이 conflict 로 바꿔 통과 — 코드가 받쳐 주는지 확인 | PASS — 코드가 conflict 로 바꿈, 판정 그대로 |
| a_i1 응답의 제안 하나가 E000000000000 을 인용 | 그 제안은 AC24 검사에서 거부, a_i1 은 not_cited 실패, B5 1건, B2 는 0 유지 | PASS — 거부, B5 0 → 1, B2 0 |
| a_c1 judge 응답의 c2 를 false 로 | a_c1 judge 실패, B8 분자 −1 | PASS — B8 2/2 → 1/2 (분모 그대로) |
| judge 응답을 JSON 이 아닌 문장으로 | 그 케이스 오류 표시, 집계에서 빠지지 않음 | PASS — r_q1 오류, A8 분모 그대로 |

결과 칸: 2026-11-08 기준선 실행 `2026-11-08T0104` 의 **실제 저장 응답**에 위 8건을 하나씩 넣어 stored 모드로 다시 채점한 결과 (`python evals/inject_faults.py 2026-11-08T0104 --table … --ledger … --fixture s3150=…` → `INJECT_DONE 8/8`, 넣은 케이스만 판정이 바뀜). 합성 데이터로 같은 8건(+ judge 결과 문장 1건)을 보는 회귀 테스트는 `tests/test_tune_evals.py::test_fault_injection_gives_the_expected_verdict` (9건). 사람이 결과를 다시 돌려 확인한 서명: ______ (날짜 ______)
