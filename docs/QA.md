# QA 기록 — Auto-tune 시연 범위 (2026-11-04)

중간평가 2 시연 범위(Auto-tune 화면 · 요청 해석 · 근거 인용 제안 · 레이어 카드 RAG · 휴리스틱 배낭)를 바꾼 커밋 `6c4b406`…`b489ce5` 에 대해 돌린 검사다. 결정 층(골든 · 단위 테스트, 브라우저 검사)과 통계 층(evals) 을 나눠 적는다. 미실행 항목은 이유와 함께 적는다.

## 1. 결과 요약

| 영역 | 검사 | 명령 | 결과 |
|---|---|---|---|
| 정적 분석 | ruff (저장소 전체) | `ruff check .` | 경고 0 (`b489ce5`) |
| CPU 단위 · 통합 | `tests/` + `app/tests/` (GPU 제외) | `pytest -q --ignore=tests/gpu tests app/tests` (bath 16코어, `d83fc7a`) | **2,930 passed, 3 skipped** — skip: CUDA NF4 대조 · microxcaling 오라클(클론 없음) · GPU 골든 1건 (AGENTS.md 상 정상) |
| CPU 단위 · 통합 (2026-11-05 HEAD) | `tests/` + `app/tests/` (GPU 제외), 옵션 전이 · 재배정 라운드 번들 포함 | 같은 명령 (cpu_cap 16코어 — monaco 배정, `150b9d3` 새 클론, 27분 15초) | **2,948 passed, 3 skipped** — skip 은 위와 같은 3건 |
| CPU (이후 변경, 2026-11-05) | `app/tests` 전체 — 재배정 라운드의 칸 변경 · 옵션 전이 번들 포함 | `pytest -q app/tests` (로컬 `.venv`) | **472 passed** (새 테스트 `test_autotune_replan_bundle.py` 는 수정 전 내보내기에서 실패, 수정 후 통과) |
| CPU (이후 변경) | 튜닝 루프 재개 · DENY 감지 (`b489ce5`) | `pytest -q app/tests/test_tune_resume.py app/tests/test_tune_loop.py …` | 24 passed, 1 skipped (로컬에 optuna 없음; 클러스터 실행에서는 통과) |
| CPU 단위 · 통합 (push 전 HEAD) | `tests/` + `app/tests/` (GPU 제외), 리포트 오차 · 재개 · 평가 러너 수정 포함 | 같은 명령 (cpu_cap 16코어, `a74179f`, 18분 19초) | **2,934 passed, 3 skipped** — skip 은 위와 같은 3건. 그 뒤 `5e1ec8b` 는 `app/__init__.py` 버전 검사만 바꿈 (로컬 `test_server.py` 등 55 passed) |
| evals 러너 | 러너 단위 테스트 + 오답 주입 9행 | `pytest -q tests/test_tune_evals.py tests/test_evals.py` | **82 passed** — 오답 주입 9/9 기대 판정 ([위임 기록](./prompts/delegation_evals.md)) |
| 브라우저: Quantizer | JS ↔ Python 레시피 바이트 일치 (프리셋 5 · 무작위 격자 300 · 옵션 상태 300), 상호작용, 빈 · 오류 상태 | `node scripts/quantizer/qz_check.mjs …` | **QZ_CHECKS[static] 71/71** |
| 브라우저: tricast 수용 | 위 300 격자의 YAML 을 tricast 로더가 해석 | `python scripts/quantizer/qz_tricast_accept.py …` | `QZ_TRICAST_ACCEPT_OK` (레시피 100, 층 19,600) |
| 브라우저: 레이어 지도 | 지표 왕복 · 비유한 값 표시 | `node scripts/quantizer/cdp_check.mjs …` | **CDP_CHECKS 44/44** |
| 브라우저: Auto-tune | 배낭 · 제약 = Python, 요청 상자 · 라운드(저장 · 실시간 fixture · 오류), 칸 툴팁의 리포트 오차 = 번들 값, Tab 포커스, 감소 모션, 390 px, 대비 ≥ 4.5:1 | `node scripts/quantizer/autotune_ui_check.mjs <out> . <python>` | **49/49** (2026-11-08, `e14e78b`: 위 45 + 레시피 파일 4 — 정적 서버는 받는 방법 안내, 기록된 해 `iso_3.15` 의 파일 SHA-256 = 측정 기록 · KL · PPL, 예산을 바꾸면 비워짐 · AI 최선 파일, `--agent` 없으면 격자는 되고 제안 파일은 503) · 45/45 (track M 번들, 세션 4라운드 · 옵션 전이 5행) · 43/43 (전이 · 재배정 라운드 추가 전) |
| 리포트 오차 정직성 | 층별 MSE · SQNR · cos 가 그 방법의 균일 레시피와 같은 레시피 해시의 리포트에서만 옴 | `pytest -q app/tests/test_autotune_reports.py` · 번들 재생성 | 1 passed · 균일 15개 모두 해시 일치, 196층 × 15방법 = 2,940칸 채움, 다른 번들 필드는 이전과 동일 |
| 배낭 정합 | 브라우저 배낭 = `python -m app.autotune solve` (예산별 196층 선택 · 값 · 비트) | `node scripts/quantizer/autotune_check.mjs …` | v2 표 **10/10**, track M 표 **10/10** (불가능 · 너무 큰 예산의 거부 문구 포함; `a74179f` 에서 2.5 · 3 · 403/128 · 3.5 · 4 · 4.16 · 4.25 · 5 · 8 · 1 비트로 다시 10/10) |
| README "화면만 보기" | 저장소를 새로 clone (`a74179f`) → 새 가상환경 (Python 3.14.5 · 3.11) 에 README 의 6개 패키지만 설치 → `python -m app.server --demo` | curl · 위 브라우저 검사를 clone 과 그 가상환경으로 | 화면 6개 · 번들 · health 모두 200 (3.11 은 health), `--agent` 없이 해석 API 503 (의도), **AUTOTUNE_UI 42/42** (3.14). 이 설치로는 `app.autotune solve` 가 torch 를 못 찾는다 — 화면만 보기 범위 밖이며 README 의 전체 설치가 필요 |
| 정직성 | 화면 번들의 측정값이 정확한 레시피에만 붙음 (레시피 파일 SHA-256 → KL 기록 → PPL 기록) | `python -m app.export_autotune_demo …` | 균일 15 · 배낭 6 · 옵션 전이 5 모두 대조 통과, 거부 0. 측정하지 않은 전이 격자(`xfer_iso_3.15`: 세션 최선과 이름만 다름)는 '측정 기록 없음'으로 거부됨을 확인한 뒤 뺐다 |
| 도입 슬라이드 | 중간평가 2 도입 3장 (대상 · 문제 / 입력 → 결과 / 최종 목표), 수치는 `docs/slides/midterm2/make_data.py` 가 번들 · seed 기록에서 생성 | `python3 docs/slides/midterm2/make_data.py` → `node docs/slides/midterm2/build_deck.js <out.pptx>` → paper-slides `render_qa.sh` | 3/3 페이지 렌더 확인: 넘침 · 겹침 없음, 폰트 AppleSDGothicNeo 만, 빨강 강조 장당 1–2곳. 사용자 덱(v3–v7)은 건드리지 않음 |
| 레시피 파일 API | `POST /api/autotune/recipe`: 배낭 격자 → `solve` 가 쓰는 YAML, 기록된 6개 해 모두 SHA-256 이 측정 기록과 같음, 세션 제안 → 그 라운드 `kl.json` 의 `file_sha256` 과 대조, 잘못된 격자 422, tricast 없으면 503 | `pytest -q app/tests/test_autotune_recipe.py` | 6 passed (`act_damp05_r3` 파일 SHA `8cd7cee3…` = 원격 측정 파일) |
| judge 검증 도구 | 라벨링 시트에 judge 입력 전체(검색 문서 포함) · 판정 없음, 일치율은 같은 실행 · 같은 단위의 true/false 짝만 셈, 불일치 방향 | `pytest -q tests/test_judge_agreement.py` | 3 passed. 사람 라벨은 아직 없다 |
| CPU 단위 · 통합 (evals 커밋) | `tests/` + `app/tests/` (GPU 제외) | 같은 명령 (cpu_cap 8코어 — lyon, `b313676` 새 클론, 18분 27초; 16코어는 클러스터가 차서 DENY) | **2,954 passed, 3 skipped** — skip 은 위와 같은 3건 |
| evals 기준선 (2026-11-09 ②) | 35 케이스 live 3회 (같은 커밋 `6c039bc`, 새 클론 3개) | `evals/run_tune_evals.py --mode live` | 코드 층 A1–A7 100% · 절대 규칙 0건 (3회 모두), B1 9/19 · 11/19 · 16/19 (평균 0.63, 3회 폭 0.37, 목표 0.70 미달 — 0214 한 번은 0.84, 능력 지표), judge 지표는 사람 라벨 전이라 판정 보류. 앞선 두 단계(`b7e3eca` · `598c757`)가 케이스 r_c9 · r_q4 의 기대 결과 오류와 연산 예산 충돌의 빈틈을 찾아 고친 뒤 다시 잼 ([설계 §9](../evals/EVAL_DESIGN.md)) |
| evals 재현 | 저장소만으로 stored 재채점 (`evals/fixtures/tune_inputs`, ledger 280 레코드, `runs/` 없는 클론 `177f644`) | `--mode stored --from-run …` | 3회 모두 지표(충족 표시 포함) · 35 케이스 판정이 live 와 같음. 검토 반영 뒤 `344cb8e` 클론에서 다시 같음, 오답 주입 8/8 (27 케이스 기준선도 `b328289` 에서 같은 확인) |
| evals 오답 주입 | 위임 기록 §1-2 의 8건을 실제 저장 응답(`2026-11-08T0104`)에 넣어 stored 재채점 | `python evals/inject_faults.py 2026-11-08T0104 …` | **8/8** 기대 판정, 넣은 케이스만 바뀜 |
| 실제 서버 E2E | `python -m app.server --demo --agent <loop.json>` 위의 화면 · 번들 · 실시간 해석 · 실시간 제안 | curl (로컬 `.venv`) | health 200 · 화면 200 · 번들 200 (720 KB) · **실시간 해석 6.2초 $0.029** → `tune · 403/128 · BF16 유지 [-1]` · **실시간 제안 68초 $0.28** → 1 채택 · 1 거부(같은 레시피가 이미 세션에 있음), 원본 세션 파일 수 33 → 33 |
| CPU 단위 · 통합 (2026-11-09 기준선 커밋) | `tests/` + `app/tests/` (GPU 제외), evals 확장 · 러너 수정 · 연산 예산 충돌 · 튜닝 골든 포함 | 같은 명령 (cpu_cap 8코어, `598c757` 의 git 클론, 18분 3초) | **3,003 passed, 3 skipped** — skip 은 위와 같은 3건. `.git` 없는 `git archive` 트리에서는 실행 기록에 커밋을 남기는 테스트 9건이 git_sha None 으로 실패해(환경 문제, 코드 결함 아님) 클론으로 다시 돌렸다 |
| CPU 단위 · 통합 (2026-11-09 최종, 검토 반영) | 같은 범위 + 연산 예산 충돌 단위 테스트 · 골든 결함 주입 · 예산 문자열 거부 | 같은 명령 (cpu_cap 16코어, `344cb8e` 의 git 클론, 18분 50초) | **3,026 passed, 3 skipped** — skip 은 위와 같은 3건 |
| CPU 단위 · 통합 (2026-11-09 ② 기준선 커밋) | 위 범위 + 녹화 재생 | 같은 명령 (cpu_cap 16코어, `6c039bc` 의 git 클론, 18분 15초) | **3,032 passed, 3 skipped** — skip 은 위와 같은 3건 |
| 최종 확인 (`4102738`) | CPU 전체 스위트 (git 클론) · `runs/` 없는 클론에서 기준선 3회 stored 재채점 · 오답 주입 · 그 클론으로 띄운 오프라인 서버의 리허설 · 화면 검사 | 위 명령들 | **3,032 passed, 3 skipped** (17분 56초) · 재채점 3회 모두 지표 · 35 케이스 판정이 live 와 같음 · **8/8** · **REHEARSAL_DONE 10/10** (claude · `runs/` 없이) · **49/49** |
| 녹화 재생 (오프라인 데모) | `--record` 가 실시간 답을 그대로 남기고, `--replay` 가 claude 를 부르지 않고 같은 답을 `mode: replay` · 녹화 시각과 함께 돌려줌, 녹화 안 된 문장은 녹화 목록과 404, 다른 요청의 녹화는 거부, 플래그 검사 | `pytest -q app/tests/test_autotune_replay.py` | 6 passed. 녹화본: 시나리오 문장 + 질의 대비 6 문장 ($0.18) + Claude 라운드 5 (3 채택 · 1 거부, $0.22) — `app/web/demo/autotune/replay/` |
| 화면(tune.html) · 리허설 (자동) | 시연 5단계(모델 불러오기 → Auto-tune · AI 추천 → 직접 편집 → 편집한 셋업 평가 → 추천으로 되돌려 평가)를 화면 조작 그대로: 숫자 · "측정한 파일과 같음" · 출처 칩 · 다음에 누를 버튼이 화면 안 · 가로 넘침 | `node scripts/quantizer/demo_rehearsal.mjs <폴더> [주소] [폭]` | `--agent` 서버 1440 px **5/5** · 390 px **5/5** (2026-11-11), 페이지 오류 0, 가로 넘침 없음. `--agent` 없는 서버와 GPU 서버(실제 평가)는 이 구성에서 아직 돌리지 않았다 (미검증). 예비 녹화 `~/Documents/AI캡스톤/tricast_demo_backup.gif` |
| 화면(tune.html) · 기능 검사 | 배낭이 Worker 에서 돌고 메인 스레드가 멈추지 않음 · 측정된 예산 미리 계산 · **한 시스템**(4단계뿐 — 다른 화면 메뉴 · 채팅 · 차트 · 입력 칸 없음) · 크기 버튼(크기 % + 추천 셋업 PPL) · 6개 크기 모두 추천 = 측정된 것 중 최선(AI 세부 설정은 더 좋을 때만) · **직접 편집은 측정된 것만**: 57개 조합(세부 설정 13 · 전체를 한 방법으로 44) 모두 측정한 레시피 파일(SHA 일치)과 기록된 PPL · 주소로 작업 복원(편집 포함) · 추천 평가(숫자 · 출처 칩 · PPL 증가 막대) · 편집 평가와 추천으로 되돌리기 · 격자는 읽기 전용 · 1440 · 390 px 가로 넘침 · 1440 × 900 과 1280 × 720 에서 평가 버튼이 화면 안 · GPU 평가 경로(가짜 GPU fixture) | `node scripts/quantizer/tune_app_check.mjs <폴더> .venv/bin/python [주소]` | **15/15** (2026-11-11) — 계산 2.97초 동안 메인 스레드 최장 정지 22 ms (대조 루프 154 ms), 미리 계산분 9 ms, 추천 PPL 133.29 / 30.22 / 26.51 / 23.00 / 21.73 / 21.67 (27% 는 AI 설정이 더 나빠 쓰지 않음), 편집 57/57 측정. `pytest app/tests/test_autotune_uniform_recipe.py` 3 passed (한 방법 15종의 레시피가 측정 파일과 SHA 일치). GPU 경로는 **가짜 GPU** 로 페이지 처리만 확인했다 (실 GPU 미검증) |
| judge AI 교차 라벨 (사용자 위임) | 다른 엔진(Argon)이 판정 없는 시트로 21 단위 · 47 판정 | `python evals/judge_agreement.py agree … evals/judge_labels_ai.jsonl` | 일치율 0.915 (43/47) — 참고값, 사람 라벨이 아니라 judge 검증이 아니다 |
| 튜닝 골든 (2026-11-09) | AC24 제안 검사 · AC28 요청 검사를 저장소의 실제 데모 자료로: 측정된 제안의 재수락 · 레시피 SHA, 한 필드 변형 거부, 기준선에서 거부된 실제 응답 3건, 표에서 계산한 충돌(크기 · 연산 예산) · 되묻기, 제약 아래 배낭 | `pytest -q tests/test_tune_golden.py` | **35 passed** — 케이스 29 + 매니페스트 + 결함 주입 5: 수치 검사 · 크기 충돌 · 연산 충돌 · 예산 검사를 끄거나 제약을 무시하면 해당 케이스가 모두 실패함을 테스트가 확인 (`test_the_goldens_catch_a_broken_check`) |
| 연산 예산 충돌 (2026-11-09) | 연산 가중치가 섞인 합성 표(nvfp4 1/4 · int8 1/2 · 가중치 전용 1 · nvint4 비평가)에서 가장 싼 연산 · 허용 방법 · 비평가 층 · 충돌 판정, 숫자로 못 읽는 예산("1/0", "1.5/2")은 422 | `pytest -q app/tests/test_tune_request_compute.py` | 15 passed (데모 표는 모든 방법이 연산 1 이라 이 경로를 가르지 못한다는 독립 검토 지적으로 추가) |
| evals 러너 수정 (2026-11-09, 사용자 승인) | 소수 목표("≥ 87.5%"), 0/0 = 해당 없음, git 상태를 쓰기 전에 읽기, 기대 키 `compute_budget` | `pytest -q tests/test_tune_evals.py` · 11-08 기준선 3회 stored 재채점 · `inject_faults.py` | 73 passed (판정 회귀 테스트 포함). 재채점: 지표 분자 · 분모와 27 케이스 판정 그대로, A1 충족 표시만 바로잡힘. 오답 주입 **8/8**. live 실행의 `git_dirty` 가 false 로 기록됨 |
| 브라우저: Auto-tune (2026-11-09) | 위 49 항목, 연산 예산 충돌 검사 뒤 | `node scripts/quantizer/autotune_ui_check.mjs <out> . <python>` | **49/49** (`598c757` 트리) |
| GPU | `tests/gpu` | — | **미실행**: 이번 범위는 커널 · GPU 경로를 바꾸지 않았다. 마지막 결과는 H200 896 passed ([SPEC](./SPEC.md#6-수용-기준) AC1). |

## 2. 접근성 · 화면 (Auto-tune, `autotune_ui_check.mjs`)

- 키보드: Tab 으로 닿는 요소 60/60, 모두 포커스 링 표시. 슬라이더 · 격자 칸 방향키 동작.
- 색: 방법 계열 7색 + BF16 중립. dataviz 검증기로 밝은 · 어두운 배경에서 인접쌍 CVD ΔE ≥ 8.4 확인. 칸 글자는 두 테마 모두 대비 4.5:1 이상 (최저 4.76). 색만으로 구분하지 않도록 칸마다 방법 약어를 적는다.
- 움직임: 예산을 바꾸면 바뀐 칸만 다시 칠한다 (마지막 지연 240 ms + 전환 140 ms). `prefers-reduced-motion` 에서는 0 초.
- 폭: 1440 · 390 px 모두 페이지 가로 스크롤 없음 (격자만 자기 영역 안에서 스크롤).

## 3. 이번 QA 에서 찾아 고친 것

| 찾은 것 | 영향 | 조치 |
|---|---|---|
| 튜닝 루프가 gpu_cap 의 실제 거부 줄(`DENY host=…`)을 못 봄 | 거부된 라운드를 최대 `wait_s`(90분) 기다림 | `b489ce5`: 거부 줄도 grep, 회귀 테스트 `test_the_launch_sees_the_real_deny_line` |
| 측정이 멈춘 라운드의 요약이 사라짐 | 재실행이 그 라운드를 거부하고 분석 · 거부 사유를 잃음 | `b489ce5`: 측정 전에 `measuring` 으로 기록, 재실행이 다시 측정 |
| 레이어 프로파일의 모델 revision 누락 | 프로파일이 표의 층 문서와 갈라져 RAG 에서 따로 나옴 | `d1a8bbb`: 불러온 config 의 커밋 해시 기록, 재실행 결과 통계 동일 |
| 검색된 층 문서가 예전 표 줄로 시작 | 2,400자에서 잘려 세션 표 · 프로파일이 빠짐 | `ed83459`: 세션 표 KL → 프로파일 → 리포트 순으로 좁힘 |
| 1차원 배낭이 예산 전체를 칸으로 펼침 | 16방법 표에서 브라우저 풀이 4–7초, 6비트 이상 거부 | `ec5fcf1`: 층마다 가장 싼 선택지 위만 다룸 (같은 해, 칸 3–7배 감소) |
| evals 데이터셋 r_c6 (초안) | 어떤 응답으로도 통과 불가 | 예산 3 → 3.2 (설계서 §2 초안 수정 기록) |
| 측정이 멈춘 라운드의 재개가 실제 클러스터 경로에서 실패 (`round 2 exists`) | 데모 세션 2라운드를 다시 잴 수 없음. 단위 테스트는 가짜 `measure` 를 써서 라운드 폴더를 만드는 `measure_remote` 를 지나지 않았다 | `951abf3`: 같은 제안 · kl.json 없는 폴더는 같은 두 파일로 다시 씀, 다른 제안이면 거부. 회귀 테스트는 실제 `measure_remote` 를 가짜 ssh 로 (`app/tests/test_tune_resume_remote.py`) |
| 제안 카드의 옵션 줄이 GPTQ 계열 6종의 전체 옵션(기본값 포함)을 나열 | 바뀐 것(`act_order`, `damp`)이 15줄 속에 묻혀 시연에서 읽히지 않음 | 번들에 `options_changed` (`qr.method_options`: 기본값과 다른 것만, 검사를 통과한 것만), 같은 변경의 방법을 한 묶음으로 표시, 검사 추가 |
| 레이어 카드의 방법별 옵션이 저비트 방법(GPTQ int2/3 · RHT · AWQ …)에서 비어 있음 | 카드 JSON 이 그 방법들의 옵션 공간을 안 보여 줌 | `METHOD_OPTIONS.get` 이 `__missing__` 을 건너뜀 → 인덱싱으로 (`app/tests/test_option_summary.py`), 카드 다시 생성 |
| 재개한 2라운드 측정이 geneva 에서 `PackageNotFoundError: hqq` | 라운드 래퍼가 PYTHONPATH 를 트리 경로로 덮어써, 시작점을 잰 환경(`pydeps_qm`)의 hqq 를 못 봄 | `9c34299`: `remote.env` 의 PYTHONPATH 를 트리 경로 뒤에 붙임 (없으면 래퍼 바이트 동일), 세션 설정에 `pydeps_qm` 추가 · RESUME.txt 기록, 띄우기 전 bath 에서 hqq 0.2.8.post1 확인 |
| macOS 기본 `python3` (3.9.6) 로 README 를 따르면 설치는 되고 서버가 pydantic `TypeError: Unable to evaluate type annotation 'str \| None'` 로 죽음 | 동료가 원인을 모른 채 막힘 | `app/__init__.py` 가 3.10 미만이면 한 줄로 이유 · 해결을 말하고 종료 (3.9 에서 rc=1 확인, 3.11 · 3.14 health 200), README 첫 줄에 `python3 --version` |
| 툴팁의 "outlier 입력 채널" 이 비율(0.0088)을 개수처럼 표시 | 발표자 · 동료가 채널 수로 오독 | 비율 · 개수로 표시 (`0.88% (27/3072)`), fixture 의 가짜 값을 비율로 바꾸고 문구 검사 추가 (가짜 값이 정수 6 이라 놓쳤다) |

## 4. 남은 위험 · 미검증

- 데모 세션(`runs/20261104_demo/s3150`)은 3라운드 · 측정 8건까지 끝났다 ([증거 §2.6](./evidence/2026-11-04_lowbit_track_m.md)). AI 최선 둘의 test PPL 은 출발점과 같은 트리 · GPU 로 쟀다 (26.82 · 26.51 vs 27.04). 보정 seed 42–46 로도 쟀다: 5/5 seed 에서 AI 가 낮고 평균 −3.48%, 부호 뒤집기 p = 1/32 ([증거 §2.7](./evidence/2026-11-04_lowbit_track_m.md)). turin(PCIe) 의 seed 42 값이 geneva(SXM4) 와 float 단위로 같다. 라운드 4 (배낭 재배정, LLM 없음) 는 검증 KL 0.39309 로 채택되지 않았고 ([§2.8](./evidence/2026-11-04_lowbit_track_m.md)), AI 옵션을 다른 예산의 배낭 해 5개에 옮긴 측정은 [§2.9](./evidence/2026-11-04_lowbit_track_m.md) 에 있다 (대조군 KL 이 세션 기록과 비트 단위로 같음).
- 브라우저 배낭은 메인 스레드에서 돈다. track M 표에서 2.5–4.25 비트 풀이가 0.5–2.9초 걸려 그동안 화면이 멈춘다 (Web Worker 이전은 다음 작업).
- 브라우저 검사는 Google Chrome (macOS) 만 돌렸다. Safari · Firefox 는 손으로 확인하지 않았다.
- README 의 "화면만 보기" 설치는 Python 3.14 · 3.11 에서 확인했다 (3.10 · 3.12 · 3.13 · Windows 는 미확인).
- evals 는 합격선 · 절대 규칙 기대 결과 · judge 라벨이 팀 확정 전이라, 실행해도 결과를 근거로 쓰지 않는다 ([설계서 §7](../evals/EVAL_DESIGN.md)).
- 제안자가 받는 RAG 문서의 프로파일 줄에는 통계 이름만 있고 정의가 없다 (예: `outlier_channels` 가 비율이라는 사실). 진행 중인 데모 세션의 입력이 라운드 중간에 바뀌지 않도록 이번에는 고치지 않았다 — 다음 세션 전에 문서 머리에 정의를 넣는다.
