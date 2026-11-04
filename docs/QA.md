# QA 기록 — Auto-tune 시연 범위 (2026-11-04)

중간평가 2 시연 범위(Auto-tune 화면 · 요청 해석 · 근거 인용 제안 · 레이어 카드 RAG · 휴리스틱 배낭)를 바꾼 커밋 `6c4b406`…`b489ce5` 에 대해 돌린 검사다. 결정 층(골든 · 단위 테스트, 브라우저 검사)과 통계 층(evals) 을 나눠 적는다. 미실행 항목은 이유와 함께 적는다.

## 1. 결과 요약

| 영역 | 검사 | 명령 | 결과 |
|---|---|---|---|
| 정적 분석 | ruff (저장소 전체) | `ruff check .` | 경고 0 (`b489ce5`) |
| CPU 단위 · 통합 | `tests/` + `app/tests/` (GPU 제외) | `pytest -q --ignore=tests/gpu tests app/tests` (bath 16코어, `d83fc7a`) | **2,930 passed, 3 skipped** — skip: CUDA NF4 대조 · microxcaling 오라클(클론 없음) · GPU 골든 1건 (AGENTS.md 상 정상) |
| CPU (이후 변경) | 튜닝 루프 재개 · DENY 감지 (`b489ce5`) | `pytest -q app/tests/test_tune_resume.py app/tests/test_tune_loop.py …` | 24 passed, 1 skipped (로컬에 optuna 없음; 클러스터 실행에서는 통과) |
| CPU 단위 · 통합 (push 전 HEAD) | `tests/` + `app/tests/` (GPU 제외), 리포트 오차 · 재개 · 평가 러너 수정 포함 | 같은 명령 (cpu_cap 16코어, `a74179f`, 18분 19초) | **2,934 passed, 3 skipped** — skip 은 위와 같은 3건. 그 뒤 `5e1ec8b` 는 `app/__init__.py` 버전 검사만 바꿈 (로컬 `test_server.py` 등 55 passed) |
| evals 러너 | 러너 단위 테스트 + 오답 주입 9행 | `pytest -q tests/test_tune_evals.py tests/test_evals.py` | **82 passed** — 오답 주입 9/9 기대 판정 ([위임 기록](./prompts/delegation_evals.md)) |
| 브라우저: Quantizer | JS ↔ Python 레시피 바이트 일치 (프리셋 5 · 무작위 격자 300 · 옵션 상태 300), 상호작용, 빈 · 오류 상태 | `node scripts/quantizer/qz_check.mjs …` | **QZ_CHECKS[static] 71/71** |
| 브라우저: tricast 수용 | 위 300 격자의 YAML 을 tricast 로더가 해석 | `python scripts/quantizer/qz_tricast_accept.py …` | `QZ_TRICAST_ACCEPT_OK` (레시피 100, 층 19,600) |
| 브라우저: 레이어 지도 | 지표 왕복 · 비유한 값 표시 | `node scripts/quantizer/cdp_check.mjs …` | **CDP_CHECKS 44/44** |
| 브라우저: Auto-tune | 배낭 · 제약 = Python, 요청 상자 · 라운드(저장 · 실시간 fixture · 오류), 칸 툴팁의 리포트 오차 = 번들 값, Tab 포커스, 감소 모션, 390 px, 대비 ≥ 4.5:1 | `node scripts/quantizer/autotune_ui_check.mjs <out> . <python>` | **43/43** (track M 번들, 리포트 · 측정된 세션 3라운드 포함; 제안 카드의 옵션 줄 = 기본값과 다른 것만) · 39/39 (개발 번들, 리포트 검사 추가 전) |
| 리포트 오차 정직성 | 층별 MSE · SQNR · cos 가 그 방법의 균일 레시피와 같은 레시피 해시의 리포트에서만 옴 | `pytest -q app/tests/test_autotune_reports.py` · 번들 재생성 | 1 passed · 균일 15개 모두 해시 일치, 196층 × 15방법 = 2,940칸 채움, 다른 번들 필드는 이전과 동일 |
| 배낭 정합 | 브라우저 배낭 = `python -m app.autotune solve` (예산별 196층 선택 · 값 · 비트) | `node scripts/quantizer/autotune_check.mjs …` | v2 표 **10/10**, track M 표 **10/10** (불가능 · 너무 큰 예산의 거부 문구 포함; `a74179f` 에서 2.5 · 3 · 403/128 · 3.5 · 4 · 4.16 · 4.25 · 5 · 8 · 1 비트로 다시 10/10) |
| README "화면만 보기" | 저장소를 새로 clone (`a74179f`) → 새 가상환경 (Python 3.14.5 · 3.11) 에 README 의 6개 패키지만 설치 → `python -m app.server --demo` | curl · 위 브라우저 검사를 clone 과 그 가상환경으로 | 화면 6개 · 번들 · health 모두 200 (3.11 은 health), `--agent` 없이 해석 API 503 (의도), **AUTOTUNE_UI 42/42** (3.14). 이 설치로는 `app.autotune solve` 가 torch 를 못 찾는다 — 화면만 보기 범위 밖이며 README 의 전체 설치가 필요 |
| 정직성 | 화면 번들의 측정값이 정확한 레시피에만 붙음 (레시피 파일 SHA-256 → KL 기록 → PPL 기록) | `python -m app.export_autotune_demo …` | 균일 15 · 배낭 6 모두 대조 통과, 거부 0 |
| 실제 서버 E2E | `python -m app.server --demo --agent <loop.json>` 위의 화면 · 번들 · 실시간 해석 · 실시간 제안 | curl (로컬 `.venv`) | health 200 · 화면 200 · 번들 200 (720 KB) · **실시간 해석 6.2초 $0.029** → `tune · 403/128 · BF16 유지 [-1]` · **실시간 제안 68초 $0.28** → 1 채택 · 1 거부(같은 레시피가 이미 세션에 있음), 원본 세션 파일 수 33 → 33 |
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

- 데모 세션(`runs/20261104_demo/s3150`)은 3라운드 · 측정 8건까지 끝났다 ([증거 §2.6](./evidence/2026-11-04_lowbit_track_m.md)). AI 최선 둘의 test PPL 은 출발점과 같은 트리 · GPU 로 쟀다 (26.82 · 26.51 vs 27.04). PPL 의 보정 seed 잡음은 재지 않았다.
- 브라우저 배낭은 메인 스레드에서 돈다. track M 표에서 2.5–4.25 비트 풀이가 0.5–2.9초 걸려 그동안 화면이 멈춘다 (Web Worker 이전은 다음 작업).
- 브라우저 검사는 Google Chrome (macOS) 만 돌렸다. Safari · Firefox 는 손으로 확인하지 않았다.
- README 의 "화면만 보기" 설치는 Python 3.14 · 3.11 에서 확인했다 (3.10 · 3.12 · 3.13 · Windows 는 미확인).
- evals 는 합격선 · 절대 규칙 기대 결과 · judge 라벨이 팀 확정 전이라, 실행해도 결과를 근거로 쓰지 않는다 ([설계서 §7](../evals/EVAL_DESIGN.md)).
- 제안자가 받는 RAG 문서의 프로파일 줄에는 통계 이름만 있고 정의가 없다 (예: `outlier_channels` 가 비율이라는 사실). 진행 중인 데모 세션의 입력이 라운드 중간에 바뀌지 않도록 이번에는 고치지 않았다 — 다음 세션 전에 문서 머리에 정의를 넣는다.
