# 위임 기록 — Quantizer 개편 (2026-10-13)

> 이 기록은 지휘 Claude Code 세션이 작성한 초안이다. 팀 검토 · 추인은 아직 받지 않았다.

Quantizer 개편(브랜치 `quantizer_develop`)에서 하위 에이전트에게 넘긴 일, 사람이 직접 정한 것, 위임 결과를 지휘 쪽이 다시 돌려
확인한 것을 남긴다.

| 역할 | 누구 |
|---|---|
| 방향 · 기준 확정 · 수용 판단 | Youngmin17 |
| 지휘 · 검증 실행 | 메인 Claude Code 세션 (Youngmin17 지휘) |
| 위임 받은 쪽 | Claude Code 하위 에이전트 (`omnix-worker`), 위임마다 새 세션 · 커밋 금지 |

## 1. 사람이 정한 것

| 결정 | 내용 | 이유 |
|---|---|---|
| 방향 | 칩 계산 흉내 → 레이어별 양자화 방법 선택 도구(Quantizer)로 포지셔닝, NVINT = NVFP4 구조 + INT 원소, 발표 자료에서 MMA 내용 제거 | 사용자 지시 |
| 테스트 위치 | 새 테스트는 `app/tests/` 에 **새 파일로만** 추가 (기존 테스트 파일은 수정 금지) | AGENTS.md §4-1 — `app/tests/` 변경은 사람 승인 |
| 커밋 | 로컬 커밋만, push 없음 | 사용자 선택 |
| 서버 GPU 실행 검증 | 지금은 하지 않고 "미검증"으로 표시 | 사용자 선택 → SPEC AC20 · §10 R7 |

## 2. 위임한 일 — Quantizer 화면 옵션 · 보정 · 테스트 이관

- **태스크**: 방법별 옵션(그룹 크기 · 반올림 · 스케일 방법 · 변환 값 · GPTQ 값 · observer), KV 잔여 창, 보정 설정을 Quantizer 화면 ·
  Python 생성기 · 서버 · 러너에 넣고, 저장소 밖에 있던 검사들을 `app/tests/` 와 `scripts/quantizer/` 로 옮긴다.
- **성공 기준 (위임할 때 준 것)**: ① 기존 프리셋 5종의 YAML 이 바이트 단위로 그대로 ② 브라우저 JS 와 Python 이 옵션 포함 무작위
  상태에서 같은 YAML ③ 모든 옵션 필드가 해석된 레이어 명세에 도달 ④ 해당 없는 옵션 · 범위 밖 값은 거부 ⑤ `pytest app/tests` 통과,
  `ruff check .` 무경고 ⑥ SPEC 인용용 테스트 경로 · 함수 이름 보고.
- **위임하지 않은 것**: `docs/` 수정, 테스트 기준의 결정(무엇을 같은 레시피로 볼지, 기록된 결과를 언제 붙일지), 커밋.

## 3. 지휘 쪽 재검증 — 보고 값과 다시 돌린 값

| 검사 | 위임 쪽 보고 | 지휘 쪽 재실행 (2026-10-13) |
|---|---|---|
| `pytest app/tests -q` | `308 passed` | `308 passed, 1 warning` ✅ |
| `ruff check .` | `All checks passed!` | `All checks passed!` ✅ |
| `scripts/quantizer/qz_check.mjs` (헤드리스 Chrome) | `QZ_CHECKS[static] 71/71` | 첫 실행은 **지휘 쪽 실수**로 실패 (보조 루트를 상대 경로로 만들어 링크가 깨짐) → 절대 경로로 다시 만들어 `71/71 passed` ✅ |
| `scripts/quantizer/qz_tricast_accept.py` | `QZ_TRICAST_ACCEPT_OK` | `recipes 100, layers checked 19600 …` `QZ_TRICAST_ACCEPT_OK` ✅ |
| `scripts/quantizer/cdp_check.mjs` | `CDP_CHECKS 44/44` | `44/44 passed` ✅ |
| SPEC 인용 테스트 22개 | 이름 목록 | 22개 모두 `def` 존재 확인 ✅ |
| 프리셋 YAML 5종 | 바이트 동일 | `cmp` 동일 ✅ |

## 4. 검증에서 나온 결함과 처리

| 결함 | 어떻게 찾았나 | 처리 |
|---|---|---|
| 칸의 블록 형식 명세가 기본값과 깊게 합쳐지며 다른 방법의 필드(GPTQ 의 `zero_point`)를 물려받음 | 지휘 쪽 검증 (레이어마다 해석된 명세를 그 방법의 단독 명세와 대조) — 고친 뒤 무작위 격자 200개로 다시 확인 | 칸에는 `scheme` 을 가진 완전한 명세만 쓴다 → SPEC AC16 |
| 블록 형식 레시피가 `mma` 를 생략하면 기본 누산(cofda F23) + `scaled` 경로라 매우 느림 (PPL 1회 약 3시간으로 추정) | W2 실행 진행률 관찰 (로그 미보관) → 4창 probe 로 확인: dequant 44초 · 프리셋 지정 54초 · 기본 cofda F23 약 1,110초 (`runs/20261013_quantizer_w3/qz_w2_probe.log`) | 품질 실험용 조각은 `mma_input: dequant` → SPEC §5-4 · R4 |
| KV `mode: cache` 의 PPL 이 토큰 단위로 돌아 매우 느림 (진행률로 잡은 추정 약 15시간) | W2 실행 진행률 관찰 (로그 미보관) | Quantizer 의 KV 는 `mode: fakequant` → SPEC R4 |
| 데모 모드 서버가 생성기 모듈을 처음부터 불러와 기존 테스트 `test_demo_mode_never_imports_compute_modules` 실패 | 위임 쪽이 기존 테스트로 발견 | 테스트는 고치지 않고 서버 쪽 import 를 라이브 경로 안으로 옮김 |
| 효과가 관측되지 않은 옵션 2개 (INT8 + `mse` 스케일, GPTQ `block_size`) | 위임 쪽 tiny 모델 실행 | 숨기지 않고 보고에 남김 — `block_size` 는 수학적으로 결과가 같아야 정상, INT8 + `mse` 는 tiny 모델에서 최적 비율이 1.0 |

## 5. 사람의 수용 판단 (대기)

- 위 재검증 결과와 SPEC AC15–AC21 의 정의를 팀이 검토한 뒤 추인한다.
- 서버 GPU 실행(AC20 의 남은 부분)은 사용자가 정할 때 GPU 서버에서 프리셋 1개를 기록과 대조해 판정한다.
