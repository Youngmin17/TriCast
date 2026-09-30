# 위임 프롬프트와 검증 루프 기록 (강의 5)

> 이 저장소의 코드는 사람이 정한 계약(`docs/design/ENGINE.md`, `AGENTS.md`)을 기준으로 AI 코딩 에이전트에게
> 위임해 만들었다. 이 문서는 무엇을 어떻게 위임했고, 결과를 무엇으로 판정했는지를 사실대로 남긴다.
> 원본 프롬프트: `docs/prompts/wave1/` (공통 규칙 `00_common.md` + 레인별 `lane_*.md`).

## 위임 구조 — Wave 1 (2026-09-30)

| 레인 | 위임한 것 | 담당 | 판정 기준 (위임 전에 정함) |
|---|---|---|---|
| Q | 양자화 레퍼런스 · QTensor · API · observer | Codex | 레퍼런스 테스트 + microxcaling 오라클과 비트 일치 |
| M | MMA 누산 레퍼런스 (FDA/CoFDA/GDFS …) | Codex | 손으로 유도한 NADPE 사례 · 불변식 테스트 |
| KQ | Triton 양자화 커널 | Codex | GPU 에서 레퍼런스와 비트 일치 (`tests/gpu`) |
| KM | Triton MMA 커널 | Codex | GPU 에서 레퍼런스와 비트 일치 (`tests/gpu`) |
| A | 변환 (Hadamard · SmoothQuant · AWQ) · GPTQ | Codex | 불변식 · GPTQ 오차 < RTN 오차 |
| I | Recipe · HF 패치 · calibrate · PPL · lm-eval · CLI | Codex | tiny 모델 CPU 테스트 · 레시피 스키마 검증 |
| D | Qwen3-0.6B 데모 | Codex | 문법·`--help` (실행은 GPU 에서 사람이 확인) |
| — | 형식·반올림·명세·레퍼런스 캐스트·계약서 | Claude | torch 네이티브 캐스트와 비트 일치 |

**왜 이렇게 나눴나**
- 명세 계층(형식·반올림·QuantSpec·MMASpec)과 계약서를 먼저 고정하고 커밋했다(`64c10e3`). 모든 레인이 같은
  어휘와 시그니처로 병렬 작업하려면 경계가 먼저 있어야 한다.
- 레인마다 **소유 파일**을 지정하고, 공유 명세 파일 수정을 금지했다. 명세에 문제가 있으면 고치지 말고
  `CONCERNS` 로 보고하게 했다 — 기준을 에이전트가 바꾸면 기준으로 에이전트를 판정할 수 없다.
- 모든 레인에 같은 완료 형식(`CODEX_DONE / FILES / TESTS / CONCERNS`)을 요구했고, 보고를 그대로 믿지 않고
  사람 쪽에서 테스트를 다시 실행해 판정했다.

## 검증 루프에서 실제로 일어난 일

### 1. 7개 레인이 수 초 만에 전부 "완료" — 실제로는 전부 실패
- 관찰: 백그라운드 작업 7개가 수 초 안에 모두 종료. 종료 코드만 보면 성공처럼 보였다.
- 로그 확인: 모든 레인이 `The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account`
  로 즉시 종료 (`EXIT rc=1`). 로컬 설정의 기본 모델이 계정에서 막혀 있었다.
- 조치: 최근 세션 기록에서 실제로 쓰인 모델을 찾아 1줄 프로브(`PROBE_OK`)로 확인한 뒤 `-m gpt-6-astra` 로 재투입.
- 교훈: 완료는 종료 코드가 아니라 로그의 완료 마커로 판정한다. 대량 위임 전에 1줄 프로브를 한다.

### 2. 보호된 파일의 결함을 에이전트의 테스트가 잡았다 (Lane Q)
- Lane Q 보고: `476 passed, 9 failed` — 실패 9건은 모두 Claude 가 작성한 공유 파일 `reference/cast.py` 의 결함
  (unsigned 정수 격자의 −0, E8M0 에 +Inf 입력, `decode()` 의 범위 검사 누락).
- 에이전트는 소유권 규칙대로 공유 파일을 고치지 않고 `CONCERNS` 로 보고했다. 테스트를 결함에 맞게 느슨하게
  바꾸지도 않았다 (AGENTS.md 금지 사항 1·2 가 지켜진 사례).
- 사람 쪽 조치: 결함 3종을 수정 → 전체 CPU 스위트 `812 passed`.

### 3. 명세 간 불일치 보고
- Lane A: 명세 docstring 은 RHT 를 `H·D`, 계약서는 `D·H` 로 적고 있었다 → 계약서를 따랐다고 보고 → docstring 수정.
- Lane M: NADPE 원본의 fp32 서브노멀 누산기 정규화가 한 비트 더 시프트한다(도달 불가능한 경로). 계약서의
  정확한 정규화를 따르고 차이를 보고. E8M0 field-0 규칙의 적용 범위가 계약서에 group 레벨로만 적혀 있었음 →
  NADPE 소스 확인 후 product 레벨까지 계약서를 명확히 함.

### 4. GPU 에서만 드러난 결함 — 레퍼런스 쪽 버그였던 경우 포함 (KQ · KM)
- Triton 양자화 커널 첫 GPU 실행: 579 중 551 통과. 실패 28건은 두 원인으로 갈렸다.
  - Triton 이 libdevice 를 flush-to-zero 로 링크한다 (`set_nvvm_reflect_ftz`) → 스케일이 fp32 서브노멀(E8M0 2^-127)이면
    0/0 = NaN. 1M 개 나눗셈 프로브로 `tl.math.div_rn`(FTZ), inline PTX `div.rn.f32`(IEEE, 불일치 0) 를 비교해 확정 →
    스케일·epilogue 의 fp32 연산을 IEEE PTX 헬퍼(`kernels/ieee.py`) 로 교체.
  - 스케일 1 ulp 차이는 **레퍼런스 쪽** 문제였다: PyTorch CUDA 의 `tensor / 스칼라` 는 역수 곱셈이라 정확히 반올림되지
    않는다 (CPU 는 정확) → 레퍼런스가 장치마다 다른 값을 냈다. fp64 계산 후 fp32 1회 반올림(53 ≥ 2·24+2 라 이중 반올림
    무해)으로 바꿔 장치 무관하게.
  - 수정 후 579/579 비트 일치.
- Triton MMA 커널: 컴파일 오류 — constexpr 튜플을 인덱싱한 값이 포장이 풀린 파이썬 튜플로 중첩 함수에 전달되어
  `'int' object has no attribute 'type'`. 구성 요소를 단독 커널로 이분 탐색해도 재현되지 않아, Triton 코드 생성기의
  `call_JitFunction` 을 감싸 인자 타입을 출력해서 원인을 찾았다 → `tl.constexpr(...)` 로 재포장. 수정 후 149/149.

### 5. 독립 구현으로 레퍼런스를 판정 (AC7)
- 계약서만 보고 만든 레퍼런스를 NADPE 원본 CUDA 커널(수정 없이 단독 빌드)과 비교: FP8 1200 + FP4 516 = 1716 케이스
  전부 비트 일치. 오라클 벡터는 결정론 2400회 재실행과 독립 numpy 모델로 먼저 검증했다.

### 6. 적대적 검토 (Wave 3) 와 네트워크 단절
- Codex 적대적 검토: CRITICAL 0 / MAJOR 15 / MINOR 1, 모두 file:line 과 재현 입력 포함 (예: 구조화 출력이 지원하지 않는
  `minimum`·`maxItems` 를 보내 실제 API 에서 400, 새 프로세스의 `lm_eval --model tricast` 미등록, 보정 없이 forward 하면
  SmoothQuant/GPTQ 가 조용히 생략). 테스트는 모두 통과하던 상태였다 — 테스트가 덮지 않는 경로를 찾게 지시한 효과.
- Claude 4관점 검토 Workflow 는 네트워크 단절(ENOTFOUND)로 에이전트 4개가 모두 실패 → 코드가 바뀐 Wave 4 뒤에 재실행.
- 같은 시각 원격 GPU 작업을 기다리던 ssh 가 끊겼는데, 대기 명령 끝의 `; true` 때문에 성공(exit 0)으로 보였다 →
  원격 로그의 완료 마커로 판정해 작업이 살아 있음을 확인.

### 7. 성능 결함은 정확성 테스트가 잡지 못한다
- Qwen3 데모가 47분 동안 CPU 100% · GPU 0% — Triton MMA 커널이 M·N·K 를 `tl.constexpr` 로 선언해 행렬 크기마다
  재컴파일하고 있었다 (LLM 평가는 배치마다 시퀀스 길이가 다르다). 정확성 테스트 149개는 모두 통과한 상태였다.
  → 크기를 런타임 인자로, autotune 키는 M 을 2의 거듭제곱 버킷으로, 가중치 K-major 패킹을 한 번만.

## 결과 요약 (Wave 1)

| 레인 | 로컬 판정 (사람이 재실행) | 비고 |
|---|---|---|
| M | 129 passed, ruff 0 | |
| A | 111 passed, ruff 0 | |
| Q | 476 passed / 9 failed → 공유 파일 수정 후 통과 | microxcaling 오라클 포함 |
| KQ | 로컬 skip (GPU 전용) | GPU 판정: 진행 중 |
| KM, I, D | 진행 중 | |
