---
description: 린트와 테스트를 실행하고 완료의 정의 (AGENTS.md §6) 충족 여부를 보고한다 (수정 없음)
disable-model-invocation: true
---
파일을 수정하지 말고 다음 검사를 순서대로 실행해줘. 앞 검사가 실패해도 실행 가능한 다음 검사는 수행해.
1. `make lint` (`ruff check .`)
2. `make test` (CPU `pytest -q`). CUDA 와 triton 이 있으면 `make gpu` 도 실행한다.
3. `pytest tests/test_golden.py tests/test_compare_golden.py -q -rfs`

명령별 종료 상태와 passed / failed / skipped / xfailed 를 구분해 보고해.
골든 케이스 노드와 보조 검사 (`test_golden_manifest`, `test_golden_vectors_present`) 를 나눠 세고, skip 마다 사유를 적어.
필수 케이스의 skip 이나 xfail 은 완료로 처리하지 말 것 (CPU 에서는 GPU 전용 `e2e_bf16_passthrough_ppl` 1건만 skip 이 정상).
실패한 테스트마다 파일:줄 과 원인 한 줄, 수정 제안만 적어. 최종 수용 여부는 사람이 정한다.
