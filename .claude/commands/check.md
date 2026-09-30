---
description: 린트와 테스트를 실행하고 결과를 요약한다 (수정 없음)
---
1. `ruff check .` 을 실행하고 경고 수를 센다.
2. `pytest -q` 를 실행한다 (CPU). CUDA 와 triton 이 있으면 `pytest tests/gpu -q` 도 실행한다.
3. 실패한 테스트마다 파일:줄 과 원인 한 줄을 적는다. 이 명령에서는 코드를 고치지 않는다.
