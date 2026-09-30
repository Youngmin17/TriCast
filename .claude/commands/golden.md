---
description: 골든 케이스·골든 벡터 기준으로 현재 구현을 진단한다 (수정 없음)
---
1. `tests/harness/golden_cases.yaml` 의 케이스와 `tests/data/nadpe/` 벡터를 대상으로 관련 테스트만 실행한다.
2. 실패 케이스마다 대응 AC 번호 (docs/SPEC.md §6), 기대값, 실제값, 의심 원인 한 줄을 표로 보고한다.
3. 테스트·골든 데이터는 절대 수정하지 않는다 (AGENTS.md §4). 고칠 곳은 구현 쪽 후보만 제시한다.
