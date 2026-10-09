---
description: 골든 케이스·골든 벡터 기준으로 현재 구현을 진단한다 (진단만, 수정 없음)
disable-model-invocation: true
---
`pytest tests/test_golden.py tests/test_compare_golden.py -v -rfs` 를 실행하고, `tests/harness/golden_cases.yaml` ·
`tests/harness/golden_compare.yaml` 및 구현과 결과를 대조해줘.
명령을 실행하지 못했다면 그 이유를 보고하고 통과로 추정하지 말 것.
어떤 소스, 테스트, 케이스, 골든 벡터 (`tests/data/nadpe/`) 또는 설정 파일도 수정하지 말 것 (AGENTS.md §4).
케이스별로 대응 AC (`docs/SPEC.md` §6), 통과/실패, 판정 근거, skip 사유를 표로 보고해.
실패가 있으면 기대값, 실제값, 고칠 구현 위치와 이유만 제안해. 실제 수정은 별도의 위임으로 수행한다.
