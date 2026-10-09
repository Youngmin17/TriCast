# 전체 검증 기록 — 2026-10-09

README · SPEC 의 테스트 수치가 나온 실행이다. 두 번 돌렸다: 실행 i 는 전체 (린트 · CPU · GPU · 웹 앱), 실행 i2 는 그 뒤에
근거 표 검사 2건 (`tests/test_ontology.py`) 을 더한 최종 트리의 린트 · CPU 다. 두 실행 사이에 코드와 GPU 테스트는 바뀌지 않았다.
최종 트리는 이 파일을 담은 커밋과 같다 (수치 문장 제외).

| 항목 | 값 |
|---|---|
| 호스트 · GPU | UCL geneva, NVIDIA A100-SXM4-80GB (driver 595.71.05), `gpu_cap.sh` 가 GPU 0 을 배정 |
| 시각 | 실행 i 2026-10-09 06:06:39 UTC, 실행 i2 06:15:40 UTC |
| 소프트웨어 | torch 2.8.0+cu128, triton 3.4.0, transformers 4.55.2 (conda env `tricast`) |
| 원본 로그 | 클러스터 `/scratch/uceeeee/tricast/logs/audit_20261009i.log`, `audit_20261009i2.log` |

| 단계 | 명령 | 결과 (로그 원문) |
|---|---|---|
| 린트 | `ruff check .` | `All checks passed!` (i, i2) |
| CPU 스위트 (GPU 숨김) | `CUDA_VISIBLE_DEVICES= pytest -q --ignore=tests/gpu` | i2: `2274 passed, 2 skipped, 19 warnings in 177.71s` |
| GPU 스위트 + 골든 | `pytest -q -rs tests/gpu tests/test_golden.py` | i: `945 passed, 1 skipped, 1 warning in 255.17s` |
| 웹 앱 | `pytest app/tests -q` (fastapi · uvicorn · httpx · pillow 설치) | i: `161 passed, 1 warning in 40.73s` |

skip 사유 (로그 원문):

- CPU: `tests/test_microxcaling_parity.py:14: could not import 'mx'` (오라클 패키지 미설치), `tests/test_golden.py:309: needs a CUDA GPU with triton` (GPU 를 숨긴 실행)
- GPU: `tests/gpu/test_triton_mma.py:362: set TRICAST_MMA_PERF=1` (선택 성능 측정)
