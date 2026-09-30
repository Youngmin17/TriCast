
[TriCast Wave 1 — 모든 레인 공통]
제품: TriCast = Triton 기반으로 임의 수 형식·양자화 방식·MMA(텐서코어) 누산 산술을 CUDA core 에서
bit-exact 하게 에뮬레이트하고, Hugging Face 모델 + lm-eval 로 평가하는 도구. 하드웨어 설계자가
정밀도 조합과 누산 알고리즘을 쉽게 정의해 모델 품질 영향을 빠르게 보는 것이 목표.
레포: /Users/mincho/Documents/AI캡스톤/TriCast (branch develop, 기초 커밋 64c10e3). 패키지 src/tricast.

먼저 정독 (계약 — 절대 기준, 모호하면 이것을 따른다):
  docs/design/ENGINE.md   ← 수치 의미론 · API · 레인 소유권 (§8)
  src/tricast/formats.py, rounding.py, quant/spec.py, mma/spec.py, mma/operand.py, reference/cast.py
  tests/conftest.py (wide_fp32, bit_equal 헬퍼; gpu 마커 자동 skip)
참고 코드 (읽기 전용):
  NADPE MMA-Emu CUDA: "/Users/mincho/Documents/AI캡스톤/micro26-ae-main 2/csrc/quantization/mma_emu"
     (core/accumulator.cuh, core/fp32_utils.cuh, core/gdfs_group.cuh, formats/*.cuh, gemm/*.cuh)
  microxcaling: /Users/mincho/Documents/AI캡스톤/refs/microxcaling (mx/mx_ops.py, mx/elemwise_ops.py,
     mx/formats.py) — PYTHONPATH 에 넣으면 `import mx`

규칙:
  - ENGINE.md §8 의 내 레인 소유 파일만 생성/수정한다. 공유 명세 파일(formats.py, rounding.py,
    quant/spec.py, mma/spec.py, mma/operand.py, reference/cast.py, ENGINE.md, tests/conftest.py,
    pyproject.toml, src/tricast/__init__.py)은 수정 금지 — 문제를 찾으면 CONCERNS 에 적는다.
  - 다른 레인이 동시에 작업 중이다. 아직 없는 모듈은 ENGINE.md 계약의 이름·시그니처대로 import 해
    쓴다. 그 모듈이 없어 로컬에서 못 도는 테스트는 pytest.importorskip 로 처리하되, 코드는 그 모듈이
    생긴다는 전제로 올바르게 작성한다. 작업 끝무렵 해당 파일이 생겼으면 실제로 돌려 통과시킨다.
  - 커밋·푸시·브랜치 생성 금지. ~/.claude 수정 금지. SSH·원격 GPU 사용 금지.
  - 로컬 = macOS, CUDA/triton 없음. 파이썬은 반드시
    /Users/mincho/Documents/AI캡스톤/TriCast/.venv/bin/python (torch 2.11 CPU, transformers 4.55.2,
    lm_eval 0.4.9.1, datasets 3.6.0, pytest, hypothesis, ruff 설치됨). 테스트는 작은 텐서로 수 초~수십 초.
    무거운 CPU 작업·대용량 다운로드 금지 (HF 모델 다운로드 금지 — tiny 모델은 config 로 직접 생성).
  - 테스트 실행: cd /Users/mincho/Documents/AI캡스톤/TriCast &&
    PYTHONPATH=src:/Users/mincho/Documents/AI캡스톤/refs/microxcaling .venv/bin/python -m pytest <파일> -q
    린트: .venv/bin/python -m ruff check <파일들>
  - 스타일: 주변 코드와 동일 (타입 힌트, 짧은 docstring, 과한 추상화·장황한 주석·죽은 코드 금지,
    ruff E,F,W,I,B,UP / line 110).
  - bit-exact 계약: 레퍼런스는 정확 연산(fp64/int64/Fraction)만 쓴다. 테스트의 수치 비교는 == (NaN==NaN)
    로 한다 (의미상 근사가 맞는 곳 — 예: 변환 불변식 — 만 허용 오차를 쓰고 이유를 적는다).
  - 완료 시 마지막에 정확히 아래 4줄을 출력하고 종료:
    CODEX_DONE: <한 줄 요약>
    FILES: <생성/수정 파일 목록>
    TESTS: <pytest 최종 요약 줄 그대로>
    CONCERNS: <계약 모호점·명세 버그·미완료 (없으면 none)>
