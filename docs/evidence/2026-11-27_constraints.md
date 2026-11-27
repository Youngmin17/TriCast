# README 설치 버전 고정 (`constraints.txt`) · 2026-11-27

README의 설치 명령이 평가 당일에도 같은 버전을 깔도록 [`constraints.txt`](../../constraints.txt)를 만들고, 빈 가상 환경에
README 명령 그대로 설치해 게이트를 다시 돌린 기록이다. 코드는 `e04e29e`와 같다 (이 기록의 커밋은 문서와 고정 파일만 바꾼다).

## 1. 만든 방법

- **기준 버전:** `Makefile`의 `install-cpu` — UCL CPU 전체 테스트가 통과한 torch 2.8.0 · transformers 4.55.2 · lm_eval 0.4.13 ·
  datasets 4.8.5 · ruff 0.16.9.
- **생성:** UCL bath (Xeon Gold 6126, AVX-512), Python 3.11.15, pip 24.0, 깨끗한 트리 `e04e29e`. 빈 가상 환경에 CPU용 torch 2.8.0과
  `.[eval,dev,app,onnx]`를 위 버전으로 설치하고 `pip freeze --exclude-editable`로 102개를 적었다 (torch의 `+cpu` 표기만 뺐다).
  여기에 HQQ 어댑터가 고정한 `hqq==0.2.8.post1`(`src/tricast/weight_quant/hqq.py`의 `HQQ_VERSION`)과 그 의존성
  einops 0.8.2 · termcolor 3.3.0을 같은 방식으로 설치해 확인한 버전으로 더했다. 모두 105개, SHA-256 `d831e73e…8a60`.
- **Python 3.11에서 만든 이유:** 처음 Python 3.12에서 만든 목록은 `numpy==2.5.5`를 담았고, 이 버전은 Python 3.11 배포본이 없어
  macOS 3.11 설치가 `ResolutionImpossible`로 실패했다. 지원 범위(README: 3.11 · 3.12)의 가장 낮은 버전에서 다시 만들었다.
  torch 2.8.0은 Python 3.14를 지원하지 않는다.

## 2. 빈 환경 검증

설치는 README 명령 `pip install -c constraints.txt -e ".[eval,dev,app]"` (기본 패키지 색인)을 그대로 썼다. Linux의 torch 2.8.0은
CUDA 빌드라 nvidia 라이브러리 14개와 triton을 함께 받는데, 이것들은 torch가 정확한 버전으로 요구하므로 목록에 적지 않았다.

| 환경 | 설치된 버전 = 고정 버전 | ruff | 골든 3파일 | 앱 테스트 | README `make check` (hqq + 스레드 4) |
|---|---|---|---|---|---|
| Linux bath · Python 3.11.15 | 차이 0 | 경고 0 | 111 통과 · 1 skip | 585 통과 · 6 skip | 2,565 통과 · 903 skip · 실패 0 |
| Linux bath · Python 3.12.13 | 차이 0 | 경고 0 | 111 통과 · 1 skip | 585 통과 · 6 skip | 2,565 통과 · 903 skip · 실패 0 |
| macOS arm64 · Python 3.11.15 | 차이 0 (97개) | — | 111 통과 · 1 skip | — | 실행하지 않음 (아래) |

골든의 skip 1건은 GPU 전용 `e2e_bf16_passthrough_ppl`이다. `make check`의 skip 903건 중 884건은 CUDA GPU 전용이고, 나머지는
README 설치에 넣지 않은 선택 패키지(bitsandbytes, onnx, microxcaling)다. onnx와 microxcaling 테스트는 모듈 단위로 건너뛰어
1건씩 세므로, 선택 패키지를 경로로 넣어 돌린 2026-11-26 클러스터 실행(2,679 통과 · 899 skip)보다 모인 테스트 수가 적다.

macOS에서는 README의 다른 두 경로도 확인했다.

- **면담 명령:** 서버를 README대로 띄우고 요청 한 건을 보냈다. HTTP 200, `mode: replay`, `request.outcome: needs_clarification`
  (질문 2개), 응답의 `X-Request-ID`(`q-cbe9a427…`)가 `logs/trace.jsonl` 마지막 줄의 `request_id`와 같았다 (상태 확인 포함 2줄 추가).
- **MVP 최소 설치:** `pip install -c constraints.txt fastapi uvicorn pillow numpy jsonschema pyyaml` (torch 없음)으로 띄운 서버의
  `/tune.html`이 200이었다. Python 3.14.5에서도 같은 설치가 고정 버전 그대로(22개, 차이 0) 되고 `/tune.html`과 요청 API가
  200이었다. 이 목록의 numpy 2.4.6은 Python 3.11 이상이라 README의 최소 요구를 3.10에서 3.11로 올렸다.

macOS 전체 CPU 테스트는 로컬 부하 때문에 돌리지 않았고, 아래 실패와 관련된 두 파일(`tests/test_lowbit_methods.py`,
`tests/test_weight_methods.py`)만 hqq 설치 후 돌려 55 통과 · 3 skip(CUDA 전용)이었다.

## 3. 처음 실행에서 나온 실패 5건

hqq 없이, 스레드 3으로 돌린 첫 CPU 전체 실행은 두 Python 버전 모두 5 실패 · 2,554 통과 · 909 skip이었다. 두 원인 모두 이번 변경
전부터 있던 조건이며, 2026-11-26 클러스터 실행은 스레드 4와 경로에 넣은 hqq로 통과했다.

- **GPTQ 3건** (`test_a_low_bit_gptq_layer_applies_like_gptq_int4[4|3|2]`): `gptq`는 Hessian이 비트 단위로 대칭이어야 받고
  (`torch.equal(H, H.T)`), 테스트는 보정 행렬이 `2 XᵀX / n`과 비트 단위로 같기를 요구한다. bath의 MKL(AVX-512)에서 float64
  `XᵀX`(256 × 256)를 스레드별로 쟀다.

  | 스레드 | 1 | 2 | 3 | 4 | 12 | 24 (기본) |
  |---|---|---|---|---|---|---|
  | GPTQ 3건 | 통과 | 통과 | 실패 | 통과 | 실패 | 실패 |
  | `XᵀX` 최대 비대칭 | 0 | — | 1.99×10⁻¹³ | 0 | — | 2.42×10⁻¹³ |

  macOS(기본 8스레드)에서는 통과했다. 그래서 README의 전체 테스트 명령은 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`를 준다.
- **HQQ 2건** (`test_a_patched_layer_uses_the_replaced_weight[hqq3|hqq2]`): 테스트가 `kind == "hqq"`일 때만 hqq가 없으면
  건너뛰어, 3비트 · 2비트 사례는 hqq 없이 실행되고 `PackageNotFoundError`로 끝난다. 테스트 파일은 AGENTS.md §4의 보호 대상이라
  바꾸지 않았다. README는 전체 테스트 전에 `hqq==0.2.8.post1`을 설치하게 하며, 설치하면 `tests/test_weight_methods.py`가
  20 통과 · 3 skip이다.

남은 결정: 두 조건을 저장소 안에서 없애려면 테스트의 skip 조건(`kind.startswith("hqq")`)과 `Makefile` 테스트 대상의 스레드
고정이 필요하다. 둘 다 사람이 승인하는 파일이다.

## 4. 재현

스크립트와 로그: 로컬 `runs/20261127T0243Z_constraints/` (git 제외), 클러스터 `/scratch/uceeeee/tricast/lock_20261127T0243Z/`.

```bash
# 생성 (Python 3.11, 빈 가상 환경)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[eval,dev,app,onnx]" transformers==4.55.2 lm_eval==0.4.13 datasets==4.8.5 ruff==0.16.9
pip freeze --exclude-editable | sed -E 's/^(torch==[0-9.]+)\+cpu$/\1/' > constraints.txt   # 그 뒤 hqq 세 줄을 더했다
# 검증 (새 가상 환경, README 명령)
pip install -c constraints.txt -e ".[eval,dev,app]"
pytest tests/test_golden.py tests/test_compare_golden.py tests/test_tune_golden.py -q -rs
pip install -c constraints.txt hqq==0.2.8.post1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 make check
make test-app
```
