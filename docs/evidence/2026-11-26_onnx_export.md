# 2026-11-26 — 양자화 가중치의 실제 저장 비트폭

같은 모델의 세 가지 질문을 따로 확인한다. (1) 저장된 코드가 실제로 몇 바이트인가,
(2) 그 코드를 ONNX Runtime에서 풀면 TriCast가 곱하는 가중치와 같은가,
(3) 서로 다른 행렬곱 구현까지 포함한 모델 출력은 얼마나 달라지는가.
(1)·(2)가 (3)이나 칩의 처리 속도를 보증하지 않는다.

## 실행 조건

- Git 기준: `quantizer_develop`, `8e9066c02c1c12691998267507b028c87eb94d73` + 작업 트리.
  코드 해시·명령·로그: 로컬 `runs/20261126T0142Z/export/`;
  UCL `/scratch/uceeeee/tricast/onnx_20261126T0142Z/`. CPU 전용, 예약은 `cpu_cap.sh`.
- Python 3.12.14, PyTorch 2.8.0, transformers 4.55.2, ONNX 1.19.1.
  표준 그래프·검출기 대조에 ORT 1.23.2도 사용했다. 1.23.2 CPU는 2비트 MatMulNBits를 실행하지 못해
  범용 exporter의 ONNX extra는 1.25.1 이상으로 지정했다. 설치는 기존 공용 환경 밖에서 했다.
- 표준 그래프: opset 21. 선택된 각 층에 대해 코드 initializer의 물리 크기와
  최적화를 끈 ORT decode 출력을 검사한다. 양자화되지 않은 일반 가중치, 스케일, 영점,
  LUT 및 그래프가 따로 있으므로 코드 비트폭을 파일 전체 평균 비트폭이라고 부르지 않는다.
- 공개 데모용 Qwen 체크포인트 revision `c1899de289a04d12100db370d81485cdf75e47ca`;
  [검증했던 원 패키지](2026-11-13_model_packages.md) `model.safetensors` SHA-256
  `6fc8b3413fc275e112f7669f3797873ea165e23aef2a4d06128d36252e3bd29b`.
- 출력 비교는 양쪽 모두 같은 입력·마스크의 다음 토큰과 logits를 대조했다.
  진단의 엄격한 수치 기준은 기존 작은 Llama 통합 테스트의 cos > 0.99999 및 max_abs < 0.001이다.
  작은 모델의 허용치를 실모델 품질 척도로 간주하지 않으며, 실패를 지우기 위해 넓히지 않는다.

## 재현

```bash
# HQQ 원본 코드가 없던 기존 4층을 BF16으로 유지하는 경우에만 이 플래그를 쓴다.
python -m tricast.onnx_export from-pack --package /path/to/q06b_act_damp05_r3 \
  --allow-dense-fallback --out deploy/qwen3-06b-20pct.onnx
python -m tricast.onnx_export run --onnx deploy/qwen3-06b-20pct.onnx \
  --tokenizer /path/to/q06b_act_damp05_r3 --prompt "A triangle has" --max-new-tokens 2
# 출력은 빈 경로에만 만든다. 검사 입력이 정해진 로컬 HF 모델은 토크나이저 없이 대조할 수 있다.
python scripts/quantizer/onnx_export_check.py --model /path/to/hf-checkpoint \
  --input-ids 1,2,3,4,5 --input-ids 7,9,12 --max-new-tokens 3 --out runs/local-check
# YOLO: solutions.json이 있는 새 작업 폴더를 사용하고, 같은 배정으로 얻은 기존 COCO 결과를 넘긴다.
python scripts/quantizer/yolo_tune.py onnx --work /path/to/work --more /path/to/coco-evals \
  --setups auto_iso_u3 --threads 16 --workers 2
```

## 실제 파일

**Qwen3-0.6B · 20% 예산 레시피.** 표준 ONNX 한 파일은 505,892,145 B다(토크나이저 제외).
입력 임베딩과 출력 헤드가 같은 체크포인트 텐서인지 전치까지 대조하고 BF16 initializer를 공유했다.
미양자화 체크포인트의 매핑 누락은 0개다. 앞선 FP32 추적본은 같은 레시피인데도 1,427,900,279 B였다.
두 수치의 차이는 정밀도를 더 낮춰 얻은 것이 아니라 중복 저장을 없애고 원래 dtype을 보존한 결과가 대부분이다.

| 저장 표현 | 층 수 |
|---|---:|
| 2비트 코드열 | 28 |
| 3비트 코드열 | 118 |
| UINT4 / NF4 | 33 / 13 |
| BF16 예외 (HQQ 원 코드 없음) | 4 |

192개 층의 가중치 436,207,616개에 쓴 순수 코드 공간은 **162,267,136 B**다.
예외를 포함해 **196/196**개 층에서 복원 피연산자가 일치했다(0의 부호 제외).
부호화되지 않는 HQQ는 기본 호출에서 실패한다. 정상 파일을 덮어쓰거나 패키지 설정 한 바이트가 변조된 경우도 거부했다.
별도 3×17 Linear의 51개 INT3 코드는 20 B로 저장돼 마지막 묶음에 불필요한 바이트를 더하지 않았다.

## 반올림 대조

고정 시드로 만든 작은 Qwen3 구조(학습된 모델의 정확도 근거가 아님)에서 70개 층에 INT8·INT4·INT3·MXFP4·NF4·RaZeR를
섞었고, 70/70개 층이 복원 일치했다. 동일 체크포인트와 토큰 두 묶음에서 층의 출력 형식만 바꿨다.

| 층 출력 | 순수 코드 | 다음 토큰 일치 | 6개 비교 중 가장 큰 logits 절대 차이 | 기존 수치 게이트 |
|---|---:|---:|---:|---|
| BF16 반올림 | 880,640 B | 6/6 | 0.00217404 | 실패 |
| FP32 (반올림 대조군) | 880,640 B | 6/6 | 0.000000574 | 통과 |

두 경우 모두 내보낸 모델을 새 ORT 세션에서 열고 추적 때와 다른 길이로 생성했다. 이 대조는
연산 순서의 작은 차이가 중간 반올림을 거치며 커질 수 있다는 설명과 일치한다. FP32 대조군의 성공이
BF16 레시피의 실패를 상쇄하지 않는다. 실기기 속도·전체 PPL·Qwen3-8B ONNX는 이 실행의 측정 대상이 아니다.

## Qwen 출력 대조

두 프롬프트에서 각각 2토큰을 생성했다. 길이 3·4·12·13인 입력의 다음 토큰은 **4/4 동일**했지만,
FP64로 집계한 logits cosine은 0.999958–0.999965, 최대 절대 차이는 **0.14818**이었다.
따라서 파일의 가중치 저장·복원은 통과, **엄격한 logits 일치는 실패**다. 긴 답의 동일성이나 전체 PPL은 평가하지 않았다.

같은 입력을 주되 문맥을 늘린 별도 대조에서는 **508개 위치 중 506개(99.61%)**에서 두 모델이 같은 다음 토큰을 골랐다.
WikiText-2 test를 토큰화한 뒤 네 사분위점부터 128토큰씩 사용했다(시작 0·74,752·149,504·224,256).
실제 목표 토큰의 평균 음의 로그가능도는 패키지 4.19117, ONNX 4.19090이었지만, 세 번째 구간의 두 위치에서
최상위 토큰이 달랐고 전체 logits의 최대 차이는 0.40538이었다. 이는 같은 정답 문맥을 넣은 표본 대조이며
자유 생성 일치율, 정답률 또는 전체 WikiText-2 PPL이 아니다. 입력 해시와 구간별 점수는 `heldout_argmax.json`에 있다.

같은 3토큰 입력에서 모델에 마스크 래퍼만 붙였을 때 최대 차이는 0이었다. PyTorch에서
패키지의 Linear를 추적용 Linear로 바꾼 시점에 이미 0.11647 차이가 났다.
패키지 대비 ORT 최적화 켬/끔은 0.12555 / 0.12342였다. 최적화를 끈다고 해소되지 않는다.
각 층에 **같은 활성화**를 넣어 따로 비교해도 196개 중 175개에서 최소 한 원소가 달랐다.
첫 차이는 첫 블록의 `self_attn.q_proj`였다. 같은 FP32 입력과 가중치에서
`F.linear`와 연속 전치 행렬을 곱하는 `matmul` 간 최대 차이는 반올림 전에 0.00000763이었지만,
BF16 반올림 뒤에는 출력 6,144개 중 4개가 달랐고 최대 차이는 0.00048828이었다
(`first-linear.log`, `FIRST_LINEAR_DONE`). 다음 단계는 이 층에서 백엔드별 누산 순서를 비교하고,
원래 BF16 레시피 그대로 더 긴 문맥의 품질 차이를 측정하는 것이다.
실제 파일 SHA-256은 `b0da0f3150203df7799cecb69021e61382634de7bde942cd93ce614034fa0ee4`이다.

로컬에서 내려받은 실제 모델은 다음 명령으로 연다 (위치: TriCast 저장소 기준).

```bash
DIR=runs/20261126T0142Z/export/models/qwen3-06b
python -m tricast.onnx_export run --onnx "$DIR/qwen3-06b-20pct.onnx" \
  --tokenizer "$DIR/tokenizer" --threads 4 --prompt "A triangle has" --max-new-tokens 2
```

## 회귀 검사

### YOLO11n · COCO val2017

기존 3비트 코드를 INT4 initializer에 넣던 경로를 실제 3비트 스트림으로 교체했다.
검증용 `auto_iso_u3`은 방법 이름과 달리 **87개 층 모두 INT3**다. 순수 코드는 977,988 B,
그래프·스케일을 포함한 ONNX는 **1,357,954 B**다(같은 모델의 FP32 추적본 10,567,925 B).
파일 SHA-256: `6b781377221d26351bf9a5247d1280cc2cd8d70356da7785e86d7e37173c3631`.

보정 이미지를 제외한 4,872장에서 AP@[.50:.95] (0–1)는 PyTorch **0.000007554743**,
ONNX **0.000007554741**이었다. 형식 변환 뒤 AP 차이는 −1.73×10⁻¹²이지만, 원본 양자화 모델부터
검출 정확도가 사실상 0이다. 별도의 시연 이미지 3장에서도 두 모델 모두 박스 0개, 원시 출력 최대 차이는 0.002014였다.
**저장 형식 회귀용이지 배포 권장 모델이 아니다.** 포스터의 혼합 INT4/INT8 검출 결과와도 다른 파일이다.
측정 원본: `runs/20261126T0142Z/export/models/yolo11n/onnx_auto_iso_u3.json`
(`YOLO_ONNX_DONE`, `monaco`, ORT CPU).

### CPU 스위트

- `bath`의 최종 소스에서 `python -m pytest -q`: **2,679 passed, 899 skipped**. CPU에서 실행한 것으로 CUDA 검증을 뜻하지 않는다. GPU 커널 코드는 바꾸지 않았다.
- `bradford`: bitpack · 일반 ONNX · 패키지 · 검출기 ONNX 관련 **40 passed**. 표준 ONNX, ORT 전용 MatMulNBits, external data, 동적 batch/seq, 그룹 Conv를 포함한다. 전송 전 `ruff check .` 및 `git diff --check` 통과.
- `oslo`: 앱 스위트 587 passed, 선택 의존성 부재로 5 skipped. pytest 통과 마커 뒤 이전 임시 셸 스크립트의 재전송 때문에 wrapper 종료는 실패했다. Optuna·HQQ를 포함해 해당 모듈과 검출기 검사를 다시 실행해 26 passed (skip 0, `APP_EXTRAS_PASS`). 두 실행의 통과 수를 합산하지 않는다.
- 초기 실패: ORT 1.23.2의 INT2 MatMulNBits 로드 거부, 체크포인트 텐서 복구 단계와 코드 복원 단계에서 생성한 같은 ONNX 노드 이름. 각각 1.25.1 환경과 이름 충돌 검사로 해결했다. 모델 파일 생성과 모델 품질 검증 마커는 별도다.
