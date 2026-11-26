# 2026-11-26 — 양자화 모델의 추론 런타임 호환성

[원본 ONNX 가중치 복원 검증](2026-11-26_onnx_export.md)에서 만든 Qwen3-0.6B·YOLO11n INT3와
기존에 배포한 YOLO11n 혼합 4/8 ONNX를 입력으로 사용했다. 원본 허깅페이스 모델을 새로 받아 양자화한 것이 아니다.
기준선은 기존 패키지(PyTorch) 또는 같은 SHA의 ONNX(ORT CPU)이며, **파일 생성, 실제 추론, 저장 비트폭, 출력 일치**는 따로 판정한다.
검증 파일과 명령은 `runs/20261126T0244Z/`, UCL에서는 `/scratch/uceeeee/tricast/interop_20261126T0244Z/`에 둔다.

## 실행 경로

| 런타임 | 입력 → 출력 | 주의점 |
|---|---|---|
| ONNX Runtime | ONNX → ORT FlatBuffer | 내보낼 때 최적화를 끄고, 저장된 코드의 타입·모양·길이·SHA-256을 소스와 대조한다. 이 검사는 CPU 커널의 저비트 가속을 뜻하지 않는다. |
| OpenVINO | ONNX → XML+BIN | `--serialization as-is`와 일반 변환은 저장 결과가 다르다. CPU 플러그인으로 **새로 읽은 IR**만 추론한다. |
| TensorRT | ONNX → 엔진 plan | GPU별로 다시 빌드해야 한다. 파서 거부와 추론 오차를 따로 기록한다. YOLO만 `--decode-conv-weights`로 원 ONNX에서 실제 FP32 Conv 피연산자를 얻는 제어군을 지원한다. |
| Transformers, vLLM | TriCast 패키지 → 밀집 HF safetensors | 확장자가 같아도 표준 HF 파라미터 스키마와 TriCast 코드 스키마는 다르다. 명시적 허용 없이 밀집화하지 않는다. |
| llama.cpp | 밀집 HF → GGUF F32 → 선택적 Q4_K_M | GGUF의 Q4_K_M은 새 양자화다. 원래 혼합 INT2/3/NF4의 코드가 아니다. |

실기기 처리속도·모바일 실행·Core ML·LiteRT/TFLite·TensorRT-LLM은 이 기록의 검증 대상이 아니다.
TensorRT 엔진을 TensorRT-LLM 지원으로, 데스크톱 .ort 로드를 모바일 전용 빌드의 지원으로 일반화하지 않는다.

## 공식 규격

- [ORT model format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html)
- [OpenVINO IR as-is 직렬화](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.serialize.html)
- [TensorRT quantized types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)
- [vLLM 양자화 구현](https://docs.vllm.ai/en/stable/features/quantization/)
- [llama.cpp 고정 소스](https://github.com/ggml-org/llama.cpp/tree/9c7185dd28416cf67f5e3b268381f311b5e3da56)

## 직접 실행

다음은 저장소 루트에서 실행한다. 출력은 항상 새 경로에 쓴다. 실험에서 사용한
일반 Linux CPU 환경은 Python 3.12.14 · ONNX 1.19.1 · ORT 1.25.1 · OpenVINO 2026.4.1이었다.
TensorRT 10.13.3.9 CUDA 12와 vLLM 0.21.0은 사용한 GPU 환경에서 별도 실행했다.

```bash
# NPZ는 ONNX 입력 이름을 키로 한다. 예: Qwen input_ids/attention_mask=int64[B,S], YOLO images=float32[B,3,H,W].
python scripts/quantizer/ort_export.py --onnx "$ONNX" --manifest "$MANIFEST" \
    --inputs "$CASE1_NPZ" "$CASE2_NPZ" --out deploy/ort
# YOLO: 원본 INT4/INT8 상수를 유지하고 CPU 플러그인의 split→conv 최적화 오류 우회
python scripts/quantizer/openvino_export.py --task detector --onnx "$MIXED_YOLO" --image "$IMAGE1" \
    --image "$IMAGE2" --gather-split-conv-inputs --out deploy/openvino
# 이 Qwen 그래프용 비트 해제 우회. 원래 수치와 동등하다는 뜻은 아니다.
python scripts/quantizer/openvino_export.py --task causal-lm --onnx "$QWEN_ONNX" --tokenizer "$TRICAST_PACK" \
    --serialization as-is --lut-unpack --out deploy/qwen-ir
# 검사 결과 JSON의 strict_match와 실제 데이터셋 지표는 별도다.
python scripts/quantizer/tensorrt_export.py --model yolo --onnx "$MIXED_YOLO" \
    --image-npy "$REAL_NCHW_NPY" --out-dir deploy/tensorrt
# 이 아래는 원래의 저비트나 층별 출력 반올림을 보존하는 경로가 아니다.
python scripts/quantizer/gguf_export.py dense-hf --package "$TRICAST_PACK" \
    --out deploy/dense-hf --allow-dense-approximation
python scripts/quantizer/gguf_export.py gguf --dense-hf deploy/dense-hf --llama-cpp "$LLAMA_CPP" \
    --out deploy/dense-f32.gguf --allow-dense-approximation
python scripts/quantizer/gguf_export.py requantize --gguf deploy/dense-f32.gguf --llama-cpp "$LLAMA_CPP" \
    --llama-quantize "$LLAMA_QUANTIZE" --out deploy/q4_k_m.gguf --allow-requantization
```

GGUF는 llama.cpp의 같은 체크아웃에서 만든 실행 파일로 읽는다. **Q4_K_M은 모든 텐서가
4비트라는 뜻이 아니다.** 이 모델은 F32 113개, Q6_K 29개, Q4_K 168개 텐서로 저장됐다.

## 실제로 무엇을 보존했나

- `.ort`: Qwen3-0.6B 515,895,320 B. 192개 코드 initializer의 명칭·타입·모양·길이·해시가
  모두 원 ONNX와 같았다(162,267,136 B). XML이나 파일 크기만 보고 판단하지 않았다.
  YOLO 혼합 `.ort`는 1,896,080 B (87개, 1,457,648 B), 순수 INT3는 1,976,248 B (87개, 977,988 B)이다.
  코드가 더 작아도 파일 전체는 더 클 수 있다. 혼합 모델은 원래 쓰던 파일이므로 24층 INT8·63층 INT4;
  **63개 중 16개는 논리적 3비트 코드를 INT4에 저장한 기존 파일**이다. 출력이 같다는 말이 이 16층을 실제 INT3로 줄였다는 뜻은 아니다.
- OpenVINO Qwen: 일반 변환 IR에서는 원래의 가중치 코드 **74/192**만 남고 BIN이 2,671,152,657 B가 됐다.
  `as-is` IR은 **192/192**(162,267,136 B)를 보존하고 BIN 490,646,629 B였지만 CPU 추론이 원본과 크게 달랐다.
  원본과 동일한 zero point 버퍼 146개는 이 가중치 수에 포함하지 않았다.
- TensorRT 10.13.3.9에서 Qwen의 변경 없는 UINT4 및 순수 INT3 YOLO의 UINT32 비트 연산 상수는
  ONNX 파서가 거부했다. 반면 혼합 YOLO는 원 그래프에서 강타입 엔진을 만들 수 있었다.
  디스크에서 plan을 다시 읽어 실행했으나 엔진에 들어간 각 층의 내부 정밀도는 이 실험으로 증명하지 않는다.
- Qwen의 원래 `model.safetensors`는 stock Transformers에서 메타데이터 `tricast.pack/1`, vLLM에서
  알 수 없는 `layers.0.mlp.gate_up_proj.codes`로 거부됐다. 코드 키를 무시하고 무작위 가중치를 생성한 모델은 평가하지 않았다.
  별도 HF 체크포인트는 패키지 가중치를 해독한 뒤 선형 입력 변환을 가중치에 접고 FP32로 저장했다.
  명시적 동의 없이 이 경로로 바꾸지 않는다.

## Qwen3-0.6B: 파일과 원 패키지의 예측 비교

WikiText-2 test의 동일한 네 구간(각 128토큰)에서 정답 문맥을 주고 원 TriCast 패키지의
다음 토큰 예측과 비교한 **508개 위치**다. 별도로 두 프롬프트에서 자유 탐욕 생성을 수행했다.
이 수치는 다음 단어 정답률이나 전체 테스트셋 PPL이 아니다. 모든 체크포인트는 같은 패키지에서 유래한다.

| 경로 | 508개 위치에서 같은 예측 | 두 프롬프트의 전체 생성열 일치 | 해석 |
|---|---:|---:|---|
| OpenVINO CPU, 원 코드 보존 IR + LUT 해제 | 504/508 | 원 ONNX 대비 다음 토큰 4/4 (공통 prefix) | 원 ONNX 대비 별도 506/508. 긴 자유 생성 일치 보장은 없다. |
| Transformers, 밀집 FP32 safetensors | 505/508 | 2/2 (각 8토큰) | 2,384,234,968 B 가중치. 소스 코드는 저장하지 않는다. |
| vLLM H100, 같은 HF 파일·BF16 실행 | 502/508 | 2/2 (각 8토큰) | 표준 safetensors 로더. |
| llama.cpp CPU, GGUF F32 | 505/508 | 2/2 (각 4토큰) | 모든 310개 텐서가 F32. |
| llama.cpp CPU, GGUF Q8_0 | 489/508 | 2/2 (각 4토큰) | 639,446,656 B, 197개 Q8_0 행렬·113개 F32 벡터. |
| vLLM H100, 같은 Q8_0 | 491/508 | 2/2 (각 8토큰) | 같은 파일, 별도 실행기. |
| llama.cpp CPU, GGUF Q4_K_M | 401/508 | 1/2 (각 4토큰) | 396,704,384 B. 원본에 대한 손실이 크다. |
| vLLM H100, 같은 Q4_K_M | 402/508 | 1/2 (각 8토큰) | 실행 성공이 원래 예측 보존은 아니다. |

원래 혼합 코드를 유지해야 할 때 Q4_K_M·Q8_0로 바꿔 부르지 않는다. Q8_0 하나로
llama.cpp/vLLM을 모두 쓰려는 경우에도 원본 대비 품질은 별도로 검증한다. `requantize` 명령에
`--type Q8_0`를 지정하면 같은 F32 GGUF에서 Q8_0을 만들 수 있다. vLLM 0.21의
[GGUF 실행법](https://docs.vllm.ai/en/v0.21.0/features/quantization/gguf/)에 따라 HF 토크나이저를 별도로 지정했다.

## ONNX Runtime: 기본 설정과 대조 설정은 다르다

직렬화 전·후 ORT CPU 세션의 그래프 최적화를 모두 `ORT_DISABLE_ALL`로 고정하면 Qwen 6개 입력의 **모든 logits**와
두 YOLO 모델 각각 3개 실사진의 **모든 출력값**이 완전히 같았다. 이는 원 TriCast 패키지가 아니라 **같은 ONNX 모델**과의 대조다.
별도로 두 세션을 **기본 최적화**로 새로 열면 결과는 달랐다. Qwen의 최대 logits 차이는 0.79521,
동일 정답 문맥 508개 위치 중 같은 top-1은 **504개**였다. 혼합/INT3 YOLO의 검출 헤드 최대 차이는
각각 0.0001221/0.001648이었다. 산술까지 같아야 하는 대조에서는 아래처럼 설정한다.
두 경우의 원 파일 해시와 가중치 저장 상태는 동일하다.

```python
import onnxruntime as ort
opt = ort.SessionOptions()
opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
opt.intra_op_num_threads, opt.inter_op_num_threads = 4, 1
session = ort.InferenceSession("model.ort", opt, providers=["CPUExecutionProvider"])
# session.run(None, {"input_ids": ids, "attention_mask": mask})  # Qwen, 두 입력 모두 int64
```

## YOLO11n: 전체 COCO val2017 대조

원본 SHA `8c56e144ee70919b816b59cb4ad5de5275ad7e17aae55acd0c5e0c3246962ca7`.
5,000장의 기존 직사각형 배치 구성을 그대로 사용하고, 보정에 쓴 128장을 제외해 4,872장만 채점했다.
후처리·라벨·좌표 변환·COCOeval은 원 ONNX 평가와 같다. 아래 mAP@[.50:.95]는 **0–100** 단위다.

| 실행 경로 | mAP | 원 ONNX 대비 (mAP 포인트) |
|---|---:|---:|
| 원 ONNX · ORT CPU | 24.449907 | 기준 |
| OpenVINO CPU · 혼합 IR | 24.449883 | −0.000024 |
| TensorRT A100 · 원 그래프에서 만든 plan | 24.454982 | +0.005075 |

OpenVINO: 직접 변환은 CPU 컴파일러의 `ConvertWeightCompressedConv1x1ToMatmul` 패스에서
다중 출력 Split을 처리하지 못했다. 해당 Conv 입력만 등가인 채널 Gather로 바꿨다.
생성된 IR의 87개 가중치 코드가 원본과 같았다. 원시 헤드의 엄격한 수치 대조는 3장 중 1장만 통과했으나,
세 장 모두 신뢰도 기준 이상 후보·클래스는 같았고 전체 mAP 결과는 표와 같다. 엔진에서 같은
출력이 나온다고 보장하지 않는다. TensorRT는 원 그래프 그대로, FP32 Conv 우회 **없이** 통과했다.
배포용 plan은 7,903,452 B이며 프로파일의 최소·최적·최대 입력은 각각
`1×3×256×320`, `8×3×512×672`, `8×3×672×672`이다. 메모리·지연시간·타 GPU 호환성 비교는 하지 않았다.

## 재현과 회귀

- 저장소 기준 `quantizer_develop@8e9066c02c1c12691998267507b028c87eb94d73` + 작업 트리.
  [이전 검증](2026-11-26_onnx_export.md)의 2,679개 통과는 같은 계산 코어의 과거 결과이며 이번 실행 수에 합산하지 않는다.
- 이번 UCL 패키지/ONNX 관련 검사 **40 passed**. 최초 MKL 환경을 고정하지 않은 실행은 Hessian 대칭 검사 8개가 실패했고,
  프로젝트의 기존 `MKL_CBWR=COMPATIBLE`에서 그대로 재실행하여 통과했다. 허용오차·기존 테스트는 바꾸지 않았다.
- ORT external-data 변조/명세 누락 거부, 기존 GGUF 파일 덮어쓰기·명시 동의·실제 텐서 유형 검사, NFS 임시 파일 정리를 검사했다.
  첫 F32 GGUF 변환의 NFS 정리 오류는 mmap을 먼저 닫도록 수정했다. 이미 만든 파일은 새 프로세스에서 310개 텐서와 SHA를 다시 검증했다.
- 실행 설정·명령 `runs/20261126T0244Z/setup/`, 모델별 원본 출력 `records/results/`, 집계 `result.json`,
  로컬 검증 파일 목록 `artifact_manifest.json` 및 `models/`. 수치 집계와 모델 SHA 검사를 각각 `summary.py`, `verify_artifacts.py`로 다시 실행할 수 있다.

### Qwen OpenVINO에서 포착한 오류

기본 IR과 `as-is` IR은 실행됐지만, 같은 두 프롬프트의 네 다음 토큰에서 원 ONNX/ORT와 하나도 일치하지 않았다.
가중치 보존과 별개다. 입력 임베딩과 묶인 출력 헤드의 대조 슬라이스는 일치했다. 첫 블록의 잔차 출력은
매우 비슷했지만 모든 블록을 연결한 진단 그래프에서 일부 INT2·NF4 복원 피연산자는 크게 달랐다.
같은 파일에서 2/4비트 byte-shift 해제만 256행 정수 lookup으로 대체하는 별도 실험을 했다.
이때 4/4 다음 토큰은 원 ORT와 일치했으나 최대 logits 차이 0.43266으로 엄격한 기준은 여전히 미통과였다.

별도 BF16 경계값 36,864개에서 ORT는 반올림 오라클과 일치했다. OpenVINO CPU는
`float32 → BF16 → float32` 동적 경로에서 30,720개가 달랐고, 상수 경로에서도 6,144개가 달랐다.
이는 해당 캐스트의 실제 비호환 증거다. 모델 전체의 모든 수치 차이가 BF16 때문이라고 단정하지 않는다.


배포용 명시적 `--lut-unpack`으로 새로 만든 IR에서도 192/192 가중치 코드의 바이트 해시가
일치했다. 같은 WikiText-2 508개 위치의 top-1은 원 패키지 대비 **504/508**, 원 ONNX/ORT 대비
**506/508**이고, 원 ONNX 대비 최대 logits 차이는 **0.49979**였다. 따라서 저비트 저장과
해당 모델의 샘플 추론은 검증했지만 엄격한 logits 동등성을 선언하지 않는다. IR의 XML/BIN은
모두 필요하다. 검증한 장치는 CPU, 설정은 `INFERENCE_PRECISION_HINT=f32`,
`DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, `NUM_STREAMS=1`, 스레드 4개다.
