# 압축 상태로 추론하기: 보존 경로와 표준 런타임 경로

파일만 압축하거나 로드 직후 전체 가중치를 FP로 바꾸는 것은 여기서 말하는 압축 추론이 아니다.
**(1)** 원래 가중치 코드를 유지하는지, **(2)** 실행 중 decoder projection의 상주 자료형,
**(3)** prefill과 decode가 실제로 실행한 연산, **(4)** 모델 품질은 따로 판정했다.
W4A16은 4비트 가중치를 소비한다는 뜻이지 누산·활성값·KV가 4비트라는 뜻이 아니다.

## 선택 방법

- 이미 만든 TriCast 혼합 패키지의 수치를 최대한 유지: `--mode tiled`. 이 모드는 전체 projection을 먼저 풀지 않고 행 타일을 풀어 PyTorch FP32 Linear에 넘긴다. 표준 vLLM 포맷보다는 이식 가능한 수치 기준 경로에 해당한다. 임베딩·norm·명시적 BF16 fallback은 압축 코드로 세지 않는다.
- 범용 로컬 LLM: 원래 HF 모델에서 **한 번** 표준 GGUF로 양자화한 뒤 llama.cpp 또는 지원되는 vLLM에서 연다. Qwen3-0.6B에서는 직접 만든 Q8_0이 원래 BF16의 PPL에 가장 가까웠다. 이 결론은 표의 모델·데이터·기기에 한정된다.
- W4 모델을 stock vLLM과 llama.cpp 양쪽에 주려면 **한 번 만든 W4 코드**를 HF compressed-tensors와 GGUF Q4_0로 저장한다. embedding·scale 저장 형식과 블록 배치는 다르지만 GGUF를 다시 읽어 복원한 가중치를 모두 대조한다. 백엔드별 활성·누산 산술과 출력 일치는 별도다.
- 모델을 이미 공격적으로 압축했다면 한 번 더 양자화하지 않는 것이 기본이다. 모델 품질에서 큰 차이가 났다. 원래 코드가 없는 층을 낮은 비트폭이라고 표시하지 않는다.

## Qwen3-0.6B

원본 revision `c1899de289a04d12100db370d81485cdf75e47ca`. 기존 혼합 패키지 `q06b_act_damp05_r3`의 가중치 SHA-256은 `6fc8b3413fc275e112f7669f3797873ea165e23aef2a4d06128d36252e3bd29b`. 품질은 WikiText-2 raw test 4,358행을 두 줄바꿈으로 이어 한 번 토큰화했다. 2,048 토큰씩 146창에서 각 첫 표적과 마지막 불완전한 창(70토큰)을 제외했다. **298,862개 표적 토큰**에 대해 NLL을 합산한 뒤 한 번 지수화했다. 아래 MB는 10⁶ B다.

| 출발점 → 실행 경로 | 전체 PPL ↓ | 실행 중 모델의 등록된 GPU 가중치·버퍼 (MB) | 동일 입력 128 + 출력 8 지연 median (ms) |
|---|---:|---:|---:|
| 원본 → vLLM BF16 | 20.9513 | 1,202.6 | 100.4 |
| 원본 → GGUF Q8_0 → vLLM | 20.9093 | 643.9 | 81.4 |
| 원본 → HF CT W4A16 → vLLM | 29.7899 | 548.9 | 110.1 |
| **같은 CT W4** → GGUF Q4_0 → vLLM | 30.6001 | 569.5 | 80.4 |
| 기존 혼합 패키지를 FP32로 복원 → vLLM BF16 | 26.4625 | 1,202.6 | 133.8 |
| 기존 혼합에서 파생된 GGUF 후보 → 5.75 bpw 배정 → vLLM | 28.2457 | 473.9 | 83.0 |
| 기존 혼합을 다시 CT W4A16으로 양자화 → vLLM | 53.2980 | 548.9 | 74.8 |
| 바로 위 CT의 코드를 GGUF Q4_0로 재배치 → vLLM | 55.1266 | 569.5 | 73.8 |

H100 NVL, vLLM 0.21.0, eager, TP=1, batch 1, 고정 KV 384 MiB. 지연은 3회 워밍업 후 5회 별도 요청의 wall time이다. 서로 다른 GPU 슬롯·런 시점이므로 장비를 고정한 성능 우위나 서비스 처리량으로 일반화하지 않는다. 속도 개선이 필요한 경우에도 H100에서는 CT W4가 원본 BF16보다 느렸던 결과까지 고려해야 한다. 전체 장치 메모리, 부하를 건 서버 또는 긴 KV-cache benchmark는 아니다.

**저비트 발화:** H100의 CT는 Machete, A100의 같은 직접 CT는 Marlin(PPL 29.7908)이었다. 직접 Q8_0은 A100에서도 GGML MMQ/MMVQ로 실행됐다(PPL 20.9119). 모델 내부 196개 원래 projection이 vLLM에서 112개 모듈로 합쳐진다. 각 모듈의 추론 전후 등록 파라미터를 보고 128-token prefill·1-token decode의 **224개 범위**에서 실제 CUDA launch를 대조했다. BF16과 초기 재양자화 CT 실행의 예전 검사기는 CPU 범위와 Kineto의 GPU 복제 주석을 이중 집계했다. 저장된 원 trace 재판독 결과는 별도 JSON이며, 이 초기 파일들은 실행 시점 trace 해시가 없는 사후 판독이다. 이후 실행은 trace의 SHA도 기록한다.

**HF 포맷과 Transformers 실행은 별개다.** 동일한 직접 W4 디렉터리를 Transformers 4.57.6 / compressed-tensors 0.15.0.1 기본 로더로 열고 H100에서 8토큰을 생성했다. 로드 직후에는 밀집 `weight`가 있는 projection이 0/196개였지만, 첫 생성에서 `Decompressing model`이 출력됐고 생성 후에는 196/196개가 BF16 밀집 2차원 가중치를 보유했다. 같은 모델·기기에서 vLLM은 생성 뒤에도 W4가 유지됐다. Transformers로 파일을 읽거나 답을 얻는 것만으로 압축 상주 추론이라 하지 않는다(`stock-hf.json`).

**원래 혼합 레시피:** 196층 중 UINT2 28개·UINT3 118개·UINT4 33개·NF4 13개는 코드 그대로, 원 코드가 없는 HQQ 4개는 선언한 BF16으로 남는다. CPU, V100(FP32 입력) 및 H100에서 모든 층의 저장 바이트 및 타일별 값을 기존 pack decoder와 대조했다. H100(입력 BF16, 타일 256행) 전체 PPL은 FP32로 projection을 풀어 둔 수치 기준 **26.4456**, 타일 실행 **26.4436**이다. 이는 위 vLLM BF16이나 원본 모델과의 비교가 아니다. 가중치 복원은 일치하지만 행렬곱 모양이 달라 여섯 입력 중 일부 위치의 logits·최상위 토큰은 달랐다.

**Auto-tune:** F32 GGUF가 같은 두 표준 후보(Q4_K_M, Q8_0)에서 각 텐서를 실제로 읽는다. 제약은 decoder 코드+스케일, 목적 함수는 층별 상대 가중치 제곱오차의 합이다. 예산 5.75에서 실제 **5.746429 bpw / 316,342,272 B**를 선택했다. 파일은 450,181,728 B이며, 선택하지 않은 embedding·norm 등은 명시한 base에서 복사했다. CPU `MUL_MAT`의 실제 입력 텐서 196개 각각의 종류와 크기가 계획과 일치했다. 기존 fake-KL 후보표를 이 용도로 사용하지 않는다.

**긴 입력과 CPU 대조:** A100에서 직접 CT와 직접 Q8_0은 2,048토큰 prefill과 1토큰 decode 모두에서 압축 연산을 재확인했다. llama.cpp CPU는 Q8_0 파일(804,753,536 B)의 196개 projection을 모두 Q8_0 피연산자로 사용했다. 4×128토큰 부분 대조에서 같은 파일을 쓴 vLLM과 다음 토큰 argmax가 489/508개 일치했다. CPU 전체 코퍼스 PPL로 읽으면 안 된다.

**원본 혼합 코드의 메모리:** H100의 별도 새 프로세스 두 개에서 같은 2,048토큰 입력을 사용했다. 전체 projection을 FP32로 풀어 두는 기준과 비교해 모델 로드 후 PyTorch CUDA 할당량은 **2,077,747,712 → 493,008,896 B (−76.3%)**, forward 피크는 **2,811,948,544 → 1,229,110,272 B (−56.3%)**였다. GPU 전체 사용량, vLLM 비교, 또는 속도 우위가 아니다. 원래 **혼합 패키지**의 값과 FP32 행렬곱을 사용하는 경로다. 실제 CLI로도 16-token 한도의 greedy 생성을 실행했다(`records/generation.log`). 답에는 “three sides”와 관계없는 다국어 접두부가 함께 나왔다. 생성 성공은 답변 품질 통과와 다르다.

## 두 번째 계열: Llama-3.2-1B

원본 revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`. 다른 tokenizer로 같은 WikiText-2를 다시 처리했다(141×2,048, 288,627개 표적; 꼬리 309개 제외). H100 vLLM의 원본 BF16은 **PPL 9.7560 / 2,488.4 MB**, 원본에서 직접 만든 CT W4A16은 **12.7022 / 1,044.0 MB**였다. W4는 64개 fused 모듈의 prefill/decode 128개 범위에서 Machete 실행을 확인했다. 가중치 파일은 배포하지 않는다.

동일 W4를 담은 GGUF는 Llama의 Q/K 행 순열 32개와 upstream converter의 RoPE 텐서를 검증했고, llama.cpp CPU는 112/112개 projection에서 Q4_0을 실제로 소비해 생성·teacher-forced 평가를 마쳤다. **같은 파일을 vLLM 0.21.0에서 로드하면 `KeyError: embed_tokens.weight`로 실패**했다. 이 파일은 tied embedding을 BF16으로 보존한다. 이 vLLM 버전의 [GGUF loader](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/model_loader/weight_utils.py)는 비양자화 embedding을 `weight`로 전달하고, [Llama 구현](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/models/llama.py)은 embedding 초기화 때 quant 설정을 넘긴다. 로더를 고치거나 embedding까지 추가 양자화한 다른 모델을 동일 가중치 검증으로 대체하지 않았다. 권장 조합은 CPU=GGUF, NVIDIA 서버=HF CT다.

## 비전: YOLO11n / ONNX Runtime CPU

기존 혼합 가중치 파일의 정수 코드를 INT8로 부호 확장하고, 128장으로 UINT8 activation 범위를 보정해 QLinearConv로 내렸다(INT32 bias). ORT 프로파일에서 학습된 Conv **87/87개**의 정수 연산을 확인했다. DFL의 고정 상수 16개는 FP32 예외다. 이 경로는 원래 혼합 비트폭이나 부동소수 활성 계산을 유지하지 않는다.

| held-out COCO 4,872장 / CPU | 기존 혼합 weight + FP Conv | 정수 Conv |
|---|---:|---:|
| AP@[.50:.95] (0–100) | 24.4499 | 19.0082 |
| ONNX 파일 (B) | 1,657,313 | 2,877,608 |
| 독립 프로세스 3개 입력 실행 후 최대 RSS (MB) | 129.0 | 148.7 |

서로 다른 3개 입력의 median 지연은 각각 70.3/71.9/73.9 ms와 70.4/72.7/69.5 ms다. 정수 연산이 발화해도 이 경우에는 파일과 피크 메모리가 커지고 AP가 **5.44 pp** 떨어진다. **기본 배포 경로로 채택하지 않는다.**

## 지금 생성해 보기

저장소 루트에서, UCL 검증과 같은 vLLM 0.21.0 CUDA 환경 또는 llama.cpp 설치를 사용한다. 아래 파일은 로컬 `runs`에 있고 Git/HF에 게시하지 않았다. GGUF 파일만 넘기는 llama.cpp와 달리 vLLM GGUF 예시에는 원본 tokenizer/config 디렉터리도 필요하다.

```bash
RUN=runs/20261126T034137Z
# HF W4는 이 디렉터리 하나만으로 작동한다.
python examples/generate_packed.py "$RUN/models/hf-original-w4"
# vLLM은 GGUF의 가중치와, 같은 원본 tokenizer/config를 따로 받는다.
python examples/generate_packed.py "$RUN/models/original-q8.gguf" --tokenizer "$RUN/models/q06b_act_damp05_r3"
# llama.cpp CPU:
llama-cli -m "$RUN/models/original-q8.gguf" -p "How many sides does a triangle have?" -n 16
```

GPU 두 명령에 해당하는 실행을 네트워크 없이 새 프로세스에서 각각 검증했다. 두 경로 모두 실제로 생성한 답에 세 변이라고 답했다. 이 한 프롬프트는 사용법 확인용이고 모델 품질 평가는 위 전체 코퍼스 표 기준이다. [배포 파일 해시](../../runs/20261126T034137Z/artifact_manifest.json), [장치·코드·입력 지문](../../runs/20261126T034137Z/env.json)은 로컬 실행 기록이다.

## 재현 명령

아래 `SRC`는 원본 HF 스냅샷 디렉터리, `LLAMA`는 공식 llama.cpp 체크아웃이다. llama.cpp 실험용 commit은 `9c7185dd28416cf67f5e3b268381f311b5e3da56`. 변환기와 gguf-py는 같은 버전을 쓴다. 명령은 저장소 루트에서 실행한다. 출력을 재사용해 덮어쓰지 않는다.

```bash
# 1. 원본에서 W4를 한 번 만든 후 같은 양자화 값으로 CPU/GPU용 파일을 얻는다.
python scripts/quantizer/hf_packed_export.py --original-hf "$SRC" --out deploy/w4
python "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outtype f32 --outfile deploy/template-f32.gguf
python scripts/quantizer/hf_packed_gguf.py --original-hf "$SRC" --checkpoint deploy/w4 \
  --template deploy/template-f32.gguf --llama-cpp "$LLAMA" --out deploy/w4.gguf
# 또는 품질 우선 Q8. 이 명령은 원본에서 만든 F32 template에 적용한다.
"$LLAMA/build/bin/llama-quantize" deploy/template-f32.gguf deploy/q8.gguf Q8_0

# 2. 실행기에서 직접 사용. 검증기는 본문과 동일한 offline single-GPU 경로.
PYTHONPATH=src:. python scripts/quantizer/packed_vllm_check.py --kind compressed-tensors \
  --model deploy/w4 --tokenizer "$SRC" --inputs wiki-window.npz --corpus-ids wiki.npy \
  --seqlen 2048 --kv-mib 384 --out checks/w4
"$LLAMA/build/bin/llama-cli" -m deploy/w4.gguf -p "A triangle has" -n 16

# 3. 원래 TriCast 패키지는 다른 포맷으로 바꾸지 않고 직접 실행할 수 있다.
python -m tricast.pack run --package "$PACK" --device cuda --mode tiled --tile-rows 256 \
  --prompt 'A triangle has' --max-new-tokens 16
```

`wiki-window.npz`의 `input_ids`는 정수 [1,S], `wiki.npy`는 같은 tokenizer의 1차원 정수 배열이다. 토큰 ID를 서로 다른 모델에 돌려 쓰지 않는다. 실험별 실제 실행 명령은 `runs/20261126T034137Z/`의 `*.sh`, 원시 로그·PPL·프로파일은 같은 폴더 `records/`와 UCL `/scratch/uceeeee/tricast/packed_20261126T034137Z/` 참조. 측정 전에 정한 seed는 42다.

## 배포 범위와 후속 순서

1. **파일 규격을 새로 만들지 않는다.** 혼합 정수 코드·scale·zero·codebook·레이어 변환의 기준은 기존 패키지다. 대상이 정확히 표현하면 재배치하고, 아니면 명시적 재양자화 또는 미지원으로 반환한다. HF는 런타임이 아니라 체크포인트 규약이다. ONNX는 그래프/교환용이며, 그래프가 열려도 정수 연산이 쓰였다는 보장은 없다. 타깃별 배정 사례와 실명령은 `auto.sh`, `ct-table.sh`, `apply.sh`에 있다. 기존 Auto-tune의 모델 KL과 이 배포 후보의 가중치 오차는 서로 다른 목적함수다.
2. **다음 우선순위는 정확도다.** 서버 W4는 데이터를 이용한 직접 양자화를 원본에서 만들고 같은 비트 예산에서 비교한다. 선택에 사용하지 않은 평가셋으로 PPL과 과제 점수를 확인하기 전 품질 기본값을 바꾸지 않는다. YOLO는 재보정 후 기존 혼합 모델 대비 COCO AP와 RSS를 모두 다시 잰다.
3. **모바일은 별도의 실행 대상이다.** ExecuTorch `.pte`에 대해 먼저 모델 하나, 실제 ARM 장치와 delegate 하나를 정한다. 위임된 연산·packed 상주량·load/forward 피크 RSS·정확도와 지연을 기존 경로와 대조해야 지원으로 올린다. 지금은 기기 검증이 없고, CUDA vLLM이나 x86 llama.cpp 기록으로 대신하지 않는다.
4. **커널 생성기는 보류한다.** stock 커널이 있으면 먼저 사용한다. 원형 혼합 포맷만 필요한 타깃에서 타일 복원이 병목임을 프로파일로 보인 뒤, 가장 흔한 UINT3 projection 하나부터 fused kernel을 검토한다. 이를 모든 포맷과 GPU에 필요한 전제조건으로 두지 않는다.

## 배포 파일 식별

TriCast GitHub 기준 `8e9066c02c1c12691998267507b028c87eb94d73` 위의 미커밋 작업 트리다. 아래는 UCL에서 실행한 파일과 로컬 배포 파일을 대조한 값이다. HF·TriCast 행은 디렉터리 전체가 아니라 가중치 파일 크기다.

| 파일 | B | SHA-256 |
|---|---:|---|
| 기존 TriCast 혼합 패키지 가중치 | 490,965,128 | `6fc8b3413fc275e112f7669f3797873ea165e23aef2a4d06128d36252e3bd29b` |
| 직접 Q8 GGUF | 804,753,536 | `3d41c99eec9b5ab61d8e3bd55b8591d93f58b1084fc05bcf6166ee52eea4de6b` |
| 직접 W4 HF 가중치 | 538,459,960 | `dd93ee6efba2bd99a6cd4ae00149bf43ad28c40bc2c7555eb07a5142731aa5dc` |
| 동일한 W4의 GGUF | 565,104,736 | `f7073721b3ebef541fb70b4f6005b7b4dba189cd4d3c889ac78897af80828704` |
| 5.75 bpw 예산 GGUF | 450,181,728 | `d64c16e57bf088d901205ddd731530b1b6b4a9564d32665cccb017be550c9188` |
| 실험 YOLO INT8 ONNX | 2,877,608 | `8d27fa8505a65ebd090f75f7fa61a41785f75636caded53b0591fb4badcbbae5` |

## 회귀·재현 근거

- UCL CPU 전체 `python -m pytest -q`: **2,679 passed, 899 skipped**. 대부분 GPU 전용 테스트이며 GPU 실행 결과와 분리한다.
- UCL 앱 전체: **590 passed, 2 skipped**. Optuna가 설치된 환경에서 해당 파일을 포함한 다섯 파일을 다시 실행해 **26 passed**. 겹치는 테스트가 있어 합산하지 않는다.
- 기존 GPU 회귀 `python -m pytest tests/gpu -q -rs`: H200(berlin GPU 2), A100(geneva GPU 1), V100(london GPU 0) 각각 **896 passed, 1 skipped**. 공통 skip은 명시적 `TRICAST_MMA_PERF` opt-in 성능 시험이다.
- 측정·스크립트·모델 식별은 `runs/20261126T034137Z/{env.json,result.json,artifact_manifest.json}`; 실행 마커·trace·혼합 타일의 독립 프로세스 계측은 `records/`에 있다. 소스의 Ruff 검사와 Git 공백 검사도 통과했다. 커널 ISA나 서비스 부하에서의 성능 향상은 주장하지 않는다.
