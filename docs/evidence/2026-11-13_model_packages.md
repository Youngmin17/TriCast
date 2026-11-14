# 2026-11-13 — 모델 패키지: 고른 레시피를 저비트 파일로 (Qwen3-0.6B · Qwen3-8B safetensors, YOLO11n ONNX)

> 요청 (2026-11-13): "onnx 나 safetensor 등 필요한 패키지 출력 및 추론 구조 연결까지". 지금까지 튜닝 화면 ⑤ 는 크기를
> **계산**만 하고 레시피 묶음(zip)을 내보냈다. 이 문서는 같은 레시피로 실제 저비트 파일을 만들고, 다시 불러 추론한 기록이다.
> 수용 기준은 SPEC AC30 (언어 모델 패키지) · AC31 (검출 모델 ONNX).

## 1. 무엇을 만들었고, 무엇으로 판정하나

| | 언어 모델 (`python -m tricast.pack`) | 검출 모델 (`scripts/quantizer/yolo_tune.py onnx`) |
|---|---|---|
| 파일 | `model.safetensors` — 양자화한 Linear 마다 정수 코드(2 · 3 · 4 · 8비트로 묶은 uint8) · 스케일 · 영점, NF4 는 블록 absmax, 입력 변환(랜덤 하다마드)은 부호 대각. 나머지 텐서(임베딩 · 정규화 · 묶이지 않은 출력층)는 원래 dtype. `quant_config.json` — 레시피 원문 · SHA-256 · 모델 revision · 층별 저장 형식. 체크포인트의 `config.json` · 토크나이저. `manifest.json` — 파일 크기 · SHA-256 · 검사 결과 | `<setup>.onnx` — 양자화한 conv 가중치를 INT8 · INT4(3비트 코드도 INT4 에) 정수와 출력 채널별 FP32 스케일로 저장, `DequantizeLinear`(axis 0, opset 21)가 원래 이름의 FP32 가중치로 풀어 Conv 에 넘김. 활성값은 FP32 |
| 쓰기 전 검사 | 모든 층을 풀어낸 가중치가 TriCast 에뮬레이션이 곱하는 가중치(`EmuLinear` 의 피연산자)와 **비트 단위로 같은지** (0 의 부호 제외 — §6). 하나라도 다르면 실패 | 층마다 저장한 정수(INT8 · INT4) × 스케일 = 내보낸 FP32 가중치 (비트 단위, 0 의 부호 제외). 다르면 실패 |
| 다시 불러 잰 것 | WikiText-2 test PPL (TriCast 모델과 같은 GPU · 같은 평가 코드), 코드만 GPU 에 두는 방식의 PPL · GPU 메모리 | onnxruntime 으로 COCO val2017 4,872장 mAP (PyTorch fake-quant 평가와 같은 사진 · 같은 후처리), 시연 사진 3장의 원시 출력 차이 · 상자 수 · 클래스 |
| 화면 ⑤ 에 싣는 조건 | 위 검사 통과 + 패키지 PPL 이 TriCast PPL 과 상대 0.1% 안 (`pack_index.py`, AC4 와 같은 한계) | 파일의 SHA-256 이 기록과 같을 때 (`pack_index.py`) |
| 담지 않는 것 | TriCast 의 MMA 산술(칩 축), 활성값 양자화 · KV 캐시 · 희소화 · 이상치 · BF16/FP16/FP32 밖의 출력 형식 (모두 거부), HQQ 의 정수 코드 (공식 `hqq` 가 코드를 남기지 않아 그 층은 TriCast 가 곱하는 가중치 그대로, `fallback_layers` 에 사유) | 활성값 양자화, 칩 산술 |

0.1% 한계는 측정 전에 정한 값이 아니다. 패키지를 처음 잰 뒤(−0.017%, +0.034%) 화면 게이트로 넣으면서 AC4(무손실 레시피 PPL
0.1%)와 같은 값을 가져왔다.

## 2. 실행

- 트리: `6bc78e6` + 작업 트리 (`pack.py` · `linear.py` 훅 등, 로그의 `dirty_lines`). 클러스터 conda `tricast` (torch 2.8.0+cu128,
  transformers 4.55.2), `HF_*_OFFLINE=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed 42, bf16. ONNX: onnx 1.19.1 · onnxruntime 1.23.2
  (`pydeps_onnx`), CPU 16 스레드.
- 보정 캐시는 쓰지 않았다 — 캐시에는 풀어낸 피연산자만 있고 정수 코드가 없다. 그래서 패키지를 만들 때 다시 보정한다.

```bash
# 언어 모델 (gpu_cap 1칸; qz_pack.sh = build --ppl → run --mode codes --ppl)
T=…/qzbench_8 M=Qwen/Qwen3-0.6B R=c1899de289a04d12100db370d81485cdf75e47ca \
  RECIPE=evals/fixtures/tune_inputs/s3150/proposals/act_damp05_r3/recipe.yaml OUT=…/packages/q06b_act_damp05_r3 bash qz_pack.sh
T=…/qzbench_8 M=Qwen/Qwen3-8B R=b968826d9c46dd6066d109eabc6255188de91218 \
  RECIPE=…/q8b/table/auto_3.00/recipe.yaml OUT=…/packages/q8b_auto_3.00 HMB=103079215104 bash qz_pack.sh
# 메모리 기록을 고친 뒤 codes 모드만 다시 (qz_pack_run.sh, 트리 qzbench_10)
python -m tricast.pack run --package <dir> --mode codes --ppl --out <dir>.run_codes.json
# 검출 모델 (cpu_cap, monaco)
python scripts/quantizer/yolo_tune.py onnx --work …/qzyolo_runs/r1 --more …/r1/e1 \
  --setups native,u_int4,heur_8_4,auto_iso_u4,auto_iso_h84 --threads 16 --workers 2
# 화면 목록
python scripts/quantizer/pack_index.py --out app/web/demo/autotune/packages.json \
  --llm runs/20261113_pack/q06b_act_damp05_r3 --llm runs/20261113_pack/q8b_auto_3.00 \
  --onnx yolo11n=runs/20261113_pack/yolo_onnx/onnx --onnx-copy app/web/demo/autotune/packages/yolo11n
```

기록 (저장소 밖 `runs/20261113_pack/`, 패키지 파일 자체는 클러스터 `/scratch/uceeeee/tricast/packages/`): `*/manifest.json` ·
`*/quant_config.json` · `*.run_codes.json` · `logs/` (`pack_q06b.log` · `pack_q8b.log` · `qz10_packrun.log` · `yolo_onnx.log`) ·
`yolo_onnx/onnx/` (ONNX 5개 + `onnx_*.json`).

## 3. 결과

### 3.1 언어 모델 패키지

| | Qwen3-0.6B · 20% AI 추천 (`act_damp05_r3`) | Qwen3-8B · 19% (`auto_3.00`) |
|---|---|---|
| GPU · 시각 (UTC) | malmo H100 NVL · 04:20–04:28 | berlin H200 NVL · 04:20–04:38 |
| 레시피 SHA-256 | `8cd7cee3e93a…` (튜닝 세션 제안 파일) | `89307fc45621…` (화면 번들의 `recipe_file_sha256`) |
| 층 (저장 형식) | 196 — 정수 2 · 3 · 4비트 28 · 118 · 33, NF4 13, BF16 그대로 4 (HQQ `k_proj`: 블록 3 · 24 · 25 · 26) | 252 — 정수 2 · 3 · 4비트 46 · 152 · 54 |
| 풀어낸 가중치 = TriCast 가중치 | **196/196** 비트 단위 (0 의 부호 제외) | **252/252** 비트 단위 (0 의 부호 제외) |
| 패키지 전체 / 원본 체크포인트 | **0.51 GB** / 1.50 GB (34%) — `model.safetensors` 0.49 GB | **5.11 GB** / 16.38 GB (31%) — `model.safetensors` 5.09 GB |
| WikiText-2 PPL: TriCast → 패키지 (같은 GPU) | 26.445773 → 26.441190 (**−0.017%**) | 11.224198 → 11.228064 (**+0.034%**) |
| 코드만 GPU 에 두는 방식 (`--mode codes`) | 로드 후 GPU 메모리 **0.49 GB**, PPL 26.445626, 평가 중 최대 6.13 GB | 로드 후 **5.11 GB**, PPL 11.224693, 평가 중 최대 11.93 GB |
| 화면 ④ 의 기록과 | 다르다 — ④ 의 26.51 은 A100 기록 (같은 레시피 파일, 보정 · 측정 GPU 가 다름) | 패키지를 만들 때 잰 TriCast PPL 이 ④ 의 기록과 **비트 단위로 같다** (11.224197609462264) |

- 0.6B 패키지가 계산값(0.80 GB)보다 작은 것은 출력층이 임베딩과 묶여(tied) 한 번만 저장되기 때문이다 (원본 체크포인트와
  계산값은 둘 다 따로 센다). 8B 는 묶이지 않아 계산값 5.09 GB 와 `model.safetensors` 가 같다.
- 패키지 PPL 의 차이는 행렬곱의 합산 순서에서 온다: 가중치는 같고, TriCast 는 K 순서의 FP32 FMA 사슬, 패키지는 PyTorch 행렬곱.
  `load` 모드는 BF16 가중치를 cuBLAS BF16 GEMM 으로, `codes` 모드는 풀어낸 FP32 가중치를 FP32 GEMM 으로 곱한 뒤 BF16 으로
  반올림해, codes 쪽이 TriCast 에 더 가깝다 (0.6B −0.0006%, 8B +0.0044%). cuBLAS BF16 경로의 축약 정밀도가 원인이라고 보지만
  따로 확인하지 않았다 (미검증).
- 처음 codes 모드 실행(트리 `qzbench_8`)은 "GPU 메모리 21.5 GB" 를 적었다. 모델 골격을 BF16 Linear 로 먼저 만든 뒤 바꿔 끼우는
  동안의 최대값이라 패키지의 메모리가 아니었다. 로드 직후 상주 메모리와 평가 중 최대값을 따로 적게 고치고(`run`) 다시 쟀다.

### 3.2 YOLO11n ONNX (monaco CPU, 04:28–05:18 UTC)

| 셋업 (크기) | 저장 형식 | 파일 | COCO mAP50-95: onnxruntime / PyTorch 기록 | 차이 | 시연 사진 원시 출력 최대 차이 | 상자 수 · 클래스 (3장) |
|---|---|---|---|---|---|---|
| 원본 (native) | FP32 | 10.57 MB (저장소에 넣지 않음) | 0.3945 / 0.3945 | −1.1×10⁻⁶ | 0.0033 | 3/3 같음 |
| INT4 균일 (25%) | INT4 87층 | 1.50 MB | 0.0109 / 0.0109 | +9×10⁻⁹ | 0.0106 | 3/3 같음 |
| INT8 가장자리 + INT4 (26%) | INT8 7 · INT4 80 | 1.52 MB | 0.0136 / 0.0136 | −4×10⁻⁹ | 0.0065 | 3/3 같음 |
| **Auto-tune (25%)** | INT8 23 · INT4 64 | 1.65 MB | **0.2383** / 0.2383 | +7×10⁻⁸ | 0.0032 | 3/3 같음 |
| **Auto-tune (26%)** | INT8 24 · INT4 63 | 1.66 MB | **0.2445** / 0.2445 | −9×10⁻⁸ | 0.0070 | 3/3 같음 |

- 원시 출력은 디코딩된 상자 좌표(픽셀)와 클래스 점수라 0.003–0.01 은 픽셀 이하 차이다. onnxruntime 과 PyTorch 의 conv 커널이
  다르기 때문이며, 정수 가중치가 아니라 FP32 원본(native)에도 같은 크기로 있다.
- 교차 검토(§6 의 11번) 뒤 INT 셋업 4개를 IR 10 으로 다시 만들었다 (monaco, 트리 `qzbench_11`, 06:17–06:40 UTC, 두 작업으로
  나눠 16 스레드씩). 파일 크기 · mAP · PyTorch 대비 차이 · 원시 출력 차이가 위 표와 같았고, 새 검사(PyTorch 기록의 배정 =
  ONNX 의 배정)도 통과했다. 바뀐 것은 머리의 IR 버전뿐이라 SHA-256 만 달라졌다: `u_int4` `5ff490ff83e2…` · `heur_8_4`
  `95beaeb39eb0…` · `auto_iso_u4` `02b2d7406f1c…` · `auto_iso_h84` `8c56e144ee70…`. 원본 FP32 파일은 INT4 가 없어 IR 9 그대로다.
- ONNX 파일이 계산값(26% 에서 1.35 MB)보다 큰 것은 스케일 · 바이어스를 FP32 로 두고(계산은 FP16) 영점과 그래프 정의를 함께 담기
  때문이다.

### 3.3 테스트 (클러스터; 첫 판은 트리 `qzbench_10`, 교차 검토 반영 뒤 `qzbench_13` — 둘 다 `6bc78e6` + 작업 트리)

| 스위트 | 결과 |
|---|---|
| `pytest tests --ignore=tests/gpu` (monaco CPU) | 첫 판 2541 passed, 3 skipped (`tests/test_pack.py` 9건). 검토 반영 뒤 2546 passed, 3 skipped, 1 failed — `test_ontology.py` 가 SPEC §4 에 새로 넣은 범위 밖 문구를 막음 (온톨로지 항목과 이름 불일치). 문구를 고쳐 `test_ontology.py` · `test_spec_traceability.py` 14 passed |
| `pytest app/tests` (CPU) | 첫 판 561 passed (새 파일 추가 전 수집) · 검토 반영 뒤 575 passed (`pydeps_onnx` 포함) |
| `app/tests/test_yolo_onnx.py` (budapest CPU, `pydeps_onnx`) | 2 passed |
| `tests/test_pack.py` · `app/tests/test_pack_index.py` · `app/tests/test_yolo_onnx.py` · `tests/test_nn_patch.py` (검토 반영 뒤) | 15 · 12 · 2 · 57 = 86 passed |
| `pytest tests/gpu` + 방법 테스트 (geneva A100) | 896 passed, 1 skipped · 108 passed (`linear.py` 훅 변경 확인) |

## 4. 한계 · 미검증

- **속도는 재지 않았고 빨라지지 않는다.** 패키지는 정수 코드를 풀어 일반 행렬곱(BF16 · FP32)으로 계산한다. 저비트 커널
  (예: INT4 GEMM)은 이 저장소의 범위 밖이다 (SPEC §4). 줄어드는 것은 파일 크기와, `codes` 모드에서의 GPU 상주 메모리다.
  `codes` 모드는 층마다 매번 풀어 쓰므로 느리다 (재지 않음).
- 패키지는 두 셋업만 만들었다. 다른 셋업은 화면 ⑤ 에 만드는 명령만 나온다.
- 0.6B 패키지의 TriCast PPL(26.446, H100)은 화면 ④ 의 A100 기록(26.51)과 다르다. 같은 레시피 파일이지만 보정과 측정을 다른
  GPU 에서 했다. 같은 GPU 끼리(8B, H200)는 비트 단위로 같았다.
- GPU 한 종류에서 한 번씩만 만들었다 (seed 42). 다른 GPU 에서 같은 바이트의 패키지가 나오는지는 확인하지 않았다.
- `transformers` 의 다른 모델 계열(Llama 등)은 CPU 의 작은 Llama 로만 확인했다 (`tests/test_pack.py`).
- ONNX 는 가중치만 정수다. INT4 `DequantizeLinear` 는 opset 21 이 필요해 onnxruntime 1.20 이상에서만 열린다.

## 5. 이 과정에서 고친 것

| 문제 | 찾은 방법 | 고침 |
|---|---|---|
| 작은 Llama 에서 패키지 logits 가 TriCast 와 최대 약 1e-3 다름 | `tests/test_pack.py` 첫 실행 | TriCast 는 출력을 MMA 명세의 출력 형식(BF16)으로 반올림한다. `PackedLinear` 도 같은 규칙으로 반올림 → 층별 차이 0 |
| 레시피 원문을 경로로 읽음 | 같은 테스트 | `load` 가 `yaml.safe_load` 로 원문을 읽음 |
| codes 모드에서 키 누락 오류 | 같은 테스트 | 양자화 층이 가진 접두어를 누락 검사에서 제외 |
| codes 모드의 GPU 메모리가 골격 생성까지 포함 (21.5 GB) | 8B 결과가 패키지 크기보다 4배 큼 | 로드 후 상주 · 평가 중 최대를 나눠 기록, 다시 잼 (§3.1) |
| `pack_index.py` 가 `.run_codes.json` (JSON 뒤에 `PACK_RUN_DONE` 줄)을 읽지 못함 | 로컬 실행 전 파일 확인 | JSON 하나만 읽고 나머지 무시, `run --out` 으로 JSON 만 따로 저장 |
| onnxruntime 출력이 정수 그래프와 FP32 그래프 사이에 비트 단위로 같지 않음 | `app/tests/test_yolo_onnx.py` 첫 실행 | 가중치가 노드 출력이면 다른 conv 커널을 고르는 것으로 보임. 테스트를 둘로 나눔: `DequantizeLinear` 만 돌려 FP32 가중치와 비트 단위로 같은지, 출력은 상대 1e-5 안 |
| 원본 FP32 ONNX(10.6 MB)까지 화면 폴더로 복사 | 파일 목록 확인 | 양자화한 층이 없는 셋업은 기록만 싣고 파일은 복사하지 않음 (`test_the_original_models_file_keeps_its_record_but_is_not_copied`) |

## 6. 다른 엔진의 교차 검토 (Codex, 읽기 전용) 와 처리

구현이 끝난 뒤 Codex CLI(`-s read-only`)에 변경분 전체(`pack.py` · `linear.py` 훅 · `pack_index.py` · `yolo_tune.py onnx` ·
화면 · SPEC/ARCHITECTURE 문장)를 넘겨 반박 검토를 받았다. 15건 (MAJOR 12 · MINOR 3). 이미 만든 두 패키지와 ONNX 5개에
해당하는지 먼저 가렸다: 두 레시피는 가중치만 양자화하고(모든 규칙이 `activation: null`, KV 없음, `mma: fp32_fma`), ONNX 에 붙은
PyTorch 기록 5개는 층별 배정이 ONNX 와 87층 모두 같았다. 해당하는 것은 ONNX IR 버전(11번) 하나였다.

| # | 지적 | 이번 기록에 해당? | 처리 |
|:-:|---|---|---|
| 1 | `PackedLinear` 가 활성값 양자화 · KV 를 재현하지 않음 | 아니오 (가중치만) | 그런 레시피는 거부 (`test_what_a_package_cannot_run_is_refused` · `test_a_stored_recipe_with_a_kv_spec_does_not_load`) |
| 2 | −0 으로 반올림된 가중치가 +0 으로 풀림, `torch.equal` 은 같다고 봄 | 예 (모든 패키지) | 사실이다. 검사를 비트 비교(0 의 부호만 같게 봄)로 바꾸고 문서에 예외를 적음. +0 에서 시작하는 합에는 영향이 없다 (`test_a_weight_rounded_to_negative_zero_…`). 이미 통과한 `torch.equal` 은 같은 뜻이라 다시 만들 필요 없음 |
| 3 | 같은 폴더에 다시 쓰면 이전 PPL 검사를 물려받음 | 아니오 (새 폴더) | `build` 는 빈 폴더만 받고, `write` 는 이전 manifest 를 지움 (`test_a_new_package_does_not_inherit_old_checks`) |
| 4 | 목록 도구가 기록된 차이 값을 그대로 믿음 · 검사 층 수 · 해시 대조 없음 | 아니오 (값이 맞음) | 두 PPL 로 차이를 다시 계산, 층 수 · 레시피 해시 · `quant_config.json` 해시 대조 (`test_files_that_disagree_with_each_other_are_not_listed`) |
| 5 | 목록 키가 레시피 SHA 뿐 — 모델이 달라도 같은 레시피 파일이면 충돌 | 아니오 (두 레시피가 다름) | 목록을 (모델, 레시피 SHA) 로 찾음 (화면 · 검사 스크립트) |
| 6 | 고정소수점 정수 형식(`frac_bits`)의 소수부를 버림 | 아니오 | 그런 형식은 코드 대신 가중치 그대로 저장 |
| 7 | 코드가 없는 층을 BF16 으로 좁혀 정밀도를 잃을 수 있음 | 아니오 (HQQ 4층은 BF16 값) | TriCast 가 곱하는 값을 저장하고, BF16 으로 정확할 때만 좁힘 (`test_a_dense_weight_is_kept_exactly_…`) |
| 8 | BF16 · FP16 밖의 출력 형식(FP8 등)을 무시 | 아니오 (BF16) | 거부 |
| 9 | 두 이름에 공유된 Linear 가 한쪽만 바뀜 | 아니오 | 거부 (`named_modules(remove_duplicate=False)` 로 찾음) |
| 10 | `.to(bfloat16)` 뒤 bias 가 있는 층이 오류 | 아니오 (bias 없음) | bias 를 FP32 로 계산하고, 층의 dtype 변환은 장치 이동만 하도록 (가중치 · 코드 · bias 를 반올림하지 않음) |
| 11 | INT4 를 넣고도 ONNX IR 버전을 9 로 둠 (INT4 는 IR 10 부터) | **예** (ONNX 4개) | IR 10 으로 올리고 INT 셋업 4개를 다시 만듦 — 결과 같음, 화면 파일 교체 (§3.2) |
| 12 | PyTorch 기록이 다른 배정의 것일 수 있음 | 아니오 (5/5 배정 같음) | 기록이 없거나 배정이 다르면 실패 |
| 13 | 범위 밖 영점을 int16 으로 무조건 줄임 | 아니오 | int16 밖이면 int32 |
| 14 | 저장 dtype 으로 바꾸기 전에만 검사 | 아니오 | 코드 범위를 먼저 확인하고, 저장한 정수에서 다시 계산해 대조 |
| 15 | 화면이 PPL 차이의 원인을 "다른 GPU" 로 단정 | 문구 | "따로 보정 · 측정한 실행의 값" 으로 바꿈 |

고친 뒤 패키지 관련 테스트(`tests/test_pack.py` · `app/tests/test_pack_index.py` · `app/tests/test_yolo_onnx.py` ·
`tests/test_nn_patch.py`) 86건 통과 (bath → cpu_cap, 트리 `qzbench_12`).
