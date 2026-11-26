# YOLO11n 크기 스윕과 배포 크기 추천 · 2026-11-26

## 1. 사전 등록 (결과를 보기 전, 2026-11-26 08:50Z)

**질문.** 같은 층별 민감도 표로 평균 4.5–7비트 예산을 풀면, 균일 INT8(평균 8.05비트)보다 작은 크기에서
원본에 가까운 COCO mAP를 지키는 지점이 있는가. 있다면 그중 가장 작은 크기를 배포 크기로 추천한다.

- **표:** `/scratch/uceeeee/tricast/qzyolo_runs/r1/table.json` (SHA-256 `dcdd83ec…72aa`, 2026-11-11 측정). 새로 재지 않는다.
- **셋업:** `scripts/quantizer/yolo_tune.py`의 `SWEEP` — `auto_b450`, `auto_b500`, `auto_b550`, `auto_b600`, `auto_b700`
  (예산 = 평균 비트 9/2, 5, 11/2, 6, 7). 기존 셋업(`native`, 균일 INT8/4/3, 경험칙, `auto_iso_*`)은 다시 재지 않는다.
- **풀이·평가:** 기존과 같은 다중 선택 배낭(`solve`)과 같은 평가(`evaluate`: COCO val2017 중 보정 128장을 뺀 4,872장,
  AP50–95, CPU FP32, seed 42). 같은 표로 다시 푼 `auto_iso_*` 세 해가 기존 해와 같아야 한다(결정론 확인).
- **추천 규칙 (고정):** Auto-tune 셋업 가운데 원본 대비 mAP50–95 손실이 **1.0 포인트(0–100 단위) 이하**인 것 중
  평균 비트가 가장 작은 셋업. 해당 셋업이 없으면 추천하지 않는다. 손실 허용치는 사용자 입력이며 1.0은 이 기록의 기본값이다.
- **읽는 법:** 추천은 평가 4,872장으로 고른다. 이것은 배포 전 수용 시험이며, 고르지 않은 별도 데이터에서의 일반화 주장이 아니다.
  균일 INT8과의 비교는 같은 4,872장, 같은 전처리·후처리에서의 크기 차이와 mAP 차이로만 말한다.

실행 트리: 작업 트리 스냅샷 `/scratch/uceeeee/tricast/qzsweep_20261126T0840Z` (HEAD `26fc532` + 미커밋 export 작업 +
`SWEEP`), `yolo_tune.py` SHA-256 `5f3a131c26f219a4f3cd3fab6a4b1b032b615e50bed93c786bc8ea95cc82c832`.
결과 파일의 `env.git_dirty`는 그래서 true다. ONNX·배포 단계는 두 번째 스냅샷 `qzsweep_20261126T0853Z`
(`yolo_tune.py` SHA-256 `f689f2ae…6c0c`, `--int3` 저장 옵션 추가)에서 돌렸다. 커밋한 파일(`62d4e615…5630`)은 그 뒤
리뷰 지적 두 가지만 더했다: `solutions.json`에 없는 셋업을 KeyError 대신 메시지로 거절하고, 검증기가 `budget_bits` 없는
스윕 셋업을 KeyError 없이 문제로 보고한다. 측정 경로의 연산은 바뀌지 않았다.

## 2. 결과

결정론: 같은 표로 다시 푼 `auto_iso_u4`·`auto_iso_h84`·`auto_iso_u3`의 배정·예산·예측 D가 2026-11-11 해와 모두 같았다.
`solutions.json` SHA-256 `9a8853a1…44a3f`. 평가는 bath·bradford(16코어, CPU FP32), 셋업당 413–491초.

| 셋업 | 예산 (평균 비트) | 원본 대비 크기 | 배정 (FP16 · INT8 · INT4 · INT3 층) | 측정 D | mAP50–95 | 원본 대비 |
|---|---:|---:|---|---:|---:|---:|
| 원본 FP16 | 16 | 100% | — | 0 | **39.45** | — |
| INT8 균일 (기존) | 8.051 | 50.3% | 0 · 87 · 0 · 0 | 0.0032 | 39.37 | −0.08 |
| `auto_b700` | 7 | 43.7% | 9 · 69 · 3 · 6 | 0.0055 | 35.98 | −3.47 |
| `auto_b600` | 6 | 37.5% | 1 · 65 · 12 · 9 | 0.0199 | 34.21 | −5.24 |
| `auto_b550` | 11/2 | 34.4% | 1 · 60 · 17 · 9 | 0.0275 | 33.55 | −5.90 |
| `auto_b500` | 5 | 31.2% | 1 · 47 · 29 · 10 | 0.0508 | 31.07 | −8.38 |
| `auto_b450` | 9/2 | 28.1% | 0 · 39 · 36 · 12 | 0.0923 | 28.63 | −10.82 |
| `auto_iso_h84` (11-11) | 4.100 | 25.6% | 0 · 24 · 47 · 16 | — | 24.45 | −15.00 |
| `auto_iso_u4` (11-11) | 4.051 | 25.3% | 0 · 23 · 47 · 17 | — | 23.83 | −15.62 |
| INT4 균일 (기존) | 4.051 | 25.3% | 0 · 0 · 87 · 0 | — | 1.09 | −38.36 |

**판정 (사전 등록 규칙):** 손실 1.0 이하인 Auto-tune 셋업이 측정 범위(25–44%)에 없다 → **추천하지 않는다.**
이 기준을 만족하는 가장 작은 측정 셋업은 균일 INT8(50.3%)이다. 곡선은 크기에 따라 단조롭게 오르며(측정 D도 같은 순서),
INT4 크기에서 무너지는 균일 양자화(1.09)와 INT8(39.37) 사이를 층별 배정으로 채운다. 7비트에서도 INT3 6층·INT4 3층이 남아
손실 3.47이 생긴다. 층 하나씩 잰 D의 합(예측 0.0064)과 전체 측정 D(0.0055)는 가깝지만, 이 범위의 D가 mAP에서는
수 포인트 차이다. 방법 후보(RTN INT8/4/3)를 넓히는 것은 별도 사전 등록 실험으로 다룬다.

### 배포 파일과 런타임

각 셋업의 ONNX(`yolo_tune.py onnx`)를 onnxruntime 1.23.2 CPU로 같은 4,872장에서 다시 채점했다. 모든 파일이
PyTorch fake-quant 평가와 mAP 차이 1.3×10⁻⁶ 이하였다. INT3 층은 기본으로 3비트씩 묶어 저장한다.

| 셋업 | ONNX 크기 | INT8 · INT4 · INT3 층 | INT3 저장 | ORT mAP |
|---|---:|---|---|---:|
| `auto_b450` | 1,674,869 B | 39 · 36 · 12 | 3비트 묶음 | 28.63 |
| `auto_b500` | 1,864,945 B | 47 · 29 · 10 | INT4 칸 (`--int3 int4`) | 31.07 |
| `auto_b550` | 1,998,721 B | 60 · 17 · 9 | 3비트 묶음 | 33.55 |
| `auto_b600` | 2,161,964 B | 65 · 12 · 9 | 3비트 묶음 | 34.21 |
| `auto_b700` | 2,548,556 B | 69 · 3 · 6 | 3비트 묶음 | 35.98 |

원본 FP32 ONNX는 10,567,925 B다. `auto_b500`의 3비트 묶음 파일(1,837,032 B, ORT mAP 31.07)도 만들었지만, TensorRT 파서가
비트 해제 그래프를 읽지 못한다는 기존 기록 때문에 세 런타임이 같은 파일을 쓰도록 INT4 칸 파일을 화면에 올렸다.
런타임 확인 대상(`auto_b500`, 스윕의 가운데 크기)은 런타임 결과를 보기 전에 정했다.

| 런타임 | 장치 | 실행 파일 | mAP50–95 | ORT 대비 | 판정 (±0.1) |
|---|---|---|---:|---:|---|
| ONNX Runtime 1.23.2 | Xeon CPU (oslo) | ONNX `f73d6fbf…` | 31.0675 | 기준 | — |
| OpenVINO 2026.4.1 | Xeon Gold 6126 CPU (bath) | IR XML+BIN (BIN `c1bcc190…`) | 31.0673 | −0.00015 | 통과 |
| TensorRT 10.13.3.9 | H200 NVL (berlin) | engine plan | 31.0675 | +0.00003 | 통과 |

OpenVINO는 Split→Conv 최적화 오류를 피하려고 Conv 입력 1곳을 동등한 채널 Gather로 바꿨고(`--gather-split-conv-inputs`),
IR의 86개 코드 묶음이 원본과 같았다. 시연 사진 3장의 원시 헤드 출력은 비트 단위로 같지 않았다(최대 0.0009, cosine 1.0, 신뢰
후보·클래스 동일). TensorRT는 원 그래프에서 plan을 만들었다. A100 두 대(turin 응답 없음, geneva는 다른 세션이 GPU 한도를
사용)를 쓸 수 없어 H200에서 실행했다. 처리 속도·메모리·전력은 비교하지 않았다.

### 화면

앱(`app/web/tune.html`)은 같은 규칙과 허용치 1.0으로 검출 모델의 크기 버튼에 "배포 추천"을 표시한다. 이번 기록에서는 해당 크기가
없다는 문장과 기준을 만족하는 가장 작은 측정 셋업(INT8 균일)을 보인다.

## 3. 재현

UCL CPU 호스트(`cpu_cap`, 16코어)에서 실행했다. 경로는 클러스터 기준이며 로그는 각 단계의 `YOLO_*_DONE` 또는 `YOLO_FAIL`로 끝난다.

```bash
W=/scratch/uceeeee/tricast/qzyolo_runs
cp $W/r1/table.json $W/r2/                      # 같은 표 (SHA-256 dcdd83ec…)
python scripts/quantizer/yolo_tune.py solve    --work $W/r2
python scripts/quantizer/yolo_tune.py evaluate --work $W/r2  --setups auto_b450,auto_b500,auto_b550
python scripts/quantizer/yolo_tune.py evaluate --work $W/r2b --setups auto_b600,auto_b700   # 같은 solutions.json
cp $W/r2/{table,solutions}.json $W/r2o/          # ONNX 단계마다 새 작업 폴더 (r2o, r2c, r2d)
python scripts/quantizer/yolo_tune.py onnx     --work $W/r2o --setups <셋업> --more $W/r2 $W/r2b
python scripts/quantizer/yolo_tune.py onnx     --work $W/r2d --setups auto_b500 --int3 int4 --more $W/r2 $W/r2b
python scripts/quantizer/yolo_tune.py export   --work $W/r2  --more $W/r2b $W/r1 $W/r1/e1
python scripts/quantizer/pack_index.py --out app/web/demo/autotune/packages.json --llm … \
    --onnx yolo11n=<r1/onnx> --onnx yolo11n=<셋업마다 고른 ONNX 폴더> --onnx-copy app/web/demo/autotune/packages/yolo11n \
    --runtime yolo11n:auto_b500=<OpenVINO COCO result.json> --runtime yolo11n:auto_b500=<TensorRT COCO result.json>
```

OpenVINO·TensorRT 확인은 2026-11-26 런타임 기록([runtime_interop](2026-11-26_runtime_interop.md))과 같은
방법이다: OpenVINO는 `scripts/quantizer/openvino_export.py --task detector --gather-split-conv-inputs`로 만든 IR을
CPU로, TensorRT는 `scripts/quantizer/tensorrt_export.py`로 만든 plan을 A100에서 실행해 같은 4,872장으로 채점한다.
실행 스크립트·로그: 로컬 `runs/20261126T0840Z_yolo_sweep/`, 클러스터 `/scratch/uceeeee/tricast/deploy_20261126T0840Z/`.
