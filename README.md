# TriCast

**저정밀 행렬 연산기의 산술 — 수 형식, 양자화, 그리고 텐서코어 누산기 자체 — 을 CUDA 코어에서 비트 단위로
에뮬레이트하고, 그 선택이 언어 모델 품질을 어떻게 바꾸는지 잰다.**

[English](README.en.md) · [엔진 계약서](docs/design/ENGINE.md) · [지원 현황](support_matrix.yaml) ·
[전체 결과](docs/results/qwen3_0.6b.md) · [데모 대본](docs/demo/DEMO.md)

> 레시피의 값 하나를 바꾸면, 커널을 다시 쓰지 않고 같은 모델에서 그 산술이 품질과 누산 오차에 미치는 영향을
> 실행 환경 기록과 함께 돌려받는다.

**목차** · [같은 내적, 네 개의 답](#같은-내적-네-개의-답) · [왜 만들었나](#왜-만들었나) ·
[무엇을 바꿀 수 있나](#무엇을-바꿀-수-있나) · [빠른 시작](#빠른-시작) · [결과](#결과-qwen3-06b) ·
[3분 데모](#3분-데모) · [검증](#어떻게-검증하나) · [저장소 구조](#저장소-구조) · [한계](#한계)

---

## 같은 내적, 네 개의 답

32개짜리 FP8(E4M3) 내적 하나를 네 가지 데이터패스 방식으로 누산한 결과다
(`python examples/one_dot_product.py` 가 정확 레퍼런스로 CPU 에서 그대로 재현한다).

```
누산기                           결과 (fp32)              비트
fp64, 한 번 반올림               291.274658203125         0x4391a328
Blackwell FP8   CoFDA F=25       291.2746276855469        0x4391a327
Hopper FP8      CoFDA F=13       291.25                   0x4391a000
좁은 누산기     CoFDA F=7        290.0                    0x43910000
```

텐서코어는 곱을 하나씩 더하지 않는다. 곱 한 묶음을 가장 큰 지수에 맞춰 정렬하고, 정해진 소수 비트 폭 `F`
아래를 전부 잘라 낸 뒤, 남은 비트를 정확히 더하고 한 번 정규화한다.

```
             묶음 최대 지수 Emax = 8, F = 10
p0  + 1.101 · 2^8      | 1.1010000000 |
p1  − 1.011 · 2^5      | 0.0010110000 |        오른쪽으로 3칸
p2  + 1.111 · 2^-1     | 0.0000000011 | 11     오른쪽으로 9칸; F 밖의 비트는 버려진다
                         └─ F = 10 ──┘
```

이 폭, 묶음 크기, 진행 중인 누산값이 절단에 함께 들어가는지는 모두 설계 변수다. 스펙 시트에는 보이지 않지만
모델 품질을 움직인다. TriCast 는 그 하나하나를 레시피의 값으로 정할 수 있게 한다.

## 왜 만들었나

저정밀 모델과 NPU 연산기를 설계·검증하는 사람은 비트 폭·스케일링·누산 방식·희소성·이상치 처리나 GPU 세대를
바꿀 때마다, 그 효과를 확인하려고 코드와 CUDA 커널을 다시 쓴다. 팀 인터뷰 10건 가운데 실무자 5명 중 3명이
그렇게 말했다 (로그 1, 9, 10).

> “비트 폭 하나 바꾸고, 희소 비율 하나 틀고, 이상치 보존 방식 하나 수정할 때마다 CUDA 커널을 다시 깎아야
> 합니다.” — 인터뷰 로그 10

TriCast 는 이 선택들을 레시피의 값으로 분리한다. 근거와 범위는 [문제 정의서](docs/PROBLEM.md),
[인터뷰 기록](docs/research/interviews.md), [제품 스펙](docs/SPEC.md)에 있다.

## 무엇을 바꿀 수 있나

| 층 | 선택지 |
|---|---|
| **수 형식** | 임의의 float `ExMy` (IEEE / finite-NaN / fnuz / 특수값 없음, bias 지정, 서브노멀 on/off), 정수·고정소수점 (`intN`, `uintN`, `frac=F`), 2의 거듭제곱 스케일 (E8M0). 기본 등록: fp32 · tf32 · bf16 · fp16 · fp8 e4m3/e5m2 (+fnuz) · fp6 e3m2/e2m3 · fp4 e2m1 · ue4m3 · int8/4/2 · mxint8/4 |
| **반올림** | 최근접 짝수, 최근접 0에서 멀리, 0 쪽, 올림, 내림, 확률적 반올림 (noise 와 비트 수 지정) |
| **스케일** | 텐서 / 행(채널·토큰) / 그룹 / 2-D 블록; absmax, 2의 거듭제곱 내림·올림(MX), MSE 탐색, 백분위, **Four-over-Six**; 2단계(NVFP4); 정수·실수 zero point |
| **스킴** | MXFP8/6/4, MXINT8/4, **NVFP4**, 블록 부동소수점 (MSFP12/16, `bfp<m>_b<block>`), FP8 텐서/행/그룹/블록 (DeepSeek), INT8, INT4 g128 ± zero point, **KIVI** 2/4비트 KV 캐시 |
| **보정** | 정적 observer (min-max, **EMA**, Transformer-Engine delayed history, 백분위, MSE); WikiText-2 / C4 / Pile 표본 |
| **알고리즘** | **GPTQ** (모든 형식, 그룹, act-order, 순차), **AWQ** · SmoothQuant (입력 공유 그룹), Hadamard · 랜덤 Hadamard 회전, QAT 용 STE |
| **가중치 구조** | N:M (예: 2:4)·비율 기반 크기 희소성; 이상치를 고정밀 형식으로 따로 보존 (SpQR 방식, 별도 fp32 경로) |
| **MMA 누산** | CoFDA (C-fused / C-decoupled), GDFS (2단계 그룹 합), DeepSeek 방식 FP32 승격, IEEE FP32 FMA 체인, FP64, 정확 정수; 블록 스케일은 곱·그룹·승격·epilogue 중 어디서 적용할지 선택 |
| **하드웨어 프리셋** | Hopper FP8 (F=13, CS=32), Ada FP8, Blackwell FP8 (F=25), Blackwell FP4 (GDFS G=6 F=35), DeepSeek FP8 승격 — 모든 프리셋에 출처 기록 |
| **모델과 과제** | Hugging Face causal LM 전반 (`nn.Linear` 레이어, 레이어별 규칙), WikiText-2 perplexity, lm-eval 전 과제, 레이어별 오차 리포트 (MSE, SQNR, 코사인, logits KL, fp64 누산 대비 누산기 ULP 오차) |

## 빠른 시작

### 설치

```bash
pip install -e ".[triton,eval]"          # Triton 커널은 Linux + CUDA; CPU 에서는 레퍼런스로 동작
pip install -e ".[agent]"                # `tricast agent` 를 Claude API 로 쓸 때만
```

### Python

```python
import torch, tricast
from tricast.eval.ppl import perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

report = tricast.patch_model(model, "hopper_fp8_w8a8")   # 모든 decoder linear 를 Hopper 누산으로
print(perplexity(model, tok, dataset="wikitext2"))
```

### 명령줄

| 명령 | 하는 일 |
|---|---|
| `tricast formats` · `schemes` · `presets` | 수 형식, 양자화 스킴, 하드웨어 프리셋 목록 |
| `tricast cast` | 값을 수 형식 격자에 반올림 |
| `tricast recipe-check <이름·경로>` | 레시피 검증 — 잘못된 필드의 경로를 알려 준다 |
| `tricast ppl --model M --recipe R` | WikiText-2 perplexity |
| `tricast eval --model M --recipe R --tasks hellaswag,coqa` | lm-eval 과제 |
| `tricast report --model M --recipe R` | 레이어별 MSE·SQNR·코사인·누산기 ULP 오차와 모델 logits KL |
| `tricast run CONFIG.yaml` | 레시피·스윕 설정 실행 (중단한 곳부터 이어서) |
| `tricast agent "<요청>"` | 자연어 요청을 검증된 레시피로 |
| `tricast-lm-eval …` | lm-eval CLI 그대로 (`--model tricast --model_args pretrained=…,recipe=…`) |

이름으로 부르는 레시피 26개가 `src/tricast/recipes/` 에 있다 (`hopper_fp8_w8a8`, `nvfp4_w_a`, `mxfp4_w_a`,
`fp8_2of4_sparse`, `nvfp4_outliers`, `w4a16_g128_zp_gptq`, `kivi2_kv` 등).

### 레시피 예

```yaml
name: my_accelerator
defaults:
  weight:     {scheme: nvfp4}
  activation: {format: fp8_e4m3, granularity: row}
  mma:        {algorithm: cofda, f_bits: 11, chunk_size: 64, c_mode: decoupled}
  sparsity:   {kind: "n:m", n: 2, m: 4}     # 가중치 2:4 희소화
  outliers:   {fraction: 0.005, format: bf16}  # 크기 상위 0.5% 를 bf16 으로 따로 보존
overrides:
  - layers: "0,-1"                 # 첫·마지막 decoder 블록의 linear 는 양자화하지 않는다
    skip: true
  - modules: [down_proj]
    weight: {scheme: mxfp8_e4m3}
kv: {preset: kivi2, mode: cache}   # KV 캐시는 모든 블록에서 KIVI-2 (kv.layers 로 좁힌다)
```

## 결과 (Qwen3-0.6B)

WikiText-2 test 전체 (2048 토큰 창 146개), bf16 모델, A100 한 장 (네이티브 perplexity 20.966). 모든 행에서
가중치와 활성은 텐서당 스케일 하나의 FP8 E4M3 이고, 누산기만 바뀐다.

| 누산기 | perplexity | 네이티브 대비 |
| --- | ---: | ---: |
| 정확 (fp64) | 21.167 | +0.96% |
| Blackwell, F=25 | 21.184 | +1.04% |
| Hopper, F=13, 32개 묶음 | 21.204 | +1.13% |
| F=7, 누산값을 23비트 레지스터에 따로 | 21.275 | +1.47% |
| F=7, 누산값도 묶음마다 함께 절단 | **29.158** | **+39.1%** |

- 마지막 두 행은 진행 중인 누산값이 묶음의 절단에 함께 들어가는지만 다르다. 그 차이 하나가 +1.5% 와 +39% 를
  가른다.
- 누산 오차를 bf16 출력의 ULP 로 재면 (V100, 1024 토큰) Hopper 는 평균 2.62, F=7 decoupled 는 60.5 ULP 다.
  그런데도 decoupled 의 logits KL 은 Hopper 와 거의 같다 (0.024 대 0.022). ULP 예산만으로는 모델 품질을 예측할
  수 없다.
- 재학습 없이 모든 linear 를 크기 기준 2:4 로 자르면 PPL 이 65,188 로 무너진다. TriCast 없이 PyTorch 로 같은
  가지치기를 해도 앞 4개 창에서 같은 PPL (73,434) 이 나오므로 구현이 아니라 가지치기 자체의 결과다. 크기 상위 0.5% 를 bf16 으로 보존한
  NVFP4 는 26.18 로, 보존하지 않은 NVFP4 (26.22) 와 거의 같다.
- 에뮬레이트한 bf16 passthrough 는 네이티브 모델을 0.004% 이내로 재현한다.

레시피 26개의 PPL, 블록 형식·GPTQ·AWQ·KIVI·lm-eval(HellaSwag, CoQA) 결과와 각 실행의 환경은
[docs/results/qwen3_0.6b.md](docs/results/qwen3_0.6b.md) 에 있다.

## 3분 데모

```bash
python examples/demo_qwen3.py --quick     # A100 한 장에서 약 40분 → report.md · results.json · env.json
```

| 시간 | 보여 주는 것 | 핵심 수치 |
|---|---|---|
| 0:00 | 문제와 입력 | 인터뷰 인용: 조합을 바꿀 때마다 커널을 다시 깎는다 |
| 0:25 | 같은 가중치를 9개 형식으로 | 같은 4비트라도 스케일 방식에 따라 SQNR 18.6–21.2 dB |
| 1:00 | 같은 FP8, 다른 누산 | F=7 fused 상대 오차 0.158, decoupled 0.0078 — 20배 |
| 1:40 | 모델 품질 (앞 16개 윈도우 PPL) | F=7 fused 27.58 대 decoupled 20.51, 재학습 없는 2:4 는 47,112 |
| 2:15 | 생성 문장 비교 | F=7 fused 와 MXFP4 에서 깨진 단어가 나온다 |
| 2:45 | 결론과 산출물 | 비트 단위 정확성은 데모가 아니라 GPU 테스트 896개와 NADPE 골든 벡터 1716개로 따로 검증 |

최근 실행 (커밋 `9e9bfed`, A100): `DEMO_DONE status=ok`, 11개 레시피 모두 정상. 화면 구성, 말할 내용, 발표 때
밝혀야 할 주의 사항은 [docs/demo/DEMO.md](docs/demo/DEMO.md) 에 있다.

## 어떻게 검증하나

에뮬레이션은 그것이 정말 의도한 산술일 때만 쓸모가 있다. 아래 항목은 전부 이 저장소의 테스트다.

| 확인 | 근거 |
|---|---|
| 레퍼런스 캐스트 vs PyTorch 네이티브 변환 | fp8 4종, bf16, fp16 — 형식마다 무작위 값 10만 개 + 경계값에서 비트 일치 |
| 레퍼런스 MX 양자화 vs `microsoft/microxcaling` | 비트 일치 (even / nearest / floor). 단, microxcaling 이 fp32 `log2` 로 잘못 분류하는 입력은 제외 |
| 레퍼런스 MMA vs **NADPE** CUDA 커널 (MICRO'26, 단독 빌드) | **1716 / 1716** 케이스 비트 일치 — FP8 CoFDA / C-decoupled / GDFS, NVFP4, MXFP4 |
| Triton 커널 vs 레퍼런스 | GPU 테스트 (`tests/gpu`: 양자화 + MMA, 무작위·경계·특수값) A100 과 V100 에서 각각 896개 통과 (`9e9bfed`) |
| 실제 크기 GEMM 에서 Triton MMA vs NADPE | 2048×1024×3072, 2048×3072×1024, 4096³ 에서 CoFDA·C-decoupled·GDFS 모두 비트 일치 (`scripts/bench/bench_mma_vs_nadpe.py`) |
| CPU 테스트 (`pytest --ignore=tests/gpu`) | Linux torch 2.8 에서 2,337개 통과 (`82db449`) |

Triton 은 libdevice 를 flush-to-zero 로 링크한다. 그래서 서브노멀을 만날 수 있는 모든 fp32 연산은 IEEE PTX 로
계산하고, 레퍼런스는 fp32 연산 하나하나를 fp64 로 계산한 뒤 한 번 반올림한다 — 결과가 장치에 따라 달라지지
않는다.

## 저장소 구조

```
TriCast/
├─ src/tricast/                 라이브러리
│  ├─ formats.py, rounding.py   수 형식과 반올림 명세
│  ├─ quant/                    양자화 명세·API·observer, 희소성·이상치 (structure.py)
│  ├─ mma/                      MMA 누산 명세와 하드웨어 프리셋
│  ├─ reference/                정확 레퍼런스 (fp64 / int64) — 산술의 정의
│  ├─ kernels/                  Triton 커널: 양자화, MMA, 디코드용 작은 M
│  ├─ nn/                       Hugging Face 모델 패치 (EmuLinear, patch_model)
│  ├─ kv/                       KV 캐시 양자화 (KIVI)
│  ├─ weight_quant/             GPTQ
│  ├─ transforms.py             AWQ · SmoothQuant · Hadamard
│  ├─ calibration.py            보정 데이터와 observer 적합
│  ├─ eval/                     perplexity, lm-eval 어댑터, 스윕 러너, 실행 환경 기록
│  ├─ analysis.py               레이어별 오차와 누산기 ULP 리포트
│  ├─ agent/ · tools/ · rag.py  자연어 요청 → 레시피 에이전트
│  ├─ recipes/                  이름으로 부르는 레시피 26개
│  └─ cli.py                    tricast 명령
├─ tests/        CPU 테스트 · gpu/ (Triton 비트 일치) · data/nadpe/ (골든 벡터) · harness/ (골든 케이스)
├─ configs/      e2e/ (결과 문서를 재현하는 설정) · sweeps/
├─ scripts/      bench/ (성능) · e2e/ (결과 정리·교차검증) · nadpe_oracle/ (독립 구현 대조)
├─ examples/     demo_qwen3.py (3분 데모) · one_dot_product.py
├─ evals/        자연어 요청 파싱 평가셋 30건, 채점기, 판정 프롬프트
├─ docs/         스펙·문제 정의·온톨로지·인터뷰·엔진 계약서·결과·데모·스파이크
├─ AGENTS.md     코딩 에이전트 작업 규칙 (CLAUDE.md 는 이 파일을 불러온다)
└─ support_matrix.yaml   구현·검증 상태의 원천
```

브랜치는 `develop` (작업) 과 `main` (검증을 마친 상태를 fast-forward) 두 개다. 변경 내역은
[CHANGELOG.md](CHANGELOG.md) 에 있다.

### AI캡스톤 과제 산출물

| 강의 | 산출물 |
|---|---|
| 2 — 인터뷰와 온톨로지 | [docs/research/interviews.md](docs/research/interviews.md), [docs/ontology.yaml](docs/ontology.yaml) |
| 4 — 문제 정의와 스펙 | [docs/PROBLEM.md](docs/PROBLEM.md), [docs/SPEC.md](docs/SPEC.md) (수용 기준 AC1–AC10), [docs/spikes/](docs/spikes/cuda_core_bit_exact_emulation.md) |
| 5 — 골든 케이스와 평가 | [tests/harness/golden_cases.yaml](tests/harness/golden_cases.yaml), [evals/](evals/README.md) |
| 작업 규칙 | [AGENTS.md](AGENTS.md) — 절대 규칙, 금지 사항, 완료의 정의 |

## 한계

- attention 의 `QKᵀ` / `PV` 행렬곱은 아직 에뮬레이트하지 않는다. KV 캐시 양자화는 지원한다.
- 하드웨어 프리셋은 공개된 측정(NADPE)에서 왔다. TriCast 가 실리콘과 직접 대조하는 도구(`tricast.probe`)는
  계획 단계다.
- CUDA 코어 에뮬레이션은 모델링 대상인 텐서코어 GEMM 보다 느리다. A100 에서 Hopper FP8 누산기는 0.15 TMAC/s
  (NADPE CUDA 커널 0.15), Qwen3-0.6B 의 2048 토큰 창 하나에 약 6초, 토큰 하나를 디코딩하는 데 약 0.1초(V100
  0.17초)가 걸리고, 그중 대부분은 호스트에서 커널을 띄우는 시간이다. 새 구성을 처음 쓸 때 Triton 커널을 컴파일한다.
- 자연어 요청을 레시피로 바꾸는 에이전트(`tricast agent`)는 가짜 모델 client 로 테스트했다. 실제 Claude API
  경로에는 `pip install -e ".[agent]"` 와 `ANTHROPIC_API_KEY` 가 필요하다. 둘 중 하나가 없으면 `--llm auto` 는
  오프라인 파서를 쓰고, 어떤 파서가 돌았는지 출력한다.

## 바탕이 된 작업

NADPE / MMA-Emu (MICRO'26, *Not All Dot Products Are Equal*), microsoft/microxcaling, OCP Microscaling 스펙,
NVIDIA NVFP4, DeepSeek-V3 (FP8 승격), GPTQ, AWQ, SmoothQuant, QuaRot, KIVI, Four Over Six. MIT 라이선스.
