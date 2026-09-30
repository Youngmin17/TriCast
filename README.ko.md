# TriCast

**저정밀 행렬 연산기의 산술 — 수 형식, 양자화, 그리고 텐서코어 누산기 자체 — 을 CUDA core 에서 비트 단위로
에뮬레이트하고, 그것이 언어 모델 품질을 어떻게 바꾸는지 잰다.**

[English README](README.md) · [엔진 계약서](docs/design/ENGINE.md) · [지원 현황](support_matrix.yaml)

---

## 같은 내적, 네 개의 답

32개짜리 FP8(E4M3) 내적 하나를 네 가지 데이터패스 방식으로 누산한 결과다
(`python examples/one_dot_product.py` 가 정확 레퍼런스로 CPU 에서 그대로 재현한다):

```
누산기                           결과 (fp32)              비트
fp64, 한 번 반올림               291.274658203125         0x4391a328
Blackwell FP8   CoFDA F=25       291.2746276855469        0x4391a327
Hopper FP8      CoFDA F=13       291.25                   0x4391a000
좁은 누산기     CoFDA F=7        290.0                    0x43910000
```

텐서코어는 곱을 하나씩 더하지 않는다. 곱 한 묶음을 가장 큰 지수에 맞춰 정렬하고, 정해진 소수 비트 폭 `F` 아래를
전부 잘라 낸 뒤, 남은 비트를 정확히 더하고 한 번 정규화한다:

```
             묶음 최대 지수 Emax = 8, F = 10
p0  + 1.101 · 2^8      | 1.1010000000 |
p1  − 1.011 · 2^5      | 0.0010110000 |        오른쪽으로 3칸
p2  + 1.111 · 2^-1     | 0.0000000011 | 11     오른쪽으로 9칸; F 밖의 비트는 버려진다
                         └─ F = 10 ──┘
```

이 폭, 묶음 크기, 진행 중인 누산값이 절단에 함께 들어가는지 여부는 모두 설계 변수다. 스펙 시트에는 보이지 않지만
모델 품질을 움직인다. TriCast 는 그 하나하나를 값으로 정할 수 있게 한다.

## 무엇을 모델링하나

| 층 | 선택지 |
|---|---|
| **수 형식** | 임의의 float `ExMy` (IEEE / finite-NaN / fnuz / 특수값 없음, bias 지정, 서브노멀 on/off), 정수·고정소수점 (`intN`, `uintN`, `frac=F`), 2의 거듭제곱 스케일 (E8M0). 기본 등록: fp32 · tf32 · bf16 · fp16 · fp8 e4m3/e5m2 (+fnuz) · fp6 e3m2/e2m3 · fp4 e2m1 · ue4m3 · int8/4/2 · mxint8/4 |
| **반올림** | 최근접 짝수, 최근접 0에서 멀리, 0 쪽, 올림, 내림, 확률적 반올림 (noise 와 비트 수 지정) |
| **스케일** | 텐서 / 행(채널·토큰) / 그룹 / 2-D 블록; absmax, 2의 거듭제곱 내림·올림(MX), MSE 탐색, 백분위, **Four-over-Six**; 2단계(NVFP4); 정수·실수 zero point |
| **스킴** | MXFP8/6/4, MXINT8/4, **NVFP4**, 블록 부동소수점 (MSFP12/16, `bfp<m>_b<block>`), FP8 텐서/행/그룹/블록 (DeepSeek), INT8, INT4 g128 ± zero point, **KIVI** 2/4비트 KV 캐시 |
| **보정** | 정적 observer (min-max, **EMA**, Transformer-Engine delayed history, 백분위, MSE); WikiText-2 / C4 / Pile 표본 |
| **알고리즘** | **GPTQ** (모든 형식, 그룹, act-order, 순차), **AWQ** · SmoothQuant (입력 공유 그룹), Hadamard · 랜덤 Hadamard 회전, QAT 용 STE |
| **MMA 누산** | CoFDA (C-fused / C-decoupled), GDFS (2단계 그룹 합), DeepSeek 방식 FP32 승격, IEEE FP32 FMA 체인, FP64, 정확 정수; 블록 스케일은 곱·그룹·승격·epilogue 중 어디서 적용할지 선택 |
| **하드웨어 프리셋** | Hopper FP8 (F=13, CS=32), Ada FP8, Blackwell FP8 (F=25), Blackwell FP4 (GDFS G=6 F=35), DeepSeek FP8 승격 — 모든 프리셋에 출처 기록 |
| **모델과 과제** | Hugging Face causal LM 전반 (`nn.Linear` 레이어, 레이어별 규칙), WikiText-2 perplexity, lm-eval 전 과제, 레이어별 오차 리포트 (MSE, SQNR, 코사인, logits KL) |

## 어떻게 검증하나

에뮬레이션은 그것이 정말 의도한 산술일 때만 쓸모가 있다. 아래 항목은 전부 이 저장소의 테스트다.

| 확인 | 근거 |
|---|---|
| 레퍼런스 캐스트 vs PyTorch 네이티브 변환 | fp8 4종, bf16, fp16 — 형식마다 무작위 값 10만 개 + 경계값에서 비트 일치 |
| 레퍼런스 MX 양자화 vs `microsoft/microxcaling` | 비트 일치 (even / nearest / floor). 단, microxcaling 이 fp32 `log2` 로 잘못 분류하는 입력은 제외 |
| 레퍼런스 MMA vs **NADPE** CUDA 커널 (MICRO'26, 단독 빌드) | **1716 / 1716** 케이스 비트 일치 — FP8 CoFDA / C-decoupled / GDFS, NVFP4, MXFP4 |
| Triton 커널 vs 레퍼런스 | GPU 테스트 (`tests/gpu`: 양자화 + MMA, 무작위·경계·특수값) A100 과 V100 에서 각각 824 개 통과 |
| 실제 크기 GEMM 에서 Triton MMA vs NADPE | 2048×1024×3072, 2048×3072×1024, 4096³ 에서 CoFDA·C-decoupled·GDFS 모두 비트 일치 (`scripts/bench/bench_mma_vs_nadpe.py`) |

Triton 은 libdevice 를 flush-to-zero 로 링크한다. 그래서 서브노멀을 만날 수 있는 모든 fp32 연산은 IEEE PTX 로
계산하고, 레퍼런스는 fp32 연산 하나하나를 fp64 로 계산한 뒤 한 번 반올림한다 — 결과가 장치에 따라 달라지지 않는다.

## 빠른 시작

```bash
pip install -e ".[triton,eval]"          # Triton 커널은 Linux + CUDA; CPU 에서는 레퍼런스로 동작
pip install -e ".[agent]"                # `tricast agent` 를 Claude API 로 쓸 때만
```

```python
import torch, tricast
from tricast.eval.ppl import perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

report = tricast.patch_model(model, "hopper_fp8_w8a8")   # 모든 decoder linear 를 Hopper 누산으로
print(perplexity(model, tok, dataset="wikitext2"))
```

```bash
tricast schemes                                   # 무엇을 어떻게 양자화할 수 있는지
tricast ppl    --model Qwen/Qwen3-0.6B --recipe nvfp4_w_a
tricast report --model Qwen/Qwen3-0.6B --recipe mxfp4_w_a        # 레이어별 MSE / SQNR / 코사인 / KL
python -m tricast.eval.lmeval --model tricast \
    --model_args pretrained=Qwen/Qwen3-0.6B,recipe=hopper_fp8_w8a8 --tasks hellaswag,coqa
python examples/demo_qwen3.py --quick             # 형식, 누산기, 16 창 PPL, 생성 비교
```

### 레시피 예

```yaml
name: my_accelerator
defaults:
  weight:     {scheme: nvfp4}
  activation: {format: fp8_e4m3, granularity: row}
  mma:        {algorithm: cofda, f_bits: 11, chunk_size: 64, c_mode: decoupled}
overrides:
  - layers: "0,-1"                 # 첫·마지막 decoder 블록의 linear 는 양자화하지 않는다
    skip: true
  - modules: [down_proj]
    weight: {scheme: mxfp8_e4m3}
kv: {preset: kivi2, mode: cache}   # KV 캐시는 모든 블록에서 KIVI-2 (kv.layers 로 좁힌다)
```

## 결과 — Qwen3-0.6B

(`examples/demo_qwen3.py` 실행 결과로 채움; `docs/demo/DEMO.md` 참고)

## 한계

- attention 의 `QKᵀ` / `PV` 행렬곱은 아직 에뮬레이트하지 않는다. KV 캐시 양자화는 지원한다.
- 하드웨어 프리셋은 공개된 측정(NADPE)에서 왔다. TriCast 가 실리콘과 직접 대조하는 도구(`tricast.probe`)는 계획 단계다.
- CUDA core 에뮬레이션은 모델링 대상인 텐서코어 GEMM 보다 느리다. A100 에서 Hopper FP8 누산기는 0.15 TMAC/s
  (NADPE CUDA 커널 0.15), Qwen3-0.6B 의 2048 토큰 창 하나에 약 6 초가 걸린다. 새 구성을 처음 쓸 때 Triton 커널을
  컴파일한다.
- 자연어 요청을 레시피로 바꾸는 에이전트(`tricast agent`)는 가짜 모델 client 로 테스트했다. 실제 Claude API 경로에는
  `pip install -e ".[agent]"` 와 `ANTHROPIC_API_KEY` 가 필요하다. 둘 중 하나가 없으면 `--llm auto` 는
  오프라인 파서를 쓰고, 어떤 파서가 돌았는지 출력한다.

## 바탕이 된 작업

NADPE / MMA-Emu (MICRO'26, *Not All Dot Products Are Equal*), microsoft/microxcaling, OCP Microscaling 스펙,
NVIDIA NVFP4, DeepSeek-V3 (FP8 승격), GPTQ, AWQ, SmoothQuant, QuaRot, KIVI, Four Over Six. MIT 라이선스.
