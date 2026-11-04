# 2026-11-04 — 트랙 M 저비트 방법: Llama-2-7B 균일 충실도 · Qwen3-0.6B 표

> 이 문서의 판정 기준(§1.2)은 결과가 나오기 전에 커밋했다. 작업은 23:43 UTC 에 시작했고 결과 파일은 아직 없다.

## 1. Llama-2-7B 균일 충실도 (A100-SXM4-80GB, 트리 `2cdbd39`)

### 1.1 설정

- 모델: `meta-llama/Llama-2-7b-hf` snapshot `01c7f73d`, 토크나이저 `TheBloke/Llama-2-7B-AWQ` `476ced46` (AC26 게이트와 같은 glue 폴더), bf16.
- PPL: WikiText-2 test, `"\n\n"` 이어붙이기, 2048 창 166개 (`tricast run`, `tasks.ppl`).
- GEMM: 모든 레시피 `defaults.mma: {algorithm: fp32_fma}`.
  - 프리셋 `fp32_fma` 와 산술이 같다 (`MMASpec` 의 이름 · 출처 필드만 다름).
- 레시피: `runs/20261104_lowbit/recipes/l2_*.yaml` (git 밖, 손으로 씀; 등록 전 방법도 같은 필드).
  - GPTQ 계열: C4 train 128 × 2048, seed 0, 순차 보정 — 공식 `llama.py`.
  - AWQ 계열: Pile 128 × 512, seed 0, 비순차. 공식 `entry.py` 는 `mit-han-lab/pile-val-backup` validation 이고, TriCast `pile` 은 `NeelNanda/pile-10k` train 이다 (파라미터 매핑).
- 작업: `j1` gptq_int4 · rht_gptq_int4, `j2` int3 두 개, `j3` int2 두 개, `j4` awq_int4 · awq_int3 · int8_w. 로그 `/scratch/uceeeee/tricast/logs/qz_lb_j*.log`, 결과 `/scratch/uceeeee/tricast/qz_lowbit/out/j*/`.
- 1차 시도(트리 `7760166`, 23:03–23:11 UTC)는 C4 보정 로더 버그로 GPTQ 계열 4개가 실패했다. 로그는 `logs/try1/` 에 있다.
  - 원인: datasets 4.x 에서 설정 `en` 의 split 검증 (`ExpectedMoreSplitsError`). `2cdbd39` 에서 고쳤다.
  - 같은 시도의 dense bf16 기준선은 끝났다: **PPL 5.4737** (FP16 앵커 5.47).

### 1.2 판정 기준 (결과 전 커밋)

앵커 (Llama-2-7B, WikiText-2, ctx 2048, g128):

| 설정 | AWQ v5 Tab 4 | OmniQuant v3 Tab 1 |
|---|---|---|
| FP16 | 5.47 | 5.47 |
| INT4 RTN / GPTQ / AWQ | 5.73 / 5.69 / 5.60 | 5.72 / 5.61 / 5.62 |
| INT3 RTN / GPTQ / AWQ | 6.66 / 6.43 / 6.24 | 6.66 / 6.29 / 6.24 |
| INT2 RTN / GPTQ / AWQ | — | 4.2e3 / 36.77 / 2.2e5 |

- 두 논문의 GPTQ 값이 다르다 (INT4 0.08, INT3 0.14). 설정 차이로 보이고, 어느 쪽이 공식 기본값인지 이 문서에서는 확인하지 않았다.

| # | 방법 | 통과 조건 |
|:-:|---|---|
| F1 | gptq_int4 | PPL ∈ [5.61 × 0.99, 5.69 × 1.01] = [5.554, 5.747] |
| F2 | gptq_int3 | PPL ∈ [6.29 × 0.99, 6.43 × 1.01] = [6.227, 6.494] |
| F3 | awq_int4 | \|PPL − 5.60\| ≤ 0.056 또는 \|PPL − 5.62\| ≤ 0.056 (상대 1%) |
| F4 | awq_int3 | \|PPL − 6.24\| ≤ 0.0624 |
| F5 | int8_w | PPL ≤ 5.4737 × 1.005 (dense 대비 +0.5% 이내) |
| R1 | rht_gptq_int3 | PPL < gptq_int3 의 PPL (같은 실행 조건) |
| R2 | rht_gptq_int2 | PPL < gptq_int2 의 PPL |
| — | gptq_int2 · rht_gptq_int4 | 보고만 (2비트 GPTQ 는 논문마다 36.77–60.45 로 흔들린다; SliM-LLM Tab 1 은 60.45) |

- F1–F5 가 실패한 방법은 트랙 M 풀에 넣되, 비교표에 "재현 미달 (Δ=…)" 로 적는다. 원인 분석은 버전 · 보정 데이터 · 그룹 방향 순으로 한다.
- R1 · R2 가 실패하면 rht_gptq 는 풀에 남기되, "저비트에서 이득 없음" 을 기록한다. 이 판정은 RHT 블록 128(입력 쪽만)에 대한 것이다.

### 1.3 결과

결과 파일은 `qz_lowbit/out/j*/l2_*.json` 이다. 레시피 해시는 로컬 `runs/20261104_lowbit/recipes/` 와 같다. 모든 행이 166 창이다.

| # | 방법 | GPU | 레시피 해시 | PPL | 기준 | 판정 |
|:-:|---|---|---|---|---|:-:|
| — | dense bf16 (1차 시도, `7760166`) | A100-SXM4 | native | 5.4737 | FP16 5.47 | 정상 |
| F1 | gptq_int4 | A100-SXM4 | `5061e109da` | 5.6563 | [5.554, 5.747] | ✅ |
| F2 | gptq_int3 | A100-SXM4 | `d12ebde42a` | 6.4812 | [6.227, 6.494] | ✅ |
| F3 | awq_int4 | A100 PCIe | `05c83e3604` | 5.6174 | 5.60 ± 0.056 | ✅ |
| F4 | awq_int3 | A100 PCIe | `954ccc131c` | 6.4864 | 6.24 ± 0.0624 | ❌ 재현 미달 (Δ = +0.246, +3.9%) |
| F5 | int8_w | A100 PCIe | `de5bf9a5a4` | 5.4769 | ≤ 5.5011 | ✅ |
| — | gptq_int2 | A100 PCIe | `c10002c389` | 76.97 | 보고만 | — |
| — | rht_gptq_int4 | A100-SXM4 | `3ac9664d34` | 5.6109 | 보고만 (gptq_int4 5.6563 대비 −0.8%) | — |
| R1 | rht_gptq_int3 | A100-SXM4 | `3752658bb5` | 6.3110 | < gptq_int3 6.4812 (같은 작업 j2) | ✅ (−2.6%) |
| R2 | rht_gptq_int2 | A100 PCIe | `215df05099` | 60.17 | < gptq_int2 76.97 (같은 작업 j3) | ✅ (−21.8%) |

- 입력 쪽 RHT(블록 128, seed 0)는 GPTQ 를 모든 비트에서 낫게 한다. 다만 2비트는 여전히 쓸 수 없는 수준이다 (OmniQuant W2 g128 11.06). 2비트 경쟁력에는 벡터 양자화(QuIP# E8P · AQLM)나 학습형 클리핑이 필요하다.
- rht_gptq_int3 6.311 은 AWQ 앵커 6.24 보다 1.1% 높고, 우리 awq_int3(스케일만) 6.486 보다 낮다.

- gptq_int4 5.6563 은 AC26 의 공식 GPTQ 어댑터 결과 5.6570 과 0.0007 차이다 (다른 GPU · 다른 GEMM 누산).
- 두 A100 은 장치 이름이 다르다 (geneva `A100-SXM4-80GB`, turin `A100 80GB PCIe`). 세션 GPU 고정은 이름으로 비교하므로, 트랙 M 표 · 세션은 한 변형에 고정한다 (§1.5 이후 geneva).
- **F4 원인 분석** (공식 `mit-han-lab/llm-awq` `d6e797a4` 의 `awq/quantize/auto_scale.py` · `auto_clip.py` 와 `src/tricast/transforms.py` `_fit_awq` 를 대조):
  1. **TriCast 에는 `auto_clip` 이 없다.** 공식은 스케일 적용 뒤 q/k 를 뺀 모든 Linear 에서, 출력 행 × 그룹마다 최대값을 1 − i/20 (i < 10) 로 줄여 보정 토큰 512개의 출력 MSE 가 최소인 값으로 가중치를 클리핑한다. 클리핑은 비트가 낮을수록 중요하다. INT4 통과 · INT3 미달과 맞는 유력 원인이다 (아직 검증 전).
  2. α 격자: 공식 {0, 0.05, …, 0.95}, TriCast linspace(0, 1, 20).
  3. 스케일 하한: 공식 clamp(min=1e-4), TriCast 는 0 만 1e-5 로 바꾼다.
  4. 목적 함수: 공식은 모듈 출력 (q/k/v → self_attn 전체, gate/up → mlp 전체), TriCast 는 Linear 출력 오차의 합.
  5. 표본: 공식은 보정 토큰 전부 (스케일) · stride 512개 (클리핑), TriCast 는 4096행 reservoir.
- 처리: 사전 등록 규칙대로 awq_int4 · awq_int3 은 풀에 두고 "AWQ 스케일 탐색만 (auto_clip 없음)" 으로 표기한다. 클리핑 흡수는 `weight_algo` 의 새 kind 로 따로 추가한다. 다른 AWQ 칸으로 값이 새지 않고, 이미 만든 표의 해시도 바뀌지 않는다. 흡수 후 같은 F4 로 다시 판정한다.

### 1.4 AWQ 클리핑 흡수 후 재판정 (결과 전 커밋)

- 방법: `awq_clip_int3` · `awq_clip_int4` (AWQ 스케일 + 공식 auto_clip, `weight_algo: awq_clip`). 설정은 §1.1 의 AWQ 계열과 같다 (Pile 128 × 512, seed 0, fp32_fma).
- 기준 (§1.2 와 같은 수치):
  - **F4′** awq_clip_int3: \|PPL − 6.24\| ≤ 0.0624
  - **F3′** awq_clip_int4: \|PPL − 5.60\| ≤ 0.056 또는 \|PPL − 5.62\| ≤ 0.056
- F4′ 가 실패하면 남은 차이(α 격자 · 스케일 하한 · 모듈 출력 목적 · 표본)를 하나씩 격리하는 실험을 계획에 올린다. 실패한 채로도 풀에는 둔다.

결과 (트리 `9ecd7cb`, A100-SXM4, 01:43–02:12 UTC):

| # | 방법 | 레시피 해시 | PPL | 기준 | 판정 |
|:-:|---|---|---|---|:-:|
| F3′ | awq_clip_int4 | `fb9826a5c6` | 5.6048 | 5.60 ± 0.056 | ✅ |
| F4′ | awq_clip_int3 | `29a4a23ac1` | 6.3074 | 6.24 ± 0.0624 | ❌ (Δ = +0.067, +1.08%) |

- 클리핑이 F4 의 차이를 73% 줄였다 (6.4864 → 6.3074, 앵커 6.24). 남은 0.067 이 기준(1%)을 조금 넘는다. 비교표에는 "재현 미달 (Δ=+0.067)" 로 적는다.
- 남은 차이의 격리 실험(축 하나씩, G6)은 다음 계획에 올린다. 후보는 목적 함수(모듈 출력), Pile 데이터(pile-val-backup vs pile-10k), fp16 모델 dtype, α 격자 · 스케일 하한이다.
- 3비트에서는 rht_gptq_int3 6.3110 과 awq_clip_int3 6.3074 가 거의 같다 (차이 0.004, 다른 GPU 작업).

### 1.5 Qwen3-0.6B 작업 1차 실패 (turin, 01:55 UTC)

- 두 작업(표 · 균일 PPL)이 같은 시각에 호스트 메모리 할당 실패로 죽었다: `memory allocation of 335544320 bytes failed`(표, SIGABRT 134), `MemoryError`(균일 PPL).
- turin 은 엄격 overcommit(`vm.overcommit_memory=2`, ratio 95, CommitLimit 407 GB)이다. 다른 사용자의 큰 프로세스(가상 크기 140 GB · 118 GB)가 그 시각 무렵 시작됐다. 정확한 순간의 Committed_AS 는 기록하지 못했다.
- 처리: 트랙 M 을 **geneva(A100-SXM4-80GB, CommitLimit 517 GB)** 에 고정하고 두 작업을 처음부터 다시 띄웠다 (02:14 UTC). 표는 재개 식별에 GPU 이름이 들어가서 turin 항목(6층분)을 이어 쓸 수 없다. 1차 출력은 `qz_trackm_qwen_try1_turin/`, 로그는 `logs/try1/` 에 있다.
- 균일 기준선 목록에 awq_clip_int4 · awq_clip_int3 을 더했다 (15개).

## 2. Qwen3-0.6B 트랙 M 표 (개발 모델, 사전 등록 전)

- 풀: bf16, gptq_int4/3/2, rht_gptq_int4/3/2, awq_int4/3, awq_clip_int4/3, hqq_int4/3/2, nf4, int8_w (16), `--mma fp32_fma`, 보정 WikiText-2 train 128 × 2048 seed 42, KL 창 validation 8 × 512 seed 42.
- 배낭 2.5 · 3.0 · 3.5 · 4.25 비트, 균일 기준선 15개. 검증 KL (seed 43) 과 test PPL.
- 실행 스크립트 `runs/20261104_lowbit/qz_trackm_qwen.sh` (`MODE=auto|uniform`). geneva A100-SXM4 에 고정한다 (트랙 M 세션의 GPU 모델, §1.5).
- 이 결과는 논문 프로토콜 §8 판정 기준을 정하기 전에 본다. 그래서 §8 의 수치 기준을 이 결과에 맞춰 고르지 않는다는 점을 사전 등록 커밋에 적는다.

### 2.1 균일 기준선 결과 (geneva A100-SXM4, 트리 `9ecd7cb`, 02:14–03:18 UTC)

- 출력 `/scratch/uceeeee/tricast/qz_trackm_qwen/`: `kl_seed43_uniform.json`, `ppl_uniform/`
- 검증 KL: WikiText-2 validation 8 × 512 창, seed 43.
- test PPL: 2048 창, `tricast run`.
- 모든 레시피는 `defaults.mma: fp32_fma` 이고, 보정은 WikiText-2 train 128 × 2048, seed 42 다.
- 비트는 `WEIGHT_BITS` 로 센 값이다. int8_w 는 행마다 fp32 스케일 하나라 종류에 따라 조금씩 다르다.

| 방법 | 레시피 해시 | 비트 | test PPL | 검증 KL |
|---|---|---:|---:|---:|
| BF16 (native, 패치 없음) | — | 16 | 20.966 | — |
| int8_w | `7c45b40934` | ≈8.02 | 21.057 | 0.00630 |
| rht_gptq_int4 | `de0acd0d73` | 4.16 | 22.558 | 0.1131 |
| awq_clip_int4 | `21c7dcc672` | 4.16 | 23.546 | 0.1558 |
| nf4 | `baf57997ca` | 4.50 | 23.718 | 0.2048 |
| awq_int4 | `d9cb50cbab` | 4.16 | 24.013 | 0.1840 |
| hqq_int4 | `7c8d9e89cf` | 4.50 | 24.550 | 0.2402 |
| gptq_int4 | `177c15a9b6` | 4.16 | 24.725 | 0.2167 |
| awq_clip_int3 | `15cd11ec74` | 3.15 | 33.31 | 0.6148 |
| rht_gptq_int3 | `63577322c3` | 3.15 | 35.81 | 0.6467 |
| awq_int3 | `5771ca9e13` | 3.15 | 44.36 | 0.9482 |
| gptq_int3 | `1822ce0b35` | 3.15 | 48.14 | 1.0486 |
| hqq_int3 | `8cfd19c703` | 3.70 | 123.57 | 1.9411 |
| rht_gptq_int2 | `065f7e9289` | 2.14 | 1.73e6 | 11.24 |
| gptq_int2 | `d5cba7fda6` | 2.14 | 8.41e6 | 12.21 |
| hqq_int2 | `0ac165eb34` | 2.50 | 8.35e6 | 12.77 |

- 0.6B 에서는 방법 차이가 크다. 4비트에서 RHT+GPTQ 가 GPTQ 보다 8.8% 낮고 (22.56 vs 24.72), 3비트에서 AWQ+clip 이 GPTQ 보다 31% 낮다 (33.3 vs 48.1).
- 균일 3비트 이하는 무너진다. 3비트는 BF16 대비 +59% 이상이고, 2비트는 모두 붕괴했다. 그래서 2.5 · 3.0 비트 예산의 결과는 층별 혼합에 달려 있다 (H4 가 겨냥하는 영역).
- 검증 KL 순위와 test PPL 순위는 두 쌍(awq_int4 · nf4, gptq_int4 · hqq_int4)만 뒤바뀌고 나머지는 같다. 두 쌍 모두 두 지표의 차이가 약 2% 안이다.
