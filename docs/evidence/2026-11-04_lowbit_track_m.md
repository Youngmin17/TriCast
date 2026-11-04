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

(작업 완료 후 기록)

## 2. Qwen3-0.6B 트랙 M 표 (개발 모델, 사전 등록 전)

- 풀: bf16, gptq_int4/3/2, rht_gptq_int4/3/2, awq_int4/3, hqq_int4/3/2, nf4, int8_w (14), `--mma fp32_fma`, 보정 WikiText-2 train 128 × 2048 seed 42, KL 창 validation 8 × 512 seed 42.
- 배낭 2.5 · 3.0 · 3.5 · 4.25 비트, 균일 기준선 13개. 검증 KL (seed 43) 과 test PPL.
- 실행 스크립트 `runs/20261104_lowbit/qz_trackm_qwen.sh` (`MODE=auto|uniform`). A100 에 고정한다 (트랙 M 세션의 GPU 모델).
- 이 결과는 논문 프로토콜 §8 판정 기준을 정하기 전에 본다. 그래서 §8 의 수치 기준을 이 결과에 맞춰 고르지 않는다는 점을 사전 등록 커밋에 적는다.
