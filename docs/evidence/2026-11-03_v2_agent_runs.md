# 측정 기록 — 2026-11-03 (Auto-tune v2 · 공식 코드 게이트 · R7 · 에이전트 루프)

[Auto-tune v2 스파이크](../spikes/2026-11-03_autotune_v2.md), [에이전트 스파이크](../spikes/2026-11-03_tune_agent.md),
[자율 루프 스파이크](../spikes/2026-11-03_tune_loop.md), SPEC AC22~AC27 의 GPU 근거다.
- 모든 GPU 실행은 `gpu_cap.sh` 를 거쳤다.
- 트리는 표에 적은 커밋의 수정 없는 클론이다 (`git_dirty: false`).
- 원자료는 git 에 넣지 않는 `runs/20261103_tune/` 와 클러스터 `/scratch/uceeeee/tricast/` 에 있다.

| 항목 | 값 |
|---|---|
| 소프트웨어 | torch 2.8.0+cu128 · transformers 4.55.2 · datasets 4.8.5 · CUDA 12.8 (conda env `tricast`) |
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca`, PPL 은 WikiText-2 test 146 × 2048 창 (298,862 토큰, 지문 `a46124b21ac53738`) |
| Llama-3.2-1B | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Llama-2-7B (게이트) | `01c7f73d771dfac7d292323805ebc428287df4f9` |
| 측정 기록 ledger | `runs/ledger.jsonl` — 모든 레코드를 원본 해시 · 위치로 대조해 불일치 0 (`python -m app.ledger verify`) |

## Auto-tune v2 — Qwen3-0.6B (트리 `ec3afc2`, NVIDIA A100-SXM4-80GB)

| 산출물 | 경로 · 해시 |
|---|---|
| 표 | `qz_v2_qwen/table.json` (sha256 `91ca4b448afd8113…`), validation 8 × 512 seed 42, 보정 train 128 × 2048 seed 42 |
| 풀이 | `auto_4.50` `202342dc11a4…` · `auto_5.09` `afe7fc040a72…` · `auto_5.72` `50ceb708f313…` · `auto_5.72_kivi2` `340067ba25cb…` |
| C2 | `c2.json`: ρ 0.9860 (n 12), 대역 ρ 1.0000 (n 8), 쌍 3/3, 창 겹침 0 |

| 레시피 | 검증 KL (validation seed 43) | PPL | 시간 (초) |
|---|:-:|:-:|:-:|
| `auto_4.50` | 0.08768 | 21.8822 | 1,097 |
| `auto_5.09` | 0.05967 | 21.9232 | 1,057 |
| `auto_5.72` | 0.04719 | 21.6168 | 1,042 |
| `auto_5.72_kivi2` | 0.16307 | 23.9460 | 1,105 |
| `qd_nvfp4` | 0.33896 | (W2) 26.1599 | — |

## C3′ — Llama-3.2-1B (표 `ec3afc2`, PPL `eff9e7f`, NVIDIA H100 NVL)

| 산출물 | 값 |
|---|---|
| 표 | `qz_v2_llama/table.json` (sha256 `b9a17e6ca2eeb5ba…`): 112층, 방법 848 + 바닥 112 + KV 48, 2,548초, 보정 겹침 0 |
| 4.50 비트 풀이 | GPTQ 74 · INT8 27 · MXFP8 6 · BF16 5 |
| PPL | 원본 9.7573 · `llama_auto_4.50` **10.2269** (Hessian 9.46 GB, 한도 16 GiB) · `qd_nvfp4` 11.6306 |
| 다시 잰 이유 | 첫 실행(`ec3afc2`)은 GPTQ Hessian 이 기본 한도 4 GiB 를 넘어 거부됐다 (기록 `qz_v2_llama.log`). |

## AC25 — 공식 Wanda / SparseGPT 재현 게이트 (Llama-2-7B, 트리 `1e93df6`, NVIDIA H200)

PPL 은 공식 `lib.eval.eval_ppl` 로 잰다 (WikiText-2 test, seqlen 4096). 보정은 공식 C4 128 × 4096, seed 0 이다. 기준은 고정 커밋 README 다.

| 설정 | TriCast 어댑터 | 공식 코드 · 구 환경 (transformers 4.37, shim 없음) | README | 판정 |
|---|:-:|:-:|:-:|:-:|
| dense | 5.1166 | (Micro_27 5.1164) | 5.12 | ✅ |
| Wanda 50% | 6.4631 | 6.4636 | 6.42 | ✅ |
| Wanda 2:4 | 11.3392 | 11.3679 | 11.02 | ❌ README 대비 +2.9% — 공식 코드 자체 재현값과 일치 (Δ −0.25%) |
| SparseGPT 50% | 6.5168 | — | 6.51 | ✅ |
| SparseGPT 2:4 | 10.1807 | — | 10.17 | ✅ |

게이트 실행의 deviation 은 결과 JSON 의 `deviations` · `gate_deviations` 에 남겼다.
- 클러스터의 meta-llama 캐시에는 토크나이저가 없었다. 그래서 같은 SHA-256 (`9e556afd…`) 의 Llama-2 `tokenizer.model` 사본을 썼다.
- 공식 C4 `data_files` 를 풀기 위해 이 게이트만 hub 를 온라인으로 했다. 샤드는 캐시에 있었다.
- datasets 4.x 에 맞춰 C4 설정 이름을 넘기지 않게 했다 (`1e93df6`). `'en'` 으로 넘겼던 첫 시도는 분할 검증 오류로 실패했다.

## AC26 — 공식 GPTQ GPU 게이트 (Llama-2-7B W4 g128, NVIDIA H200)

> 판정 기준은 측정 전에 이 커밋으로 적었고, 결과를 본 뒤 바꾸지 않는다.

**설정**
- 공식 GPTQ `2d65066` 의 `llama_sequential` 을 쓴다. W4, 비대칭, group 128, true-sequential, act-order 없음, damp 0.01.
- 보정은 공식 `datautils` 의 C4 128 × 2048, seed 0 이다.
- 세 실행을 같은 평가(`same_eval`)로 잰다. 평가는 WikiText-2 test 를 `"\n\n"` 로 잇고 2048 토큰 창으로 나눠 fp16 전체 forward 로 잰다 (공식 `llama_eval` 과 같은 식).
  - `dense`: 양자화하지 않은 모델 (tricast env, transformers 4.55)
  - `adapter`: `tricast.official.gptq.quantize` (tricast env + compat shim)
  - `official`: 같은 공식 코드를 구 환경(OmniServe env, transformers 4.37, shim 없음)에서 직접 부른다. 공식 `llama_eval` 도 함께 잰다.
- 앵커는 AWQ (MLSys 2024) Table 4 의 Llama-2-7B INT4-g128 이다: FP16 5.47, GPTQ 5.69 (재정렬 없음), GPTQ-R 5.63. 공식 README 에는 Llama-2 수치가 없어서 제3자 값을 쓴다.
- glue 는 두 가지이고, AC25 와 같다.
  - C4 설정 이름을 넘기지 않는다 (datasets 4.x).
  - 토크나이저 없는 meta-llama 스냅샷의 config · 가중치를 Llama-2 slow tokenizer 사본(`9e556afd…`) 옆에 링크한다.

**판정 기준**

| # | 조건 | 기준 |
|:-:|---|---|
| G1 | 어댑터 = 공식 코드 (같은 평가) | \|PPL<sub>adapter</sub> − PPL<sub>official</sub>\| ≤ 0.05 또는 상대 1% 이내 |
| G2 | 공식 코드의 재현 | 공식 `llama_eval` PPL 이 앵커 5.69 의 1% 이내 (5.633–5.747) |
| G3 | 기준선 | dense PPL 이 앵커 5.47 의 0.05 이내 |
| G4 | 발화 증거 | `fasterquant` 224회 (32 블록 × 7 Linear), 그룹당 고유값 ≤ 16 |

- AC26 GPU 게이트의 판정은 G1 · G4 로 한다.
- G2 · G3 는 재현 수준을 보고한다. 실패하면 "재현 미달 (Δ=…)" 로 적고 원인을 가린다.

**결과** (트리 `d1daeab`, NVIDIA H200 NVL, berlin)

| 실행 | 환경 | same_eval PPL | 공식 `llama_eval` | 결과 파일 sha256 |
|---|---|:-:|:-:|---|
| dense | tricast env (torch 2.8.0 · transformers 4.55.2) | 5.4721 | — | `bbb5e511…` |
| adapter | tricast env + compat shim | **5.6570** | — | `9e012c68…` |
| official | OmniServe env (torch 2.2.0 · transformers 4.37.2, shim 없음) | **5.6712** | 5.6714 | `1646b84a…` |

- 같은 평가: 두 환경의 평가 토큰 sha256 이 같다 (`8e0218ee…`, 166 × 2048 창).
- 같은 보정 입력: 어댑터가 공식 loader 에서 받은 C4 128 × 2048 과 구 환경 `get_loaders` 출력의 sha256 이 같다 (`068c5674…`).

| # | 결과 | 판정 |
|:-:|---|:-:|
| G1 | \|5.6570 − 5.6712\| = 0.0142 (상대 0.25%) ≤ 0.05 | ✅ |
| G2 | 공식 `llama_eval` 5.6714 vs 앵커 5.69: Δ −0.019 (−0.33%), 1% 이내 | ✅ |
| G3 | dense 5.4721 vs 앵커 5.47: Δ +0.002 | ✅ |
| G4 | `fasterquant` 224회, 공식 층 호출 20,480회 (모두 `position_ids` 포함), 그룹당 고유값 최대 16 | ✅ |

- **판정: AC26 GPU 게이트 통과 (G1 · G4).** 공식 코드 자체도 제3자 앵커를 재현한다 (G2 · G3).
- 어댑터가 0.014 낮다. 두 실행은 torch · transformers 판이 다르다(GPTQ 풀이와 forward 수치 경로 포함). 원인은 따로 가르지 않았다.
- 1차 실행은 dense 단계에서 NFS 페이지 대기(D 상태, pyarrow `.so`)로 20분 멈췄다. 그래서 정리하고 2차 실행에서 네이티브 라이브러리를 미리 읽었다. 1차 로그는 클러스터 `qz_gptq_gate/try1/` 에 있다.

## 보정 결과 캐시 GPU 게이트 (Qwen3-0.6B, 자율 루프 2라운드의 제안 4개)

> 판정 기준은 측정 전에 이 커밋으로 적었고, 결과를 본 뒤 바꾸지 않는다.

**설정**
- 레시피는 `runs/20261103_tune/loop_s2` 2라운드의 제안 4개다: `ao_damp0.005_int8mse` · `noao_int8mse` · `ao_int8mse_swap_l2up_l21down` · `ao_int8mse_actmse`.
- 측정은 루프와 같은 `app.autotune measure --calibrate` 이다. 검증 8 × 512, seed 43, 보정 wikitext2 train 128 × 2048, seed 42.
- GPU: 루프와 같은 geneva A100 이 cap(2) 에 차 있어서(실행 전 확인) berlin H200 에서 돌린다. 이 수정은 결과를 보기 전에 했다.
  - CG1–CG4 는 같은 작업 안의 상대 비교라 GPU 와 무관하다. 아래 "참고" 비교만 GPU 가 달라 해석이 약해진다.
- 한 작업 안에서 세 번 잰다.
  - S1: 캐시 없음
  - S2: `--calib-cache` 새 디렉터리 (채우기; 제안끼리 같은 층 명세는 서로 적중)
  - S3: 같은 캐시로 다시 (전부 적중)
- 같은 작업에서 NF4 의 bitsandbytes CUDA 비트 일치 테스트(`tests/test_weight_methods.py`, B1 이 GPU 를 못 돌린 1건)도 돌린다.

**판정 기준**

| # | 조건 | 기준 |
|:-:|---|---|
| CG1 | 결과가 같다 | 제안마다 S2 · S3 의 KL 이 S1 과 float 로 같다 |
| CG2 | 전부 적중이면 빠르다 | S3 에서 보정 forward 0회, 제안당 `wall_s` 합이 S1 의 1/5 이하 |
| CG3 | 부분 적중의 절감 (보고만) | S2 의 적중 · 미스 수와 `wall_s` |
| CG4 | NF4 CUDA | `test_nf4_matches_bitsandbytes_cuda_bit_for_bit` 통과 |

- 채택은 CG1 · CG2 · CG4 로 한다. CG1 이 실패하면 캐시는 쓰지 않는다.
- 참고(판정 무관): S1 의 KL 을 루프가 트리 `1e93df6` 에서 잰 ledger 값과 비교한다 (그 뒤 코드 변경의 영향).

**결과** (트리 `967e3ea` — 캐시 코드는 리뷰 수정 전인 `7ed0f5d` 판, NVIDIA H200 NVL, berlin, 2026-11-03 14:03–14:26 UTC)

| 제안 | S1 KL (캐시 없음) | S2 · S3 KL | S1 `wall_s` | S2 `wall_s` (적중 / 미스) | S3 `wall_s` (적중 / 미스, forward) |
|---|:-:|:-:|:-:|:-:|:-:|
| `ao_damp0.005_int8mse` | 0.05841608473416018 | 같음 · 같음 | 100.4 | 68.8 (0 / 194) | 112.7 (194 / 0, 0) |
| `noao_int8mse` | 0.0614358996930136 | 같음 · 같음 | 60.5 | 101.6 (65 / 129) | 128.6 (194 / 0, 0) |
| `ao_int8mse_swap_l2up_l21down` | 0.05629070946923123 | 같음 · 같음 | 59.9 | 62.0 (63 / 131) | 16.7 (194 / 0, 0) |
| `ao_int8mse_actmse` | 0.056175341815044286 | 같음 · 같음 | 61.6 | 118.9 (166 / 28) | 36.9 (194 / 0, 0) |
| 합 | | | 282.4 | 351.3 | 294.9 |

| # | 결과 | 판정 |
|:-:|---|:-:|
| CG1 | 네 제안 모두 S2 · S3 의 KL 이 S1 과 float 로 같다 | ✅ |
| CG2 | S3 는 전 층 적중 · 보정 forward 0회지만 `wall_s` 합이 S1 의 1.04배 (기준: 1/5 이하) | ❌ |
| CG3 | 미스가 하나라도 있으면 forward 129회가 그대로 돈다. 미스 28개(`actmse`)도 118.9 s | (보고) |
| CG4 | `tests/test_weight_methods.py` 19 passed, skip 0 (NF4 bitsandbytes CUDA 비트 일치 포함) | ✅ |

- **판정: 캐시는 결과가 같지만(CG1) 빠르지 않다(CG2 실패).** 측정 가속 목적으로는 채택하지 않는다.
- 원인 가설 (미검증): 적중 항목(레시피당 약 0.9 GB 의 BF16 operand · scale)을 NFS `/scratch` 에서 읽고 쓰는 시간이다.
  - 근거: 같은 S3 안에서 앞 두 제안(S2 가 막 쓴 항목)은 113–129 s, 뒤 두 제안(앞과 같은 항목이 많음)은 17–37 s 였다.
  - 다음: 노드 로컬 디스크 캐시와 리뷰 수정판 코드(`44228cf`)로 CG2 를 다시 사전 등록해 잰다.
- **참고 (판정 무관) — 루프 ledger 값과 다르다.**
  - 같은 레시피를 자율 루프가 트리 `1e93df6` · A100 에서 잰 값은 0.05538 · 0.05912 · 0.05574 · 0.05608 이다 (위 순서).
  - 이번 값과 최대 +5.5% 차이가 나고, 순위도 바뀐다 (A100 최선 `ao_damp0.005_int8mse` 가 여기서는 4개 중 3위).
  - **귀속: GPU 차이다. 코드 회귀가 아니다.** 트리 `1e93df6` 을 같은 H200 에서 재면 네 값이 트리 `967e3ea` 와 float 로 같다 (`attr_1e93df6_h200.json`, 14:28–14:32 UTC). 레시피 해시도 같다.
  - 그러니 같은 GPU · 같은 트리 안에서는 측정이 결정적이다 (CG1, 이번 귀속, R7). A100 과 H200 사이에서는 KL 이 최대 5.5% 다르고, 서로 가까운 제안의 순위가 바뀐다.
  - 논문 실험에 주는 규칙: 한 칸(모델 × 예산)의 모든 arm 은 같은 GPU 모델과 같은 트리에서 잰다. GPU 간 순위 상관을 따로 보고한다 (protocol §4 · §7).

## 보정 결과 캐시 GPU 게이트 2 — 노드 로컬 디스크 (Qwen3-0.6B, H200)

> 판정 기준은 측정 전에 이 커밋으로 적었고, 결과를 본 뒤 바꾸지 않는다.

- 동기: 1차 게이트는 CG1 통과, CG2 실패였다. 원인 가설은 NFS 입출력이다.
- 바꾼 것은 두 가지다.
  - 캐시 디렉터리를 berlin 의 노드 로컬 NVMe (`/tmp/uceeeee_calib_cache`) 에 둔다. 끝나면 지운다.
  - 코드는 리뷰 수정판이다 (`44228cf` 포함 트리, `CACHE_FORMAT` 2).
- 나머지(제안 4개, 측정 명령, S1 · S2 · S3 순서)는 1차와 같다.

| # | 조건 | 기준 |
|:-:|---|---|
| CG1′ | 결과가 같다 | 제안마다 S2 · S3 의 KL 이 S1 과 float 로 같고, 1차 게이트의 S1 값과도 같다 |
| CG2′ | 전부 적중이면 빠르다 | S3 에서 보정 forward 0회, 제안당 `wall_s` 합이 S1 의 1/5 이하 |
| CG5 | 리뷰 수정이 실제 모델에서 캐시를 끄지 않는다 | S3 의 `uncacheable` · `corrupt` 가 비어 있고 `inputs` 가 verified |

- 채택은 CG1′ · CG2′ · CG5 로 한다. CG2′ 가 실패하면 측정 가속용 캐시는 쓰지 않고, 원인 분석(프로파일)으로 넘어간다.

## R7 — 서버 GPU 작업 경로 (트리 `eff9e7f`, Tesla V100-PCIE-32GB, london)

`POST /api/quantizer/jobs` (FastAPI TestClient, 포트를 열지 않는 배치) 로 `qd_nvfp4` PPL 을 재고, 같은 GPU 에서 `tricast run` 과 짝지어 비교했다.

| 경로 | PPL | 레시피 해시 |
|---|:-:|---|
| 서버 작업 | 26.1968475695289 | `9f2612422bcc…` |
| CLI `tricast run` | 26.1968475695289 | `9f2612422bcc…` |

- 판정: **같음 (비트 단위)**. 스크립트가 찍은 자동 판정은 `FAIL` 이었다. 원인은 판정 스크립트가 작업 결과의 `ppl` 을 dict 로 가정한 버그다. 같은 기록 파일로 다시 판정했다.
- 함께 찾은 결함: 서버 작업 기록에 모델 revision 이 비어 있었다 (`model_sha: null`). `74fb16f` 에서 고쳤다.
- 이전 시도:
  - 1차: 클러스터 env 에 fastapi 가 없었다.
  - 2차: turin 에서 pyarrow dlopen 이 autofs 에서 멈췄다.

## 에이전트 스파이크 (지휘 세션, 4.50 비트, 트리 `1e93df6`, A100-SXM4)

세션 `agent_s1` 은 제안 33개로 구성된다. A0 하나, A1 16개, A2 16개다. 측정 결과는 `rounds/1~4/kl.json` 에 있다.

| arm | 최종 선택 | 레시피 해시 | 검증 KL | test PPL |
|---|---|---|:-:|:-:|
| A0 | `auto_4.50` | `202342dc11a4…` | 0.08768 | 21.8822 |
| A1 | `a1_r4_v4int8` | `bbae404aea12…` | 0.07787 | **21.6348** |
| A2 | `a2_r4_n2` | `39b6877b8493…` | 0.08371 | 21.8588 |

## 자율 루프 (`app/tune_loop.py` @ `3ecf566`, 285/56 비트)

세션 `loop_s2` 는 2라운드로 끝났다. 제안 8개 중 6개가 통과했다. 제안 호출 비용은 $0.71 · $0.64 다 (claude-opus-5-5).

| 레시피 | 검증 KL | test PPL |
|---|:-:|:-:|
| 시작점 `auto_5.09` | 0.05967 | 21.9232 |
| 루프 최선 `ao_damp0.005_int8mse` | 0.05538 | **21.7820** |

## C3 사후 대조군 (사전 등록 아님, 트리 `1e93df6`, A100-SXM4)

| 레시피 | 레시피 해시 | 비트 | test PPL |
|---|---|:-:|:-:|
| `auto_4.50_nogptq` (GPTQ 를 뺀 파생 표로 풀이) | `a0edd441b7cf…` | 4.4996 | 25.4941 |
| `qd_gptq_int4` (균일 W4A16) | `f8d801d2622d…` | 4.156 | 24.7281 |
