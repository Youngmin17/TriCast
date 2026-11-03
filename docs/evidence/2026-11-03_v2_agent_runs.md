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
