# Quantizer 측정 기록 — 2026-10-13

[문제 정의서 v2.0](../PROBLEM.md) §3-5, [스펙 v2.0](../SPEC.md) §5 · §7 · §10, [스파이크](../spikes/2026-10-13_layer_sqnr_proxy.md)와
Quantizer 화면의 기록된 결과(`app/web/demo/quantizer/presets.json`, `app/web/demo/layers/`)가 나온 실행이다. 모든 GPU 실행은 TriCast
`ab489f1` 의 수정 없는 클론(`git_dirty: false`)에서 돌았고, 원본 JSON 은 git 에 넣지 않는 `runs/20261013_quantizer_w{1,2,3}/` 와
클러스터 `/scratch/uceeeee/tricast/qz_w1` 에 있다.

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` |
| 코드 | TriCast `ab489f17cdc01cbaa8074613ca30c89204ca6a9e` (클러스터 트리 `/scratch/uceeeee/tricast/qz_ab489f1`) |
| 소프트웨어 | torch 2.8.0+cu128 · transformers 4.55.2 · lm_eval 0.4.13 · CUDA 12.8 (conda env `tricast`) |
| GPU 배정 | 모든 실행이 `gpu_cap.sh` 경유 |
| PPL 데이터 | WikiText-2 test 전체, 146 × 2048 창, 298,862 토큰, 지문 `a46124b21ac53738` |

## W1 — 레이어 리포트 8종 · PPL 기준점 · 스파이크

**레이어 리포트** (`tricast report`, WikiText-2 train 8 × 512 토큰, seed 42, Tesla V100-PCIE-16GB, 번들 레시피 그대로 — 누산은 각
레시피의 프리셋)

| 레시피 | 레시피 해시 | 모델 logits KL | top-1 일치 | 시간 (초) |
|---|---|:-:|:-:|:-:|
| `int8_row_w8a8` | `78f8822fe042…` | 0.0339 | 0.8967 | 198.4 |
| `mxfp8_w_a` | `a5103a05d087…` | 0.0323 | 0.9053 | 268.4 |
| `w4a16_g128_zp_gptq` | `637ecdc4d0e9…` | 0.1973 | 0.7749 | 509.9 |
| `nvfp4_awq_shared` | `28d7a5dd177a…` | 0.2147 | 0.7605 | 361.5 |
| `nvfp4_smoothquant` | `a9617c515020…` | 0.2437 | 0.7468 | 307.9 |
| `nvfp4_w_a` | `7c25643a2d88…` | 0.2964 | 0.7244 | 297.7 |
| `mxfp4_w_a` | `a40da150ce58…` | 0.5937 | 0.6084 | 296.7 |
| `msfp12_bfp` | `5e34d0a5478f…` | 0.6953 | 0.5952 | 198.7 |

**PPL** (geneva, NVIDIA A100-SXM4-80GB)

| 실행 | 레시피 해시 | PPL | 시간 (초) |
|---|---|:-:|:-:|
| 원본 (`native`) | — | 20.9662 (`support_matrix.yaml` 의 과거 기록과 같음) | 17.1 |
| `mxfp4_w_a` | `a40da150ce58…` | 33.9409 | 1,290.1 |
| `mixed_first_last_bf16` | `5d94e42d2d71…` | 29.6883 | 1,170.5 |
| 스파이크 `top14` | `940cb638f1ec…` | 33.1232 | 1,209.8 |
| 스파이크 무작위 seed 42 · 1 · 2 · 3 · 4 | `3816dc7b89f0…` · `ad8572b5e8af…` · `2c74915e3fa2…` · `2f13b38cdaa2…` · `e0cda0175e99…` | 32.1474 · 33.3784 · 32.6875 · 32.8613 · 31.8504 | 1,158 ~ 1,194 |
| 레이어 혼합 시연 (`mixed_per_layer_demo`, 생성기 이전 손작성본) | `0484257e1149…` | 25.9249 | 1,382.1 |

## W2 — Quantizer 프리셋 5종 PPL · 리포트

프리셋은 `python -m app.quantizer_recipe --write-presets` 의 출력 그대로다 (`app/tests/data/quantizer/recipes/` 와 바이트 동일).
블록 형식은 모두 `mma_input: dequant`.

| 프리셋 | 레시피 해시 | PPL (A100, geneva) | 시간 (초) | 리포트 logits KL · top-1 (H100 NVL, malmo) |
|---|---|:-:|:-:|:-:|
| `qd_mixed` | `11b6ac487694…` | 25.9797 | 1,069.9 | 0.2406 · 0.7476 |
| `qd_nvfp4` | `9f2612422bcc…` | 26.1599 | 989.6 | 0.2996 · 0.7188 |
| `qd_nvint4` | `ba9c8928b96f…` | 27.9498 | 988.8 | 0.4146 · 0.6707 |
| `qd_mxfp4_ends_bf16` | `483ec568f4a0…` | 29.6883 | 913.9 | 0.4762 · 0.6548 |
| `qd_mxfp4` | `e6f0c97f769b…` | 33.9409 | 983.2 | 0.5864 · 0.6194 |

- `qd_mxfp4` · `qd_mxfp4_ends_bf16` 의 PPL 은 W1 의 `mxfp4_w_a` · `mixed_first_last_bf16` 과 소수 넷째 자리까지 같다 — 레시피 표현
  (dequant ↔ 번들 프리셋 누산)이 달라도 같은 값.
- `qd_mixed` (25.9797) 와 W1 의 손작성 `mixed_per_layer_demo` (25.9249) 는 다른 레시피다 (해시 다름 — 손작성본은 누산 프리셋
  `nvidia_blackwell_fp4` · `nvidia_blackwell_fp8` 과 보정 seed 기본값 0 을 쓰고, 생성기는 `mma_input: dequant` 와 보정 seed 42 를 쓴다).

## W3 — lm-eval 5과제 (과제마다 앞 300문항)

malmo, NVIDIA H100 NVL. `acc_norm`: ARC-e · ARC-c · PIQA · HellaSwag, `acc`: WinoGrande. 표본 수는 lm-eval 의
`n-samples.effective` (모두 300).

| 레시피 | ARC-e | ARC-c | PIQA | WinoGrande | HellaSwag | 평균 | 시간 (초) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 원본 | 56.0 | 33.7 | 67.7 | 60.3 | 50.3 | 53.6 | 74.2 |
| `qd_mixed` | 54.3 | 31.0 | 68.3 | 53.0 | 50.7 | 51.5 | 1,583.5 |
| `qd_nvfp4` | 51.3 | 28.7 | 60.7 | 50.7 | 47.0 | 47.7 | 1,303.9 |
| `qd_nvint4` | 44.7 | 29.0 | 64.0 | 54.0 | 46.7 | 47.7 | 1,354.7 |
| `qd_mxfp4` | 41.7 | 28.3 | 61.3 | 56.7 | 44.3 | 46.5 | 372.0 |
| `qd_mxfp4_ends_bf16` | 44.3 | 28.3 | 61.7 | 46.7 | 49.0 | 46.0 | 390.6 |

과제당 300문항의 표준오차는 약 2.6~2.9점이다 (ARC-c 2.6~2.7). 평균 1~2점 차이와 그 순서는 이 범위 안에서 흔들릴 수 있다 (스펙 §10 R6).

## W4 — 사진 검출 (발표 자료용, 양자화만 바꿈)

YOLO11n (체크포인트 sha256 `0ebbc80d4a76…`), 안전 시나리오 사진 11장 (Wikimedia Commons, 라이선스는
`artifacts/safety_demo/src/credits.json`), Studio 의 `run_detect` (신뢰도 ≥ 0.25, 원본과 IoU ≥ 0.5 로 짝지음), geneva A100.
레시피는 모두 `mma_input: dequant` 이고 누산 설계는 바꾸지 않았다 (모든 arm 이 Conv2d 88개를 패치 — `evidence.patched_conv2d`).

| arm | 가중치 | 입력(활성) | 원본이 찾은 사람 22명 중 다시 찾은 수 |
|---|---|---|:-:|
| `int8` | INT8 행 단위 | INT8 행 단위 | **22** |
| `nvfp4` | NVFP4 | NVFP4 | 1 |
| `mxfp4` | MXFP4 | MXFP4 | **0** |
| `mxfp4_ends` | MXFP4 (첫 Conv · 검출 머리 26개는 원본) | MXFP4 | 0 |
| `mxfp4_w_int8_a` | MXFP4 | INT8 행 단위 | 0 |
| `mxfp4_w_only` | MXFP4 | 원본 | 0 |

- **독립 교차 확인** (TriCast 를 쓰지 않음): 순수 PyTorch 로 OCP MX 규칙(32개마다 공유 지수, E2M1 짝수 반올림)을 짜서 Conv2d 88개의
  가중치만 바꾸고 ultralytics 로 그대로 검출 — 사람 수(짝짓지 않은 개수) 원본 23 · INT8 24 · MXFP4 **1**. TriCast 결과와 같은 방향이다.
  원본 수 22 ↔ 23 의 차이는 두 경로의 전처리 · 후처리 차이다.
- 원자료: `runs/20261013_quantizer_w4_photo/` (`results.json` sha256 `a70d1766c81b…`, `results_set2.json` `30f51277c370…`,
  `independent.json` `16d8848b09d9…`, 그림 `fig/photo_c03_int8_mxfp4.png`), 클러스터 `/scratch/uceeeee/tricast/qz_w4`.
- 읽을 때: RTN(그냥 반올림) 4비트이고 GPTQ · AWQ 같은 보정 방법은 Conv2d 에 쓰지 않았다 (TriCast 의 Conv2d 는 RTN · 동적
  양자화만 지원). "4비트로 줄이면 반드시 이렇게 된다"가 아니라 "아무 방법 없이 4비트로 줄이면 이렇게 될 수 있다"의 예시다.

## W5 — Auto-tune v1 스파이크 (2026-10-13 14:20~16:27 UTC)

판정 규칙: [스파이크](../spikes/2026-10-13_autotune_mckp.md) 1~4절 (커밋 `6893e32`) · 보충 2 와 분석 스크립트 (커밋 `ae8bf10`,
measure 결과가 생기기 전). 코드: TriCast `73f7893` 의 수정 없는 클론 `/scratch/uceeeee/tricast/qz_at_73f7893`. 로컬 사본:
`runs/20261014_autotune/{a100,v100}/` (git 밖).

| 실행 | 호스트 · GPU | 결과 | 원자료 (sha256) |
|---|---|---|---|
| 층 × 방법 KL 표 | turin · A100 80GB PCIe | 196 × 7, 668초, 메타 `git_sha 73f7893` · `git_dirty false` · `model_sha c1899de` | `a100/table.json` (`d2280062…`) |
| 같은 표 (교차 확인) | cork · V100-PCIE-16GB | 1,455초, A100 표와 Spearman 0.9995 | `v100/table.json` |
| 풀이 5개 | (CPU) | `auto_4.25` · `auto_4.50` · `auto_5.09` (285/56) · `auto_5.72` (1831/320) · `auto_5.72_kivi2` | `a100/auto_*/solution.json` |
| 동치 증거 | turin | 표 값과 정확히 같음 | `a100/kl_equiv_seed42.json` |
| 검증 KL (seed 43, 11개) + 예측 | turin | C2 ρ 0.973 · 대역 ρ 1.000 · 쌍 3/3 · 중복 0 · 창 겹침 0 | `a100/kl_seed43.json` (`9de91330…`) · `a100/predict.json` (`f5048bff…`) · `a100/c2.json` (`9e5a0fe8…`) |
| PPL (WikiText-2 test 전체) | turin | 아래 표 | `a100/ppl_auto/*.json` · `a100/ppl_followup/*.json` |
| 패치 바닥 대조군 | cork · turin | V100 층 1개 0.00079 · A100 0.00086 | `{v100,a100}/floor/kl_floor_seed42.json` |

| 레시피 | 레시피 해시 | PPL (turin A100) |
|---|---|:-:|
| `auto_4.50` | `624ff1d50b67…` | 25.5308 |
| `auto_5.09` | `41bf10258478…` | 23.8655 |
| `auto_5.72` (KV 없음) | `f1aa6f6c0fbd…` | 23.1649 |
| `auto_5.72_kivi2` | `3b09f59b6928…` | 26.1453 |
| `qd_nvfp4` (같은 GPU 재측정) | `9f2612422bcc…` | 26.1599 — W2 (geneva A100-SXM4) 와 같음 |
| `qd_mixed` (같은 GPU 재측정) | `11b6ac487694…` | 25.9797 — W2 와 같음 |
| `qd_mixed_nokv` (사후 진단 — `qd_mixed` 에서 KV 만 뺌) | `5670837d062d…` | 23.0293 |

## 로컬 검증 (2026-10-13, macOS CPU, repo `.venv`)

| 단계 | 명령 | 결과 (출력 원문) |
|---|---|---|
| 앱 테스트 | `PYTHONPATH=src:. python -m pytest app/tests -q` | `308 passed, 1 warning` |
| 문서 테스트 | `pytest tests/test_ontology.py tests/test_spec_traceability.py tests/test_rag.py -q` | `18 passed` |
| 린트 | `ruff check .` | `All checks passed!` |
| Quantizer 화면 (헤드리스 Chrome) | `node scripts/quantizer/qz_check.mjs …` | `QZ_CHECKS[static] 71/71 passed` |
| tricast 수용 | `python scripts/quantizer/qz_tricast_accept.py …` | `recipes 100, layers checked 19600, methods seen 10/10, method variants 190 …` / `QZ_TRICAST_ACCEPT_OK` |
| 레이어 지도 (헤드리스 Chrome) | `node scripts/quantizer/cdp_check.mjs …` | `CDP_CHECKS 44/44 passed` |

**전체 CPU 스위트** (클러스터 CPU 노드, `cpu_cap.sh` 8코어, conda env `tricast`, GPU 숨김, `pytest -q --ignore=tests/gpu`) — 이번 변경을 담은
트리 스냅샷 `qz_docs_202610131307` 과 수정 없는 `ab489f1` 트리를 같은 조건으로 나란히 돌렸다.

| 트리 | 환경 | 결과 (출력 원문) |
|---|---|---|
| 수정 없는 `ab489f1` | `MKL_CBWR` 없음 | `61 failed, 2213 passed, 2 skipped` |
| 수정 없는 `ab489f1` | `MKL_CBWR=COMPATIBLE` | `1 failed, 2273 passed, 2 skipped` |
| 이번 변경 (`dirty_lines=30`) | `MKL_CBWR=COMPATIBLE` | `1 failed, 2273 passed, 2 skipped` |
| 이번 변경, 남은 1건만 | `MKL_CBWR=COMPATIBLE` + `PYTHONPATH=src` | `tests/test_eval.py::test_lmeval_entrypoint_registers_in_fresh_process` `1 passed` |

- 61건(`tests/test_gptq.py` 등 정확 일치 비교)은 `MKL_CBWR` 를 빼면 수정 없는 트리에서도 똑같이 실패한다 — 이번 변경과 무관한 실행 환경
  차이다. 팀 감사 관례(`MKL_CBWR=COMPATIBLE`)를 처음 실행에서 빠뜨린 지휘 쪽 실수였다.
- 남은 1건은 자식 프로세스가 `tricast` 를 import 하는 테스트라 `PYTHONPATH=src` 가 필요하다 (pytest 의 `pythonpath` 설정은 자식에
  전달되지 않음). 이 조건을 맞추면 2026-10-09 감사와 같은 **2,274 passed · 2 skipped** 다.

## 확인하지 않은 것 · 읽을 때 주의

- **서버 GPU 실행** (`POST /api/quantizer/jobs`): 거부 · 큐 · 결과 형태는 CPU 테스트만 했고, GPU 서버에서 끝까지 돈 적은 없다.
- **레이어 지도의 줄마다 조건이 다르다**: 지도의 9개 방법 중 8개는 W1 리포트(V100, 번들 레시피)이고 NVINT4 만 W2 리포트(H100, dequant)다. 번들 레시피 8개 중 7개는 `scaled` + 칩 누산 프리셋(`int_exact` · `nvidia_blackwell_fp4` · `nvidia_blackwell_fp8` — Blackwell 프리셋의 칩 확인은 보류)이고 GPTQ 만 dequant 다.
  같은 MXFP4 도 두 조건에서 모델 KL 이 0.5937 ↔ 0.5864 로 약 1% 다르므로, NVINT4 를 다른 방법과 맞대 볼 때는 같은 W2 조건의
  `qd_nvfp4` 와 비교한다.
- **결정론**: CLI 는 `torch.use_deterministic_algorithms(True)` 를 부르지 않는다. 같은 트리의 원본 PPL 이 과거 기록과 같아 기준선은
  흔들리지 않았지만, 레시피 실행의 반복 측정은 하지 않았다.
- **W3 슬롯 B 의 로컬 로그 사본** (`runs/20261013_quantizer_w3/qz_w3_B.log` · `qz_w3_lmeval_B.log`)은 실행 도중에 복사한 것이라 DONE 마커가 없다. 완료는 각 JSON 의 `status: complete` 와 `summary.md` 로 판정했다 (클러스터 원본 로그는 `/scratch/uceeeee/tricast/logs/`).
- **크기 · 속도**: 이 실행들은 fake-quant 라 압축된 가중치를 저장하지도, 저정밀 커널로 돌리지도 않는다. 크기는 형식 정의로 계산한
  값이고 속도 이득은 재지 않았다.
