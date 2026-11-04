# 레인 노트 — 외부 공식 비교군 재현 가능성 (2026-11-04)

> 정적 읽기만 했다. 공식 소스와 논문을 읽었고, 어떤 코드도 실행하지 않았다. GPU 시간은 2 × params × tokens / 약 170 TFLOP/s 로 계산한 모델값이고 실측이 아니다.
> 원본 소스 사본은 세션 scratchpad 에만 있고 저장소에는 넣지 않았다.

## 요약

| 비교군 | 공식 repo @ commit (라이선스) | 탐색 진입점 | 탐색 공간 | 적합도 | 측정 수 상한 | 앵커 | 재현성 |
|---|---|---|---|---|---|---|:-:|
| EvoPress (ICML'25) | IST-DASLab/EvoPress @ `a66a6974` (Apache-2.0) | DB `quant.py` → `evo_quant_search.py:main` | Linear 별 정수 비트 level 1개 (공식 GPTQ g128 사본 2–6 bit), 같은 크기 층끼리 ±1 swap | KL(dense‖quant), 3단 선택 2048 / 16384 / 131072 토큰 | `--generations` · `--offspring` · `--survivors_per_selection` 등. 기본 약 22,350회 | arXiv 2410.14649v2 Tab 7: Llama-2-7B 3-bit Wiki2/C4 균일 6.16/7.96 → 5.70/7.87 (seq 4096, 우리 2048 하네스와 직접 비교 불가) | 높음 |
| ModelOpt AutoQuantize | NVIDIA/Model-Optimizer @ `ba053c3f` (Apache-2.0). 우리 env 용 pin `nvidia-modelopt==0.43.0` | `mtq.auto_quantize` (`kl_div` · `gradient`) | 그룹(q/k/v · gate/up · o · down) 별 형식 하나 또는 무양자화. HQQ · NF4 표현 불가 | `kl_div`: 그룹 하나만 바꾼 모델의 KL 합 → minimax 이진 탐색. `gradient`: fwd+bwd → LP | 없음 (측정 수 = 그룹 × 형식으로 고정) | 공개 수치 없음 | 중간 |
| AMQ (EMNLP'25) | dlwns147/amq @ `61833490` (Apache-2.0) | `amq_sensitivity` → `amq_search` → `amq_quantization --method awq` | Linear 별 {2,3,4} g128 (하드코딩). 탐색은 HQQ proxy, 최종은 AWQ/GPTQ/OWQ | JSD, RBF 대리 모델 + NSGA-II | `--n_doe` + `--iterations` × `--n_iter`. 기본 10,250회 + 민감도 n_linear | KB 카드: Llama-2-7B 3.0 bit Wiki2 6.83 (Tab 1) | 중간 |

## 같은 측정 수 비교로 맞추는 방법

### EvoPress (1순위)

1. **공식 env 앵커 게이트.** torch 2.4 / transformers 4.43.4 에서 Llama-2-7B 3-bit 를 돌려 Tab 7 의 ±3% 안인지 본다.
2. **우리 env 로 이식.** model_type 허용 목록에 qwen 이 없어서 Qwen3 는 "EvoPress-adapted" 다.
3. **등측정 arm.**
   - DB(`<layer>/<level>.pth`)에 우리 풀의 GPTQ int2/3/4 가중치를 넣는다.
   - 다음 세 값을 같게 둔다: `--calibration_tokens` = `--tokens_per_selection` = 우리 창 토큰 수. `--survivors_per_selection 1`.
   - 이렇게 하면 평가 1회 = 측정 1회다.
   - 비용식에는 scale overhead 가 없으므로 `target = B − 0.25` 로 바꿔 넣는다.
4. **방법 혼합 · INT8.** level 이 연속 정수여야 하므로, 방법을 섞거나 INT8 을 넣으려면 패치가 필요하다 ("adapted" 로 표기).

### ModelOpt AutoQuantize

- 측정 수를 상한으로 맞출 수 없다. 그래서 any-time 곡선 위에 고정점 하나로만 표시한다.
- 우리 풀 가운데 INT{2,3,4} · AWQ · INT8 부분집합 위에서만 비교된다.
- recipe 비용에서 scale overhead 가 빠진다 (algorithms.py 206–269).
- 0.43.0 에 GPTQ · RotateConfig · per-format effective_bits 가 있는지는 확인하지 않았다 (unknown).

### AMQ

- 측정 수는 `n_linear + n_doe + iterations × n_iter` 로 맞춘다 (예: 224 + 100 + 13 × 50 ≈ 974).
- 한 번의 탐색 결과에서 예산 4개를 모두 고를 수 있다.
- Qwen3 config 를 추가해야 한다 (데이터 추가일 뿐 알고리즘 변경은 아님).

## 위험 · 미확인

- **EvoPress**
  - target logits 를 CPU 에 전부 둔다: 524k 토큰 fp16 기준 Llama-2 34 GB, Qwen3 159 GB.
  - DB 는 8B 모델에서 level 하나당 약 14 GB 다. `/scratch` 사용률이 98% 라 용량을 먼저 확보해야 한다.
  - Tab 16 의 Llama-2-7B 행은 요약 단계에서 잘못 옮겨졌을 수 있다 (C4 < Wiki2). PDF 원문과 대조하기 전에는 쓰지 않는다.
- **ModelOpt**
  - main 브랜치는 우리 env 와 맞지 않는다 (torch ≥ 2.10, transformers ≥ 5.5).
  - 혼합 recipe 에서 GPTQ 가 가중치를 제자리에서 바꾸면 다른 recipe 를 오염시킬 수 있다 (unknown).
- **AMQ**
  - transformers 4.45.2 고정과 HQQ 버전의 호환성이 unknown 이다.
  - KB 카드의 탐색 비용 "~5 A100-h" 와 FLOP 추정(약 55 h)이 10배 어긋난다. Tab 4 를 다시 확인해야 한다.
- **공통: "같은 측정 수"의 정의.** EvoPress 공식판은 단계마다 토큰 수가 다르다. 그래서 주 비교는 단일 단계 + 고정 창 판으로 하고, 공식 다단 판은 토큰 환산 수치로 함께 적는다. 이 정의는 논문 본문에 명시한다.

## 근거 위치 (요지)

- **EvoPress**
  - `evo_quant_search.py`: 26–42 (DB), 45–51 (fitness), 145–197 (탐색 knob), 299–304 · 362 (비용 = numel × level)
  - `scripts/run_quant_search.sh`: 7, 24–28
  - `src/model_utils.py`: 44–50 (model_type 허용 목록)
  - 논문 v2 Tab 7 · 8 · 16, App F.2
- **ModelOpt**
  - `model_quant.py`: 763–960
  - `algorithms.py`: 165–269, 1936–2066
  - `config.py`: 1731–1870
  - `pyproject.toml`: 22–90
  - PyPI 0.43.0 requires_dist
- **AMQ**
  - `amq/search/optimizer.py`: 53–78, 93–107, 258–267
  - `amq/evaluation/evaluator.py`: 42–103
  - `amq/utils/func.py`: 101–114
