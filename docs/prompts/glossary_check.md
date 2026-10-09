# 용어집 작동 확인 — AGENTS.md 제거 실험 (2026-10-08)

강의 4 지침의 "용어집 작동 확인"과 강의 5의 에이전트 하니스 검증(지침을 넣은 상태와 뺀 상태에서 같은 요청을 비교)을
한 번에 수행한 기록이다. 판정은 TriCast 레시피 로더(`tricast.recipe.load_recipe`, 스키마
`src/tricast/schemas/recipe.schema.json`)로 하고, 회차별 출력 원문과 차이를 아래에 남겼다.

## 1. 설계

| 항목 | 내용 |
|---|---|
| 에이전트 | Claude Code general-purpose 하위 에이전트, 회차·조건마다 새 세션 (2026-10-08) |
| 조건 A | `AGENTS.md` §2 용어집 발췌를 프롬프트 앞에 넣음 (§3 절대 규칙은 넣지 않음) |
| 조건 B | 용어집 없이 같은 요청만 |
| 공통 지시 | 파일 읽기·검색·도구 사용 없이 주어진 텍스트만으로 답함 |
| 판정 | 출력 YAML 에 `name` 만 채워 `load_recipe` 로 읽음. 통과하면 해석된 값이 기대값과 같은지 대조 |

요청 (모든 회차에서 같은 문장):

```text
TriCast 레시피 YAML 조각을 써줘. 조건: 어떤 NPU 칩은 곱 32개를 한 묶음으로 가장 큰 지수에 맞춰 정렬한 뒤
누산기에 소수 13비트만 남기고(나머지는 버림) 누산기 값과 함께 더한다. 가중치는 FP8 E4M3 행 단위 스케일,
활성값은 텐서 단위 스케일이고 칩이 반올림 대신 버림을 쓴다. 첫 번째와 마지막 디코더 블록은 양자화하지 않는다.
YAML 만 출력하고, 마지막 줄에 사용한 필드 이름을 쉼표로 나열해.
```

기대값 (요청을 용어집 어휘로 옮긴 것): `mma: {algorithm: cofda, chunk_size: 32, f_bits: 13, c_mode: fused}`,
`weight: {format: fp8_e4m3, granularity: row, rounding: rtz}`, `activation: {granularity: tensor, rounding: rtz}`,
`overrides: [{layers: "0,-1", skip: true}]`.

## 2. 결과

| 회차 | 조건 | 에이전트가 쓴 필드 | 로더 판정 |
|---|---|---|---|
| 1 | B (용어집 없음) | `recipe`, `weights`, `scale_granularity`, `accumulator.block_size`, `fraction_bits`, `skip_blocks` … (13개) | 거부 — `recipe: Additional properties are not allowed ('recipe' was unexpected)` |
| 1 | A (용어집 있음) | `defaults.weight/activation` (`format`, `granularity`, `rounding: rtz`), `mma` (`algorithm: cofda`, `chunk_size`, `f_bits`, `c_mode: fused`), `overrides[].layers`, `skip` | 거부 — `overrides[0].layers: [0, -1] is not of type 'string'` |
| 2 | A (보완한 용어집) | 1회차 A 의 필드에 `scale.method: absmax` (가중치·활성값) 추가, `layers: "0,-1"` | **통과** — 레시피 SHA-256 `3685227b5ece…`, mma `cofda F=13 CS=32 fused`, weight `fp8_e4m3 row rtz`, activation `fp8_e4m3 tensor rtz` |

- 조건 B 의 필드 이름 13개 중 스키마와 같은 이름은 `name`·`format`·`rounding` 3개였고, 셋 다 스키마와 다른 위치에 있었다.
- 조건 A 는 필드 이름이 모두 스키마와 같았고, `F`(`f_bits`)와 `G`(`g_bits`)를 구분했으며, "버림"을 `rtz`로 옮겼다.
  1회차의 거부 원인(`layers` 의 타입)은 용어집이 `layers` 를 "decoder 블록 번호"라고만 적고 문자열이라는 것을 적지 않은 데 있었다.
- 2회차가 더한 `scale.method: absmax` 는 기본값과 같아 해석 결과가 바뀌지 않는다 (1회차 A 출력의 `layers` 만 문자열로 바꾼
  사본도 같은 SHA-256 으로 통과).
- 요청에 없는 활성값 형식(`fp8_e4m3`)을 1회차 A·B 는 주석으로 가정이라고 적었고 2회차는 적지 않았다. 이 실험의 발췌에는
  §3 절대 규칙 4(요청에 없는 값은 가정으로 적는다)를 넣지 않았으므로 판정 대상에서 뺐다.

## 3. 반영

- `AGENTS.md` §2 용어집: `layers` 를 "decoder 블록 번호 **문자열**, 예 `"0,-1"` · `"0-7"`"로 고쳤다.
- 2회차에서 같은 요청이 로더를 통과했다.
- 이전 확인 기록 (2026-09-30)은 [delegation_examples.md §7](./delegation_examples.md#7-용어집-작동-확인-강의-4-지침--2026-09-30)에 있다.

## 4. 출력 원문

<details>
<summary>1회차 B (용어집 없음)</summary>

```yaml
# 스키마를 확인하지 않았다(파일 조회 금지 조건). 아래 필드 이름은 추정이므로 실제 TriCast 스키마와 맞춰 봐야 한다.
recipe:
  name: npu_fp8_e4m3_trunc_acc13
  rounding: truncate            # 칩 전체에 적용: 양자화 변환과 누산 모두 반올림 대신 버림
  weights:
    format: fp8_e4m3
    scale_granularity: per_row
  activations:
    format: fp8_e4m3            # 가정: 요청에 활성값 포맷이 없어 E4M3로 둠
    scale_granularity: per_tensor
  accumulator:
    block_size: 32              # 곱 32개를 한 묶음으로 처리
    alignment: max_exponent     # 묶음 안에서 가장 큰 지수에 맞춰 정렬
    fraction_bits: 13           # 정렬 후 소수 13비트만 남기고 나머지는 버림
    include_accumulator: true   # 누산기 값과 함께 더함. 가장 큰 지수를 고를 때 누산기도 포함하는지는 요청만으로 알 수 없음
  skip_blocks: [0, -1]          # 첫 번째와 마지막 디코더 블록은 양자화하지 않음. -1(마지막 블록) 같은 음수 인덱스를 스키마가 받는지는 확인 못 함
```

</details>

<details>
<summary>1회차 A (용어집 있음)</summary>

```yaml
defaults:
  weight:
    format: fp8_e4m3
    granularity: row
    rounding: rtz
  activation:
    format: fp8_e4m3        # 활성값 포맷은 요청에 없어서 가중치와 같은 E4M3로 가정
    granularity: tensor
    rounding: rtz
  mma:
    algorithm: cofda        # 곱 32개를 한 묶음으로 최대 지수에 정렬하는 1단계 방식이라 gdfs가 아님
    chunk_size: 32
    f_bits: 13              # 누산기 소수 13비트(F)이고 g_bits(G)가 아님. 버림은 cofda 기본 동작
    c_mode: fused           # 누산기 값을 같은 정렬 단계에 넣어 함께 더함
overrides:
  - layers: [0, -1]
    skip: true
```

</details>

<details>
<summary>2회차 A (보완한 용어집)</summary>

```yaml
defaults:
  weight:
    format: fp8_e4m3
    granularity: row
    scale:
      method: absmax
    rounding: rtz
  activation:
    format: fp8_e4m3
    granularity: tensor
    scale:
      method: absmax
    rounding: rtz
  mma:
    algorithm: cofda
    chunk_size: 32
    f_bits: 13
    c_mode: fused
overrides:
  - layers: "0,-1"
    skip: true
```

</details>
