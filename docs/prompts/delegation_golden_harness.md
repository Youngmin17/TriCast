> 실행 기록은 2026-10-08 자동 실행 원본이고, 수용 판단은 Youngmin17 이 2026-10-09 에 내렸다 (§6).

# 골든 테스트 하니스 결함 주입 검사 (강의 5 지침 C절)

## 1. 대상과 범위

| 항목 | 값 |
|---|---|
| 하니스 파일 | `tests/test_golden.py` (303줄), `tests/harness/golden_cases.yaml` (318줄, 케이스 29개) |
| 기준 커밋 | `3820b5b` (docs: rebuild README slides from the v3 deck) |
| 작업 위치 | 기준 커밋의 별도 클론 (작업 트리는 건드리지 않음) |
| NADPE 벡터 | `git ls-files tests/data/nadpe` = 21개 (fp8 12, fp4 8, manifest.json 1) — 클론에도 모두 존재 |
| 실행 명령 | `PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_golden.py -q -rs -p no:cacheprovider` |
| 환경 | Python 3.14.5, torch 2.11.0, pytest 9.0.2, macOS-26.6.2-arm64 (CUDA 없음) |

검사 범위는 CPU 에서 실행되는 골든 케이스와 보조 검사(`test_golden_manifest`)다. GPU 전용 `e2e_bf16_passthrough_ppl` 은 이 환경에서 skip 되므로 결함 주입 검사 대상에서 빠진다.

## 2. 수집과 실행 확인

`--collect-only -q` 결과는 `48 tests collected in 0.23s`. 노드 구성은 다음과 같다.

| 구분 | 수 | 근거 |
|---|---|---|
| YAML 골든 케이스 | 29 | `yaml.safe_load` 길이 29 |
| 그중 펼쳐지지 않는 케이스 | 27 | cast 7, quantize 3, mma 4, recipe_error 5, preset 1, runner_env 1, parser 5, e2e 1 |
| 그중 벡터 파일별로 펼쳐지는 케이스 | 2 | `nadpe_all_fp8_vectors` → 12 노드, `nadpe_all_fp4_vectors` → 8 노드 |
| 펼쳐진 골든 노드 (`test_golden[...]`) | 47 | 27 + 12 + 8 |
| 보조 검사 (`test_golden_manifest`) | 1 | 골든 케이스가 아님. id 중복, AC 포괄 (기준 커밋은 AC1~AC7, 이후 SPEC v1.0 번호로 AC1~AC6), 필수 키, marks 허용값 검사 |
| 수집 합계 | 48 | 47 + 1 |

기준 실행:

| passed | failed | skipped | 요약 줄 원문 |
|---|---|---|---|
| 47 (골든 46 + 보조 1) | 0 | 1 | `47 passed, 1 skipped in 22.84s` |

skip 사유 원문: `SKIPPED [1] tests/test_golden.py:294: needs a CUDA GPU with triton` — `e2e_bf16_passthrough_ppl` 이 `marks: [gpu, slow]` 를 달고 있어 `tests/conftest.py` 가 CUDA 없음을 보고 skip 한다. nadpe 케이스는 벡터가 있으므로 "vectors are absent" skip 이 하나도 발생하지 않았다.

## 3. 결함 주입 결과

절차: 테스트 파일과 YAML 은 고정하고 구현 파일 하나만 바꾼 뒤 전체 실행, 실패 노드와 단언 메시지 첫 줄을 기록하고 `git checkout -- <file>` 로 원복했다. 각 결함 직후 `git status --short` 가 비어 있음을 확인한 다음 다음 결함으로 넘어갔다. 주입 diff 와 전체 출력 로그는 실행 환경에만 남기고 레포에는 넣지 않았다. 아래 표는 그 로그에서 옮긴 원문이다.

### 3.1 요약표

| # | 주입한 결함 (파일:함수) | 실패해야 할 케이스 (사전 예측) | 실제 결과 | 판정 |
|---|---|---|---|---|
| F1 | `src/tricast/reference/cast.py:_round_up` — RNE 분기가 항상 올림 없음(RTZ)을 반환 | cast 동점: fp8 비포화, fp4, bf16 / RNE 에 기대는 quantize 3개 / 출력을 bf16 RNE 로 반올림하는 nadpe 20개. RTZ 와 결과가 같은 cast 4개(fp8 포화, int4, ue4m3 ×2)는 통과 예상 | 26 failed: `cast_fp8_rne_nonsaturating_boundary`, `cast_fp4_rne_four_ties`, `cast_bf16_rne_ties`, quantize 3개, nadpe fp8 12개, nadpe fp4 8개. 예측한 cast 4개는 통과 | 검출 (예측과 일치) |
| F2 | `src/tricast/reference/mma.py:_cofda` — 함수 시작에서 `spec = spec.with_(f_bits=40)` 로 바꿔 F 절삭을 사실상 없앰 | `mma_cofda_three_fraction_bits`, `mma_cofda_c_fused`, CoFDA 사례가 든 nadpe 벡터. F=13 에서 이미 절삭이 없는 `mma_cofda_thirteen_fraction_bits` 와 `mma_cofda_c_decoupled` 는 통과 예상 | 22 failed: `mma_cofda_three_fraction_bits`, `mma_cofda_c_fused`, nadpe fp8 12개, nadpe fp4 8개. thirteen, decoupled 는 통과 | 검출 (예측과 일치) |
| F3 | `src/tricast/recipe.py:load_recipe` — `_validate(data)` 호출을 `pass` 로 대체 | 과제 지시: `recipe_error_*` 5개. 코드 읽기 예측: 하위 단계(`_quant_spec`, `SparsitySpec`, `LinearSpec.__post_init__`)가 같은 경로 접두어로 다시 거부하는 3개는 통과, `group_size`·`c_mode` 2개만 실패 | 2 failed: `recipe_error_negative_group_size`, `recipe_error_invalid_c_mode`. 나머지 3개 통과 | 검출 (5개 중 2개 케이스만 반응) |
| F4 | `src/tricast/mma/spec.py:PRESETS` — `int_exact` 의 provenance 를 `""` 로 | `presets_all_have_provenance` | 1 failed: `presets_all_have_provenance` | 검출 (예측과 일치) |
| F5 | `src/tricast/agent/offline.py:parse_offline` — `mma["f_bits"]` 가 None 이면 assumption 없이 13 을 채움 | `f_bits` 를 unspecified 로 둔 parser 4개 (`precision_ko`, `accumulator_ko`, `precision_en`, `f_bits_ko`). F=13 을 명시한 `parser_missing_chunk_size_en` 은 통과 예상 | 4 failed: 예측한 4개. `chunk_size_en` 통과 | 검출 (예측과 일치) |

### 3.2 실행별 요약 줄과 단언 메시지 (원문)

**F1** — `26 failed, 21 passed, 1 skipped in 4.80s`

| 실패 노드 | 단언 위치 | 메시지 첫 줄 |
|---|---|---|
| `cast_fp8_rne_nonsaturating_boundary` | `test_golden.py:52` (`_equal`) | `AssertionError: (tensor([ 448.,  448.,  448., -448., -448.]), tensor([ 448.,  448.,   nan, -448.,   nan]))` |
| `cast_fp4_rne_four_ties` | `:52` | `AssertionError: (tensor([0.0000, 0.5000, 2.0000, 4.0000]), tensor([0., 1., 2., 4.]))` |
| `cast_bf16_rne_ties` | `:52` | `AssertionError: (tensor([ 1.0000,  1.0078, -1.0000, -1.0078]), tensor([ 1.0000,  1.0156, -1.0000, -1.0156]))` |
| `quantize_mxfp4_one_block` | `:52` | `AssertionError: (tensor([[ 0.0000,  0.5000,  2.0000,  6.0000, -0.0000, -0.5000, ...` |
| `quantize_nvfp4_one_block` | `:52` | `AssertionError: (tensor([[ 0.0000,  0.5000,  2.0000,  4.0000,  6.0000, -0.0000, ...` |
| `quantize_int4_zero_point_one_group` | `:52` | `AssertionError: (tensor([[-7., -4., -3., -1.,  7.]]), tensor([[-7., -5., -3., -1.,  7.]]))` |
| nadpe fp8 12개 | `:203` (`torch.equal(actual, expected)`) | 예: `fp8_realistic-scaled_33x17x96.pt case 0: MMASpec(algorithm='cofda', f_bits=3, chunk_size=16, ...` (unit 6개는 `case 3`, f_bits=9) |
| nadpe fp4 8개 | `:242` | 예: `fp4_mxfp4_realistic_33x17x128.pt case 4: MMASpec(algorithm='gdfs', f_bits=9, ...` (nvfp4 4개는 `case 0`, gdfs f_bits=7) |

**F2** — `22 failed, 25 passed, 1 skipped in 5.48s`

| 실패 노드 | 단언 위치 | 메시지 첫 줄 |
|---|---|---|
| `mma_cofda_three_fraction_bits` | `:52` | `AssertionError: (tensor([[2.2656]]), tensor([[2.2500]]))` |
| `mma_cofda_c_fused` | `:52` | `AssertionError: (tensor([[18.]]), tensor([[16.]]))` |
| nadpe fp8 12개 | `:203` | 모두 `case 0: MMASpec(algorithm='cofda', f_bits=3, chunk_size=16, c_mode='fused', ...` |
| nadpe fp4 8개 | `:242` | mxfp4 4개는 `case 64`, nvfp4 4개는 `case 32`, 모두 `MMASpec(algorithm='cofda', f_bits=3, chunk_size=16, ...` |

참고: 메시지의 `f_bits=3` 은 테스트가 만든 spec 을 출력한 것이다. 주입은 `_cofda` 안에서 40 으로 덮어썼다.

**F3** — `2 failed, 45 passed, 1 skipped in 22.30s`

| 실패 노드 | 단언 위치 | 메시지 |
|---|---|---|
| `recipe_error_negative_group_size` | `:81` (`pytest.raises(..., match=...)`) | `AssertionError: Regex pattern did not match.` / `Expected regex: '^defaults\\.weight\\.group_size:'` / `Actual message: "defaults.weight: granularity 'group' needs group_size > 0"` |
| `recipe_error_invalid_c_mode` | `:81` | `AssertionError: Regex pattern did not match.` / `Expected regex: '^defaults\\.mma\\.c_mode:'` / `Actual message: "defaults.mma: c_mode must be 'fused' or 'decoupled'"` |

통과한 3개(`unknown_format`, `sparsity_n_not_below_m`, `outliers_without_weight_spec`)는 스키마 검증 없이도 하위 생성자가 같은 경로 접두어(`defaults.weight.format:`, `defaults.sparsity.n...`, `defaults.outliers...`)로 `ValueError` 를 내기 때문에 통과했다. 이 3개 케이스는 "어느 단계에서 거부했는가"를 구분하지 못한다.

**F4** — `1 failed, 46 passed, 1 skipped in 22.34s`

| 실패 노드 | 단언 위치 | 메시지 첫 줄 |
|---|---|---|
| `presets_all_have_provenance` | `:89` | `AssertionError: int_exact` |

**F5** — `4 failed, 43 passed, 1 skipped in 22.75s`

| 실패 노드 | 단언 위치 | 메시지 첫 줄 |
|---|---|---|
| `parser_missing_precision_ko` | `:171` | `AssertionError: unrequested recipes[0].mma.f_bits=13 has no exact-path assumption` |
| `parser_missing_accumulator_ko` | `:171` | 동일 |
| `parser_missing_precision_en` | `:171` | 동일 |
| `parser_missing_f_bits_ko` | `:171` | 동일 |

다섯 결함 모두 최소 1개 이상의 골든 노드를 실패시켰다. 예측과 어긋난 결과는 없었다. F3 은 과제가 적은 예측("recipe_error_* 실패")보다 실제 실패가 적었고(5개 중 2개), 이는 주입 전에 코드를 읽고 예측한 결과와 같다.

## 4. 부록 C2·C7 결과

| 검사 | YAML 변경 (클론에서만, 검사 후 원복) | 수집 결과 | 실행 결과 | 판정 |
|---|---|---|---|---|
| C2 유효 케이스 추가 | `cast_bf16_rne_ties` 를 복제하고 id 를 `cast_bf16_rne_ties_tmp_c2` 로 변경 | `49 tests collected in 0.17s`, 노드 `test_golden[cast_bf16_rne_ties_tmp_c2]` 수집 | `-k tmp_c2 -v`: `test_golden[cast_bf16_rne_ties_tmp_c2] PASSED` / `1 passed, 48 deselected in 0.17s`. 전체: `48 passed, 1 skipped in 22.19s` | 새 케이스가 코드 수정 없이 수집·실행됨 |
| C7a `expected` 없는 케이스 | cast 케이스에서 `expected` 를 빼고 `note` 만 둠 (`cast_bf16_tmp_c7_no_expected`) | 49개 수집 (수집 단계에서는 막히지 않음) | `2 failed, 46 passed, 1 skipped in 22.03s` — `test_golden_manifest`: `AssertionError: assert {'ac', 'expected', 'id', 'input', 'kind', 'note'} <= dict_keys(['id', 'ac', 'kind', 'input', 'note'])`, 해당 노드: `KeyError: 'expected'` | 보조 검사에서 걸림 |
| C7b 오타 키 `expectd` | 위와 같은 케이스에 `expectd: [1]` (`cast_bf16_tmp_c7_typo_key`) | 49개 수집 | `2 failed, 46 passed, 1 skipped in 22.21s` — `test_golden_manifest`: `AssertionError: assert {...} <= dict_keys(['id', 'ac', 'kind', 'input', 'expectd', 'note'])`, 해당 노드: `KeyError: 'expected'` | 보조 검사에서 걸림 |

C7 두 경우 모두 수집 단계가 아니라 실행 단계(`test_golden_manifest` 의 필수 키 단언과 해당 노드의 `KeyError`)에서 걸렸다.

## 5. 원복 확인

- 각 결함과 C2·C7 검사 직후 `git checkout -- <file>` 를 실행하고 `git status --short` 가 빈 출력인지 확인했다.
- 최종: `git status --short` 줄 수 0, `git rev-parse --short HEAD` = `3820b5b`.
- 최종 재실행 요약 줄: `47 passed, 1 skipped in 22.01s` (skip 사유 `needs a CUDA GPU with triton`, 기준 실행과 같음).
- 모든 변경은 클론 안에서만 했다. 원본 레포에는 쓰지 않았다.

## 6. 수용 판단

| 항목 | 판단 |
|---|---|
| 수집·실행 확인 결과 | 수용 — 골든 노드 47 (YAML 29 케이스) + 보조 검사 1, skip 은 GPU 전용 1건뿐 |
| F1~F5 검출 결과 | 수용 — 다섯 결함 모두 예측한 케이스의 단언문에서 실패 |
| F3 부분 검출 (5개 중 2개) | 현재 케이스로 충분 — 스키마를 빼도 하위 생성자가 같은 필드 경로로 거부하므로 AC5 ("모델을 건드리기 전에, 위반한 필드 경로를 담아 실패")는 그대로 판정된다. 스키마 전용 케이스는 추가하지 않는다 |
| §7 보완점 | 1·3 반영 (허용 키 검사, `test_golden_vectors_present`), 2 는 위 F3 판단으로 대체, 4 는 `/check` · `/golden` 명령을 `-rfs` 로 바꿔 반영, 5 는 의도된 케이스 구성 |
| 검토자 / 날짜 | Youngmin17 / 2026-10-09 |

## 7. 발견한 하니스 보완점

검사 중에는 하니스를 수정하지 않았다. 검사 뒤 1·3·4번을 반영했다 — `test_golden_manifest` 가 허용 키 밖의 키를 거부하고, 새 보조 검사 `test_golden_vectors_present` 가 `manifest.json` 의 sha256 으로 FP8·FP4 벡터 20개를 대조하며 nadpe glob 이 비면 실패한다. `/check` · `/golden` 은 `-rfs` 로 실패 요약까지 낸다. 2번은 §6 의 F3 판단으로 정리했다.

1. **알 수 없는 키를 검사하지 않음 (실행으로 확인)**: `test_golden_manifest` 는 필수 키가 들어 있는지(`<=`)만 본다. `marks` 를 `mark: [gpu]` 로 잘못 쓴 임시 케이스 `cast_bf16_tmp_extra_typo_marks` 를 넣었을 때 `test_golden_manifest PASSED`, 해당 노드 `PASSED` (`2 passed, 47 deselected in 0.19s`) 였다. GPU 전용 케이스에서 이 오타가 나면 CPU 환경에서 skip 되지 않고 실행된다. 허용 키 집합과 비교하는 단언이 있으면 잡힌다. (C7b 의 `expectd` 는 `expected` 가 함께 빠졌기 때문에 걸린 것이다. `expected` 와 다른 오타 키가 함께 있으면 걸리지 않는다는 점은 위 `mark` 실행과 같은 원리에 따른 추론이다.)
2. **recipe_error 케이스가 거부 단계를 구분하지 못함 (F3 결과)**: 5개 중 3개는 스키마 검증을 통째로 빼도 통과한다. 스키마가 실제로 작동하는지 확인하려면 스키마만 거부하고 하위 생성자는 받아들이는 입력이 필요하다. 또는 메시지 본문(예: jsonschema 메시지 형식)까지 단언해야 한다.
3. **nadpe glob 이 비면 skip 으로 바뀜 (코드 읽기 관찰, 이번에는 실행으로 재현하지 않음)**: `_parameters()` 는 glob 결과가 비면 펼치지 않은 단일 노드를 만들고, `_nadpe`/`_nadpe_fp4` 는 이를 "vectors are absent" skip 으로 처리한다. glob 오타나 벡터 누락이 실패가 아니라 skip 으로 보고된다. 이번 실행에서는 벡터 20개가 모두 있어 skip 이 없었으나, 벡터 개수(`manifest.json` 기준)를 단언하는 보조 검사가 없다.
4. **실행 명령의 `-rs` 는 실패 요약을 출력하지 않음**: 지침의 명령(`-q -rs`)은 skip 사유만 요약하므로 실패 노드 목록은 본문 traceback 에서 직접 추려야 했다. 결함 주입 기록용으로는 `-rfs` 가 편하다. (하니스가 아닌 명령 문제)
5. **RNE→RTZ 에 둔감한 cast 케이스 (F1 결과, 결함은 아님)**: `cast_fp8_rne_saturating_boundary`, `cast_int4_symmetric_saturation`, `cast_ue4m3_*` 2개는 RTZ 로 바꿔도 통과했다. 이 케이스들의 목적은 포화·부호 처리이므로 문제는 아니다. 다만 RNE 동점 판정을 직접 검사하는 cast 케이스는 3개(fp8 비포화, fp4, bf16)뿐이다.
