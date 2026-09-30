
[Lane I — 통합: Recipe · EmuLinear · patch · calibrate · eval (ppl, lm-eval) · runner · CLI · 레시피]
소유 파일 (이것만 생성/수정):
  src/tricast/recipe.py, src/tricast/schemas/recipe.schema.json, src/tricast/nn/__init__.py,
  src/tricast/nn/linear.py, src/tricast/nn/patch.py, src/tricast/calibrate.py, src/tricast/eval/__init__.py,
  src/tricast/eval/ppl.py, src/tricast/eval/lmeval.py, src/tricast/eval/runner.py, src/tricast/eval/envinfo.py,
  src/tricast/cli.py, configs/recipes/*.yaml, configs/sweeps/*.yaml,
  tests/test_recipe.py, tests/test_nn_patch.py, tests/test_eval.py, tests/test_cli.py

의존 (동시 구현 중 — ENGINE 계약 이름 그대로 사용): tricast.quant.api.{quantize, fake_quant, resolve_backend},
tricast.quant.qtensor.QTensor, tricast.quant.observer.ObserverState, tricast.mma.api.{gemm, as_operand},
tricast.transforms.{CalibStats, StatsCollector, fit_transform, LinearTransform},
tricast.weight_quant.quantize_weight.

구현 (ENGINE §6):
1) src/tricast/recipe.py — @dataclass LinearSpec(weight: QuantSpec|None, activation: QuantSpec|None,
   mma: MMASpec, transform: TransformSpec, weight_algo: WeightAlgoSpec); @dataclass Recipe(name, description,
   defaults: LinearSpec, include, exclude, overrides: list[tuple[str, dict]], calibration: dict | None,
   backend). load_recipe(path | dict | name) — 이름이면 configs/recipes/<name>.yaml (레포 루트 기준; 패키지
   설치 시에도 찾도록 경로 탐색). Recipe.spec_for(module_name) -> LinearSpec | None (include/exclude fnmatch,
   overrides 첫 매치를 defaults 위에 필드 단위 병합). needs_calibration 속성 (observer / smoothquant / awq /
   gptq 존재 시). to_dict(), sha256 해시(정규 JSON). 문자열 축약: weight: "nvfp4" == {scheme: nvfp4},
   mma: "nvidia_hopper_fp8" == {preset: ...}, transform: "hadamard" == {kind: hadamard},
   weight_algo: "gptq" == {kind: gptq}. 검증 오류 메시지에 레시피 경로 (예: "overrides[1].weight.scale.method").
2) src/tricast/schemas/recipe.schema.json — JSON Schema draft 2020-12, 위 구조 전체. enum 값은
   spec 모듈 상수와 일치 (테스트로 확인). description 을 사람이 읽기 좋게 (이 스키마는 이후 LLM 구조화
   출력 스키마로도 재사용). load_recipe 가 jsonschema 로 1차 검증.
3) src/tricast/nn/linear.py — EmuLinear(nn.Module) (§6.2): from_linear(linear, spec: LinearSpec, name,
   backend); transform 적용 후 가중치를 1회 quantize (weight_algo rtn; gptq 는 calibrate 후 재양자화) →
   as_operand 결과 캐시; bias fp32. forward(x): [..., K] → [rows, K] → transform.apply_activation →
   activation 양자화 (observer 있으면 ObserverState.observe 로 amax) → gemm → [..., N], x.dtype 반환.
   training 모드: custom autograd (forward = 에뮬 gemm, backward: grad_x = grad_y @ Ŵ, grad_W = grad_yᵀ @ x̂,
   양자화기는 STE). 모드 "calibrate"(통계 수집 + 동적 양자화) / "frozen". extra_repr 에 요약.
   src/tricast/nn/patch.py — patch_model(model, recipe, *, backend=None) -> PatchReport(patched: list[(name,
   summary)], skipped); unpatch_model(model); iter_emulinear(model). lm_head 등 exclude 존중.
4) src/tricast/calibrate.py — calibrate(model, recipe, tokenizer=None, *, texts=None, input_ids=None,
   samples=None, seqlen=None, seed=None, device=None): 데이터는 recipe.calibration.dataset ∈
   {wikitext2 (Salesforce/wikitext, wikitext-2-raw-v1, train), c4 (allenai/c4, en, validation 첫 shard),
   pile (NeelNanda/pile-10k)} 에서 seed 로 무작위 seqlen 윈도우 samples 개 (texts/input_ids 인자가 있으면
   그것 사용). 한 패스로 EmuLinear 들을 calibrate 모드로 돌려 StatsCollector·observer·Hessian(XᵀX) 수집 →
   fit_transform → 가중치 재양자화 (gptq 는 quantize_weight(hessian=…)) → observer freeze → frozen 모드.
5) src/tricast/eval/ppl.py — perplexity(model, tokenizer, *, dataset="wikitext2", split="test", seqlen=2048,
   max_windows=None, batch_size=1, device=None, texts=None) -> dict(ppl, nll, n_tokens, n_windows,
   dataset_fingerprint). GPTQ 규약 (§6.4): "\n\n".join(texts) 1회 토크나이즈, 겹치지 않는 seqlen 윈도우,
   토큰 평균 NLL, ppl = exp(nll). texts 인자로 데이터셋 대체 가능 (테스트용).
   src/tricast/eval/lmeval.py — @register_model("tricast") class TriCastLM(HFLM):
   __init__(self, pretrained, recipe=None, backend="auto", calibrate=True, **kwargs) → super().__init__(
   pretrained=pretrained, **kwargs) → patch_model(self.model, load_recipe(recipe)) (+ 필요 시 calibrate).
   evaluate(model, tokenizer, tasks, *, num_fewshot=None, limit=None, batch_size=8, log_samples=False) -> dict
   (lm_eval.simple_evaluate(model=HFLM(pretrained=model_obj, tokenizer=tok, batch_size=...), tasks=...)).
   lm_eval import 는 함수/클래스 정의 시점에 필요하므로 모듈 import 시 lm_eval 이 없으면 명확한 ImportError.
   src/tricast/eval/envinfo.py — capture_env(model_id=None, extra=None) -> dict: TriCast git SHA + dirty,
   python/torch/triton/transformers/lm_eval/datasets 버전, GPU 이름·드라이버·CUDA, hostname, HF 모델 commit
   SHA (huggingface_hub 로 조회, 실패 시 None), UTC 시각.
   src/tricast/eval/runner.py — run_config(cfg: dict | path) — cfg: {model, dtype, device, recipes: [...],
   tasks: {ppl: {dataset, seqlen, max_windows}, lm_eval: {tasks: [...], limit, num_fewshot, batch_size}},
   output_dir, seed} → 레시피별 결과 JSON (<output_dir>/<recipe>.json: metrics + env + recipe dict + recipe
   hash + wall time) + resume (완료 JSON 이 있으면 skip) + summary.md (markdown 표).
6) src/tricast/cli.py — `tricast` (argparse, main(argv=None)): formats [--name N], schemes, presets,
   cast --format F --rounding R [--no-saturate] VALUES..., recipe-check PATH|NAME, ppl --model M --recipe R
   [--seqlen --max-windows --dtype --device], eval --model M --recipe R --tasks a,b [--limit --num-fewshot
   --batch-size], run CONFIG.yaml. 출력은 사람이 읽기 좋은 표.
7) configs/recipes/ — 각 파일 맨 위 주석 1~2줄 (무엇을 모델링하는지 + 출처). 최소:
   bf16_passthrough (weight/activation null, mma fp32_fma), fp64_reference (null/null, mma fp64),
   hopper_fp8_w8a8, ada_fp8_w8a8, blackwell_fp8_w8a8, fp8_f7_lowacc (cofda F=7 fused),
   fp8_f7_decoupled, deepseek_fp8_block (activation fp8_group128, weight fp8_block128,
   mma deepseek_fp8_promote128), mxfp8_w_a, mxfp6_w_a, mxfp4_w_a (mma nvidia_blackwell_fp4),
   nvfp4_w_a (nvidia_blackwell_fp4), nvfp4_4o6, msfp12_bfp, int8_row_w8a8 (int_exact),
   w4a16_g128_zp_gptq (weight int4_g128_zp + gptq, activation null, mma cofda F=23 CS=32 — bf16 텐서코어
   가정은 미검증이라고 주석), fp8_ema_static (activation fp8_tensor_ema), fp8_delayed_history,
   mxfp4_rht (transform random_hadamard), nvfp4_smoothquant (transform smoothquant α=0.5).
   configs/sweeps/fp8_cofda_f_sweep.yaml (NADPE Fig.6(a) 축소: F ∈ {25,21,17,13,11,9,7,5,3} × c_mode
   {fused, decoupled}, CS 32) 와 nvfp4_gdfs_sweep.yaml (G ∈ {6,5,4,3} × F ∈ {35,25,13,10}) — runner 가 읽는
   sweep 형식 (base recipe + 축 목록 → 조합 레시피 생성) 을 runner.py 에 구현.

테스트 (로컬 CPU, reference 백엔드, 작은 랜덤 모델 — HF 모델 다운로드 금지):
 - transformers 의 Qwen3Config / LlamaConfig 로 tiny 모델 (hidden 64, layers 2, heads 4, kv heads 2,
   intermediate 128, vocab 256) 직접 생성.
 - tests/test_recipe.py: 모든 configs/recipes/*.yaml 로딩·스키마 검증, 축약형, overrides/include/exclude,
   오류 경로 메시지, 스키마 enum == spec 상수, 해시 안정성, sweep 전개.
 - tests/test_nn_patch.py: patch 후 forward 동작·shape·dtype, fp32 모델 + fp64_reference 레시피 출력이 원본과
   1e-5 이내, bf16_passthrough 동작, unpatch 복원, training 모드 backward (STE) 동작, lm_head 제외.
 - tests/test_eval.py: calibrate (짧은 texts; observer ema / gptq / smoothquant 레시피 경로), perplexity
   (tiny 모델 + texts), runner (tmp 디렉터리, resume skip), capture_env 키, lm-eval 어댑터는
   pytest.importorskip("lm_eval") 후 register 확인 + tiny 모델 객체로 evaluate 스모크 (limit 아주 작게 —
   네트워크 필요한 task 면 skip).
 - tests/test_cli.py: formats / schemes / presets / cast / recipe-check 를 main(argv) 로.
의존 레인 산출물이 없으면 해당 테스트는 importorskip 로 skip 되게 하되, 작업 막바지에 파일이 생겼으면
실제 실행해 통과시킨다.

완료 기준: 테스트 PASS (의존 모듈 존재 시), ruff 0.
