# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- README: "Claude Code 로 쓰기" — what to ask Claude for each task, the command it runs, what to check, and what it
  will not do (AGENTS.md §3 · §4).
- An image generator on the tool: Tongyi-MAI/Z-Image-Turbo (6B diffusion transformer, revision f332072,
  Apache-2.0), measured on a V100 in FP32 at 512 px (`scripts/quantizer/zimage_tune.py`: refs / table / merge / solve /
  evaluate / export; record `app/web/demo/autotune/zimage.json`). At the same size Auto-tune stays closer to the
  original images than INT3 uniform (20%, SSIM 0.588 -> 0.614, 18 of 24 prompts, Holm p 0.026) and the common rule
  (27%, 0.679 -> 0.707, p 0.049); against INT4 uniform (26%) the difference is not significant (p 0.060).
  Two between-step sizes added after those results (rule written before they were measured): at 19% and 16% the only
  existing way that fits is INT2 uniform (SSIM 0.194, noise); Auto-tune keeps 0.575 and 0.373 (24 of 24 prompts each).
  275 Linears × BF16 · INT8 · INT4 · INT3 · INT2 (RTN, exact FP32 dequantization); the table replays the unquantized
  8-forward generation; the knapsack is solved as an integer program (HiGHS) with the cost re-checked exactly;
  setups are scored by SSIM and PSNR against the original model's images on 24 prompts, the verdict by a paired
  exact sign-flip test against the pair fixed before each setup was measured (Holm within each family of sizes).
  A run fingerprint binds table rows, merge, evaluations and export. The page's detector views take a `KINDS`
  config and show the generated images (original / existing way / Auto-tune).
- Low-bit model packages (`python -m tricast.pack build|run`, `src/tricast/pack.py`): a recipe applied to a Hugging Face
  causal LM is written as integer codes bit-packed into `model.safetensors` (2/3/4/8 bits, scales, zero points; NF4
  codes + absmax; a Hadamard transform's diagonal) with `quant_config.json` (recipe text and SHA-256) and the
  checkpoint's config and tokenizer, and loads back as `PackedLinear` layers (`load` keeps the decoded weight, `codes`
  keeps only the codes on the GPU). Every layer must decode to the weight TriCast's emulation multiplies with, bit for
  bit (the sign of a zero aside), or nothing is written; activation quantization, KV specs, sparsity, outliers and other
  MMA output formats are refused. `EmuLinear.on_quantized` reports each quantization result. Qwen3-0.6B 20%: 0.51 GB,
  PPL 26.446 → 26.441; Qwen3-8B 19%: 5.11 GB, 11.224 → 11.228 (`docs/evidence/2026-11-13_model_packages.md`).
- YOLO11n ONNX files with integer weights (`scripts/quantizer/yolo_tune.py onnx`): INT8/INT4 codes with per-channel
  scales decoded by `DequantizeLinear` (opset 21, IR 10), run by onnxruntime; COCO mAP equals the PyTorch records.
- Step 5 of the tool shows the package built from the setup's own recipe file (files, checks, the low-memory run) or
  the command that builds one; the detector gets an ONNX download. `scripts/quantizer/pack_index.py` writes
  `app/web/demo/autotune/packages.json` and refuses packages whose checks or files disagree.
- The one tool `app/web/tune.html`: a single column of steps and nothing else (no menu to other pages, no
  chat, no chart) — 1 put a model in (the ones with a sensitivity table); 2 Auto-tune recommends its setup for a
  target size (the exact knapsack in a module worker, `js/mckp_worker.js`, plus the option settings the AI tune
  session measured as better, said with how it was reached); 3 edit it by hand; 4 evaluate (the recipe run on the
  server's GPU through `POST /api/quantizer/jobs` in live mode with CUDA; otherwise the recorded measurement with
  the recipe-file SHA check, as bars of the PPL increase over BF16 for every measured way at that size).
  Everything that can be chosen was measured on GPU: six target sizes; in the edit step one method on every layer
  (the uniform recipes measured within the size) and the option settings measured on the recommended assignment —
  57 setups over the six sizes; free per-layer edits wait for measurements. Sizes read as a share of the original,
  methods by name; the address keeps the job. One screen at 1440 × 900. Checks:
  `scripts/quantizer/tune_app_check.mjs` with a fake-GPU live fixture (`scripts/quantizer/tune_fixture_server.py`,
  made-up job numbers); the analysis, Quantizer and layer-map pages stay at their addresses, off this page.
- Step 5 of the tool ("결과물", after the evaluation): the model file's size under the setup by part (computed
  from the original checkpoint's safetensors header and the setup's Linear bits; embedding, output layer and
  norms stay BF16 — 1.50 GB → 0.80 GB at 20%) and the recipe bundle `POST /api/autotune/export` (zip: the
  measured recipe file, its measurement, the size, a README with the commands). `app/autotune_export.py`; the
  recipe route says the size too.
- Several models on the tool: `app/web/demo/autotune/models.json` lists them (`kind` llm | detector | image; only the
  `live` one, the model the job runner loads, gets the GPU path). `POST /api/autotune/recipe {model}` picks that
  model's bundle (unknown ids, detector entries and paths outside the folder: 404); a bundle without an AI session
  shows the per-layer assignment alone and refuses session recipes. Step 4 sets the same-size existing way (one
  method on every layer, or the common rule) beside Auto-tune: detection boxes on three COCO photos for a
  detector (tabs by arrow keys); a language model shows numbers only (no generated answers).
- YOLO11n data (`scripts/quantizer/yolo_tune.py` → `app/web/demo/autotune/yolo11n.json`): weight-only fake-quant
  of the 87 convs (per-output-channel absmax, FP16 scale, RNE; activations FP32), per-layer distortion on 128
  calibration images, the exact knapsack at the size of INT4 uniform and of the INT8-edge rule, COCO mAP50-95 on
  the 4,872 other val2017 images (CPU): Auto-tune 23.8 / 24.4 vs INT4 uniform 1.1 / INT8-edge + INT4 1.4
  (FP16 39.5).
- Qwen3-8B data (`app/web/demo/autotune/qwen3-8b/`): the per-layer KL table (252 Linear × BF16 / GPTQ 4·3·2 /
  INT8, H200, 4.3 h), knapsack solutions at 2.50 / 3.15 / 4.16 bits and, per recipe, WikiText-2 test PPL with
  greedy answers to five prompts: Auto-tune 10.92 vs GPTQ-3 uniform 11.15 at 3.15 bits, 9.93 vs GPTQ-4 9.96 at
  4.16 bits (BF16 9.72) — the drift from BF16 16% and 12% smaller. Size 16.38 GB → 5.22 GB at 3.15 bits
  (computed). `MODEL_FILES` entry and a calibration note (GPTQ on every layer at once needs ~58 GB of Hessians).
- `scripts/quantizer/tune_outputs.py`: the uniform recipe files through the page's own route, then per recipe on
  a fresh model the WikiText-2 test PPL (greedy answers kept in the records as evidence, not shown);
  `app.export_autotune_demo --outputs` takes a grid's PPL from those records when it has no KL measurement
  (records of another model, of another loaded commit, or smoke runs are refused).
- Benchmarks beyond WikiText-2 on the tool (step 4, "같은 크기, 여러 벤치마크"): `scripts/quantizer/tune_bench.py`
  records, per page recipe file (SHA-256-checked against the bundle), WikiText-2 and C4 PPL and lm-eval 0-shot
  LAMBADA / ARC-E / ARC-C / HellaSwag / PIQA / WinoGrande / MMLU with every document's result, on a model
  prepared as `tune_outputs.py` prepares it; the export sets each Auto-tune size against, per benchmark, the
  best one-method setup within the size, with the paired difference, its standard error and a 2-SE verdict.
  Documents are capped (first 1,000; MMLU 20 per subject) — the patched 8B runs ~620 tokens/s on H200.
  Qwen3-0.6B (H100, 19 setups): at 19% the only uniform that fits is 2-bit and collapses (LAMBADA 0.0 →
  15.7, accuracy mean 30.2 → 36.6, +6.3 ± 0.7); at 22% +4.7 ± 0.5; at the uniform sizes 20% / 26% PPL is
  lower but most tasks show no difference and one task each favors the uniform. Qwen3-8B (H200; WikiText-2
  PPL bit-identical to the 11-11 answer records): +1.4 ± 0.4 at 20%, +0.2 ± 0.3 at 26%. YOLO11n's step 4 shows
  COCOeval's AP75 and small / medium / large AP of the same run. Check: `scripts/quantizer/tune_models_check.mjs`.
- Qwen3-8B sizes 17% and 19% (2.75 / 3.00 bits, knapsack on the same table): the only uniform within them is
  GPTQ 2-bit (13%, PPL 915, garbled answers in the records); Auto-tune at 19% has PPL 11.22 (GPTQ-3 at 20%:
  11.15) and accuracy mean 61.8 vs 28.8 (+33.0 ± 0.7, all seven tasks); at 17% PPL 13.22, mean 47.9, but four
  of five recorded answers loop. The step-4 gain sentence shows one decimal near 100% (99.8%, not 100%).
- The 8B demo offers only the sizes where the existing way collapses and ours holds (`models.json` `sizes`:
  18% and 19%; 18% newly measured — PPL 11.94, accuracy mean 28.8 → 56.3, +27.5 ± 0.7); the data files keep
  every size. Language models no longer show generated answers (removed with their page file and export);
  only YOLO11n (boxes) and Z-Image-Turbo (images) show outputs.
- `POST /api/autotune/recipe` with a uniform grid (one method on every layer) answers with the measured uniform
  recipe `u_<method>` (SHA equals the measured file's).
- `POST /api/autotune/recipe {"transfer": id}` (with `--agent`): an option transfer's recipe rebuilt byte for byte
  as the transfer script wrote it (SHA equals the measured file's).
- Offline demo (`python -m app.server --demo --agent <loop.json> --replay DIR`, `--record DIR`): the request
  parse and the Claude round answer from answers recorded on a live run, badged on the page as a recording with
  its time; recordings of the scenario sentence, six Q&A sentences and one round. `scripts/quantizer/
  demo_rehearsal.mjs` clicks through the ten demo steps and checks badges, numbers and SHA prefixes;
  `scripts/quantizer/demo_backup.py` turns a rehearsal into a captioned backup GIF.
- AI cross-labels of the judge's 21 units by another engine (`evals/judge_labels_ai.jsonl`, user-delegated,
  kept apart from human labels; agreement 0.915 as a reference, not judge validation).
- Tune golden cases (`tests/harness/golden_tune.yaml`, `tests/test_tune_golden.py`): AC24 proposal checks and
  AC28 request checks on the demo data in the repository — a measured proposal re-accepted with its recipe
  SHA-256, one-field edits of it refused, three real refused replies refused for the same numbers, conflicts
  computed from the table and the constrained knapsack keeping every constraint.
- Tune evals (`evals/EVAL_DESIGN.md`, `evals/tune_evalset.jsonl`, `evals/tune_judge_prompt.md`,
  `evals/run_tune_evals.py`): 27 cases over tune-request parsing (AC28) and evidence-cited proposals (AC24/27),
  code graders first and an LLM judge only for clarification and rationale quality, stored and live modes,
  results and `history.csv`; the runner was delegated (`docs/prompts/delegation_evals.md`) and checked by fault
  injection. Thresholds, absolute-rule expectations and judge labels are drafts for the team to set.
- Auto-tune page (`app/web/autotune.html`): a budget slider or a request sentence, the exact knapsack solved
  in the browser (`app/web/js/mckp.js`, equal to `app.autotune solve` per budget in
  `scripts/quantizer/autotune_check.mjs`), the tune agent's evidence-cited proposals with their checks, and
  the measured results; the bundle (`app/export_autotune_demo.py`) attaches a number only to its exact recipe.
  With `python -m app.server --agent <loop.json>`, `/api/autotune/parse` reads one tune request sentence
  (`app/tune_request.py`, SPEC AC28) and `/api/autotune/propose` asks for a checked round, both live.
- Layer profiles and layer cards (`app/layer_profile.py`, `app/layer_cards.py`): per decoder Linear, the
  unquantized weight and input-activation distribution and the output error of magnitude / Wanda pruning
  (50% and 2:4), joined with every method's bits, size, single-layer KL and options; the ledger records the
  profile (`layer_stat`), and the tune loop's analysis retrieves the loss layers' documents (RAG, mask `no-rag`).
- Per-layer report errors on the Auto-tune page: the cell tooltip shows the weight MSE and layer-output SQNR /
  cosine from the `tricast report` of the method's uniform recipe, attached only when the report's recipe hash
  is that uniform KL record's (`export_autotune_demo.layer_reports`).
- Tune loop proposer `replan` (`--replan-columns`): the session table with columns measured under the best
  grid's options merged in, solved again at the session budget, one proposal with those options; the Auto-tune
  page marks such a round as a knapsack replan with no LLM call.
- Eval tools that drive the frozen runner: `evals/judge_agreement.py` (a labeling sheet with the judge's whole
  input and no verdict; agreement per criterion with its direction) and `evals/inject_faults.py` (the
  delegation §1-2 wrong answers put into a real run's stored replies, re-graded in stored mode).
- Recipe file on the Auto-tune page (`POST /api/autotune/recipe`, `app/autotune_recipe.py`): the YAML
  `app.autotune solve` writes for the shown grid, its SHA-256 held against the GPU-measured file of a recorded
  solution, and with `--agent` the session's measured proposal file; copy and download, no LLM call.
- Option transfer (`scripts/quantizer/autotune_xfer.py`, `export_autotune_demo --transfer`): knapsack solutions
  with a session's options and their cells unchanged; the Auto-tune chart draws them as "AI 옵션 적용", joined to
  the session best when the options are its own. The chart axis now fits the mixed series; collapsed uniform
  recipes sit on the top edge, their values in the tooltip and the table.
- `app.autotune table --options FILE` measures methods with options (meta.options, other recipe sha256s), `merge`
  puts one table's method columns into another of the same measurement (equal patch floors), and `solve` writes
  a table's options into the recipe: the knapsack can be solved again under options the tune agent found.
- Proposal cards on the Auto-tune page list only the options set away from their defaults, methods with the same
  changes grouped (`export_autotune_demo.changed_options`); layer cards now carry the option space of the
  Python-only low-bit methods, which `METHOD_OPTIONS.get` had skipped.
- Evidence ledger (`app/ledger.py`, SPEC AC23): every measured number of auto-tune tables, measures, solutions,
  `tricast run` and `tricast report` records becomes one record with its environment and source pointer;
  `tricast.rag.search_evidence` ranks per-layer, per-recipe and per-method documents plus the method catalog.
- Autonomous tune loop (`app/tune_loop.py`, AC27): per-layer evidence analysis, Claude Code headless (or
  auto-greedy) proposals, checks, GPU measurement through gpu_cap and ingest, round after round without a human.
- Per-layer NF4, HQQ and RaZeR (`weight_algo: nf4|hqq|razer`, Python-only Quantizer methods `nf4`,
  `hqq_int4`, `razer_nvfp4`): bit-identical on CPU to bitsandbytes NF4, HQQ 0.2.8.post1 and the pinned RaZeR
  function; refused with official GPTQ or a transform. Four-over-Six does not match fouroversix 1.0.5 (recorded).
- Paper-track Quantizer methods (`app.quantizer_recipe.LOWBIT`, Python-only): GPTQ / HQQ 3- and 2-bit
  (`weight_algo: hqq3|hqq2`), weight-only AWQ INT4 and Hadamard + NVFP4 / MXFP4, with official-code fidelity
  tests (`tests/test_lowbit_methods.py`); the page, `CALIBRATED` and `ALL_METHODS` are unchanged.
- Opt-in calibration result cache (`calibrate(..., cache_dir=...)`): per-layer states keyed by model, source,
  runtime, calibration windows and spec; bit-identical to a fresh calibration, no forward when every layer hits.
- Grids for models deeper than the page's 28 blocks (Llama-3.1-8B, Qwen3-8B): one row per decoder block in
  `quantizer_recipe.build`, auto-tune tables, `solve` and the tune loop; 28-block grids are unchanged.
- One GPU model per tune session (loop.json `gpu`, `tune_agent.pin_gpu`): KL measured on another GPU model is
  neither ranked nor accepted, since A100 and H200 differ by up to 5.5% on the same recipe.
- Lower-bit track M methods (LOWBIT): `rht_gptq_int4|3|2` (seeded random Hadamard on the input side, then
  GPTQ with the rotated inputs' Hessian: QuIP#'s incoherence without its codebook), `awq_int3` and `int8_w`
  (W8A16). Every proposer, Claude Code included, now chooses among all Python-side methods. `hqq_int3` counts
  3.7 bits (ten 3-bit codes per int32), not 3.5.
- Track C compute budget (`autotune solve --compute-budget`): Linear MACs weighted by the inverse dense B200
  Tensor Core throughput of each method's GEMM (FP4 : FP8 = INT8 : BF16 = 4 : 2 : 1, NVIDIA HGX B200 spec
  page), solved with bits by an exact two-dimensional DP (`solve_mckp2`); unrated methods need `--drop-unrated`.
  The tune loop's `compute_budget` holds every proposal, neighbour and arm to it as well.
- AWQ's weight clipping (`weight_algo: awq_clip`, the official auto_clip_layer bit for bit with its quantizer)
  and the Quantizer methods `awq_clip_int4|3` (AWQ scales, then the clip); `awq_int4|3` stay scale-only.
- One GEMM accumulation per recipe (`quantizer_recipe.build(mma=...)`, `autotune table --mma fp32_fma`): the
  table's method recipes and floors use it, solve writes it, and the tune loop refuses a grid with another.
- Information-matched controls: `--proposer evolution-table` (evolution drawing one of the parent's 50 tabled
  moves by rank) and `--proposer claude-table-select` (Claude proposes 3x, the table picks what is measured).
- Anonymous-methods ablation of the Claude proposer (`llm.mask: anonymous`, `llm.alias_seed`): method ids
  become seeded aliases in the prompt and schema, cards are withheld, replies are decoded before the checks.
- Comparison arms for the tune loop: Optuna TPE (`--proposer tpe`, study rebuilt from the session each round),
  option-only coordinate search (`option-greedy`), LLM-seeded TPE (`claude-then-tpe`), analysis masks for the
  Claude proposer (`llm.mask`) and the no-check ablation (`checks: false`, budget still enforced).
- Black-box controls for the tune loop (`--proposer evolution|random`): EvoPress-style level-switch mutations
  plus option mutations under the same checks and measurement; `--measurements N` runs a fixed budget (refused
  proposals made up, shortfall recorded); proposals are deduplicated by recipe hash.
- Agent tune tools (`app/tune_agent.py`, AC24) and their protocol (`docs/prompts/tune_agent.md`): proposals are
  refused unless they build, fit the budget, cite ledger records and state only supported numbers.
- Auto-tune v2 (`app/autotune.py`): per-layer patch floor, calibrated methods (GPTQ; AWQ and SmoothQuant on
  `o_proj`/`down_proj`), per-block KV entries, models with fewer blocks, calibrated `measure`.
- Official-code pruning (`src/tricast/official`, recipe `prune:`, AC25): Wanda, SparseGPT and magnitude from
  locuslab/wanda at a pinned commit, called unchanged through a transformers ≥ 4.48 compat shim.
- Official-code GPTQ (recipe `official_gptq:`): IST-DASLab/gptq `llama_sequential` at a pinned commit; `tricast run`
  takes `hessian_max_bytes` for GPTQ's live-Hessian budget.
- Method catalog (`docs/research/methods_catalog.yaml`): 91 quantization and sparsity methods with official
  code, license, per-layer entry point and TriCast status; pre-registered spikes for auto-tune v2 and the agent.
- Quantizer page (`app/web/quantizer.html`) and its reference generator `app/quantizer_recipe.py`: pick one of
  nine methods per Linear on a 28 × 7 grid (per-method options, KV cache, calibration) and get the same recipe
  bytes from the browser and from `python -m app.quantizer_recipe`; recorded results attach only on a hash match.
- Layer map (`app/web/layers.html`, `app/export_layers.py`): per-layer SQNR / MSE / cosine from `tricast report`.
- `POST /api/quantizer/jobs` live job API (PPL, layer report, lm-eval); GPU execution is not verified yet.
- Recorded Quantizer demo data for Qwen3-0.6B (five presets: PPL, layer reports, lm-eval 5 tasks × 300 docs).

### Changed
- Tune eval set 27 → 35 cases (2026-11-09, user-approved): English and compound constraints, compute-only
  clarification, an English out-of-scope request, allowed-method and compute-budget conflicts, an
  admin-impersonation injection and an unknown excluded method; expect key `compute_budget`. The first
  baseline of the set (b7e3eca) showed that `r_c9` expected `tune` for a compute budget the weight-only
  table cannot meet; it is now the conflict case `r_k5`. `r_q4` asks with a compute budget the table can meet
  (two reviews found 1/4 left `conflict` a fair answer too).
- Midterm-2 demo (`docs/DEMO.md`): one unbroken live chain (sentence → parse → knapsack → recipe file whose
  SHA-256 is the measured file's → AI rounds → AI best's file), the live Claude round asked early, a blanket
  disclosure of what is recorded, and the final goals confirmed with S2 and evals rows; the tune evals'
  thresholds and absolute-rule expectations approved, baseline measured three times on one commit (§9).
- README slides now show the 10-slide v7 business deck (Quantizer and Auto-tune), slide 6 animated with the
  Quantizer page recording; the v1 chip-arithmetic slide images were removed.
- Docs v2.0: PROBLEM, SPEC and the ontology are reframed around choosing a quantization method per layer;
  SPEC adds AC15–AC21, a layer-selection spike and the 2026-10-13 run record are added.
- Interview record: the ten-log record of 2026-10-01 was replaced on 2026-10-08 by the 28-log record now in
  `docs/research/interviews.md`; old logs 9 and 10 are now logs 01 and 02, so earlier entries that cite
  "log 9" or "log 10" refer to 01 and 02. The PROBLEM falsification approved on 2026-10-01 belongs to the
  ten-log record.
- Course deliverables moved to the standard paths: `docs/PROBLEM.md`, `docs/SPEC.md`,
  `docs/research/interviews.md` and `docs/ontology.yaml` (split out of `docs/ontology.md`). Golden
  cases use the SPEC v1.0 AC numbers (NADPE vectors are AC1), AGENTS.md gains stop conditions and
  ontology class names, and references to removed documents now point at existing files.
- The earlier design, results and slide documents were removed from `docs/`, and the git history was
  squashed into one commit. RAG retrieval reads `docs/SPEC.md`.
- The README now shows the 9-slide business overview as images (`assets/readme/`, slide 7
  animated), rendered from the v3 deck PDF, followed by a short Korean text section.

### Fixed
- The recipe route's `measured.gpu` / `git_sha` come from the PPL record when a row has no KL record (they were
  empty for setups measured for PPL only); the tool's "한 방법 최선" says how many uniform recipes it was picked
  from.
- `tricast.eval.lmeval.evaluate(bootstrap_iters=…)`: passed on to lm-eval when given (0 turns its standard
  errors off). lm-eval's bootstrap forks one process per CPU and failed with ENOMEM on strict-overcommit GPU
  hosts holding a model; without the argument the call is unchanged.
- A live Claude round on the Auto-tune page is numbered after every recorded round (LLM and measured), so
  it no longer shares a number with the replan round.
- A tune request whose compute budget is below the least MAC weight the constraints allow (any budget
  below 1 on a weight-only table: every A16 method runs a BF16 GEMM) is now a computed `conflict`, as a
  size budget already was; before, a `tune` parse went on to a knapsack that ignores the compute budget.
- A tune request whose budget the schema pattern lets through but no number reads ("1/0", "1.5/2") is refused
  with a RequestError (422) instead of a ZeroDivisionError / ValueError (500 on the server, a crashed eval run).
- Tune evals runner: a target with decimals ("≥ 87.5%") was read as its digits after the point (5%), a
  metric with no case to count (0/0) was reported as met, and a live run read `git status` after writing its
  kept replies, so it was always dirty; values and case verdicts of the 2026-11-08 runs are unchanged.
- A tune-loop round with no LLM reply on file (`replan`) listed no cell changes in the Auto-tune bundle; the
  changes now come from the recorded grid against its parent's (`export_autotune_demo.grid_changes`).
- C4 calibration (`calibration.dataset: c4`) loads its shard with no config name: under datasets 4.x config
  `en` carries the split sizes of all of C4 and one shard failed split verification (ExpectedMoreSplitsError,
  found on the 2026-11-04 Llama-2-7B GPU run), as the official prune path already did.
- On Python 3.9 (macOS `/usr/bin/python3`) the app exits with the required version and how to get it, instead
  of a pydantic TypeError after a successful install; the Auto-tune tooltip reads `outlier_channels` as a fraction.
- A tune round whose measurement stopped is measured again on the cluster: `measure_remote` takes back the round
  folder of the same proposals instead of refusing it as an existing round; a `remote.env` PYTHONPATH (the hqq
  install) now follows the tree's paths in the round wrapper instead of being overwritten.

### Added
- `docs/PROBLEM.md` §3-0 lists behaviour, cost and workaround evidence and the unaffected group in the interview
  log's own words; `docs/ontology.md` §4-4 shows the interview passage behind each class. `tests/test_ontology.py`
  checks that every quote and tag in both tables appears in the cited log row.
- `scripts/harness/compare_fault_injection.py` reruns the AC13/AC14 fault injection on a copy;
  `scripts/bench/count_code_lines.py` reproduces the README code-size comparison; `docs/evidence/` keeps the
  2026-10-09 full-check record.
- `tricast.eval.compare.compare_ppl` (SPEC AC13 · AC14): the PPL verdict for two evaluation records, refused
  with the offending field path unless the model revision, dataset fingerprint and token count are valid and
  equal. Golden cases in `tests/harness/golden_compare.yaml`, harness `tests/test_compare_golden.py`, delegation
  record `docs/prompts/delegation_compare.md`.
- `tests/test_ac4_passthrough.py` judges SPEC AC4 on the CPU reference backend, with a lossy recipe as the
  negative control; `make test-app` and the `app` extra run the TriCast Studio tests.
- Course records restored at the course paths: the CUDA-core emulation spike
  (`docs/spikes/cuda_core_bit_exact_emulation.md`, team-ratified 2026-10-01) and the Wave 1 delegation
  prompts with their verification loop (`docs/prompts/delegation_examples.md`, `docs/prompts/wave1/`).
- `docs/prompts/glossary_check.md`: AGENTS.md removal experiment judged by the recipe loader; it found that
  the glossary did not say `layers` is a string, and the corrected glossary passes.
- `tests/test_spec_traceability.py` checks that every SPEC AC names test, config or script files that
  exist (AC12 is the one marked pending), that golden cases and AGENTS rules cite real ACs, and that the
  SPEC traceability table points at existing files.
- `docs/ontology.yaml` gains interview synonyms and a three-tier scope (managed / external reference /
  out of scope with reasons and evidence); `tests/test_ontology.py` checks that the out-of-scope items
  match SPEC §4's excluded column.
- pytest puts the checkout's `src/` first on `sys.path` (`pythonpath`), so in-process tests import this
  tree even when another tricast is installed; the lm-eval entry-point test still needs `pip install -e .`.
- Claude Code harness: `/check` and `/golden` are user-invoked only (`disable-model-invocation`) and report
  golden and auxiliary checks separately; `git push` asks first; the guard hook protects
  `src/tricast/reference/` instead of the removed design document.
- `tests/test_ontology.py` checks that every ontology class, attribute and relation cites an
  interview log that exists and that relations only target defined classes.
- README text below the slides: install, quick start, document map and verification status.
- `docs/SPEC.md` reference list: every source was opened and checked on 2026-10-08 (NADPE MICRO'26
  artifact doi:10.5281/zenodo.21505180, OCP MX v1.0, IEEE 754-2019, DeepSeek-V3, GPTQ, AWQ, KIVI and
  others). Preset provenance now names the NADPE DOI and source path; the DeepSeek preset states that
  its F and CS come from NADPE.
- `docs/prompts/delegation_golden_harness.md`: fault-injection check of the golden harness (five
  injected faults, all detected). `test_golden_manifest` now rejects unknown case keys, and
  `test_golden_vectors_present` checks the FP8 and FP4 NADPE vectors against `manifest.json` sha256.
- TriCast Studio (`app/`), a web app that compares native and emulated outputs of one input for a
  user-designed accumulation algorithm (CoFDA, GDFS, FP32 FMA, FP64) and input format, on Qwen3-0.6B,
  Llama-3.2-1B, YOLO11n and ResNet18. Three modes: recorded examples, live runs on a server GPU
  (FastAPI job queue), and a WebGPU lab whose WGSL CoFDA GEMM matches the exact JS reference bit for
  bit on 295 golden cases (1,470 outputs) and TriCast's Triton bits on a real Qwen3-0.6B layer.
- Project logo in both READMEs.
- Development harness: `make check` (ruff + CPU pytest), Claude Code hooks that ask before Edit/Write
  changes tests, golden data, shared specs or the gate files and lint every edited Python file, and a
  GitHub Actions workflow (lint, CPU tests, wheel build) switched by the `CI_ENABLED` repository
  variable and off by default.
- `make ci` replays the three CI jobs locally. On macOS arm64 (Python 3.11, torch 2.8.0 CPU, no
  microxcaling) at fa13ba1: lint clean, 2,229 passed / 899 skipped (CUDA and oracle only), wheel and
  sdist pass `twine check`.
- Opt-in inference-only `EmuConv2d` lowering through the existing quantization and MMA engine
  (`patch_model(..., include_conv2d=True)`), including groups/depthwise, padding modes and reversible patching.
- Separate opt-in `AttentionSpec` / `patch_attention` for Transformers 4.55.2 Llama/Qwen3 eager
  QK/PV arithmetic, with explicit per-product specs and plain DynamicCache guards. Projection patching
  can coexist; TriCast KV-cache combinations and training are unsupported. H200/A100 Triton and
  V100 reference passed 47 arithmetic cases, two tiny models across five recipes with 20-token
  generation each, and 29 guards; this does not establish pretrained attention quality.
- A narrow native Hopper FP8 instruction probe with raw operand/lane-mapping controls, exact
  native/reference/Triton comparisons, preserved mismatches and PTX/SASS witnesses. The mma.sync
  path was compiler-lowered to FP16; a separate WGMMA probe passed its native witness and 140/141
  numerical cases. The direct FP32-C mismatch is retained; overall native bit parity and other GPU
  generations are not verified.
- Model-family examples and a reproducible integration validator for Linear/Conv2d, Llama, Qwen,
  ResNet and YOLO, with bit parity, accumulator propagation controls and run metadata.
- Full paired YOLO11n COCO2017 validation: 5,000 images/625 batches, finite outputs, selected Triton
  dispatch and native restoration. `fp8_f7_lowacc` bbox AP fell from 39.4194% to 19.3953%
  (−20.0241 pp); full-evaluation execution passed, not an accuracy-preservation claim.
- Full Llama-3.2-1B projection/MLP WikiText-2 and Winogrande comparisons, retaining native SDPA.
  H200 `hopper_fp8_w8a8` PPL was 9.7573 → 9.8798. A100 `bf16_passthrough` passed AC4's `1e-3`
  relative PPL gate (`4.4411e-5`); A100 `fp8_f7_lowacc` PPL rose 9.7564 → 141.2488 and Winogrande
  fell 60.2210% → 49.5659%. All full execution gates passed, not an acceptable-quality claim.
  Each comparison uses its same-GPU native baseline; F7 changes both quantization and accumulation.
- Architecture diagrams, a model coverage guide and a twenty-six-slide Korean technical overview
  covering Tensor Core motivation, numerical/model-quality decisions, individual interview excerpts,
  Python-to-GPU Triton execution, configurable MMA, model application and recorded metric/demo results.
  The Triton section separates compiler/execution from numerical rules, quotes the actual
  alignment helper and FP8 recipe, and explains specialization, Linear forward and validation scope.
  The strengths comparison ties tile abstraction, arithmetic control, JIT reuse and autotune to
  implementation evidence. Input/CoFDA equations and promotion pseudocode use editable Office Math.
  Six arithmetic walkthrough slides explain FDA, CoFDA fused/decoupled, GDFS, periodic FP32
  promotion and comparison baselines, including the existing hand-derived 16/18 example.
  The deck follows hardware comparison, algorithms and LLM outputs, interviews, then Triton
  design/features/results, with a native side-by-side CUDA Core/Tensor Core comparison table.
- Number formats (`ExMy` float with IEEE / fn / fnuz / no-special, custom bias, subnormals on/off; `intN`/`uintN`
  with fixed point; E8M0), six rounding modes including stochastic rounding with explicit noise.
- Exact reference backend (fp64 / int64) for casting, quantization and MMA accumulation.
- Quantization: tensor / row / group / 2-D block scales; absmax, power-of-two, MSE search, percentile,
  Four-over-Six; two-level NVFP4; integer (GPTQ/AWQ) and float (KIVI/HQQ) zero points; MX, NVFP4, BFP and
  FP8/INT schemes; static observers (min-max, EMA, delayed history, percentile, MSE).
- MMA accumulation models: CoFDA (C-fused / C-decoupled), GDFS, DeepSeek-style FP32 promotion, FP32 FMA
  chain, FP64, exact integer; hardware presets with provenance (Hopper, Ada, Blackwell FP8/FP4).
- Triton kernels for quantization and MMA emulation on CUDA cores, bit-identical to the reference.
- Hugging Face integration: `patch_model` with per-layer rules (`match` / `layers` / `modules`, `skip`),
  calibration (WikiText-2 / C4 / Pile), GPTQ (incl. sequential), AWQ, SmoothQuant, Hadamard rotations,
  STE for QAT, KIVI-style KV-cache quantization.
- Evaluation: WikiText-2 perplexity, lm-eval adapter (`tricast-lm-eval`, `tricast eval`), sweep runner with
  resume and environment capture, per-layer error report (`tricast report`).
- `tricast agent`: natural-language request to a validated recipe (Claude API or offline parser).
- NADPE golden vectors (1716 cases) and `scripts/nadpe_oracle/check_triton.py` for Triton-vs-NADPE checks.
- Qwen3-0.6B results (`docs/results/qwen3_0.6b.md`): WikiText-2 perplexity of 26 recipes on an A100,
  KIVI perplexity and CoQA, HellaSwag/CoQA of five recipes and the accumulator ULP of five FP8
  accumulators on a V100, each with its environment;
  `scripts/e2e/summarize.py` also tabulates lm-eval records.
- `scripts/bench/bench_decode.py`: decode-step latency of a patched model (median / mean / p99), with
  the host load, other processes on the GPU and the source commit recorded.
- Weight structure in recipes: `sparsity` (N:M or unstructured magnitude pruning) and `outliers` (the
  largest weights kept in a higher-precision format and added back through an fp32 path), with the
  bundled recipes `fp8_2of4_sparse` and `nvfp4_outliers` (team interviews, log 10).
- Accumulator ULP error: `tricast.analysis.ulp_distance` / `ulp_error`, the per-layer `mma_ulp` of
  `tricast report` (against fp64 accumulation of the same operands) and ULP columns in the demo
  (team interviews, log 9).
- Course deliverables grounded in the team's ten interviews: `docs/research/interviews.md`,
  `docs/ontology.yaml`, `docs/PROBLEM.md`, `docs/SPEC.md` (AC8–AC10).

### Changed
- Korean technical overview, 30 slides: agenda, validation dashboard, closing summary; annotated recipe
  and API code slides; Llama/YOLO results relabeled as `fp8_f7_lowacc` (F=7, C-fused); interview roles
  restored to the recorded logs; Qwen PPL split into quantization and accumulator shares.
- Synchronize the latest problem/specification review without relabeling new AC/V-plan contracts as
  verified; align the demo guide with its implemented calibration and retained execution record.
- Minimum Triton version is 3.4; 3.0 cannot compile the current constexpr tuple-based kernels.
- Recipe loading supports Python 3.10's location of the `Traversable` protocol.
- MMA emulation kernel: integer fast path for finite operands (int32/int64 aligned sums, int32
  operand decode, runtime pass loops, no register spills). On an A100 the Hopper FP8 accumulator
  went from 0.011 to 0.15 TMAC/s (NADPE CUDA kernel: 0.15) and a Qwen3-0.6B 2048-token window
  from 83 s to 6 s; results stay bit-identical.
- MMA emulation kernel for small M (decoding, M ≤ 16; CoFDA, GDFS, exact integer): each chunk,
  group or K span is loaded as one tile and reduced in registers, and for CoFDA and GDFS the
  Inf/NaN flags are reduced in the kernel instead of a host finiteness check (a device sync per
  linear). A Qwen3-0.6B Hopper-FP8 decode step went from 375–401 to 168–170 ms on a V100 and from
  292–294 to 101 ms on an A100 (`scripts/bench/bench_decode.py`); results and generated tokens stay
  identical.
- Recipe overrides replace a `sparsity`, `format` or `dequant_format` mapping instead of merging
  it field by field: a partial merge could combine two formats' fields into a third format.
- Interview log (`docs/research/interviews.md`): the respondents of logs 9 and 10 appear by role
  instead of by name.
- `README.md` is now the Korean README; the English one moved to `README.en.md`. Both gained a
  command table, the three-minute demo outline, the repository layout and the verification of the CPU
  suite; the Korean one also lists the course deliverables.
- Course deliverables approved by the team on 2026-10-01: AC1–AC10, the problem statement and its
  falsification, the absolute rules and definition of done, the golden cases, the evaluation set and
  judge rubric, and the spike's success conditions; ontology entries without interview or
  observation evidence were removed.
- Six CPU tests compared native (unemulated) results bit for bit and failed on Linux, where CPU
  BLAS and SDPA round differently by shape. They now allow that rounding only where native ops are
  involved (perplexity across batch sizes: relative 1e-12; STE gradients: relative 1e-6; SDPA
  single-chunk cache logits: 1e-6), and the KV-only report test uses a model whose logits the 2-bit
  cache measurably changes (KL 2.7e-8; it was rounding noise before).

### Fixed
- `app/tests/test_webgpu_golden.py` compares the `fp64` field's NaNs by class, as `webgpu_check.html`
  does. On x86_64 Linux (geneva) the `nan` case rebuilt `fff00000` where the macOS-recorded golden has
  `7ff00000` and failed at 3820b5b; with the change all 161 app tests pass there. `expected` stays bitwise.
- `tests/test_cli.py` passes `--device cpu`: its fake model loader ignores the device, so three CLI
  tests failed on any machine with a visible GPU (reproduced at 3820b5b on an A100).
- Named-task dataset fingerprint capture for lm-eval 0.4.13's `TaskManager.load` API. Five CPU
  provenance checks and the full UCL CPU suite passed (2,337 tests, one CUDA-only skip). Numerical
  and model execution are unchanged. Existing quality records retain their original source and
  empty fingerprints; a separately labeled after-run cache audit is not historical in-memory capture.
- Reference arithmetic on CUDA tensors: powers of two are built from their fp64 bit pattern instead of
  `torch.ldexp`, whose `pow(2, e)` is inexact on CUDA. Before, a few casts differed from the CPU
  (A100: 1 of ~500k bf16 values, up to 7 tf32) and decoding, hence every reference GEMM, raised on
  CUDA. The Triton path runs this code on the GPU for MSE/percentile scale search, zero points and
  dequantized operands, and GPTQ and the AWQ search call it directly, so the Qwen3-0.6B rows of those
  recipes were rerun: WikiText-2 PPL `nvfp4_4o6` 25.5801 → 25.5673, `w4a16_gptq_sequential`
  24.4976 → 24.0839, KIVI-2 23.0081 → 22.9740 (8 windows streamed through the cache: 19.6035 →
  19.5115), and the demo's 16-window `nvfp4_4o6` 24.3761 → 24.2218; the other rerun rows moved by at
  most 0.15% or within lm-eval's standard error, and `nvfp4_awq_shared` did not change. A reference
  cast on CUDA takes about 30% longer per call (idle V100).
