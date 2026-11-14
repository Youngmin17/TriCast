// TriCast (tune.html): one tool, one flow, top to bottom.
//   1 put a model in (one with a measured sensitivity table);
//   2 Auto-tune recommends its setup for a target size: the exact knapsack in the browser (js/mckp.js, in a worker)
//     picks a method per layer, and the option settings the AI tune session measured as better go on top;
//   3 edit the recommendation by hand;  4 evaluate the recipe end to end.
// Everything that can be chosen was measured on GPU, so every number on the screen is a measurement: the six target
// sizes; in the edit step the assignment (the recommended per-layer one, or one method on every layer — the uniform
// recipes measured within that size) and the option settings measured for the recommended assignment. Free
// per-layer edits are not offered yet: their setups have no measurement.
// The evaluation shows the GPU measurement recorded for the setup on screen, with whether the recipe file is the
// very file measured (SHA-256); on a GPU server (live mode with CUDA) it runs the recipe there instead.
// Sizes are said as a share of the original (16 bits = 100%) with the bits beside; methods by a readable name
// (methodLabel), their id in the tooltip.
// The models come from demo/autotune/models.json: a language model has a bundle (and may have `benchmarks`, its
// benchmark file, and `sizes`, the sizes offered), a detector (YOLO11n) its own file of per-layer bit-width setups
// with COCO mAP and the boxes they drew, an image generator (Z-Image-Turbo) the same with its images and their SSIM to
// the unquantized model's. Only a model marked "live" runs on a GPU server.
import {parseBits, solve, summary} from "./mckp.js";
import {h, s, withTip} from "./ui.js";

const KIND = {q_proj: "q", k_proj: "k", v_proj: "v", o_proj: "o", gate_proj: "gate", up_proj: "up", down_proj: "down"};
const LMEVAL = [["arc_easy", "ARC-Easy"], ["arc_challenge", "ARC-Challenge"], ["piqa", "PIQA"], ["winogrande", "WinoGrande"],
  ["hellaswag", "HellaSwag"]];
const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 2) => (value == null ? "—" : Number(value).toFixed(digits));
const signed = (value, digits = 2) => (value == null ? "—" : `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`);
const bitsOf = (text) => {
  const [p, q] = parseBits(text);
  return Number(p) / Number(q);
};
const pct = (bits) => `${fmt(100 * bits / 16, 0)}%`;
const pplShort = (v) => (v >= 100 ? Math.round(v).toLocaleString("en-US") : fmt(v, 1));
const pplText = (v) => (v >= 1000 ? Math.round(v).toLocaleString("en-US") : fmt(v));
// a share in %: whole numbers, but one decimal near 100 so 99.8% does not read as 100%
const pctText = (v) => fmt(v, v >= 99.5 && v < 100 ? 1 : 0);
// A row of tabs: the arrow keys, Home and End move the pick, and the picked tab keeps the focus through the redraw
// the pick causes (only the picked tab is in the Tab order). items: [{content, attrs}]; pick(k) redraws.
function tabRow(id, label, rowClass, tabClass, items, at, pick) {
  const row = h("div", {class: rowClass, role: "tablist", "aria-label": label, id});
  const select = (k) => {
    pick(k);
    document.querySelector(`#${id} [role="tab"][aria-selected="true"]`)?.focus();
  };
  items.forEach((item, k) => {
    const tab = h("button", {type: "button", role: "tab", class: tabClass, "aria-selected": String(k === at),
      tabindex: k === at ? "0" : "-1", ...item.attrs}, item.content);
    tab.addEventListener("click", () => select(k));
    tab.addEventListener("keydown", (event) => {
      const next = {ArrowRight: k + 1, ArrowLeft: k - 1, Home: 0, End: items.length - 1}[event.key];
      if (next == null) return;
      event.preventDefault();
      select((next + items.length) % items.length);
    });
    row.append(tab);
  });
  return row;
}

// a method id the way an engineer says it: "rht_gptq_int3" → "회전+GPTQ 3비트". Built from the id's parts
// (app/quantizer_recipe.py: rht = QuIP#'s random-Hadamard incoherence processing before GPTQ; awq_clip = AWQ with the
// official auto_clip; int8_w = per-row INT8 weights with BF16 activations); an id it cannot read is shown as it is.
const ALGO = {gptq: "GPTQ", awq: "AWQ", awq_clip: "AWQ+클립", hqq: "HQQ", rht_gptq: "회전+GPTQ"};
function methodLabel(id) {
  if (id === "bf16") return "BF16 (원본 그대로)";
  if (id === "nf4") return "NF4 4비트";
  if (id === "int8_w") return "INT8 8비트";
  const parts = /^(.+)_int(\d+)$/.exec(id ?? "");
  return parts && ALGO[parts[1]] ? `${ALGO[parts[1]]} ${parts[2]}비트` : id;
}
// uniform: the edit "one method on every layer" (a method id), or null = the recommended per-layer assignment.
// option: the id of the AI-measured option settings on the recommended assignment, or null = the default settings.
// exporting: the step-5 bundle download in flight; exportError: why it failed.
// models: the recorded model list; entry: the loaded one; benchmarks: its benchmark file (language models,
// scripts/quantizer/tune_bench.py); det: a detector's file; pick: the detector setup picked by hand (null =
// Auto-tune's); image: the photo shown in 4.
const st = {health: null, bundle: null, model: null, budget: null, plan: null, planError: null, computing: false, planTicket: 0,
  uniform: null, option: null, recipe: null, recipeError: null, recipeSeq: 0, lm: new Set(), limit: 200, evaluation: null,
  exporting: false, exportError: null, models: [], entry: null, benchmarks: null, det: null, pick: null, image: 0,
  packages: null,
  loading: false, loadError: null};
const isDet = () => ["detector", "image"].includes(st.entry?.kind);  // a file of measured setups, not a bundle
const isGen = () => st.entry?.kind === "image";
const serverGpu = () => st.health?.mode === "live" && st.health?.device === "cuda";
// the live job runner applies recipes to the one model it loads (app/runners/quantizer.MODEL_ID): the entry says so
const gpuLive = () => serverGpu() && st.entry?.live === true;

async function postJSON(path, payload) {
  const response = await fetch(path, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload)});
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.error?.message ?? `${path}: HTTP ${response.status}`);
  return body;
}

const chip = (text, kind = "plain", title = null) => h("span", {class: "ta-chip", "data-kind": kind, title}, text);
// a "?" that explains a term on hover or focus
const help = (term, text) => withTip(h("button", {type: "button", class: "ta-help", "aria-label": `${term} 설명`}, "?"),
  () => h("div", {class: "ta-help-tip"}, h("b", null, term), h("p", null, text)));

// ---------- the table and what was measured on it ----------
const table = () => st.bundle.table;
const elementBits = (method) => (method === "bf16" ? 16 : table().weight_bits[method]?.[0] ?? 16);
const sameGrid = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const uniformGrid = (method) => Array.from({length: table().blocks}, () => table().kinds.map(() => method));

// the knapsack setups GPU-measured without constraints, smallest first: {budget, bits, row}. A model-list entry's
// `sizes` (solution ids) narrows the offer to those — the bundle keeps every measurement either way
function measuredBudgets() {
  const shown = st.entry?.sizes ? new Set(st.entry.sizes) : null;
  const rows = st.bundle.solutions.filter((row) => row.ppl != null && (row.kv ?? "none") === "none" && (!shown || shown.has(row.id)));
  return [...new Map(rows.map((row) => [row.budget_bits, row])).values()].sort((a, b) => a.bits - b.bits)
    .map((row) => ({budget: row.budget_bits, bits: row.bits, row}));
}

// the uniform recipes (one method on every layer) measured within `budget`, best PPL first
function uniformsWithin(budget) {
  const limit = bitsOf(budget) + 1e-9;  // the budget, not the plan's bits: a uniform at the budget is "within the size"
  return st.bundle.baselines.filter((b) => b.ppl != null && b.bits <= limit).sort((a, b) => a.ppl - b.ppl);
}

// What was measured for an assignment besides its default settings: the option settings the AI tune session measured
// on it with a test PPL — its own proposals where the session ran, its best options transferred to the other sizes.
function optionsFor(base) {
  if (!base) return [];
  const {session, transfer} = st.bundle;
  if (session && base.id === session.start) {
    return session.rounds.flatMap((round) => round.proposals ?? [])
      .filter((p) => p.status === "accepted" && p.ppl != null && sameGrid(p.grid, base.grid))
      .map((p) => ({kind: "proposal", id: p.name, ppl: p.ppl, kl: p.kl, options: p.options_changed ?? {},
        record: session.records?.[p.ppl_id] ?? base.ppl_record}));
  }
  const row = (transfer ?? []).find((item) => item.base === base.id);
  return row && row.ppl != null && sameGrid(row.grid, base.grid)
    ? [{kind: "transfer", id: row.id, ppl: row.ppl, kl: row.kl, options: row.options_changed ?? {}, record: row.ppl_record}] : [];
}

// the settings Auto-tune recommends for an assignment: the measured option settings with the lowest PPL, or null
// when the default settings measured best
function recommendedOption(base) {
  const best = optionsFor(base).sort((a, b) => a.ppl - b.ppl)[0];
  return best && best.ppl < base.ppl ? best : null;
}

// option settings in words: "act_order = true, damp = 0.05" and the methods that got them
function optionText(option) {
  const texts = new Set(Object.values(option.options).map((changes) =>
    Object.entries(changes).map(([k, v]) => `${k.replace(/^weight_algo\./, "")} = ${v}`).join(", ")));
  return texts.size === 1 ? [...texts][0] : `설정 ${texts.size}가지`;
}
function optionScope(option) {
  const methods = Object.keys(option.options);
  return methods.every((m) => m.includes("gptq")) ? `GPTQ 계열 ${methods.length}종` : `${methods.length}개 방법`;
}

// the setup on screen: the recommended assignment with default or AI-measured settings, or a hand edit to one
// method on every layer — each with its recorded measurement
function current() {
  const p = st.plan;
  if (!p) return null;
  if (st.uniform) {
    const b = st.bundle.baselines.find((row) => row.method === st.uniform);
    return {kind: "uniform", id: b.id, method: b.method, grid: uniformGrid(b.method), bits: b.bits, ppl: b.ppl, kl: b.kl, record: b.ppl_record};
  }
  const base = p.recorded;
  const option = optionsFor(base).find((o) => o.id === st.option) ?? null;
  return option ? {kind: option.kind, id: option.id, option, grid: p.grid, bits: p.bits, ppl: option.ppl, kl: option.kl, record: option.record}
    : {kind: "auto", id: base?.id ?? null, grid: p.grid, bits: p.bits, ppl: base?.ppl ?? null, kl: base?.kl ?? null, record: base?.ppl_record ?? null};
}
const edited = () => Boolean(st.uniform) || st.option !== (recommendedOption(st.plan?.recorded)?.id ?? null);

// ---------- the knapsack off the main thread ----------
// One module worker (js/mckp_worker.js) solves. A request from the user goes before any warm-up and replaces a user
// request not started yet (the last pick wins); results are kept per budget, so a budget solved once — or warmed up
// in the background — answers at once. Without workers it solves here.
// gen: the loaded model's generation — a job of an earlier model resolves as superseded and caches nothing
const solver = {worker: undefined, busy: false, current: null, user: null, warm: [], seq: 0, cache: new Map(), jobs: new Map(), gen: 0};

function resetSolver() {
  solver.gen += 1;
  const stale = [solver.user, ...solver.warm].filter(Boolean);
  Object.assign(solver, {user: null, warm: [], cache: new Map(), jobs: new Map()});
  for (const job of stale) job.done({superseded: true});
  pump();
}

function solverWorker() {
  if (solver.worker === undefined) {
    try {
      solver.worker = new Worker(new URL("./mckp_worker.js", import.meta.url), {type: "module"});
      solver.worker.addEventListener("message", ({data}) => finish(data));
      solver.worker.addEventListener("error", (event) => {  // a worker that cannot start: solve here from now on
        event.preventDefault();
        solver.worker = null;
        if (solver.current) finish(solveHere(solver.current));
      });
    } catch {
      solver.worker = null;
    }
  }
  return solver.worker;
}

function solveHere(job) {
  const started = performance.now();
  try {
    return {id: job.id, solution: solve(job.table, parseBits(job.budget)), seconds: (performance.now() - started) / 1000, main: true};
  } catch (error) {
    return {id: job.id, error: error.message};
  }
}

function pump() {
  document.body.dataset.solver = solver.busy || solver.user || solver.warm.length ? "busy" : "idle";
  if (solver.busy) return;
  const job = solver.user ?? solver.warm.shift();
  if (!job) return;
  if (job === solver.user) solver.user = null;
  solver.busy = true;
  solver.current = job;
  document.body.dataset.solver = "busy";
  const worker = solverWorker();
  if (worker) worker.postMessage({id: job.id, table: job.table, budget: job.budget});
  else setTimeout(() => finish(solveHere(job)), 0);
}

function finish(data) {
  const job = solver.current;
  if (!job || job.id !== data.id) return;
  solver.busy = false;
  solver.current = null;
  job.done(data);
  pump();
}

// the plan for `budget`: {grid, bits, counts, seconds, recorded, budget} or {error}, or {superseded: true} when a
// later user request replaced it before it started
function planFor(budget, warm = false) {
  const key = budget;
  if (solver.cache.has(key)) return Promise.resolve(solver.cache.get(key));
  const queued = solver.jobs.get(key);
  if (queued) {
    const at = solver.warm.indexOf(queued.job);
    if (!warm && at >= 0) {  // the user wants a warm-up that has not started: it goes first now
      solver.warm.splice(at, 1);
      solver.user?.done({superseded: true});
      solver.user = queued.job;
      pump();
    }
    return queued.promise;
  }
  const t = table();
  const gen = solver.gen;
  let job;
  const promise = new Promise((resolve) => {
    job = {id: ++solver.seq, table: t, budget, done: (data) => {
      if (gen !== solver.gen) return resolve({superseded: true});
      solver.jobs.delete(key);
      if (data.superseded) return resolve(data);
      let result;
      if (data.error) result = {error: data.error, budget};
      else {
        const grid = Array.from({length: t.blocks}, () => t.kinds.map(() => null));
        t.layers.forEach((layer, i) => {
          grid[layer.block][t.kinds.indexOf(layer.kind)] = t.methods[data.solution.choice[i]];
        });
        const {bits, counts} = summary(t, data.solution);
        const recorded = st.bundle.solutions.find((s) => sameGrid(s.grid, grid)) ?? null;
        result = {grid, bits, counts, seconds: data.seconds, recorded, budget, warm, thread: data.main ? "main" : "worker"};
      }
      solver.cache.set(key, result);
      resolve(result);
    }};
  });
  solver.jobs.set(key, {job, promise});
  if (warm) solver.warm.push(job);
  else {
    solver.user?.done({superseded: true});
    solver.user = job;
  }
  pump();
  return promise;
}

// the measured budgets, solved in the background while the page is idle, so picking one is instant
function warmUp() {
  for (const {budget} of measuredBudgets()) planFor(budget, true);
}

// ---------- the recipe file ----------
async function refreshRecipe() {
  const seq = ++st.recipeSeq;
  st.recipe = null;
  st.recipeError = null;
  const now = current();
  if (!now) return;
  try {
    const recipe = await postJSON("api/autotune/recipe", recipeBody(now));
    if (seq === st.recipeSeq) st.recipe = recipe;
  } catch (error) {
    if (seq === st.recipeSeq) st.recipeError = error.message;
  }
  if (seq === st.recipeSeq) {
    drawEdit();
    drawEval();
    drawOut();
  }
}

// what the server needs to build the recipe of a setup (the recipe and the export routes take the same body)
const recipeBody = (now) => ({model: st.model, ...(now.kind === "uniform" ? {grid: now.grid} : now.option ? {[now.kind]: now.id}
  : {grid: now.grid, budget_bits: st.plan.budget})});

// the recipe bundle of the setup on screen: a zip the server builds (app/autotune_export.py)
async function exportBundle() {
  const now = current();
  if (!now || st.exporting) return;
  Object.assign(st, {exporting: true, exportError: null});
  drawOut();
  try {
    // on a GPU server the evaluation this page just ran on this recipe goes into the bundle too
    const e = st.evaluation;
    const job = e?.kind === "job" && e.status === "done" ? {job: e.id} : {};
    const response = await fetch("api/autotune/export", {method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({...recipeBody(now), ...job})});
    if (!response.ok) throw new Error((await response.json().catch(() => null))?.error?.message ?? `HTTP ${response.status}`);
    const name = /filename="([^"]+)"/.exec(response.headers.get("content-disposition") ?? "")?.[1] ?? "tricast_recipe.zip";
    const a = h("a", {href: URL.createObjectURL(await response.blob()), download: name});
    document.body.append(a);
    a.click();
    a.remove();
  } catch (error) {
    st.exportError = error.message;
  }
  st.exporting = false;
  drawOut();
}

function download(recipe, type = "text/yaml") {
  const a = h("a", {href: URL.createObjectURL(new Blob([recipe.yaml], {type})), download: recipe.file});
  document.body.append(a);
  a.click();
  a.remove();
}

// ---------- actions ----------
const getJSON = (path) => fetch(path).then((r) => {
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
});

// load a recorded model (its bundle or detector file, and its benchmark file): everything chosen before is cleared
async function loadModel(id = $("model").value) {
  const entry = st.models.find((m) => m.id === id);
  if (!entry || st.loading) return false;
  st.loading = true;
  st.loadError = null;
  drawModel();
  try {
    const dir = "demo/autotune/";
    const measured = ["detector", "image"].includes(entry.kind);
    const [data, benchmarks] = await Promise.all([getJSON(dir + (measured ? entry.data : entry.bundle)),
      entry.benchmarks ? getJSON(dir + entry.benchmarks) : null]);
    st.planTicket += 1;
    resetSolver();
    Object.assign(st, {entry, model: entry.id, bundle: measured ? null : data, det: measured ? data : null,
      benchmarks, budget: null, plan: null, planError: null, computing: false, uniform: null, option: null, pick: null, recipe: null,
      recipeError: null, evaluation: null, exportError: null, image: 0});
  } catch (error) {
    st.loadError = error.message;
  }
  st.loading = false;
  drawAll();
  saveState();
  if (st.model === entry.id && !isDet()) warmUp();
  return st.model === entry.id;
}

// Auto-tune for the target size: the knapsack's assignment, with the AI-measured settings on top where they measured
// better; false when a later pick took over meanwhile
async function recommend(budget) {
  const ticket = ++st.planTicket;
  Object.assign(st, {budget, plan: null, planError: null, uniform: null, option: null, recipe: null, recipeError: null,
    evaluation: null, computing: true});
  drawAll();
  const result = await planFor(budget);
  if (ticket !== st.planTicket || result.superseded) return false;
  st.computing = false;
  if (result.error) st.planError = result.error;
  else {
    st.plan = result;
    st.option = recommendedOption(result.recorded)?.id ?? null;
  }
  refreshRecipe().then(showNext);
  drawAll();
  saveState();
  return true;
}

// a hand edit of the setup on screen: {uniform} and / or {option}
function edit(change) {
  Object.assign(st, change, {evaluation: null});
  refreshRecipe().then(showNext);
  drawAll();
  saveState();
}

// bring `el` into view only as far as needed: on a screen that already shows it, nothing moves
const reveal = (el) => el?.scrollIntoView({block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
// once the setup and its recipe file are on screen, the next thing to press is the evaluation's title row: on a short
// screen it comes into view; where the steps are stacked taller than the screen (a phone) the page is left alone
const showNext = () => matchMedia("(min-width: 761px)").matches && st.plan && reveal(document.querySelector("#sec-eval .ta-sec-head"));

// the job in the address (model, target size, a hand edit), so a reload or a shared link returns to it
function saveState() {
  const params = new URLSearchParams();
  if (st.model) params.set("model", st.model);
  if (st.model && st.budget) params.set("size", st.budget);
  if (isDet()) {
    if (st.plan && st.pick) params.set("pick", st.pick);
  } else if (st.plan && st.uniform) params.set("uniform", st.uniform);
  else if (st.plan && edited()) params.set("option", st.option ?? "default");
  const query = params.toString();
  window.history.replaceState(null, "", `${location.pathname}${query ? `#${query}` : ""}`);
}

async function restoreState() {
  const params = new URLSearchParams(location.hash.slice(1));
  const id = params.get("model");
  if (!st.models.some((m) => m.id === id)) return;
  $("model").value = id;
  if (!(await loadModel(id))) return;
  const size = params.get("size");
  if (isDet()) {
    if (!detSizes().some((row) => row.id === size)) return;
    detRecommend(size);
    const pick = params.get("pick");
    if (detWithin().some((row) => row.id === pick)) detEdit(pick);
    return;
  }
  if (!measuredBudgets().some((mb) => mb.budget === size) || !(await recommend(size))) return;
  const uniform = params.get("uniform");
  const option = params.get("option");
  if (uniformsWithin(size).some((b) => b.method === uniform)) edit({uniform});
  else if (option === "default") edit({option: null});
  else if (optionsFor(st.plan?.recorded).some((o) => o.id === option)) edit({option});
}

async function runEval() {
  const now = current();
  if (!now) return;
  if (gpuLive()) {
    if (!st.recipe) return;
    // the evaluation this press started: a reply that comes after the setup or the model changed is dropped
    const evaluation = {kind: "job", status: "queued", started: Date.now()};
    st.evaluation = evaluation;
    drawEval();
    try {
      const tasks = ["ppl", ...st.lm];
      const job = await postJSON("api/quantizer/jobs", {name: st.recipe.name, yaml: st.recipe.yaml, tasks,
        limit: st.lm.size ? st.limit : null});
      if (st.evaluation !== evaluation) return;
      st.evaluation = {...evaluation, id: job.id, status: job.status};
      poll(job.id);
    } catch (error) {
      if (st.evaluation !== evaluation) return;
      st.evaluation = {kind: "job", status: "error", error: error.message};
    }
    drawEval();
    reveal($("sec-eval"));
    return;
  }
  st.evaluation = {kind: now.ppl != null ? "record" : "none"};
  drawEval();
  drawOut();
  reveal($("sec-eval"));
}

async function poll(id) {
  try {
    const response = await fetch(`api/quantizer/jobs/${id}`, {headers: {accept: "application/json"}});
    const record = await response.json();
    if (!response.ok) throw new Error(record.error?.message ?? `HTTP ${response.status}`);
    if (st.evaluation?.id !== id) return;
    st.evaluation = {...st.evaluation, status: record.status, record};
    drawEval();
    drawOut();
    if (!["done", "error"].includes(record.status)) setTimeout(() => poll(id), 1500);
  } catch (error) {
    if (st.evaluation?.id === id) st.evaluation = {...st.evaluation, status: "error", error: error.message};
    drawEval();
  }
}

// ---------- drawing ----------
const gpuName = (record) => (record?.gpu ?? table().meta.gpu_names?.[0] ?? "A100").replace("NVIDIA ", "");
const blockHead = (title, ...rest) => h("div", {class: "ta-block-head"}, h("b", null, title), rest);
const preset = (attrs, on, title, value, note, run) => {
  const button = h("button", {type: "button", class: "ta-preset", "aria-pressed": String(on), title, ...attrs},
    h("b", null, value), h("small", null, note));
  button.addEventListener("click", run);
  return button;
};

function drawServer() {
  $("server").textContent = gpuLive() ? "GPU 서버 (CUDA) — 평가를 실제로 돌립니다"
    : serverGpu() ? "GPU 서버 — 이 모델의 평가는 측정 기록으로" : "이 PC 에는 GPU 가 없습니다 — 평가는 측정 기록으로";
}

// 1 model: one row — the pick and the load button, then what was loaded
function drawModel() {
  $("model-state").replaceChildren(st.loading ? chip("불러오는 중", "live") : st.model ? chip("준비됨", "ok") : "");
  document.body.dataset.model = st.loading ? "" : st.model ?? "";  // the loaded model, for the browser checks
  $("load").classList.toggle("primary", !st.model);
  $("load").disabled = st.loading;
  const fact = (k, v) => h("div", null, h("dt", null, k), h("dd", null, v));
  if (st.loadError) {
    $("model-info").replaceChildren(h("p", {class: "ta-err"}, `측정 기록을 읽지 못했습니다: ${st.loadError}`));
  } else if (st.model && isDet()) {
    const d = st.det;
    const k = KD();
    $("model-info").replaceChildren(h("dl", {class: "ta-facts", title: `${d.model.label} · ${d.model.revision ? `revision ${d.model.revision}` : `체크포인트 SHA-256 ${d.model.checkpoint_sha256}`} · ${k.quant}`},
      fact("모델", k.what),
      fact(k.unit, `${d.layers.length}개`),
      fact("방법", `${d.methods.length + (isGen() ? 0 : 1)}종`),
      fact("평가", k.evalText(d))));
  } else if (st.model) {
    const meta = table().meta;
    $("model-info").replaceChildren(h("dl", {class: "ta-facts", title: `revision ${meta.revision} · 민감도 표: ${gpuName()} · ${meta.source?.dataset ?? ""} ${meta.windows}창`},
      fact("revision", meta.revision.slice(0, 7)),
      fact("Linear", `${table().layers.length}개`),
      fact("방법", `${table().methods.length}종`),
      fact("민감도 표", gpuName().split("-")[0])));
  } else {
    $("model-info").replaceChildren(h("p", {class: "ta-muted"}, "층별 민감도를 재 둔 모델만 고를 수 있습니다."));
  }
  const now = st.model ? (isDet() ? detCurrent() : current()) : null;
  document.title = `${st.model ? `${(st.entry?.label ?? st.model).split("/").pop()}${now ? ` ${pct(now.bits)}` : ""} · ` : ""}TriCast`;
}

// 2 Auto-tune: the target size is the one thing asked; the answer is the recommended setup and how it was reached
function drawTune() {
  if (isDet()) return drawTuneDet();
  const on = Boolean(st.model);
  $("sec-tune").toggleAttribute("data-off", !on);
  if (!on) {
    $("tune").replaceChildren();
    $("tune-state").replaceChildren(h("span", {class: "ta-muted"}, "모델을 불러오면 목표 크기를 고릅니다"));
    return;
  }
  const p = st.plan;
  const base = p?.recorded ?? null;
  const best = base ? recommendedOption(base) : null;
  $("tune-state").replaceChildren(...(base ? [chip(`추천 · 크기 ${pct(p.bits)} · PPL ${fmt(best?.ppl ?? base.ppl)}`, "ok")] : []));
  const sizes = measuredBudgets().map((mb) => {
    const ppl = recommendedOption(mb.row)?.ppl ?? mb.row.ppl;
    return preset({"data-budget": mb.budget, "aria-label": `원본의 ${pct(mb.bits)}, ${fmt(mb.bits, 2)}비트, 추천 셋업의 측정 PPL ${pplShort(ppl)}`},
      mb.budget === st.budget, `평균 ${fmt(mb.bits, 3)} 비트 (예산 ${mb.budget}) · 추천 셋업의 GPU 측정 PPL ${fmt(ppl)}`,
      pct(mb.bits), [h("span", {class: "k"}, "PPL"), ` ${pplShort(ppl)}`], () => mb.budget !== st.budget && recommend(mb.budget));
  });
  const parts = [h("div", {class: "ta-block"},
    blockHead("목표 크기", help("크기", `Linear 층(${table().layers.length}개) 가중치 하나를 저장하는 평균 비트 수로 정한 크기입니다. 원본(BF16)은 16비트 = 100%, 4비트면 25% 입니다. 임베딩 · 출력층은 BF16 그대로라 파일 전체 크기는 ⑤ 에서 따로 보여 줍니다.`),
      h("span", null, "원본 대비 · 아래 숫자는 그 크기 추천 셋업의 GPU 측정 PPL")),
    h("div", {class: "ta-presets", role: "group", "aria-label": `목표 크기 — GPU 로 측정된 ${sizes.length}개`}, sizes))];
  if (!st.budget) parts.push(h("p", {class: "ta-muted"}, st.bundle.session ? "목표 크기를 고르면 그 안에서 층마다 방법을 고르고, AI 가 GPU 로 재서 찾은 세부 설정을 더해 추천합니다."
    : "목표 크기를 고르면 그 안에서 모델 전체 오차가 가장 작도록 층마다 방법을 골라 추천합니다."));
  if (st.computing) parts.push(h("p", {class: "ta-muted", id: "computing"}, `층 ${table().layers.length}개의 배정을 정확히 계산하는 중… (몇 초)`));
  if (st.planError) parts.push(h("p", {class: "ta-err"}, st.planError));
  if (p && !base) parts.push(h("p", {class: "ta-warn"}, "이 배정은 GPU 측정 기록의 배정과 다릅니다 (번들의 표나 배낭이 바뀌었을 때). 번들을 다시 만드세요."));
  if (base) parts.push(recommendation(p, base, best));
  $("tune").replaceChildren(...parts);
}

// what Auto-tune did, in the two things it does: 1 a method per layer, 2 the AI-measured settings where they help
function recommendation(p, base, best) {
  const native = st.bundle.native.ppl;
  const uniform = uniformsWithin(p.budget)[0] ?? null;
  const ppl = best?.ppl ?? base.ppl;
  const other = best ? null : optionsFor(base).sort((a, b) => a.ppl - b.ppl)[0] ?? null;  // measured, but worse
  return h("div", {class: "ta-reco", id: "reco"},
    h("div", {class: "ta-reco-head"}, h("b", null, st.bundle.session ? "AI 추천" : "Auto-tune 추천"), h("span", {class: "ppl"}, `PPL ${fmt(ppl)}`),
      h("span", {class: "ta-muted"}, `원본 ${fmt(native)} 대비 ${signed(ppl - native)} · ${gpuName(base.ppl_record).split("-")[0]} 측정`)),
    h("ol", {class: "ta-steps"},
      h("li", null, h("span", null, "층마다 방법을 다르게 골랐습니다"),
        h("span", {class: "num"}, uniform ? `한 방법 최선 ${pplText(uniform.ppl)} → ${fmt(base.ppl)}` : `PPL ${fmt(base.ppl)}`),
        h("span", {class: "ta-muted", id: "solver-note", title: "크기 예산 안에서 모델 전체 오차가 가장 작은 조합을 브라우저가 정확히 계산했습니다 (다중 선택 배낭)."},
          `정확 해 · ${fmt(p.seconds, 2)}초 (${p.thread === "main" ? "메인 스레드" : "Worker"}${p.warm ? " · 미리 계산" : ""})`)),
      best ? h("li", null, h("span", null, "AI 가 GPU 로 재서 찾은 세부 설정을 더했습니다"),
        h("span", {class: "num"}, `${fmt(base.ppl)} → ${fmt(best.ppl)}`), h("code", null, optionText(best)))
        : other ? h("li", {class: "skip"}, h("span", null, "AI 가 찾은 세부 설정은 이 크기에서 더 나빠 쓰지 않았습니다"),
          h("span", {class: "num"}, `${fmt(base.ppl)} → ${fmt(other.ppl)}`), h("code", null, optionText(other))) : null));
}

// the assignment as blocks × 7 layer kinds, colored by bits; a cell says its layer's method on hover
function gridView(grid) {
  const t = table();
  const count = new Map();
  const cells = grid.flatMap((row, b) => row.map((method, k) => {
    count.set(method, (count.get(method) ?? 0) + 1);
    const cell = h("i", {class: "ta-cell", "data-b": String(elementBits(method))});
    const layer = t.layers.find((item) => item.block === b && item.kind === t.kinds[k]);
    const kl = method === "bf16" ? 0 : layer?.kl[t.methods.indexOf(method)];
    return withTip(cell, () => h("div", null, h("b", null, `블록 ${b} · ${t.kinds[k]}`), h("table", null, h("tbody", null,
      h("tr", null, h("td", null, "방법"), h("td", null, `${methodLabel(method)} (${method})`)),
      h("tr", null, h("td", null, "비트"), h("td", null, String(elementBits(method)))),
      h("tr", null, h("td", null, "이 층만 바꿨을 때 오차"), h("td", null, `KL ${kl == null ? "—" : kl === 0 ? "0" : kl < 0.001 ? kl.toExponential(1) : kl.toFixed(4)}`))))));
  }));
  const used = [...count].sort((a, b) => b[1] - a[1]);
  const TOP = 3;
  return h("div", {class: "ta-block"},
    blockHead("층별 배정", h("span", null, `Linear ${cells.length}개 · 칸에 올리면 그 층의 방법`)),
    h("div", {class: "ta-plan"},
      h("div", {class: "ta-kinds", "aria-hidden": "true"}, t.kinds.map((k) => h("span", null, KIND[k] ?? k))),
      h("div", {class: "ta-grid", style: `grid-template-columns: repeat(${t.blocks}, minmax(4px, 1fr))`, role: "img", "aria-label": `블록 ${t.blocks} × 층 종류 ${t.kinds.length} 의 방법 배정: ${used.map(([m, n]) => `${methodLabel(m)} ${n}층`).join(", ")}`}, cells),
      h("div", {class: "ta-axis"}, [0, 1, 2, 3].map((q) => Math.round(q * t.blocks / 4)).concat(t.blocks - 1).map((b) => h("span", null, `블록 ${b}`)))),
    h("div", {class: "ta-legend"}, [2, 3, 4, 8, 16].map((b) => h("span", null, h("i", {style: `background: var(--b${b})`}),
      b === 16 ? "BF16 유지" : `${b}비트`))),
    h("p", {class: "ta-muted", id: "used"}, used.slice(0, TOP).map(([m, n]) => `${methodLabel(m)} ${n}층`).join(" · ") +
      (used.length > TOP ? ` · 그 밖 ${used.length - TOP}종 ${used.slice(TOP).reduce((sum, [, n]) => sum + n, 0)}층` : "")));
}

function recipeView() {
  if (st.recipeError) return h("p", {class: "ta-err"}, `레시피 파일: ${st.recipeError}`);
  const r = st.recipe;
  if (!r) return h("p", {class: "ta-muted"}, "레시피 파일을 만드는 중…");
  const copy = h("button", {type: "button", class: "ta-btn small"}, "복사");
  copy.addEventListener("click", () => navigator.clipboard?.writeText(r.yaml).then(() => (copy.textContent = "복사됨")));
  const save = h("button", {type: "button", class: "ta-btn small"}, "다운로드");
  save.addEventListener("click", () => download(r));
  // the file's identity: its name, whether it is the very file the GPU measured, its SHA-256 (12 digits shown, all in data-sha)
  return h("div", {class: "ta-block"},
    h("div", {class: "ta-file", id: "recipe-file", "data-sha": r.sha256}, h("span", {class: "ta-file-icon", "aria-hidden": "true"}, "YAML"),
      h("div", {class: "ta-file-meta"}, h("b", null, r.file),
        r.measured?.same_file ? h("span", {class: "ok"}, "✓ GPU 가 측정한 바로 그 파일") : h("span", null, "이 파일의 GPU 측정 기록은 없습니다"),
        h("span", {title: `SHA-256 ${r.sha256}`}, `SHA-256 ${r.sha256.slice(0, 12)}…`)),
      copy, save),
    h("details", {class: "ta-more"}, h("summary", null, `YAML 보기 · ${r.yaml.split("\n").length - 1}줄`), h("pre", {class: "ta-code"}, r.yaml)));
}

// 3 edit by hand, among what was measured: the assignment (the recommended per-layer one, or one method on every
// layer) and, on the recommended assignment, the option settings
function drawEdit() {
  if (isDet()) return drawEditDet();
  const now = current();
  const on = Boolean(st.model && now);
  $("sec-edit").toggleAttribute("data-off", !on);
  if (!on) {
    $("edit").replaceChildren();
    $("edit-state").replaceChildren(h("span", {class: "ta-muted"}, "추천을 받은 뒤 직접 바꿉니다"));
    return;
  }
  const base = st.plan.recorded;
  const recommended = recommendedOption(base)?.id ?? null;
  const changed = edited();
  const back = h("button", {type: "button", class: "ta-btn small", id: "reset"}, "추천으로 되돌리기");
  back.addEventListener("click", () => edit({uniform: null, option: recommended}));
  $("edit-state").replaceChildren(chip(changed ? "직접 편집함" : "추천 그대로", changed ? "ai" : "plain"), changed ? back : "");

  const uniforms = uniformsWithin(st.plan.budget);
  const assign = h("select", {id: "assign", "aria-label": "층별 배정"},
    h("option", {value: ""}, "Auto-tune 배정 — 층마다 다른 방법"),
    h("optgroup", {label: `전체를 한 방법으로 — 이 크기 안에서 GPU 로 잰 ${uniforms.length}개`},
      uniforms.map((b) => h("option", {value: b.method}, `${methodLabel(b.method)} · 크기 ${pct(b.bits)} · PPL ${pplText(b.ppl)}`))));
  assign.value = st.uniform ?? "";
  assign.addEventListener("change", () => edit({uniform: assign.value || null}));
  const settings = st.uniform ? h("p", {class: "ta-muted"}, "한 방법 배정은 기본 설정으로 잰 기록만 있습니다.")
    : h("div", {class: "ta-presets list", role: "group", "aria-label": "세부 설정 — 이 배정에서 측정된 것"},
      preset({"data-option": "default"}, st.option == null, `기본 설정 · GPU 측정 PPL ${fmt(base?.ppl)}`,
        "기본 설정", `PPL ${fmt(base?.ppl)}${recommended == null ? " · 추천" : ""}`, () => st.option != null && edit({option: null})),
      optionsFor(base).sort((a, b) => b.ppl - a.ppl).map((o) => preset({"data-option": o.id}, st.option === o.id,
        `${o.id} · ${optionScope(o)} · GPU 측정 PPL ${fmt(o.ppl)}`, h("code", null, optionText(o)),
        [`PPL ${fmt(o.ppl)} (${signed(o.ppl - base.ppl)})`, o.id === recommended ? " · 추천" : null,
          o.ppl >= base.ppl ? h("span", {class: "warn"}, " · 권하지 않음") : null],
        () => st.option !== o.id && edit({option: o.id}))));
  $("edit").replaceChildren(h("div", {class: "ta-edit"},
    h("div", {class: "ta-col"},
      h("div", {class: "ta-block"}, blockHead("층별 배정", h("span", null, "추천대로 두거나, 전체를 한 방법으로 바꿉니다")), assign),
      h("div", {class: "ta-block"}, blockHead("세부 설정", st.uniform ? null : h("span", null, `${optionsFor(base)[0] ? optionScope(optionsFor(base)[0]) : "GPTQ 계열"}의 옵션`)), settings),
      h("p", {class: "ta-muted"}, "지금은 GPU 로 재 둔 조합만 고를 수 있습니다. 층 하나씩 바꾸는 편집은 측정 기록이 생기면 엽니다.")),
    h("div", {class: "ta-col"}, gridView(now.grid), recipeView())));
}

// How far each way of reaching this size drifts from the original: zero-based bars of the PPL increase, one row per
// measured way, the setup on screen in bold.
// a metric's ladder words: how far a value is from the original's (gap ≥ 0 = worse), and how it is written
const PPL_UNIT = {head: "원본에서 벌어진 정도 (PPL 증가)", col: "증가", name: "PPL", gap: (v, base) => v - base, delta: (g) => signed(g), text: (v) => pplText(v)};
function ladder(rows, nativeValue, size, unit = PPL_UNIT) {
  const max = Math.max(...rows.map((r) => unit.gap(r.ppl, nativeValue)), 1e-9);
  return h("div", {class: "ta-ladder", id: "ladder"},
    h("div", null, h("span", {class: "th"}, `크기 ${size} 안에서`), h("span", {class: "th"}, unit.head),
      h("span", {class: "th num"}, unit.col), h("span", {class: "th num"}, unit.name)),
    rows.map((r) => h("div", {class: `${r.series}${r.cur ? " cur" : ""}`},
      h("span", {class: "name"}, r.series === "base" ? null : h("i", {class: `ta-sw ${r.series}`}), r.name,
        r.sub ? h("small", {title: r.title ?? null}, r.sub) : null, r.help ?? null),
      h("span", {class: "track"}, r.series === "base" ? null
        : h("i", {style: `width: ${(100 * Math.max(0, unit.gap(r.ppl, nativeValue)) / max).toFixed(1)}%`})),
      h("span", {class: "num"}, r.series === "base" ? "기준" : unit.delta(unit.gap(r.ppl, nativeValue))),
      h("span", {class: "num"}, unit.text(r.ppl)))));
}

function recordView(now) {
  const {native} = st.bundle;
  const base = st.plan.recorded;
  const within = uniformsWithin(st.plan.budget);
  const uniform = within[0] ?? null;
  const rows = [{series: "base", name: "원본", sub: "BF16 · 100%", ppl: native.ppl}];
  if (now.kind === "uniform" && now.method !== uniform?.method) {
    rows.push({series: "uni", name: "직접 편집", sub: `전체 ${methodLabel(now.method)} · ${pct(now.bits)}`, title: now.method, ppl: now.ppl, cur: true});
  }
  if (uniform) {
    rows.push({series: "uni", name: "한 방법 최선", sub: `${methodLabel(uniform.method)} · ${pct(uniform.bits)}`, title: uniform.method, ppl: uniform.ppl,
      cur: now.kind === "uniform" && now.method === uniform.method,
      help: help("한 방법", `모든 층에 같은 양자화 방법을 쓰는 보통 방식입니다. 이 크기 안에서 GPU 로 잰 ${within.length}가지 가운데 PPL 이 가장 낮은 것과 비교합니다 (${within.map((b) => methodLabel(b.method)).join(", ")}).`)});
  }
  if (base?.ppl != null) rows.push({series: "heur", name: "Auto-tune 배정", sub: "기본 설정", ppl: base.ppl, cur: now.kind === "auto"});
  for (const o of optionsFor(base).sort((a, b) => b.ppl - a.ppl)) {
    rows.push({series: "ai", name: "+ AI 세부 설정", sub: optionText(o), title: o.id, ppl: o.ppl, cur: now.id === o.id});
  }
  const loss = now.ppl - native.ppl;
  const reco = recommendedOption(base)?.ppl ?? base?.ppl ?? null;
  const gain = uniform ? 100 * (1 - loss / (uniform.ppl - native.ppl)) : null;
  const verdict = edited() && reco != null
    ? ["Auto-tune 추천(PPL ", fmt(reco), ")은 이 편집보다 원본에서 벌어진 정도가 ", h("b", null, `${fmt(Math.abs(100 * (1 - (reco - native.ppl) / loss)), 0)}% ${reco <= now.ppl ? "작습니다" : "큽니다"}`), "."]
    : gain != null ? ["같은 크기의 한 방법 최선보다 원본에서 벌어진 정도가 ", h("b", null, `${pctText(Math.abs(gain))}% ${gain >= 0 ? "작습니다" : "큽니다"}`), "."] : null;
  return [
    h("div", {class: "ta-result", id: "eval-result"},
      h("div", {class: "ta-hero"},
        h("span", null, "WikiText-2 test PPL", help("PPL (퍼플렉서티)", `모델이 다음 단어를 얼마나 헷갈리는지 잰 값으로, 낮을수록 좋습니다. 원본(BF16)은 ${fmt(native.ppl)} 입니다.`)),
        h("b", null, pplText(now.ppl)), h("small", null, `원본 ${fmt(native.ppl)} 대비 ${signed(loss)}`)),
      h("div", {class: "ta-col"}, ladder(rows, native.ppl, pct(st.plan.bits)),
        verdict ? h("p", {class: "ta-gain", id: "gain"}, verdict) : null)),
    benchView(),
    h("p", {class: "ta-muted"}, `PPL 은 셋업을 고르는 데 쓰지 않은 데이터 (${native.n_tokens?.toLocaleString("en-US") ?? "—"} 토큰) 로 쟀습니다.` +
      (st.benchmarks ? "" : " lm-eval 과제별 정확도는 이 레시피의 기록이 없습니다."))].filter(Boolean);
}

// one benchmark per row: the original, the existing way — the best of the existing ways within the size on that
// row, named under its value — and ours, then ours − existing, with its standard error and verdict under it where
// it has one. groups: [{name, rows: [{label, sub, title, native, theirs, best, ours, diff, se, verdict, total}]}]
const VERDICT = {better: "우리 우세", same: "차이 없음", worse: "기존 우세"};
function benchTable(id, label, groups) {
  const head = (series, text) => h("th", {class: "num", scope: "col"}, series ? h("i", {class: `ta-sw ${series}`}) : null, text);
  return h("div", {class: "ta-bench-wrap"}, h("table", {class: "ta-table ta-bench", id, "aria-label": label},
    h("thead", null, h("tr", null, h("th", {scope: "col"}, "벤치마크"), head(null, "원본"), head("uni", "기존 방법"),
      head("heur", "우리 방법"), head(null, "우리 − 기존"))),
    groups.map((g) => h("tbody", null, g.name ? h("tr", {class: "grp"}, h("th", {colspan: "5", scope: "rowgroup"}, g.name)) : null,
      g.rows.map((r) => h("tr", {class: r.total ? "total" : null, title: r.title ?? null},
        h("th", {scope: "row"}, r.label, r.sub ? h("small", null, r.sub) : null),
        h("td", {class: "num"}, r.native),
        h("td", {class: "num"}, r.theirs, r.best ? h("small", null, r.best) : null),
        h("td", {class: "num"}, r.ours),
        h("td", {class: "num"}, r.diff, r.se || r.verdict
          ? h("small", {class: `ta-vd ${r.verdict ?? ""}`}, [r.se, r.verdict ? VERDICT[r.verdict] : null].filter(Boolean).join(" · ")) : null)))))));
}

// The size on screen on more benchmarks than WikiText-2 (scripts/quantizer/tune_bench.py, recorded on a GPU with the
// same recipe files — SHA-256 — as the PPL above): the Auto-tune assignment against, per benchmark, the best one-method
// setup measured within the size. Accuracy differences are paired over the same documents.
function benchView() {
  const bm = st.benchmarks;
  const base = st.plan.recorded;
  const cmp = base ? bm?.compare?.[base.recipe_file_sha256] : null;
  if (!bm) return null;
  if (!cmp) return h("p", {class: "ta-muted", id: "bench"}, "이 크기는 여러 벤치마크 기록이 없습니다.");
  const baseline = (id) => st.bundle.baselines.find((row) => row.id === id);
  const method = (id) => methodLabel(baseline(id)?.method ?? id);
  const named = (id) => `${method(id)} · ${pct(baseline(id)?.bits ?? 0)}`;  // the existing way's own size: it can be smaller
  const value = (b, v) => (b.kind === "ppl" ? pplText(v) : fmt(100 * v, 1));
  const byId = new Map(cmp.rows.map((r) => [r.bench, r]));
  const rows = bm.benchmarks.map((b) => {
    const r = byId.get(b.id);
    return {group: b.group, label: b.kind === "ppl" ? `${b.label} PPL` : b.label, title: b.what,
      sub: b.kind === "ppl" ? "낮을수록 좋음" : `${b.num_fewshot ? `예제 ${b.num_fewshot}개` : "예제 없이"} · ${b.n.toLocaleString("en-US")}문항` +
        (b.n_total > b.n ? ` (전체 ${b.n_total.toLocaleString("en-US")})` : ""),
      native: value(b, bm.native.values[b.id]), theirs: value(b, r.theirs), best: named(r.best), ours: value(b, r.ours),
      diff: b.kind === "ppl" ? `${signed(100 * (r.ours / r.theirs - 1), 1)}%` : signed(100 * r.diff, 1),
      se: b.kind === "ppl" ? null : `± ${fmt(100 * r.se, 1)}`, verdict: r.verdict};
  });
  const groups = [...new Set(rows.map((r) => r.group))].map((name) => ({name, rows: rows.filter((r) => r.group === name)}));
  const m = cmp.macro;
  const accs = bm.benchmarks.filter((b) => b.kind === "acc");
  const nativeMean = accs.reduce((sum, b) => sum + bm.native.values[b.id], 0) / accs.length;
  groups.push({name: null, rows: [{total: true, label: "정확도 평균", sub: `${m.tasks}과제 · 기존은 평균이 가장 높은 한 방법`,
    native: fmt(100 * nativeMean, 1), theirs: fmt(100 * m.theirs, 1), best: named(m.best), ours: fmt(100 * m.ours, 1),
    diff: signed(100 * m.diff, 1), se: `± ${fmt(100 * m.se, 1)}`, verdict: m.verdict}]});
  const count = (v) => cmp.rows.filter((r) => r.verdict === v).length;
  const env = bm.native.env ?? {};
  // one method on every layer comes in steps (2 · 3 · 4 bits …): between two steps the existing way is the step below
  const largest = Math.max(...cmp.uniforms.map((id) => baseline(id)?.bits ?? 0));
  const step = st.plan.bits - largest > 0.1
    ? `이 크기에 맞는 한 방법이 없어, 기존 방법은 그 아래 단계(가장 큰 것이 ${pct(largest)})입니다. ` : "";
  // the table's own measurement: on another GPU than the bars above, the same recipe's PPL differs a little
  const gpu = (env.gpu_names ?? [])[0];
  const other = gpu && st.bundle.native.gpu && gpu !== st.bundle.native.gpu
    ? `이 표는 위 막대(${gpuName({gpu: st.bundle.native.gpu})})와 다른 GPU 에서 다시 재서 같은 레시피의 PPL 이 조금 다릅니다 (원본 ${fmt(st.bundle.native.ppl)} → ${fmt(bm.native.values.wikitext2)}). ` : "";
  return h("div", {class: "ta-block", id: "bench"},
    blockHead("같은 크기, 여러 벤치마크", help("여러 벤치마크", `PPL 하나로 고른 셋업이 다른 과제에서도 나은지 봅니다. '기존 방법' 칸은 과제마다 이 크기 안에서 GPU 로 잰 한 방법 ${cmp.uniforms.length}가지(${cmp.uniforms.map(method).join(", ")}) 가운데 그 과제 점수가 가장 좋은 것입니다. 정확도 차이는 같은 문항끼리 짝지어 잰 표준오차(±)와 함께 보이고, 2 표준오차를 넘을 때만 우세로 적습니다 (과제마다 따로 판정하고 다중 비교 보정은 하지 않음). PPL 은 문항별 기록이 없어 판정하지 않습니다.`),
      h("span", null, `lm-eval ${bm.lm_eval_version} · ${(env.gpu_names ?? [])[0] ?? ""} · git ${String(env.git_sha ?? "").slice(0, 7)}`)),
    benchTable("bench-table", "같은 크기에서 기존 방법과 우리 방법의 벤치마크별 점수", groups),
    h("p", {class: "ta-muted", id: "bench-sum"}, `정확도 ${cmp.rows.filter((r) => r.verdict).length}과제: 우리 우세 ${count("better")} · 차이 없음 ${count("same")} · 기존 우세 ${count("worse")} (같은 문항 짝 비교, 2 표준오차 기준). ${step}${other}` +
      "정확도는 %, 차이는 %포인트이고 ± 는 그 표준오차입니다. 우리 방법은 Auto-tune 배정의 기본 설정입니다."));
}

function jobView() {
  const e = st.evaluation;
  const record = e.record ?? {};
  if (e.status === "error") return [h("p", {class: "ta-err"}, `평가 실패: ${e.error ?? record.error?.message ?? record.error ?? "원인 불명"}`)];
  if (e.status !== "done") {
    const done = Math.round((record.progress ?? 0) * 100);
    return [h("div", {class: "ta-row"}, chip(e.status === "queued" ? "대기" : `${record.stage ?? "실행"} 중`, "live"), chip(`${done}%`)),
      h("div", {class: "ta-progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(done)},
        h("i", {style: `width: ${done}%`}))];
  }
  const results = record.results ?? {};
  const run = results.run ?? {};
  const native = st.bundle.native;
  const out = [h("div", {class: "ta-row"}, chip("이 서버 GPU 에서 방금 측정", "ok"), chip(`레시피 SHA ${String(record.recipe?.sha256 ?? "").slice(0, 12)}`))];
  if (run.ppl != null) {
    out.push(h("div", {class: "ta-result", id: "eval-result"},
      h("div", {class: "ta-hero"}, h("span", null, "WikiText-2 test PPL"), h("b", null, fmt(run.ppl)),
        h("small", null, `원본 ${fmt(native.ppl)} 대비 ${signed(run.ppl - native.ppl)}`)),
      h("p", {class: "ta-muted"}, `${run.n_tokens?.toLocaleString("en-US") ?? "—"} 토큰 · ${record.env?.gpu ?? ""}`)));
  }
  const tasks = results.lmeval?.tasks ?? [];
  if (tasks.length) {
    out.push(h("table", {class: "ta-table"}, h("thead", null, h("tr", null, h("th", null, "과제"), h("th", null, "지표"),
      h("th", {class: "num"}, "정확도"), h("th", {class: "num"}, "±"), h("th", {class: "num"}, "표본"))),
    h("tbody", null, tasks.map((t) => h("tr", null, h("td", null, t.label ?? t.task), h("td", {class: "mono"}, t.metric),
      h("td", {class: "num"}, fmt(100 * t.value, 1)), h("td", {class: "num"}, t.stderr == null ? "—" : fmt(100 * t.stderr, 1)),
      h("td", {class: "num"}, t.n ?? "—"))))));
  }
  return out;
}

// 4 evaluation of the setup on screen
// step 4 is redrawn whenever its data changes (a recipe file arriving late included): a comparison tab that had the
// focus keeps it
function drawEval() {
  const tabs = document.activeElement?.closest?.('#eval [role="tablist"]')?.id ?? null;
  if (isDet()) drawEvalDet();
  else drawEvalLm();
  if (tabs) document.querySelector(`#${tabs} [role="tab"][aria-selected="true"]`)?.focus();
}

function drawEvalLm() {
  const now = current();
  const on = Boolean(st.model && now);
  $("sec-eval").toggleAttribute("data-off", !on);
  if (!on) {
    $("eval").replaceChildren();
    $("eval-state").replaceChildren(h("span", {class: "ta-muted"}, "셋업이 정해지면 레시피 파일 그대로 평가합니다"));
    return;
  }
  const live = gpuLive();
  const e = st.evaluation;
  // where the result came from, beside the title: a record says which GPU and tree measured it and whether the recipe
  // file on screen is the very file measured
  const same = st.recipe?.measured?.same_file ?? null;
  $("eval-state").replaceChildren(...(!e ? [chip(live ? "GPU 준비됨" : "GPU 없음 · 기록 대조", live ? "ok" : "plain")]
    : e.kind === "record" ? [chip(`기록된 GPU 측정 · ${gpuName(now.record)} · git ${(now.record?.git_sha ?? "").slice(0, 7)}`, "ok"),
      same === true ? chip("레시피 파일 SHA 일치", "ok") : same === false ? chip("레시피 파일 SHA 다름", "bad") : chip("레시피 파일 대조 안 됨")]
      : e.kind === "none" ? [chip("측정 기록 없음", "bad")]
        : [chip({queued: "대기", running: "실행 중", done: "완료", error: "실패"}[e.status] ?? e.status, e.status === "error" ? "bad" : "live")]));
  const parts = [];
  if (live) {  // a GPU server: what to measure besides PPL
    const limit = h("input", {type: "number", min: "1", max: "300", value: String(st.limit), "aria-label": "과제당 표본 수"});
    limit.addEventListener("change", () => (st.limit = Math.max(1, Math.min(300, Number(limit.value) || 200))));
    parts.push(h("div", {class: "ta-tasks"},
      h("label", null, h("input", {type: "checkbox", checked: true, disabled: true}), h("span", null, "PPL · WikiText-2 test")),
      LMEVAL.map(([id, label]) => {
        const box = h("input", {type: "checkbox", checked: st.lm.has(id)});
        box.addEventListener("change", () => (box.checked ? st.lm.add(id) : st.lm.delete(id)));
        return h("label", null, box, h("span", null, label));
      }),
      h("label", null, h("span", null, "과제당 표본"), limit)));
  }
  // the one action of this step, in the title row; a record needs no second press — it stays until the setup changes
  if (live || !e) {
    const runningJob = e?.kind === "job" && !["done", "error"].includes(e.status);
    const go = h("button", {type: "button", class: "ta-btn primary", id: "run-eval", disabled: runningJob || (live && !st.recipe)},
      live ? "GPU 에서 평가 실행" : "평가 결과 불러오기");
    go.addEventListener("click", runEval);
    $("eval-state").append(go);
    if (!e) {
      parts.push(h("p", {class: "ta-muted"}, live ? "레시피 파일 그대로 이 서버 GPU 에서 PPL 과 고른 과제를 잽니다."
        : "이 PC 에는 GPU 가 없어, 지금 셋업의 GPU 측정 기록(WikiText-2 test PPL)을 레시피 파일 지문(SHA-256)과 대조해 보여 줍니다."));
    }
  }
  if (e?.kind === "record") parts.push(...recordView(now));
  else if (e?.kind === "none") {
    parts.push(h("div", {class: "ta-warn"}, h("p", {class: "ta-muted"}, "이 배정은 GPU 측정 기록의 배정과 다릅니다 (번들의 표나 배낭이 바뀌었을 때). 번들을 다시 만드세요."),
      h("pre", {class: "ta-cmd"}, `tricast ppl --model ${st.model} --recipe ${st.recipe?.file ?? "recipe.yaml"}`)));
  } else if (e?.kind === "job") parts.push(...jobView());
  $("eval").replaceChildren(...parts);
}

// ---------- 5 what the evaluated setup hands over ----------
const GB = (bytes) => `${(bytes / 1e9).toFixed(2)} GB`;
const PART = {linear: "Linear 가중치", embedding: "임베딩 · 출력층 · 정규화"};

// the model file under this setup against the original, by part (computed by the server: app/autotune_export.py)
function sizeView(size) {
  if (!size) return h("p", {class: "ta-muted"}, "이 모델의 원본 파일 정보가 없어 크기를 계산하지 않았습니다.");
  const max = size.original_bytes;
  // two parts on screen: the Linear weights, and the rest kept in BF16 (the norms, 0.01% of the file, go with it)
  const two = (parts) => [parts[0], {part: "embedding", params: parts[1].params + parts[2].params, bits: 16,
    bytes: parts[1].bytes + parts[2].bytes}];
  const bar = (label, sub, parts, total, cls) => h("div", {class: cls},
    h("span", {class: "name"}, label, h("small", null, sub)),
    h("span", {class: "track"}, parts.map((p) => h("i", {class: p.part, style: `width: ${(100 * p.bytes / max).toFixed(2)}%`,
      title: `${PART[p.part]} · ${p.params.toLocaleString("en-US")}개 × ${fmt(p.bits, Number.isInteger(p.bits) ? 0 : 3)}비트 = ${GB(p.bytes)}`}))),
    h("span", {class: "num"}, GB(total)));
  const linear = size.setup[0];
  return h("div", {class: "ta-block", id: "size"},
    blockHead("압축 크기", help("압축 크기 (계산값)", `원본 체크포인트(model.safetensors)의 텐서를 종류별로 세고, Linear 가중치만 고른 비트(스케일 포함)로, 나머지는 BF16 으로 계산했습니다. 파일을 재서 얻은 값이 아닙니다.`),
      h("span", null, "계산값")),
    h("div", {class: "ta-size"},
      bar("원본", "BF16", two(size.original), size.original_bytes, "orig"),
      bar("이 셋업", `Linear ${fmt(linear.bits, 3)}비트`, two(size.setup), size.setup_bytes, "cur")),
    h("div", {class: "ta-legend"}, ["linear", "embedding"].map((part) => h("span", null, h("i", {class: `ta-size-sw ${part}`}),
      part === "linear" ? "Linear 가중치 (양자화)" : "임베딩 · 출력층 · 정규화 (BF16 유지)"))),
    h("p", {class: "ta-gain", id: "size-note"}, "파일 전체 ", h("b", null, `${GB(size.original_bytes)} → ${GB(size.setup_bytes)} (${fmt(100 * size.ratio, 0)}%)`),
      `. Linear 가중치는 원본의 ${fmt(100 * size.linear_ratio, 0)}% 로 줄고, 임베딩 · 출력층 ${GB(size.original[1].bytes)} 은 BF16 그대로입니다.`),
    h("p", {class: "ta-muted"}, "저비트로 저장할 때의 크기를 계산한 값입니다. 실제로 만든 파일은 옆의 모델 패키지에 있습니다 (기록이 있는 셋업만)."));
}

// the low-bit package written for this very recipe file (scripts/quantizer/pack_index.py): its files as measured,
// the checks run when it was written and how to build and run it
const KIND_TEXT = {affine: "정수 코드 + 스케일 · 영점", nf4: "NF4 코드 + 블록 absmax", dense: "가중치 그대로 (코드를 꺼낼 수 없는 방법)"};
function packageView(pkg) {
  const r = st.recipe;
  if (!pkg) {
    return h("div", {class: "ta-block", id: "package"},
      blockHead("모델 패키지", h("span", null, "이 셋업은 아직 만든 파일이 없습니다")),
      h("pre", {class: "ta-cmd"}, `python -m tricast.pack build --model ${st.model} --revision ${table().meta.revision ?? "<rev>"} \\\n  --recipe ${r.file} --out pkg --ppl`),
      h("p", {class: "ta-muted"}, "GPU 에서 레시피를 적용 · 보정하며 정수 코드를 기록해, 저비트 safetensors 패키지와 읽어 들이는 설정을 씁니다."));
  }
  const orig = pkg.original_bytes;
  const bar = (label, sub, bytes, cls) => h("div", {class: cls}, h("span", {class: "name"}, label, h("small", null, sub)),
    h("span", {class: "track"}, h("i", {class: "linear", style: `width: ${(100 * bytes / orig).toFixed(2)}%`, title: GB(bytes)})),
    h("span", {class: "num"}, GB(bytes)));
  const delta = pkg.ppl_package / pkg.ppl_tricast - 1;
  const kinds = Object.entries(pkg.kinds).map(([k, n]) => `${KIND_TEXT[k] ?? k} ${n}층`).join(" · ");
  const seen = current()?.ppl;  // step 4's recorded PPL of this recipe file
  const same = seen != null && seen === pkg.ppl_tricast;
  const memory = pkg.gpu_memory_loaded_bytes != null && pkg.ppl_codes != null;
  return h("div", {class: "ta-block", id: "package"},
    blockHead("모델 패키지", help("모델 패키지 (실제 파일)", "레시피를 적용 · 보정하면서 정수 코드와 스케일을 기록해 쓴 저비트 파일입니다 (python -m tricast.pack). 읽을 때 레시피로 각 층을 해석해 TriCast 와 같은 가중치로 되돌리고, PyTorch 행렬곱으로 추론합니다."),
      chip(`실제 파일 · ${String(pkg.gpu ?? "").replace("NVIDIA ", "")}`, "ok")),
    h("div", {class: "ta-size"}, bar("원본", "체크포인트", orig, "orig"), bar("패키지", "safetensors · 설정 · 토크나이저", pkg.bytes, "cur")),
    h("table", {class: "ta-table", id: "package-files"}, h("tbody", null, pkg.files.map((f) => h("tr", null,
      h("td", {class: "mono"}, f.name), h("td", {class: "num"}, f.bytes >= 1e8 ? GB(f.bytes) : f.bytes >= 1e6 ? `${(f.bytes / 1e6).toFixed(1)} MB` : `${(f.bytes / 1e3).toFixed(1)} KB`))))),
    h("ul", {class: "ta-checks", id: "package-checks"},
      h("li", null, `${pkg.decoded_equal.layers}개 층 모두 TriCast 가 곱하는 가중치와 비트 단위로 같음 (0 의 부호만 예외: 정수 코드에는 −0 이 없음)`),
      h("li", null, `WikiText-2 PPL: TriCast ${fmt(pkg.ppl_tricast, 3)} · 패키지 ${fmt(pkg.ppl_package, 3)} (${signed(100 * delta, 2)}%, 같은 GPU)`,
        seen == null ? null : h("span", {class: "ta-muted"}, same ? " — ④ 의 기록과 비트 단위로 같음"
          : ` — ④ 의 기록(${fmt(seen, 2)})은 따로 보정 · 측정한 실행의 값 (같은 레시피 파일)`)),
      memory ? h("li", null, `코드만 올려 두고 층마다 풀어 쓰면 GPU 메모리 ${GB(pkg.gpu_memory_loaded_bytes)} (BF16 원본 ${GB(orig)}) · PPL ${fmt(pkg.ppl_codes, 3)}`) : null,
      h("li", null, kinds),
      pkg.tied.length ? h("li", null, "출력층은 임베딩과 묶여(tied) 한 번만 저장 — 원본 체크포인트는 두 번 저장해 계산값보다 작음") : null),
    h("pre", {class: "ta-cmd"}, `python -m tricast.pack run --package ${pkg.dir} --prompt "…"`),
    h("p", {class: "ta-muted"}, `파일은 GPU 클러스터에 있습니다 (${GB(pkg.bytes)} — 저장소에는 넣지 않음). 위 명령으로 같은 레시피에서 다시 만들 수 있습니다.`));
}

function exportView() {
  const r = st.recipe;
  const go = h("button", {type: "button", class: "ta-btn primary", id: "export", disabled: !r || st.exporting},
    st.exporting ? "묶는 중…" : "레시피 묶음 받기 (.zip)");
  go.addEventListener("click", exportBundle);
  const files = r ? [r.file, "evaluation.json", r.size ? "size.json" : null, "README.md"].filter(Boolean) : [];
  return h("div", {class: "ta-block", id: "export-block"},
    blockHead("내보내기", h("span", null, "이 셋업을 다른 곳에서 그대로 쓰는 데 필요한 것")),
    h("div", {class: "ta-row"}, go, h("span", {class: "ta-muted mono"}, files.join(" · "))),
    st.exportError ? h("p", {class: "ta-err"}, `내보내기: ${st.exportError}`) : null,
    h("p", {class: "ta-muted"}, `레시피 파일(GPU 가 측정한 바로 그 파일), 측정 기록, ${gpuLive() ? "이 서버 GPU 에서 방금 잰 결과, " : ""}크기 계산, 적용 명령이 든 README 를 묶습니다. 모델 가중치 파일은 넣지 않습니다 — 레시피를 적용하면 같은 모델이 됩니다.`));
}

// 5 opens once the setup has an evaluation on screen ("평가 후"): the computed size beside the recipe bundle
function drawOut() {
  if (isDet()) return drawOutDet();
  const e = st.evaluation;
  const on = Boolean(st.model && current() && st.recipe && (e?.kind === "record" || (e?.kind === "job" && e.status === "done")));
  $("sec-out").toggleAttribute("data-off", !on);
  if (!on) {
    $("out").replaceChildren();
    $("out-state").replaceChildren(h("span", {class: "ta-muted"}, "평가 결과를 보면 압축 크기와 내보내기가 열립니다"));
    return;
  }
  $("out-state").replaceChildren(st.recipe.size ? chip(`파일 ${GB(st.recipe.size.setup_bytes)} · 원본의 ${fmt(100 * st.recipe.size.ratio, 0)}%`, "plain") : "");
  $("out").replaceChildren(h("div", {class: "ta-outgrid"}, sizeView(st.recipe.size), exportView()),
    packageView((st.packages?.llm ?? []).find((p) => p.model === st.model && p.recipe_sha256 === st.recipe.sha256) ?? null));
}

// ---------- a detector (YOLO11n): a bit-width per Conv layer, scored by COCO mAP ----------
// Its file (scripts/quantizer/yolo_tune.py) holds every setup measured: one bit-width on every Conv ("uniform"), a
// common rule ("heuristic") — the existing ways — and Auto-tune's, the exact knapsack over a per-layer table of the
// detection output's error, at the existing ways' sizes. Weights only (fake-quant); activations stay FP32.
const AP = (v) => (v == null ? "—" : (100 * v).toFixed(1));
const AP_UNIT = {head: "원본에서 떨어진 정도 (mAP 감소)", col: "감소", name: "mAP", gap: (v, base) => base - v,
  delta: (g) => signed(-100 * g, 1), text: (v) => AP(v)};
const KEEP = new Set(["fp16", "bf16"]);  // the weight as it is: FP16 for the detector, BF16 for the image generator
const detLabel = (method) => (KEEP.has(method) ? `${method.toUpperCase()} (원본 그대로)` : method.toUpperCase());
const detBits = (method) => (method === "fp16" ? 16 : Number(/\d+/.exec(method)?.[0] ?? 16));
const MB = (bytes) => `${(bytes / 1e6).toFixed(2)} MB`;
// the detector file's quantization (scripts/quantizer/yolo_tune.py) in the page's words; the file's own text in the tooltip
const DET_QUANT = "가중치만 양자화 (fake-quant: 출력 채널마다 대칭 absmax 스케일 FP16, RNE) · 활성값은 FP32";
const GEN_QUANT = "가중치만 양자화 (fake-quant, RTN: INT8 은 출력 채널마다, INT4 · 3 · 2 는 입력 128개마다 FP16 스케일 + 영점) · 활성값 · 텍스트 인코더 · VAE 는 그대로";
const dB = (v) => (v == null ? "—" : `${v.toFixed(1)} dB`);
// what the page says and ranks by for each kind of measured-dataset model: the detector's COCO mAP; the image
// generator's SSIM to the unquantized model's image (×100: 100 = the same image), PSNR beside it
const KINDS = {
  detector: {key: "AP_50_95", short: "mAP", name: "COCO mAP50-95", unit: "Conv", what: "물체 검출", origin: "FP16", out: "검출 출력",
    toward: "입력 쪽 → 검출 헤드", quant: DET_QUANT, scale: "출력 채널마다 FP16 스케일 포함",
    evalText: (d) => `COCO ${d.evaluation.images.toLocaleString("en-US")}장`,
    explain: (v) => `검출 상자가 정답 상자와 얼마나 잘 맞는지를 IoU 0.5–0.95 에 걸쳐 평균한 값으로, 높을수록 좋습니다. 원본(FP16)은 ${AP(v)} 입니다.`,
    extra: (m) => `mAP50 ${AP(m.AP_50)}`,
    evalNote: (d) => `mAP 는 층별 오차 표에 쓰지 않은 COCO val2017 ${d.evaluation.images.toLocaleString("en-US")}장으로 쟀습니다 (pycocotools). ${DET_QUANT}.`,
    evalIntro: "측정해 둔 COCO mAP 과, 같은 사진에서 원본 · 기존 방법 · Auto-tune 이 찾은 상자를 보여 줍니다.",
    metricsTitle: "같은 크기, COCO 세부 지표", metricsUnit: "AP × 100 · pycocotools",
    metricsHelp: (d, within) => `같은 평가 한 번(${d.evaluation.images.toLocaleString("en-US")}장)을 IoU 기준과 물체 크기별로 나눠 본 값입니다. '기존 방법' 칸은 지표마다 이 크기 안 기존 방법 ${within.length}가지(${within.map((row) => row.label).join(", ")}) 가운데 가장 좋은 것입니다. 한 번 잰 값이라 표준오차는 없습니다.`,
    metricsNote: "다른 데이터셋이 아니라 같은 COCO 평가를 나눠 본 값입니다. 차이는 AP 포인트입니다."},
  image: {key: "ssim", short: "SSIM", name: "원본 이미지와의 SSIM", unit: "Linear", what: "이미지 생성", origin: "BF16", out: "생성 출력(속도장)",
    toward: "입력 쪽 → 마지막 층", quant: GEN_QUANT, scale: "스케일 · 영점 포함",
    evalText: (d) => `프롬프트 ${d.evaluation.prompts}개`,
    explain: () => "같은 프롬프트 · 같은 초기 잡음으로 만든 이미지가 원본 모델의 이미지와 구조적으로 얼마나 같은지(SSIM × 100)입니다. 100 이면 원본과 같은 이미지이고, 높을수록 좋습니다.",
    extra: (m) => `PSNR ${dB(m.psnr)}`,
    evalNote: (d) => `SSIM · PSNR 은 층별 오차 표에 쓰지 않은 프롬프트 ${d.evaluation.prompts}개로, 원본 모델이 같은 시드로 만든 이미지와 비교해 쟀습니다 (${d.model.size} px, ${d.model.steps}단계). 품질 자체(사람 평가 · CLIP 점수)가 아니라 원본과의 차이입니다. ${GEN_QUANT}.`,
    evalIntro: "측정해 둔 SSIM · PSNR 과, 같은 프롬프트로 원본 · 기존 방법 · Auto-tune 이 만든 이미지를 보여 줍니다.",
    metricsTitle: "같은 크기, 원본과의 차이", metricsUnit: "SSIM × 100 · PSNR dB",
    metricsHelp: (d, within) => `같은 평가(프롬프트 ${d.evaluation.prompts}개)의 두 지표입니다. '기존 방법' 칸은 지표마다 이 크기 안 기존 방법 ${within.length}가지(${within.map((row) => row.label).join(", ")}) 가운데 가장 좋은 것입니다. 한 번 잰 값이라 표준오차는 없습니다.`,
    metricsNote: "둘 다 원본 모델의 이미지와 비교한 값입니다 (높을수록 원본에 가까움)."},
};
const KD = () => KINDS[st.entry?.kind] ?? KINDS.detector;
const KD_UNIT = () => ({...AP_UNIT, head: `원본에서 떨어진 정도 (${KD().short} 감소)`, name: KD().short});
// the sizes Auto-tune was measured at, smallest first — but not a size where the knapsack had no choice (its
// assignment is an existing way's: at INT3's size with nothing below INT3, every layer is INT3)
const sameAssignment = (a, b) => Object.keys(a.methods).length === Object.keys(b.methods).length
  && Object.entries(a.methods).every(([layer, method]) => b.methods[layer] === method);
const detSizes = () => st.det.setups.filter((row) => row.kind === "auto" && row.metrics?.[KD().key] != null
  && !st.det.setups.some((other) => other.kind !== "auto" && sameAssignment(row, other))).sort((a, b) => a.bits - b.bits);
// the existing ways measured within the size on screen — the budget Auto-tune was given, not the bits it used —,
// best mAP first
const detWithin = () => {
  if (!st.plan) return [];
  const limit = (st.plan.budget_bits ? bitsOf(st.plan.budget_bits) : st.plan.bits) + 1e-9;
  return st.det.setups.filter((row) => row.kind !== "auto" && row.metrics?.[KD().key] != null && row.bits <= limit)
    .sort((a, b) => b.metrics[KD().key] - a.metrics[KD().key]);
};
const detCurrent = () => (st.plan ? st.det.setups.find((row) => row.id === (st.pick ?? st.plan.id)) ?? st.plan : null);

function detRecommend(id) {
  const plan = st.det.setups.find((row) => row.id === id);
  if (!plan) return;
  Object.assign(st, {budget: id, plan, pick: null, evaluation: null, image: 0});
  drawAll();
  saveState();
  showNext();
}

function detEdit(pick) {
  Object.assign(st, {pick: pick === st.plan.id ? null : pick, evaluation: null});
  drawAll();
  saveState();
}

function drawTuneDet() {
  $("sec-tune").toggleAttribute("data-off", false);
  const p = st.plan;
  const native = st.det.native.metrics[KD().key];
  const k = KD();
  $("tune-state").replaceChildren(...(p ? [chip(`추천 · 크기 ${pct(p.bits)} · ${k.short} ${AP(p.metrics[k.key])}`, "ok")] : []));
  const sizes = detSizes().map((row) => preset({"data-budget": row.id, "aria-label": `원본의 ${pct(row.bits)}, ${fmt(row.bits, 2)}비트, 추천 셋업의 ${k.name} ${AP(row.metrics[k.key])}`},
    row.id === st.budget, `평균 ${fmt(row.bits, 3)} 비트 (예산 ${row.bits_fraction}) · 추천 셋업의 ${k.name} ${AP(row.metrics[k.key])}`,
    pct(row.bits), [h("span", {class: "k"}, k.short), ` ${AP(row.metrics[k.key])}`], () => row.id !== st.budget && detRecommend(row.id)));
  const parts = [h("div", {class: "ta-block"},
    blockHead("목표 크기", help("크기", `${k.unit} 층(${st.det.layers.length}개) 가중치 하나를 저장하는 평균 비트 수(${k.scale})로 정한 크기입니다. 원본(${k.origin})은 16비트 = 100% 입니다.`),
      h("span", null, `원본 대비 · 아래 숫자는 그 크기 추천 셋업의 ${k.name}`)),
    h("div", {class: "ta-presets", role: "group", "aria-label": `목표 크기 — 측정된 ${sizes.length}개`}, sizes))];
  if (!p) parts.push(h("p", {class: "ta-muted"}, `목표 크기를 고르면 그 안에서 ${k.out}의 오차가 가장 작도록 ${k.unit} 층마다 비트 수를 골라 추천합니다.`));
  else {
    const best = detWithin()[0] ?? null;
    parts.push(h("div", {class: "ta-reco", id: "reco"},
      h("div", {class: "ta-reco-head"}, h("b", null, "Auto-tune 추천"), h("span", {class: "ppl"}, `${k.short} ${AP(p.metrics[k.key])}`),
        h("span", {class: "ta-muted"}, `원본 ${AP(native)} 대비 ${signed(100 * (p.metrics[k.key] - native), 1)}`)),
      h("ol", {class: "ta-steps"}, h("li", null, h("span", null, `${k.unit} 층마다 비트 수를 다르게 골랐습니다`),
        h("span", {class: "num"}, best ? `기존 최선 ${AP(best.metrics[k.key])} → ${AP(p.metrics[k.key])}` : `${k.short} ${AP(p.metrics[k.key])}`),
        h("span", {class: "ta-muted", title: `층 하나만 바꿔 잰 ${k.out} 오차의 합이 가장 작은 조합을 크기 예산 안에서 정확히 풀었습니다 (다중 선택 배낭).`},
          best ? `기존 최선 = ${best.label}` : "정확 해")))));
  }
  $("tune").replaceChildren(...parts);
}

// the Conv layers from the input to the detection head, one cell each, colored by bits
function stripView(row) {
  const d = st.det;
  const count = new Map();
  const cells = d.layers.map((layer) => {
    const method = row.methods[layer.name] ?? (isGen() ? "bf16" : "fp16");
    count.set(method, (count.get(method) ?? 0) + 1);
    const err = KEEP.has(method) ? 0 : layer.d[method];
    return withTip(h("i", {class: "ta-cell", "data-b": String(detBits(method))}), () => h("div", null, h("b", null, layer.name),
      h("table", null, h("tbody", null,
        h("tr", null, h("td", null, "방법"), h("td", null, detLabel(method))),
        h("tr", null, h("td", null, "가중치"), h("td", null, `${layer.params.toLocaleString("en-US")}개`)),
        h("tr", null, h("td", null, "이 층만 바꿨을 때 출력 오차"), h("td", null, err == null ? "—" : err === 0 ? "0" : err.toExponential(1)))))));
  });
  const used = [...count].sort((a, b) => b[1] - a[1]);
  const bits = [...new Set(used.map(([m]) => detBits(m)))].sort((a, b) => a - b);
  return h("div", {class: "ta-block"},
    blockHead("층별 배정", h("span", null, `${KD().unit} ${cells.length}개 · ${KD().toward} · 칸에 올리면 그 층의 방법`)),
    h("div", {class: "ta-strip", role: "img", "aria-label": `${KD().unit} ${cells.length}개의 비트 배정: ${used.map(([m, n]) => `${detLabel(m)} ${n}층`).join(", ")}`}, cells),
    h("div", {class: "ta-legend"}, bits.map((b) => h("span", null, h("i", {style: `background: var(--b${b})`}), b === 16 ? `${KD().origin} 유지` : `${b}비트`))),
    h("p", {class: "ta-muted", id: "used"}, used.map(([m, n]) => `${detLabel(m)} ${n}층`).join(" · ")));
}

function drawEditDet() {
  const now = detCurrent();
  $("sec-edit").toggleAttribute("data-off", !now);
  if (!now) {
    $("edit").replaceChildren();
    $("edit-state").replaceChildren(h("span", {class: "ta-muted"}, "추천을 받은 뒤 직접 바꿉니다"));
    return;
  }
  const back = h("button", {type: "button", class: "ta-btn small", id: "reset"}, "추천으로 되돌리기");
  back.addEventListener("click", () => detEdit(st.plan.id));
  $("edit-state").replaceChildren(chip(st.pick ? "직접 편집함" : "추천 그대로", st.pick ? "ai" : "plain"), st.pick ? back : "");
  const within = detWithin();
  const group = (kind, label) => {
    const rows = within.filter((row) => row.kind === kind);
    return rows.length ? h("optgroup", {label}, rows.map((row) => h("option", {value: row.id},
      `${row.label} · 크기 ${pct(row.bits)} · ${KD().short} ${AP(row.metrics[KD().key])}`))) : null;
  };
  const assign = h("select", {id: "assign", "aria-label": "층별 배정"},
    h("option", {value: ""}, "Auto-tune 배정 — 층마다 다른 비트"),
    group("uniform", "전체를 한 방법으로 — 이 크기 안"), group("heuristic", "흔한 규칙 — 이 크기 안"));
  assign.value = st.pick ?? "";
  assign.addEventListener("change", () => detEdit(assign.value || st.plan.id));
  $("edit").replaceChildren(h("div", {class: "ta-edit"},
    h("div", {class: "ta-col"},
      h("div", {class: "ta-block"}, blockHead("층별 배정", h("span", null, "추천대로 두거나, 이 크기 안의 기존 방법으로 바꿉니다")), assign),
      h("p", {class: "ta-muted", title: st.det.quantization}, `${KD().quant}. 지금은 측정해 둔 셋업만 고를 수 있습니다.`)),
    h("div", {class: "ta-col"}, stripView(now))));
}

// native boxes a setup kept: a box of the same class with IoU ≥ 0.5 (each setup box used once) → [kept, of]
function keptBoxes(native, boxes) {
  const iou = (a, b) => {
    const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const hgt = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = w * hgt;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter || 1);
  };
  const used = new Set();
  let kept = 0;
  for (const box of native) {
    const k = boxes.findIndex((other, i) => !used.has(i) && other[4] === box[4] && iou(box, other) >= 0.5);
    if (k >= 0) {
      used.add(k);
      kept += 1;
    }
  }
  return [kept, native.length];
}

function boxesSvg(boxes, img) {
  const names = st.det.names;
  const size = Math.max(12, Math.round(img.width / 26));
  return s("svg", {class: "ta-det-svg", viewBox: `0 0 ${img.width} ${img.height}`, "aria-hidden": "true",
    style: `--sw: ${Math.max(2, img.width / 260).toFixed(1)}px`},
  // every box is drawn; only the confident ones (≥ 0.5) are labeled, so a crowded photo stays readable — each box
  // says its class and score on hover
  boxes.map(([x1, y1, x2, y2, c, score]) => s("g", null,
    s("title", null, `${names[c] ?? c} ${score.toFixed(2)}`),
    s("rect", {x: x1, y: y1, width: Math.max(1, x2 - x1), height: Math.max(1, y2 - y1)}),
    score >= 0.5 ? s("text", {x: x1 + 2, y: y1 > size + 4 ? y1 - 4 : y1 + size + 2, "font-size": size}, `${names[c] ?? c} ${score.toFixed(2)}`) : null)));
}

// the same photos through the original, the existing way (the hand pick, or the best within the size) and Auto-tune
function detectView(now) {
  const d = st.det;
  const auto = st.plan;
  const existing = now.id !== auto.id ? now : detWithin()[0] ?? null;
  const at = Math.min(st.image, d.images.length - 1);
  const img = d.images[at];
  const cols = [{id: "native", title: "원본", sub: "FP16 · 100%", series: "base"},
    existing ? {id: existing.id, title: "기존 방법", sub: `${existing.label} · ${pct(existing.bits)}`, series: "uni", cur: now.id === existing.id} : null,
    {id: auto.id, title: "우리 방법 · Auto-tune", sub: `층별 배정 · ${pct(auto.bits)}`, series: "heur", cur: now.id === auto.id}].filter(Boolean);
  const thumbs = tabRow("photo-tabs", "사진", "ta-thumbs", "ta-thumb", d.images.map((item, k) => ({
    content: h("img", {src: `demo/${item.file}`, alt: "", loading: "lazy"}), attrs: {"aria-label": `사진 ${k + 1} (COCO ${item.coco_id})`}})),
  at, (k) => {
    st.image = k;
    drawEval();
  });
  const native = img.detections.native ?? [];
  return h("div", {class: "ta-block", id: "detections"},
    blockHead("같은 사진, 같은 크기 — 검출 결과", help("검출 비교", "같은 사진을 원본, 모든 층에 같은 방법을 쓴 모델(기존 방법), 층마다 비트 수를 고른 모델(우리 방법)에 넣어 얻은 상자입니다 (점수 0.25 이상). '유지' 는 원본 상자 가운데 같은 클래스 · IoU 0.5 이상으로 다시 찾은 수입니다."),
      h("span", null, "점수 0.25 이상 상자")),
    thumbs,
    h("div", {class: "ta-dets"}, cols.map((c) => {
      const boxes = img.detections[c.id] ?? [];
      const [kept, of] = c.id === "native" ? [null, null] : keptBoxes(native, boxes);
      return h("figure", {class: `ta-det ${c.series}${c.cur ? " cur" : ""}`},
        h("div", {class: "ta-det-img"}, h("img", {src: `demo/${img.file}`, alt: `${c.title}: 상자 ${boxes.length}개`, width: String(img.width), height: String(img.height)}),
          boxesSvg(boxes, img)),
        h("figcaption", null, c.series === "base" ? null : h("i", {class: `ta-sw ${c.series}`}), h("b", null, c.title), h("small", null, c.sub),
          h("span", {class: "num"}, kept == null ? `상자 ${boxes.length}개` : `원본 상자 ${kept}/${of} 유지 · ${boxes.length}개`)));
    })),
    h("p", {class: "ta-muted"}, "COCO val2017 사진 (CC BY 2.0, demo/media/SOURCES.md). 이 사진들은 층별 오차 표를 잰 보정 사진에 넣지 않았습니다."));
}

// the same prompts and starting noise through the original, the existing way (the hand pick, or the best within the
// size) and Auto-tune
function imagesView(now) {
  const d = st.det;
  const auto = st.plan;
  const existing = now.id !== auto.id ? now : detWithin()[0] ?? null;
  const at = Math.min(st.image, d.images.length - 1);
  const img = d.images[at];
  const cols = [{id: "native", title: "원본", sub: `${KD().origin} · 100%`, series: "base"},
    existing ? {id: existing.id, title: "기존 방법", sub: `${existing.label} · ${pct(existing.bits)}`, series: "uni", cur: now.id === existing.id} : null,
    {id: auto.id, title: "우리 방법 · Auto-tune", sub: `층별 배정 · ${pct(auto.bits)}`, series: "heur", cur: now.id === auto.id}].filter(Boolean);
  const thumbs = tabRow("photo-tabs", "프롬프트", "ta-thumbs", "ta-thumb", d.images.map((item, k) => ({
    content: h("img", {src: `demo/${item.files.native}`, alt: "", loading: "lazy"}), attrs: {"aria-label": `프롬프트 ${k + 1}: ${item.prompt}`}})),
  at, (k) => {
    st.image = k;
    drawEval();
  });
  return h("div", {class: "ta-block", id: "detections"},
    blockHead("같은 프롬프트, 같은 크기 — 생성한 이미지", help("생성 비교", "같은 프롬프트와 같은 초기 잡음(시드)을 원본, 모든 층에 같은 방법을 쓴 모델(기존 방법), 층마다 비트 수를 고른 모델(우리 방법)에 넣어 만든 이미지입니다. 숫자는 원본 이미지와의 PSNR 입니다."),
      h("span", null, `시드 ${img.seed} · ${d.model.steps}단계`)),
    thumbs,
    h("p", {class: "ta-prompt"}, `“${img.prompt}”`),
    h("div", {class: "ta-dets"}, cols.map((c) => {
      const v = c.id === "native" ? null : img.psnr[c.id];
      return h("figure", {class: `ta-det ${c.series}${c.cur ? " cur" : ""}`},
        h("div", {class: "ta-det-img"}, h("img", {src: `demo/${img.files[c.id]}`, alt: `${c.title}: ${img.prompt}`, width: "384", height: "384"})),
        h("figcaption", null, c.series === "base" ? null : h("i", {class: `ta-sw ${c.series}`}), h("b", null, c.title), h("small", null, c.sub),
          h("span", {class: "num"}, c.id === "native" ? "기준" : v == null ? "원본과 같음" : `PSNR ${v.toFixed(1)} dB`)));
    })),
    h("p", {class: "ta-muted"}, "프롬프트는 이 데이터를 위해 쓴 것이고 층별 오차 표를 잰 프롬프트와 겹치지 않습니다. 이미지는 384 px 로 줄여 보입니다."));
}

function detRecordView(now) {
  const native = st.det.native.metrics;
  const auto = st.plan;
  const within = detWithin();
  const k = KD();
  const rows = [{series: "base", name: "원본", sub: `${k.origin} · 100%`, ppl: native[k.key]}];
  for (const row of within) {
    rows.push({series: "uni", name: row.label, sub: `${row.kind === "heuristic" ? "흔한 규칙" : "한 방법"} · ${pct(row.bits)}`, title: row.id, ppl: row.metrics[KD().key], cur: now.id === row.id});
  }
  rows.push({series: "heur", name: "Auto-tune 배정", sub: `층별 · ${pct(auto.bits)}`, ppl: auto.metrics[KD().key], cur: now.id === auto.id});
  const best = within[0] ?? null;
  const loss = native[k.key] - auto.metrics[k.key];
  const gain = best ? 100 * (1 - loss / (native[k.key] - best.metrics[k.key])) : null;
  return [
    h("div", {class: "ta-result", id: "eval-result"},
      h("div", {class: "ta-hero"},
        h("span", null, k.name, help(k.short, k.explain(native[k.key]))),
        h("b", null, AP(now.metrics[k.key])), h("small", null, `원본 ${AP(native[k.key])} 대비 ${signed(100 * (now.metrics[k.key] - native[k.key]), 1)} · ${k.extra(now.metrics)}`)),
      h("div", {class: "ta-col"}, ladder(rows, native[k.key], pct(auto.bits), KD_UNIT()),
        gain != null ? h("p", {class: "ta-gain", id: "gain"}, `같은 크기의 기존 최선(${best.label})보다 원본에서 떨어진 정도가 `,
          h("b", null, `${fmt(Math.abs(gain), 0)}% ${gain >= 0 ? "작습니다" : "큽니다"}`), ".") : null)),
    detMetricsView(),
    isGen() ? imagesView(now) : detectView(now),
    h("p", {class: "ta-muted", title: st.det.quantization}, k.evalNote(st.det))];
}

// COCOeval's standard breakdown of the same run (scripts/quantizer/yolo_tune.py METRIC_KEYS): by IoU threshold and by
// object size, the Auto-tune assignment against, per metric, the best existing way within the size
const DET_METRICS = [["IoU 기준", [["AP_50_95", "mAP50-95", "IoU 0.5–0.95 평균"], ["AP_50", "AP50", "IoU 0.5 이상이면 맞음"],
  ["AP_75", "AP75", "IoU 0.75 이상 — 상자가 더 정확해야 맞음"]]],
["물체 크기", [["AP_small", "작은 물체", "넓이 32² 픽셀 미만"], ["AP_medium", "중간 물체", "32²–96² 픽셀"], ["AP_large", "큰 물체", "96² 픽셀 초과"]]]];
// the image generator's two comparisons with the unquantized model's image: SSIM (× 100) and PSNR (dB)
const GEN_METRICS = [["원본과 비교", [["ssim", "SSIM", "구조 유사도 × 100 · 100 = 같은 이미지"], ["psnr", "PSNR", "dB · 높을수록 원본에 가까움", "dB"]]]];
function detMetricsView() {
  const d = st.det;
  const auto = st.plan;
  const within = detWithin();
  const k = KD();
  const metrics = isGen() ? GEN_METRICS : DET_METRICS;
  // a PSNR of null is the same image (infinite PSNR), not a missing value; the table needs the ranking key only
  const value = (row, key, unit) => (unit === "dB" ? row.metrics[key] ?? Infinity : row.metrics[key]);
  const missing = isGen() ? auto.metrics[k.key] == null : metrics.some(([, items]) => items.some(([key]) => auto.metrics[key] == null));
  if (!within.length || missing) return null;
  const groups = metrics.map(([name, items]) => ({name, rows: items.map(([key, label, sub, unit]) => {
    const best = within.reduce((a, b) => (value(b, key, unit) > value(a, key, unit) ? b : a));
    const text = unit === "dB" ? (v) => (v == null ? "같음" : v.toFixed(1)) : AP;
    const [ours, theirs] = [value(auto, key, unit), value(best, key, unit)];
    const diff = !Number.isFinite(ours) || !Number.isFinite(theirs) ? "—" : unit === "dB" ? signed(ours - theirs, 1) : signed(100 * (ours - theirs), 1);
    return {label, sub, native: unit === "dB" ? "기준" : text(d.native.metrics[key]), theirs: text(best.metrics[key]), best: `${best.label} · ${pct(best.bits)}`,
      ours: text(auto.metrics[key]), diff};
  })}));
  return h("div", {class: "ta-block", id: "bench"},
    blockHead(k.metricsTitle, help(k.metricsTitle, k.metricsHelp(d, within)), h("span", null, k.metricsUnit)),
    benchTable("bench-table", `같은 크기에서 기존 방법과 우리 방법의 ${k.metricsTitle.replace("같은 크기, ", "")}`, groups),
    h("p", {class: "ta-muted", id: "bench-sum"}, k.metricsNote),
    isGen() && auto.paired ? pairedView(auto) : null);
}

// the image generator's verdict, fixed before the setups were measured (scripts/quantizer/zimage_tune.py compare):
// Auto-tune against its pair, per-prompt SSIM, exact sign-flip test, Holm within the pair's family of sizes. The
// between-step sizes were added after the first results — the page says so
const VERDICT_TEXT = {ours: "우리 우세", same: "차이 없음", theirs: "기존 우세"};
function pairedView(auto) {
  const p = auto.paired;
  const ref = st.det.setups.find((row) => row.id === p.vs);
  const pText = (v) => (v < 0.001 ? "< 0.001" : v.toFixed(3));
  const later = !String(p.fixed ?? "").startsWith("before");
  const pair = later
    ? `첫 결과를 본 뒤 더한 크기 — 짝: ${ref?.label ?? p.vs}, 이 크기 안의 유일한 기존 방법. 규칙은 재기 전에 정함`
    : `측정 전에 정한 짝: ${ref?.label ?? p.vs}`;
  const family = p.family?.length ?? st.det.setups.filter((row) => row.kind === "auto").length;
  return h("p", {class: "ta-gain", id: "paired"},
    `판정 (${pair}) — 프롬프트 ${p.n}개 중 ${p.wins}개에서 원본에 더 가깝고 ${p.losses}개에서 멀었습니다. SSIM 평균 차이 ${signed(100 * p.mean_diff, 1)}, sign-flip p ${pText(p.p_sign_flip)} (크기 ${family}개 Holm 보정 ${pText(p.p_holm)}) → `,
    h("b", null, VERDICT_TEXT[p.verdict] ?? p.verdict), ".");
}

function drawEvalDet() {
  const now = detCurrent();
  $("sec-eval").toggleAttribute("data-off", !now);
  if (!now) {
    $("eval").replaceChildren();
    $("eval-state").replaceChildren(h("span", {class: "ta-muted"}, "셋업이 정해지면 측정 기록으로 평가합니다"));
    return;
  }
  const e = st.evaluation;
  const env = now.env ?? st.det.env ?? {};
  const where = env.gpu ? String(env.gpu).replace("Tesla ", "").replace("NVIDIA ", "") : `CPU ${String(env.host ?? "").split(".")[0]}`;
  $("eval-state").replaceChildren(...(e ? [chip(`기록된 측정 · ${where} · git ${String(env.git_sha ?? "").slice(0, 7)}`, "ok")]
    : [chip("측정 기록 대조", "plain")]));
  if (!e) {
    const go = h("button", {type: "button", class: "ta-btn primary", id: "run-eval"}, "평가 결과 불러오기");
    go.addEventListener("click", () => {
      st.evaluation = {kind: "record"};
      drawEvalDet();
      drawOutDet();
      reveal($("sec-eval"));
    });
    $("eval-state").append(go);
    $("eval").replaceChildren(h("p", {class: "ta-muted"}, KD().evalIntro));
    return;
  }
  $("eval").replaceChildren(...detRecordView(now));
}

function drawOutDet() {
  const now = detCurrent();
  const on = Boolean(now && st.evaluation);
  $("sec-out").toggleAttribute("data-off", !on);
  if (!on) {
    $("out").replaceChildren();
    $("out-state").replaceChildren(h("span", {class: "ta-muted"}, "평가 결과를 보면 압축 크기와 내보내기가 열립니다"));
    return;
  }
  const orig = st.det.model.original_bytes;
  $("out-state").replaceChildren(chip(`파일 ${MB(now.bytes)} · 원본의 ${fmt(100 * now.bytes / orig, 0)}%`, "plain"));
  const bar = (label, sub, bytes, cls) => h("div", {class: cls}, h("span", {class: "name"}, label, h("small", null, sub)),
    h("span", {class: "track"}, h("i", {class: "linear", style: `width: ${(100 * bytes / orig).toFixed(2)}%`, title: MB(bytes)})),
    h("span", {class: "num"}, MB(bytes)));
  const file = {file: `${st.det.model.id}_${now.id}.json`, yaml: `${JSON.stringify({model: st.det.model, quantization: st.det.quantization, setup: now.id,
    label: now.label, bits: now.bits_fraction, methods: now.methods, metrics: now.metrics, env: now.env ?? null}, null, 1)}\n`};
  const save = h("button", {type: "button", class: "ta-btn primary", id: "export"}, "층별 배정 파일 받기 (.json)");
  save.addEventListener("click", () => download(file, "application/json"));
  $("out").replaceChildren(h("div", {class: "ta-outgrid"},
    h("div", {class: "ta-block", id: "size"},
      blockHead("압축 크기", help("압축 크기 (계산값)", st.det.model.size_rule), h("span", null, "계산값")),
      h("div", {class: "ta-size"}, bar("원본", KD().origin, orig, "orig"), bar("이 셋업", `${KD().unit} ${fmt(now.bits, 2)}비트`, now.bytes, "cur")),
      h("p", {class: "ta-gain", id: "size-note"}, "파일 전체 ", h("b", null, `${MB(orig)} → ${MB(now.bytes)} (${fmt(100 * now.bytes / orig, 0)}%)`), ".")),
    h("div", {class: "ta-block", id: "export-block"},
      blockHead("내보내기", h("span", null, "이 셋업의 층별 비트와 측정값")),
      h("div", {class: "ta-row"}, save, h("span", {class: "ta-muted mono"}, file.file)),
      h("p", {class: "ta-muted"}, `${KD().unit} 층마다 고른 방법, 크기, ${KD().short}, 실행 환경을 담습니다.`))),
    isGen() ? h("p", {class: "ta-muted", id: "package"}, "이 모델의 저비트 파일(모델 패키지)은 아직 만들지 않았습니다. 위 크기는 계산값입니다.")
      : onnxView(st.packages?.onnx?.yolo11n?.[now.id] ?? null, now));
}

// the setup as an ONNX file with integer weights (scripts/quantizer/yolo_tune.py onnx), run by onnxruntime
function onnxView(pkg, now) {
  if (!pkg) return h("p", {class: "ta-muted", id: "package"}, "이 셋업은 ONNX 파일 기록이 없습니다.");
  const kinds = Object.entries(pkg.kinds).map(([k, n]) => `${k.toUpperCase()} ${n}층`).join(" · ");
  const ap = pkg.metrics.AP_50_95;
  const torchAp = pkg.pytorch_metrics?.AP_50_95;
  const boxes = Object.values(pkg.demo_boxes);
  const same = boxes.filter((b) => b.onnx === b.pytorch && b.same_classes).length;
  const name = `${now.id}.onnx`;
  const link = pkg.file ? h("a", {class: "ta-btn primary", id: "onnx-download", href: `demo/autotune/${pkg.file}`, download: name}, "ONNX 파일 받기")
    : h("span", {class: "ta-muted"}, "파일은 저장소에 넣지 않았습니다 (기록만)");
  return h("div", {class: "ta-block", id: "package"},
    blockHead("모델 패키지 · ONNX", help("ONNX 패키지 (실제 파일)", "Conv 가중치를 정수(INT8 · INT4)와 출력 채널마다 FP32 스케일로 저장하고 DequantizeLinear 로 되돌리는 ONNX 파일입니다 (opset 21). onnxruntime 으로 실행해 같은 COCO 사진에서 PyTorch 결과와 비교했습니다."),
      chip(`실제 파일 · onnxruntime ${pkg.env.onnxruntime}`, "ok")),
    h("div", {class: "ta-row"}, link, h("span", {class: "ta-muted mono"}, `${name} · ${MB(pkg.bytes)}`)),
    h("ul", {class: "ta-checks", id: "package-checks"},
      h("li", null, kinds ? `정수 코드 × 스케일 = PyTorch 가중치 (파일을 쓸 때 층마다 대조) · ${kinds}` : "가중치 FP32 그대로 (양자화한 층 없음)"),
      h("li", null, `COCO mAP50-95 (${pkg.images.toLocaleString("en-US")}장): onnxruntime ${AP(ap)} · PyTorch ${AP(torchAp)}`),
      h("li", null, `사진 ${boxes.length}장의 상자 수 · 클래스가 PyTorch 와 같음: ${same}/${boxes.length}`)),
    h("pre", {class: "ta-cmd"}, `python -c "import onnxruntime as ort; s = ort.InferenceSession('${name}'); print([i.name for i in s.get_inputs()])"`),
    h("p", {class: "ta-muted"}, `입력 images [N, 3, H, W] (0–1, 32의 배수), 출력 output0 [N, 84, 앵커] — ultralytics 와 같은 후처리(NMS)를 씁니다. ${now.label}.`,
      kinds ? " 파일이 위 계산값보다 조금 큰 것은 스케일 · 바이어스를 FP32 로 두고(계산은 FP16) 영점 · 그래프 정의를 함께 담기 때문입니다." : ""));
}

function drawAll() {
  // the step's subtitle says who recommends: the AI where a tune session ran on the model, else the knapsack alone
  document.querySelector("#sec-tune .ta-sub").textContent = st.model && !st.bundle?.session ? "층별 자동 배정" : "AI 추천";
  drawServer();
  drawModel();
  drawTune();
  drawEdit();
  drawEval();
  drawOut();
}

async function start() {
  try {
    st.models = (await getJSON("demo/autotune/models.json")).models;
    st.packages = await getJSON("demo/autotune/packages.json").catch(() => null);  // step 5's measured files
  } catch {
    try {  // a bundle directory without a model list: its one bundle
      const bundle = await getJSON("demo/autotune/bundle.json");
      st.models = [{id: bundle.table.meta.model_id, kind: "llm", bundle: "bundle.json", live: true}];
    } catch (error) {
      $("main").replaceChildren(h("p", {class: "ta-err"}, `측정 기록 번들을 읽지 못했습니다: ${error.message}`));
      return;
    }
  }
  st.health = await fetch("api/health").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  $("model").replaceChildren(...st.models.map((m) => h("option", {value: m.id}, m.label ?? m.id)));
  $("model-form").addEventListener("submit", (event) => {
    event.preventDefault();
    loadModel();
  });
  drawServer();
  drawAll();
  document.body.dataset.solver = "idle";
  document.body.dataset.ready = "yes";
  await restoreState();
}

start();
