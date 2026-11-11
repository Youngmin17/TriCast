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
import {parseBits, solve, summary} from "./mckp.js";
import {h, withTip} from "./ui.js";

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
const st = {health: null, bundle: null, model: null, budget: null, plan: null, planError: null, computing: false, planTicket: 0,
  uniform: null, option: null, recipe: null, recipeError: null, recipeSeq: 0, lm: new Set(), limit: 200, evaluation: null};
const gpuLive = () => st.health?.mode === "live" && st.health?.device === "cuda";

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

// the knapsack setups GPU-measured without constraints, smallest first: {budget, bits, row}
function measuredBudgets() {
  const rows = st.bundle.solutions.filter((row) => row.ppl != null && (row.kv ?? "none") === "none");
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
  if (base.id === session.start) {
    return session.rounds.flatMap((round) => round.proposals ?? [])
      .filter((p) => p.status === "accepted" && p.ppl != null && sameGrid(p.grid, base.grid))
      .map((p) => ({kind: "proposal", id: p.name, ppl: p.ppl, kl: p.kl, options: p.options_changed ?? {},
        record: session.records?.[p.ppl_id] ?? base.ppl_record}));
  }
  const row = transfer.find((item) => item.base === base.id);
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
const solver = {worker: undefined, busy: false, current: null, user: null, warm: [], seq: 0, cache: new Map(), jobs: new Map()};

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
  let job;
  const promise = new Promise((resolve) => {
    job = {id: ++solver.seq, table: t, budget, done: (data) => {
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
  const body = now.kind === "uniform" ? {grid: now.grid} : now.option ? {[now.kind]: now.id} : {grid: now.grid, budget_bits: st.plan.budget};
  try {
    const recipe = await postJSON("api/autotune/recipe", body);
    if (seq === st.recipeSeq) st.recipe = recipe;
  } catch (error) {
    if (seq === st.recipeSeq) st.recipeError = error.message;
  }
  if (seq === st.recipeSeq) {
    drawEdit();
    drawEval();
  }
}

function download(recipe) {
  const a = h("a", {href: URL.createObjectURL(new Blob([recipe.yaml], {type: "text/yaml"})), download: recipe.file});
  document.body.append(a);
  a.click();
  a.remove();
}

// ---------- actions ----------
function loadModel() {
  st.model = table().meta.model_id;
  drawAll();
  saveState();
  warmUp();
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
  if (st.plan && st.uniform) params.set("uniform", st.uniform);
  else if (st.plan && edited()) params.set("option", st.option ?? "default");
  const query = params.toString();
  window.history.replaceState(null, "", `${location.pathname}${query ? `#${query}` : ""}`);
}

async function restoreState() {
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get("model") !== table().meta.model_id) return;
  st.model = table().meta.model_id;
  drawAll();
  warmUp();
  const size = params.get("size");
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
    st.evaluation = {kind: "job", status: "queued", started: Date.now()};
    drawEval();
    try {
      const tasks = ["ppl", ...st.lm];
      const job = await postJSON("api/quantizer/jobs", {name: st.recipe.name, yaml: st.recipe.yaml, tasks,
        limit: st.lm.size ? st.limit : null});
      st.evaluation = {...st.evaluation, id: job.id, status: job.status};
      poll(job.id);
    } catch (error) {
      st.evaluation = {kind: "job", status: "error", error: error.message};
    }
    drawEval();
    reveal($("sec-eval"));
    return;
  }
  st.evaluation = {kind: now.ppl != null ? "record" : "none"};
  drawEval();
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
  $("server").textContent = `${gpuLive() ? "GPU 서버 (CUDA) — 평가를 실제로 돌립니다" : "이 PC 에는 GPU 가 없습니다 — 평가는 GPU 측정 기록으로"}`;
}

// 1 model: one row — the pick and the load button, then what was loaded
function drawModel() {
  const meta = table().meta;
  $("model-state").replaceChildren(st.model ? chip("준비됨", "ok") : "");
  $("load").classList.toggle("primary", !st.model);
  if (st.model) {
    const fact = (k, v) => h("div", null, h("dt", null, k), h("dd", null, v));
    $("model-info").replaceChildren(h("dl", {class: "ta-facts", title: `revision ${meta.revision} · 민감도 표: ${gpuName()} · ${meta.source?.dataset ?? ""} ${meta.windows}창`},
      fact("revision", meta.revision.slice(0, 7)),
      fact("Linear", `${table().layers.length}개`),
      fact("방법", `${table().methods.length}종`),
      fact("민감도 표", gpuName().split("-")[0])));
  } else {
    $("model-info").replaceChildren(h("p", {class: "ta-muted"}, "GPU 로 층별 민감도를 재 둔 모델입니다."));
  }
  document.title = `${st.model ? `${st.model.split("/").pop()}${st.plan ? ` ${pct(current().bits)}` : ""} · ` : ""}TriCast`;
}

// 2 Auto-tune: the target size is the one thing asked; the answer is the recommended setup and how it was reached
function drawTune() {
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
    blockHead("목표 크기", help("크기", "가중치 하나를 저장하는 평균 비트 수로 정한 모델 크기입니다. 원본(BF16)은 16비트 = 100%, 4비트면 25% 입니다."),
      h("span", null, "원본 대비 · 아래 숫자는 그 크기 추천 셋업의 GPU 측정 PPL")),
    h("div", {class: "ta-presets", role: "group", "aria-label": `목표 크기 — GPU 로 측정된 ${sizes.length}개`}, sizes))];
  if (!st.budget) parts.push(h("p", {class: "ta-muted"}, "목표 크기를 고르면 그 안에서 층마다 방법을 고르고, AI 가 GPU 로 재서 찾은 세부 설정을 더해 추천합니다."));
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
    h("div", {class: "ta-reco-head"}, h("b", null, "AI 추천"), h("span", {class: "ppl"}, `PPL ${fmt(ppl)}`),
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

// the assignment as 28 blocks × 7 layer kinds, colored by bits; a cell says its layer's method on hover
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
      h("div", {class: "ta-grid", role: "img", "aria-label": `블록 ${t.blocks} × 층 종류 ${t.kinds.length} 의 방법 배정: ${used.map(([m, n]) => `${methodLabel(m)} ${n}층`).join(", ")}`}, cells),
      h("div", {class: "ta-axis"}, [0, 7, 14, 21, t.blocks - 1].map((b) => h("span", null, `블록 ${b}`)))),
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
function ladder(rows, nativePpl, size) {
  const max = Math.max(...rows.map((r) => r.ppl - nativePpl), 1e-9);
  return h("div", {class: "ta-ladder", id: "ladder"},
    h("div", null, h("span", {class: "th"}, `크기 ${size} 안에서`), h("span", {class: "th"}, "원본에서 벌어진 정도 (PPL 증가)"),
      h("span", {class: "th num"}, "증가"), h("span", {class: "th num"}, "PPL")),
    rows.map((r) => h("div", {class: `${r.series}${r.cur ? " cur" : ""}`},
      h("span", {class: "name"}, r.series === "base" ? null : h("i", {class: `ta-sw ${r.series}`}), r.name,
        r.sub ? h("small", {title: r.title ?? null}, r.sub) : null, r.help ?? null),
      h("span", {class: "track"}, r.series === "base" ? null
        : h("i", {style: `width: ${(100 * Math.max(0, r.ppl - nativePpl) / max).toFixed(1)}%`})),
      h("span", {class: "num"}, r.series === "base" ? "기준" : signed(r.ppl - nativePpl)),
      h("span", {class: "num"}, pplText(r.ppl)))));
}

function recordView(now) {
  const {native} = st.bundle;
  const base = st.plan.recorded;
  const uniform = uniformsWithin(st.plan.budget)[0] ?? null;
  const rows = [{series: "base", name: "원본", sub: "BF16 · 100%", ppl: native.ppl}];
  if (now.kind === "uniform" && now.method !== uniform?.method) {
    rows.push({series: "uni", name: "직접 편집", sub: `전체 ${methodLabel(now.method)} · ${pct(now.bits)}`, title: now.method, ppl: now.ppl, cur: true});
  }
  if (uniform) {
    rows.push({series: "uni", name: "한 방법 최선", sub: `${methodLabel(uniform.method)} · ${pct(uniform.bits)}`, title: uniform.method, ppl: uniform.ppl,
      cur: now.kind === "uniform" && now.method === uniform.method,
      help: help("한 방법", "모든 층에 같은 양자화 방법을 쓰는 보통 방식입니다. 이 크기 안에서 PPL 이 가장 낮은 것과 비교합니다.")});
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
    : gain != null ? ["같은 크기의 한 방법 최선보다 원본에서 벌어진 정도가 ", h("b", null, `${fmt(Math.abs(gain), 0)}% ${gain >= 0 ? "작습니다" : "큽니다"}`), "."] : null;
  return [
    h("div", {class: "ta-result", id: "eval-result"},
      h("div", {class: "ta-hero"},
        h("span", null, "WikiText-2 test PPL", help("PPL (퍼플렉서티)", `모델이 다음 단어를 얼마나 헷갈리는지 잰 값으로, 낮을수록 좋습니다. 원본(BF16)은 ${fmt(native.ppl)} 입니다.`)),
        h("b", null, pplText(now.ppl)), h("small", null, `원본 ${fmt(native.ppl)} 대비 ${signed(loss)}`)),
      h("div", {class: "ta-col"}, ladder(rows, native.ppl, pct(st.plan.bits)),
        verdict ? h("p", {class: "ta-gain", id: "gain"}, verdict) : null)),
    h("p", {class: "ta-muted"}, `PPL 은 셋업을 고르는 데 쓰지 않은 데이터 (${native.n_tokens?.toLocaleString("en-US") ?? "—"} 토큰) 로 쟀습니다. ` +
      "lm-eval 과제별 정확도는 이 레시피의 기록이 없습니다.")];
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
function drawEval() {
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

function drawAll() {
  drawModel();
  drawTune();
  drawEdit();
  drawEval();
}

async function start() {
  try {
    st.bundle = await fetch("demo/autotune/bundle.json").then((r) => {
      if (!r.ok) throw new Error(`demo/autotune/bundle.json: HTTP ${r.status}`);
      return r.json();
    });
  } catch (error) {
    $("main").replaceChildren(h("p", {class: "ta-err"}, `측정 기록 번들을 읽지 못했습니다: ${error.message}`));
    return;
  }
  st.health = await fetch("api/health").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const meta = table().meta;
  $("model").replaceChildren(h("option", {value: meta.model_id}, meta.model_id));
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
