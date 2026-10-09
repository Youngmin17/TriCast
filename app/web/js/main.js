// TriCast Studio: state, selection rules and rendering.
import {canonical, mmaKey} from "./algo.js";
import {findRecorded, follow, inputOf, listRuns, loadRun, serverResources, submit} from "./api.js";
import {SIZES, runLab} from "./lab.js";
import {renderRail} from "./rail.js";
import {renderScope} from "./results.js";
import {detectRuntimes, modeSwitch, resourcePanel} from "./runtime.js";
import {h} from "./ui.js";

const rail = document.getElementById("rail");
const scope = document.getElementById("scope");
const status = document.getElementById("status");

const state = {
  sel: {task: "llm.generate", model: null, mma: {algorithm: "cofda"}, format: "fp8_tensor", baseline: "native",
    prompt: "", maxTokens: 48, image: null},
  run: null, busy: false, error: null, tab: "compare", diffOnly: false, advanced: false, spaceMetric: null,
  runtime: "offline", lab: {size: "m", result: null, stage: null},
};
let ctx = null;
let runtimes = null;
let labCtx = null;
let loadToken = 0;

const demo = () => ctx.mode === "demo";
const lab = () => ctx.mode === "webgpu";
const inputKey = (sel) => (sel.task === "llm.generate" ? sel.prompt : sel.image);
const runInput = (run) => run.input?.prompt ?? run.input?.image;
const algorithms = () => ctx.catalog.algorithms;
const kDomains = (format) => ctx.catalog.formats.find((f) => (f.id ?? null) === (format ?? null))?.k_domains ?? [];

function inputRuns(sel = state.sel) {
  return ctx.runs.filter((run) => run.task === sel.task && run.model === sel.model && runInput(run) === inputKey(sel));
}

// Lower is closer: same baseline and format first, then the same algorithm, then nearby parameter values.
function distance(run, sel) {
  let d = 0;
  if ((run.baseline || "native") !== sel.baseline) d += 100;
  if ((run.format ?? null) !== (sel.format ?? null)) d += 50;
  if (run.mma.algorithm !== sel.mma.algorithm) d += 1000;
  for (const [key, value] of Object.entries(sel.mma)) {
    const other = run.mma[key];
    if (key === "algorithm" || other === value) continue;
    d += typeof value === "number" && typeof other === "number"
      ? 1 + (10 * Math.abs(other - value)) / Math.max(other, value) : 10;
  }
  return d;
}

// Demo mode: move to the recorded run closest to the selection, keeping `keep(run)` true when possible.
function snap(keep = () => true) {
  const runs = inputRuns();
  if (!runs.length) return;
  const pool = runs.filter(keep).length ? runs.filter(keep) : runs;
  const best = pool.reduce((a, b) => (distance(b, state.sel) < distance(a, state.sel) ? b : a));
  state.sel.mma = {...best.mma};
  state.sel.format = best.format ?? null;
  state.sel.baseline = best.baseline || "native";
}

// Live mode: keep the selection inside MMASpec's rules and the algorithm's formats.
function repair() {
  const sel = state.sel;
  sel.mma = canonical(sel.mma, algorithms());
  const algo = algorithms().find((item) => item.id === sel.mma.algorithm);
  if (algo && !algo.formats.includes(sel.format)) sel.format = algo.formats[0] ?? null;
  const fits = (span) => kDomains(sel.format).every((domain) => domain % span === 0);
  const span = sel.mma.promote_interval;
  if (span && (sel.mma.c_mode === "decoupled" || span % sel.mma.chunk_size || !fits(span))) sel.mma.promote_interval = 0;
  if (sel.mma.algorithm === "gdfs") {
    const sizes = algo.params.find((param) => param.key === "group_size")?.options ?? [];
    if (!fits(sel.mma.group_size)) sel.mma.group_size = [...sizes].reverse().find(fits) ?? sel.mma.group_size;
    const groups = sel.mma.k_tile / sel.mma.group_size;
    if (!Number.isInteger(groups) || groups < 1 || groups > 8) sel.mma.k_tile = sel.mma.group_size * 4;
  }
}

function firstInput(sel) {
  const run = ctx.runs.find((item) => item.task === sel.task && item.model === sel.model && item.input);
  if (!run) return;
  if (run.input.prompt != null) sel.prompt = run.input.prompt;
  if (run.input.image != null) sel.image = run.input.image;
}

function enterLab() {
  Object.assign(state.sel, {task: "kernel.gemm", model: "random", format: "fp8_tensor", baseline: "fp64_exact"});
  repairLab();
}

function repairLab() {
  const sel = state.sel;
  sel.mma = canonical({...sel.mma, algorithm: "cofda"}, algorithms());
  sel.mma.promote_interval = 0;
  sel.format = "fp8_tensor";
  sel.baseline = "fp64_exact";
}

function changed(keep) {
  state.error = null;
  if (demo()) snap(keep);
  else if (lab()) {
    repairLab();
    state.lab.result = null;
  } else {
    repair();
    state.run = null;
  }
  refresh();
  if (demo()) openRecorded();
}

const act = {
  pick(patch) {
    const sel = state.sel;
    Object.assign(sel, patch);
    if ("task" in patch) {
      const models = ctx.catalog.models.filter((m) => m.tasks.includes(sel.task));
      sel.model = (models.find((m) => act.available("model", m.id)) || models[0])?.id ?? null;
    }
    if ("task" in patch || "model" in patch) firstInput(sel);
    const key = Object.keys(patch)[0];
    const keep = key === "format" ? (run) => (run.format ?? null) === sel.format
      : key === "baseline" ? (run) => (run.baseline || "native") === sel.baseline : () => true;
    changed(keep);
  },
  edit(patch) {
    Object.assign(state.sel, patch);
    const button = rail.querySelector(".run-btn");
    if (button) button.disabled = state.busy || !(inputKey(state.sel) || "").trim();
  },
  setAlgorithm(id) {
    // Start from the new algorithm's own defaults: its parameters mean different things (CoFDA F13 is not GDFS F13).
    state.sel.mma = canonical({algorithm: id}, algorithms());
    changed((run) => run.mma.algorithm === id);
  },
  setParam(key, value) {
    state.sel.mma = canonical({...state.sel.mma, [key]: value}, algorithms());
    changed((run) => run.mma.algorithm === state.sel.mma.algorithm && run.mma[key] === value);
  },
  usePreset(preset) {
    state.sel.mma = {...preset.mma};
    state.sel.format = preset.format ?? null;
    changed((run) => mmaKey(run.mma) === mmaKey(preset.mma) && (run.format ?? null) === (preset.format ?? null));
  },
  openRow(run) {
    state.tab = "compare";
    act.openCase(run);
  },
  selectMMA(mma) {
    state.sel.mma = {...mma};
    changed((run) => mmaKey(run.mma) === mmaKey(mma));
  },
  presetAvailable(preset) {
    if (lab()) return preset.mma.algorithm === "cofda" && !preset.mma.promote_interval;
    return !demo() || inputRuns().some((run) => mmaKey(run.mma) === mmaKey(preset.mma)
      && (run.format ?? null) === (preset.format ?? null));
  },
  available(key, value) {
    if (lab()) {
      if (key === "algorithm") return value === "cofda";
      if (key === "promote_interval") return value === 0;
      return true;
    }
    if (!demo()) {
      if (key !== "format") return true;
      return algorithms().find((a) => a.id === state.sel.mma.algorithm)?.formats.includes(value) ?? false;
    }
    if (key === "task") return ctx.runs.some((run) => run.task === value);
    if (key === "model") return ctx.runs.some((run) => run.task === state.sel.task && run.model === value);
    const runs = inputRuns();
    if (key === "algorithm") return runs.some((run) => run.mma.algorithm === value);
    if (key === "format") return runs.some((run) => (run.format ?? null) === value && run.mma.algorithm === state.sel.mma.algorithm);
    if (key === "baseline") return runs.some((run) => (run.baseline || "native") === value);
    return runs.some((run) => run.mma.algorithm === state.sel.mma.algorithm && run.mma[key] === value);
  },
  // Demo mode: the values of one integer parameter that were recorded with everything else unchanged.
  recordedValues(key) {
    if (!demo()) return null;
    const sel = state.sel;
    const same = inputRuns().filter((run) => run.mma.algorithm === sel.mma.algorithm
      && (run.format ?? null) === (sel.format ?? null) && (run.baseline || "native") === sel.baseline);
    const strict = same.filter((run) => Object.keys(sel.mma).every((k) => k === key || run.mma[k] === sel.mma[k]));
    return [...new Set((strict.length ? strict : same).map((run) => run.mma[key]))].sort((a, b) => a - b);
  },
  recorded: () => Boolean(findRecorded(ctx, state.sel)),
  openCase(run) {
    Object.assign(state.sel, {task: run.task, model: run.model, mma: {...run.mma}, format: run.format ?? null,
      baseline: run.baseline || "native"});
    if (run.input.prompt != null) state.sel.prompt = run.input.prompt;
    if (run.input.max_new_tokens != null) state.sel.maxTokens = run.input.max_new_tokens;
    if (run.input.image != null) state.sel.image = run.input.image;
    if (demo()) snap((item) => mmaKey(item.mma) === mmaKey(run.mma));
    refresh();
    openRecorded(run.id);
  },
  setTab(tab) {
    state.tab = tab;
    renderScope(scope, state, ctx, act);
  },
  setDiffOnly(value) {
    state.diffOnly = value;
    renderScope(scope, state, ctx, act);
  },
  setAdvanced(value) {
    state.advanced = value;
    renderRail(rail, state, ctx, act);
  },
  setSpaceMetric(key) {
    state.spaceMetric = key;
    renderScope(scope, state, ctx, act);
  },
  upload(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => act.pick({image: reader.result});
    reader.readAsDataURL(file);
  },
  async run() {
    if (lab()) return act.runLab();
    const sel = state.sel;
    const request = {task: sel.task, model: sel.model, mma: sel.mma, format: sel.format, baseline: sel.baseline,
      input: inputOf(sel)};
    state.busy = true;
    state.error = null;
    state.run = {status: "queued", progress: 0, stage: null};
    ++loadToken;  // a recorded run still loading must not replace this one
    refresh();
    try {
      state.run = await submit(request, (update) => {
        state.run = update;
        renderScope(scope, state, ctx, act);
      });
    } catch (error) {
      state.run = null;
      state.error = error.message;
    }
    state.busy = false;
    // The result is already shown; the run list and the server panel refresh on a best-effort basis.
    ctx.runs = await listRuns().catch(() => ctx.runs);
    runtimes.server.resources = await serverResources().catch(() => runtimes.server.resources);
    renderStatus();
    refresh();
  },
};

// While a run is in flight the selection stays as it was sent (refresh also disables the rail's controls).
for (const name of ["pick", "edit", "setAlgorithm", "setParam", "usePreset", "openRow", "selectMMA", "openCase",
  "upload"]) {
  const call = act[name];
  act[name] = (...args) => (state.busy ? undefined : call(...args));
}

act.setRuntime = (id) => {
  if (state.busy || !runtimes[id]?.ok || state.runtime === id) return;
  ++loadToken;  // a recorded run still loading belongs to the old mode
  state.runtime = id;
  state.run = null;
  state.error = null;
  state.tab = "compare";
  ctx = id === "webgpu" ? labCtx : runtimes[id].ctx;
  renderStatus();
  if (id === "offline") {
    const start = firstCase();
    if (start) {
      act.openCase(start);
      return;
    }
  } else if (id === "webgpu") {
    enterLab();
  } else {
    const models = ctx.catalog.models.filter((m) => m.tasks.includes(state.sel.task));
    if (!ctx.catalog.baselines.some((b) => b.id === state.sel.baseline)) state.sel.baseline = "native";
    if (!ctx.catalog.algorithms.some((a) => a.id === state.sel.mma.algorithm)) {
      state.sel.mma = canonical({algorithm: ctx.catalog.algorithms[0].id}, algorithms());
    }
    if (!models.some((m) => m.id === state.sel.model)) {
      state.sel.task = "llm.generate";
      state.sel.model = ctx.catalog.models.find((m) => m.tasks.includes("llm.generate"))?.id ?? null;
    }
    repair();
  }
  refresh();
};

act.setLabSize = (id) => {
  if (state.busy) return;
  state.lab.size = id;
  state.lab.result = null;
  refresh();
};

act.envPanel = () => resourcePanel(state, runtimes);

act.runLab = async () => {
  const source = ctx.catalog.models.find((m) => m.id === state.sel.model);
  const size = SIZES.find((item) => item.id === state.lab.size) || SIZES[1];
  state.busy = true;
  state.error = null;
  state.lab.result = null;
  refresh();
  try {
    state.lab.result = await runLab({mma: state.sel.mma, presetKey: mmaKey(state.sel.mma)}, source, size, (stage) => {
      state.lab.stage = stage;
      renderScope(scope, state, ctx, act);
    });
  } catch (error) {
    state.error = `브라우저 GPU 실행에 실패했습니다: ${error.message}`;
  }
  state.busy = false;
  state.lab.stage = null;
  refresh();
};

act.labProbe = () => runtimes.webgpu.probe;

async function labCatalog(base) {
  let packs = [];
  try {
    const response = await fetch("demo/webgpu/operands/index.json");
    if (response.ok) packs = (await response.json()).packs || [];
  } catch {
    packs = [];
  }
  return {
    ...base,
    tasks: [{id: "kernel.gemm", kind: "kernel", label: "연산 실험실", description: "FP8 행렬곱을 설계한 산술로 브라우저 GPU에서 실행"}],
    models: [{id: "random", label: "무작위 FP8 행렬", tasks: ["kernel.gemm"], support: "experimental",
      note: "가우시안 값을 FP8 E4M3로 반올림한 피연산자"},
    ...packs.map((pack) => ({id: pack.id, label: pack.label, tasks: ["kernel.gemm"], support: "full_eval",
      note: `${pack.model} · ${pack.layer} · ${pack.M}×${pack.N}×${pack.K} (서버 TriCast 결과와 대조)`, pack}))],
    formats: [{id: "fp8_tensor", label: "FP8 E4M3 · 텐서", bits: 8, note: ""}],
    baselines: [{id: "fp64_exact", label: "FP64 정확 누산", description: "같은 FP8 피연산자를 FP64로 누산한 값과 ULP로 비교합니다."}],
  };
}

// Opens run `id`, else the recorded run that matches the selection; an unfinished live run is followed.
async function openRecorded(id) {
  const entryId = id ?? findRecorded(ctx, state.sel)?.id;
  const token = ++loadToken;
  if (!entryId) {
    state.run = null;
    renderScope(scope, state, ctx, act);
    return;
  }
  const show = (run) => {
    state.run = run;
    state.error = null;
    renderScope(scope, state, ctx, act);
  };
  try {
    const run = await loadRun(ctx, entryId);
    if (token !== loadToken) return;
    show(run);
    if (ctx.source === "server" && (run.status === "queued" || run.status === "running")) {
      await follow(entryId, show, () => token === loadToken);
    }
  } catch (error) {
    if (token !== loadToken) return;
    state.run = null;
    state.error = `기록을 불러오지 못했습니다: ${error.message}`;
    renderScope(scope, state, ctx, act);
  }
}

function renderStatus() {
  status.replaceChildren(modeSwitch(state, runtimes, act));
}

function refresh() {
  renderRail(rail, state, ctx, act);
  if (state.busy) {
    for (const control of rail.querySelectorAll("button:not(.run-btn), input, textarea, select")) control.disabled = true;
  }
  renderScope(scope, state, ctx, act);
}

const ANCHORS = {"#llm": "llm.generate", "#detect": "vision.detect", "#classify": "vision.classify"};

// Opens on the Hopper-like preset of the first recorded input; #llm, #detect, #classify and #webgpu pick the view.
function firstCase() {
  const task = ANCHORS[location.hash] || "llm.generate";
  const native = (run) => run.task === task && (run.baseline || "native") === "native";
  const hopper = ctx.catalog.presets?.find((preset) => preset.id === "hopper");
  const start = hopper && ctx.runs.find((run) => native(run) && mmaKey(run.mma) === mmaKey(hopper.mma)
    && (run.format ?? null) === (hopper.format ?? null));
  return start || ctx.runs.find(native) || ctx.runs[0] || null;
}

async function boot() {
  runtimes = await detectRuntimes();
  const base = runtimes.offline.ctx?.catalog ?? runtimes.server.ctx?.catalog;
  if (base && runtimes.webgpu.ok) labCtx = {mode: "webgpu", source: "browser", catalog: await labCatalog(base), runs: []};
  if (runtimes.webgpu.ok && !labCtx) runtimes.webgpu = {ok: false, reason: "알고리즘 목록을 읽지 못해 실험실을 열 수 없습니다"};
  state.runtime = location.hash === "#webgpu" && runtimes.webgpu.ok ? "webgpu"
    : runtimes.server.ok ? "server" : runtimes.offline.ok ? "offline" : runtimes.webgpu.ok ? "webgpu" : null;
  if (!state.runtime) {
    scope.replaceChildren(h("section", {class: "state"}, h("h2", null, "실행할 수 있는 모드가 없습니다"),
      h("p", null, `예시 데이터: ${runtimes.offline.reason} / 서버: ${runtimes.server.reason} / 브라우저 GPU: ${runtimes.webgpu.reason}`)));
    return;
  }
  ctx = state.runtime === "webgpu" ? labCtx : runtimes[state.runtime].ctx;
  renderStatus();
  if (lab()) {
    enterLab();
    refresh();
    return;
  }
  const start = firstCase();
  if (start) {
    act.openCase(start);
    return;
  }
  // Nothing recorded yet (a fresh server): start from the Hopper preset, else the first algorithm's defaults.
  const hopper = ctx.catalog.presets?.find((preset) => preset.id === "hopper");
  const algo = algorithms()[0];
  state.sel.mma = hopper ? {...hopper.mma} : canonical({algorithm: algo.id}, algorithms());
  state.sel.format = hopper ? hopper.format ?? null : algo.formats[0] ?? null;
  state.sel.model = ctx.catalog.models.find((m) => m.tasks.includes(state.sel.task))?.id ?? null;
  refresh();
}

boot();
