// Auto-tune page (autotune.html): pick a size budget for the Linear weights of the model, and the exact
// multiple-choice knapsack over the per-layer KL table (js/mckp.js, the same DP as app/autotune.solve_mckp)
// assigns a quantization method to every Linear layer of the block x kind floorplan, in the browser. The recorded
// tune-loop session then shows how the AI agent (Claude) edits that starting point: evidence-cited proposals, the
// deterministic checks each one passed or failed, and the GPU measurements that decide. "Claude 에게 새 제안 받기"
// asks the server for one more round (POST /api/autotune/propose, app/autotune_live.py; checked, not measured).
// A sentence typed above the budget ("요청 해석", POST /api/autotune/parse, app/tune_request.py) becomes a checked
// request whose budget and constraints (BF16-kept blocks, excluded or allowed methods) the knapsack then solves in.
// Data: demo/autotune/bundle.json (app/export_autotune_demo.py). Every measured value shown comes from it.
import {SolveError, parseBits, problem, solve, summary, weightBits} from "./mckp.js";
import {h, hideTip, s, showTip} from "./ui.js";

const SCHEMA = "tricast.autotune_demo/1";
const DATA = "demo/autotune/bundle.json";
const EXPORT = "python -m app.export_autotune_demo --table runs/<실행>/table.json --out app/web/demo/autotune";
const PROPOSE = "api/autotune/propose";
const PARSE = "api/autotune/parse";
const AGENT_SERVER = "python -m app.server --demo --agent <loop.json> 으로 띄운 주소에서 다시 시도하세요.";
const MAX_BITS = 8.5;
const UNITS = 128;          // slider positions per bit: the slider moves in 1/128 bit
const DEBOUNCE_MS = 120;
const STAGGER_MS = 240;     // the last changed cell starts this late; each fade is 140 ms (autotune.css), < 400 ms
const SHORT_KIND = {q_proj: "q", k_proj: "k", v_proj: "v", o_proj: "o", gate_proj: "gate", up_proj: "up", down_proj: "down"};

// ---------- method families: fill color by family, the digit in the code carries the bit width ----------
const FAMILY_RULES = [
  [/^gptq_int(\d)$/, "gptq", "GPTQ", (m) => `G${m[1]}`],
  [/^rht_gptq_int(\d)$/, "rht", "RHT+GPTQ", (m) => `R${m[1]}`],
  [/^awq_clip_int(\d)$/, "clip", "AWQ+clip", (m) => `C${m[1]}`],
  [/^awq_int(\d)$/, "awq", "AWQ", (m) => `A${m[1]}`],
  [/^hqq_int(\d)$/, "hqq", "HQQ", (m) => `H${m[1]}`],
  [/^nf4$/, "nf4", "NF4", () => "NF"],
  [/^int8_w$/, "int8w", "INT8 W8A16", () => "I8"],
  [/^bf16$/, "bf16", "BF16", () => "BF"],
];
// Weight × activation formats of older tables share the "other" fill; these codes keep them apart.
const OTHER_CODES = {nvfp4: "NV4", nvint4: "NI4", mxfp4: "MX4", mxfp8: "MX8", int8: "8A8", bfp12: "B12",
  awq_nvfp4: "AN4", smoothquant_nvfp4: "SQ4"};

function describe(method, taken) {
  for (const [pattern, family, label, code] of FAMILY_RULES) {
    const match = pattern.exec(method);
    if (match) return {family, label, code: code(match)};
  }
  let code = OTHER_CODES[method];
  if (!code || taken.has(code)) {
    const letters = method.replace(/[^a-z]/gi, "").toUpperCase();
    const digit = (method.match(/\d/g) || []).pop() ?? "";
    code = `${letters.slice(0, 2)}${digit}`;
    for (let i = 0; taken.has(code); i += 1) code = `${letters.slice(0, 1)}${digit}${i}`.slice(0, 3);
  }
  return {family: "other", label: "기타 W×A 형식", code};
}

// ---------- formatting ----------
const NONE = "기록 없음";
const finite = (x) => typeof x === "number" && Number.isFinite(x);
const fmtBits = (x) => (finite(x) ? x.toFixed(4) : NONE);
const fmtKL = (x) => (!finite(x) ? NONE : x === 0 ? "0" : Math.abs(x) >= 1 ? x.toFixed(3) : x.toPrecision(4));
const fmtPPL = (x) => (finite(x) ? x.toFixed(2) : NONE);
const fmtUsd = (x) => (finite(x) ? `$${x.toFixed(3)}` : "비용 기록 없음");
const fmtSec = (x) => (finite(x) ? `${x.toFixed(1)}초` : "시간 기록 없음");
const short = (sha) => (sha ? String(sha).slice(0, 7) : null);
const gpuName = (name) => (name ? String(name).replace(/^NVIDIA\s+/, "") : null);
const frac = ([p, q]) => Number(p * 1000000000n / q) / 1e9;
const sameFrac = (a, b) => Boolean(a && b) && a[0] * b[1] === b[0] * a[1];
// A value in the readout font, or the muted "기록 없음" when there is none.
const val = (text) => (text === NONE ? h("span", {class: "at-none"}, NONE) : h("b", null, text));

// A budget [p, q] as the text field shows it: a decimal when it has at most 4 places, else the reduced fraction.
function budgetText([p, q]) {
  const value = frac([p, q]);
  if (Math.abs(Math.round(value * 10000) / 10000 - value) < 1e-12) return String(Math.round(value * 10000) / 10000);
  let [a, b] = [p, q];
  while (b) [a, b] = [b, a % b];
  return `${p / a}/${q / a}`;
}

// ---------- data, filled by boot() ----------
let bundle;
let baseTable;             // the table as exported
let table;                 // the table the solver uses: baseTable, or baseTable within a tune request's constraints
let methods = [];          // method ids, table order
let info = {};             // method id -> {family, label, code, uniform: params-weighted bits of the uniform recipe}
let kinds = [];
let blocks = 0;
let layerAt = [];          // [block][kind index] -> layer index | undefined
let cheapestBits = 0;      // the cheapest assignment (every layer at its smallest tabled option)
let cheapestUnits = 0;     // slider minimum, in 1/128 bit (the cheapest assignment rounded up)
let session = null;
let proposals = new Map(); // proposal name -> proposal (stored rounds, then live rounds)
let sessionBudget = null;

const state = {
  text: "", budget: null, solution: null, counts: [], bits: null, rawKL: null, grid: null, ms: null, error: "",
  shown: null,              // the grid the floorplan shows now (the solution, or a hovered proposal's)
  hover: null, metric: "kl", tableView: false, liveRounds: [], liveError: "",
  constraints: null,        // {keep_bf16_blocks, exclude_methods, only_methods} of an applied tune request, or null
};
const ui = {};
let solveTimer = 0;
let moveTimer = 0;
const root = document.getElementById("at");

// Uniform (one method everywhere) average stored bits, weighted by params: equal to the per-layer bits except for
// a per-row scale (int8), whose bits depend on in_features.
function uniformBits(method) {
  let bits = 0;
  let params = 0;
  for (const layer of table.layers) {
    const [num, den] = weightBits(table.weight_bits[method], layer.in_features);
    bits += layer.params * (Number(num) / Number(den));
    params += layer.params;
  }
  return bits / params;
}

const layerBits = (layer, method) => {
  const [num, den] = weightBits(table.weight_bits[method], layer.in_features);
  return Number(num) / Number(den);
};

function gridOf(choice) {
  return Array.from({length: blocks}, (_, b) => kinds.map((_, k) => {
    const i = layerAt[b][k];
    return i === undefined ? null : methods[choice[i]];
  }));
}

const sameGrid = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length
  && a.every((row, i) => Array.isArray(row) && row.length === b[i].length && row.every((m, k) => m === b[i][k]));

// A solution measured with a quantized KV cache (kv != "none") is labeled as such: "auto_5.72 · KV kivi2".
const hasKV = (row) => typeof row.kv === "string" && row.kv !== "none";
function solutionLabel(row) {
  if (!hasKV(row)) return row.id;
  const base = row.id.endsWith(`_${row.kv}`) ? row.id.slice(0, -row.kv.length - 1) : row.id;
  return `${base} · KV ${row.kv}`;
}

// ---------- tune-request constraints: port of app/tune_request.constrained ----------
const blockOf = (block) => ((block % blocks) + blocks) % blocks;  // Python's block % blocks (-1 = the last block)

// `base` with every option the request rules out set to null: BF16 only on the kept blocks, no excluded method,
// only the allowed methods (BF16 stays allowed) elsewhere.
function constrained(base, request) {
  const kept = new Set((request.keep_bf16_blocks ?? []).map(blockOf));
  const excluded = new Set(request.exclude_methods ?? []);
  let allowed = base.methods.filter((m) => !excluded.has(m));
  if (request.only_methods?.length) allowed = allowed.filter((m) => m === "bf16" || request.only_methods.includes(m));
  const out = structuredClone(base);
  for (const layer of out.layers) {
    base.methods.forEach((method, j) => {
      if (!allowed.includes(method) || (kept.has(layer.block) && method !== "bf16")) {
        layer.value[j] = null;
        layer.kl[j] = null;
      }
    });
  }
  return out;
}

const constraintCount = (c) => c.keep_bf16_blocks.length + c.exclude_methods.length + (c.only_methods ? 1 : 0);

// ---------- proposals: base grids and their own grids ----------
function baseGrid(name, seen = new Set()) {
  const solution = bundle.solutions.find((item) => item.id === name);
  if (solution?.grid) return solution.grid;
  const proposal = proposals.get(name);
  if (!proposal || seen.has(name)) return null;
  seen.add(name);
  return proposalGrid(proposal, seen);
}

function proposalGrid(proposal, seen = new Set()) {
  if (Array.isArray(proposal.grid)) return proposal.grid;
  const base = baseGrid(proposal.base, seen);
  if (!base) return null;
  const grid = base.map((row) => [...row]);
  for (const change of validChanges(proposal)) grid[change.block][kinds.indexOf(change.kind)] = change.method;
  return grid;
}

// The changes the page can place on the floorplan (agent output is checked here before it is used as an index).
function validChanges(proposal) {
  if (!Array.isArray(proposal.changes)) return [];
  return proposal.changes.filter((c) => c && Number.isInteger(c.block) && c.block >= 0 && c.block < blocks
    && kinds.includes(c.kind) && typeof c.method === "string");
}

// ---------- shared bits ----------
function announce(text) {
  if (ui.live) ui.live.textContent = text;
}

async function copy(button, text) {
  const label = button.dataset.label;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "복사됨";
  } catch {
    button.textContent = "복사하지 못했습니다 · 직접 선택하세요";
  }
  setTimeout(() => (button.textContent = label), 1600);
}

const copyButton = (label, text) => {
  const button = h("button", {type: "button", class: "copy", "data-label": label}, label);
  button.addEventListener("click", () => copy(button, typeof text === "function" ? text() : text));
  return button;
};

const code = (method) => {
  const item = info[method] ?? describe(String(method), new Set());
  return h("span", {class: "at-code", "data-f": item.family, "aria-hidden": "true"}, item.code);
};

// A tooltip of label/value rows under a monospaced heading.
function tipRows(head, rows) {
  return h("div", null, head ? h("div", {class: "tip-head"}, head) : null,
    h("table", null, rows.filter(Boolean).map(([k, v]) => h("tr", null, h("td", {class: "muted"}, k), h("td", null, v)))));
}

function tipOn(el, build) {
  el.addEventListener("mouseenter", () => showTip(el, build()));
  el.addEventListener("focus", () => showTip(el, build()));
  el.addEventListener("mouseleave", hideTip);
  el.addEventListener("blur", hideTip);
  return el;
}

// "GPU 기록값" plus the record's GPU and git sha; a record the bundle lacks says so instead of guessing.
function recordNote(record, id) {
  const where = [gpuName(record?.gpu), short(record?.git_sha)].filter(Boolean).join(" · ");
  return where || (id ? `기록 ${id} · GPU·git sha 는 번들에 없음` : "GPU·git sha 기록 없음");
}

function recordBadge(record, id) {
  const badge = h("span", {class: "at-badge rec", tabindex: "0"}, "GPU 기록값");
  return tipOn(badge, () => tipRows(id ?? record?.name ?? "측정 기록", [
    ["GPU", record?.gpu ?? NONE], ["git", short(record?.git_sha) ?? NONE], record?.source ? ["파일", record.source] : null,
    record?.windows ? ["창", `${record.windows} × ${record.seqlen ?? "?"} · ${record.split ?? ""} · seed ${record.seed ?? "?"}`] : null,
  ]));
}

const step = (no, title, hint, ...body) => h("section", {class: "panel at-step", "aria-labelledby": `at-step-${no}`},
  h("div", {class: "step-head"}, h("span", {class: "step-no"}, no), h("h2", {class: "step-title", id: `at-step-${no}`}, title),
    hint ? h("span", {class: "step-hint"}, hint) : null), ...body);

// A paragraph clamped to `lines` lines with a toggle, shown only when the text is longer than that.
function clamped(text, lines) {
  const p = h("p", {class: "at-clamp", style: `--lines: ${lines}`}, text);
  const more = h("button", {type: "button", class: "at-more", "aria-expanded": "false", hidden: true}, "전체 보기");
  more.addEventListener("click", () => {
    const open = p.classList.toggle("open");
    more.setAttribute("aria-expanded", String(open));
    more.textContent = open ? "접기" : "전체 보기";
  });
  requestAnimationFrame(() => {
    if (p.scrollHeight > p.clientHeight + 2) more.hidden = false;
  });
  return [p, more];
}

// ---------- context strip ----------
function contextStrip() {
  const meta = table.meta ?? {};
  const pill = (label, value) => h("span", {class: "pill"}, label, " ", h("b", null, value));
  const mma = meta.mma == null ? null
    : typeof meta.mma === "string" && meta.mma.includes("fp32") ? h("span", {class: "pill"}, "가중치 전용 · fp32 누산")
      : pill("누산", typeof meta.mma === "string" ? meta.mma : JSON.stringify(meta.mma));
  const windows = meta.windows ? `${meta.windows} 창 × ${meta.seqlen ?? "?"} 토큰 · seed ${meta.seed ?? "?"}` : NONE;
  return h("div", {class: "at-context", "aria-label": "표의 실행 환경"},
    pill("모델", `${meta.model_id ?? NONE}${meta.revision ? ` @ ${short(meta.revision)}` : ""}`), mma,
    pill("방법", String(methods.length)), pill("KL 표", windows),
    pill("GPU", gpuName((meta.gpu_names ?? [])[0]) ?? NONE),
    pill("트리", meta.git_sha ? `${short(meta.git_sha)}${meta.git_dirty ? " (수정됨)" : ""}` : NONE));
}

// The cheapest assignment of the solver's table, exact: every layer at its smallest remaining option.
function computeCheapest() {
  const {costs, scale, params} = problem(table);
  const cheapest = costs.reduce((sum, row) => sum + row.reduce((a, b) => (b < a ? b : a)), 0n);
  cheapestBits = frac([cheapest, params * scale]);
  const units = cheapest * BigInt(UNITS);
  const den = params * scale;
  cheapestUnits = Number(units / den + (units % den ? 1n : 0n));
}

// ---------- 1 budget ----------
function budgetStep() {
  const max = Math.max(cheapestUnits, Math.round(MAX_BITS * UNITS));
  ui.slider = h("input", {type: "range", min: String(cheapestUnits), max: String(max), step: "1", id: "at-slider",
    "aria-label": "크기 예산 (가중치당 비트)"});
  ui.slider.addEventListener("input", () => {
    const text = budgetText([BigInt(ui.slider.value), BigInt(UNITS)]);
    ui.input.value = text;
    ui.slider.setAttribute("aria-valuetext", `${text} 비트`);
    queueSolve(text, false);
  });
  ui.ticks = h("div", {class: "at-ticks", "aria-hidden": "true"});
  ui.input = h("input", {type: "text", class: "at-input", id: "at-budget", inputmode: "decimal", autocomplete: "off",
    spellcheck: "false", "aria-describedby": "at-budget-error"});
  ui.input.addEventListener("input", () => queueSolve(ui.input.value, false));
  ui.input.addEventListener("change", () => queueSolve(ui.input.value, true, true));
  ui.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") queueSolve(ui.input.value, true, true);
  });
  ui.error = h("p", {class: "at-error", id: "at-budget-error", role: "alert"});
  ui.time = h("span", {class: "at-solve-time"});
  const sessionButton = sessionBudget ? h("button", {type: "button", class: "at-btn",
    onclick: () => { ui.input.value = session.budget_bits; queueSolve(session.budget_bits, true, true); }},
  `AI 세션 예산으로 (${session.budget_bits})`) : null;
  ui.minLabel = h("span", null, (cheapestUnits / UNITS).toFixed(4));
  ui.maxLabel = h("span", null, (Number(ui.slider.max) / UNITS).toFixed(1));
  ui.constraints = h("div", {class: "at-constraints", hidden: true, "aria-label": "적용 중인 제약"});
  return step(1, "크기 예산", null,
    requestBox(),
    h("p", {class: "at-lead"}, "Linear 가중치 하나에 평균 몇 비트를 쓸지 정합니다. 눈금은 한 방법을 모든 층에 쓴 균일 레시피의 크기입니다."),
    h("div", {class: "at-budget"},
      h("div", {class: "at-slider"}, ui.slider),
      ui.ticks,
      h("div", {class: "at-ends", "aria-hidden": "true"}, ui.minLabel, ui.maxLabel)),
    h("div", {class: "at-budget-row"},
      h("label", {for: "at-budget"}, "예산"), ui.input, h("span", {class: "at-unit"}, "비트/가중치 · 3.15 또는 403/128"),
      sessionButton, ui.time),
    ui.constraints,
    ui.error);
}

// ---------- 1 natural-language request ----------
const OUTCOME = {tune: ["튜닝", "ok"], needs_clarification: ["확인 필요", ""], out_of_scope: ["범위 밖", ""], conflict: ["충돌", ""]};
const storedRequests = () => (Array.isArray(bundle.requests)
  ? bundle.requests.filter((item) => item && typeof item.text === "string" && item.request && typeof item.request === "object") : []);

function requestBox() {
  ui.reqInput = h("input", {type: "text", class: "at-req-input", id: "at-request", autocomplete: "off", maxlength: "1000",
    placeholder: "Qwen3-0.6B 를 3.2비트 안에서, 첫 블록과 마지막 블록은 BF16 으로"});
  ui.reqInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !ui.reqAsk.disabled) parseRequest();
  });
  ui.reqAsk = h("button", {type: "button", class: "at-btn primary"}, "요청 해석");
  ui.reqAsk.addEventListener("click", parseRequest);
  ui.reqCancel = h("button", {type: "button", class: "at-btn", hidden: true}, "취소");
  ui.reqWait = h("span", {class: "at-wait", "aria-live": "polite"});
  ui.reqError = h("p", {class: "at-live-error", role: "alert", hidden: true});
  ui.reqResult = h("div", {class: "at-req-result", hidden: true});
  const stored = storedRequests();
  ui.examples = stored.map((item) => {
    const chip = h("button", {type: "button", class: "at-example", title: item.text, "aria-pressed": "false"}, item.text);
    chip.addEventListener("click", () => {
      ui.reqInput.value = item.text;
      ui.reqError.hidden = true;
      ui.examples.forEach((other) => other.setAttribute("aria-pressed", String(other === chip)));
      showRequest(item, false);
    });
    return chip;
  });
  return h("div", {class: "at-request"},
    h("label", {class: "at-req-label", for: "at-request"}, "요청을 문장으로 입력"),
    h("div", {class: "at-req-row"}, ui.reqInput, ui.reqAsk, ui.reqCancel),
    ui.reqWait,
    stored.length ? h("div", {class: "at-examples"}, h("span", {class: "at-muted"}, "저장된 예시"), ui.examples) : null,
    ui.reqError, ui.reqResult);
}

async function parseRequest() {
  const text = ui.reqInput.value.trim();
  ui.reqError.hidden = true;
  if (!text) {
    ui.reqError.textContent = `요청 문장을 입력하세요. 예: ${ui.reqInput.placeholder}`;
    ui.reqError.hidden = false;
    return;
  }
  try {
    const body = await postLive(PARSE, {text}, {button: ui.reqAsk, cancel: ui.reqCancel, wait: ui.reqWait,
      waiting: "Claude 가 요청을 해석하는 중", usual: "보통 5–60초", what: "요청 해석"});
    if (!body.request || typeof body.request !== "object") throw new Error("요청 해석 API 의 응답에 request 가 없습니다. 서버 버전을 확인하세요.");
    ui.examples.forEach((chip) => chip.setAttribute("aria-pressed", "false"));
    showRequest(body, true);
  } catch (error) {
    let message = error.name === "AbortError" ? "요청 해석을 취소했습니다. 화면은 그대로입니다." : error.message;
    if (error.code === "agent_unavailable") {
      message += storedRequests().length ? " 아래 저장된 예시를 누르면 같은 흐름을 볼 수 있습니다." : " 저장된 예시도 번들에 없습니다.";
    }
    ui.reqError.textContent = message;
    ui.reqError.hidden = false;
  }
}

// What in a "tune" request this table cannot take (the server refuses these with 422; a stored reply is checked
// again here before its values become indices).
function requestProblems(req) {
  const problems = [];
  if (typeof req.budget_bits !== "string" || !/^\d+(\.\d+)?(\/\d+)?$/.test(req.budget_bits)) {
    problems.push(`크기 예산 ${req.budget_bits ?? "없음"} 을 읽을 수 없습니다`);
  }
  const kept = Array.isArray(req.keep_bf16_blocks) ? req.keep_bf16_blocks : [];
  const outside = kept.filter((b) => !Number.isInteger(b) || b < -blocks || b >= blocks);
  if (outside.length) problems.push(`모델에 없는 블록 ${outside.join(", ")}`);
  const named = [...(Array.isArray(req.exclude_methods) ? req.exclude_methods : []),
    ...(Array.isArray(req.only_methods) ? req.only_methods : [])];
  const unknown = [...new Set(named.filter((m) => !methods.includes(m)))];
  if (unknown.length) problems.push(`표에 없는 방법 ${unknown.join(", ")}`);
  return problems;
}

// A parsed request, live or stored: its fields, what the code decided, and for "tune" the budget and constraints
// applied to the heuristic. The other outcomes leave the floorplan as it is.
function showRequest(item, live) {
  const req = item.request;
  const date = (item.recorded_utc ?? item.llm?.started_utc ?? "").slice(0, 10) || "날짜 기록 없음";
  const [outcomeLabel, outcomeClass] = OUTCOME[req.outcome] ?? [`결과 ${req.outcome ?? "없음"}`, ""];
  const lines = (values) => (Array.isArray(values) && values.length
    ? h("ul", {class: "at-req-list"}, values.map((value) => h("li", null, String(value)))) : "없음");
  const kept = Array.isArray(req.keep_bf16_blocks) && req.keep_bf16_blocks.length
    ? req.keep_bf16_blocks.map((b) => (Number.isInteger(b) && b < 0 ? `${b} (블록 ${blockOf(b)})` : String(b))).join(", ") : "없음";
  const row = (k, v, wide = false) => h("div", {class: wide ? "wide" : null}, h("dt", null, k), h("dd", null, v));
  let verdict;
  if (req.outcome === "tune") {
    const problems = requestProblems(req);
    const failed = problems.length ? null : applyRequest(req);
    verdict = problems.length || failed
      ? h("p", {class: "at-live-error"}, `이 응답은 적용하지 않았습니다: ${problems.length ? problems.join(" · ") : failed}`)
      : h("p", {class: "at-req-note"}, `적용했습니다: 예산 ${req.budget_bits} 비트`,
        state.constraints ? ` · 제약 ${constraintCount(state.constraints)}개` : " · 제약 없음",
        req.compute_budget ? " · 연산 예산은 브라우저 DP 에 넣지 않았습니다 (Python solve --compute-budget)" : "");
  } else if (req.outcome === "needs_clarification") {
    verdict = h("div", {class: "at-notice"}, h("p", null, "확인이 필요해 격자는 바꾸지 않았습니다. 아래 질문에 답하도록 요청을 다시 쓰세요."),
      lines(req.questions));
  } else {
    verdict = h("div", {class: "at-notice"},
      h("p", null, req.outcome === "conflict" ? "요청의 조건이 서로 맞지 않아 격자는 바꾸지 않았습니다."
        : req.outcome === "out_of_scope" ? "TriCast 가 하지 않는 일이라 격자는 바꾸지 않았습니다." : "격자는 바꾸지 않았습니다."),
      h("code", null, req.reason ?? "사유 기록 없음"));
  }
  const llm = item.llm;
  ui.reqResult.replaceChildren(
    h("div", {class: "at-round-head"},
      live ? h("span", {class: "at-badge live"}, "실시간 Claude 응답") : h("span", {class: "at-badge stored"}, `저장된 응답 · ${date}`),
      h("span", {class: `at-badge ${outcomeClass}`}, outcomeLabel),
      llm ? h("span", {class: "at-muted"}, `${fmtUsd(llm.cost_usd)} · ${fmtSec(llm.duration_s)}${(llm.models ?? []).length ? ` · ${llm.models.join(", ")}` : ""}`) : null),
    h("p", {class: "at-req-text"}, `“${item.text ?? ""}”`),
    h("dl", {class: "at-dl"},
      row("모델", req.model ?? "지정 안 함"), row("크기 예산", req.budget_bits ? `${req.budget_bits} 비트` : "지정 안 함"),
      row("연산 예산", req.compute_budget ?? "지정 안 함"), row("BF16 유지 블록", kept),
      row("제외 방법", Array.isArray(req.exclude_methods) && req.exclude_methods.length ? req.exclude_methods.join(", ") : "없음"),
      row("허용 방법", Array.isArray(req.only_methods) && req.only_methods.length ? `${req.only_methods.join(", ")} (+ bf16)` : "제한 없음"),
      row("가정", lines(req.assumptions), Boolean(req.assumptions?.length)),
      row("코드 판정", lines(req.checks), Boolean(req.checks?.length))),
    verdict);
  ui.reqResult.hidden = false;
  announce(`요청 해석: ${outcomeLabel}`);
}

// Apply a checked "tune" request: its constraints, then its budget. Returns null, or why it could not be applied.
function applyRequest(req) {
  const constraints = {keep_bf16_blocks: [...(req.keep_bf16_blocks ?? [])], exclude_methods: [...(req.exclude_methods ?? [])],
    only_methods: req.only_methods?.length ? [...req.only_methods] : null};
  return setConstraints(constraintCount(constraints) ? constraints : null, req.budget_bits);
}

// Solve within `constraints` (null = the plain table) at `budget`: the slider minimum and the ticks follow the
// cheapest assignment the constrained table allows. Returns null, or why the constraints cannot be solved.
function setConstraints(constraints, budget = ui.input.value) {
  const next = constraints ? constrained(baseTable, constraints) : baseTable;
  const empty = next.layers.find((layer) => layer.value.every((v) => v === null));
  if (empty) return `제약 때문에 ${empty.name} 에 남는 방법이 없습니다. 제약을 줄이세요.`;
  table = next;
  state.constraints = constraints;
  computeCheapest();
  ui.slider.min = String(cheapestUnits);
  ui.slider.max = String(Math.max(cheapestUnits, Math.round(MAX_BITS * UNITS)));
  ui.minLabel.textContent = (cheapestUnits / UNITS).toFixed(4);
  ui.maxLabel.textContent = (Number(ui.slider.max) / UNITS).toFixed(1);
  layoutTicks();
  renderConstraints();
  ui.input.value = budget;
  runSolve(String(budget), true);
  return null;
}

// The active constraints as removable chips; "제약 해제" goes back to the plain table.
function renderConstraints() {
  const c = state.constraints;
  ui.constraints.hidden = !c;
  if (!c) {
    ui.constraints.replaceChildren();
    return;
  }
  const without = (patch) => () => {
    const next = {...c, ...patch};
    setConstraints(constraintCount(next) ? next : null);
    (ui.constraints.querySelector("button") ?? ui.slider).focus();
  };
  const chip = (label, remove) => h("span", {class: "at-chip"}, label,
    h("button", {type: "button", class: "at-chip-x", "aria-label": `${label} 제약 해제`, onclick: remove}, "×"));
  ui.constraints.replaceChildren(...[h("span", {class: "at-muted"}, "적용 중인 제약"),
    ...c.keep_bf16_blocks.map((b, i) => chip(`BF16 유지 · 블록 ${blockOf(b)}`,
      without({keep_bf16_blocks: c.keep_bf16_blocks.filter((_, j) => j !== i)}))),
    ...c.exclude_methods.map((m, i) => chip(`제외 · ${m}`, without({exclude_methods: c.exclude_methods.filter((_, j) => j !== i)}))),
    c.only_methods ? chip(`허용 · ${c.only_methods.join(", ")} (+ bf16)`, without({only_methods: null})) : null,
    h("button", {type: "button", class: "at-btn", onclick: () => {
      setConstraints(null);
      ui.slider.focus();
    }}, "제약 해제")].filter(Boolean));
}

// Tick marks at the uniform recipes' sizes inside the slider range; codes at the same size share one tick, and
// labels that would touch move to a lower row.
function layoutTicks() {
  const lo = cheapestUnits / UNITS;
  const hi = Number(ui.slider.max) / UNITS;
  const groups = new Map();
  for (const [j, method] of methods.entries()) {
    if (!table.layers.some((layer) => layer.value[j] !== null)) continue;  // ruled out everywhere by the constraints
    const bits = info[method].uniform;
    if (!(bits >= lo - 1e-9 && bits <= hi + 1e-9)) continue;
    const key = bits.toFixed(6);
    if (!groups.has(key)) groups.set(key, {bits, methods: []});
    groups.get(key).methods.push(method);
  }
  const sorted = [...groups.values()].sort((a, b) => a.bits - b.bits);
  ui.ticks.replaceChildren();
  const width = ui.ticks.clientWidth || 400;
  const rowEnds = [];
  let rows = 1;
  for (const group of sorted) {
    const label = h("span", null, group.methods.map((m) => info[m].code).join(" "));
    const stem = h("i");
    const tick = h("div", {class: "at-tick", style: `left: ${((group.bits - lo) / (hi - lo || 1)) * 100}%`}, stem, label);
    ui.ticks.append(tick);
    const x = ((group.bits - lo) / (hi - lo || 1)) * width;
    const half = label.offsetWidth / 2 + 3;
    let row = rowEnds.findIndex((end) => end < x - half);
    if (row < 0) row = rowEnds.length;
    rowEnds[row] = x + half;
    rows = Math.max(rows, row + 1);
    const overflow = Math.min(0, width + 8 - (x + half - 3)) || Math.max(0, -8 - (x - half + 3));
    if (overflow) label.style.transform = `translateX(${overflow}px)`;
    stem.style.height = `${4 + row * 13}px`;
  }
  ui.ticks.style.height = `${8 + rows * 13}px`;
}

function queueSolve(text, immediate, showFormatError = false) {
  clearTimeout(solveTimer);
  solveTimer = setTimeout(() => runSolve(text, showFormatError), immediate ? 0 : DEBOUNCE_MS);
}

function budgetError(error, text) {
  const cheapest = cheapestBits;
  if (/^infeasible budget/.test(error.message)) {
    return `${text.trim()} 비트는 가장 작은 배정(${cheapest.toFixed(4)} 비트)보다 작습니다. ${cheapest.toFixed(4)} 이상으로 정하세요.`;
  }
  if (/must be positive|zero denominator/.test(error.message)) return "예산은 0보다 큰 수여야 합니다. 예: 4.5 또는 576/128";
  if (/write a number/.test(error.message)) return "예산을 읽지 못했습니다. 3.15 처럼 소수로 쓰거나 403/128 처럼 분수로 쓰세요.";
  if (/too large/.test(error.message)) return "이 예산은 브라우저 DP 의 칸 수 한도를 넘습니다. 더 작은 예산으로 정하세요.";
  return `이 예산으로 풀지 못했습니다 (${error.message}). 다른 예산으로 다시 정하세요.`;
}

function runSolve(text, showFormatError) {
  let budget;
  try {
    budget = parseBits(text);
    const started = performance.now();
    const solution = solve(table, budget);
    const ms = performance.now() - started;
    const {bits, counts} = summary(table, solution);
    let rawKL = 0;
    for (const [i, j] of solution.choice.entries()) {
      const kl = table.layers[i].kl[j];
      rawKL = finite(kl) && rawKL !== null ? rawKL + kl : null;
    }
    Object.assign(state, {text: text.trim(), budget, solution, counts, bits, rawKL, ms, error: "", grid: gridOf(solution.choice)});
  } catch (error) {
    if (!(error instanceof SolveError)) throw error;
    const formatOnly = /write a number/.test(error.message);
    if (formatOnly && !showFormatError) return;
    state.error = budgetError(error, String(text));
    ui.error.textContent = state.error;
    ui.input.setAttribute("aria-invalid", "true");
    return;
  }
  ui.error.textContent = "";
  ui.input.removeAttribute("aria-invalid");
  if (document.activeElement !== ui.input) ui.input.value = state.text;
  const units = Math.round(frac(state.budget) * UNITS);
  if (document.activeElement !== ui.slider) {
    ui.slider.value = String(Math.min(Math.max(units, Number(ui.slider.min)), Number(ui.slider.max)));
  }
  ui.slider.setAttribute("aria-valuetext", `${state.text} 비트`);
  ui.time.textContent = `브라우저 정확 DP · ${(state.ms / 1000).toFixed(2)}초`;
  if (!state.hover) showGrid(state.grid, true);
  else setHover(state.hover);
  updateStats();
  updateAgentNote();
  updateCommands();
  announce(`${fmtBits(state.bits)} 비트 · 예측 KL ${fmtKL(state.rawKL)}`);
}

// ---------- 2 floorplan ----------
function planStep() {
  ui.cells = [];
  const attn = kinds.filter((k) => ["q_proj", "k_proj", "v_proj", "o_proj"].includes(k)).length;
  const split = attn > 0 && attn < kinds.length;
  ui.grid = h("div", {class: "at-grid", role: "grid", "aria-label": `레이어 배정: 블록 ${blocks} × 종류 ${kinds.length}`,
    "aria-describedby": "at-gridnote", style: split
      ? `grid-template-columns: 22px repeat(${attn}, var(--cw)) 6px repeat(${kinds.length - attn}, var(--cw))`
      : `grid-template-columns: 22px repeat(${kinds.length}, var(--cw))`});
  const head = h("div", {role: "row"}, h("span", {role: "columnheader"}),
    ...kinds.map((kind, k) => [split && k === attn ? h("span", {class: "at-gap"}) : null,
      h("span", {class: "at-colhead", role: "columnheader", title: kind}, SHORT_KIND[kind] ?? kind)]));
  const groups = split ? h("div", {role: "presentation", style: "display: contents"},
    h("span"), h("span", {class: "at-group", style: `grid-column: span ${attn}`}, "attention"), h("span"),
    h("span", {class: "at-group", style: `grid-column: span ${kinds.length - attn}`}, "MLP")) : null;
  const rows = [];
  for (let b = 0; b < blocks; b += 1) {
    ui.cells.push([]);
    const cells = kinds.map((kind, k) => {
      const cell = h("span", {class: "at-cell", role: "gridcell", tabindex: b === 0 && k === 0 ? "0" : "-1",
        "data-b": String(b), "data-k": String(k)});
      ui.cells[b].push(cell);
      return [split && k === attn ? h("span", {class: "at-gap"}) : null, cell];
    });
    rows.push(h("div", {role: "row"}, h("span", {class: "at-rowhead", role: "rowheader"}, String(b)), ...cells));
  }
  ui.grid.append(...[groups, head, ...rows].filter(Boolean));
  ui.grid.addEventListener("mouseover", (event) => {
    const cell = event.target.closest(".at-cell");
    if (cell) showTip(cell, cellTip(Number(cell.dataset.b), Number(cell.dataset.k)));
  });
  ui.grid.addEventListener("mouseleave", hideTip);
  ui.grid.addEventListener("focusin", (event) => {
    const cell = event.target.closest(".at-cell");
    if (cell) showTip(cell, cellTip(Number(cell.dataset.b), Number(cell.dataset.k)));
  });
  ui.grid.addEventListener("focusout", hideTip);
  ui.grid.addEventListener("keydown", moveFocus);
  ui.note = h("p", {class: "at-gridnote", id: "at-gridnote"}, "칸에 마우스를 올리거나 방향키로 옮기면 단층 KL 과 대안을 봅니다.");

  ui.bits = h("div", {class: "v"});
  ui.bitsSub = h("div", {class: "s"});
  ui.kl = h("div", {class: "v"});
  ui.klSub = h("div", {class: "s"});
  ui.bar = h("div", {class: "at-bar", role: "img"});
  ui.keys = h("ul", {class: "at-keys"});
  ui.uniform = h("div", {class: "at-box"});
  ui.record = h("div", {class: "at-box"});
  ui.live = h("p", {class: "sr-only", "aria-live": "polite"});
  const stats = h("div", {class: "at-stats"},
    h("div", {class: "at-read"}, h("div", {class: "k"}, "달성 비트"), ui.bits, ui.bitsSub),
    h("div", {class: "at-read"}, h("div", {class: "k"}, "예측 KL (단층 KL 합)", h("span", {class: "at-tag"}, "예측")), ui.kl, ui.klSub),
    h("div", {class: "at-mix"}, h("div", {class: "at-read"}, h("div", {class: "k"}, "방법 구성 (층 수)")), ui.bar, ui.keys),
    ui.uniform, ui.record);
  return step(2, "휴리스틱 배정 (정확 배낭)", null,
    h("p", {class: "at-lead"}, "층 하나만 양자화했을 때의 모델 KL 을 층 × 방법마다 재 둔 표에서, 예산 안에 들면서 KL 합이 가장 작은 조합을 정확한 DP 로 고릅니다. 칸 색은 방법 계열, 글자는 방법과 비트입니다."),
    ui.note,
    h("div", {class: "at-plan"}, h("div", {class: "at-gridwrap"}, ui.grid), stats), ui.live);
}

function moveFocus(event) {
  const cell = event.target.closest(".at-cell");
  if (!cell) return;
  let b = Number(cell.dataset.b);
  let k = Number(cell.dataset.k);
  const moves = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]};
  if (moves[event.key]) [b, k] = [b + moves[event.key][0], k + moves[event.key][1]];
  else if (event.key === "Home") k = 0;
  else if (event.key === "End") k = kinds.length - 1;
  else return;
  event.preventDefault();
  b = Math.min(Math.max(b, 0), blocks - 1);
  k = Math.min(Math.max(k, 0), kinds.length - 1);
  cell.tabIndex = -1;
  const next = ui.cells[b][k];
  next.tabIndex = 0;
  next.focus();
}

// Paint the floorplan with `grid`; with `stagger`, only the cells whose method changed fade in, top to bottom.
function showGrid(grid, stagger) {
  state.shown = grid;
  clearTimeout(moveTimer);
  for (const row of ui.cells) {
    for (const cell of row) {
      cell.classList.remove("moved");
      cell.style.transitionDelay = "";
    }
  }
  const changed = [];
  for (let b = 0; b < blocks; b += 1) {
    for (let k = 0; k < kinds.length; k += 1) {
      const method = grid?.[b]?.[k] ?? null;
      const cell = ui.cells[b][k];
      if (cell.dataset.m === (method ?? "")) continue;
      changed.push([cell, method, b, k]);
    }
  }
  const last = Math.max(1, changed.length - 1);
  changed.forEach(([cell, method, b, k], i) => {
    if (stagger) {
      cell.classList.add("moved");
      cell.style.transitionDelay = `${Math.round((i / last) * STAGGER_MS)}ms`;
    }
    cell.dataset.m = method ?? "";
    if (method === null) {
      delete cell.dataset.f;
      cell.textContent = "";
      cell.setAttribute("aria-label", `블록 ${b} ${kinds[k]}: 표에 없음`);
      return;
    }
    const item = info[method] ?? describe(method, new Set());
    cell.dataset.f = item.family;
    cell.textContent = item.code;
    cell.setAttribute("aria-label", `블록 ${b} ${kinds[k]}: ${method}`);
  });
  if (stagger && changed.length) {
    moveTimer = setTimeout(() => {
      for (const [cell] of changed) {
        cell.classList.remove("moved");
        cell.style.transitionDelay = "";
      }
    }, STAGGER_MS + 200);
  }
}

function cellTip(b, k) {
  const i = layerAt[b][k];
  const method = state.shown?.[b]?.[k];
  if (i === undefined || !method) return tipRows(`블록 ${b} ${kinds[k]}`, [["표", "이 층은 표에 없습니다"]]);
  // a hovered proposal's grid was not solved within the request's constraints: read it against the plain table
  const layer = (state.shown === state.grid ? table : baseTable).layers[i];
  const j = methods.indexOf(method);
  const kl = j >= 0 ? layer.kl[j] : null;
  // the other tabled method with the lowest single-layer KL, BF16 left out (it is the unquantized skip, KL 0)
  let alt = null;
  methods.forEach((m, jj) => {
    if (m === method || m === "bf16" || layer.value[jj] === null || !finite(layer.kl[jj])) return;
    if (alt === null || layer.kl[jj] < layer.kl[alt]) alt = jj;
  });
  const altText = alt === null ? "없음"
    : `${methods[alt]} · KL ${fmtKL(layer.kl[alt])} · ${layerBits(layer, methods[alt]).toFixed(4)} 비트`;
  const shownNote = state.hover && state.shown !== state.grid ? ["격자", `제안 ${state.hover.name} 의 격자`] : null;
  return tipRows(layer.name, [
    ["방법", `${method} (${info[method]?.code ?? "?"})`],
    ["비트", j >= 0 ? layerBits(layer, method).toFixed(4) : NONE],
    ["단층 KL", fmtKL(kl)],
    ["floor", fmtKL(layer.floor)],
    ...reportRows(j >= 0 ? layer.report?.[j] : null),
    ["대안 (BF16 제외 KL 최저)", altText],
    shownNote,
    ...profileRows(layer.profile, layer.in_features),
  ]);
}

// A proposal's options as one line: the options it sets away from their defaults (options_changed, checked by
// the exporter), methods with the same changes in one group; the options as written when they did not check.
function optionSummary(p) {
  const changed = p.options_changed;
  if (changed && typeof changed === "object") {
    const groups = new Map();
    for (const [m, opts] of Object.entries(changed)) {
      const text = Object.entries(opts ?? {}).map(([k, v]) => `${k} = ${v}`).join(", ");
      groups.set(text, [...(groups.get(text) ?? []), m]);
    }
    return groups.size ? `기본값과 다른 옵션 ${[...groups].map(([text, ms]) => `${ms.join(" · ")}: ${text}`).join(" / ")}` : null;
  }
  if (!p.options || typeof p.options !== "object") return null;
  return `옵션 ${Object.entries(p.options).map(([m, opts]) => `${m}: ${Object.entries(opts ?? {}).map(([k, v]) => `${k} = ${v}`).join(", ")}`).join(" · ")}`;
}

// Rows for the layer's errors in the tricast report of the method's uniform recipe: [weight MSE, output SQNR,
// output cosine], the output compared with the original Linear on the same input. Nothing when not reported.
function reportRows(report) {
  if (!Array.isArray(report)) return [];
  const [mse, sqnr, cos] = report;
  return [
    finite(mse) ? ["가중치 MSE (균일 리포트)", mse.toExponential(2)] : null,
    finite(sqnr) || finite(cos)
      ? ["층 출력 SQNR · cos", `${finite(sqnr) ? `${sqnr.toFixed(1)} dB` : "—"} · ${finite(cos) ? cos.toFixed(5) : "—"}`]
      : null,
  ];
}

// Rows for a layer's profile (weights, input activations, output error of pruning), only for the fields present.
// outlier_channels is the fraction of input channels whose activation absmax exceeds 6 × their median.
function profileRows(profile, inFeatures) {
  if (!profile || typeof profile !== "object") return [];
  const w = profile.weight ?? {};
  const a = profile.activation ?? {};
  const p = profile.sparsity ?? {};
  const pair = (x, y) => (finite(x) || finite(y)
    ? `크기 ${finite(x) ? x.toPrecision(3) : "—"} · Wanda ${finite(y) ? y.toPrecision(3) : "—"}` : null);
  const rows = [
    finite(w.kurtosis) ? ["가중치 첨도", w.kurtosis.toFixed(1)] : null,
    finite(w.outlier_3sigma) ? ["3σ 밖 가중치", `${(w.outlier_3sigma * 100).toFixed(2)}%`] : null,
    finite(a.channel_absmax_ratio) ? ["입력 채널 absmax 비", `${a.channel_absmax_ratio.toFixed(1)}×`] : null,
    finite(a.outlier_channels) ? ["중앙값 6× 넘는 입력 채널", `${(a.outlier_channels * 100).toFixed(2)}%`
      + (finite(inFeatures) ? ` (${Math.round(a.outlier_channels * inFeatures)}/${inFeatures})` : "")] : null,
    pair(p.magnitude_50, p.wanda_50) ? ["50% 가지치기 오차", pair(p.magnitude_50, p.wanda_50)] : null,
    pair(p.magnitude_2of4, p.wanda_2of4) ? ["2:4 가지치기 오차", pair(p.magnitude_2of4, p.wanda_2of4)] : null,
  ];
  const hist = w.histogram;
  if (hist && Array.isArray(hist.fraction) && hist.fraction.length && Array.isArray(hist.edges)
    && hist.edges.length === hist.fraction.length + 1) rows.push(["가중치 분포", sparkline(hist)]);
  return rows.filter(Boolean);
}

// The weight histogram as bars in the tooltip's own text color, with the range ends and the mass outside it.
function sparkline(hist) {
  const n = hist.fraction.length;
  const width = 132;
  const height = 22;
  const bar = (width - (n - 1)) / n;
  const top = Math.max(...hist.fraction.filter(finite), 1e-12);
  const ends = [hist.edges[0], hist.edges[n]].map((v) => (finite(v) ? Number(v.toPrecision(2)).toString() : "?"));
  const svg = s("svg", {class: "at-spark", width, height: height + 11, viewBox: `0 0 ${width} ${height + 11}`, "aria-hidden": "true"},
    hist.fraction.map((f, i) => {
      const tall = finite(f) && f > 0 ? Math.max(1, (f / top) * height) : 0;
      return s("rect", {x: (i * (bar + 1)).toFixed(2), y: (height - tall).toFixed(2), width: bar.toFixed(2), height: tall.toFixed(2)});
    }),
    s("text", {x: 0, y: height + 9}, ends[0]), s("text", {x: width, y: height + 9, "text-anchor": "end"}, ends[1]));
  const outside = (finite(hist.below) ? hist.below : 0) + (finite(hist.above) ? hist.above : 0);
  return h("span", null, svg, outside > 0 ? h("span", {class: "muted"}, `범위 밖 ${(outside * 100).toFixed(2)}%`) : null);
}

// Every number beside the floorplan: bits, predicted KL, method mix, the best uniform recipe that fits and the
// recorded measurement when this exact assignment was measured.
function updateStats() {
  ui.bits.textContent = fmtBits(state.bits);
  ui.bitsSub.replaceChildren("예산 ", h("b", null, `${state.text}`), ` = ${frac(state.budget).toFixed(4)} 비트`);
  ui.kl.textContent = fmtKL(state.rawKL);
  const objective = table.objective === "kl" ? "목적값 Σ KL " : "목적값 Σ max(0, KL − floor) ";
  ui.klSub.replaceChildren(objective, h("b", null, fmtKL(state.solution.value)),
    " · 층마다 따로 잰 KL 의 합이라 실제 모델 KL 과 다릅니다");
  ui.klSub.title = table.floor_rule ?? "";

  const used = methods.map((m, j) => [m, state.counts[j]]).filter(([, n]) => n > 0);
  const total = used.reduce((sum, [, n]) => sum + n, 0) || 1;
  ui.bar.replaceChildren(...used.map(([m, n]) => h("span", {class: "at-seg", "data-f": info[m].family,
    style: `flex: ${n} 1 0`, title: `${m} ${n}층`})));
  ui.bar.setAttribute("aria-label", used.map(([m, n]) => `${m} ${n}층`).join(", "));
  ui.keys.replaceChildren(...used.map(([m, n]) => h("li", null, code(m), h("span", {class: "id"}, m),
    h("span", {class: "n"}, String(n)), h("span", {class: "at-muted"}, `${Math.round((n / total) * 100)}%`))));

  // best measured uniform recipe whose size fits the budget: lowest validation KL, else lowest test PPL
  const budget = state.budget;
  const fits = (row) => {
    try {
      const [p, q] = parseBits(row.bits_fraction);
      return p * budget[1] <= budget[0] * q;
    } catch {
      return finite(row.bits) && row.bits <= frac(budget) + 1e-12;
    }
  };
  const within = (bundle.baselines ?? []).filter(fits);
  const byKL = within.filter((row) => finite(row.kl)).sort((a, b) => a.kl - b.kl)[0];
  const best = byKL ?? within.filter((row) => finite(row.ppl)).sort((a, b) => a.ppl - b.ppl)[0];
  const uniformBody = best
    ? [h("p", {class: "at-line"}, h("span", {class: "at-mono"}, best.id), ` · ${fmtBits(best.bits)} 비트`),
      h("p", {class: "at-line"}, "검증 KL ", val(fmtKL(best.kl)), " ", recordBadge(best.kl_record ?? best.ppl_record, best.id),
        " · test PPL ", val(fmtPPL(best.ppl))),
      h("p", {class: "at-why"}, recordNote(best.kl_record ?? best.ppl_record))]
    : [h("p", {class: "at-none"}, NONE),
      h("p", {class: "at-why"}, (bundle.baselines ?? []).length
        ? "이 예산 이하 크기의 균일 레시피 측정 기록이 없습니다."
        : "번들에 균일 레시피 측정 기록이 없습니다.")];
  ui.uniform.replaceChildren(h("h3", null, "같은 크기 이하 균일 최선"), ...uniformBody);

  const matches = (bundle.solutions ?? []).filter((row) => sameGrid(row.grid, state.grid))
    .sort((a, b) => Number(hasKV(a)) - Number(hasKV(b)));
  if (matches.length) {
    ui.record.replaceChildren(h("h3", null, "이 배정의 측정", h("span", {class: "at-badge rec"}, "GPU 측정 기록")),
      ...matches.flatMap((row) => [
        h("p", {class: "at-line"}, h("span", {class: "at-mono"}, solutionLabel(row)), " · 검증 KL ", val(fmtKL(row.kl)), " ",
          recordBadge(row.kl_record, row.id), " · test PPL ", val(fmtPPL(row.ppl))),
        h("p", {class: "at-why"}, [recordNote(row.kl_record),
          ...(row.rejected ?? []).map((reason) => reason)].join(" · "))]));
  } else {
    ui.record.replaceChildren(h("h3", null, "이 배정의 측정"),
      h("p", {class: "at-why"}, "이 배정은 예측만 있습니다 (측정은 GPU 작업이 필요합니다)"));
  }
}

// ---------- 3 agent ----------
function agentStep() {
  if (!session) {
    return step(3, "AI 에이전트 개선", null,
      h("p", {class: "at-none"}, "AI 세션 기록 없음"),
      h("p", {class: "at-lead"}, "tune loop 세션을 돌리고 그 세션과 함께 번들을 다시 내보내면 여기에 Claude 의 제안과 측정이 나옵니다."),
      h("pre", {class: "code"}, "python -m app.tune_loop run --config <loop.json> --rounds 2 --per-round 4 --proposer claude-code"));
  }
  const rounds = session.rounds ?? [];
  const first = rounds.find((r) => r.llm?.started_utc)?.llm?.started_utc;
  const date = first ? first.slice(0, 10) : "날짜 기록 없음";
  const start = (bundle.solutions ?? []).find((row) => row.id === session.start);
  const best = session.best;
  const delta = best && finite(best.kl) && finite(start?.kl) ? ` (시작 대비 ${(((best.kl - start.kl) / start.kl) * 100).toFixed(1)}%)` : "";
  const pill = (label, value) => h("span", {class: "pill"}, label, " ", h("b", null, value));
  const budgetLabel = sessionBudget ? `${session.budget_bits} = ${frac(sessionBudget).toFixed(4)} 비트` : NONE;
  ui.agentNote = h("p", {class: "at-gridnote"});
  ui.ask = h("button", {type: "button", class: "at-btn primary"}, "Claude 에게 새 제안 받기");
  ui.ask.addEventListener("click", askClaude);
  ui.cancel = h("button", {type: "button", class: "at-btn", hidden: true}, "취소");
  ui.wait = h("span", {class: "at-wait", "aria-live": "polite"});
  ui.liveError = h("p", {class: "at-live-error", role: "alert", hidden: true});
  ui.rounds = h("div", {class: "at-rounds"}, rounds.map((round) => roundView(round, false)));
  ui.liveRounds = h("div", {class: "at-rounds"});
  return step(3, "AI 에이전트 개선", null,
    h("div", {class: "at-agent-head"}, h("span", {class: "at-badge stored"}, `저장된 응답 · ${date} 세션`),
      pill("제안 엔진", session.proposer ?? NONE), pill("세션", session.name ?? NONE), pill("예산", budgetLabel),
      pill("시작", session.start ?? NONE)),
    h("p", {class: "at-lead"}, "휴리스틱 배정을 시작점으로, Claude 가 ledger 기록 id 를 근거로 붙여 바꿀 칸과 옵션을 제안합니다. 예산·인용·수치 검사를 통과한 제안만 GPU 로 검증 KL 을 잽니다. 카드에 마우스를 올리면 바뀌는 칸이 격자에 점선으로 표시됩니다."),
    best ? h("p", {class: "at-line"}, "세션 최선 ", h("span", {class: "at-mono"}, best.name), " · 검증 KL ", val(fmtKL(best.kl)),
      delta, " · test PPL ", val(fmtPPL(best.ppl)), ` · ${fmtBits(best.avg_bits)} 비트`) : h("p", {class: "at-none"}, "측정된 제안 없음"),
    h("ul", {class: "at-spend", "aria-label": "라운드별 Claude 비용"}, rounds.map((round) => (isReplan(round)
      ? h("li", null, `라운드 ${round.round} · LLM 호출 없음 (휴리스틱 재배정)`)
      : h("li", null, `라운드 ${round.round} · `, h("b", null, fmtUsd(round.llm?.cost_usd)), " · ",
        h("b", null, fmtSec(round.llm?.duration_s)), " · ", (round.llm?.models ?? []).join(", ") || "모델 기록 없음")))),
    ui.agentNote,
    h("div", {class: "at-askrow"}, ui.ask, ui.cancel, ui.wait),
    ui.liveError, ui.rounds, ui.liveRounds);
}

function updateAgentNote() {
  if (!ui.agentNote) return;
  const differs = sessionBudget && !sameFrac(sessionBudget, state.budget);
  ui.agentNote.textContent = differs
    ? `지금 예산(${state.text})이 세션 예산(${session.budget_bits})과 다릅니다. 카드에 올리면 격자가 그 제안의 격자로 바뀝니다.`
    : "";
  ui.agentNote.hidden = !differs;
}

// A round with no LLM call whose proposal the knapsack made again under the best grid's options (tune_loop replan).
const REPLAN = "knapsack-replan";
const isReplan = (round) => !round.llm && (round.proposals ?? []).some((p) => p.author === REPLAN);

function roundView(round, live) {
  const llm = round.llm;
  const head = h("div", {class: "at-round-head"}, h("h3", null, `라운드 ${round.round ?? "?"}`),
    live ? h("span", {class: "at-badge live"}, "실시간 Claude 응답")
      : isReplan(round) ? h("span", {class: "at-badge stored"}, "휴리스틱 재배정 · LLM 없음")
        : h("span", {class: "at-badge stored"}, "저장된 응답"),
    llm ? h("span", {class: "at-muted"}, `${fmtUsd(llm.cost_usd)} · ${fmtSec(llm.duration_s)}`) : null,
    round.stop ? h("span", {class: "at-badge"}, "에이전트가 멈춤을 제안") : null);
  const analysis = round.analysis ? clamped(round.analysis, 3) : [h("p", {class: "at-none"}, "분석 기록 없음")];
  const cards = (round.proposals ?? []).map((p) => cardView(p, live));
  return h("section", {class: "at-round", "aria-label": `라운드 ${round.round ?? ""}`}, head, ...analysis,
    round.note ? h("p", {class: "at-live-note"}, round.note) : null,
    cards.length ? h("div", {class: "at-cards"}, cards) : h("p", {class: "at-none"}, "제안 기록 없음"));
}

// The check that refused a proposal, read off the checker's message (app/tune_agent.propose).
function refusedCheck(reason) {
  const text = String(reason ?? "");
  if (/exceed the budget|exceeds the compute budget/.test(text)) return ["예산 ✗", "평균 비트가 크기 예산을 넘습니다."];
  if (/cited ids not in the ledger|not ledger records|cites at least one/.test(text)) return ["인용 ✗", "인용한 기록 id 가 ledger 에 없습니다."];
  if (/rationale numbers not supported/.test(text)) return ["수치 ✗", "근거 글의 숫자 일부가 인용한 기록이나 제안 값에 없습니다."];
  if (/does not build/.test(text)) return ["격자 ✗", "이 격자로 레시피를 만들 수 없습니다."];
  return ["검사 ✗", "결정론 검사를 통과하지 못했습니다."];
}

function cardView(p, live) {
  const records = session?.records ?? {};
  const accepted = p.status === "accepted";
  const isBest = !live && session?.best?.name === p.name;
  const base = typeof p.base === "string" ? baseGrid(p.base) : null;
  const changes = validChanges(p);
  const card = h("article", {class: `at-card${accepted ? "" : " refused"}${isBest ? " best" : ""}`, tabindex: "0",
    "aria-label": `제안 ${p.name ?? ""}`});
  card.addEventListener("mouseenter", () => setHover(p));
  card.addEventListener("mouseleave", () => setHover(null));
  card.addEventListener("focusin", () => setHover(p));
  card.addEventListener("focusout", (event) => {
    if (!card.contains(event.relatedTarget)) setHover(null);
  });

  const status = accepted ? h("span", {class: "at-badge ok"}, "채택")
    : p.status === "refused" ? h("span", {class: "at-badge bad"}, "거부") : h("span", {class: "at-badge"}, "상태 기록 없음");
  const shownChanges = changes.length > 8 ? changes.slice(0, 6) : changes;  // a replan moves dozens of cells
  const changeList = changes.length
    ? h("ul", {class: "at-changes", "aria-label": "바꾸는 칸"}, shownChanges.map((c) => {
      const from = base?.[c.block]?.[kinds.indexOf(c.kind)];
      return h("li", null, `블록 ${c.block} ${c.kind}`, from ? code(from) : null, h("span", {class: "arrow"}, "→"), code(c.method),
        h("span", {class: "at-mono"}, c.method));
    }), shownChanges.length < changes.length
      ? h("li", {class: "at-muted"}, `… 외 ${changes.length - shownChanges.length}칸 · 카드에 올리면 격자에 점선`) : null)
    : h("p", {class: "at-card-sub"}, "칸 변경 없음");
  const optionText = optionSummary(p);
  const cites = (Array.isArray(p.cites) ? p.cites : []).map((id) => {
    const record = records[id];
    if (!record) return h("span", {class: "at-cite unknown", tabindex: "0", title: "ledger 에 없음"}, `${id} · ledger 에 없음`);
    return tipOn(h("span", {class: "at-cite", tabindex: "0"}, id), () => tipRows(id, [
      ["종류", record.kind ?? NONE], record.layer ? ["레이어", record.layer] : null, record.method ? ["방법", record.method] : null,
      ["값", `${record.metric ?? ""} ${fmtKL(record.value)}`], ["GPU", record.gpu ?? NONE], ["git", short(record.git_sha) ?? NONE]]));
  });
  let checks;
  if (accepted) {
    checks = h("div", {class: "at-checks", "aria-label": "검사"}, ["예산 ✓", "인용 ✓", "수치 ✓"].map((t) => h("span", {class: "at-badge ok"}, t)));
  } else if (p.status === "refused") {
    const [badge, text] = refusedCheck(p.reason);
    checks = h("div", {class: "at-checks"}, h("span", {class: "at-badge bad"}, badge),
      h("p", {class: "at-reason"}, text, h("code", null, p.reason ?? "사유 기록 없음")));
  }
  const predicted = p.predicted?.predicted_kl_sum;
  const predictedView = h("div", {class: "at-kl"}, h("span", {class: "k"}, "예측 KL (단층 합)"),
    finite(predicted) ? h("span", {class: "v"}, fmtKL(predicted))
      : h("span", {class: "v none"}, "예측 없음"),
    !finite(predicted) && p.predicted?.unpredicted ? h("span", {class: "s", title: p.predicted.unpredicted},
      /^method or KV options/.test(p.predicted.unpredicted) ? "옵션을 바꾼 제안입니다. 단층 표에는 기본 옵션 값만 있습니다."
        : p.predicted.unpredicted) : null);
  const klRecord = p.kl_id ? records[p.kl_id] : null;
  const measuredView = h("div", {class: "at-kl"}, h("span", {class: "k"}, "측정 검증 KL"),
    finite(p.kl) ? h("span", {class: "v"}, fmtKL(p.kl), recordBadge(klRecord, p.kl_id))
      : h("span", {class: "v none"}, live ? "측정 전 · GPU 작업 필요" : accepted ? "측정 기록 없음" : "측정하지 않음"),
    finite(p.kl) ? h("span", {class: "s"}, recordNote(klRecord, p.kl_id)) : null,
    finite(p.ppl) ? h("span", {class: "s"}, `test PPL ${fmtPPL(p.ppl)} · ${recordNote(p.ppl_id ? records[p.ppl_id] : null, p.ppl_id)}`) : null);
  const [rationale, more] = p.rationale ? clamped(p.rationale, 4) : [h("p", {class: "at-none"}, "근거 기록 없음"), null];
  card.append(...[
    h("div", {class: "at-card-head"}, h("span", {class: "at-card-name"}, p.name ?? "이름 없음"), status,
      isBest ? h("span", {class: "at-badge best"}, "세션 최선") : null),
    h("p", {class: "at-card-sub"}, "기준 ", h("b", null, p.base ?? NONE), finite(p.avg_bits) ? ` · ${p.avg_bits.toFixed(4)} 비트` : ""),
    p.author === REPLAN ? h("p", {class: "at-card-sub"}, "휴리스틱 재배정: 기준의 옵션으로 다시 잰 단층 표에서 같은 예산의 배낭을 다시 풂") : null,
    changeList,
    optionText ? h("p", {class: "at-opts"}, optionText) : null,
    rationale, more,
    cites.length ? h("div", {class: "at-cites"}, h("span", {class: "at-muted"}, "인용"), cites) : h("p", {class: "at-none"}, "인용 없음"),
    checks,
    h("div", {class: "at-klrow"}, predictedView, measuredView)].filter(Boolean));
  return card;
}

// Hovering a proposal outlines its changed cells. When the slider budget is not the session's, the floorplan shows
// the proposal's own grid meanwhile (outlining the current solution would point at the wrong assignment).
function setHover(p) {
  state.hover = p;
  for (const row of ui.cells) for (const cell of row) cell.classList.remove("hl");
  if (!p) {
    showGrid(state.grid, false);
    ui.note.classList.remove("on");
    ui.note.textContent = "칸에 마우스를 올리거나 방향키로 옮기면 단층 KL 과 대안을 봅니다.";
    return;
  }
  const changes = validChanges(p);
  const differs = sessionBudget && !sameFrac(sessionBudget, state.budget);
  const own = differs ? proposalGrid(p) : null;
  showGrid(own ?? state.grid, false);
  for (const c of changes) ui.cells[c.block][kinds.indexOf(c.kind)].classList.add("hl");
  ui.note.classList.add("on");
  const what = changes.length ? `바뀌는 칸 ${changes.length}개를 점선으로 표시합니다` : "칸은 그대로 두고 옵션만 바꿉니다";
  ui.note.textContent = own ? `제안 ${p.name} 의 격자 (세션 예산 ${session.budget_bits}) · ${what}`
    : differs ? `이 제안의 격자가 번들에 없어 지금 배정 위에 표시합니다 · ${what}` : `제안 ${p.name} · ${what}`;
}

// POST to a live Claude endpoint with the elapsed seconds and a cancel button showing. Resolves to the JSON body
// ({mode: "live", ...}) or throws an Error whose message is what to show: the server's {"error": {"message"}} when
// it sent one (with .code), else what happened and how to fix it. A cancel throws an AbortError.
async function postLive(path, payload, {button, cancel, wait, waiting, usual, what}) {
  const controller = new AbortController();
  const started = performance.now();
  const abort = () => controller.abort();
  button.disabled = true;
  cancel.hidden = false;
  cancel.addEventListener("click", abort);
  const tick = () => {
    wait.textContent = `${waiting} · ${Math.round((performance.now() - started) / 1000)}초 (${usual})`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    let response;
    try {
      response = await fetch(path, {method: "POST", headers: {"content-type": "application/json", accept: "application/json"},
        body: JSON.stringify(payload), signal: controller.signal});
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new Error(`서버에 닿지 못했습니다 (${error.message}). python -m app.server 가 켜져 있는지 확인하세요.`);
    }
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const error = new Error(body?.error?.message ?? `${what} API 가 HTTP ${response.status} 로 답했습니다. ${AGENT_SERVER}`);
      error.code = body?.error?.code ?? null;
      throw error;
    }
    if (!body || body.mode !== "live") throw new Error(`${what} API 의 응답에 mode: live 가 없습니다. 서버 버전을 확인하세요.`);
    return body;
  } finally {
    clearInterval(timer);
    cancel.removeEventListener("click", abort);
    button.disabled = false;
    cancel.hidden = true;
    wait.textContent = "";
  }
}

async function askClaude() {
  const base = session.best?.name ?? session.start;
  ui.liveError.hidden = true;
  try {
    const body = await postLive(PROPOSE, {base, count: 4}, {button: ui.ask, cancel: ui.cancel, wait: ui.wait,
      waiting: "Claude 응답을 기다리는 중", usual: "보통 1–4분", what: "제안"});
    if (!Array.isArray(body.proposals)) throw new Error("제안 API 의 응답에 proposals 가 없습니다. 서버 버전을 확인하세요.");
    for (const p of body.proposals) if (p?.name && !proposals.has(p.name)) proposals.set(p.name, p);
    state.liveRounds.push(body);
    ui.liveRounds.append(roundView(body, true));
    announce(`실시간 Claude 응답: 제안 ${body.proposals.length}개`);
  } catch (error) {
    ui.liveError.textContent = error.name === "AbortError" ? "요청을 취소했습니다. 저장된 응답은 그대로 있습니다."
      : error.code ? error.message : `${error.message} 지금 화면의 AI 제안은 저장된 응답입니다.`;
    ui.liveError.hidden = false;
  }
}

// ---------- 4 measured results ----------
function chartPoints() {
  const records = session?.records ?? {};
  const uni = (bundle.baselines ?? []).map((row) => ({series: "uni", id: row.id, method: row.method, bits: row.bits,
    kl: row.kl, ppl: row.ppl, klRecord: row.kl_record, pplRecord: row.ppl_record}));
  const heur = (bundle.solutions ?? []).map((row) => ({series: "heur", id: solutionLabel(row), kv: hasKV(row), bits: row.bits,
    kl: row.kl, ppl: row.ppl, klRecord: row.kl_record, pplRecord: row.ppl_record}));
  const ai = [];
  if (session?.best) {
    const p = proposals.get(session.best.name);
    ai.push({series: "ai", id: session.best.name, bits: session.best.avg_bits, kl: session.best.kl, ppl: session.best.ppl,
      klRecord: p?.kl_id ? records[p.kl_id] : null, klId: p?.kl_id, pplRecord: p?.ppl_id ? records[p.ppl_id] : null, pplId: p?.ppl_id});
  }
  return {uni, heur, ai};
}

const SERIES_NAME = {uni: "균일", heur: "휴리스틱", ai: "AI 최선"};

function chartStep() {
  const {uni, heur, ai} = chartPoints();
  const anyPPL = [...uni, ...heur, ...ai].some((pt) => finite(pt.ppl));
  state.metric = anyPPL ? "ppl" : "kl";
  ui.metric = h("div", {class: "seg compact", role: "radiogroup", "aria-label": "지표"});
  for (const [id, label] of [["ppl", "test PPL"], ["kl", "검증 KL"]]) {
    const button = h("button", {type: "button", role: "radio", "data-metric": id}, label);
    button.addEventListener("click", () => {
      state.metric = id;
      drawChart();
    });
    ui.metric.append(button);
  }
  ui.metric.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    state.metric = state.metric === "ppl" ? "kl" : "ppl";
    drawChart();
    ui.metric.querySelector(`[data-metric="${state.metric}"]`).focus();
  });
  ui.tableToggle = h("button", {type: "button", class: "at-btn", "aria-pressed": "false"}, "표로 보기");
  ui.tableToggle.addEventListener("click", () => {
    state.tableView = !state.tableView;
    drawChart();
  });
  ui.legend = h("div", {class: "at-legend"});
  ui.chart = h("div", {class: "at-chart-wrap"});
  let chartWidth = 0;
  new ResizeObserver(() => {
    if (state.tableView || !bundle || ui.chart.clientWidth === chartWidth) return;
    chartWidth = ui.chart.clientWidth;
    drawChart();
  }).observe(ui.chart);
  return step(4, "측정 결과", null,
    h("p", {class: "at-lead"}, "같은 크기에서 어느 쪽이 나은지 GPU 측정값으로 봅니다. 가로축은 평균 비트, 세로축은 고른 지표(로그 축)입니다."),
    h("div", {class: "at-chart-tools"}, ui.metric, ui.tableToggle),
    ui.legend, ui.chart,
    h("p", {class: "at-note"}, "모든 점은 GPU 기록값입니다. 점에 마우스를 올리거나 표로 보면 기록마다 GPU 와 git sha 가 나옵니다."));
}

const legendMark = (kind) => {
  if (kind === "uni") return s("svg", {width: 12, height: 12}, s("circle", {cx: 6, cy: 6, r: 4.5, fill: "var(--at-uni)"}));
  if (kind === "heur") {
    return s("svg", {width: 22, height: 12}, s("line", {x1: 1, y1: 6, x2: 21, y2: 6, stroke: "var(--at-heur)", "stroke-width": 2}),
      s("circle", {cx: 11, cy: 6, r: 4, fill: "var(--at-heur)"}));
  }
  if (kind === "ai") return s("svg", {width: 14, height: 14}, s("path", {d: starPath(7, 7, 6.5), fill: "var(--at-ai)"}));
  return s("svg", {width: 22, height: 12}, s("line", {x1: 1, y1: 6, x2: 21, y2: 6, stroke: "var(--ink-2)", "stroke-width": 1.5,
    "stroke-dasharray": "5 4"}));
};

function starPath(cx, cy, r) {
  const points = [];
  for (let i = 0; i < 10; i += 1) {
    const radius = i % 2 ? r * 0.45 : r;
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    points.push(`${(cx + radius * Math.cos(angle)).toFixed(2)},${(cy + radius * Math.sin(angle)).toFixed(2)}`);
  }
  return `M${points.join("L")}Z`;
}

// Ticks for a log axis over [lo, hi]: 1-2-5 per decade when the range spans a decade or more, else nice linear
// steps placed on the log scale.
function logTicks(lo, hi) {
  if (hi / lo >= 10) {
    for (const set of [[1], [1, 2, 5], [1, 2, 3, 5, 7]]) {
      const ticks = [];
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
        for (const m of set) {
          const v = m * 10 ** e;
          if (v >= lo && v <= hi) ticks.push(v);
        }
      }
      if (ticks.length >= 3) return ticks;
    }
  }
  const raw = (hi - lo) / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const stepSize = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw) ?? raw;
  const ticks = [];
  for (let v = Math.ceil(lo / stepSize) * stepSize; v <= hi + 1e-12; v += stepSize) ticks.push(Number(v.toPrecision(6)));
  return ticks;
}

function pointTip(pt) {
  const kl = [fmtKL(pt.kl), finite(pt.kl) ? recordNote(pt.klRecord, pt.klId) : null].filter(Boolean).join(" · ");
  const ppl = [fmtPPL(pt.ppl), finite(pt.ppl) ? recordNote(pt.pplRecord, pt.pplId) : null].filter(Boolean).join(" · ");
  return tipRows(`${pt.id} (${SERIES_NAME[pt.series]})`, [["비트", fmtBits(pt.bits)], ["검증 KL", kl], ["test PPL", ppl],
    ["", "GPU 기록값"]]);
}

function drawChart() {
  for (const button of ui.metric.querySelectorAll("[role=radio]")) {
    const on = button.dataset.metric === state.metric;
    button.setAttribute("aria-checked", String(on));
    button.tabIndex = on ? 0 : -1;
  }
  ui.tableToggle.setAttribute("aria-pressed", String(state.tableView));
  ui.tableToggle.textContent = state.tableView ? "차트로 보기" : "표로 보기";
  const metric = state.metric;
  const {uni, heur, ai} = chartPoints();
  const native = bundle.native;
  const counts = (list) => list.filter((pt) => finite(pt[metric]) && pt[metric] > 0 && finite(pt.bits)).length;
  const nativeLabel = metric === "ppl" ? (finite(native?.ppl) ? `BF16 원본 PPL ${fmtPPL(native.ppl)}` : "BF16 원본 · 기록 없음")
    : "BF16 원본 · KL 0 (로그 축 밖)";
  ui.legend.replaceChildren(
    h("span", null, legendMark("uni"), `균일 (한 방법을 모든 층에)${counts(uni) ? "" : " · 기록 없음"}`),
    h("span", null, legendMark("heur"), `휴리스틱 (정확 배낭)${counts(heur) ? "" : " · 기록 없음"}`),
    h("span", null, legendMark("ai"), `AI 에이전트 최선${counts(ai) ? "" : " · 기록 없음"}`),
    h("span", null, legendMark("native"), nativeLabel));
  if (state.tableView) {
    drawTable([...uni, ...heur, ...ai]);
    return;
  }
  const valid = (pt) => finite(pt[metric]) && pt[metric] > 0 && finite(pt.bits);
  const all = [...uni, ...heur, ...ai].filter(valid);
  if (!all.length) {
    ui.chart.replaceChildren(h("p", {class: "at-chart-empty"},
      `${metric === "ppl" ? "test PPL" : "검증 KL"} 측정 기록이 없습니다. 다른 지표를 고르거나 GPU 측정 뒤 번들을 다시 내보내세요.`));
    return;
  }
  const width = Math.max(300, ui.chart.clientWidth || 640);
  const height = width < 520 ? 250 : 300;
  const m = {l: 50, r: 16, t: 22, b: 38};
  const showNative = metric === "ppl" && finite(native?.ppl) && native.ppl > 0;
  const values = all.map((pt) => pt[metric]).concat(showNative ? [native.ppl] : []);
  let lo = Math.min(...values) / 1.08;
  let hi = Math.max(...values) * 1.08;
  if (hi / lo < 1.3) [lo, hi] = [Math.sqrt(lo * hi) / 1.18, Math.sqrt(lo * hi) * 1.18];
  const bits = all.map((pt) => pt.bits);
  let x0 = Math.min(...bits) - 0.3;
  let x1 = Math.max(...bits) + 0.3;
  if (x1 - x0 < 1) [x0, x1] = [(x0 + x1) / 2 - 0.5, (x0 + x1) / 2 + 0.5];
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (width - m.l - m.r);
  const Y = (v) => m.t + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (height - m.t - m.b);
  const svg = s("svg", {class: "at-chart", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": `평균 비트 대 ${metric === "ppl" ? "test PPL" : "검증 KL"} (로그 축)`});
  // grid and axes
  for (const v of logTicks(lo, hi)) {
    svg.append(s("line", {class: "grid", x1: m.l, x2: width - m.r, y1: Y(v), y2: Y(v)}),
      s("text", {x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end"}, String(Number(v.toPrecision(3)))));
  }
  const xStep = x1 - x0 > 4 ? 1 : 0.5;
  for (let v = Math.ceil(x0 / xStep) * xStep; v <= x1 + 1e-9; v += xStep) {
    svg.append(s("line", {class: "grid", x1: X(v), x2: X(v), y1: m.t, y2: height - m.b}),
      s("text", {x: X(v), y: height - m.b + 15, "text-anchor": "middle"}, String(v)));
  }
  svg.append(s("line", {class: "axis", x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b}),
    s("text", {x: width - m.r, y: height - 6, "text-anchor": "end"}, "평균 비트 / 가중치"),
    s("text", {x: m.l, y: 12, "text-anchor": "start"}, `${metric === "ppl" ? "test PPL" : "검증 KL"} · 로그 축`));
  const placed = [];
  const label = (text, x, y, cls, anchor = "start") => {
    const w = text.length * 6.6;
    let left = anchor === "start" ? x : x - w;
    if (left + w > width - 2) [left, anchor, x] = [x - w - 16, "end", x - 16];
    let top = y - 9;
    for (let tries = 0; tries < 4 && placed.some((b) => left < b.r && left + w > b.l && top < b.b && top + 12 > b.t); tries += 1) top += 13;
    if (placed.some((b) => left < b.r && left + w > b.l && top < b.b && top + 12 > b.t)) return;
    placed.push({l: left, r: left + w, t: top, b: top + 12});
    svg.append(s("text", {class: cls, x, y: top + 9, "text-anchor": anchor}, text));
  };
  if (showNative) {
    svg.append(s("line", {class: "native", x1: m.l, x2: width - m.r, y1: Y(native.ppl), y2: Y(native.ppl)}));
    label(`BF16 원본 ${fmtPPL(native.ppl)}`, width - m.r - 2, Y(native.ppl) - 6, "lbl", "end");
  }
  // heuristic line: the weight-only solutions (no KV variant) by bits, the best value where a size repeats; the
  // others stay labeled dots
  const heurPts = heur.filter(valid).sort((a, b) => a.bits - b.bits);
  const line = [];
  for (const pt of heurPts.filter((item) => !item.kv)) {
    const last = line[line.length - 1];
    if (last && Math.abs(last.bits - pt.bits) < 1e-9) {
      if (pt[metric] < last[metric]) line[line.length - 1] = pt;
    } else line.push(pt);
  }
  if (line.length > 1) svg.append(s("polyline", {class: "heur-line", points: line.map((pt) => `${X(pt.bits)},${Y(pt[metric])}`).join(" ")}));
  const marks = [];
  for (const pt of uni.filter(valid)) marks.push([pt, s("circle", {class: "pt-uni", cx: X(pt.bits), cy: Y(pt[metric]), r: 4.5})]);
  for (const pt of heurPts) marks.push([pt, s("circle", {class: "pt-heur", cx: X(pt.bits), cy: Y(pt[metric]), r: 4.5})]);
  for (const pt of ai.filter(valid)) marks.push([pt, s("path", {class: "pt-ai", d: starPath(X(pt.bits), Y(pt[metric]), 8)})]);
  for (const [, mark] of marks) svg.append(mark);
  // direct labels: the series first, then up to 6 uniform recipes on their own frontier (no smaller and better one)
  const aiPt = ai.find(valid);
  if (aiPt) label("AI 최선", X(aiPt.bits) + 11, Y(aiPt[metric]) + 14, "series-lbl");
  if (line.length) {
    const end = line[line.length - 1];
    label("휴리스틱", X(end.bits) + 9, Y(end[metric]) + 4, "series-lbl");
  }
  // a solution off the line (a KV variant, or the same size measured again) is named, not left anonymous
  for (const pt of heurPts.filter((pt) => !line.includes(pt))) label(pt.id, X(pt.bits) + 8, Y(pt[metric]) + 4, "lbl");
  const frontier = uni.filter(valid).sort((a, b) => a.bits - b.bits)
    .filter((pt, i, list) => !list.some((other, j) => j !== i && other.bits <= pt.bits && other[metric] < pt[metric]));
  for (const pt of frontier.slice(0, 6)) label(pt.method ?? pt.id, X(pt.bits) + 8, Y(pt[metric]) + 4, "lbl");
  // hit targets larger than the marks, focusable for the keyboard
  for (const [pt] of marks) {
    const hit = s("circle", {class: "hit", cx: X(pt.bits), cy: Y(pt[metric]), r: 11, tabindex: "0",
      "aria-label": `${SERIES_NAME[pt.series]} ${pt.id}: ${fmtBits(pt.bits)} 비트, 검증 KL ${fmtKL(pt.kl)}, test PPL ${fmtPPL(pt.ppl)}`});
    tipOn(hit, () => pointTip(pt));
    svg.append(hit);
  }
  ui.chart.replaceChildren(svg);
}

function drawTable(points) {
  const native = bundle.native;
  const row = (series, id, bits, kl, ppl, record, recordId) => h("tr", null, h("td", null, series), h("td", {class: "mono"}, id),
    h("td", {class: "num"}, bits), h("td", {class: "num"}, kl), h("td", {class: "num"}, ppl),
    h("td", {class: "mono"}, gpuName(record?.gpu) ?? (recordId ? `기록 ${recordId}` : NONE)),
    h("td", {class: "mono"}, short(record?.git_sha) ?? NONE));
  const rows = points.map((pt) => row(SERIES_NAME[pt.series], pt.id, fmtBits(pt.bits), fmtKL(pt.kl), fmtPPL(pt.ppl),
    pt.klRecord ?? pt.pplRecord, pt.klId ?? pt.pplId));
  rows.push(row("BF16 원본", "native", "16.0000", "0 (정의)", fmtPPL(native?.ppl), native, null));
  ui.chart.replaceChildren(h("div", {class: "at-tablewrap"}, h("table", {class: "at-table"},
    h("caption", {class: "sr-only"}, "측정 결과 표 (GPU 기록값)"),
    h("thead", null, h("tr", null, ["계열", "id", "비트", "검증 KL", "test PPL", "GPU", "git sha"].map((t, i) =>
      h("th", {class: i >= 2 && i <= 4 ? "num" : null, scope: "col"}, t)))),
    h("tbody", null, rows))));
}

// ---------- 5 reproduce ----------
function commandStep() {
  ui.cmds = h("ul", {class: "at-cmds"});
  ui.cmdNote = h("p", {class: "at-note", hidden: true},
    "요청 제약(BF16 유지 블록 · 제외 · 허용 방법)은 solve 명령에 해당 옵션이 없어, 첫 명령은 제약 없는 표를 풉니다.");
  return step(5, "재현", null, ui.cmds, ui.cmdNote,
    h("p", {class: "at-note"}, "실시간 제안 버튼은 python -m app.server --demo --agent <loop.json> 으로 띄운 서버에서만 동작합니다 (로그인된 claude CLI 필요)."));
}

function commands() {
  const budget = state.text.replace(/\s+/g, "");
  return [
    ["지금 예산의 배정을 Python 으로 풀기 (같은 DP · 같은 결과)",
      `python -m app.autotune solve --table ${baseTable.path ?? "<table.json>"} --budget-bits ${budget} --out runs/auto_${(state.bits ?? 0).toFixed(2)}`],
    ["AI 세션 돌리기 (분석 → Claude 제안 → 검사 → GPU 검증 KL 측정)",
      "python -m app.tune_loop run --config <loop.json> --rounds 2 --per-round 4 --proposer claude-code"],
    ["이 화면 띄우기", "python -m app.server --demo"],
  ];
}

function updateCommands() {
  ui.cmdNote.hidden = !state.constraints;
  ui.cmds.replaceChildren(...commands().map(([label, command], i) => h("li", null, h("span", {class: "at-cmd-label"}, label),
    h("code", {class: "at-cmd"}, command), copyButton("복사", () => commands()[i][1]))));
}

// ---------- boot ----------
function render() {
  const left = h("div", {class: "at-left"}, budgetStep(), planStep());
  const right = h("div", {class: "at-right"}, agentStep(), chartStep(), commandStep());
  root.replaceChildren(contextStrip(), h("div", {class: "at-layout"}, left, right));
  layoutTicks();
  let tickWidth = ui.ticks.clientWidth;
  new ResizeObserver(() => {
    if (ui.ticks.clientWidth !== tickWidth) {
      tickWidth = ui.ticks.clientWidth;
      layoutTicks();
    }
  }).observe(ui.ticks);
}

function failure(title, message, command) {
  const parts = [h("h2", null, title), h("p", null, message)];
  if (command) parts.push(h("pre", {class: "code"}, command), copyButton("명령 복사", command));
  root.replaceChildren(h("section", {class: "state", role: "alert"}, parts));
}

function useBundle(data) {
  bundle = data;
  baseTable = data.table;
  table = baseTable;
  methods = [...table.methods];
  kinds = [...table.kinds];
  blocks = table.blocks;
  const taken = new Set();
  info = {};
  for (const method of methods) {
    info[method] = describe(method, taken);
    taken.add(info[method].code);
    info[method].uniform = uniformBits(method);
  }
  layerAt = Array.from({length: blocks}, () => new Array(kinds.length));
  table.layers.forEach((layer, i) => {
    const k = kinds.indexOf(layer.kind);
    if (layer.block >= 0 && layer.block < blocks && k >= 0) layerAt[layer.block][k] = i;
  });
  computeCheapest();
  session = data.session ?? null;
  proposals = new Map();
  for (const round of session?.rounds ?? []) for (const p of round.proposals ?? []) if (p?.name) proposals.set(p.name, p);
  try {
    sessionBudget = session?.budget_bits ? parseBits(session.budget_bits) : null;
  } catch {
    sessionBudget = null;
  }
}

async function boot() {
  window.__autotune = {
    setBudget: (text) => {
      ui.input.value = text;
      runSolve(text, true);
      return window.__autotune.state();
    },
    state: () => ({text: state.text, bits: state.bits, value: state.solution?.value ?? null, rawKL: state.rawKL,
      counts: Object.fromEntries(methods.map((m, j) => [m, state.counts[j]]).filter(([, n]) => n)), error: state.error,
      ms: state.ms, grid: state.grid, matched: (bundle.solutions ?? []).filter((row) => sameGrid(row.grid, state.grid)).map((row) => row.id)}),
    hover: (name) => setHover(name ? proposals.get(name) ?? null : null),
    constrained: (request) => constrained(baseTable, request),
    showRequest: (item, live = false) => showRequest(item, live),
    constraints: () => state.constraints,
    sliderMin: () => Number(ui.slider.min) / UNITS,
    cheapest: () => cheapestBits,
  };
  let response;
  try {
    response = await fetch(DATA, {headers: {accept: "application/json"}});
  } catch (error) {
    failure("Auto-tune 데이터를 읽지 못했습니다", `${DATA}: ${error.message}`, EXPORT);
    return;
  }
  if (response.status === 404) {
    failure("Auto-tune 데이터가 아직 없습니다",
      "레이어별 KL 표(table.json)를 번들로 내보내면 이 화면이 열립니다. 저장소 루트에서 실행하세요.", EXPORT);
    return;
  }
  let data;
  try {
    if (!response.ok) throw new Error(`${DATA} 요청이 HTTP ${response.status} 로 끝났습니다. 번들을 다시 내보내세요.`);
    data = await response.json();
    if (data.schema !== SCHEMA) throw new Error(`bundle.json 형식이 ${data.schema ?? "알 수 없음"} 입니다. ${SCHEMA} 형식으로 다시 내보내세요.`);
    if (!data.table?.layers?.length) throw new Error("bundle.json 에 KL 표(table.layers)가 없습니다. 표를 지정해 다시 내보내세요.");
  } catch (error) {
    failure("Auto-tune 데이터를 열지 못했습니다", error.message, EXPORT);
    return;
  }
  useBundle(data);
  render();
  const first = session?.budget_bits ?? bundle.solutions?.[0]?.budget_bits ?? "4.5";
  ui.input.value = first;
  runSolve(first, true);
  if (state.error) {
    const fallback = budgetText([BigInt(Math.max(cheapestUnits, Math.round(4.5 * UNITS))), BigInt(UNITS)]);
    ui.input.value = fallback;
    runSolve(fallback, true);
  }
  drawChart();
}

if (location.protocol !== "file:") boot();
