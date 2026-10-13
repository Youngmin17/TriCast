// Layer map: the error of every Linear layer under every recorded quantization method, side by side.
// Reads the bundle app/export_layers.py writes (demo/layers/index.json + one data file per report).
import {copyText, fmt, h, radioGroup, s} from "./ui.js";

const SCHEMA = "tricast.layer_map/1";
const BUNDLE = "demo/layers/";
const COMMAND = "python -m app.export_layers runs/<실행>/report --out app/web/demo/layers --label <레시피>=<표시 이름>";
// Strip geometry in px: row height, gap between rows, marker band above the rows, axis band below them.
const ROW = 24;
const GAP = 4;
const TOP = 12;
const AXIS = 24;
const PITCH = [4, 12];     // column pitch range (cell + 1 px gap)
const GROUP_GAP = [2, 6];  // extra space between blocks (or Linear kinds)

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
const minus = (text) => text.replaceAll("-", "−");
const finite = (body) => (v) => (v === Infinity ? "∞" : v === -Infinity ? "−∞" : minus(body(v)));
const fixed = (digits) => finite((v) => v.toFixed(digits));
// |Δ| with two decimals, switching to an exponent below 0.01 so a near tie never reads as "0.00".
const step2 = finite((v) => (v >= 0.01 ? v.toFixed(2) : v.toExponential(1)));
const compact = finite((v) => (v === 0 ? "0" : Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toExponential(2)));
const sci = finite((v) => (v === 0 ? "0" : v.toExponential(2)));
const sig = finite((v) => (v === 0 ? "0" : String(Number(v.toPrecision(4)))));
const int = finite((v) => String(v));
const safe = (show) => (v) => (v == null ? "—" : Number.isFinite(v) ? show(v) : v > 0 ? "∞" : "−∞");

// better: the direction with less error. log: draw log10 of the value. nines: color by -log10(1 - x), which spreads
// cosines that crowd near 1 (the numbers shown stay plain cosines). step formats |Δ| when show is too coarse.
const METRICS = [
  {id: "out_sqnr", label: "출력 SQNR (dB)", short: "출력 SQNR", col: "output.sqnr_db", better: "high", unit: "dB", show: fixed(2), step: step2},
  {id: "out_mse", label: "출력 MSE (log10)", short: "log10 출력 MSE", col: "output.mse", better: "low", unit: "", log: true,
    show: fixed(2), step: step2},
  {id: "out_cos", label: "출력 코사인", short: "출력 코사인", col: "output.cosine", better: "high", unit: "", nines: true,
    show: fixed(5), step: compact},
  {id: "w_sqnr", label: "가중치 SQNR", short: "가중치 SQNR", col: "weight.sqnr_db", better: "high", unit: "dB", show: fixed(2), step: step2},
  {id: "a_sqnr", label: "활성 SQNR", short: "활성 SQNR", col: "activation.sqnr_db", better: "high", unit: "dB", show: fixed(2), step: step2},
  {id: "rel_fro", label: "상대 Frobenius", short: "출력 상대 Frobenius", col: "output.relative_frobenius", better: "low",
    unit: "", show: compact},
  {id: "ulp_mean", label: "MMA ULP 평균", short: "MMA ULP 평균", col: "mma_ulp.mean", better: "low", unit: "ULP", show: compact},
];

// The table shows every exported metric of one method.
const TABLE = [
  ...[["weight", "가중치"], ["activation", "활성"], ["output", "출력"]].flatMap(([group, label]) => [
    ["sqnr_db", "SQNR (dB)", fixed(2)], ["mse", "MSE", sci], ["max_abs_error", "최대 |오차|", sig],
    ["relative_frobenius", "상대 Frobenius", compact], ["cosine", "코사인", fixed(5)],
  ].map(([key, name, show], i) => ({col: `${group}.${key}`, group: label, name, show, first: i === 0}))),
  ...[["mean", "평균", compact], ["max", "최대", int], ["p99", "p99", compact], ["exact_fraction", "정확 비율", fixed(4)]]
    .map(([key, name, show], i) => ({col: `mma_ulp.${key}`, group: "MMA ULP", name, show, first: i === 0})),
];

const root = document.getElementById("map");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const state = {metric: METRICS[0].id, scale: "abs", order: "block", hidden: new Set(), pin: null, hover: null,
  table: null, sort: null};
let model = null;     // {reports, layers, kinds}
let view = null;      // what the current state shows: order, values, per-layer medians, color domain
let geo = null;       // strip geometry: column x, pitch, crosshair groups
let palette = null;   // theme ramps from layers.css
let readoutRows = [];
let readoutHead = null;

// ---------- data ----------
const TOKENS = {inf: Infinity, "-inf": -Infinity};

async function getJSON(url) {
  const response = await fetch(url, {headers: {accept: "application/json"}});
  if (!response.ok) throw Object.assign(new Error(`${url} 요청이 ${response.status} 로 끝났습니다`), {status: response.status});
  return response.json();
}

function prepare(entry, data) {
  if (data.schema !== SCHEMA) throw new Error(`${entry.file} 의 형식이 ${data.schema} 입니다 (기대: ${SCHEMA})`);
  return {...entry, rows: data.rows, specs: data.specs, spec: data.spec, kinds: data.kinds,
    col: Object.fromEntries(data.columns.map((name, i) => [name, i])),
    at: new Map(data.rows.map((row, i) => [row[0], i])),
    cellTokens: new Map(data.nonfinite.map(([row, col, token]) => [`${row} ${col}`, token]))};
}

// One list of layers for all methods: first-seen order, which is the exporter's block order.
function buildModel(reports) {
  const byName = new Map();
  const kinds = [];
  for (const report of reports) {
    for (const kind of report.kinds) if (!kinds.includes(kind)) kinds.push(kind);
    for (const [name, block, kind] of report.rows) {
      if (!byName.has(name)) byName.set(name, {name, block, kind, pos: byName.size});
    }
  }
  return {reports, layers: [...byName.values()], kinds};
}

// A number, ±Infinity (the report's "inf" strings), or null (not measured / NaN).
function raw(report, i, col) {
  const value = report.rows[i][report.col[col]];
  return value ?? TOKENS[report.cellTokens.get(`${i} ${col}`)] ?? null;
}

function valueOf(metric, report, name) {
  const i = report.at.get(name);
  const v = i == null ? null : raw(report, i, metric.col);
  if (v == null) return null;
  const x = metric.log ? Math.log10(v) : v;
  return Number.isNaN(x) ? null : x;
}

const modelValue = (report, key) => report.metrics[key] ?? TOKENS[report.nonfinite?.[key]] ?? null;
const good = (metric, v) => (metric.better === "high" ? v : -v);
const colorX = (metric, v) => (metric.nines ? (v >= 1 ? Infinity : -Math.log10(1 - v)) : v);
const fromX = (metric, x) => (metric.nines ? 1 - 10 ** -x : x);
const deltaOf = (v, med) => (v == null || med == null ? null : v === med ? 0 : v - med);

function median(values) {
  const xs = values.filter((v) => v != null).sort(cmp);
  if (!xs.length) return null;
  const m = xs.length >> 1;
  if (xs.length % 2 || xs[m - 1] === xs[m]) return xs[m];
  const mid = (xs[m - 1] + xs[m]) / 2;
  return Number.isNaN(mid) ? null : mid;
}

// Linear interpolation between order statistics of an ascending list (numpy's default).
function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const x = (sorted.length - 1) * q;
  const i = Math.floor(x);
  return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * (x - i) : sorted[i];
}

// Both medians run over every method and every layer in the bundle, and so does the within-method clip:
// hiding a method never repaints the others.
function computeView() {
  const metric = METRICS.find((item) => item.id === state.metric);
  const list = [...model.layers];
  if (state.order === "kind") {
    const rank = new Map(model.kinds.map((kind, i) => [kind, i]));
    list.sort((a, b) => cmp(rank.get(a.kind), rank.get(b.kind)) || cmp(a.block ?? Infinity, b.block ?? Infinity) || a.pos - b.pos);
  }
  const values = new Map(model.reports.map((report) => [report.id, list.map((layer) => valueOf(metric, report, layer.name))]));
  const medians = list.map((_, c) => median(model.reports.map((report) => values.get(report.id)[c])));
  const deltas = new Map(model.reports.map((report) => [report.id, values.get(report.id).map((v, c) => deltaOf(v, medians[c]))]));
  const rowMedians = new Map(model.reports.map((report) => [report.id, median(values.get(report.id))]));
  const within = new Map(model.reports.map((report) => [report.id,
    values.get(report.id).map((v) => deltaOf(v, rowMedians.get(report.id)))]));
  const spread = [...within.values()].flat().filter((d) => d != null && Number.isFinite(d)).map(Math.abs).sort(cmp);
  const clip = quantile(spread, 0.98);
  const shown = model.reports.filter((report) => !state.hidden.has(report.id));
  let lo = Infinity;
  let hi = -Infinity;
  let span = 0;
  let exact = false;
  let missing = false;
  for (const report of shown) {
    values.get(report.id).forEach((v, c) => {
      if (v == null) {
        missing = true;
        return;
      }
      const x = colorX(metric, v);
      if (Number.isFinite(x)) {
        lo = Math.min(lo, x);
        hi = Math.max(hi, x);
      } else if (good(metric, x) > 0) exact = true;
      const g = good(metric, deltas.get(report.id)[c]);
      if (Number.isFinite(g)) span = Math.max(span, Math.abs(g));
    });
  }
  return {metric, list, index: new Map(list.map((layer, c) => [layer.name, c])), shown, values, medians, deltas,
    rowMedians, within, clip, lo, hi, span, exact, missing};
}

// ---------- color ----------
const rgbOf = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const luminance = (hex) => rgbOf(hex).reduce((sum, v, k) => sum + v * [0.2126, 0.7152, 0.0722][k], 0);

function readPalette() {
  const css = getComputedStyle(document.documentElement);
  const list = (name) => css.getPropertyValue(name).split(",").map((hex) => hex.trim()).filter(Boolean);
  return {seq: list("--lm-seq"), div: list("--lm-div")};
}

function mix(stops, t) {
  const x = clamp(t, 0, 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const a = rgbOf(stops[i]);
  const b = rgbOf(stops[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * (x - i))).join(" ")})`;
}

// 0 = the worst value on screen, 1 = the best; an infinite value sits at its end of the scale.
function quality(v) {
  const metric = view.metric;
  const x = colorX(metric, v);
  if (!Number.isFinite(x)) return good(metric, x) > 0 ? 1 : 0;
  const t = view.hi > view.lo ? (x - view.lo) / (view.hi - view.lo) : 1;
  return metric.better === "high" ? t : 1 - t;
}

function cellColor(id, c) {
  if (view.values.get(id)[c] == null) return null;
  if (state.scale !== "abs") {
    // delta: same layer vs the method median (scaled to the largest |Δ|); within: this method vs its own layer
    // median, clipped symmetrically at the 98th percentile of |Δ|.
    const [deltas, span] = state.scale === "delta" ? [view.deltas, view.span] : [view.within, view.clip];
    const g = good(view.metric, deltas.get(id)[c]);
    const t = Number.isFinite(g) ? (span ? clamp(g / span, -1, 1) : 0) : Math.sign(g);
    return mix(palette.div, (t + 1) / 2);
  }
  return mix(palette.seq, 1 - quality(view.values.get(id)[c]));
}

// ---------- text ----------
const withUnit = (metric, text) => (metric.unit ? `${text} ${metric.unit}` : text);
const display = (metric, v) => (v == null ? "—" : withUnit(metric, metric.show(v)));
const displayDelta = (metric, d) => (d == null ? "—" : d === 0 ? "0"
  : withUnit(metric, `${d > 0 ? "+" : "−"}${(metric.step ?? metric.show)(Math.abs(d))}`));
const shortName = (layer) => (layer.block == null ? layer.name : `블록 ${layer.block} · ${layer.kind.split(".").pop()}`);
const kindLabel = (kind) => kind.split(".").pop().replace(/_proj$/, "");

function quantText(spec) {
  if (!spec) return "양자화 안 함";
  const unit = spec.granularity === "group" || spec.granularity === "block"
    ? `${spec.granularity} ${spec.group_size}` : spec.granularity;
  return `${spec.format} · ${unit}`;
}

function specOf(report, name) {
  const i = report.at.get(name);
  return i == null ? null : report.specs[report.spec[i]] ?? null;
}

// [weight format, activation format, algorithm · transform, MMA] for one layer of one method. The GEMM output
// format is named only for the ULP metric, whose unit it defines.
function specCells(spec, ulp = false) {
  if (!spec) return ["—", "—", "—", "—"];
  if (spec.skipped) return [`양자화 제외 (${spec.skipped})`, "—", "—", "—"];
  const sparse = spec.sparsity && spec.sparsity !== "none" ? ` · 희소 ${JSON.stringify(spec.sparsity).replaceAll("\"", "")}` : "";
  const algo = `${spec.weight_algo ?? "—"} · 변환 ${!spec.transform || spec.transform === "none" ? "없음" : spec.transform}${sparse}`;
  const m = spec.mma;
  // F (and G for GDFS) only parameterize CoFDA and GDFS; other algorithms record them without meaning.
  const bits = m?.algorithm === "cofda" ? ` F${m.f_bits}` : m?.algorithm === "gdfs" ? ` F${m.f_bits} G${m.g_bits}` : "";
  const mma = m ? `${m.algorithm}${bits}${m.name && m.name !== m.algorithm ? ` · ${m.name}` : ""}`
    + (ulp && spec.ulp_format ? ` · 출력 ${spec.ulp_format}` : "") : "—";
  return [quantText(spec.weight), quantText(spec.activation), algo, mma];
}

// ---------- page ----------
function render(focusKey) {
  const key = focusKey ?? document.activeElement?.dataset?.key;
  palette = readPalette();
  view = computeView();
  state.hover = null;
  readoutRows = [];
  root.replaceChildren(h("div", {class: "lm-grid"},
    h("div", {class: "lm-main"}, controls(), mapPanel(), tablePanel()), sidePanel()), provenance());
  drawStrip();
  showColumn();
  reveal(view.index.get(state.pin), false);
  if (key) root.querySelector(`[data-key="${CSS.escape(key)}"]`)?.focus();
}

function controls() {
  const pick = (key) => (id) => {
    state[key] = id;
    render(`${key}:${id}`);
  };
  const choice = (key, id, text) => ({id, render: () => h("button", {type: "button", "data-key": `${key}:${id}`}, text)});
  const shownCount = model.reports.length - state.hidden.size;
  return h("section", {class: "lm-controls", "aria-label": "보기 설정"},
    h("label", {class: "lm-field"}, "지표",
      h("select", {"data-key": "metric", onchange: (event) => {
        state.metric = event.target.value;
        render("metric");
      }}, METRICS.map((m) => h("option", {value: m.id, selected: m.id === state.metric}, m.label)))),
    h("div", {class: "lm-field"}, h("span", null, "색 기준"),
      radioGroup("색 기준", [choice("scale", "abs", "절대값"), choice("scale", "delta", "같은 레이어 · 방법 중앙값 대비 Δ"),
        choice("scale", "within", "방법 안 레이어 중앙값 대비 Δ")], state.scale, pick("scale"), "seg compact")),
    h("div", {class: "lm-field"}, h("span", null, "레이어 순서"),
      radioGroup("레이어 순서", [choice("order", "block", "블록 순"), choice("order", "kind", "Linear 종류별")],
        state.order, pick("order"), "seg compact")),
    h("div", {class: "lm-field methods"}, h("span", {id: "lm-methods-label"}, "표시할 방법"),
      h("div", {class: "lm-methods", role: "group", "aria-labelledby": "lm-methods-label"}, model.reports.map((report) => {
        const on = !state.hidden.has(report.id);
        const last = on && shownCount === 1;
        return h("label", {class: "lm-check", title: last ? "적어도 한 방법은 보여야 합니다" : report.description || null},
          h("input", {type: "checkbox", checked: on, disabled: last, "data-key": `method:${report.id}`, onchange: (event) => {
            if (event.target.checked) state.hidden.delete(report.id);
            else state.hidden.add(report.id);
            render();
          }}), report.label);
      }))));
}

function legend() {
  const m = view.metric;
  const keys = [];
  let scale;
  if (state.scale !== "abs") {
    const within = state.scale === "within";
    const span = within ? view.clip : view.span;
    const worse = m.better === "high" ? -1 : 1;
    // The within-method scale is clipped, so its ends read "at most" / "at least".
    const end = (sign) => (span ? `${within ? (sign < 0 ? "≤ " : "≥ ") : ""}${displayDelta(m, sign * span)}` : "0");
    scale = h("span", {class: "lm-scale"},
      h("span", {class: "end"}, h("small", null, "나쁨"), end(worse)),
      h("span", {class: "lm-bar", style: `background: linear-gradient(90deg, ${palette.div.join(", ")})`}),
      h("span", {class: "end"}, end(-worse), h("small", null, " 좋음")),
      h("span", {class: "mid"}, within ? "0 = 이 방법의 레이어 중앙값" : "0 = 같은 레이어의 방법 중앙값"));
    keys.push(h("span", {class: "caption"}, within
      ? `각 방법의 값에서 그 방법 모든 레이어의 중앙값을 뺀 값 · 모든 방법 |Δ|의 98백분위 `
        + `(${withUnit(m, (m.step ?? m.show)(span))})에서 대칭으로 자름`
      :"같은 레이어에서 모든 방법의 중앙값을 뺀 값 · 가장 큰 |Δ|가 양 끝"));
  } else if (view.hi >= view.lo) {
    const [bestX, worstX] = m.better === "high" ? [view.hi, view.lo] : [view.lo, view.hi];
    const darkWorse = luminance(palette.seq[0]) > luminance(palette.seq.at(-1));
    scale = h("span", {class: "lm-scale"},
      h("span", {class: "end"}, h("small", null, "좋음"), display(m, fromX(m, bestX))),
      h("span", {class: "lm-bar", style: `background: linear-gradient(90deg, ${palette.seq.join(", ")})`}),
      h("span", {class: "end"}, display(m, fromX(m, worstX)), h("small", null, " 나쁨")),
      h("span", {class: "mid"}, display(m, fromX(m, (view.lo + view.hi) / 2))));
    keys.push(h("span", {class: "caption"},
      `${darkWorse ? "어두울수록" : "밝을수록"} 나쁨 · ${m.better === "high" ? "값이 낮을수록" : "값이 클수록"} 오차가 큽니다`));
  } else {
    scale = h("span", {class: "caption"}, "보이는 값이 모두 오차 0입니다");
  }
  if (view.exact && state.scale === "abs") {
    keys.unshift(h("span", {class: "lm-key"}, h("i", {class: "lm-sw", style: `background: ${palette.seq[0]}`}), "∞ = 오차 0"));
  }
  if (view.missing) keys.unshift(h("span", {class: "lm-key"}, h("i", {class: "lm-sw na"}), "값 없음"));
  return h("div", {class: "lm-legend", role: "group", "aria-label": "색 범례"}, scale, keys);
}

function mapPanel() {
  return h("section", {class: "panel lm-map", "aria-labelledby": "lm-map-title"},
    h("div", {class: "lm-map-head"}, h("h2", {id: "lm-map-title"}, `방법 × Linear 레이어 · ${view.metric.label}`), legend()),
    h("p", {class: "lead"}, "한 줄이 방법 하나, 한 칸이 Linear 레이어 하나입니다. 포인터를 올리거나 지도를 선택한 뒤 ←·→ 키로 "
      + "레이어를 옮기면 아래 판독값에 모든 방법의 값이 한 번에 나옵니다. 누르면 그 레이어에 고정됩니다. "
      + "Shift+←·→ 는 블록(또는 종류) 단위로 건너뜁니다."),
    h("div", {class: "lm-strip", style: `--row: ${ROW}px; --gap: ${GAP}px; --top: ${TOP}px`},
      h("div", {class: "lm-labels", "aria-hidden": "true"},
        view.shown.map((report) => h("div", {title: report.description || report.label}, report.label)),
        h("div", {class: "cap"}, state.order === "block" ? "블록" : "종류")),
      h("div", {class: "lm-scroll", id: "lm-scroll", tabindex: "0", role: "group", "data-key": "strip",
        "aria-label": `${view.metric.label} 지도. 왼쪽·오른쪽 화살표로 레이어를 옮깁니다`, "aria-describedby": "lm-readout-title",
        onkeydown: onStripKey})),
    readout(),
    h("p", {class: "sr-only", id: "lm-live", "aria-live": "polite"}));
}

function groupOf(layer) {
  return state.order === "kind" ? layer.kind : layer.block;
}

const marker = (kind) => s("g", {class: `xh ${kind}`, visibility: "hidden"},
  s("rect", {class: "xh-outer"}), s("rect", {class: "xh-inner"}), kind === "pin" ? s("path", {class: "pin-mark"}) : null);

function drawStrip() {
  const scroller = document.getElementById("lm-scroll");
  const list = view.list;
  const n = list.length;
  const starts = list.map((layer, c) => c === 0 || groupOf(layer) !== groupOf(list[c - 1]));
  const breaks = starts.filter(Boolean).length - 1;
  const room = scroller.clientWidth;
  const pitch = clamp(Math.floor((room - breaks * GROUP_GAP[0]) / n), ...PITCH);
  const gap = breaks ? clamp(Math.floor((room - n * pitch) / breaks), ...GROUP_GAP) : 0;
  const xs = [];
  const groups = [];
  let x = 0;
  list.forEach((layer, c) => {
    if (starts[c]) {
      if (c) x += gap;
      groups.push({key: groupOf(layer), x});
    }
    xs.push(x);
    x += pitch;
  });
  const width = x - 1;
  const rowsH = view.shown.length * ROW + (view.shown.length - 1) * GAP;
  const svg = s("svg", {class: "lm-svg", width, height: TOP + rowsH + AXIS, "aria-hidden": "true"},
    s("defs", null, s("pattern", {id: "lm-na", width: 4, height: 4, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"},
      s("line", {class: "na-line", x1: 0, y1: 0, x2: 0, y2: 4}))));
  view.shown.forEach((report, r) => {
    const y = TOP + r * (ROW + GAP);
    const row = s("g", {class: "row"});
    for (let c = 0; c < n; c += 1) {
      const color = cellColor(report.id, c);
      row.append(s("rect", {class: color ? "cell" : "cell na", x: xs[c], y, width: pitch - 1, height: ROW,
        fill: color ?? "url(#lm-na)"}));
    }
    svg.append(row);
  });
  const axis = s("g", {class: "axis"});
  const base = TOP + rowsH;
  groups.forEach((group, i) => {
    const right = (groups[i + 1]?.x ?? width + 1 + gap) - gap - 1;
    if (state.order === "kind") {
      if (right - group.x >= 24) {
        axis.append(s("text", {x: (group.x + right) / 2, y: base + 16, "text-anchor": "middle"}, kindLabel(String(group.key))));
      }
    } else if (group.key == null || group.key % 4 === 0 || i === groups.length - 1) {
      axis.append(s("line", {x1: group.x + 0.5, x2: group.x + 0.5, y1: base + 3, y2: base + 7}),
        s("text", {x: group.x, y: base + 18}, group.key == null ? "기타" : String(group.key)));
    }
  });
  const hover = marker("hover");
  const pinned = marker("pin");
  svg.append(axis, hover, pinned);
  geo = {xs, pitch, rowsH, hover, pinned};
  const columnAt = (event) => {
    const left = event.clientX - svg.getBoundingClientRect().left;
    let lo = 0;
    let hi = xs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (xs[mid] <= left) lo = mid;
      else hi = mid - 1;
    }
    return list[lo].name;
  };
  svg.addEventListener("pointermove", (event) => {
    const name = columnAt(event);
    if (name === state.hover) return;
    state.hover = name;
    showColumn();
  });
  svg.addEventListener("pointerleave", () => {
    state.hover = null;
    showColumn();
  });
  svg.addEventListener("click", (event) => pin(columnAt(event)));
  scroller.replaceChildren(svg);
}

// Outer ring in ink, inner ring in panel color: the column stays visible on any cell color.
function place(group, c) {
  if (c == null) {
    group.setAttribute("visibility", "hidden");
    return;
  }
  const x = geo.xs[c];
  const w = geo.pitch - 1;
  const [outer, inner] = group.querySelectorAll("rect");
  for (const [rect, pad] of [[outer, 2], [inner, 0.5]]) {
    rect.setAttribute("x", x - pad);
    rect.setAttribute("y", TOP - pad);
    rect.setAttribute("width", w + 2 * pad);
    rect.setAttribute("height", geo.rowsH + 2 * pad);
  }
  const mark = group.querySelector(".pin-mark");
  if (mark) mark.setAttribute("d", `M${x + w / 2 - 4} 1 L${x + w / 2 + 4} 1 L${x + w / 2} ${TOP - 4} Z`);
  group.setAttribute("visibility", "visible");
}

function showColumn() {
  const following = state.hover != null && state.hover !== state.pin;
  place(geo.hover, following ? view.index.get(state.hover) : null);
  place(geo.pinned, view.index.get(state.pin));
  fillReadout(view.index.get(following ? state.hover : state.pin), following);
}

function reveal(c, smooth = true) {
  const scroller = document.getElementById("lm-scroll");
  if (c == null || !scroller) return;
  const x = geo.xs[c];
  if (x >= scroller.scrollLeft && x + geo.pitch <= scroller.scrollLeft + scroller.clientWidth) return;
  scroller.scrollTo({left: Math.max(0, x - scroller.clientWidth / 2), behavior: smooth && !reducedMotion.matches ? "smooth" : "auto"});
}

function pin(name) {
  state.pin = name;
  showColumn();
  reveal(view.index.get(name));
  for (const row of root.querySelectorAll("table.lm-t tbody tr")) row.classList.toggle("on", row.dataset.name === name);
  for (const button of root.querySelectorAll(".lm-worst button")) button.setAttribute("aria-pressed", String(button.dataset.name === name));
  const c = view.index.get(name);
  document.getElementById("lm-live").textContent = `${name}: `
    + view.shown.map((report) => `${report.label} ${display(view.metric, view.values.get(report.id)[c])}`).join(", ");
}

function onStripKey(event) {
  const list = view.list;
  const start = (i) => {
    let j = i;
    while (j > 0 && groupOf(list[j - 1]) === groupOf(list[j])) j -= 1;
    return j;
  };
  let c = view.index.get(state.pin) ?? 0;
  if (event.key === "ArrowRight" && event.shiftKey) {
    const group = groupOf(list[c]);
    while (c < list.length - 1 && groupOf(list[c]) === group) c += 1;
  } else if (event.key === "ArrowLeft" && event.shiftKey) {
    const first = start(c);
    c = first === c && c > 0 ? start(c - 1) : first;
  } else if (event.key === "ArrowRight") c += 1;
  else if (event.key === "ArrowLeft") c -= 1;
  else if (event.key === "Home") c = 0;
  else if (event.key === "End") c = list.length - 1;
  else return;
  event.preventDefault();
  state.hover = null;
  pin(list[clamp(c, 0, list.length - 1)].name);
}

// The MMA column is shown only for the ULP metric, which measures the accumulator it names.
function readout() {
  const m = view.metric;
  const ulp = m.id === "ulp_mean";
  readoutHead = {name: h("span", {class: "name"}), meta: h("span", {class: "lm-pills"})};
  const rows = view.shown.map((report) => {
    const cells = {sw: h("i", {class: "lm-sw"}), v: h("td", {class: "v"}), d: h("td", {class: "d"}),
      w: h("td", {class: "d"}), spec: (ulp ? [0, 1, 2, 3] : [0, 1, 2]).map((i) => h("td", {class: i === 3 ? "spec mma" : "spec"}))};
    readoutRows.push({report, cells});
    return h("tr", null, h("td", null, cells.sw), h("th", {scope: "row"}, report.label), cells.v, cells.d, cells.w,
      cells.spec);
  });
  return h("section", {class: "lm-readout", "aria-labelledby": "lm-readout-title"},
    h("div", {class: "lm-readout-head"}, h("h3", {id: "lm-readout-title"}, "판독값"), readoutHead.name, readoutHead.meta),
    h("div", {class: "table-wrap"}, h("table", {class: "diff lm-ro"},
      h("thead", null, h("tr", null, h("th", null, h("span", {class: "sr-only"}, "색")), h("th", {scope: "col"}, "방법"),
        h("th", {scope: "col", class: "num"}, m.label),
        h("th", {scope: "col", class: "num", title: "같은 레이어에서 모든 방법의 중앙값 대비"}, "같은 레이어 Δ"),
        h("th", {scope: "col", class: "num", title: "이 방법의 모든 레이어 중앙값 대비"}, "방법 안 Δ"),
        ["가중치 형식", "활성 형식", "가중치 알고리즘 · 변환", ...(ulp ? ["MMA"] : [])].map((t) => h("th", {scope: "col"}, t)))),
      h("tbody", null, rows))));
}

function fillReadout(c, following) {
  if (c == null) return;
  const m = view.metric;
  const layer = view.list[c];
  readoutHead.name.textContent = layer.name;
  readoutHead.meta.replaceChildren(
    h("span", {class: "pill"}, "블록 ", h("b", null, layer.block ?? "—")),
    h("span", {class: "pill"}, "종류 ", h("b", null, layer.kind)),
    h("span", {class: "pill"}, "같은 레이어 방법 중앙값 ", h("b", null, display(m, view.medians[c]))),
    h("span", {class: "pill"}, following ? "포인터 위치" : "고정됨"));
  for (const {report, cells} of readoutRows) {
    const color = cellColor(report.id, c);
    cells.sw.className = color ? "lm-sw" : "lm-sw na";
    cells.sw.style.background = color ?? "";
    cells.v.textContent = display(m, view.values.get(report.id)[c]);
    cells.d.textContent = displayDelta(m, view.deltas.get(report.id)[c]);
    cells.w.textContent = displayDelta(m, view.within.get(report.id)[c]);
    cells.w.title = `${report.label} 레이어 중앙값 ${display(m, view.rowMedians.get(report.id))}`;
    specCells(specOf(report, layer.name), m.id === "ulp_mean").forEach((text, i) => {
      if (cells.spec[i]) cells.spec[i].textContent = text;
    });
  }
}

function sidePanel() {
  const windows = new Set(model.reports.map((report) => `${report.samples} × ${report.seqlen}`));
  const first = model.reports[0];
  const span = windows.size === 1 ? `샘플 ${first.samples}개 × ${first.seqlen} 토큰` : "리포트마다 다름, 재현 정보 참고";
  return h("aside", {class: "lm-side", "aria-labelledby": "lm-side-title"},
    h("h2", {id: "lm-side-title"}, "방법별 요약"),
    h("p", {class: "lm-note"}, `logits KL·top-1 일치·PPL은 각 리포트가 직접 잰 창(${span})의 값입니다. 테스트 세트 PPL이 아닙니다.`),
    h("div", {class: "lm-cards"}, view.shown.map(card)));
}

function card(report) {
  const m = view.metric;
  const values = view.values.get(report.id);
  const measured = values.filter((v) => Number.isFinite(v));
  const exact = values.filter((v) => v != null && !Number.isFinite(v) && good(m, v) > 0).length;
  const mean = measured.length ? measured.reduce((a, b) => a + b, 0) / measured.length : null;
  const worst = values.map((v, c) => ({v, c})).filter((item) => item.v != null)
    .sort((a, b) => cmp(good(m, a.v), good(m, b.v)) || a.c - b.c).slice(0, 5);
  const stat = (label, value, note) => h("div", null, h("dt", null, label), h("dd", null, value, note ? h("small", null, note) : null));
  const ppl = safe((v) => v.toFixed(2));
  return h("article", {class: "lm-card", "aria-label": report.label},
    h("header", null, h("h3", null, report.label), h("span", {class: "rid", title: report.description || null}, report.id)),
    h("dl", {class: "lm-stats"},
      stat(`평균 ${m.short}`, mean == null ? (exact ? "∞" : "—") : display(m, mean),
        exact ? (measured.length ? `오차 0인 ${exact}개 제외` : "모든 레이어 오차 0") : null),
      stat("logits KL", safe(fmt.kl)(modelValue(report, "logits_kl"))),
      stat("top-1 일치", safe(fmt.pct)(modelValue(report, "top1_agreement")))),
    h("p", {class: "lm-ppl"}, "PPL 기준 ", h("b", null, ppl(modelValue(report, "ppl_reference"))),
      " → 에뮬레이션 ", h("b", null, ppl(modelValue(report, "ppl_emulated")))),
    h("p", {class: "lm-worst-title"}, `가장 나쁜 레이어 5 · ${m.short}`),
    measured.length ? h("ol", {class: "lm-worst"}, worst.map((item, i) => {
      const layer = view.list[item.c];
      return h("li", null, h("button", {type: "button", title: layer.name, "aria-pressed": String(layer.name === state.pin),
        dataset: {name: layer.name}, onclick: () => pin(layer.name)},
      h("span", {class: "rank"}, i + 1), h("span", {class: "nm"}, shortName(layer)), h("span", {class: "val"}, display(m, item.v))));
    })) : h("p", {class: "lm-note"}, "모든 레이어의 오차가 0입니다."));
}

function tablePanel() {
  const section = h("section", {class: "panel lm-table", "aria-labelledby": "lm-table-title"});
  fillTable(section);
  return section;
}

function fillTable(section, focusKey) {
  const report = view.shown.find((item) => item.id === state.table) ?? view.shown[0];
  state.table = report.id;
  const items = view.list.map((layer, c) => ({layer, c, i: report.at.get(layer.name)})).filter((item) => item.i != null);
  if (state.sort) {
    const {col, dir} = state.sort;
    const value = (item) => raw(report, item.i, col);
    items.sort((a, b) => {
      const x = value(a);
      const y = value(b);
      if (x == null || y == null) return (x == null) - (y == null) || a.c - b.c;
      return dir * cmp(x, y) || a.c - b.c;
    });
  }
  const refill = (key) => fillTable(section, key);
  const sortButton = (col, text) => {
    const on = state.sort?.col === col;
    return h("button", {type: "button", "data-key": `sort:${col}`, onclick: () => {
      state.sort = !on ? {col, dir: 1} : state.sort.dir === 1 ? {col, dir: -1} : null;
      refill(`sort:${col}`);
    }}, text, on ? (state.sort.dir === 1 ? " ↑" : " ↓") : "");
  };
  const ariaSort = (col) => (state.sort?.col === col ? (state.sort.dir === 1 ? "ascending" : "descending") : null);
  const groups = [...new Set(TABLE.map((col) => col.group))];
  const select = h("select", {"data-key": "table-method", onchange: (event) => {
    state.table = event.target.value;
    refill("table-method");
  }}, view.shown.map((item) => h("option", {value: item.id, selected: item.id === report.id}, item.label)));
  section.replaceChildren(
    h("div", {class: "toolbar lm-table-head"}, h("h2", {id: "lm-table-title"}, "레이어 표"),
      h("label", {class: "lm-field"}, "표에 보일 방법", select)),
    h("p", {class: "lead"}, `${report.label}의 Linear ${items.length}개와 모든 지표입니다. 열 이름을 누르면 오름차순, 내림차순, `
      + "레이어 순서 차례로 정렬합니다. 행을 누르면 지도에서 그 레이어에 고정됩니다."),
    h("div", {class: "lm-twrap", tabindex: "0", role: "region", "aria-label": `${report.label} 레이어 표`},
      h("table", {class: "diff lm-t"},
        h("thead", null,
          h("tr", null,
            h("th", {class: "nm", rowspan: 2, scope: "col"},
              h("button", {type: "button", "data-key": "sort:order", onclick: () => {
                state.sort = null;
                refill("sort:order");
              }}, "레이어")),
            h("th", {class: "tx", rowspan: 2, scope: "col"}, "블록"), h("th", {class: "tx", rowspan: 2, scope: "col"}, "종류"),
            groups.map((group) => h("th", {class: "group", scope: "colgroup",
              colspan: TABLE.filter((col) => col.group === group).length}, group))),
          h("tr", null, TABLE.map((col) => h("th", {scope: "col", class: col.first ? "first" : null, "aria-sort": ariaSort(col.col)},
            sortButton(col.col, col.name))))),
        h("tbody", null, items.map(({layer, i}) => h("tr", {class: layer.name === state.pin ? "on" : null,
          dataset: {name: layer.name}, onclick: () => pin(layer.name)},
        h("td", {class: "nm"}, layer.name), h("td", {class: "tx"}, layer.block ?? "—"), h("td", {class: "tx"}, layer.kind),
        TABLE.map((col) => {
          const v = raw(report, i, col.col);
          return h("td", {class: col.first ? "first" : null}, v == null ? "—" : col.show(v));
        })))))));
  if (focusKey) section.querySelector(`[data-key="${CSS.escape(focusKey)}"]`)?.focus();
}

function provenance() {
  const short = (text, n) => (text ? String(text).slice(0, n) : "—");
  const dirty = (value) => (value === true ? "수정 있음" : value === false ? "깨끗함" : "알 수 없음");
  return h("footer", {class: "panel lm-prov", "aria-labelledby": "lm-prov-title"},
    h("h2", {id: "lm-prov-title"}, "재현 정보"),
    h("p", {class: "lead"}, "한 줄이 tricast report 실행 하나입니다. 수치는 이 환경 기록과 함께일 때만 근거가 됩니다."),
    h("div", {class: "table-wrap"}, h("table", {class: "diff"},
      h("thead", null, h("tr", null, ["방법", "git", "작업 트리", "모델", "GPU", "샘플 × 길이", "데이터 지문"]
        .map((text) => h("th", {scope: "col"}, text)))),
      h("tbody", null, model.reports.map((report) => h("tr", null,
        h("th", {scope: "row"}, report.label),
        h("td", {class: "mono"}, short(report.env?.git_sha, 7)),
        h("td", {class: report.env?.git_dirty === false ? null : "warn"}, dirty(report.env?.git_dirty)),
        h("td", {class: "mono"}, `${report.model.id} @ ${short(report.model.sha, 7)}`),
        h("td", null, (report.env?.gpu_names ?? []).join(", ") || "—"),
        h("td", {class: "mono"}, `${report.samples} × ${report.seqlen}`),
        h("td", {class: "mono"}, short(report.dataset_fingerprint, 8))))))));
}

function emptyState() {
  const copy = h("button", {type: "button", class: "copy", onclick: () => copyText(copy, COMMAND)}, "복사");
  root.replaceChildren(h("section", {class: "state"},
    h("h2", null, "레이어 지도 묶음이 아직 없습니다"),
    h("p", null, "tricast report 결과(report.json)를 내보내면 이 화면이 채워집니다. 저장소 루트에서 실행하세요."),
    h("pre", {class: "code"}, COMMAND),
    h("p", {class: "note"}, "결과는 app/web/demo/layers/ 에 index.json 과 방법별 파일로 저장됩니다."), copy));
}

function errorState(error) {
  root.replaceChildren(h("section", {class: "state", role: "alert"},
    h("h2", null, "레이어 지도를 열지 못했습니다"), h("p", null, error.message),
    h("p", {class: "note"}, "묶음을 다시 만들려면 저장소 루트에서 실행하세요."), h("pre", {class: "code"}, COMMAND)));
}

// The worst layer by the median over methods: where the page opens.
function worstLayer() {
  let worst = 0;
  view.medians.forEach((v, c) => {
    const current = view.medians[worst];
    if (v != null && (current == null || good(view.metric, v) < good(view.metric, current))) worst = c;
  });
  return view.list[worst].name;
}

async function boot() {
  let index;
  try {
    index = await getJSON(`${BUNDLE}index.json`);
  } catch (error) {
    if (error.status === 404) emptyState();
    else errorState(error);
    return;
  }
  let reports;
  try {
    if (index.schema !== SCHEMA) throw new Error(`묶음 형식이 ${index.schema ?? "알 수 없음"} 입니다. ${SCHEMA} 형식으로 다시 내보내세요.`);
    reports = await Promise.all((index.reports ?? []).map(async (entry) => prepare(entry, await getJSON(BUNDLE + entry.file))));
  } catch (error) {
    errorState(error);
    return;
  }
  if (!reports.length) {
    emptyState();
    return;
  }
  model = buildModel(reports);
  palette = readPalette();
  view = computeView();
  state.pin = worstLayer();
  render();
  let queued = false;
  window.addEventListener("resize", () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      drawStrip();
      showColumn();
    });
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render());
}

if (location.protocol !== "file:") boot();
