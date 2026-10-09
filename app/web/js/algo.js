// The virtual-algorithm designer: canonical MMA dicts, preset starting points and parameter controls.
import {flow} from "./flow.js";
import {datapath, keptLabel, mmaSummary} from "./glyph.js";
import {h, radioGroup, withTip} from "./ui.js";

export const STATUS_LABEL = {
  reference: "레퍼런스", modeled: "모델링", partial_mismatch: "부분 불일치", design_point: "설계점",
};
const OPTION_LABEL = {fused: "fused", decoupled: "decoupled", rtz: "0 방향 (RTZ)", rne: "최근접 (RNE)", 0: "끔"};

function holds(param, mma) {
  return !param.when || Object.entries(param.when).every(([key, value]) => mma[key] === value);
}

// Keys whose `when` condition holds, missing values filled with defaults (app/README.md "정규형 mma").
export function canonical(mma, algorithms) {
  const algo = algorithms.find((item) => item.id === mma.algorithm);
  if (!algo) return {algorithm: mma.algorithm};
  const out = {algorithm: algo.id};
  for (const param of algo.params) out[param.key] = mma[param.key] ?? param.default;
  for (const param of algo.params) if (!holds(param, out)) delete out[param.key];
  return out;
}

export function mmaKey(mma) {
  return JSON.stringify(Object.keys(mma).sort().map((key) => [key, mma[key]]));
}

export function presetFor(mma, format, presets) {
  const key = mmaKey(mma);
  return presets.find((preset) => mmaKey(preset.mma) === key && (preset.format ?? null) === (format ?? null)) || null;
}

// `domains`: the input format's K-varying scale blocks (catalog formats[].k_domains); a span must divide each.
function validOptions(param, mma, domains) {
  const fits = (span) => domains.every((domain) => domain % span === 0);
  if (param.key === "promote_interval") {
    // TriCast applies promotion to fused CoFDA only, every multiple of the chunk size inside the scale blocks.
    return param.options.filter((v) => v === 0 || (mma.c_mode !== "decoupled" && v % mma.chunk_size === 0 && fits(v)));
  }
  if (param.key === "group_size") return param.options.filter(fits);
  if (param.key === "k_tile") {
    return param.options.filter((v) => v % mma.group_size === 0 && v / mma.group_size >= 1 && v / mma.group_size <= 8);
  }
  return param.options;
}

function intControl(param, mma, act, recorded) {
  const id = `p-${param.key}`;
  const value = mma[param.key];
  const lo = param.ui_min ?? param.min;
  const hi = param.ui_max ?? param.max;
  const snap = (raw) => (recorded?.length
    ? recorded.reduce((best, v) => (Math.abs(v - raw) < Math.abs(best - raw) ? v : best), recorded[0]) : raw);
  const slider = h("input", {type: "range", id, min: String(lo), max: String(hi), step: "1", value: String(value),
    "aria-valuetext": `${value}${param.unit ?? ""}`,
    oninput: (event) => {
      const next = snap(Number(event.target.value));
      event.target.value = String(next);
      output.textContent = `${next}${param.unit ? ` ${param.unit}` : ""}`;
    },
    onchange: (event) => act.setParam(param.key, snap(Number(event.target.value)))});
  const output = h("output", {for: id, class: "param-value num"}, `${value}${param.unit ? ` ${param.unit}` : ""}`);
  const ticks = recorded?.length ? h("div", {class: "ticks", "aria-hidden": "true"}, recorded.map((v) =>
    h("i", {style: `left: ${((v - lo) / (hi - lo)) * 100}%`, class: v === value ? "on" : null}))) : null;
  return h("div", {class: "param"},
    h("label", {class: "param-head", for: id}, h("span", null, param.label), output),
    h("div", {class: "slider"}, slider, ticks),
    param.help ? h("p", {class: "param-help"}, param.help) : null);
}

function choiceControl(param, mma, act, available, domains) {
  const options = validOptions(param, mma, domains);
  const items = options.map((option) => ({
    id: option,
    disabled: !available(param.key, option),
    render: () => h("button", {type: "button"}, OPTION_LABEL[option] ?? String(option)),
  }));
  return h("div", {class: "param"},
    h("div", {class: "param-head"}, h("span", null, param.label)),
    radioGroup(param.label, items, mma[param.key], (option) => act.setParam(param.key, option), "seg"),
    param.help ? h("p", {class: "param-help"}, param.help) : null);
}

function presetChips(state, ctx, act) {
  const {presets} = ctx.catalog;
  const current = presetFor(state.sel.mma, state.sel.format, presets);
  return h("div", {class: "presets", role: "group", "aria-label": "preset 시작점"}, presets.map((preset) => {
    const button = h("button", {type: "button", class: "preset", "aria-pressed": String(current?.id === preset.id),
      disabled: !act.presetAvailable(preset), onclick: () => act.usePreset(preset)},
    h("span", {class: `dot ${preset.status}`}), preset.label);
    return withTip(button, () => h("div", null, h("div", null, `${preset.source} · ${STATUS_LABEL[preset.status] ?? preset.status}`),
      h("div", {class: "muted"}, mmaSummary(preset.mma)), h("div", {class: "muted"}, `명세 출처: ${preset.provenance}`),
      h("div", {class: "muted"}, `검증 기록: ${preset.status_note}`),
      act.presetAvailable(preset) ? null : h("div", null, "이 입력으로 기록된 실행이 없습니다")));
  }));
}

function origin(state, ctx) {
  const preset = presetFor(state.sel.mma, state.sel.format, ctx.catalog.presets);
  if (preset) {
    return h("div", {class: "origin"}, h("span", {class: `badge ${preset.status}`}, STATUS_LABEL[preset.status] ?? preset.status),
      h("span", null, `${preset.source}. 검증 기록: ${preset.status_note}`),
      h("span", {class: "muted"}, `명세 출처: ${preset.provenance}`));
  }
  return h("p", {class: "origin"}, h("span", {class: "badge design_point"}, "사용자 정의"),
    "직접 설계한 가상 알고리즘입니다. 하드웨어 측정값이 아니라 TriCast가 정의대로 계산한 결과입니다.");
}

export function designer(state, ctx, act) {
  const {algorithms} = ctx.catalog;
  const mma = state.sel.mma;
  const algo = algorithms.find((item) => item.id === mma.algorithm) || algorithms[0];
  const families = algorithms.map((item) => ({
    id: item.id, disabled: !act.available("algorithm", item.id),
    render: () => h("button", {type: "button", title: item.description}, item.label),
  }));
  const params = algo.params.filter((param) => holds(param, mma) && (!param.advanced || state.advanced));
  const domains = ctx.catalog.formats.find((f) => (f.id ?? null) === (state.sel.format ?? null))?.k_domains ?? [];
  const hasAdvanced = algo.params.some((param) => param.advanced);
  return [
    h("div", {class: "param-head"}, h("span", {class: "muted"}, "시작점")),
    presetChips(state, ctx, act),
    radioGroup("알고리즘", families, mma.algorithm, (id) => act.setAlgorithm(id), "seg"),
    h("div", {class: "viz"},
      h("div", {class: "viz-row"}, datapath(mma, 300)),
      h("p", {class: "param-help"}, keptLabel(mma)),
      flow(mma)),
    ...params.map((param) => (param.kind === "int"
      ? intControl(param, mma, act, act.recordedValues(param.key))
      : choiceControl(param, mma, act, (key, value) => act.available(key, value), domains))),
    hasAdvanced ? h("label", {class: "toggle"}, h("input", {type: "checkbox", id: "advanced", checked: state.advanced,
      onchange: (event) => act.setAdvanced(event.target.checked)}), "고급 설정 보기") : null,
    origin(state, ctx),
  ];
}
