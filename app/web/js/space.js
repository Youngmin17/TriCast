// Design-space chart: the same input under every recorded design point, metric vs F, one line per C mode.
import {mmaKey} from "./algo.js";
import {mmaSummary} from "./glyph.js";
import {fmt, h, s, withTip} from "./ui.js";

const METRICS = {
  "llm.generate": [
    {key: "top1_agreement", label: "top-1 일치율", lo: 0, hi: 1, show: (v) => fmt.pct(v)},
    {key: "kl_mean", label: "KL 평균 (로그)", log: true, show: (v) => fmt.kl(v)},
    {key: "prefix_match", label: "같은 앞부분 (토큰)", lo: 0, show: (v) => `${v}토큰`},
  ],
  "vision.detect": [
    {key: "mean_iou", label: "평균 IoU", lo: 0, hi: 1, show: (v) => fmt.num(v)},
    {key: "changed", label: "한쪽에만 있는 상자", lo: 0, show: (v) => `${v}개`,
      pick: (sum) => (sum.baseline_only ?? 0) + (sum.emulated_only ?? 0)},
  ],
  "vision.classify": [
    {key: "kl", label: "KL (로그)", log: true, show: (v) => fmt.kl(v)},
    {key: "top5_overlap", label: "top-5 겹침", lo: 0, hi: 5, show: (v) => `${v}/5`},
  ],
};

const SAME_CONTEXT = (run, sel, inputKey) => run.task === sel.task && run.model === sel.model
  && (run.format ?? null) === (sel.format ?? null) && (run.baseline || "native") === sel.baseline
  && (run.input?.prompt ?? run.input?.image) === inputKey && run.summary;

function value(metric, run) {
  return metric.pick ? metric.pick(run.summary) : run.summary[metric.key];
}

function sweepOf(runs, mma) {
  const fixed = (run) => run.mma.algorithm === "cofda" && run.mma.chunk_size === mma.chunk_size
    && (run.mma.promote_interval ?? 0) === (mma.promote_interval ?? 0) && run.mma.norm_rounding === mma.norm_rounding;
  return runs.filter((run) => fixed(run) && (run.mma.c_mode === "fused"
    || (run.mma.c_mode === "decoupled" && run.mma.f2_bits === (mma.f2_bits ?? run.mma.f2_bits))));
}

function chart(points, metric, current, act, ref) {
  const width = 680;
  const height = 210;
  const pad = {l: 56, r: 16, t: 14, b: 34};
  const fs = [...new Set(points.map((run) => run.mma.f_bits))].sort((a, b) => a - b);
  const values = [...points.map((run) => value(metric, run)), ref].filter((v) => v != null);
  const floor = 1e-6;
  let lo = metric.lo ?? Math.min(...values);
  let hi = metric.hi ?? Math.max(...values);
  if (metric.log) {
    lo = Math.max(floor, Math.min(...values.map((v) => Math.max(v, floor))));
    hi = Math.max(lo * 10, Math.max(...values));
  }
  if (hi === lo) hi = lo + 1;
  const x = (f) => pad.l + ((f - fs[0]) / Math.max(1, fs[fs.length - 1] - fs[0])) * (width - pad.l - pad.r);
  const y = (v) => {
    const t = metric.log ? (Math.log10(Math.max(v, floor)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
      : (v - lo) / (hi - lo);
    return height - pad.b - t * (height - pad.t - pad.b);
  };
  const svg = s("svg", {class: "chart space", viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
    "aria-label": `${metric.label}, F 비트에 따라`});
  const ticksY = metric.log ? [lo, Math.sqrt(lo * hi), hi] : [lo, (lo + hi) / 2, hi];
  for (const t of ticksY) {
    svg.append(s("line", {class: "axis", x1: pad.l, x2: width - pad.r, y1: y(t), y2: y(t)}),
      s("text", {x: pad.l - 8, y: y(t) + 3.5, "text-anchor": "end"}, metric.show(metric.log ? Number(t.toPrecision(2)) : t)));
  }
  for (const f of fs) svg.append(s("text", {x: x(f), y: height - 12, "text-anchor": "middle"}, `F${f}`));
  if (ref != null) {
    svg.append(s("line", {class: "refline", x1: pad.l, x2: width - pad.r, y1: y(ref), y2: y(ref)}),
      s("text", {class: "reflabel", x: width - pad.r, y: y(ref) - 5, "text-anchor": "end"}, `FP64 누산 ${metric.show(ref)}`));
  }
  for (const mode of ["fused", "decoupled"]) {
    const series = points.filter((run) => run.mma.c_mode === mode && value(metric, run) != null)
      .sort((a, b) => a.mma.f_bits - b.mma.f_bits);
    if (!series.length) continue;
    svg.append(s("path", {class: `series ${mode}`, d: series.map((run, i) =>
      `${i ? "L" : "M"}${x(run.mma.f_bits).toFixed(1)} ${y(value(metric, run)).toFixed(1)}`).join(" ")}));
    for (const run of series) {
      const on = mmaKey(run.mma) === mmaKey(current);
      const dot = s("circle", {class: `pt ${mode}${on ? " on" : ""}`, cx: x(run.mma.f_bits), cy: y(value(metric, run)),
        r: on ? 7 : 5, tabindex: "0", role: "button", "aria-label": `${mmaSummary(run.mma)} 선택`,
        onclick: () => act.selectMMA(run.mma), onkeydown: (e) => (e.key === "Enter" || e.key === " ") && act.selectMMA(run.mma)});
      withTip(dot, () => h("div", null, mmaSummary(run.mma), h("div", null, `${metric.label}: ${metric.show(value(metric, run))}`)));
      svg.append(dot);
    }
  }
  return svg;
}

export function designSpace(state, ctx, act) {
  const sel = state.sel;
  const inputKey = sel.task === "llm.generate" ? sel.prompt : sel.image;
  const runs = ctx.runs.filter((run) => SAME_CONTEXT(run, sel, inputKey));
  if (runs.length < 2) return null;
  const metrics = METRICS[sel.task];
  const metric = metrics.find((item) => item.key === state.spaceMetric) || metrics[0];
  const cofda = sel.mma.algorithm === "cofda" ? sel.mma : {chunk_size: 32, promote_interval: 0, norm_rounding: "rtz"};
  const sweep = sweepOf(runs, cofda);
  const others = runs.filter((run) => !sweep.includes(run));
  const fp64 = runs.find((run) => run.mma.algorithm === "fp64");
  const metricItems = metrics.map((item) => h("button", {type: "button", role: "radio",
    "aria-checked": String(item.key === metric.key), onclick: () => act.setSpaceMetric(item.key)}, item.label));
  return h("section", {class: "panel"},
    h("div", {class: "toolbar", style: "justify-content: space-between"},
      h("h3", null, "설계 공간 · 같은 입력에서 F를 바꾸면"),
      h("div", {class: "seg compact", role: "radiogroup", "aria-label": "지표"}, metricItems)),
    h("p", {class: "lead"}, `CS ${cofda.chunk_size}${cofda.promote_interval ? ` · 승격 ${cofda.promote_interval}` : ""} `
      + "CoFDA의 기록된 설계점입니다. 점을 누르면 그 설계로 바뀝니다. F를 늘리면 같은 형식에 FP64로 누산한 수준(점선)으로 다가가고, "
      + "그래도 남는 차이는 양자화 몫입니다. 한 입력에서는 F에 따라 단조롭지 않을 수 있습니다."),
    sweep.length ? h("div", {class: "chart-wrap"}, chart(sweep, metric, sel.mma, act, fp64 ? value(metric, fp64) : null)) : null,
    h("div", {class: "legend"},
      h("span", null, h("i", {class: "sw fused"}), "fused (C 함께 정렬)"),
      h("span", null, h("i", {class: "sw decoupled"}), "decoupled (C 따로 결합)")),
    others.length ? h("div", {class: "suggest"}, others.map((run) => h("button", {type: "button", class: "sample",
      "aria-pressed": String(mmaKey(run.mma) === mmaKey(sel.mma)), onclick: () => act.selectMMA(run.mma)},
    `${mmaSummary(run.mma)} · ${metric.show(value(metric, run))}`))) : null);
}
