// Result data: every design point run on the current input, next to the native output, in one table.
import {mmaKey} from "./algo.js";
import {mmaSummary} from "./glyph.js";
import {fmt, h} from "./ui.js";

const ALGO_ORDER = {cofda: 0, gdfs: 1, fp32_fma: 2, fp64: 3};

function order(a, b) {
  const base = (run) => ((run.baseline || "native") === "native" ? 0 : 1);
  return base(a) - base(b)
    || (ALGO_ORDER[a.mma.algorithm] ?? 9) - (ALGO_ORDER[b.mma.algorithm] ?? 9)
    || (a.mma.chunk_size ?? 0) - (b.mma.chunk_size ?? 0)
    || (a.mma.promote_interval ?? 0) - (b.mma.promote_interval ?? 0)
    || String(a.mma.c_mode).localeCompare(String(b.mma.c_mode))
    || (a.mma.f_bits ?? 0) - (b.mma.f_bits ?? 0);
}

// Common prefix plain, the diverging remainder marked.
function diffText(reference, text) {
  let i = 0;
  while (i < reference.length && i < text.length && reference[i] === text[i]) i += 1;
  if (i === text.length && i === reference.length) return h("span", null, text, h("span", {class: "same-tag"}, " 기준과 같음"));
  return h("span", null, text.slice(0, i), h("mark", {class: "diverge"}, text.slice(i) || "∅"));
}

const formatLabel = (ctx, id) => ctx.catalog.formats.find((f) => (f.id ?? null) === (id ?? null))?.label ?? String(id);
const baselineLabel = (run) => ((run.baseline || "native") === "native" ? "원본" : "FP64 누산");

function row(run, cells, current, act) {
  const on = mmaKey(run.mma) === mmaKey(current.mma) && (run.format ?? null) === (current.format ?? null)
    && (run.baseline || "native") === current.baseline;
  return h("tr", {class: on ? "on" : null, tabindex: "0", onclick: () => act.openRow(run),
    onkeydown: (event) => event.key === "Enter" && act.openRow(run)}, cells);
}

function llmTable(runs, ctx, sel, act) {
  const native = runs.find((run) => run.preview?.baseline != null && (run.baseline || "native") === "native")?.preview.baseline ?? "";
  return [
    h("section", {class: "channel"}, h("div", {class: "channel-head"}, h("span", {class: "ch-tag a"}, "A"),
      h("span", {class: "label"}, "원본(native) 출력 · 기준이 원본인 행의 비교 대상")), h("div", {class: "channel-body tokens"}, native)),
    h("div", {class: "table-wrap"}, h("table", {class: "diff results"},
      h("thead", null, h("tr", null, ["설계점", "형식", "기준", "첫 분기", "top-1 일치", "KL 평균", "에뮬레이션 출력 (주황 = 기준과 달라진 부분)"]
        .map((t) => h("th", null, t)))),
      h("tbody", null, runs.map((run) => {
        const sum = run.summary;
        const reference = (run.baseline || "native") === "native" ? native : run.preview?.baseline ?? "";
        return row(run, [
          h("td", {class: "mono"}, mmaSummary(run.mma)), h("td", null, formatLabel(ctx, run.format)), h("td", null, baselineLabel(run)),
          h("td", {class: "mono"}, sum.first_divergence == null ? "없음" : `${sum.first_divergence + 1}`),
          h("td", {class: "mono"}, fmt.pct(sum.top1_agreement)), h("td", {class: "mono"}, fmt.kl(sum.kl_mean)),
          h("td", {class: "text"}, run.preview ? diffText(reference, run.preview.emulated ?? "") : "—"),
        ], sel, act);
      })))),
  ];
}

function detectTable(runs, ctx, sel, act) {
  return h("div", {class: "table-wrap"}, h("table", {class: "diff results"},
    h("thead", null, h("tr", null, ["설계점", "형식", "기준", "상자 A → B", "A에만", "B에만", "평균 IoU"].map((t) => h("th", null, t)))),
    h("tbody", null, runs.map((run) => {
      const sum = run.summary;
      return row(run, [
        h("td", {class: "mono"}, mmaSummary(run.mma)), h("td", null, formatLabel(ctx, run.format)), h("td", null, baselineLabel(run)),
        h("td", {class: "mono"}, run.preview ? `${run.preview.baseline_boxes} → ${run.preview.emulated_boxes}` : "—"),
        h("td", {class: sum.baseline_only ? "mono hot" : "mono"}, String(sum.baseline_only)),
        h("td", {class: sum.emulated_only ? "mono hot" : "mono"}, String(sum.emulated_only)),
        h("td", {class: "mono"}, fmt.num(sum.mean_iou)),
      ], sel, act);
    }))));
}

function classifyTable(runs, ctx, sel, act) {
  const native = runs.find((run) => run.preview?.baseline_top1)?.preview.baseline_top1;
  return [
    native ? h("p", {class: "lead"}, `원본(native) top-1: ${native}`) : null,
    h("div", {class: "table-wrap"}, h("table", {class: "diff results"},
      h("thead", null, h("tr", null, ["설계점", "형식", "기준", "에뮬레이션 top-1", "확률", "top-1", "top-5 겹침", "KL"].map((t) => h("th", null, t)))),
      h("tbody", null, runs.map((run) => {
        const sum = run.summary;
        return row(run, [
          h("td", {class: "mono"}, mmaSummary(run.mma)), h("td", null, formatLabel(ctx, run.format)), h("td", null, baselineLabel(run)),
          h("td", null, run.preview?.emulated_top1 ?? "—"), h("td", {class: "mono"}, fmt.pct(run.preview?.emulated_top1_p)),
          h("td", {class: sum.top1_same ? "mono" : "mono hot"}, sum.top1_same ? "같음" : "다름"),
          h("td", {class: "mono"}, `${sum.top5_overlap}/5`), h("td", {class: "mono"}, fmt.kl(sum.kl)),
        ], sel, act);
      })))),
  ];
}

export function resultsTable(state, ctx, act) {
  const sel = state.sel;
  const inputKey = sel.task === "llm.generate" ? sel.prompt : sel.image;
  const runs = ctx.runs.filter((run) => run.task === sel.task && run.model === sel.model && run.summary && run.mma
    && (run.input?.prompt ?? run.input?.image) === inputKey).sort(order);
  if (!runs.length) {
    return [h("section", {class: "state"}, h("h2", null, "이 입력의 결과가 아직 없습니다"),
      h("p", null, "왼쪽에서 설계를 고르고 실행하면 결과가 여기에 쌓입니다."))];
  }
  const intro = h("p", {class: "lead"}, `이 입력으로 ${ctx.mode === "demo" ? "기록된" : "실행한"} 설계점 ${runs.length}개입니다. `
    + "행을 누르면 그 설계의 상세 비교가 열립니다.");
  const body = sel.task === "llm.generate" ? llmTable(runs, ctx, sel, act)
    : sel.task === "vision.detect" ? [detectTable(runs, ctx, sel, act)] : classifyTable(runs, ctx, sel, act);
  return [h("section", {class: "panel"}, h("h3", null, "결과 데이터 · 기준과 모든 설계점"), intro, ...body)];
}
