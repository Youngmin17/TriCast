// Interaction checks for app/web/layers.html (the layer map, 9 reports in app/web/demo/layers) in headless Chrome.
// Prints PASS/FAIL per check and CDP_CHECKS n/m.
// usage (static servers as in qz_check.mjs; <alt> = prep_alt_roots.sh output):
//   node scripts/quantizer/cdp_check.mjs <out_dir> http://127.0.0.1:8765 http://127.0.0.1:8766
import {mkdirSync, writeFileSync} from "node:fs";

import {connect} from "./cdp_lib.mjs";

const [, , outDir, base, alt] = process.argv;
mkdirSync(outDir, {recursive: true});
const b = await connect(outDir);
const {evaluate, pause} = b;
const name = () => evaluate(`document.querySelector(".lm-readout .name").textContent`);
const pills = () => evaluate(`[...document.querySelectorAll(".lm-readout .lm-pills .pill")].map((p) => p.textContent)`);
const count = (selector) => evaluate(`document.querySelectorAll(${JSON.stringify(selector)}).length`);
const click = (key) => evaluate(`document.querySelector('[data-key="${key}"]').click()`);
const legendText = () => evaluate(`document.querySelector(".lm-legend").textContent`);
const rowFills = (r) => evaluate(`[...document.querySelectorAll("g.row")[${r}].children].map((x) => x.getAttribute("fill"))`);
const results = {checks: [], shots: []};
const check = (label, pass, detail) => results.checks.push({label, pass: Boolean(pass), detail});
const shot = async (file) => results.shots.push(await b.shot(`${outDir}/${file}.png`));

// ---------- main page, light, 1600 ----------
await b.setup(1600, 1000);
check("main page loads", await b.open(`${base}/layers.html`));
check("1764 heatmap cells (9 x 196)", (await count("rect.cell")) === 1764, await count("rect.cell"));
check("9 readout rows", (await count(".lm-readout tbody tr")) === 9, await count(".lm-readout tbody tr"));
const scaleLabels = await evaluate(`[...document.querySelectorAll('[data-key^="scale:"]')].map((x) => x.textContent)`);
check("three color bases with unambiguous names", JSON.stringify(scaleLabels)
  === JSON.stringify(["절대값", "같은 레이어 · 방법 중앙값 대비 Δ", "방법 안 레이어 중앙값 대비 Δ"]), scaleLabels);
const heads = await evaluate(`[...document.querySelectorAll(".lm-ro thead th")].map((x) => x.textContent)`);
check("readout shows both deltas", heads.includes("같은 레이어 Δ") && heads.includes("방법 안 Δ"), heads);
const first = await name();
check("opens on the worst-median layer", first === "model.layers.13.self_attn.o_proj", first);
await evaluate(`document.getElementById("lm-scroll").focus()`);
await b.key("ArrowRight");
check("ArrowRight moves to the next Linear", (await name()) === "model.layers.13.mlp.gate_proj", await name());
check("strip focus is visible", (await evaluate(`getComputedStyle(document.getElementById("lm-scroll")).outlineStyle`)) !== "none");
await b.key("ArrowRight", true);
check("Shift+ArrowRight jumps to the next block", (await name()) === "model.layers.14.self_attn.q_proj", await name());
await b.key("ArrowLeft", true);
check("Shift+ArrowLeft jumps to the previous block start", (await name()) === "model.layers.13.self_attn.q_proj", await name());
await b.key("Home");
check("Home goes to the first layer", (await name()) === "model.layers.0.self_attn.q_proj", await name());
await b.key("End");
check("End goes to the last layer", (await name()) === "model.layers.27.mlp.down_proj", await name());
check("live region announces the pin", (await evaluate(`document.getElementById("lm-live").textContent`))
  .startsWith("model.layers.27.mlp.down_proj:"));
const cell = await evaluate(`(() => { const r = document.querySelectorAll("g.row")[3].children[50].getBoundingClientRect();
  return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()`);
await b.mouse("mouseMoved", cell.x, cell.y);
const hoverName = await name();
check("pointer moves the crosshair over all rows", hoverName === "model.layers.7.self_attn.k_proj"
  && (await pills()).includes("포인터 위치")
  && (await evaluate(`document.querySelector(".xh.hover").getAttribute("visibility")`)) === "visible", {hoverName, pills: await pills()});
await b.mouse("mousePressed", cell.x, cell.y);
await b.mouse("mouseReleased", cell.x, cell.y);
check("click pins the column", (await name()) === hoverName && (await pills()).includes("고정됨"), await pills());
await shot("cdp_light_1600_pinned");

await click("scale:delta");
await pause(100);
const legend = await legendText();
check("same-layer delta: legend centred at the same-layer method median", legend.includes("0 = 같은 레이어의 방법 중앙값")
  && legend.includes("나쁨") && legend.includes("좋음"), legend);
check("same-layer delta keeps focus on the chosen toggle", (await evaluate(`document.activeElement.dataset.key`)) === "scale:delta");
const fills = (await rowFills(0)).slice(0, 5);
check("same-layer delta repaints cells with the diverging ramp", fills.every((f) => f.startsWith("rgb(")), fills);
await shot("cdp_light_1600_delta");

// ---------- within-method delta ----------
await b.mouse("mouseMoved", 5, 5);
await click("scale:within");
await pause(100);
check("within-method delta: toggle checked and focused", (await evaluate(`[document.querySelector('[data-key="scale:within"]').getAttribute("aria-checked"),
  document.activeElement.dataset.key].join()`)) === "true,scale:within");
const withinLegend = await legendText();
check("within-method delta: legend names the per-method median and the p98 clip", withinLegend.includes("0 = 이 방법의 레이어 중앙값")
  && withinLegend.includes("98백분위") && withinLegend.includes("≤ ") && withinLegend.includes("≥ "), withinLegend);
// Independent oracle: medians and the clip straight from the bundle JSON, formatted like the page.
const oracle = await evaluate(`(async () => {
  const index = await (await fetch("demo/layers/index.json")).json();
  const pinned = document.querySelector(".lm-readout .name").textContent;
  const med = (xs) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const q = (s, p) => { const x = (s.length - 1) * p; const i = Math.floor(x); return i + 1 < s.length ? s[i] + (s[i + 1] - s[i]) * (x - i) : s[i]; };
  const fmt = (d) => d === 0 ? "0" : (d > 0 ? "+" : "−") + (Math.abs(d) >= 0.01 ? Math.abs(d).toFixed(2) : Math.abs(d).toExponential(1).replace("-", "−")) + " dB";
  const rows = []; const all = [];
  for (const entry of index.reports) {
    const data = await (await fetch("demo/layers/" + entry.file)).json();
    const col = data.columns.indexOf("output.sqnr_db");
    const xs = data.rows.map((r) => r[col]).filter((v) => v != null);
    const m = med(xs);
    xs.forEach((v) => all.push(Math.abs(v - m)));
    const v = data.rows.find((r) => r[0] === pinned)[col];
    rows.push({label: entry.label, expected: fmt(v - m), median: m});
  }
  all.sort((a, b) => a - b);
  const shown = [...document.querySelectorAll(".lm-ro tbody tr")].map((tr) => [tr.querySelector("th").textContent, tr.querySelectorAll("td.d")[1].textContent]);
  return {pinned, rows, shown, clip: q(all, 0.98).toFixed(2) + " dB"};
})()`);
check("within-method delta in the readout matches an independent median for all 9 methods",
  oracle.rows.every((row, i) => oracle.shown[i][0] === row.label && oracle.shown[i][1] === row.expected), oracle);
check("legend clip equals the independent 98th percentile of |Δ| over all methods", withinLegend.includes(`(${oracle.clip})`),
  {clip: oracle.clip, withinLegend});
const ends = await evaluate(`(() => { const stops = getComputedStyle(document.documentElement).getPropertyValue("--lm-div").split(",").map((x) => x.trim());
  const rgb = (h) => "rgb(" + [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)).join(" ") + ")";
  const lo = rgb(stops[0]); const hi = rgb(stops.at(-1));
  const all = [...document.querySelectorAll("rect.cell")].map((r) => r.getAttribute("fill"));
  return {low: all.filter((f) => f === lo).length, high: all.filter((f) => f === hi).length, total: all.length}; })()`);
check("within-method clip: about 2% of cells sit at the ramp ends", ends.low + ends.high >= 1 && ends.low + ends.high <= 0.04 * ends.total, ends);
const int8Before = await rowFills(0);
await click("method:mxfp4_w_a");
await pause(100);
const int8After = await rowFills(0);
check("within-method: hiding a method does not repaint the others (fills and clip unchanged)",
  int8Before.join() === int8After.join() && (await legendText()) === withinLegend, {same: int8Before.join() === int8After.join()});
await click("method:mxfp4_w_a");
await pause(100);
await evaluate(`document.getElementById("lm-scroll").focus()`);
const beforeKey = await name();
await b.key("ArrowRight");
check("within-method: arrow keys still move the pin", (await name()) !== beforeKey, {beforeKey, after: await name()});
await b.key("ArrowLeft");
await shot("cdp_light_1600_within");

await evaluate(`(() => { const s = document.querySelector('[data-key="metric"]'); s.value = "a_sqnr"; s.dispatchEvent(new Event("change")); })()`);
await click("scale:abs");
await pause(100);
const gptqFills = [...new Set(await rowFills(2))];
const best = await evaluate(`(() => { const sw = [...document.querySelectorAll(".lm-legend .lm-key")].find((k) => k.textContent.includes("오차 0")); return sw ? sw.textContent : null; })()`);
check("activation SQNR: GPTQ W4A16 row is all exact (one fill) and the legend explains it", gptqFills.length === 1 && best === "∞ = 오차 0",
  {gptqFills, best});
check("activation SQNR: GPTQ card says every layer has zero error", (await evaluate(`document.querySelectorAll(".lm-card")[2].textContent`))
  .includes("모든 레이어의 오차가 0입니다"));
await click("scale:within");
await pause(100);
check("activation SQNR, within-method: an all-exact row is neutral (Δ = 0)", [...new Set(await rowFills(2))].length === 1
  && (await evaluate(`document.querySelectorAll(".lm-ro tbody tr")[2].querySelectorAll("td.d")[1].textContent`)) === "0");

// The MMA column (header and cells) exists only for the ULP metric; there it names the GEMM output format.
const mmaColumn = () => evaluate(`({head: [...document.querySelectorAll(".lm-ro thead th")].some((th) => th.textContent === "MMA"),
  cells: document.querySelectorAll(".lm-ro td.spec.mma").length,
  first: document.querySelectorAll(".lm-ro tbody tr")[0].querySelector("td.spec.mma")?.textContent ?? null})`);
const mmaSqnr = await mmaColumn();
await evaluate(`(() => { const s = document.querySelector('[data-key="metric"]'); s.value = "ulp_mean"; s.dispatchEvent(new Event("change")); })()`);
const mmaUlp = await mmaColumn();
check("MMA column only for MMA ULP 평균 (header included), naming the GEMM output format there",
  !mmaSqnr.head && mmaSqnr.cells === 0 && mmaUlp.head && mmaUlp.cells === 9 && mmaUlp.first.endsWith("출력 bf16"), {mmaSqnr, mmaUlp});
await evaluate(`(() => { const s = document.querySelector('[data-key="metric"]'); s.value = "out_sqnr"; s.dispatchEvent(new Event("change")); })()`);
await click("order:kind");
await pause(100);
const axis = await evaluate(`[...document.querySelectorAll(".lm-svg .axis text")].map((t) => t.textContent)`);
check("Linear-kind order: axis shows q k v o gate up down; pinned layer kept", axis.join(",") === "q,k,v,o,gate,up,down"
  && (await name()) === hoverName, {axis, name: await name()});
await click("scale:delta");
await pause(100);
await shot("cdp_light_1600_kind_delta");
await click("order:block");
await click("scale:abs");

const medianBefore = (await pills()).find((p) => p.startsWith("같은 레이어 방법 중앙값"));
await click("method:mxfp4_w_a");
await pause(100);
check("unchecking a method hides its row, card and readout row", (await count("rect.cell")) === 8 * 196
  && (await count(".lm-card")) === 8 && (await count(".lm-readout tbody tr")) === 8);
check("the same-layer median does not change when a method is hidden", medianBefore && (await pills()).includes(medianBefore),
  {medianBefore, after: await pills()});
await click("method:mxfp4_w_a");

const worst = await evaluate(`(() => { const x = document.querySelectorAll(".lm-card")[1].querySelector(".lm-worst button"); x.click(); return x.title; })()`);
await pause(100);
check("worst-5 click pins that column and marks the table row", (await name()) === worst
  && (await evaluate(`document.querySelector("table.lm-t tbody tr.on")?.dataset.name`)) === worst, worst);
await click("sort:output.sqnr_db");
const asc = await evaluate(`[document.querySelector("table.lm-t tbody tr").dataset.name, document.querySelector('[data-key="sort:output.sqnr_db"]').closest("th").getAttribute("aria-sort")]`);
await click("sort:output.sqnr_db");
const desc = await evaluate(`[document.querySelector("table.lm-t tbody tr").dataset.name, document.querySelector('[data-key="sort:output.sqnr_db"]').closest("th").getAttribute("aria-sort")]`);
check("table sorts by output SQNR ascending then descending (INT8)", asc[0] === "model.layers.1.mlp.up_proj" && asc[1] === "ascending"
  && desc[1] === "descending" && desc[0] !== asc[0], {asc, desc});
await evaluate(`(() => { const s = document.querySelector('[data-key="table-method"]'); s.value = "mxfp4_w_a"; s.dispatchEvent(new Event("change")); })()`);
check("table method select switches the table", (await evaluate(`document.querySelector(".lm-table .lead").textContent`)).startsWith("MXFP4"));
check("page has no horizontal scroll at 1600", await evaluate(`document.documentElement.scrollWidth <= innerWidth && document.getElementById("map").scrollWidth <= document.getElementById("map").clientWidth`));

// ---------- dark ----------
await b.setup(1600, 1000, true);
await b.open(`${base}/layers.html`);
check("dark theme ramp is active", await evaluate(`getComputedStyle(document.documentElement).getPropertyValue("--lm-seq").trim().startsWith("#172d44")`));
check("dark legend says brighter is worse", (await legendText()).includes("밝을수록 나쁨"));
await shot("cdp_dark_1600");
await click("scale:within");
await pause(100);
check("dark within-method legend keeps the same wording", (await legendText()) === withinLegend);
await shot("cdp_dark_1600_within");

// ---------- 900 px, reduced motion ----------
await b.setup(900, 1000, false, true);
await b.open(`${base}/layers.html`);
const overflow = await evaluate(`(() => { const s = document.getElementById("lm-scroll");
  return {page: document.documentElement.scrollWidth, viewport: innerWidth, stripScroll: s.scrollWidth, stripBox: s.clientWidth}; })()`);
check("900 px: no page-level horizontal scroll; the heatmap scrolls inside", overflow.page <= overflow.viewport && overflow.stripScroll > overflow.stripBox, overflow);
check("reduced motion: no transitions", (await evaluate(`getComputedStyle(document.querySelector(".lm-check")).transitionDuration`)) === "0s");
await shot("cdp_light_900");

// ---------- empty and error states ----------
await b.setup(1600, 1000);
await b.open(`${alt}/empty/layers.html`);
const empty = await evaluate(`[document.querySelector(".state h2")?.textContent, document.querySelector(".state pre")?.textContent]`);
check("missing bundle: instruction with the export command", empty[0] === "레이어 지도 묶음이 아직 없습니다" && empty[1].startsWith("python -m app.export_layers"), empty);
await shot("cdp_empty");
await b.open(`${alt}/bad/layers.html`);
const bad = await evaluate(`[document.querySelector(".state h2")?.textContent, document.querySelector(".state p")?.textContent]`);
check("wrong bundle schema: error state names the problem", bad[0] === "레이어 지도를 열지 못했습니다" && bad[1].includes("tricast.layer_map/1"), bad);

const problems = b.events.filter((e) => e.sessionId === b.sessionId && (e.method === "Runtime.exceptionThrown"
  || (e.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(e.params.type))
  || (e.method === "Log.entryAdded" && e.params.entry.level === "error")));
// The empty-state page fetches a missing index.json on purpose; that 404 is the expected network log line.
const unexpected = problems.filter((e) => !(e.method === "Log.entryAdded" && e.params.entry.url?.endsWith("/empty/demo/layers/index.json")));
results.console = problems.map((e) => e.params.exceptionDetails?.exception?.description ?? e.params.entry?.text ?? JSON.stringify(e.params.args));
check("no JS exceptions or console errors (besides the intended 404)", unexpected.length === 0, results.console);
results.oracle = oracle;
results.ends = ends;
writeFileSync(`${outDir}/cdp_results.json`, JSON.stringify(results, null, 1));
for (const item of results.checks) console.log(`${item.pass ? "PASS" : "FAIL"}  ${item.label}${item.pass ? "" : `  ${JSON.stringify(item.detail)}`}`);
console.log(`CDP_CHECKS ${results.checks.filter((c) => c.pass).length}/${results.checks.length} passed`);
await b.close();
