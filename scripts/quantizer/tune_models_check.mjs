// Browser check of the tuning app's recorded models (app/web/tune.html, demo/autotune/models.json) against a running
// server: every model loads, a size recommends (only the entry's `sizes` when it lists them), step 4 shows the
// benchmarks table where the model has one (one row per benchmark, every cell a number, verdicts from the fixed set),
// the detector its boxes on three photos while a language model shows no generated answers, step 5 opens, an
// existing way can be picked by hand, the model package under step 5 matches packages.json (or says how to build
// one), the photo tabs move by keyboard, and nothing overflows sideways. Headless
// Chrome through cdp_lib.mjs, outside pytest.
//   node scripts/quantizer/tune_models_check.mjs <out dir> <url> [width]   → TUNE_MODELS_CHECKS <passed>/<total>
import {connect} from "./cdp_lib.mjs";

const [out, url, width = "1440"] = process.argv.slice(2);
if (!out || !url) {
  console.error("usage: node tune_models_check.mjs <out dir> <url> [width]");
  process.exit(2);
}
const b = await connect(out, {logging: true});
await b.setup(Number(width), 1000);
const results = [];
const check = (name, ok, detail = "") => results.push({ok, line: `${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`});
const ok = await b.open(`${url}/tune.html`, `document.body.dataset.ready === "yes"`);
check("page ready", ok);
const models = await b.evaluate(`[...document.querySelectorAll("#model option")].map(o => o.value)`);
const listed = await b.evaluate(`fetch("demo/autotune/models.json").then((r) => r.json()).then((d) => d.models)`);
check("model list = models.json", JSON.stringify(models) === JSON.stringify(listed.map((m) => m.id)), models.join(", "));

async function loadModel(id) {
  await b.evaluate(`(() => { const s = document.getElementById("model"); s.value = ${JSON.stringify(id)}; document.getElementById("load").click(); })()`);
  return b.waitFor(`document.body.dataset.model === ${JSON.stringify(id)} && document.querySelectorAll("#tune [data-budget]").length > 0`);
}
async function pick(index) {
  await b.evaluate(`document.querySelectorAll("#tune [data-budget]")[${index}].click()`);
  return b.waitFor(`!!document.querySelector("#reco") && !document.querySelector("#sec-eval").hasAttribute("data-off") && !!document.getElementById("run-eval")`, 20000);
}
// evaluate the setup on screen (picking the size already on screen keeps its evaluation: nothing to press)
async function evalNow() {
  await b.evaluate(`document.getElementById("run-eval")?.click()`);
  return b.waitFor(`!!document.getElementById("eval-result")`);
}
// the benchmarks table: [rows, cells per row, cells that are not numbers, verdict labels outside the set]
const benchTable = (words = false) => b.evaluate(`(() => {
  const rows = [...document.querySelectorAll("#bench-table tbody tr:not(.grp)")];
  const cells = rows.map((r) => [...r.querySelectorAll("td")].map((td) => td.firstChild?.textContent?.trim() ?? ""));
  // a number; for the image generator also its PSNR words: "기준" (the original), "같음" (the same image), "—" (no finite difference)
  const bad = cells.flat().filter((t) => !/^[+−-]?[0-9][0-9,.]*%?( ± [0-9.]+)?$/.test(t) && !(${words} && ["기준", "같음", "—"].includes(t)));
  const verdicts = [...document.querySelectorAll("#bench-table .ta-vd")].map((v) => v.textContent.split(" · ").pop()).filter((v) => !v.startsWith("±"));
  const wrap = document.querySelector(".ta-bench-wrap");
  return {rows: rows.length, widths: [...new Set(cells.map((c) => c.length))], bad, verdicts, inside: !!wrap && wrap.scrollWidth <= wrap.clientWidth + 1,
    odd: verdicts.filter((v) => !["우리 우세", "차이 없음", "기존 우세"].includes(v)), sum: document.getElementById("bench-sum")?.textContent ?? ""};
})()`);

for (const [i, entry] of listed.entries()) {
  const id = entry.id;
  const loaded = await loadModel(id);
  check(`${id}: loads`, loaded);
  if (!loaded) continue;
  const sizes = await b.evaluate(`document.querySelectorAll("#tune [data-budget]").length`);
  const picked = await pick(sizes > 1 ? 1 : 0);
  check(`${id}: recommends`, picked, `${sizes} sizes`);
  if (!picked) continue;
  check(`${id}: evaluation`, await evalNow());
  const answers = await b.evaluate(`document.querySelectorAll("#answers").length`);
  const dets = await b.evaluate(`document.querySelectorAll("#detections .ta-det").length`);
  const measured = entry.kind === "detector" || entry.kind === "image";  // a file of measured setups with 3 outputs
  check(`${id}: outputs`, measured ? dets === 3 && answers === 0 : answers === 0 && dets === 0, `answers=${answers} detections=${dets}`);
  if (entry.sizes) check(`${id}: sizes offered = the entry's sizes`, sizes === entry.sizes.length, `${sizes} / ${entry.sizes.length}`);
  if (entry.benchmarks || measured) {
    const t = await benchTable(entry.kind === "image");
    const expected = entry.kind === "detector" ? 6 : entry.kind === "image" ? 2 : (await b.evaluate(`fetch("demo/autotune/${entry.benchmarks}").then((r) => r.json()).then((d) => d.benchmarks.length + 1)`));
    check(`${id}: benchmarks table`, t.rows === expected && t.widths.length === 1 && t.widths[0] === 4 && t.bad.length === 0 && t.odd.length === 0 && t.inside,
      `${t.rows}/${expected} rows · cells ${t.widths} · non-numbers ${JSON.stringify(t.bad.slice(0, 3))} · verdicts ${t.verdicts.length} · fits ${t.inside}`);
    if (!measured) check(`${id}: benchmarks summary`, /우리 우세 \d+ · 차이 없음 \d+ · 기존 우세 \d+/.test(t.sum), t.sum.slice(0, 60));
  }
  check(`${id}: step 5`, await b.waitFor(`!document.querySelector("#sec-out").hasAttribute("data-off") && !!document.getElementById("size")`, 8000));
  // the model package under step 5: a recorded package shows its files and checks as packages.json has them (an ONNX
  // file downloads with the recorded size), a setup without one shows how to build it
  const pk = await b.evaluate(`(async () => {
    const block = document.getElementById("package");
    if (!block) return {block: false};
    const index = await fetch("demo/autotune/packages.json").then((r) => r.json());
    const link = document.getElementById("onnx-download");
    if (link) {
      const entry = Object.values(index.onnx?.yolo11n ?? {}).find((e) => link.getAttribute("href") === "demo/autotune/" + e.file);
      const bytes = (await (await fetch(link.getAttribute("href"))).arrayBuffer()).byteLength;
      return {block: true, kind: "onnx", ok: !!entry && bytes === entry.bytes, detail: (entry?.file ?? "no entry") + " " + bytes};
    }
    const cmd = block.querySelector(".ta-cmd")?.textContent ?? "";
    if (!cmd) return {block: true, kind: "none", ok: block.textContent.includes("아직"), detail: block.textContent.slice(0, 30)};
    const entry = (index.llm ?? []).find((e) => cmd.includes("--package " + e.dir));
    if (!entry) return {block: true, kind: "build", ok: cmd.includes("python -m tricast.pack build"), detail: cmd.slice(0, 40)};
    const rows = document.querySelectorAll("#package-files tr").length;
    const checks = document.getElementById("package-checks")?.textContent ?? "";
    return {block: true, kind: "package", ok: rows === entry.files.length && checks.includes(entry.decoded_equal.layers + "개 층 모두")
      && checks.includes(entry.ppl_package.toFixed(3)), detail: entry.dir + " rows " + rows + "/" + entry.files.length};
  })()`);
  check(`${id}: model package`, pk.block && pk.ok, `${pk.kind ?? "no block"} ${pk.detail ?? ""}`);
  // the edit: the first existing way in the select, then the comparison again
  const opts = await b.evaluate(`[...document.querySelectorAll("#assign option")].map(o => o.value).filter(Boolean)`);
  if (opts.length) {
    await b.evaluate(`(() => { const s = document.getElementById("assign"); s.value = ${JSON.stringify(opts[0])}; s.dispatchEvent(new Event("change")); })()`);
    check(`${id}: edit`, await b.waitFor(`document.querySelector("#edit-state").textContent.includes("직접 편집함")`), opts[0]);
    await evalNow();
  }
  // the photo tabs by keyboard: ArrowRight moves the pick and the focus stays on the picked tab
  const tablist = await b.evaluate(`document.querySelector("#photo-tabs")?.id ?? null`);
  if (tablist) {
    await b.evaluate(`document.querySelector("#${tablist} [aria-selected=true]").focus()`);
    await b.key("ArrowRight");
    const moved = await b.evaluate(`(() => { const t = [...document.querySelectorAll("#${tablist} [role=tab]")]; const k = t.findIndex((x) => x.getAttribute("aria-selected") === "true"); return k === 1 && document.activeElement === t[1]; })()`);
    check(`${id}: tabs by keyboard`, moved, tablist);
  }
  await b.shot(`${out}/models_${i}_${width}.png`, {x: 0, y: 0, width: Number(width), height: await b.evaluate("document.documentElement.scrollHeight")});
  check(`${id}: no sideways overflow`, !(await b.evaluate(`document.documentElement.scrollWidth > window.innerWidth + 1`)));
}
const errors = b.events.filter((e) => e.method === "Runtime.exceptionThrown").map((e) => e.params.exceptionDetails?.exception?.description?.split("\n")[0]);
check("no page exceptions", errors.length === 0, errors.join(" | "));
console.log(results.map((r) => r.line).join("\n"));
console.log(`TUNE_MODELS_CHECKS ${results.filter((r) => r.ok).length}/${results.length}`);
await b.close();
process.exit(results.every((r) => r.ok) ? 0 : 1);
