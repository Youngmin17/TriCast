// Checks for the one tool (app/web/tune.html: 1 model, 2 Auto-tune, 3 edit by hand, 4 evaluation, 5 output) in headless Chrome,
// beyond the demo rehearsal (demo_rehearsal.mjs). On a running server with --agent (python -m app.server --demo
// --agent …):
//    1 the knapsack runs in the worker: no main-thread stall over 100 ms during the solve (a timer heartbeat; a 150 ms
//      busy loop must show up as the control);  2 measured budgets are warmed up: picking one answers within 150 ms;
//    3 one system: five steps and nothing else — no menu to other pages, no chat, no chart, nothing typed, and the
//      model pick offers exactly the recorded models (demo/autotune/models.json);
//    4 the size buttons say the size and the measured PPL of the setup recommended there;  5 at every size the
//      recommendation is the best measured setup (the AI's settings on top only where they measured better), said
//      with how it was reached, and its recipe is the measured file;  6 the edit step offers only what was measured:
//      every "one method on every layer" choice and every option setting, at every size, is a GPU-measured setup
//      whose recipe is the measured file and whose evaluation is its recorded PPL;  7 the address keeps the job (a
//      hand edit included) across a reload;  8 the evaluation of the recommendation: the number, where it came from
//      (GPU, tree, recipe SHA) and bars of the drift from the original for every measured way at that size;
//    9 the evaluation of a hand edit shows it against the recommendation, and "back to the recommendation" restores
//      it;  10 the grid says a layer's method on hover and cannot be edited;  11 nothing overflows sideways at 1440
//      and 390 px;  12 one screen: at 1440 × 900 and 1280 × 720 the evaluation's button is on screen once a size is
//      picked and after an edit;  13 step 5 opens after the evaluation: the computed file size by part beside the
//      recipe bundle download (no answer comparison).
// On the live fixture (scripts/quantizer/tune_fixture_server.py: the real app on a FAKE GPU whose job runner returns
// made-up numbers — the page's handling is what is checked, not a measurement): 14 the evaluation runs as a GPU job
// (progress, PPL, the chosen lm-eval task, then step 5 opens); 15 a failing job says it failed and why.
//
// usage: node scripts/quantizer/tune_app_check.mjs <out_dir> <python> [base=http://127.0.0.1:8765]
// Prints PASS/FAIL per check and TUNE_APP_CHECKS <passed>/<total>; writes <out_dir>/tune_app_check.json and PNGs.
import {spawn} from "node:child_process";
import {mkdirSync, readdirSync, writeFileSync} from "node:fs";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

import {connect} from "./cdp_lib.mjs";

const [, , outArg, python, base = "http://127.0.0.1:8765"] = process.argv;
if (!outArg || !python) {
  console.error("usage: node scripts/quantizer/tune_app_check.mjs <out_dir> <python> [base]");
  process.exit(2);
}
const out = resolve(outArg);
mkdirSync(out, {recursive: true});
const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const results = [];
const check = (name, pass, detail = "") => {
  results.push({name, pass: Boolean(pass), detail});
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

const b = await connect(out);
const text = (selector) => b.evaluate(`(document.querySelector(${JSON.stringify(selector)})?.innerText ?? "").replace(/\\s+/g, " ")`);
const click = (selector) => b.evaluate(`(() => { const el = document.querySelector(${JSON.stringify(selector)});
  if (!el) return false; el.dispatchEvent(new MouseEvent("click", {bubbles: true})); return true; })()`);
const overflow = () => b.evaluate(`document.documentElement.scrollWidth > window.innerWidth + 1`);
const idle = () => b.waitFor(`document.body.dataset.solver === "idle"`, 60000);
const open = (url) => b.open(url, `document.body.dataset.ready === "yes"`);
const visible = (selector) => b.evaluate(`(() => { const r = document.querySelector(${JSON.stringify(selector)})?.getBoundingClientRect();
  return Boolean(r) && r.top >= 0 && r.bottom <= window.innerHeight; })()`);
const hover = async (selector) => {  // the text of the tooltip `selector` opens
  await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).dispatchEvent(new MouseEvent("mouseenter"))`);
  const shown = await b.evaluate(`document.getElementById("tip").hidden ? "" : document.getElementById("tip").innerText.replace(/\\s+/g, " ")`);
  await b.evaluate(`document.querySelector(${JSON.stringify(selector)}).dispatchEvent(new MouseEvent("mouseleave"))`);
  return shown;
};
const load = async () => {
  await click("#load");
  return b.waitFor(`!document.getElementById("sec-tune").hasAttribute("data-off")`, 30000);
};
// pick a target size and wait for its recommendation and recipe file
const pick = async (budget) => {
  await click(`.ta-preset[data-budget="${budget}"]`);
  return b.waitFor(`document.querySelector('.ta-preset[data-budget="${budget}"]')?.getAttribute("aria-pressed") === "true" && !!document.querySelector("#reco") &&
    !!document.querySelector("#recipe-file")`, 30000);
};
const file = () => b.evaluate(`(() => { const f = document.querySelector("#recipe-file"); return f ? {text: f.innerText.replace(/\\s+/g, " "), sha: f.dataset.sha,
  name: f.querySelector("b").innerText} : null; })()`);
// change the assignment select / press a settings button, and wait for the recipe file named `name`
const assign = async (method, name) => {
  await b.evaluate(`(() => { const s = document.querySelector("#assign"); s.value = ${JSON.stringify(method)}; s.dispatchEvent(new Event("change", {bubbles: true})); })()`);
  return b.waitFor(`document.querySelector("#recipe-file b")?.innerText === ${JSON.stringify(name)}`, 30000);
};
const evaluate = async () => {
  await click("#run-eval");
  await b.waitFor(`!!document.querySelector("#eval-result")`, 10000);
  return {hero: await text("#eval-result .ta-hero"), ladder: await text("#ladder"), gain: await text("#gain"), from: await text("#eval-state")};
};
const pplOf = (s) => /PPL ([\d.,]+)/.exec(s)?.[1] ?? "";
const heroPpl = (s) => /([\d.,]+) 원본/.exec(s)?.[1] ?? "";  // "WikiText-2 test PPL ? 26.51 원본 20.97 대비 …"

// ---------- the server with --agent ----------
await b.setup(1440, 900);
await open(`${base}/tune.html`);
// a heartbeat: the longest gap between 20 ms timer ticks is how long the main thread was busy at a stretch
// (headless Chrome reports no longtask entries, so the Long Tasks API cannot be the meter)
const beat = `window.__gap = 0; window.__last = performance.now(); clearInterval(window.__hb);
  window.__hb = setInterval(() => { const now = performance.now(); window.__gap = Math.max(window.__gap, now - window.__last); window.__last = now; }, 20);`;
const blank = await b.evaluate(`({sections: [...document.querySelectorAll(".ta-sec")].map((s) => s.querySelector("h2").innerText + (s.hasAttribute("data-off") ? " (대기)" : "")),
  links: [...document.querySelectorAll("a")].map((a) => a.getAttribute("href")), nav: document.querySelectorAll("nav, .ta-link").length,
  chat: document.querySelectorAll("#msgs, #sugg, .ta-ai, .ta-msg").length, chart: document.querySelectorAll("svg, #tradeoff").length,
  typed: document.querySelectorAll("input[type=text], input[type=range], input:not([type]), textarea, datalist").length,
  model: document.querySelector("#model").tagName, options: document.querySelectorAll("#model option").length})`);
// the model pick offers exactly the recorded models (demo/autotune/models.json; one bundle without a list)
const recordedModels = await b.evaluate(`fetch("demo/autotune/models.json").then((r) => (r.ok ? r.json() : {models: [0]})).then((d) => d.models.length)`);
await b.evaluate(beat);
await load();
await pick("17/4");  // the first solve: the user's pick goes before the background warm-up
const note = await text("#solver-note");
const longest = await b.evaluate(`window.__gap`);
const firstScreen = {evalButton: await visible("#run-eval")};
// positive control: a 150 ms busy loop on the main thread must show up, or the meter proves nothing
await b.evaluate(beat);
await b.evaluate(`(() => { const t = performance.now(); while (performance.now() - t < 150); })()`);
await b.pause(200);
const control = await b.evaluate(`window.__gap`);
check("1 knapsack in the worker, main thread free", note.includes("Worker") && longest <= 100 && control >= 150,
  `solver note "${note}" · longest main-thread stall during the solve ${longest.toFixed(0)} ms · control busy loop seen as ${control.toFixed(0)} ms`);

await idle();
const t0 = await b.evaluate(`performance.now()`);
await click(`.ta-preset[data-budget="403/128"]`);
await b.waitFor(`document.querySelector("#tune-state")?.innerText.includes("20%")`, 10000);
const ms = (await b.evaluate(`performance.now()`)) - t0;
await b.waitFor(`!!document.querySelector("#recipe-file")`, 30000);
check("2 a warmed-up measured budget answers at once", ms <= 150 && (await text("#solver-note")).includes("미리 계산"), `403/128 → 20% in ${ms.toFixed(0)} ms`);

check("3 one system: five steps, no menu to other pages, no chat, no chart, nothing typed",
  blank.sections.join(" | ") === "모델 | Auto-tune (대기) | 직접 편집 (대기) | 평가 (대기) | 결과물 (대기)" && blank.links.join() === "tune.html" && blank.nav === 0 &&
  blank.chat === 0 && blank.chart === 0 && blank.typed === 0 && blank.model === "SELECT" && blank.options === recordedModels,
  `steps ${blank.sections.join(" → ")} · links ${blank.links.join(",")} · menus ${blank.nav} · chat ${blank.chat} · charts ${blank.chart} · typed inputs ${blank.typed} · models ${blank.options}/${recordedModels}`);

const presets = await b.evaluate(`[...document.querySelectorAll(".ta-preset[data-budget]")].map((x) => x.innerText.replace(/\\s+/g, " "))`);
check("4 size buttons say the size and the recommended setup's measured PPL",
  presets.join(" | ") === "16% PPL 133 | 19% PPL 30.2 | 20% PPL 26.5 | 22% PPL 23.0 | 26% PPL 21.7 | 27% PPL 21.7", presets.join(" | "));

// ---------- every size: the recommendation, then every edit ----------
const budgets = await b.evaluate(`[...document.querySelectorAll(".ta-preset[data-budget]")].map((x) => x.dataset.budget)`);
const recos = [];
const edits = [];
for (const budget of budgets) {
  await pick(budget);
  const settings = await b.evaluate(`[...document.querySelectorAll(".ta-preset[data-option]")].map((x) => ({id: x.dataset.option, text: x.innerText.replace(/\\s+/g, " "),
    on: x.getAttribute("aria-pressed") === "true"}))`);
  const best = [...settings].sort((x, y) => Number(pplOf(x.text)) - Number(pplOf(y.text)))[0];
  const on = settings.find((s) => s.on);
  const reco = await text("#reco");
  const recipe = await file();
  recos.push({budget, ppl: pplOf(reco), setting: on.id, pass: on.id === best.id && on.text.includes("추천") && pplOf(reco) === pplOf(on.text) &&
    settings.filter((s) => s.text.includes("추천")).length === 1 && recipe.text.includes("GPU 가 측정한") && (await text("#edit-state")) === "추천 그대로" &&
    reco.includes(on.id === "default" ? "쓰지 않았습니다" : "더했습니다")});
  // the settings measured on the recommended assignment
  for (const s of settings) {
    await click(`.ta-preset[data-option="${s.id}"]`);
    await b.waitFor(`document.querySelector('.ta-preset[data-option="${s.id}"]')?.getAttribute("aria-pressed") === "true" && !!document.querySelector("#recipe-file")`, 30000);
    const f = await file();
    const ev = await evaluate();
    edits.push({budget, edit: `설정 ${s.id}`, ppl: pplOf(s.text), pass: f.text.includes("GPU 가 측정한") && heroPpl(ev.hero) === pplOf(s.text) && ev.from.includes("SHA 일치")});
  }
  // one method on every layer
  const uniforms = await b.evaluate(`[...document.querySelectorAll("#assign optgroup option")].map((o) => ({method: o.value, text: o.innerText}))`);
  for (const u of uniforms) {
    await assign(u.method, `u_${u.method}.yaml`);
    const f = await file();
    const bits = await b.evaluate(`new Set([...document.querySelectorAll(".ta-cell")].map((c) => c.dataset.b)).size`);
    const ev = await evaluate();
    edits.push({budget, edit: `전체 ${u.method}`, ppl: pplOf(u.text), pass: f.text.includes("GPU 가 측정한") && bits === 1 && heroPpl(ev.hero) === pplOf(u.text) &&
      ev.from.includes("SHA 일치") && (await text("#edit-state")).includes("직접 편집함")});
  }
}
check("5 at every size the recommendation is the best measured setup, and its recipe is the measured file",
  recos.length === 6 && recos.every((r) => r.pass) && recos.map((r) => r.ppl).join() === "133.29,30.22,26.51,23.00,21.73,21.67" &&
  recos.filter((r) => r.setting === "default").map((r) => r.budget).join() === "17/4",
  recos.map((r) => `${r.budget} → ${r.ppl} (${r.setting})`).join(" | "));
const failed = edits.filter((e) => !e.pass);
check("6 the edit step offers only what was measured: every choice is a measured setup, file and evaluation",
  edits.length >= 50 && failed.length === 0,
  `${edits.length} edits over 6 sizes (settings ${edits.filter((e) => e.edit.startsWith("설정")).length}, one method ${edits.filter((e) => e.edit.startsWith("전체")).length}) · ` +
  `not measured: ${failed.length ? failed.map((e) => `${e.budget} ${e.edit}`).join(", ") : "none"} · e.g. 403/128 ${edits.filter((e) => e.budget === "403/128").map((e) => `${e.edit.replace("전체 ", "")} ${e.ppl}`).join(" / ")}`);

// ---------- a hand edit at the demo size: its evaluation, the address, back to the recommendation ----------
await pick("403/128");
await assign("gptq_int3", "u_gptq_int3.yaml");
const editedFile = await file();
const hash = await b.evaluate(`location.hash`);
await b.S("Page.navigate", {url: "about:blank"});  // a real reload, not a same-document hash change
await b.pause(200);
await open(`${base}/tune.html${hash}`);
await b.waitFor(`document.querySelector("#recipe-file b")?.innerText === "u_gptq_int3.yaml"`, 30000);
const restored = await file();
check("7 a reload returns to the job, the hand edit included", hash.includes("size=403%2F128") && hash.includes("uniform=gptq_int3") &&
  /^[0-9a-f]{64}$/.test(editedFile.sha) && restored.sha === editedFile.sha && (await text("#edit-state")).includes("직접 편집함") &&
  (await b.evaluate(`document.querySelector("#assign").value`)) === "gptq_int3", `${decodeURIComponent(hash)} · SHA ${editedFile.sha.slice(0, 12)}`);

const edited = await evaluate();
const editedRows = await b.evaluate(`[...document.querySelectorAll("#ladder > div")].slice(1).map((r) => ({name: r.querySelector(".name").innerText.replace(/\\s+/g, " "), cur: r.classList.contains("cur")}))`);
await b.shot(`${out}/edit_1440.png`);
await click("#reset");
await b.waitFor(`document.querySelector("#recipe-file b")?.innerText === "act_damp05_r3.yaml"`, 30000);
const back = {state: await text("#edit-state"), assign: await b.evaluate(`document.querySelector("#assign").value`),
  setting: await b.evaluate(`document.querySelector('.ta-preset[data-option][aria-pressed="true"]')?.dataset.option ?? ""`), hash: await b.evaluate(`location.hash`)};
const reco = await evaluate();
const recoTip = await hover("#eval-result .ta-hero .ta-help");
const bars = await b.evaluate(`[...document.querySelectorAll("#ladder .track i")].map((i) => parseFloat(i.style.width))`);
const again = await b.evaluate(`!!document.querySelector("#run-eval")`);
await b.shot(`${out}/reco_1440.png`);
check("8 the evaluation of the recommendation: the number, its source, bars of the drift from the original",
  reco.hero.includes("26.51") && ["20.97", "33.31", "27.04", "26.82", "26.51", "한 방법 최선", "Auto-tune 배정", "AI 세부 설정"].every((v) => reco.ladder.includes(v)) &&
  reco.gain.includes("55% 작습니다") && bars.length === 4 && bars[0] === 100 && bars[3] < bars[2] && bars[2] < bars[1] && bars[1] < 50 &&
  reco.from.includes("기록된 GPU 측정") && reco.from.includes("SHA 일치") && recoTip.includes("퍼플렉서티") && !again,
  `${reco.hero} · ${reco.gain} · bars ${bars.join("/")}% · ${reco.from}`);
check("9 the evaluation of a hand edit stands against the recommendation; back to the recommendation restores it",
  edited.hero.includes("48.14") && editedRows[1]?.name.includes("직접 편집") && editedRows[1].cur && editedRows.filter((r) => r.cur).length === 1 &&
  editedRows.some((r) => r.name.includes("한 방법 최선")) && edited.gain.includes("Auto-tune 추천(PPL 26.51)") && edited.gain.includes("80% 작습니다") &&
  back.state === "추천 그대로" && back.assign === "" && back.setting === "act_damp05_r3" && !back.hash.includes("uniform") && !back.hash.includes("option"),
  `edit: ${edited.hero} · rows ${editedRows.map((r) => `${r.cur ? "▶" : ""}${r.name}`).join(" | ")} · ${edited.gain} · back: ${back.state}, ${back.setting}`);

// the grid: a layer's method on hover, nothing to edit
const cellTip = await hover(".ta-cell");
const gridBefore = await b.evaluate(`[...document.querySelectorAll(".ta-cell")].map((c) => c.dataset.b).join("")`);
await click(".ta-cell");
await b.pause(200);
check("10 the grid says a layer's method on hover and cannot be edited",
  cellTip.includes("블록 0 · q_proj") && cellTip.includes("비트") && /(GPTQ|AWQ|HQQ|NF4|INT8|BF16)/.test(cellTip) &&
  gridBefore === (await b.evaluate(`[...document.querySelectorAll(".ta-cell")].map((c) => c.dataset.b).join("")`)) &&
  (await text("#edit-state")) === "추천 그대로" && (await b.evaluate(`document.querySelectorAll("#edit button:not(.ta-preset):not(.ta-btn), #edit input").length`)) === 0,
  `tip "${cellTip.slice(0, 70)}" · used "${(await text("#used")).slice(0, 60)}"`);

const wideOverflow = await overflow();
await b.setup(390, 844);
await open(`${base}/tune.html`);
await load();
await pick("17/4");  // the longest settings note ("권하지 않음")
await b.pause(300);
const narrowOverflow = await overflow();
await b.shot(`${out}/page_390.png`);
check("11 no sideways overflow", !wideOverflow && !narrowOverflow, `1440 px ${wideOverflow} · 390 px ${narrowOverflow}`);

await b.setup(1280, 720);
await open(`${base}/tune.html`);
await load();
await pick("403/128");
await b.pause(900);  // the evaluation's title row comes into view
const small = {picked: await visible("#run-eval")};
await assign("awq_clip_int3", "u_awq_clip_int3.yaml");
await b.pause(900);
small.edited = await visible("#run-eval");
await b.shot(`${out}/screen_1280x720.png`);
check("12 one screen: the evaluation's button is on screen once a size is picked, at 1440 × 900 and 1280 × 720, and after an edit",
  firstScreen.evalButton && small.picked && small.edited,
  `1440 × 900 after the pick ${firstScreen.evalButton} · 1280 × 720 after the pick ${small.picked}, after an edit ${small.edited}`);

// ---------- step 5: what the evaluated setup hands over ----------
const downloads = `${out}/downloads`;
mkdirSync(downloads, {recursive: true});
await b.S("Browser.setDownloadBehavior", {behavior: "allow", downloadPath: downloads});
await b.setup(1440, 900);
await open(`${base}/tune.html`);
await load();
await pick("403/128");
await b.waitFor(`document.querySelector("#recipe-file b")?.innerText === "act_damp05_r3.yaml"`, 30000);
const shut = await b.evaluate(`document.getElementById("sec-out").hasAttribute("data-off")`);
await click("#run-eval");
await b.waitFor(`!document.getElementById("sec-out").hasAttribute("data-off") && !!document.querySelector("#size")`, 10000);
const sizeText = await text("#size");
const segments = await b.evaluate(`[...document.querySelectorAll("#size .track")].map((t) => [...t.children].map((i) => parseFloat(i.style.width)))`);
await b.evaluate(`document.getElementById("sec-out").scrollIntoView({block: "start"})`);
await b.pause(400);
await b.shot(`${out}/out_1440.png`);
await click("#export");
let zipped = [];
for (let i = 0; i < 50 && !(zipped = readdirSync(downloads).filter((f) => f.endsWith(".zip"))).length; i++) await b.pause(200);
const listed = await text("#export-block");
const hasAnswers = await b.evaluate(`!!document.querySelector("#answers")`);
check("13 step 5 after the evaluation: the computed size by part beside the recipe bundle",
  shut && sizeText.includes("1.50 GB → 0.80 GB (53%)") && sizeText.includes("원본의 20%") && sizeText.includes("계산값") &&
  segments.length === 2 && segments[0].length === 2 && Math.abs(segments[0][0] + segments[0][1] - 100) < 0.1 &&
  segments[1][0] < segments[0][0] / 4 && zipped.join() === "act_damp05_r3_tricast.zip" &&
  listed.includes("act_damp05_r3.yaml · evaluation.json · size.json · README.md") && !hasAnswers,
  `closed before the evaluation ${shut} · "${sizeText.slice(0, 60)}…" bars ${segments.map((s) => s.map((v) => v.toFixed(1)).join("+")).join(" | ")} · ` +
  `download ${zipped.join()} · listed "${listed.slice(listed.indexOf("act_damp05"), listed.indexOf("README.md") + 9)}"`);

// ---------- live fixture: a GPU job with made-up numbers ----------
const fixture = spawn(python, [`${repo}/scripts/quantizer/tune_fixture_server.py`, "0"], {cwd: repo, stdio: ["ignore", "pipe", "pipe"],
  env: {...process.env, PYTHONPATH: `${repo}/src`}});
const port = await new Promise((resolvePort, reject) => {
  let log = "";
  const onData = (chunk) => {
    log += chunk;
    const match = /TUNE_FIXTURE_READY (\d+)/.exec(log);
    if (match) resolvePort(Number(match[1]));
  };
  fixture.stdout.on("data", onData);
  fixture.stderr.on("data", onData);
  fixture.on("exit", (code) => reject(new Error(`fixture exited (${code}): ${log.slice(-400)}`)));
  setTimeout(() => reject(new Error(`fixture did not start: ${log.slice(-400)}`)), 30000);
});
try {
  const live = `http://127.0.0.1:${port}`;
  await b.setup(1440, 900);
  await open(`${live}/tune.html`);
  await load();
  await click(`.ta-preset[data-budget="403/128"]`);
  await b.waitFor(`!!document.querySelector('.ta-preset[data-option="default"]')`, 30000);
  await click(`.ta-preset[data-option="default"]`);  // the fixture serves no session files: the default settings' recipe
  await b.waitFor(`document.querySelector("#recipe-file b")?.innerText === "iso_3.15.yaml" && !document.querySelector("#run-eval").disabled`, 30000);
  const label = await text("#run-eval");
  const bar = await text("#server");
  await b.evaluate(`[...document.querySelectorAll(".ta-tasks label")].find((l) => l.innerText.includes("ARC-Easy")).querySelector("input").click()`);
  await click("#run-eval");
  const progressed = await b.waitFor(`/중/.test(document.querySelector("#eval-state")?.innerText ?? "") && !!document.querySelector("#eval .ta-progress")`, 10000);
  await b.waitFor(`!!document.querySelector("#eval-result") || !!document.querySelector("#eval .ta-err")`, 30000);
  const ev = await text("#eval");
  const opened = await b.waitFor(`!document.getElementById("sec-out").hasAttribute("data-off") && !!document.querySelector("#size")`, 5000);
  await b.shot(`${out}/live_job.png`);
  check("14 a GPU job: progress, PPL, the lm-eval task, then step 5 opens (fixture numbers)",
    label.includes("GPU 에서 평가 실행") && bar.includes("GPU 서버") && progressed && ev.includes("26.60") && ev.includes("arc_easy") && ev.includes("방금 측정") &&
    opened,
    `button "${label}" · bar "${bar}" · progress shown ${progressed} · ${ev.includes("26.60") ? "PPL 26.60" : "no PPL"} · arc_easy ${ev.includes("arc_easy")} · ` +
    `step 5 opened ${opened}`);
  await b.evaluate(`[...document.querySelectorAll(".ta-tasks label")].find((l) => l.innerText.includes("HellaSwag")).querySelector("input").click()`);
  await click("#run-eval");
  await b.waitFor(`(document.querySelector("#eval .ta-err")?.innerText ?? "").includes("평가 실패")`, 30000);
  const err = await text("#eval .ta-err");
  check("15 a failed GPU job says so and why", err.includes("평가 실패") && err.includes("hellaswag"), err);
} finally {
  fixture.kill("SIGTERM");
}

const errors = b.events.filter((e) => e.method === "Runtime.exceptionThrown").map((e) => e.params?.exceptionDetails?.exception?.description ?? "");
check("page errors", errors.length === 0, errors.length ? errors.slice(0, 2).join(" / ") : "none");
await b.close().catch(() => {});
const passed = results.filter((r) => r.pass).length;
writeFileSync(`${out}/tune_app_check.json`, JSON.stringify({base, results}, null, 1));
console.log(`TUNE_APP_CHECKS ${passed}/${results.length}`);
process.exit(passed === results.length ? 0 : 1);
