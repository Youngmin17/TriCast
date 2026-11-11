// Rehearse the demo on a running server the way the presenter runs it, in the one tool (app/web/tune.html): put the
// model in, let Auto-tune recommend its setup for the demo size (the AI's measured settings on top), edit it by hand
// to one method on every layer, evaluate the edit, then return to the recommendation and evaluate that. Each step
// checks what the audience sees (numbers, "the file the GPU measured", where a result came from), times itself and
// keeps a screenshot. On a wide screen (over 760 px) the next thing to press must already be on screen.
// Meant for a server started with --agent (the AI-measured settings' recipe files come from the session folder);
// without --agent the page still shows every number and says those files need --agent.
//
// usage: node scripts/quantizer/demo_rehearsal.mjs <out_dir> [base_url=http://127.0.0.1:8765] [width=1440]
// Prints one STEP line per step and REHEARSAL_DONE <passed>/5; writes <out_dir>/rehearsal.json and step*.png.
import {mkdirSync, writeFileSync} from "node:fs";
import {resolve} from "node:path";

import {connect} from "./cdp_lib.mjs";

const [, , outArg, base = "http://127.0.0.1:8765", widthArg = "1440"] = process.argv;
if (!outArg) {
  console.error("usage: node scripts/quantizer/demo_rehearsal.mjs <out_dir> [base_url] [width]");
  process.exit(2);
}
const out = resolve(outArg);
mkdirSync(out, {recursive: true});
const health = await fetch(`${base}/api/health`).then((r) => r.json()).catch(() => null);
if (!health) {
  console.error(`no server at ${base}: start python -m app.server --demo --agent … first`);
  process.exit(2);
}
const mode = await fetch(`${base}/api/autotune/mode`).then((r) => r.json()).catch(() => ({agent: "off"}));
const off = mode.agent === "off";  // no --agent: the AI settings' recipe files are not served
const SIZE = "403/128";            // the demo size: 20% of the original
const EDIT = "awq_clip_int3";      // the hand edit: AWQ+clip 3-bit on every layer

const b = await connect(out);
await b.setup(Number(widthArg), 900);
const started = Date.now();
const steps = [];
const NAMES = ["모델 불러오기", "Auto-tune · AI 추천", "직접 편집", "편집한 셋업 평가", "추천으로 되돌려 평가"];
const text = (selector) => b.evaluate(`(document.querySelector(${JSON.stringify(selector)})?.innerText ?? "").replace(/\\s+/g, " ")`);
const click = (selector) => b.evaluate(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el) return false; el.click(); return true; })()`);
const isOff = (id) => b.evaluate(`document.getElementById(${JSON.stringify(id)}).hasAttribute("data-off")`);
const missing = (t, values) => values.filter((value) => !t.includes(value));
const wide = Number(widthArg) > 760;
// on a wide screen, whether `selector` is wholly inside the window (nothing to scroll to reach it)
const onScreen = async (selector) => !wide || await b.evaluate(`(() => { const r = document.querySelector(${JSON.stringify(selector)})?.getBoundingClientRect();
  return Boolean(r) && r.top >= 0 && r.bottom <= window.innerHeight; })()`);
// the recipe file card once it names `name`, or the page's reason there is none
const recipe = async (name) => {
  await b.waitFor(`(document.querySelector("#recipe-file")?.innerText ?? "").includes(${JSON.stringify(name)}) || !!document.querySelector("#edit .ta-err")`, 30000);
  return {file: await text("#recipe-file"), error: await text("#edit .ta-err")};
};
const evaluate = async () => {
  await click("#run-eval");
  await b.waitFor(`!!document.querySelector("#eval-result") || !!document.querySelector("#eval .ta-err")`, health.mode === "live" ? 3600000 : 20000);
  return text("#sec-eval");
};

async function step(number, run) {
  const t0 = Date.now();
  let pass = false;
  let detail = "";
  try {
    [pass, detail] = await run();
  } catch (error) {
    detail = `error: ${error.message}`;
  }
  await b.pause(900);  // a result scrolls into view smoothly
  const shot = await b.shot(`${out}/step${String(number).padStart(2, "0")}.png`);
  const row = {step: number, title: NAMES[number - 1], pass: Boolean(pass), detail, seconds: (Date.now() - t0) / 1000,
    at_seconds: (Date.now() - started) / 1000, shot};
  steps.push(row);
  console.log(`STEP ${pass ? "PASS" : "FAIL"} ${number} ${row.title}: ${detail} (${row.seconds.toFixed(1)}s)`);
}

const opened = await b.open(`${base}/tune.html`, `document.body.dataset.ready === "yes"`);
if (!opened) {
  console.error("the tool did not render");
  process.exit(1);
}

await step(1, async () => {
  await click("#load");
  await b.waitFor(`!document.getElementById("sec-tune").hasAttribute("data-off")`, 20000);
  const model = `${await b.evaluate(`document.querySelector("#model").value`)} ${await text("#sec-model")}`;
  const sizes = await b.evaluate(`document.querySelectorAll(".ta-preset[data-budget]").length`);
  const lack = missing(model, ["Qwen/Qwen3-0.6B", "196개", "16종", "준비됨"]);
  const waiting = (await isOff("sec-edit")) && (await isOff("sec-eval"));
  return [!lack.length && sizes === 6 && waiting,
    lack.length ? `없음: ${lack}` : `Qwen/Qwen3-0.6B · Linear 196개 · 방법 16종 · 목표 크기 ${sizes}개 · 편집 · 평가는 추천 뒤에 열림 ${waiting}`];
});

await step(2, async () => {
  await click(`.ta-preset[data-budget="${SIZE}"]`);
  await b.waitFor(`!!document.querySelector("#reco") && document.querySelectorAll(".ta-cell").length === 196`, 30000);
  const got = await recipe("act_damp05_r3.yaml");
  await b.pause(700);  // the evaluation's title row comes into view on a short screen
  const reco = await text("#reco");
  const lack = missing(reco, ["AI 추천", "26.51", "33.31 → 27.04", "27.04 → 26.51", "act_order = true, damp = 0.05", "Worker"])
    .concat(missing(await text("#tune-state"), ["20%", "26.51"]));
  const fileOk = off ? got.error.includes("--agent") : !missing(got.file, ["act_damp05_r3.yaml", "GPU 가 측정한 바로 그 파일"]).length;
  const next = await onScreen("#run-eval");
  return [!lack.length && fileOk && next,
    lack.length ? `없음: ${lack}` : `크기 20% 추천 PPL 26.51 — 층별 배정 33.31 → 27.04, AI 세부 설정 27.04 → 26.51 · ` +
      `${off ? "레시피 파일은 --agent 서버 안내" : `act_damp05_r3.yaml 측정 파일과 같음 ${/SHA-256 ([0-9a-f]{12})/.exec(got.file)?.[1] ?? ""}`}` +
      `${wide ? ` · 평가 버튼 화면 안 ${next}` : ""}`];
});

await step(3, async () => {
  const offered = await b.evaluate(`[...document.querySelectorAll("#assign optgroup option")].map((o) => o.value)`);
  await b.evaluate(`(() => { const s = document.querySelector("#assign"); s.value = ${JSON.stringify(EDIT)}; s.dispatchEvent(new Event("change", {bubbles: true})); })()`);
  const got = await recipe(`u_${EDIT}.yaml`);
  await b.pause(700);
  const bits = await b.evaluate(`[...new Set([...document.querySelectorAll(".ta-cell")].map((c) => c.dataset.b))].join(",")`);
  const state = await text("#edit-state");
  const lack = missing(got.file, [`u_${EDIT}.yaml`, "GPU 가 측정한 바로 그 파일"]).concat(missing(state, ["직접 편집함", "추천으로 되돌리기"]));
  const next = await onScreen("#run-eval");
  return [offered.includes(EDIT) && !lack.length && bits === "3" && next,
    lack.length ? `없음: ${lack}` : `전체를 AWQ+클립 3비트로 (측정된 한 방법 ${offered.length}개 중) · 격자 196칸 모두 3비트 · u_${EDIT}.yaml 측정 파일과 같음 ` +
      `${/SHA-256 ([0-9a-f]{12})/.exec(got.file)?.[1] ?? ""}${wide ? ` · 평가 버튼 화면 안 ${next}` : ""}`];
});

await step(4, async () => {
  const ev = await evaluate();
  const want = health.mode === "live" && health.device === "cuda" ? ["PPL"] : ["33.31", "20.97", "+12.35", "SHA 일치", "기록된 GPU 측정", "55% 작습니다"];
  const lack = missing(ev, want);
  return [!lack.length, lack.length ? `없음: ${lack}` : "편집한 셋업 PPL 33.31 (원본 20.97 대비 +12.35) · 기록된 GPU 측정 · 레시피 SHA 일치 · 추천이 55% 작음"];
});

await step(5, async () => {
  await click("#reset");
  const got = await recipe("act_damp05_r3.yaml");
  const state = await text("#edit-state");
  const ev = await evaluate();
  const want = health.mode === "live" && health.device === "cuda" ? ["PPL"]
    : ["26.51", "20.97", "33.31", "27.04", "26.82", "55% 작습니다", off ? "대조 안 됨" : "SHA 일치"];
  const lack = missing(ev, want).concat(missing(state, ["추천 그대로"]));
  const fileOk = off ? got.error.includes("--agent") : got.file.includes("act_damp05_r3.yaml");
  return [!lack.length && fileOk, lack.length ? `없음: ${lack}` : `추천으로 되돌림 · PPL 26.51 · 원본 20.97 · 한 방법 33.31 · Auto-tune 27.04 · AI 설정 26.82 / 26.51` +
    `${off ? "" : " · 레시피 SHA 일치"}`];
});

const errors = b.events.filter((e) => e.method === "Runtime.exceptionThrown").length;
const overflow = await b.evaluate(`document.documentElement.scrollWidth > window.innerWidth + 1`);
await b.close().catch(() => {});  // Chrome can still hold its profile for a moment; the run is done
const passed = steps.filter((row) => row.pass).length;
writeFileSync(`${out}/rehearsal.json`, JSON.stringify({base, health, mode, width: Number(widthArg), steps, page_errors: errors,
  horizontal_overflow: overflow, total_seconds: (Date.now() - started) / 1000}, null, 1));
console.log(`TUNE_APP agent=${mode.agent} width=${widthArg} overflow=${overflow}`);
console.log(`REHEARSAL_DONE ${passed}/${steps.length} page_errors=${errors} total=${((Date.now() - started) / 1000).toFixed(1)}s out=${out}`);
process.exit(passed === steps.length && !errors && !overflow ? 0 : 1);
