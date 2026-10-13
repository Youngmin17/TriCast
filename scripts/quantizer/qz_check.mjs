// Checks for app/web/quantizer.html in headless Chrome (Google Chrome at the macOS path in cdp_lib.mjs, Node >= 22):
// JS ↔ Python recipe parity (presets, 300 random grids, 300 random states with per-method options, calibration and KV
// options, each one batch against `python -m app.quantizer_recipe --from-json`), interactions (grid, method options,
// calibration panel, KV options, invalid values), copy/download, layout, themes, empty/error/recorded states.
// Prints PASS/FAIL per check and QZ_CHECKS[<label>] n/m; writes <out_dir>/qz_accept_input.json for qz_tricast_accept.py.
//
// usage (from the repo root, python = an env with pyyaml; static servers started separately):
//   bash scripts/quantizer/prep_alt_roots.sh . <alt_dir> <python>
//   python -m http.server 8765 -d app/web &  python -m http.server 8766 -d <alt_dir> &
//   node scripts/quantizer/qz_check.mjs <out_dir> http://127.0.0.1:8765 http://127.0.0.1:8766 . <python> [label]
//   PYTHONPATH=src:. python scripts/quantizer/qz_tricast_accept.py <out_dir>/qz_accept_input.json
// <alt_base_url> serves prep_alt_roots.sh output (empty/, bad/, synth/ — synth numbers are made up).
import {execFileSync} from "node:child_process";
import {mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync} from "node:fs";

import {connect} from "./cdp_lib.mjs";
import {MIXED_FLOW, READY, helpers, mockLive} from "./qz_flow.mjs";

const [, , outDir, base, alt, repo, python, label = "static"] = process.argv;
mkdirSync(outDir, {recursive: true});
const py = (args) => execFileSync(python, args, {cwd: repo, encoding: "utf-8", maxBuffer: 1 << 28,
  env: {...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: repo}});
const b = await connect(outDir, {logging: true});
const {evaluate, pause} = b;
const q = helpers(b);
const results = {label, checks: [], shots: []};
const check = (name, pass, detail) => results.checks.push({label: name, pass: Boolean(pass), detail});
const shot = async (file) => results.shots.push(await b.shot(`${outDir}/${file}.png`));
const presetIds = ["qd_mxfp4", "qd_mxfp4_ends_bf16", "qd_nvfp4", "qd_nvint4", "qd_mixed"];
const cliYaml = Object.fromEntries(presetIds.map((id) => [id, py(["-m", "app.quantizer_recipe", "--preset", id])]));

// ---------- load, light 1600 ----------
await b.setup(1600, 1000);
check("page loads", await b.open(`${base}/quantizer.html`, READY));
const st0 = await q.state();
check("opens as 새 레시피: NVFP4 everywhere, name my_recipe, page YAML trusted", st0.default === "nvfp4"
  && st0.name === "my_recipe" && st0.trusted === true && st0.grid.flat().every((m) => m === "nvfp4"), st0);
check("196 grid cells, 28 column and 7 row headers", (await evaluate(`[document.querySelectorAll(".qz-cell").length,
  document.querySelectorAll(".qz-colhead button").length, document.querySelectorAll(".qz-rowhead button").length].join()`)) === "196,28,7");
check("every cell shows a text code (identity is not color-only)", await evaluate(`[...document.querySelectorAll(".qz-cell")]
  .every((c) => /^[A-Z0-9]{2,3}$/.test(c.textContent))`));

// ---------- 1. preset parity via the chips ----------
for (const id of presetIds) {
  await q.click(`[data-preset="${id}"]`);
  const st = await q.state();
  const shown = await evaluate(`document.querySelector(".qz-yaml").textContent`);
  check(`preset ${id}: chip YAML == python -m app.quantizer_recipe --preset ${id}`, `${st.yaml}\n` === cliYaml[id]
    && shown === st.yaml && st.name === id && st.matched === id, {name: st.name, matched: st.matched, len: st.yaml.length});
}

// ---------- 2. 300 random states: page build/toYaml == python --from-json ----------
function rng(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(42);
const pick = (list) => list[Math.floor(rand() * list.length)];
const METHODS = ["bf16", "gptq_int4", "awq_nvfp4", "smoothquant_nvfp4", "nvfp4", "nvint4", "mxfp4", "mxfp8", "int8", "bfp12"];
const QUANT = METHODS.filter((m) => m !== "bf16");
const KVS = ["none", "kivi2", "kivi4", "kv_fp8"];
const NAMES = ["my_recipe", "qd_mixed", "none", "null", "yes", "No", "on", "OFF", "true", "", "1abc", "_x", "-lead", ".dot",
  "a b", "a-b.c_d", "a:b", "a#b", "café", "café", "레시피", "名前", "Ⅻ", "x²", "٣abc", "ǅx", "ﬁ", "ẞ", "𝐀bc",
  "tab\tname", "quote\"s", "back\\slash", "nl\nname", "cr\rname", "sep x", "nel\u0085x", "emoji😀", "bell\u0007",
  "nb sp", "1"];
function grid(def, mode) {
  const g = Array.from({length: 28}, () => Array.from({length: 7}, () => def));
  const fill = (p) => g.forEach((row) => row.forEach((_, k) => { if (rand() < p) row[k] = pick(METHODS); }));
  if (mode === 1) fill(0.3);
  if (mode === 4) fill(1);
  if (mode === 2 || mode === 5) for (let i = 0, n = 1 + Math.floor(rand() * 4); i < n; i += 1) g[Math.floor(rand() * 28)] = Array(7).fill(pick(METHODS));
  if (mode === 3 || mode === 5) {
    for (let i = 0, n = 1 + Math.floor(rand() * 3); i < n; i += 1) {
      const k = Math.floor(rand() * 7);
      const m = pick(METHODS);
      g.forEach((row) => (row[k] = m));
    }
  }
  if (mode === 5) fill(0.05);
  if (mode === 6) g.forEach((row) => row.fill("bf16"));
  if (mode === 7) g.forEach((row, b) => row.fill(b % 2 ? pick(METHODS) : def));
  return g;
}
const specs = Array.from({length: 300}, (_, i) => {
  const def = pick(QUANT);
  const name = i < NAMES.length ? NAMES[i] : `${pick(NAMES)}_${i}`;
  return {name, default: def, grid: grid(def, i % 8), kv: pick(KVS)};
});
const specFile = `${outDir}/qz_parity_specs.json`;
writeFileSync(specFile, JSON.stringify(specs));
const pageYaml = await evaluate(`(${JSON.stringify(specs)}).map((s) => window.__quantizer.toYaml(window.__quantizer.build(s.name, s.default, s.grid, s.kv)))`);
const expected = py(["-m", "app.quantizer_recipe", "--from-json", specFile]);
const joined = pageYaml.map((y) => `---\n${y}\n`).join("");
let firstBad = null;
if (joined !== expected) {
  let i = 0;
  while (i < joined.length && joined[i] === expected[i]) i += 1;
  firstBad = {at: i, page: JSON.stringify(joined.slice(Math.max(0, i - 80), i + 80)), python: JSON.stringify(expected.slice(Math.max(0, i - 80), i + 80))};
}
const kinds = {rules: pageYaml.filter((y) => y.includes("\noverrides:")).length, kv: pageYaml.filter((y) => y.includes("\nkv:")).length,
  quoted: pageYaml.filter((y) => y.startsWith("name: \"")).length, calib: pageYaml.filter((y) => y.includes("\ncalibration:")).length};
check("300 random states: page YAML byte-equal to python --from-json (one batch)", joined === expected && pageYaml.length === 300,
  {bytes: joined.length, firstBad, kinds});
// 50 more with file-safe names for the tricast acceptance check (load_recipe + spec_for, run separately).
const accept = Array.from({length: 50}, (_, i) => {
  const def = pick(QUANT);
  return {name: `qz_rand_${String(i).padStart(2, "0")}`, default: def, grid: grid(def, (i % 7) + 1), kv: pick(KVS)};
});
const acceptYaml = await evaluate(`(${JSON.stringify(accept)}).map((s) => window.__quantizer.toYaml(window.__quantizer.build(s.name, s.default, s.grid, s.kv)))`);

// ---------- 2b. 300 random states with per-method options, calibration and KV options ----------
// Raw random options are normalized by the page (what its controls would hold), then the same specs go to the page's
// build() and to python --from-json in one batch each.
const gen = JSON.parse(readFileSync(`${repo}/app/web/demo/quantizer/presets.json`, "utf8")).generator;
const dec = (v) => (Array.isArray(v) ? v.map(dec) : v && typeof v === "object" ? Object.fromEntries(v.items.map(([k, x]) => [k, dec(x)])) : v);
const OPT = dec(gen.options);
const MOPT = dec(gen.method_options);
const KVO = dec(gen.kv_options);
const CAL = dec(gen.calibration);
const FLOATS = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1, 2, 12.5, 50, 99, 99.9, 99.99, 100];
const inside = (spec, v) => (v === 0 || Math.abs(v) >= gen.float_min) && v >= spec.min && v <= spec.max
  && !(v === spec.min && spec.min_open) && !(v === spec.max && spec.max_open);
function randValue(spec) {
  if (spec.type === "choice") return pick(spec.choices);
  if (spec.type === "bool") return rand() < 0.5;
  if (spec.type === "int") {
    const step = spec.multiple ?? 1;
    const hi = Math.min(spec.max, spec.min + 64 * step);
    return spec.min + step * Math.floor(rand() * ((hi - spec.min) / step + 1));
  }
  for (;;) {
    const v = rand() < 0.7 ? pick(FLOATS) : Math.round((spec.min + rand() * (spec.max - spec.min)) * 1e4) / 1e4;
    if (inside(spec, v)) return v;
  }
}
function randOptions() {
  const raw = {};
  for (const m of QUANT) {
    if (rand() >= 0.55) continue;
    const o = Object.fromEntries(MOPT[m].filter(() => rand() < 0.5).map((id) => [id, randValue(OPT[id])]));
    if (m === "int8" && rand() < 0.6) Object.assign(o, {"activation.granularity": "tensor", "activation.observer.kind": pick(OPT["activation.observer.kind"].choices)});
    raw[m] = o;
  }
  return raw;
}
function optionState(name, mode) {
  const def = pick(QUANT);
  const kv = pick(KVS);
  return {name, default: def, grid: grid(def, mode), kv, raw: randOptions(),
    calibration: Object.fromEntries(Object.entries(CAL).filter(() => rand() < 0.25).map(([k, spec]) => [k, randValue(spec)])),
    kv_options: Object.fromEntries(Object.entries(KVO[kv]).filter(() => rand() < 0.6).map(([k, spec]) => [k, randValue(spec)]))};
}
async function normalize(states) {
  const options = await evaluate(`(${JSON.stringify(states.map((s) => s.raw))}).map((raw) => Object.fromEntries(Object.entries(raw)
    .map(([m, o]) => [m, window.__quantizer.normalizeOptions(m, o)]).filter(([, o]) => Object.keys(o).length)))`);
  return states.map(({raw, ...s}, i) => ({...s, options: options[i]}));
}
const buildAll = (list) => evaluate(`(${JSON.stringify(list)}).map((s) => window.__quantizer.toYaml(window.__quantizer.build(s.name, s.default, s.grid,
  s.kv, undefined, {options: s.options, calibration: s.calibration, kv_options: s.kv_options})))`);
const optSpecs = await normalize(Array.from({length: 300}, (_, i) => optionState(i < NAMES.length ? NAMES[i] : `${pick(NAMES)}_o${i}`, i % 8)));
const optFile = `${outDir}/qz_parity_option_specs.json`;
writeFileSync(optFile, JSON.stringify(optSpecs));
const optYaml = await buildAll(optSpecs);
const optExpected = py(["-m", "app.quantizer_recipe", "--from-json", optFile]);
const optJoined = optYaml.map((y) => `---\n${y}\n`).join("");
let optBad = null;
if (optJoined !== optExpected) {
  let i = 0;
  while (i < optJoined.length && optJoined[i] === optExpected[i]) i += 1;
  optBad = {at: i, page: JSON.stringify(optJoined.slice(Math.max(0, i - 120), i + 80)), python: JSON.stringify(optExpected.slice(Math.max(0, i - 120), i + 80))};
}
const optKinds = {withOptions: optSpecs.filter((s) => Object.keys(s.options).length).length,
  optionFields: new Set(optSpecs.flatMap((s) => Object.values(s.options).flatMap(Object.keys))).size,
  completeMappings: optYaml.filter((y) => /\n {4}block: 0\n/.test(y)).length, observers: optYaml.filter((y) => y.includes("observer:")).length,
  residual: optYaml.filter((y) => y.includes("residual:")).length, calib: optYaml.filter((y) => y.includes("\ncalibration:")).length,
  calibChanged: optYaml.filter((y) => /\ncalibration:\n {2}(dataset|samples|seqlen)/.test(y)).length};
check("300 random states WITH options/calibration/kv_options: page YAML byte-equal to python --from-json (one batch)",
  optJoined === optExpected && optYaml.length === 300 && optKinds.optionFields === Object.keys(OPT).length, {bytes: optJoined.length, optBad, optKinds});
const acceptOpt = await normalize(Array.from({length: 50}, (_, i) => optionState(`qz_opt_${String(i).padStart(2, "0")}`, (i % 7) + 1)));
const acceptOptYaml = await buildAll(acceptOpt);
writeFileSync(`${outDir}/qz_accept_input.json`, JSON.stringify([...accept.map((s, i) => ({...s, yaml: acceptYaml[i]})),
  ...acceptOpt.map((s, i) => ({...s, yaml: acceptOptYaml[i]}))]));

// ---------- 3. interactions ----------
await q.click('[data-preset="new"]');
await q.click('[data-default="awq_nvfp4"]');
let st = await q.state();
check("default card switches every default-following cell", st.default === "awq_nvfp4" && st.grid.flat().every((m) => m === "awq_nvfp4"), st.default);
await q.drag(q.cell(2, 1), q.cell(5, 3));
st = await q.state();
const rect = [];
for (let blk = 2; blk <= 5; blk += 1) for (let k = 1; k <= 3; k += 1) rect.push([blk, k]);
check("drag paints a rectangle selection (blocks 2–5 × k,v,o = 12 cells)", JSON.stringify(st.selection) === JSON.stringify(rect), st.selection);
await q.click('[data-apply="mxfp4"]');
st = await q.state();
check("palette applies MXFP4 to exactly the selection; 12 single-cell rules", rect.every(([blk, k]) => st.grid[blk][k] === "mxfp4")
  && st.grid.flat().filter((m) => m === "mxfp4").length === 12 && (st.yaml.match(/^ {2}- match: model\./gm) || []).length === 12,
  {count: st.grid.flat().filter((m) => m === "mxfp4").length});
await q.key("z", 2);
st = await q.state();
check("Ctrl+Z undoes the apply", st.grid.flat().every((m) => m === "awq_nvfp4") && !st.yaml.includes("overrides"));
await q.click('[data-col="7"]');
await q.click('[data-col="8"]', 8);
st = await q.state();
check("column header selects a block, Shift+click adds the next one (14 cells)", st.selection.length === 14
  && st.selection.every(([blk]) => blk === 7 || blk === 8), st.selection.length);
await q.click('[data-row="3"]');
st = await q.state();
check("row header selects one kind in all 28 blocks", st.selection.length === 28 && st.selection.every(([, k]) => k === 3));
await q.click(q.cell(10, 0));
await q.click(q.cell(12, 2), 8);
st = await q.state();
check("click then Shift+click extends to the rectangle between them (3 × 3)", st.selection.length === 9, st.selection);
await q.key("Escape");
st = await q.state();
check("Esc clears the selection", st.selection.length === 0);
await evaluate(`document.querySelector('${q.cell(0, 0)}').focus()`);
await q.key("ArrowRight");
await q.key("ArrowRight");
await q.key("ArrowDown");
await q.key(" ");
await q.key("ArrowRight", 8);
st = await q.state();
const focusVisible = await evaluate(`getComputedStyle(document.activeElement).outlineStyle !== "none" && document.activeElement.matches(".qz-cell")`);
check("keyboard: arrows move focus, Space selects, Shift+→ extends (cells b2,b3 × k)", JSON.stringify(st.selection) === "[[2,1],[3,1]]"
  && focusVisible, {sel: st.selection, focusVisible});
await q.key("7");
st = await q.state();
check("digit 7 applies the 7th palette method (MXFP8) to the selection", st.grid[2][1] === "mxfp8" && st.grid[3][1] === "mxfp8"
  && st.grid.flat().filter((m) => m === "mxfp8").length === 2);
await q.key("Escape");
await q.key("0");
st = await q.state();
check("digit 0 with nothing selected applies the 10th method (BF16 유지) to the focused cell", st.grid[3][1] === "bf16"
  && st.yaml.includes("match: model.layers.3.self_attn.k_proj\n    skip: true"));
await q.key("z", 4);
await q.key("z", 4);
st = await q.state();
check("⌘Z twice goes back to the plain AWQ grid", st.grid.flat().every((m) => m === "awq_nvfp4"), st.history);
await q.click('[data-quick="기본값으로 되돌리기"]');
await shot(`qz_${label}_light_1600_interactions`);

// ---------- 3b. method options, calibration panel, KV options ----------
const opt = (id) => `#qz-f-${id.replaceAll(".", "-")}`;
const press = async (key, code, keyCode, text) => {
  for (const type of ["keyDown", "keyUp"]) await b.S("Input.dispatchKeyEvent", {type, key, code, windowsVirtualKeyCode: keyCode, ...(type === "keyDown" && text ? {text} : {})});
  await pause(60);
};
const choose = async (id, value) => {  // what picking an <option> does: the value changes, then "change" fires
  await evaluate(`(() => { const el = document.querySelector(${JSON.stringify(opt(id))}); el.focus(); el.value = ${JSON.stringify(String(value))};
    el.dispatchEvent(new Event("change", {bubbles: true})); })()`);
  await pause(60);
};
const typeInto = async (id, text) => {  // real text input into a focused, selected number field
  await evaluate(`(() => { const el = document.querySelector(${JSON.stringify(opt(id))}); el.focus(); el.select(); })()`);
  await b.S("Input.insertText", {text});
  await pause(60);
};
const blurAll = async () => { await evaluate("document.activeElement?.blur()"); await pause(60); };
const fieldState = (id) => evaluate(`(() => { const el = document.querySelector(${JSON.stringify(opt(id))}); const why = document.getElementById(el.id + "-why");
  return {value: el.type === "checkbox" ? el.checked : el.value, disabled: el.disabled, hidden: el.closest(".qz-field").hidden,
    invalid: el.getAttribute("aria-invalid"), why: why.textContent}; })()`);
const dots = (m) => evaluate(`[...document.querySelectorAll('[data-optdot="${m}"]')].map((d) => !d.hidden)`);
await q.click('[data-preset="new"]');
const collapsed = await evaluate(`[...document.querySelectorAll(".qz-method .qz-opt-toggle")].map((t) => [t.getAttribute("aria-expanded"),
  document.getElementById(t.getAttribute("aria-controls")).hidden])`);
check("9 method cards each have a collapsed 옵션 disclosure (aria-expanded=false, panel hidden); KV 옵션 hidden while KV is 없음",
  collapsed.length === 9 && collapsed.every(([e, h]) => e === "false" && h === true)
  && (await evaluate(`document.querySelector('[data-options="kv"]').hidden`)) === true, collapsed);
await evaluate(`document.querySelector('[data-options="nvfp4"]').focus()`);
await press("Enter", "Enter", 13, "\r");
const opened = await evaluate(`[document.querySelector('[data-options="nvfp4"]').getAttribute("aria-expanded"), document.getElementById("qz-opt-nvfp4").hidden,
  getComputedStyle(document.activeElement).outlineStyle, document.querySelector("#qz-opt-nvfp4 .qz-fixed").textContent]`);
check("keyboard: Enter on the focused NVFP4 옵션 toggle opens its panel (focus ring visible); the panel says block 16 is fixed by NVFP4",
  opened[0] === "true" && opened[1] === false && opened[2] !== "none" && opened[3].includes("블록 16") && opened[3].includes("고정"), opened);
const nvOpts = await evaluate(`[...document.querySelectorAll("#qz-opt-nvfp4 .qz-field:not([hidden]) [data-field]")].map((e) => e.dataset.field)`);
check("NVFP4 offers only its valid options: scale method + rounding for weight and activation (no group size, no observer)",
  JSON.stringify(nvOpts) === JSON.stringify(["nvfp4.weight.scale.method", "nvfp4.weight.rounding", "nvfp4.activation.scale.method", "nvfp4.activation.rounding"]), nvOpts);
let before = await q.state();
await evaluate(`document.querySelector(${JSON.stringify(opt("nvfp4.weight.rounding"))}).focus()`);
await press("s", "KeyS", 83, "s");  // type-ahead on the focused select picks "sr · 확률적 반올림"
st = await q.state();
const shownYaml = await evaluate(`document.querySelector(".qz-yaml").textContent`);
check("typing 's' on 가중치 원소 반올림 picks sr: YAML defaults.weight gains rounding: sr (inside the scheme mapping), live in the panel",
  st.yaml.includes("  weight:\n    scheme: nvfp4\n    rounding: sr\n    mma_input: dequant") && shownYaml === st.yaml
  && st.options.nvfp4?.["weight.rounding"] === "sr" && st.history === before.history + 1, {options: st.options, yaml: st.yaml.slice(0, 220)});
check("a changed option lights the amber dot on the NVFP4 card and in the legend, and the toggle says 1개 바꿈",
  JSON.stringify(await dots("nvfp4")) === "[true,true]" && (await evaluate(`document.querySelector('[data-optn="nvfp4"]').textContent`)).includes("1개 바꿈")
  && JSON.stringify(await dots("mxfp4")) === "[false,false]", await dots("nvfp4"));
await blurAll();
await q.key("z", 2);
st = await q.state();
check("Ctrl+Z takes the option back: YAML without rounding, dots off, the select shows rne again",
  !st.yaml.includes("rounding:") && JSON.stringify(st.options) === "{}" && JSON.stringify(await dots("nvfp4")) === "[false,false]"
  && (await fieldState("nvfp4.weight.rounding")).value === "rne", st.options);
await q.click('[data-default="int8"]');
await q.click('[data-options="int8"]');
let obs = await fieldState("int8.activation.observer.kind");
check("INT8 with per-token activations: the observer control is disabled and says why (static observers need one tensor scale)",
  obs.disabled === true && obs.hidden === false && obs.why.includes("텐서 하나") && obs.value === "none", obs);
await choose("int8.activation.granularity", "tensor");
await choose("int8.activation.observer.kind", "ema");
st = await q.state();
const scaleBlocked = await fieldState("int8.activation.scale.method");
const decay = await fieldState("int8.activation.observer.decay");
const badge = await evaluate(`!document.querySelector('[data-calib="int8"]').hidden`);
const calibReadout = await evaluate(`document.querySelector('[data-readout="calib"]').textContent`);
check("tensor + ema: observer written inside the activation scheme mapping, decay field appears, scale method blocked with its reason",
  st.yaml.includes("  activation:\n    scheme: int8_row\n    granularity: tensor\n    observer:\n      kind: ema\n    mma_input: dequant"),
  {yaml: st.yaml.slice(0, 400)});
check("…the static observer needs calibration: card badge 보정 필요, calibration {seed: 42} written, summary 보정 필요 with samples × seqlen tokens",
  badge && st.needs_calibration && st.yaml.endsWith("calibration:\n  seed: 42") && calibReadout.includes("보정 필요")
  && calibReadout.includes("128 × 2,048 = 262,144 토큰") && decay.hidden === false && decay.value === "0.99"
  && scaleBlocked.disabled && scaleBlocked.why.includes("absmax"), {badge, calibReadout, decay, scaleBlocked});
await typeInto("int8.activation.observer.decay", "1");
const badDecay = await fieldState("int8.activation.observer.decay");
const blocked = await evaluate(`[document.querySelector(".qz-yamlpanel .copy + .copy").disabled, document.querySelector(".qz-yamlpanel .note").textContent]`);
st = await q.state();
check("decay 1 is refused in one line (0 이상 1 미만), the YAML keeps the last good value and download waits",
  badDecay.invalid === "true" && badDecay.why === "0 이상 1 미만이어야 합니다" && blocked[0] === true && blocked[1].includes("고칠 입력")
  && !st.yaml.includes("decay:") && st.invalid.includes("int8.activation.observer.decay"), {badDecay, blocked});
await typeInto("int8.activation.observer.decay", "0.9");
st = await q.state();
check("decay 0.9 is accepted: decay: 0.9 in the YAML, download enabled again", st.yaml.includes("    observer:\n      kind: ema\n      decay: 0.9\n")
  && !(await evaluate(`document.querySelector(".qz-yamlpanel .copy + .copy").disabled`)), st.yaml.slice(0, 400));
await typeInto("int8.activation.observer.decay", "1.5");
await blurAll();
const restored = await fieldState("int8.activation.observer.decay");
st = await q.state();
check("leaving the field with 1.5 restores 0.9 (nothing invalid reaches the recipe)", restored.value === "0.9" && restored.invalid === "false"
  && st.invalid.length === 0 && st.options.int8["activation.observer.decay"] === 0.9, restored);
await choose("int8.activation.granularity", "row");
st = await q.state();
check("back to per-token: the observer stops applying, YAML drops it and calibration (no other calibrated method)",
  !st.yaml.includes("observer") && !st.yaml.includes("calibration") && (await fieldState("int8.activation.observer.kind")).disabled, st.yaml.slice(-120));
await typeInto("calibration.samples", "64");
st = await q.state();
const calibSub = await evaluate(`document.querySelector('[data-readout="calib"]').textContent`);
check("calibration panel: samples 64 is written even without a calibrated method (changed value + seed 42); status says it is unused",
  st.yaml.endsWith("calibration:\n  samples: 64\n  seed: 42") && calibSub.includes("없음")
  && (await evaluate(`document.querySelector(".qz-calib-status").textContent`)).includes("쓰이지 않습니다"), st.yaml.slice(-80));
await choose("int8.activation.granularity", "tensor");
st = await q.state();
check("with the ema observer back, the summary estimates 64 × 2,048 = 131,072 calibration tokens",
  (await evaluate(`document.querySelector('[data-readout="calib"]').textContent`)).includes("64 × 2,048 = 131,072 토큰") && st.needs_calibration);
await typeInto("calibration.samples", "600");
const samples = await fieldState("calibration.samples");
const whileBad = await q.state();
await typeInto("calibration.seqlen", "1");  // moving on restores samples to its last good value
const seqlen = await fieldState("calibration.seqlen");
const samplesAfter = await fieldState("calibration.samples");
st = await q.state();
check("calibration validation: samples 600 → '1 이상 512 이하의 정수여야 합니다' (restored to 64 on leaving), seqlen 1 → '2 이상 4096 이하…'; YAML keeps 64 × 2048",
  samples.why === "1 이상 512 이하의 정수여야 합니다" && JSON.stringify(whileBad.invalid) === '["calibration.samples"]'
  && samplesAfter.value === "64" && samplesAfter.invalid === "false" && seqlen.why === "2 이상 4096 이하의 정수여야 합니다"
  && st.yaml.includes("samples: 64") && !st.yaml.includes("seqlen") && JSON.stringify(st.invalid) === '["calibration.seqlen"]',
  {samples, samplesAfter, seqlen, invalid: [whileBad.invalid, st.invalid]});
await blurAll();
await choose("calibration.dataset", "c4");
st = await q.state();
check("dataset C4 → calibration: {dataset: c4, samples: 64, seed: 42}", st.yaml.endsWith("calibration:\n  dataset: c4\n  samples: 64\n  seed: 42")
  && st.invalid.length === 0, st.yaml.slice(-90));
await q.click('[data-kv="kivi2"]');
await q.click('[data-options="kv"]');
await typeInto("kv.kivi2.residual", "48");
const res48 = await fieldState("kv.kivi2.residual");
await typeInto("kv.kivi2.residual", "64");
st = await q.state();
check("KV options: KIVI-2 residual 48 is refused (32의 배수), 64 gives kv: {preset: kivi2, mode: fakequant, residual: 64}",
  res48.why === "32의 배수여야 합니다" && st.yaml.includes("kv:\n  preset: kivi2\n  mode: fakequant\n  residual: 64")
  && JSON.stringify(st.kv_options) === '{"residual":64}', {res48, kv: st.kv_options});
await blurAll();
await evaluate("document.querySelector('.qz-config').scrollTo?.(0, 0); document.getElementById('qz').scrollTo(0, 0)");
await shot(`qz_${label}_light_1600_options`);
await q.click('[data-preset="new"]');
st = await q.state();
check("새 레시피 resets options, calibration and KV fields; the YAML is the plain NVFP4 recipe again",
  JSON.stringify([st.options, st.calibration, st.kv_options]) === "[{},{},{}]" && !st.yaml.includes("calibration") && !st.yaml.includes("kv:")
  && JSON.stringify(await dots("int8")) === "[false,false]", [st.options, st.calibration, st.kv_options]);

// ---------- 4. click flow to qd_mixed ----------
for (const [, run] of MIXED_FLOW) await run(q);
st = await q.state();
const mixedPython = cliYaml.qd_mixed.slice(0, -1);
check("click flow (새 레시피 → AWQ → o_proj MXFP8 → down_proj GPTQ → ends BF16 → KIVI-2) reaches preset qd_mixed",
  st.matched === "qd_mixed" && st.name === "my_recipe" && st.yaml === mixedPython.replace("name: qd_mixed", "name: my_recipe"), st.matched);
const chipOn = await evaluate(`document.querySelector('[data-preset="qd_mixed"]').getAttribute("aria-pressed")`);
const record = await evaluate(`document.querySelector(".qz-record").textContent`);
const recorded = JSON.parse(readFileSync(`${repo}/app/web/demo/quantizer/presets.json`, "utf8")).presets.find((p) => p.id === "qd_mixed");
check("results column never prints null/undefined", !/null|undefined|NaN/.test(await evaluate(`(() => { const c = document.querySelector(".qz-results").cloneNode(true);
  c.querySelectorAll("pre, code").forEach((x) => x.remove()); return c.textContent; })()`)));
check("qd_mixed chip lights up and the record panel matches what presets.json holds", chipOn === "true" && (recorded.run || recorded.report
  ? record.includes("qd_mixed") && record.includes(recorded.recipe_sha256.slice(0, 12))
  : record.includes("이 조합은 프리셋 qd_mixed 와 같지만 아직 기록이 없습니다")), record.slice(0, 160));
await evaluate(`(() => { const i = document.getElementById("qz-name"); i.focus(); i.select(); })()`);
await b.S("Input.insertText", {text: "qd_mixed"});
await pause(60);
st = await q.state();
check("typing the name qd_mixed makes the YAML byte-identical to the preset", st.yaml === mixedPython && st.name === "qd_mixed");
await shot(`qz_${label}_light_1600_mixed`);

// ---------- 5. copy and download ----------
await evaluate(`window.__copied = []; navigator.clipboard.writeText = (t) => { window.__copied.push(t); return Promise.resolve(); }`);
await q.click(".qz-yamlpanel .copy");
await q.click(".qz-cmds li:nth-child(2) .copy");
await pause(100);
const copied = await evaluate("window.__copied");
const copyLabel = await evaluate(`document.querySelector(".qz-yamlpanel .copy").textContent`);
check("YAML 복사 copies the YAML (+ newline); a command button copies its command; label confirms",
  copied[0] === cliYaml.qd_mixed && copied[1] === "tricast report --model Qwen/Qwen3-0.6B --recipe qd_mixed.yaml" && copyLabel === "복사됨",
  {copied: copied.map((t) => t.slice(0, 40)), copyLabel});
const downloads = `${outDir}/downloads_${label}`;
rmSync(downloads, {recursive: true, force: true});
mkdirSync(downloads, {recursive: true});
await b.S("Page.setDownloadBehavior", {behavior: "allow", downloadPath: downloads});
await q.click(".qz-yamlpanel .copy + .copy");
let files = [];
for (let i = 0; i < 50 && !files.includes("qd_mixed.yaml"); i += 1) {
  await pause(100);
  files = readdirSync(downloads);
}
const fileText = files.includes("qd_mixed.yaml") ? readFileSync(`${downloads}/qd_mixed.yaml`, "utf8") : null;
check("내려받기 saves qd_mixed.yaml byte-identical to python --preset qd_mixed", fileText === cliYaml.qd_mixed, files);
const cmds = await evaluate(`[...document.querySelectorAll(".qz-cmds .qz-cmd")].map((c) => c.textContent)`);
check("commands: recipe-check, report, ppl, lm-eval 5 tasks --limit 300, tricast run <name>_bench.yaml, export_layers",
  cmds.join("\n") === ["tricast recipe-check qd_mixed.yaml", "tricast report --model Qwen/Qwen3-0.6B --recipe qd_mixed.yaml",
    "tricast ppl --model Qwen/Qwen3-0.6B --recipe qd_mixed.yaml",
    "tricast eval --model Qwen/Qwen3-0.6B --recipe qd_mixed.yaml --tasks arc_easy,arc_challenge,piqa,winogrande,hellaswag --limit 300",
    "tricast run qd_mixed_bench.yaml", "python -m app.export_layers runs/<run>/report --out app/web/demo/layers"].join("\n"), cmds);
await q.click(".qz-commands .qz-actions .copy");
files = [];
for (let i = 0; i < 50 && !files.includes("qd_mixed_bench.yaml"); i += 1) {
  await pause(100);
  files = readdirSync(downloads);
}
const benchCfg = files.includes("qd_mixed_bench.yaml") ? `${downloads}/qd_mixed_bench.yaml` : null;
// The `tricast run` config of the recorded lm-eval runs (runs/20261013_quantizer_w3/lmeval_A.yaml), for this recipe.
const LMEVAL_RUN = {model: "Qwen/Qwen3-0.6B", dtype: "bfloat16", device: "cuda", seed: 42, native_baseline: true,
  tasks: {lm_eval: {tasks: ["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"], batch_size: 16, limit: 300}},
  recipes: ["qd_mixed.yaml"], output_dir: "runs/qd_mixed_bench"};
const cfgSame = benchCfg && py(["-c", `import json, sys, yaml
a = yaml.safe_load(open(sys.argv[1])); b = json.loads(sys.argv[2])
print(json.dumps([a == b, sorted(a)]))`, benchCfg, JSON.stringify(LMEVAL_RUN)]).trim();
check("벤치 설정 내려받기: same keys and lm_eval block as the recorded lm-eval run config, recipe = qd_mixed.yaml",
  cfgSame && JSON.parse(cfgSame)[0] === true, cfgSame);
const api = await evaluate(`document.querySelector(".qz-api").textContent`);
check("Python API snippet: 8 lines, existing tricast functions, this recipe file", api.split("\n").length === 8
  && api.includes('load_recipe("qd_mixed.yaml")') && api.includes("from tricast import load_recipe, patch_model, calibrate"), api);
const gpuNote = await evaluate(`[document.querySelector(".qz-gpu-run").disabled, document.getElementById("qz-gpu-note").textContent]`);
check(`server-GPU button disabled with the reason (${label})`, gpuNote[0] === true && gpuNote[1].includes("python -m app.server --device cuda")
  && gpuNote[1].includes(label === "demo" ? "데모 서버" : "정적 파일 서버"), gpuNote);
await evaluate(`(() => { const i = document.getElementById("qz-name"); i.focus(); i.select(); })()`);
await b.S("Input.insertText", {text: "bad name/x"});
await pause(60);
const nameErr = await evaluate(`[document.getElementById("qz-name-error").textContent, document.querySelector(".qz-yamlpanel .copy + .copy").disabled]`);
check("an unsafe file name says how to fix it and disables 내려받기", nameErr[0].includes("영문·숫자로 시작") && nameErr[1] === true, nameErr);
await evaluate(`(() => { const i = document.getElementById("qz-name"); i.select(); })()`);
await b.S("Input.insertText", {text: "qd_mixed"});
await evaluate(`document.activeElement.blur()`);

// ---------- 6. NVINT4 preset, layout ----------
await q.click('[data-preset="qd_nvint4"]');
const nvint4 = await evaluate(`document.querySelector('[data-default="nvint4"]').textContent`);
check("NVINT4 card carries the new-format note", nvint4.includes("새 형식 · NVFP4 구조 + INT4"), nvint4);
check("1600: no page-level horizontal scroll", await evaluate(`document.documentElement.scrollWidth <= innerWidth`));
await shot(`qz_${label}_light_1600_nvint4`);
await b.setup(1000, 900, false, true);
await b.open(`${base}/quantizer.html`, READY);
const narrow = await evaluate(`(() => { const w = document.querySelector(".qz-gridwrap");
  return {page: document.documentElement.scrollWidth, viewport: innerWidth, main: document.getElementById("qz").scrollWidth,
    mainBox: document.getElementById("qz").clientWidth, grid: w.scrollWidth, gridBox: w.clientWidth}; })()`);
check("1000 px: no page-level horizontal scroll; the grid scrolls inside its panel", narrow.page <= narrow.viewport
  && narrow.main <= narrow.mainBox && narrow.grid > narrow.gridBox, narrow);
check("reduced motion: no transitions", (await evaluate(`getComputedStyle(document.querySelector(".qz-cell")).transitionDuration`)) === "0s");
await shot(`qz_${label}_light_1000`);

// ---------- 7. dark ----------
await b.setup(1600, 1000, true);
await b.open(`${base}/quantizer.html`, READY);
await q.click('[data-preset="qd_mixed"]');
await q.click('[data-options="awq_nvfp4"]');
await pause(400);  // fills animate for 0.12 s
const darkPanel = await evaluate(`[getComputedStyle(document.getElementById("qz-opt-awq_nvfp4")).backgroundColor,
  getComputedStyle(document.querySelector("#qz-opt-awq_nvfp4 .qz-select")).backgroundColor]`);
check("dark theme: the options panel and its controls use the dark tokens (sunken #0f1c29, panel #122130)",
  darkPanel[0] === "rgb(15, 28, 41)" && darkPanel[1] === "rgb(18, 33, 48)", darkPanel);
const dark = await evaluate(`[getComputedStyle(document.querySelector('.qz-cell[data-m="awq_nvfp4"]')).backgroundColor,
  getComputedStyle(document.body).backgroundColor, getComputedStyle(document.querySelector('.qz-cell[data-m="mxfp8"]')).color]`);
check("dark theme: method fills and paper switch to the dark tokens", dark[0] === "rgb(0, 115, 74)" && dark[1] === "rgb(12, 23, 34)"
  && dark[2] === "rgb(255, 255, 255)", dark);
await shot(`qz_${label}_dark_1600_mixed`);

// ---------- 8. empty, error and (synthetic) recorded states ----------
await b.setup(1600, 1000);
await b.open(`${alt}/empty/quantizer.html`, READY);
const empty = await evaluate(`[document.querySelector(".state h2")?.textContent, document.querySelector(".state pre")?.textContent]`);
check("missing presets.json: says what to run", empty[0] === "Quantizer 데이터가 아직 없습니다" && empty[1]?.startsWith("python -m app.export_quantizer_demo"), empty);
await b.open(`${alt}/bad/quantizer.html`, READY);
const bad = await evaluate(`[document.querySelector(".state h2")?.textContent, document.querySelector(".state p")?.textContent]`);
check("wrong schema: error names the expected schema", bad[0] === "Quantizer 데이터를 열지 못했습니다" && bad[1].includes("tricast.quantizer_demo/1"), bad);
await b.open(`${alt}/synth/quantizer.html`, READY);
await q.click('[data-preset="qd_mixed"]');
const synth = await evaluate(`fetch("demo/quantizer/presets.json").then((r) => r.json())`);
const sp = synth.presets.find((p) => p.id === "qd_mixed");
const panel = await evaluate(`(() => { const r = document.querySelector(".qz-record");
  return {text: r.textContent, minis: r.querySelectorAll(".qz-mini:not(i)").length, na: r.querySelectorAll(".qz-minigrid .qz-na").length,
    link: r.querySelector(".qz-maplink")?.getAttribute("href")}; })()`);
const delta = `${sp.run.ppl_delta_pct > 0 ? "+" : "−"}${Math.abs(sp.run.ppl_delta_pct).toFixed(1)}%`;
check("SYNTHETIC record: PPL, BF16, Δ, windows, GPU, commit, SHA, 196-cell SQNR map with 14 BF16 hatches, layer-map link",
  panel.text.includes(sp.run.ppl.toFixed(2)) && panel.text.includes(synth.native.ppl.toFixed(2)) && panel.text.includes(delta)
  && panel.text.includes("146개 창") && panel.text.includes("A100") && panel.text.includes(sp.run.env.git_sha.slice(0, 7))
  && panel.text.includes(sp.recipe_sha256.slice(0, 12)) && panel.minis === 196 && panel.na === 14 && panel.link === "layers.html",
  {delta, minis: panel.minis, na: panel.na});
const bench = await evaluate(`[...document.querySelectorAll(".qz-btable tbody tr")].map((tr) => [...tr.children].map((c) => c.textContent))`);
const lm = sp.lmeval;
const benchOk = bench.length === 6 && lm.tasks.every((t, i) => bench[i][1] === (t.value * 100).toFixed(1)
  && bench[i][3] === `${t.delta_pts > 0 ? "+" : t.delta_pts < 0 ? "−" : ""}${Math.abs(t.delta_pts).toFixed(1)}`
  && bench[i][4] === `±${(t.stderr * 100).toFixed(1)}` && bench[i][5] === String(t.n) && bench[i][0].includes(t.metric))
  && bench[5][0].startsWith("매크로 평균") && bench[5][1] === (lm.macro * 100).toFixed(1);
const prov = await evaluate(`document.querySelector(".qz-bench .qz-env").textContent`);
check("SYNTHETIC 벤치마크 (E2E): 5 tasks + macro row with score, BF16, Δ pt, ±stderr, n; provenance GPU · lm-eval · limit",
  benchOk && prov.includes("0.4.13") && prov.includes("limit 300") && prov.includes("A100"), {bench, prov});
await q.click('[data-preset="qd_mxfp4"]');
const benchNone = await evaluate(`document.querySelector(".qz-bench").textContent`);
check("벤치마크 without a record: 기록 없음 — 아래 명령으로 실행", benchNone.includes("기록 없음 — 아래 명령으로 실행"), benchNone);
const none = await evaluate(`document.querySelector(".qz-record").textContent`);
check("SYNTHETIC: results column never prints null/undefined", !/null|undefined|NaN/.test(await evaluate(`(() => { const c = document.querySelector(".qz-results").cloneNode(true);
  c.querySelectorAll("pre, code").forEach((x) => x.remove()); return c.textContent; })()`)));
check("SYNTHETIC bundle, preset without a record: says so and offers recorded presets", none.includes("아직 기록이 없습니다")
  && none.includes("qd_nvint4") && none.includes("qd_mixed"), none);
await q.click('[data-preset="qd_nvint4"]');
const ppOnly = await evaluate(`document.querySelector(".qz-record").textContent`);
check("SYNTHETIC record with PPL but no report: report part says 기록 없음", ppOnly.includes("31.25") && ppOnly.includes("레이어별 리포트는 아직 기록이 없습니다"));
await q.click('[data-preset="qd_mixed"]');
await shot(`qz_${label}_SYNTHETIC_record`);

// ---------- 8b. MOCKED live mode (DevTools Fetch answers /api/*; no server, no GPU) ----------
const mock = await mockLive(b, {run: sp.run, report: sp.report, lmeval: sp.lmeval});
await b.open(`${alt}/synth/quantizer.html`, READY);
await q.click('[data-preset="qd_mixed"]');
await pause(300);
const enabled = await evaluate(`[document.querySelector(".qz-gpu-run").disabled, document.getElementById("qz-gpu-note").textContent]`);
check("MOCKED live+CUDA catalog enables 서버 GPU에서 실행 and names the device", enabled[0] === false && enabled[1].includes("MOCK GPU"), enabled);
await evaluate(`document.querySelector('.qz-gpu input[data-task="arc_easy"]').click()`);
await q.click(".qz-gpu-run");
const seenStages = new Set();
for (let i = 0; i < 120; i += 1) {
  const now = await evaluate(`[...document.querySelectorAll(".qz-job .pipeline span.now")].map((x) => x.textContent).join()`);
  if (now) seenStages.add(now);
  if (await evaluate(`!!document.querySelector(".qz-job .qz-match")`)) break;
  await pause(100);
}
const jobText = await evaluate(`document.querySelector(".qz-job").textContent`);
check("MOCKED job: queued → running stages → results in the record/bench shapes → export command",
  ["대기", "모델 불러오기", "레이어별 리포트", "lm-eval"].every((x) => seenStages.has(x)) && jobText.includes("서버 실행 결과")
  && jobText.includes(sp.run.ppl.toFixed(2)) && jobText.includes("매크로 평균") && jobText.includes("python -m app.export_layers")
  && jobText.includes("MOCK GPU"), {stages: [...seenStages], polls: mock.polls(), log: mock.log, jobText: jobText.slice(0, 300)});
await shot(`qz_${label}_live_panel_MOCKED`);
await mock.stop();

// ---------- 9. errors: DevTools events and Chrome's own stderr log, with a positive control ----------
await evaluate(`setTimeout(() => { throw new Error("QZ_POSITIVE_CONTROL"); }, 0)`);
await pause(400);
const problems = b.events.filter((e) => e.sessionId === b.sessionId && (e.method === "Runtime.exceptionThrown"
  || (e.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(e.params.type))
  || (e.method === "Log.entryAdded" && e.params.entry.level === "error")));
const text = (e) => e.params.exceptionDetails?.exception?.description ?? e.params.entry?.text ?? JSON.stringify(e.params.args);
// Intended 404s: the empty root has no presets.json, and a static server has no /api (the page probes it).
const expected404 = (e) => e.method === "Log.entryAdded" && (e.params.entry.url?.endsWith("/empty/demo/quantizer/presets.json")
  || (label === "static" && e.params.entry.url?.endsWith("/api/catalog")) || /\/(empty|bad|synth)\/api\/catalog$/.test(e.params.entry.url ?? ""));
const control = problems.filter((e) => text(e).includes("QZ_POSITIVE_CONTROL"));
const unexpected = problems.filter((e) => !expected404(e) && !text(e).includes("QZ_POSITIVE_CONTROL"));
const log = b.stderr();
const uncaught = log.split("\n").filter((line) => /Uncaught|CONSOLE.*(Error|error)/.test(line));
check("positive control: a thrown error shows up in DevTools and in Chrome's --enable-logging stderr", control.length === 1
  && uncaught.some((line) => line.includes("QZ_POSITIVE_CONTROL")), {control: control.length, uncaught});
check("no other JS exceptions or console errors (DevTools + stderr; the empty root's 404 is intended)", unexpected.length === 0
  && uncaught.every((line) => line.includes("QZ_POSITIVE_CONTROL")), {unexpected: unexpected.map(text), uncaught});
writeFileSync(`${outDir}/qz_stderr_${label}.log`, log);
writeFileSync(`${outDir}/qz_results_${label}.json`, JSON.stringify(results, null, 1));
for (const item of results.checks) console.log(`${item.pass ? "PASS" : "FAIL"}  ${item.label}${item.pass ? "" : `  ${JSON.stringify(item.detail)}`}`);
console.log(`QZ_CHECKS[${label}] ${results.checks.filter((c) => c.pass).length}/${results.checks.length} passed`);
await b.close();
