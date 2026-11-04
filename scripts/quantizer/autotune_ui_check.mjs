// Checks for app/web/autotune.html in headless Chrome (Google Chrome at the macOS path in cdp_lib.mjs, Node >= 22):
// 1 solver: 9/2 and the session budget reproduce the bundle's recorded solution grid and value;
// 2 constraints: JS constrained() == app.tune_request.constrained (null pattern) for three requests on the served
//   table, and the solve within each equals app.autotune.solve_mckp and respects the request;
// 3 request box: a stored tune reply applies (badge, budget, slider minimum = constrained cheapest, chips, removing a
//   chip, 제약 해제), the other outcomes leave the grid alone, live parse, cancel, 501 / 503 / 422 messages;
// 4 agent rounds: stored badges, hover outlines exactly the changed cells, a live round appends, errors keep the
//   stored rounds;
// 5 layout and access: no page scroll at 1440 / 390, no console errors (positive control; the intended 4xx/5xx
//   network lines are listed), reduced motion, Tab reaches every focusable with a focus ring, dark theme, cell code
//   contrast >= 4.5:1 for every method family in both themes;
// 6 missing bundle: the export command.
// Prints PASS/FAIL per check and AUTOTUNE_UI_CHECKS[<label>] n/m; exits 0 only when every check passes.
//
// usage (from the repo root; python = an env that imports app.tune_request and numpy, e.g. the repo's .venv):
//   node scripts/quantizer/autotune_ui_check.mjs <out_dir> <repo> <python> [label]
// The script starts and stops its own servers: `python -m http.server` for app/web (the bundle as it is; POSTs answer
// 501) and scripts/quantizer/autotune_fixture_server.py (fake bundle extras and the two /api/autotune endpoints; its
// numbers are made up). Screenshots and autotune_ui_results_<label>.json go to <out_dir>.
import {execFileSync, spawn} from "node:child_process";
import {mkdirSync, writeFileSync} from "node:fs";
import {resolve} from "node:path";
import {pathToFileURL} from "node:url";

import {connect} from "./cdp_lib.mjs";

const [, , outArg, repoArg, pythonArg, label = "local"] = process.argv;
if (!outArg || !repoArg || !pythonArg) {
  console.error("usage: node scripts/quantizer/autotune_ui_check.mjs <out_dir> <repo> <python> [label]");
  process.exit(2);
}
const outDir = resolve(outArg);
const repo = resolve(repoArg);
// a path such as .venv/bin/python is relative to where the check is run, not to <repo> (the servers' cwd)
const python = pythonArg.includes("/") ? resolve(pythonArg) : pythonArg;
mkdirSync(outDir, {recursive: true});
const results = {label, checks: [], shots: []};
const check = (name, pass, detail) => results.checks.push({label: name, pass: Boolean(pass), detail});
const env = {...process.env, PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1", PYTHONPATH: repo};
const {parseBits} = await import(pathToFileURL(`${repo}/app/web/js/mckp.js`).href);
const READY = `!!document.querySelector(".at-cell[data-f]") || !!document.querySelector(".state")`;
const FAMILIES = ["gptq", "awq", "rht", "clip", "hqq", "nf4", "int8w", "other", "bf16"];

// Start a server and wait for the line that names its port.
function serve(args, pattern) {
  const proc = spawn(python, args, {cwd: repo, stdio: ["ignore", "pipe", "pipe"], env});
  return new Promise((resolvePort, reject) => {
    let text = "";
    const onData = (chunk) => {
      text += chunk;
      const match = pattern.exec(text);
      if (match) resolvePort({proc, port: Number(match[1])});
    };
    proc.stdout.on("data", onData);
    proc.stderr.on("data", onData);
    proc.on("error", (error) => reject(new Error(`${python} ${args.join(" ")}: ${error.message}`)));
    proc.on("exit", (code) => reject(new Error(`${args.join(" ")} exited (${code}): ${text.slice(-300)}`)));
    setTimeout(() => reject(new Error(`${args.join(" ")} did not start: ${text.slice(-300)}`)), 20000);
  });
}

const servers = [];
let b = null;
try {
  const staticServer = await serve(["-m", "http.server", "0", "--bind", "127.0.0.1", "-d", `${repo}/app/web`], /port (\d+)/);
  servers.push(staticServer.proc);
  const fixtureServer = await serve([`${repo}/scripts/quantizer/autotune_fixture_server.py`, repo], /AUTOTUNE_FIXTURE_READY (\d+)/);
  servers.push(fixtureServer.proc);
  const STATIC = `http://127.0.0.1:${staticServer.port}`;
  const FIXTURE = `http://127.0.0.1:${fixtureServer.port}`;
  const bundle = await (await fetch(`${STATIC}/demo/autotune/bundle.json`)).json();
  const fixtureBundle = await (await fetch(`${FIXTURE}/fixture/demo/autotune/bundle.json`)).json();
  const {table, session} = bundle;
  const quant = table.methods.filter((m) => m !== "bf16");
  const blockOf = (block) => ((block % table.blocks) + table.blocks) % table.blocks;
  const sameFrac = (x, y) => {
    const [p, q] = parseBits(x);
    const [r, t] = parseBits(y);
    return p * t === r * q;
  };
  const ceil128 = ([num, den]) => Number((BigInt(num) * 128n + BigInt(den) - 1n) / BigInt(den)) / 128;

  b = await connect(outDir, {logging: true});
  const {evaluate, pause, waitFor, S} = b;
  const shot = async (file) => results.shots.push(await b.shot(`${outDir}/${file}.png`));
  const state = () => evaluate("window.__autotune.state()");
  const layerMethods = (grid) => table.layers.map((layer) => grid[layer.block][table.kinds.indexOf(layer.kind)]);
  const open = async (url) => {
    const ok = await b.open(url, READY);
    await pause(300);
    return ok;
  };
  const typeRequest = (text) => evaluate(`(() => { const i = document.getElementById("at-request"); i.value = ${JSON.stringify(text)};
    document.querySelector(".at-request .at-btn.primary").click(); })()`);
  const requestError = () => evaluate(`(() => { const e = document.querySelector(".at-request .at-live-error"); return e.hidden ? "" : e.innerText; })()`);
  const proposeError = () => evaluate(`(() => { const e = document.querySelector(".at-askrow + .at-live-error"); return e.hidden ? "" : e.innerText; })()`);
  const roundBadges = () => evaluate(`[...document.querySelectorAll(".at-round")].map((r) => r.querySelector(".at-round-head .at-badge").textContent)`);
  const clearConstraints = () => evaluate(`(() => { const b = [...document.querySelectorAll(".at-constraints .at-btn")].find((x) => x.textContent === "제약 해제"); if (b) b.click(); })()`);

  // ---------- static server, light 1440: the repo bundle as it is ----------
  await b.setup(1440, 1800);
  check("page loads on the repo bundle", await open(`${STATIC}/autotune.html`));
  const initialMin = await evaluate("window.__autotune.sliderMin()");

  // 1 solver
  const budgets = [...new Set(["9/2", session?.budget_bits].filter(Boolean))];
  for (const budget of budgets) {
    const st = await evaluate(`window.__autotune.setBudget(${JSON.stringify(budget)})`);
    const recorded = (bundle.solutions ?? []).find((row) => (row.kv ?? "none") === "none" && row.budget_bits && sameFrac(row.budget_bits, budget));
    if (recorded) {
      const value = table.objective === "kl" ? recorded.predicted_kl_sum : recorded.predicted_kl_sum_minus_floor;
      check(`solver ${budget}: choice == recorded ${recorded.id} grid, value == its recorded value, raw KL == predicted_kl_sum`,
        JSON.stringify(st.grid) === JSON.stringify(recorded.grid) && st.value === value
        && Math.abs(st.rawKL - recorded.predicted_kl_sum) < 1e-12 && st.matched.includes(recorded.id),
        {value: st.value, recorded: value, rawKL: st.rawKL, predicted: recorded.predicted_kl_sum, matched: st.matched});
    } else {
      const [p, q] = parseBits(budget);
      check(`solver ${budget}: no recorded solution at this budget; solved within it`, !st.error && st.bits <= Number(p) / Number(q) + 1e-12,
        {bits: st.bits, error: st.error});
    }
  }

  // 2 constraints: three requests from the methods the session-budget solve uses most
  await evaluate(`window.__autotune.setBudget(${JSON.stringify(session?.budget_bits ?? "9/2")})`);
  const counts = Object.entries((await state()).counts).filter(([m]) => m !== "bf16").sort((x, y) => y[1] - x[1]).map(([m]) => m);
  const ranked = [...counts, ...quant.filter((m) => !counts.includes(m))];
  const fixtureTune = fixtureBundle.requests.find((item) => item.request.outcome === "tune").request;
  const requests = [
    {name: `keep [0,-1] + exclude ${ranked[0]}`, keep_bf16_blocks: [0, -1], exclude_methods: [ranked[0]], only_methods: null},
    {name: `only [${ranked[0]}, ${ranked[1]}]`, keep_bf16_blocks: [], exclude_methods: [], only_methods: [ranked[0], ranked[1]]},
    {name: "keep [-1]", keep_bf16_blocks: [-1], exclude_methods: [], only_methods: null},
    {name: "fixture tune reply", keep_bf16_blocks: fixtureTune.keep_bf16_blocks, exclude_methods: fixtureTune.exclude_methods,
      only_methods: fixtureTune.only_methods},
  ];
  const PY = `
import json, math, sys
from fractions import Fraction
from app import autotune as at
from app import tune_request as tr
table = json.load(open(sys.argv[1], encoding="utf-8"))["table"]
out = []
for req in json.loads(sys.argv[2]):
    ct = tr.constrained(table, req)
    cheapest = tr.cheapest_bits(ct)
    budget = Fraction(math.ceil((cheapest + Fraction(3, 4)) * 128), 128)
    rows, costs, values, params = [], [], [], 0
    for layer in ct["layers"]:
        opts = [j for j, v in enumerate(layer["value"]) if v is not None]
        rows.append(opts)
        costs.append([(Fraction(ct["weight_bits"][ct["methods"][j]][0]) + Fraction(ct["weight_bits"][ct["methods"][j]][1],
                       ct["weight_bits"][ct["methods"][j]][2] or layer["in_features"])) * layer["params"] for j in opts])
        values.append([layer["value"][j] for j in opts])
        params += layer["params"]
    sol = at.solve_mckp(costs, values, budget * params)
    out.append({"value_nulls": [[i, j] for i, l in enumerate(ct["layers"]) for j, v in enumerate(l["value"]) if v is None],
                "kl_nulls": [[i, j] for i, l in enumerate(ct["layers"]) for j, v in enumerate(l["kl"]) if v is None],
                "cheapest": [cheapest.numerator, cheapest.denominator], "budget": f"{budget.numerator}/{budget.denominator}",
                "methods": [ct["methods"][rows[i][k]] for i, k in enumerate(sol.choice)], "value": sol.value})
print(json.dumps(out))
`;
  const pyOut = JSON.parse(execFileSync(python, ["-c", PY, `${repo}/app/web/demo/autotune/bundle.json`, JSON.stringify(requests)],
    {cwd: repo, encoding: "utf-8", maxBuffer: 1 << 28, env}));
  for (const [i, req] of requests.slice(0, 3).entries()) {
    const ref = pyOut[i];
    const js = await evaluate(`(() => { const t = window.__autotune.constrained(${JSON.stringify(req)}); const v = [], k = [];
      t.layers.forEach((l, i) => l.value.forEach((x, j) => { if (x === null) v.push([i, j]); if (l.kl[j] === null) k.push([i, j]); }));
      return {v, k}; })()`);
    check(`constrained ${req.name}: JS null pattern == app.tune_request.constrained (value and kl)`,
      JSON.stringify(js.v) === JSON.stringify(ref.value_nulls) && JSON.stringify(js.k) === JSON.stringify(ref.kl_nulls),
      {js: [js.v.length, js.k.length], python: [ref.value_nulls.length, ref.kl_nulls.length]});
    await evaluate(`window.__autotune.showRequest({text: "check", request: {outcome: "tune", model: null, budget_bits: ${JSON.stringify(ref.budget)},
      compute_budget: null, keep_bf16_blocks: ${JSON.stringify(req.keep_bf16_blocks)}, exclude_methods: ${JSON.stringify(req.exclude_methods)},
      only_methods: ${JSON.stringify(req.only_methods)}, questions: [], reason: null, assumptions: [], checks: []}}, false)`);
    const st = await state();
    const used = layerMethods(st.grid);
    const kept = new Set(req.keep_bf16_blocks.map(blockOf));
    const allowed = new Set(req.only_methods ? [...req.only_methods, "bf16"] : table.methods);
    const respects = table.layers.every((layer, n) => (!kept.has(layer.block) || used[n] === "bf16")
      && !req.exclude_methods.includes(used[n]) && allowed.has(used[n]));
    const differ = used.filter((m, n) => m !== ref.methods[n]).length;
    const cheapest = Number(BigInt(ref.cheapest[0]) * 1000000000000n / BigInt(ref.cheapest[1])) / 1e12;
    check(`constrained ${req.name}: solve at ${ref.budget} respects it and == app.autotune.solve_mckp (all layers, value), slider minimum = cheapest`,
      !st.error && respects && differ === 0 && st.value === ref.value
      && Math.abs((await evaluate("window.__autotune.cheapest()")) - cheapest) < 2e-9
      && (await evaluate("window.__autotune.sliderMin()")) === ceil128(ref.cheapest),
      {error: st.error, respects, differ, value: [st.value, ref.value], cheapest});
    await clearConstraints();
  }
  check("제약 해제 returns to the plain table and its slider minimum", (await evaluate("window.__autotune.constraints()")) === null
    && (await evaluate("window.__autotune.sliderMin()")) === initialMin, {min: await evaluate("window.__autotune.sliderMin()"), initialMin});

  // 4 stored rounds and hover (at the session budget)
  if (session) {
    await evaluate(`window.__autotune.setBudget(${JSON.stringify(session.budget_bits)})`);
    const badges = await roundBadges();
    const date = (session.rounds.find((r) => r.llm?.started_utc)?.llm.started_utc ?? "").slice(0, 10);
    check("stored rounds render, each with the 저장된 응답 badge, and the session header names its date", badges.length === session.rounds.length
      && badges.every((t) => t === "저장된 응답")
      && (await evaluate(`document.querySelector(".at-agent-head .at-badge").textContent`)) === `저장된 응답 · ${date} 세션`, {badges, date});
    const withChanges = session.rounds.flatMap((r) => r.proposals).filter((p) => (p.changes ?? []).length);
    const hovers = [];
    for (const p of withChanges) {
      const got = await evaluate(`(() => { const card = [...document.querySelectorAll(".at-card")].find((c) => c.querySelector(".at-card-name").textContent === ${JSON.stringify(p.name)});
        card.dispatchEvent(new MouseEvent("mouseenter")); const hl = [...document.querySelectorAll(".at-cell.hl")].map((c) => c.dataset.b + ":" + c.dataset.k).sort();
        card.dispatchEvent(new MouseEvent("mouseleave")); return {hl, after: document.querySelectorAll(".at-cell.hl").length}; })()`);
      const want = p.changes.map((c) => `${c.block}:${table.kinds.indexOf(c.kind)}`).sort();
      hovers.push({name: p.name, ok: JSON.stringify(got.hl) === JSON.stringify(want) && got.after === 0, got: got.hl, want});
    }
    check(`hover outlines exactly the changed cells (${withChanges.length} proposals with changes), and leaving clears them`,
      withChanges.length > 0 && hovers.every((x) => x.ok), hovers.filter((x) => !x.ok));
    const withGrid = withChanges.find((p) => Array.isArray(p.grid));
    if (withGrid) {
      await evaluate(`window.__autotune.setBudget("9/2")`);
      const shown = await evaluate(`(() => { const card = [...document.querySelectorAll(".at-card")].find((c) => c.querySelector(".at-card-name").textContent === ${JSON.stringify(withGrid.name)});
        card.dispatchEvent(new MouseEvent("mouseenter")); const cells = [...document.querySelectorAll(".at-cell")];
        const grid = []; for (const c of cells) (grid[c.dataset.b] ??= [])[c.dataset.k] = c.dataset.m;
        const note = document.getElementById("at-gridnote").textContent; card.dispatchEvent(new MouseEvent("mouseleave")); return {grid, note}; })()`);
      check("hover at another budget shows the proposal's own grid and says so", JSON.stringify(shown.grid) === JSON.stringify(withGrid.grid)
        && shown.note.includes(withGrid.name), shown.note);
      await evaluate(`window.__autotune.setBudget(${JSON.stringify(session.budget_bits)})`);
    }
    // propose against the static server (POST 501): message, stored rounds kept
    await evaluate(`document.querySelector(".at-askrow .at-btn.primary").click()`);
    await waitFor(`!document.querySelector(".at-askrow + .at-live-error").hidden`, 8000);
    const message = await proposeError();
    check("propose on a server without the API (HTTP 501): says so and how to start one; stored rounds kept",
      message.includes("HTTP 501") && message.includes("--agent") && (await roundBadges()).length === session.rounds.length, message);
  }
  // request box against the static server (POST 501)
  await typeRequest("check 501");
  await waitFor(`!document.querySelector(".at-request .at-live-error").hidden`, 8000);
  const parse501 = await requestError();
  check("request on a server without the API (HTTP 501): says so and how to start one", parse501.includes("HTTP 501") && parse501.includes("--agent"), parse501);

  // 5 layout, focus, contrast, themes, motion
  const scroll = () => evaluate("({w: document.documentElement.scrollWidth, c: document.documentElement.clientWidth})");
  const wide = await scroll();
  check("1440: no horizontal page scroll", wide.w <= wide.c, wide);
  await shot(`autotune_${label}_light_1440`);
  await evaluate(`window.__autotune.setBudget(${JSON.stringify(session?.budget_bits ?? "9/2")})`);
  await open(`${STATIC}/autotune.html`);
  const listed = await evaluate(`(() => { const sel = 'a[href], button, input, select, textarea, [tabindex]';
    const list = [...document.querySelectorAll(sel)].filter((el) => el.tabIndex >= 0 && !el.disabled && el.getClientRects().length > 0
      && !el.closest("[hidden]") && getComputedStyle(el).visibility !== "hidden");
    list.forEach((el, i) => { el.dataset.qaFocus = String(i); }); document.activeElement?.blur(); return list.length; })()`);
  const reached = new Map();
  const unlisted = [];
  for (let i = 0; i < listed + 8; i += 1) {
    for (const type of ["rawKeyDown", "keyUp"]) await S("Input.dispatchKeyEvent", {type, key: "Tab", code: "Tab", windowsVirtualKeyCode: 9});
    const focus = await evaluate(`(() => { const a = document.activeElement; if (!a || a === document.body) return null; const cs = getComputedStyle(a);
      return {id: a.dataset.qaFocus ?? null, tag: a.tagName, ring: (cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0) || (cs.boxShadow && cs.boxShadow !== "none")}; })()`);
    if (!focus) continue;
    if (focus.id === null) unlisted.push(focus);
    else if (reached.has(focus.id)) break;
    else reached.set(focus.id, focus);
  }
  const noRing = [...reached.values(), ...unlisted].filter((f) => !f.ring);
  check(`Tab reaches every focusable element (${listed}) and each shows a focus ring`, reached.size === listed && noRing.length === 0,
    {reached: reached.size, listed, noRing: noRing.slice(0, 5), unlisted: unlisted.length});
  const contrast = async () => evaluate(`(() => {
    const lum = (rgb) => { const [r, g, b] = rgb.match(/\\d+(\\.\\d+)?/g).slice(0, 3).map(Number).map((v) => v / 255)
      .map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
    const host = document.querySelector(".at-grid"); const out = {};
    for (const f of ${JSON.stringify(FAMILIES)}) { const el = document.createElement("span"); el.className = "at-cell"; el.dataset.f = f; host.append(el);
      const cs = getComputedStyle(el); const [a, b] = [lum(cs.backgroundColor), lum(cs.color)];
      out[f] = Math.round(((Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)) * 100) / 100; el.remove(); }
    return out; })()`);
  const lightRatios = await contrast();
  // the stagger (light): drop the focus the Tab walk left (a focused card holds the hover, which repaints without fading)
  await evaluate("document.activeElement?.blur(); window.__autotune.hover(null)");
  await evaluate(`window.__autotune.setBudget("9/2")`);
  await pause(500);
  const stagger = await evaluate(`(() => { window.__autotune.setBudget(${JSON.stringify(session?.budget_bits ?? "5")}); const moved = [...document.querySelectorAll(".at-cell.moved")];
    return {moved: moved.length, last: Math.max(0, ...moved.map((c) => parseFloat(c.style.transitionDelay) || 0)),
      duration: moved.length ? parseFloat(getComputedStyle(moved[0]).transitionDuration) * 1000 : 0}; })()`);
  check("budget change: only changed cells fade, the last one ends within 400 ms", stagger.moved > 0 && stagger.last + stagger.duration <= 400, stagger);
  await b.setup(1440, 900, false, true);
  await open(`${STATIC}/autotune.html`);
  const reduced = await evaluate(`(() => { window.__autotune.setBudget("9/2"); const c = document.querySelector(".at-cell.moved");
    return c ? getComputedStyle(c).transitionDuration : "no changed cell"; })()`);
  check("prefers-reduced-motion: changed cells have no transition (0s)", reduced === "0s", reduced);
  await b.setup(1440, 1800, true);
  await open(`${STATIC}/autotune.html`);
  const darkBg = await evaluate("getComputedStyle(document.body).backgroundColor");
  const darkRatios = await contrast();
  await shot(`autotune_${label}_dark_1440`);
  check("dark theme renders (body on the dark paper token)", darkBg === "rgb(12, 23, 34)", darkBg);
  check("cell code contrast >= 4.5:1 for every method family, light and dark (computed styles)",
    Object.values(lightRatios).every((r) => r >= 4.5) && Object.values(darkRatios).every((r) => r >= 4.5), {light: lightRatios, dark: darkRatios});
  await b.setup(390, 1600);
  await open(`${STATIC}/autotune.html`);
  const narrow = await scroll();
  check("390: no horizontal page scroll", narrow.w <= narrow.c, narrow);
  await shot(`autotune_${label}_light_390`);

  // ---------- fixture server: stored replies, live parse and propose, cancel, 422 ----------
  await b.setup(1440, 1800);
  check("fixture page loads (fake requests + profiles)", await open(`${FIXTURE}/fixture/autotune.html`));
  const fixtureMin = await evaluate("window.__autotune.sliderMin()");
  const examples = fixtureBundle.requests;
  const tuneIndex = examples.findIndex((item) => item.request.outcome === "tune");
  const tune = examples[tuneIndex];
  await evaluate(`document.querySelectorAll(".at-example")[${tuneIndex}].click()`);
  await pause(600);
  const applied = await evaluate(`({badge: document.querySelector(".at-req-result .at-badge").textContent, text: window.__autotune.state().text,
    constraints: window.__autotune.constraints(), chips: document.querySelectorAll(".at-chip").length, min: window.__autotune.sliderMin(),
    grid: window.__autotune.state().grid, error: window.__autotune.state().error})`);
  const fixtureRef = pyOut[3];
  const keptOk = fixtureTune.keep_bf16_blocks.map(blockOf).every((blk) => applied.grid[blk].every((m) => m === "bf16"))
    && !applied.grid.flat().some((m) => fixtureTune.exclude_methods.includes(m));
  const chipCount = fixtureTune.keep_bf16_blocks.length + fixtureTune.exclude_methods.length + (fixtureTune.only_methods ? 1 : 0);
  check("stored tune reply applies: 저장된 응답 badge, its budget, its constraints in the grid, one chip each, slider minimum = constrained cheapest",
    applied.badge === `저장된 응답 · ${tune.recorded_utc.slice(0, 10)}` && applied.text === tune.request.budget_bits && !applied.error && keptOk
    && applied.chips === chipCount && applied.min === ceil128(fixtureRef.cheapest), {...applied, grid: undefined, want: ceil128(fixtureRef.cheapest)});
  await shot(`autotune_${label}_request_applied_1440`);
  await evaluate(`document.querySelector(".at-chip .at-chip-x").click()`);
  await pause(400);
  const removed = await evaluate(`({chips: document.querySelectorAll(".at-chip").length, constraints: window.__autotune.constraints()})`);
  check("removing a chip drops that one constraint", removed.chips === chipCount - 1 && removed.constraints
    && removed.constraints.keep_bf16_blocks.length + removed.constraints.exclude_methods.length + (removed.constraints.only_methods ? 1 : 0) === chipCount - 1, removed);
  await clearConstraints();
  await pause(300);
  check("제약 해제 after a stored reply: no constraints, chips hidden, slider minimum back", (await evaluate("window.__autotune.constraints()")) === null
    && (await evaluate(`document.querySelector(".at-constraints").hidden`)) && (await evaluate("window.__autotune.sliderMin()")) === fixtureMin);
  for (const outcome of ["needs_clarification", "out_of_scope", "conflict"]) {
    const index = examples.findIndex((item) => item.request.outcome === outcome);
    const before = JSON.stringify((await state()).grid);
    await evaluate(`document.querySelectorAll(".at-example")[${index}].click()`);
    await pause(300);
    const text = await evaluate(`document.querySelector(".at-req-result").innerText`);
    const expect = outcome === "needs_clarification" ? examples[index].request.questions[0] : examples[index].request.reason;
    check(`stored ${outcome}: the grid is unchanged and the result shows ${outcome === "needs_clarification" ? "the questions" : "the reason"}`,
      JSON.stringify((await state()).grid) === before && text.includes(expect) && (await evaluate("window.__autotune.constraints()")) === null, text.slice(0, 160));
  }
  await typeRequest("fixture 422");
  await waitFor(`!document.querySelector(".at-request .at-live-error").hidden`, 8000);
  const refused = await requestError();
  check("request refused (HTTP 422): the server's message", refused.includes("fixture 422"), refused);
  await typeRequest("fixture live request");
  await pause(500);
  const waiting = await evaluate(`({wait: document.querySelector(".at-request .at-wait").textContent, cancel: !document.querySelectorAll(".at-request .at-req-row .at-btn")[1].hidden,
    disabled: document.querySelector(".at-request .at-btn.primary").disabled})`);
  await waitFor(`document.querySelector(".at-req-result .at-badge")?.textContent === "실시간 Claude 응답"`, 8000);
  const live = await evaluate(`({badge: document.querySelector(".at-req-result .at-badge").textContent, constraints: window.__autotune.constraints(), error: window.__autotune.state().error,
    grid: window.__autotune.state().grid})`);
  const liveUsed = new Set(live.grid.flat());
  check("live parse: elapsed seconds and 취소 while waiting, then 실시간 Claude 응답 with its constraints applied",
    /· \d+초/.test(waiting.wait) && waiting.cancel && waiting.disabled && live.badge === "실시간 Claude 응답" && !live.error
    && JSON.stringify(live.constraints?.only_methods) === JSON.stringify(quant.slice(0, 2)) && live.grid[table.blocks - 1].every((m) => m === "bf16")
    && [...liveUsed].every((m) => m === "bf16" || quant.slice(0, 2).includes(m)), {waiting, badge: live.badge, constraints: live.constraints, used: [...liveUsed]});
  await clearConstraints();
  await typeRequest("slow fixture request");
  await pause(600);
  await evaluate(`[...document.querySelectorAll(".at-request .at-btn")].find((x) => x.textContent === "취소").click()`);
  await pause(300);
  const cancelled = await evaluate(`({error: document.querySelector(".at-request .at-live-error").innerText, enabled: !document.querySelector(".at-request .at-btn.primary").disabled,
    wait: document.querySelector(".at-request .at-wait").textContent})`);
  check("cancel stops the wait and says nothing changed", cancelled.error === "요청 해석을 취소했습니다. 화면은 그대로입니다." && cancelled.enabled && cancelled.wait === "", cancelled);
  if (session) {
    const storedCount = (await roundBadges()).length;
    await evaluate(`document.querySelector(".at-askrow .at-btn.primary").click()`);
    await waitFor(`document.querySelectorAll(".at-round").length === ${storedCount + 1}`, 8000);
    const badges = await roundBadges();
    check("live propose appends a round with the 실시간 Claude 응답 badge; stored rounds keep theirs", badges.length === storedCount + 1
      && badges.at(-1) === "실시간 Claude 응답" && badges.slice(0, -1).every((t) => t === "저장된 응답"), badges);
  }
  const tip = await evaluate(`(() => { const c = document.querySelector('.at-cell[data-b="3"][data-k="1"]'); c.dispatchEvent(new MouseEvent("mouseover", {bubbles: true}));
    return {text: document.getElementById("tip").innerText, bars: document.querySelectorAll("#tip svg.at-spark rect").length}; })()`);
  check("profile in the cell tooltip (fake profile): kurtosis, 3σ, channel ratio, outlier channels, pruning errors, 24-bar sparkline",
    ["가중치 첨도", "3σ 밖 가중치", "입력 채널 absmax 비", "outlier 입력 채널", "50% 가지치기 오차", "2:4 가지치기 오차"].every((k) => tip.text.includes(k)) && tip.bars === 24, tip);

  // ---------- fixture server, /down/: 503 for both live calls ----------
  await open(`${FIXTURE}/down/autotune.html`);
  await typeRequest("fixture request while the agent is down");
  await waitFor(`!document.querySelector(".at-request .at-live-error").hidden`, 8000);
  const down = await requestError();
  check("request with no agent (HTTP 503): the server's message and a pointer to the stored examples", down.includes("fixture 503") && down.includes("저장된 예시"), down);
  if (session) {
    await evaluate(`document.querySelector(".at-askrow .at-btn.primary").click()`);
    await waitFor(`!document.querySelector(".at-askrow + .at-live-error").hidden`, 8000);
    const message = await proposeError();
    check("propose with no agent (HTTP 503): the server's message; stored rounds kept", message.includes("fixture 503")
      && (await roundBadges()).length === session.rounds.length, message);
  }

  // ---------- 6 missing bundle ----------
  await open(`${FIXTURE}/missing/autotune.html`);
  const missing = await evaluate(`document.querySelector(".state")?.innerText ?? ""`);
  check("missing bundle: says so and shows the export command with a copy button", missing.includes("python -m app.export_autotune_demo")
    && (await evaluate(`!!document.querySelector(".state .copy")`)), missing.slice(0, 160));
  await shot(`autotune_${label}_missing`);

  // ---------- console: DevTools events and Chrome's stderr, with a positive control ----------
  await evaluate(`setTimeout(() => { throw new Error("AT_POSITIVE_CONTROL"); }, 0)`);
  await pause(400);
  const problems = b.events.filter((e) => e.sessionId === b.sessionId && (e.method === "Runtime.exceptionThrown"
    || (e.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(e.params.type))
    || (e.method === "Log.entryAdded" && e.params.entry.level === "error")));
  const text = (e) => e.params.exceptionDetails?.exception?.description ?? `${e.params.entry?.text ?? JSON.stringify(e.params.args)} ${e.params.entry?.url ?? ""}`;
  // intended: POSTs to the static server (501), the fixture's 422 / 503, the missing bundle (404)
  const intended = (e) => e.method === "Log.entryAdded" && (new RegExp(`^${STATIC}/api/autotune/(parse|propose)$`).test(e.params.entry.url ?? "")
    || /\/fixture\/api\/autotune\/parse$|\/down\/api\/autotune\/(parse|propose)$|\/missing\/demo\/autotune\/bundle\.json$/.test(e.params.entry.url ?? ""));
  const control = problems.filter((e) => text(e).includes("AT_POSITIVE_CONTROL"));
  const unexpected = problems.filter((e) => !intended(e) && !text(e).includes("AT_POSITIVE_CONTROL"));
  const uncaught = b.stderr().split("\n").filter((line) => /Uncaught|CONSOLE.*(Error|error)/.test(line));
  check("positive control: a thrown error shows up in DevTools and in Chrome's stderr log", control.length === 1
    && uncaught.some((line) => line.includes("AT_POSITIVE_CONTROL")), {control: control.length});
  check("no other JS exceptions or console errors (intended network lines listed in the detail)", unexpected.length === 0
    && uncaught.every((line) => line.includes("AT_POSITIVE_CONTROL")), {unexpected: unexpected.map(text), intended: problems.filter(intended).map(text)});
  results.intended = problems.filter(intended).map(text);
} catch (error) {
  check("the check script ran to the end", false, String(error?.stack ?? error).slice(0, 600));
} finally {
  if (b) await Promise.race([b.close(), new Promise((done) => setTimeout(done, 4000))]);
  for (const proc of servers) proc.kill("SIGTERM");
}
writeFileSync(`${outDir}/autotune_ui_results_${label}.json`, JSON.stringify(results, null, 1));
for (const item of results.checks) console.log(`${item.pass ? "PASS" : "FAIL"}  ${item.label}${item.pass ? "" : `  ${JSON.stringify(item.detail)}`}`);
for (const line of results.intended ?? []) console.log(`NOTE  intended network error: ${line.trim()}`);
const passed = results.checks.filter((c) => c.pass).length;
console.log(`AUTOTUNE_UI_CHECKS[${label}] ${passed}/${results.checks.length}`);
process.exit(passed === results.checks.length && results.checks.length > 0 ? 0 : 1);
