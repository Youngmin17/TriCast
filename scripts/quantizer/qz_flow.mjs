// Shared helpers for the Quantizer CDP scripts: real mouse/keyboard input and the click flow that builds preset
// qd_mixed from "새 레시피" (used by qz_check.mjs and quantizer_deck.mjs).
export const READY = `!!document.querySelector(".qz-cell") || !!document.querySelector(".state")`;

export function helpers(b) {
  const {S, evaluate, pause} = b;
  const center = (selector) => evaluate(`(() => { const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return null; el.scrollIntoView({block: "nearest", inline: "nearest"});
    const r = el.getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()`);
  async function mouse(type, x, y, {modifiers = 0, buttons = 0} = {}) {
    await S("Input.dispatchMouseEvent", {type, x, y, modifiers, buttons, clickCount: type === "mouseMoved" ? 0 : 1,
      button: type === "mouseMoved" && !buttons ? "none" : "left"});
  }
  async function click(selector, modifiers = 0) {
    const at = await center(selector);
    if (!at) throw new Error(`no element ${selector}`);
    await mouse("mouseMoved", at.x, at.y, {modifiers});
    await mouse("mousePressed", at.x, at.y, {modifiers, buttons: 1});
    await mouse("mouseReleased", at.x, at.y, {modifiers});
    await pause(60);
  }
  async function drag(from, to) {
    const a = await center(from);
    const z = await center(to);
    await mouse("mouseMoved", a.x, a.y);
    await mouse("mousePressed", a.x, a.y, {buttons: 1});
    for (let i = 1; i <= 4; i += 1) {
      await mouse("mouseMoved", a.x + ((z.x - a.x) * i) / 4, a.y + ((z.y - a.y) * i) / 4, {buttons: 1});
    }
    await mouse("mouseReleased", z.x, z.y);
    await pause(60);
  }
  const KEYS = {ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, Escape: 27, " ": 32};
  async function key(name, modifiers = 0) {
    const digit = /^[0-9]$/.test(name);
    const letter = /^[a-z]$/.test(name);
    const code = digit ? `Digit${name}` : letter ? `Key${name.toUpperCase()}` : name === " " ? "Space" : name;
    const keyCode = KEYS[name] ?? name.toUpperCase().charCodeAt(0);
    for (const type of ["rawKeyDown", "keyUp"]) {
      await S("Input.dispatchKeyEvent", {type, key: name, code, windowsVirtualKeyCode: keyCode, modifiers});
    }
    await pause(40);
  }
  const state = () => evaluate("window.__quantizer.getState()");
  const cell = (b, k) => `.qz-cell[data-b="${b}"][data-k="${k}"]`;
  return {center, mouse, click, drag, key, state, cell};
}

// The qd_mixed flow, one meaningful step at a time.
export const MIXED_FLOW = [
  ["새 레시피", (q) => q.click('[data-preset="new"]')],
  ["기본 방법 AWQ + NVFP4", (q) => q.click('[data-default="awq_nvfp4"]')],
  ["모든 o_proj 선택", (q) => q.click('[data-quick="모든 o_proj"]')],
  ["MXFP8 적용", (q) => q.click('[data-apply="mxfp8"]')],
  ["모든 down_proj 선택", (q) => q.click('[data-quick="모든 down_proj"]')],
  ["GPTQ INT4 적용", (q) => q.click('[data-apply="gptq_int4"]')],
  ["첫·마지막 블록 BF16", (q) => q.click('[data-quick="첫·마지막 블록 BF16"]')],
  ["KV 캐시 KIVI-2", (q) => q.click('[data-kv="kivi2"]')],
];

// MOCKED live mode for screenshots and checks: answers the page's /api/catalog and /api/quantizer/jobs through
// the DevTools Fetch domain (no server, no GPU). `done` is the finished job's "results"; GPU and commit carry
// "MOCK" so a capture can never pass for a real run. Call before navigating; stop() when finished.
export async function mockLive(b, done) {
  await b.S("Fetch.enable", {patterns: [{urlPattern: "*/api/*"}]});
  const id = "mock0000000000000000000000000000";
  const env = {gpu_names: ["MOCK GPU (CDP 모의 응답)"], git_sha: "mock000000", git_dirty: null};
  const stages = [["queued", null, 0], ["running", "모델 불러오기", 0.02], ["running", "레이어별 리포트", 0.1],
    ["running", "lm-eval", 0.7]];
  const results = Object.fromEntries(Object.entries(done).map(([key, value]) => [key,
    value && {...value, env: {...value.env, ...env}, source: "MOCK"}]));
  let seen = 0;
  let polls = 0;
  let running = true;
  const log = [];
  const reply = (requestId, status, body) => b.S("Fetch.fulfillRequest", {requestId, responseCode: status,
    responseHeaders: [{name: "content-type", value: "application/json"}],
    body: Buffer.from(JSON.stringify(body)).toString("base64")});
  const job = () => {
    const [status, stage, progress] = stages[Math.min(polls - 1, stages.length - 1)];
    const finished = polls > stages.length;
    return {id, status: finished ? "done" : status, stage: finished ? "정리" : stage, progress: finished ? 1 : progress,
      request: {kind: "quantizer"}, env: finished ? env : null, error: null,
      evidence: finished ? {layer_map_command: "python -m app.export_layers runs/studio/quantizer/MOCK/report --out app/web/demo/layers"} : null,
      results: finished ? results : null};
  };
  const loop = (async () => {
    while (running) {
      const events = b.events.slice(seen);
      seen = b.events.length;
      for (const event of events) {
        if (event.method !== "Fetch.requestPaused") continue;
        const {requestId, request} = event.params;
        if (request.url.endsWith("/api/catalog")) await reply(requestId, 200, {mode: "live", device: {kind: "cuda", name: env.gpu_names[0]}});
        else if (request.method === "POST") await reply(requestId, 202, {id, status: "queued"});
        else {
          polls += 1;
          const answer = job();
          log.push(`${answer.status}:${answer.stage}`);
          await reply(requestId, 200, answer);
        }
      }
      await b.pause(20);
    }
  })();
  return {polls: () => polls, log, async stop() {
    running = false;
    await loop;
    await b.S("Fetch.disable");
  }};
}
