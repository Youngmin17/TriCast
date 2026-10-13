// Minimal DevTools-protocol client for headless Chrome (no dependencies; Node >= 22 has WebSocket).
import {spawn} from "node:child_process";
import {mkdirSync, rmSync, writeFileSync} from "node:fs";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const CODES = {ArrowRight: 39, ArrowLeft: 37, ArrowUp: 38, ArrowDown: 40, Home: 36, End: 35, Escape: 27, " ": 32};

// logging: also start Chrome with --enable-logging=stderr so page console lines and uncaught errors reach
// stderr, readable with b.stderr() (an independent channel from the DevTools events).
export async function connect(profileRoot, {logging = false} = {}) {
  const port = 9300 + Math.floor(Math.random() * 400);
  const profile = `${profileRoot}/cdp_profile_${port}`;
  mkdirSync(profile, {recursive: true});
  const proc = spawn(CHROME, ["--headless=new", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`, ...(logging ? ["--enable-logging=stderr", "--v=0"] : []), "about:blank"],
  {stdio: ["ignore", "ignore", "pipe"]});
  let buffer = "";
  const wsUrl = await new Promise((resolve, reject) => {
    proc.stderr.on("data", (chunk) => {
      buffer += chunk;
      const match = buffer.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) resolve(match[1]);
    });
    setTimeout(() => reject(new Error("Chrome did not open DevTools")), 20000);
  });
  const ws = new WebSocket(wsUrl);
  await new Promise((resolve) => ws.addEventListener("open", resolve));
  let seq = 0;
  const pending = new Map();
  const events = [];
  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const {resolve, reject} = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error)));
      else resolve(msg.result);
    } else events.push(msg);
  });
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params, sessionId}));
  });
  const {targetId} = await send("Target.createTarget", {url: "about:blank"});
  const {sessionId} = await send("Target.attachToTarget", {targetId, flatten: true});
  const S = (method, params) => send(method, params, sessionId);
  await S("Runtime.enable");
  await S("Log.enable");
  await S("Page.enable");

  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  async function evaluate(expression) {
    const result = await S("Runtime.evaluate", {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw new Error(`${expression.slice(0, 80)}: ${result.exceptionDetails.exception?.description}`);
    return result.result.value;
  }
  async function waitFor(expression, ms = 15000) {
    const start = Date.now();
    while (Date.now() - start < ms) {
      if (await evaluate(expression).catch(() => false)) return true;
      await pause(100);
    }
    return false;
  }
  return {
    S, evaluate, waitFor, pause, events, sessionId, stderr: () => buffer,
    async shot(path, clip) {
      const {data} = await S("Page.captureScreenshot", clip ? {format: "png", clip: {...clip, scale: 1}} : {format: "png"});
      writeFileSync(path, Buffer.from(data, "base64"));
      return path;
    },
    async setup(width, height, dark = false, reduce = false) {
      await S("Emulation.setDeviceMetricsOverride", {width, height, deviceScaleFactor: 1, mobile: false});
      await S("Emulation.setEmulatedMedia", {features: [{name: "prefers-color-scheme", value: dark ? "dark" : "light"},
        {name: "prefers-reduced-motion", value: reduce ? "reduce" : "no-preference"}]});
    },
    async open(url, ready = `document.querySelectorAll("rect.cell").length > 0 || !!document.querySelector(".state")`) {
      await S("Page.navigate", {url});
      const ok = await waitFor(`document.readyState === "complete" && (${ready})`);
      await waitFor("document.fonts.status === 'loaded'", 5000);
      await pause(200);
      return ok;
    },
    async key(name, shift = false) {
      for (const type of ["rawKeyDown", "keyUp"]) {
        await S("Input.dispatchKeyEvent", {type, key: name, code: name, windowsVirtualKeyCode: CODES[name], modifiers: shift ? 8 : 0});
      }
      await pause(50);
    },
    async mouse(type, x, y) {
      await S("Input.dispatchMouseEvent", {type, x, y, button: type === "mouseMoved" ? "none" : "left",
        clickCount: type === "mouseMoved" ? 0 : 1});
      await pause(50);
    },
    async close() {
      await send("Browser.close").catch(() => {});
      proc.kill("SIGTERM");
      await pause(300);
      rmSync(profile, {recursive: true, force: true});
    },
  };
}
