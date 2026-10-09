// Browser-GPU operator lab: run the designed CoFDA arithmetic on an FP8 GEMM with WebGPU, measure its error
// against exact FP64 accumulation, and check the GPU bits against the exact JS reference (and the server's
// TriCast bits when the operands came from a real model layer).
import {mmaSummary} from "./glyph.js";
import {fmt, h, s} from "./ui.js";

export const SIZES = [
  {id: "s", label: "64 × 64 × 256", M: 64, N: 64, K: 256},
  {id: "m", label: "128 × 256 × 1024", M: 128, N: 256, K: 1024},
  {id: "l", label: "256 × 1024 × 1024", M: 256, N: 1024, K: 1024},
];

// Finite FP8 E4M3 (fn) values, used to round random Gaussian samples to the nearest code (ties to even).
let table = null;
function e4m3Table(decode) {
  if (table) return table;
  const items = [];
  for (let code = 0; code < 256; code += 1) {
    const value = decode(code);
    if (Number.isFinite(value)) items.push({code, value});
  }
  items.sort((a, b) => a.value - b.value || a.code - b.code);
  table = items;
  return table;
}

function nearestCode(value, decode) {
  const items = e4m3Table(decode);
  let lo = 0;
  let hi = items.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (items[mid].value <= value) lo = mid;
    else hi = mid;
  }
  const a = items[lo];
  const b = items[hi];
  if (value <= a.value) return a.code;
  if (value >= b.value) return b.code;
  const da = value - a.value;
  const db = b.value - value;
  if (da !== db) return da < db ? a.code : b.code;
  return (a.code & 1) === 0 ? a.code : b.code;
}

function gaussianCodes(count, seed, decode) {
  let state = seed >>> 0;
  const rand = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return (state + 0.5) / 2 ** 32;
  };
  const codes = new Uint8Array(count);
  for (let i = 0; i < count; i += 1) {
    const g = Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand());
    codes[i] = nearestCode(g * 64, decode);
  }
  return codes;
}

async function operands(source, size, decode) {
  if (source.id === "random") {
    return {M: size.M, N: size.N, K: size.K, a: gaussianCodes(size.M * size.K, 42, decode),
      b: gaussianCodes(size.N * size.K, 7, decode), scaleA: 1, scaleB: 1, server: null};
  }
  const meta = source.pack;
  const fetchBytes = async (name) => new Uint8Array(await (await fetch(`demo/webgpu/operands/${name}`)).arrayBuffer());
  const [a, b] = await Promise.all([fetchBytes(meta.a_file), fetchBytes(meta.b_file)]);
  return {M: meta.M, N: meta.N, K: meta.K, a, b, scaleA: meta.scale_a, scaleB: meta.scale_b, server: meta.server};
}

// Correct significand bits of each FP32 output against the exact result: 24 − log2(ULP + 1), so 0 ULP is 24 bits,
// 1 ULP is 23, 2^16 ULP is about 8, and results on the wrong side of zero count as 0. Plus the normwise error.
function precisionStats(bits, exact, ulpDistance) {
  const n = bits.length;
  const got = new Float32Array(bits.buffer.slice(0));
  const want = new Float32Array(exact.buffer.slice(0));
  const correct = new Float32Array(n);
  let exactCount = 0;
  let diff = 0;
  let norm = 0;
  for (let i = 0; i < n; i += 1) {
    const u = ulpDistance(bits[i], exact[i]);
    correct[i] = Math.max(0, 24 - Math.log2(u + 1));
    if (u === 0) exactCount += 1;
    if (Number.isFinite(got[i]) && Number.isFinite(want[i])) {
      diff += (got[i] - want[i]) ** 2;
      norm += want[i] ** 2;
    }
  }
  const sorted = Float32Array.from(correct).sort();
  const at = (q) => sorted[Math.min(n - 1, Math.floor(n * q))];
  return {correct, median: at(0.5), low: at(0.01), exact: exactCount / n, rel: norm ? Math.sqrt(diff / norm) : 0};
}

// Times one GPU call: three warmups, then the median of five.
async function timed(call) {
  for (let i = 0; i < 3; i += 1) await call();
  const times = [];
  let last = null;
  for (let i = 0; i < 5; i += 1) {
    last = await call();
    times.push(last.ms);
  }
  times.sort((x, y) => x - y);
  return {...last, ms: times[2]};
}

function referenceInWorker(args) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./refworker.js", import.meta.url), {type: "module"});
    worker.onmessage = ({data}) => {
      worker.terminate();
      if (data.error) reject(new Error(data.error));
      else resolve(data);
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message || "레퍼런스 계산 작업을 시작하지 못했습니다"));
    };
    worker.postMessage(args);
  });
}

export function adapterName(probe) {
  const adapter = probe?.adapter;
  return adapter?.description || [adapter?.vendor, adapter?.architecture].filter(Boolean).join(" · ") || "WebGPU";
}

export async function runLab(request, source, size, onStage) {
  const engineModule = await import("./webgpu/engine.js");
  const reference = await import("./webgpu/reference.js");
  onStage("operands");
  const ops = await operands(source, size, reference.decodeE4M3);
  const args = {a: ops.a, b: ops.b, M: ops.M, N: ops.N, K: ops.K, scaleA: ops.scaleA, scaleB: ops.scaleB,
    mma: request.mma};
  const exactRun = referenceInWorker(args);
  onStage("engine");
  const engine = await engineModule.createEngine();
  try {
    onStage("emulated");
    const emulated = await timed(() => engine.gemm(args));
    onStage("native");
    const native = await timed(() => engine.gemmF32(args));
    onStage("reference");
    const {cofda, exact, ms: referenceMs} = await exactRun;
    let same = 0;
    for (let i = 0; i < cofda.length; i += 1) if (cofda[i] === emulated.bits[i]) same += 1;
    let server = null;
    if (ops.server && ops.server[request.presetKey]) {
      const expected = new Uint32Array(await (await fetch(`demo/webgpu/operands/${ops.server[request.presetKey]}`)).arrayBuffer());
      let equal = 0;
      for (let i = 0; i < expected.length; i += 1) if (expected[i] === emulated.bits[i]) equal += 1;
      server = {equal, total: expected.length};
    }
    return {
      status: "done", request, M: ops.M, N: ops.N, K: ops.K, source: source.label,
      emulated: precisionStats(emulated.bits, exact, reference.ulpDistance),
      native: precisionStats(native.bits, exact, reference.ulpDistance),
      parity: {equal: same, total: cofda.length}, server,
      timing: {emulated_ms: emulated.ms, native_ms: native.ms, reference_ms: referenceMs},
    };
  } finally {
    engine.destroy();
  }
}

function histChart(stats, side) {
  const counts = new Array(25).fill(0);
  for (const c of stats.correct) counts[Math.min(24, Math.floor(c))] += 1;
  const total = stats.correct.length;
  const width = 375;
  const height = 110;
  const bar = width / 25;
  const svg = s("svg", {class: "chart hist", viewBox: `0 0 ${width} ${height + 18}`, width, height: height + 18, role: "img",
    "aria-label": `유효 비트 분포, 중앙값 ${stats.median.toFixed(1)}비트`});
  counts.forEach((count, i) => {
    const frac = count / total;
    const barHeight = Math.max(count ? 2 : 0, Math.sqrt(frac) * height);
    svg.append(s("rect", {class: `bar ${side}`, x: i * bar + 1, y: height - barHeight, width: bar - 2, height: barHeight, rx: 1},
      s("title", null, `${i}비트: ${count}개 (${fmt.pct(frac)})`)));
    if (i % 4 === 0) svg.append(s("text", {x: i * bar + bar / 2, y: height + 13, "text-anchor": "middle"}, String(i)));
  });
  return svg;
}

function heatmap(stats, M, N) {
  const canvas = h("canvas", {class: "heat", width: String(Math.min(N, 512)), height: String(Math.min(M, 256)),
    role: "img", "aria-label": "출력 위치별 ULP 크기"});
  const ctx = canvas.getContext("2d");
  const sx = N / canvas.width;
  const sy = M / canvas.height;
  const image = ctx.createImageData(canvas.width, canvas.height);
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const t = (24 - stats.correct[Math.floor(y * sy) * N + Math.floor(x * sx)]) / 24;
      const o = (y * canvas.width + x) * 4;
      image.data[o] = Math.round(0 + t * 196);
      image.data[o + 1] = Math.round(124 - t * 22);
      image.data[o + 2] = Math.round(134 - t * 107);
      image.data[o + 3] = Math.round(40 + t * 215);
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function readout(key, value, sub, changed) {
  return h("div", {class: "readout"}, h("span", {class: "k"}, key),
    h("span", {class: changed ? "v changed" : "v"}, value), h("span", {class: "s"}, sub));
}

function channel(tag, side, label, stats) {
  return h("section", {class: "channel"},
    h("div", {class: "channel-head"}, h("span", {class: `ch-tag ${side}`}, tag), h("span", {class: "label"}, label),
      h("span", {class: "aside"}, `중앙 ${stats.median.toFixed(1)}비트`)),
    h("div", {class: "channel-body"}, histChart(stats, side),
      h("p", {class: "note"}, `유효 비트 (FP32 가수 24비트 중) · 하위 1% ${stats.low.toFixed(1)}비트 · 정확히 같음 ${fmt.pct(stats.exact)}`
        + ` · 상대 오차 ${stats.rel.toExponential(2)}`)));
}

export function renderLab(result, probe) {
  const parityOk = result.parity.equal === result.parity.total;
  return [
    h("header", {class: "runhead"},
      h("h1", null, `연산 실험실 · ${result.M} × ${result.N} × ${result.K}`),
      h("div", {class: "meta"},
        h("span", {class: "pill"}, "누산 알고리즘", h("b", null, mmaSummary(result.request.mma))),
        h("span", {class: "pill"}, "피연산자", h("b", null, result.source)),
        h("span", {class: "pill"}, "기준", h("b", null, "같은 FP8 + FP64 정확 누산")),
        h("span", {class: "pill"}, "GPU", h("b", null, adapterName(probe))))),
    h("div", {class: "channels"},
      channel("A", "a", "f32 FMA (WebGPU)", result.native),
      channel("B", "b", "가상 알고리즘 (WebGPU)", result.emulated)),
    h("div", {class: "readouts"},
      readout("레퍼런스 대조", `${result.parity.equal}/${result.parity.total}`,
        parityOk ? "WebGPU 결과가 JS 정확 레퍼런스와 비트 단위로 같음" : "비트가 다른 출력이 있습니다", !parityOk),
      result.server ? readout("서버 TriCast 대조", `${result.server.equal}/${result.server.total}`, "같은 피연산자, 서버 결과와 비트 비교",
        result.server.equal !== result.server.total) : null,
      readout("B 유효 비트", `${result.emulated.median.toFixed(1)}비트`,
        `중앙값 · A(f32 FMA) ${result.native.median.toFixed(1)}비트 · 같은 FP8 피연산자의 FP64 정확 누산 기준`,
        result.emulated.median < result.native.median - 1),
      readout("WebGPU 시간", `${result.timing.emulated_ms.toFixed(1)} ms`,
        `f32 ${result.timing.native_ms.toFixed(1)} ms · JS 레퍼런스 ${Math.round(result.timing.reference_ms)} ms (별도 스레드) · 벤치마크 아님`)),
    h("section", {class: "panel"}, h("h3", null, "출력 위치별 잃은 비트 (B, FP64 대비)"),
      h("p", {class: "lead"}, "각 픽셀이 출력 원소 하나입니다. 진할수록 정답과 맞는 유효 비트가 적습니다 (24비트 = 완전히 같음)."),
      heatmap(result.emulated, result.M, result.N)),
  ];
}
