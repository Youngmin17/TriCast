// The three run modes and what each can run: recorded examples, the server's GPU, the browser's GPU (WebGPU).
import {adapterName} from "./lab.js";
import {h, withTip} from "./ui.js";

export const MODES = [
  {id: "offline", label: "예시 데이터", sub: "기록된 실행"},
  {id: "server", label: "서버 GPU", sub: "실시간 · 모델 작업"},
  {id: "webgpu", label: "브라우저 GPU", sub: "실시간 · 연산 실험실"},
];

async function json(url) {
  const response = await fetch(url, {headers: {accept: "application/json"}});
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

async function bundle() {
  try {
    const [catalog, index] = await Promise.all([json("demo/catalog.json"), json("demo/index.json")]);
    return {ok: true, ctx: {mode: "demo", source: "bundle", catalog, runs: index.runs || []}};
  } catch (error) {
    return {ok: false, reason: `기록 묶음을 읽지 못했습니다 (${error.message})`};
  }
}

async function server() {
  try {
    const catalog = await json("api/catalog");
    if (catalog.mode !== "live") return {ok: false, reason: "서버가 기록 재생 모드로 실행 중입니다 (python -m app.server --device cuda 로 켜면 사용 가능)"};
    const [listing, resources] = await Promise.all([json("api/runs"), json("api/resources").catch(() => null)]);
    const gpus = resources?.server?.gpus ?? [];
    return {ok: true, resources, ctx: {mode: "live", source: "server", catalog, runs: listing.runs || []},
      reason: gpus.length ? null : "서버에 GPU가 보이지 않습니다. CPU 레퍼런스로 실행하면 매우 느립니다."};
  } catch {
    return {ok: false, reason: "이 페이지는 서버 없이 열렸습니다. GPU 노드에서 python -m app.server --device cuda 를 실행하고 그 주소로 여세요."};
  }
}

async function webgpu() {
  try {
    const {probeWebGPU} = await import("./webgpu/probe.js");
    const probe = await probeWebGPU();
    return {ok: probe.available, probe, reason: probe.available ? null : probe.reason};
  } catch (error) {
    return {ok: false, reason: `WebGPU 엔진을 불러오지 못했습니다 (${error.message})`};
  }
}

export async function detectRuntimes() {
  const [offline, live, gpu] = await Promise.all([bundle(), server(), webgpu()]);
  return {offline, server: live, webgpu: gpu};
}

export function modeSwitch(state, runtimes, act) {
  const group = h("div", {class: "modes", role: "radiogroup", "aria-label": "실행 모드"});
  for (const mode of MODES) {
    const rt = runtimes[mode.id];
    const button = h("button", {type: "button", role: "radio", class: "mode", "aria-checked": String(state.runtime === mode.id),
      disabled: !rt?.ok || (state.busy && state.runtime !== mode.id), onclick: () => act.setRuntime(mode.id)},
    h("span", {class: `dot ${rt?.ok ? "on" : ""}`}), h("span", {class: "mode-label"}, mode.label),
    h("small", null, mode.sub));
    withTip(button, () => h("div", null, rt?.ok ? "사용 가능" : rt?.reason ?? "확인 중"));
    group.append(button);
  }
  return group;
}

function gpuRows(gpus) {
  return gpus.map((gpu) => h("li", null,
    h("b", null, `GPU ${gpu.index} · ${gpu.name}`),
    ` · ${Math.round(gpu.memory_used_mb / 1024)} / ${Math.round(gpu.memory_total_mb / 1024)} GB · 사용률 ${gpu.utilization_pct}%`));
}

function bytes(n) {
  return n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : `${Math.round(n / 2 ** 20)} MB`;
}

export function resourcePanel(state, runtimes) {
  const rt = runtimes[state.runtime];
  if (state.runtime === "offline") {
    const runs = rt.ctx.runs;
    const models = [...new Set(runs.map((run) => run.model))];
    return h("section", {class: "env"},
      h("div", {class: "env-head"}, h("b", null, "예시 데이터"), h("span", null, "미리 실행해 둔 결과를 봅니다. 입력은 기록된 것만 고를 수 있습니다.")),
      h("p", {class: "env-line"}, `기록된 실행 ${runs.length}건 · 모델 ${models.length}개 · 실시간으로 직접 넣으려면 서버 GPU 또는 브라우저 GPU 모드를 쓰세요.`));
  }
  if (state.runtime === "server") {
    const s = rt.resources?.server;
    return h("section", {class: "env"},
      h("div", {class: "env-head"}, h("b", null, `서버 GPU · ${s?.hostname ?? "알 수 없음"}`),
        h("span", null, "직접 넣은 프롬프트·이미지를 서버 GPU에서 실시간으로 실행합니다.")),
      s?.gpus?.length ? h("ul", {class: "env-list"}, gpuRows(s.gpus)) : h("p", {class: "env-line warn"}, rt.reason),
      h("p", {class: "env-line"}, `torch ${s?.torch ?? "—"} · triton ${s?.triton ?? "—"} · CUDA ${s?.cuda ?? "—"} · 대기 ${s?.queue?.queued ?? 0} · 실행 중 ${s?.queue?.running ?? 0}`),
      s ? h("p", {class: "env-line"}, `실행 가능한 작업: ${(s.runnable_tasks || []).join(", ") || "없음"} · 캐시된 모델: ${(s.models_cached || []).join(", ") || "없음"}`) : null);
  }
  const probe = rt.probe;
  return h("section", {class: "env"},
    h("div", {class: "env-head"}, h("b", null, `브라우저 GPU · ${adapterName(probe)}`),
      h("span", null, "이 컴퓨터의 GPU에서 설계한 산술로 행렬곱을 바로 실행하고 레퍼런스와 비트 단위로 대조합니다.")),
    h("p", {class: "env-line"}, `${probe.adapter?.vendor ?? ""} ${probe.adapter?.architecture ?? ""} · 버퍼 최대 ${bytes(probe.limits.maxBufferSize)} · `
      + `저장 버퍼 바인딩 최대 ${bytes(probe.limits.maxStorageBufferBindingSize)} · 워크그룹 공유 메모리 ${Math.round(probe.limits.maxComputeWorkgroupStorageSize / 1024)} KB`
      + `${probe.fallback ? " · 소프트웨어 어댑터 (느림)" : ""}`),
    h("p", {class: "env-line"}, "모델 전체(LLM·이미지)를 브라우저에서 에뮬레이션하는 기능은 아직 없습니다. 모델 작업은 서버 GPU 모드에서 실행하세요."));
}
