// Data source: the TriCast Studio server when one answers, otherwise the recorded bundle in demo/.
// Shapes follow app/README.md.
import {mmaKey} from "./algo.js";

const BUNDLE = "demo/";

async function getJSON(url, init) {
  const response = await fetch(url, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body?.error?.message || `${response.status} ${url}`);
    error.status = response.status;
    error.code = body?.error?.code;
    throw error;
  }
  return body;
}

export function mediaURL(ctx, name) {
  const file = encodeURIComponent(name);
  return ctx.source === "server" ? `api/media/${file}` : `${BUNDLE}media/${file}`;
}

export function inputOf(sel) {
  return sel.task === "llm.generate" ? {prompt: sel.prompt, max_new_tokens: sel.maxTokens} : {image: sel.image};
}

// The server keys runs on the exact prompt and token budget; recorded bundle entries carry no budget.
function sameInput(a = {}, b = {}) {
  if ("prompt" in b) return a.prompt === b.prompt && (a.max_new_tokens == null || a.max_new_tokens === b.max_new_tokens);
  return a.image === b.image;
}

export function findRecorded(ctx, sel) {
  const key = mmaKey(sel.mma);
  return ctx.runs.find((run) =>
    run.task === sel.task && run.model === sel.model && run.mma && mmaKey(run.mma) === key
    && (run.format ?? null) === (sel.format ?? null) && (run.baseline || "native") === sel.baseline
    && run.status !== "error" && sameInput(run.input, inputOf(sel)));
}

export async function serverResources() {
  return getJSON("api/resources");
}

export async function listRuns() {
  return (await getJSON("api/runs")).runs || [];
}

export function loadRun(ctx, id) {
  return ctx.source === "server" ? getJSON(`api/runs/${encodeURIComponent(id)}`)
    : getJSON(`${BUNDLE}runs/${encodeURIComponent(id)}.json`);
}

// Live server: poll a run once a second until it finishes or `wanted()` turns false; onUpdate gets every state.
export async function follow(id, onUpdate, wanted = () => true) {
  for (;;) {
    const run = await getJSON(`api/runs/${encodeURIComponent(id)}`);
    if (!wanted()) return run;
    onUpdate(run);
    if (run.status === "done" || run.status === "error") return run;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

// Live server: queue the run and follow it. `reused` marks a finished run the server returned from its cache.
export async function submit(request, onUpdate) {
  const {id, cached} = await getJSON("api/runs", {
    method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(request),
  });
  const mark = (run) => ({...run, reused: Boolean(cached)});
  return mark(await follow(id, (run) => onUpdate(mark(run))));
}
