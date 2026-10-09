// The scope: recorded cases for the current task, the run header, and the task view or a state screen.
import {mediaURL} from "./api.js";
import {STATUS_LABEL} from "./algo.js";
import {renderEvidence} from "./evidence.js";
import {datapath, mmaSummary} from "./glyph.js";
import {renderLab} from "./lab.js";
import {renderLLM} from "./llm.js";
import {designSpace} from "./space.js";
import {resultsTable} from "./table.js";
import {fmt, h} from "./ui.js";
import {renderClassify, renderDetect} from "./vision.js";

const STAGES = [["load", "모델 로드"], ["baseline", "기준 실행"], ["patch", "에뮬레이션 연결"],
  ["emulated", "에뮬레이션 실행"], ["metrics", "지표 계산"]];

const byId = (list, id) => list.find((item) => item.id === id);

function caseCards(state, ctx, act) {
  const groups = new Map();
  for (const run of ctx.runs) {
    if (run.task !== state.sel.task || !run.input) continue;
    const key = `${run.model}|${run.input.prompt ?? run.input.image}`;
    if (!groups.has(key)) groups.set(key, run);
  }
  if (!groups.size) return null;
  const task = byId(ctx.catalog.tasks, state.sel.task);
  const count = (run) => ctx.runs.filter((item) => item.task === run.task && item.model === run.model
    && (item.input?.prompt ?? item.input?.image) === (run.input.prompt ?? run.input.image)).length;
  // The bundle's selection text is long; the card shows the part about this input and keeps the rest as a tooltip.
  const role = (run) => run.role ?? (run.selection || "").split(" · ").find((part) => part.startsWith("이 이미지:"))?.slice(6).trim();
  // COCO media by id; other recorded images by the subject that opens their title; uploads have neither.
  const imageTitle = (run, name) => {
    const coco = /^coco_0*(\d+)/.exec(name);
    if (coco) return `COCO ${coco[1]}`;
    return run.title ? run.title.split(" · ")[0] : "올린 이미지";
  };
  const cards = [...groups.values()].map((run) => {
    const model = byId(ctx.catalog.models, run.model);
    const active = run.model === state.sel.model
      && (run.input.prompt ?? run.input.image) === (state.sel.task === "llm.generate" ? state.sel.prompt : state.sel.image);
    const image = run.input.image;
    return h("button", {type: "button", class: "case", "aria-pressed": String(active), title: run.selection || "",
      onclick: () => act.openCase(run)},
    image ? h("img", {src: mediaURL(ctx, image), alt: "", loading: "lazy", class: "case-img"}) : null,
    h("span", {class: "kind"}, model?.label ?? run.model),
    h("span", {class: "title"}, image ? imageTitle(run, image) : run.input.prompt),
    h("span", {class: "kind"}, [role(run), `설계점 ${count(run)}개`].filter(Boolean).join(" · ")));
  });
  return h("section", {class: "cases"},
    h("div", {class: "cases-head"}, h("h2", null, ctx.mode === "demo" ? "기록된 사례" : "최근 실행"),
      h("span", null, `${task?.label ?? ""} · ${cards.length}개 입력`)),
    h("div", {class: "case-strip"}, cards));
}

function header(run, ctx) {
  const {catalog} = ctx;
  const req = run.request;
  const model = byId(catalog.models, req.model);
  const task = byId(catalog.tasks, req.task);
  const preset = run.preset ? byId(catalog.presets, run.preset) : null;
  const format = catalog.formats.find((item) => item.id === (req.format ?? null));
  const baseline = byId(catalog.baselines, req.baseline || "native");
  const env = run.env || {};
  const pill = (key, value) => h("span", {class: "pill"}, key, h("b", null, value));
  return h("header", {class: "runhead"},
    h("h1", null, `${model?.label ?? req.model} · ${task?.label ?? req.task}`),
    h("div", {class: "meta"},
      pill("누산 알고리즘", run.mma_label ?? mmaSummary(req.mma)),
      pill("출처", preset ? `${preset.label} · ${STATUS_LABEL[preset.status] ?? preset.status}` : "사용자 정의 가상 설계"),
      pill("형식", format?.label ?? String(req.format)),
      pill("비교 기준", baseline?.label ?? req.baseline),
      pill("backend", run.evidence?.backend ?? "—"),
      pill("GPU", (env.gpu_names || [])[0] ?? "—"),
      pill("소스", fmt.short(env.git_sha)),
      env.utc ? pill("기록", env.utc.slice(0, 10)) : null,
      run.reused ? pill("서버 캐시", "같은 요청의 저장된 결과") : null));
}

function tabs(state, act) {
  const tab = (id, label) => h("button", {type: "button", role: "tab", "aria-selected": String(state.tab === id),
    onclick: () => act.setTab(id)}, label);
  return h("div", {class: "tabs", role: "tablist"}, tab("compare", "비교"), tab("table", "결과 데이터"),
    tab("evidence", "재현 정보"));
}

function taskView(run, state, ctx, act) {
  const glyph = datapath(run.request.mma, 112);
  const space = designSpace(state, ctx, act);
  if (run.request.task === "llm.generate") {
    const [prompt, channels, ruler, readouts, chart] = renderLLM(run, glyph);
    return [prompt, channels, ruler, readouts, space, chart];
  }
  if (run.request.task === "vision.detect") return [...renderDetect(run, ctx, state, act, glyph), space];
  return [...renderClassify(run, ctx, glyph), space];
}

// While a live run is going: the pipeline, and the baseline output as soon as it exists.
function progress(run) {
  const index = STAGES.findIndex(([id]) => id === run.stage);
  const parts = [h("section", {class: "state"},
    h("h2", null, run.status === "queued" ? "대기 중" : `실행 중 · ${Math.round((run.progress || 0) * 100)}%`),
    h("p", null, run.baseline ? "기준(A) 결과가 나왔습니다. 같은 모델에 가상 산술을 연결해 다시 실행하는 중입니다."
      : "기준(A)을 먼저 실행하고, 같은 모델에 가상 산술을 연결해 다시 실행합니다."),
    h("div", {class: "pipeline"}, STAGES.map(([id, label], i) =>
      h("span", {class: i < index ? "done" : i === index ? "now" : null}, label))))];
  if (run.baseline) {
    const b = run.baseline;
    const body = b.text != null ? h("div", {class: "tokens"}, b.text)
      : b.boxes ? h("p", null, `검출 ${b.boxes.length}개: ${b.boxes.slice(0, 8).map((x) => `${x.cls} ${x.conf.toFixed(2)}`).join(", ")}`)
        : h("p", null, `top-1 ${b.top[0].label} (${fmt.pct(b.top[0].p)})`);
    parts.push(h("div", {class: "channels"},
      h("section", {class: "channel"}, h("div", {class: "channel-head"}, h("span", {class: "ch-tag a"}, "A"),
        h("span", {class: "label"}, b.label), h("span", {class: "aside"}, run.cached_baseline ? "캐시" : "방금 실행")),
      h("div", {class: "channel-body"}, body)),
      h("section", {class: "channel"}, h("div", {class: "channel-head"}, h("span", {class: "ch-tag b"}, "B"),
        h("span", {class: "label"}, "에뮬레이션 실행 중…")), h("div", {class: "channel-body"},
        h("p", {class: "note"}, "끝나면 여기에 같은 형식으로 표시됩니다.")))));
  }
  return parts;
}

function missing(state, ctx, act) {
  const options = ctx.runs.filter((run) => run.task === state.sel.task && run.model === state.sel.model
    && (run.input?.prompt ?? run.input?.image) === (state.sel.task === "llm.generate" ? state.sel.prompt : state.sel.image));
  return h("section", {class: "state"},
    h("h2", null, "이 조합으로 기록된 실행이 없습니다"),
    h("p", null, ctx.mode === "demo"
      ? "데모는 GPU에서 미리 실행한 결과만 보여 줍니다. 아래 기록 중 하나를 고르거나, 라이브 서버에서 직접 실행하세요."
      : "왼쪽에서 조합을 고르고 비교 실행을 누르세요."),
    options.length ? h("div", {class: "suggest"}, options.map((run) => {
      return h("button", {type: "button", class: "sample", onclick: () => act.openCase(run)},
        `${mmaSummary(run.mma)} · ${run.format ?? "양자화 없음"} · ${run.baseline === "same_quant_fp64" ? "FP64 기준" : "원본 기준"}`);
    })) : null);
}

const LAB_STAGES = [["operands", "피연산자 준비"], ["engine", "GPU 준비"], ["emulated", "가상 산술 실행"],
  ["native", "f32 실행"], ["reference", "레퍼런스 대조"]];

function labScope(state, ctx, act) {
  if (state.error) return [h("section", {class: "state"}, h("h2", null, "실행하지 못했습니다"), h("p", null, state.error))];
  if (state.lab.result) return renderLab(state.lab.result, act.labProbe());
  if (state.busy) {
    const index = LAB_STAGES.findIndex(([id]) => id === state.lab.stage);
    return [h("section", {class: "state"}, h("h2", null, "브라우저 GPU에서 실행 중"),
      h("div", {class: "pipeline"}, LAB_STAGES.map(([id, label], i) =>
        h("span", {class: i < index ? "done" : i === index ? "now" : null}, label))))];
  }
  return [h("section", {class: "state"}, h("h2", null, "설계한 산술을 이 컴퓨터의 GPU에서 바로 실행합니다"),
    h("p", null, "왼쪽에서 피연산자와 알고리즘을 고르고 '브라우저 GPU에서 실행'을 누르세요. F를 바꿔 가며 다시 실행하면 오차 분포가 어떻게 달라지는지 볼 수 있습니다."))];
}

export function renderScope(root, state, ctx, act) {
  if (ctx.mode === "webgpu") {
    root.replaceChildren(act.envPanel(), ...labScope(state, ctx, act));
    return;
  }
  const parts = [act.envPanel(), caseCards(state, ctx, act)];
  if (state.error) {
    parts.push(h("section", {class: "state"}, h("h2", null, "실행하지 못했습니다"), h("p", null, state.error)));
  } else if (state.run && state.run.status === "done") {
    const view = state.tab === "evidence" ? renderEvidence(state.run)
      : state.tab === "table" ? resultsTable(state, ctx, act) : taskView(state.run, state, ctx, act);
    parts.push(header(state.run, ctx), tabs(state, act), ...view);
  } else if (state.run && state.run.status === "error") {
    parts.push(h("section", {class: "state"}, h("h2", null, "실행 중 오류가 났습니다"),
      h("p", null, state.run.error?.message ?? "알 수 없는 오류"), h("p", {class: "note"}, state.run.error?.code ?? "")));
  } else if (state.run) {
    parts.push(...progress(state.run));
  } else {
    parts.push(missing(state, ctx, act), ...(ctx.runs.length ? resultsTable(state, ctx, act) : []));
  }
  root.replaceChildren(...parts.filter(Boolean));
}
