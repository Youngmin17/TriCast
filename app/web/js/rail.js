// Configuration rail: task → model → accumulation algorithm → format/baseline → input, then run.
import {designer} from "./algo.js";
import {mediaURL} from "./api.js";
import {SIZES} from "./lab.js";
import {h, radioGroup} from "./ui.js";

const SUPPORT_LABEL = {full_eval: "전체 평가", operator: "연산자 검증", experimental: "실험적"};
const KIND_LABEL = {llm: "LLM", vision: "비전"};

function step(no, title, hint, ...body) {
  return h("section", {class: "step"},
    h("div", {class: "step-head"}, h("span", {class: "step-no"}, no), h("h2", {class: "step-title"}, title),
      hint ? h("span", {class: "step-hint"}, hint) : null),
    ...body);
}

function taskStep(state, ctx, act) {
  const items = ctx.catalog.tasks.map((task) => ({
    id: task.id, disabled: !act.available("task", task.id),
    render: () => h("button", {type: "button", title: task.description}, task.label,
      h("small", null, KIND_LABEL[task.kind] || task.kind)),
  }));
  return step("1", "작업", null, radioGroup("작업", items, state.sel.task, (id) => act.pick({task: id}), "seg"));
}

function modelStep(state, ctx, act) {
  const items = ctx.catalog.models.filter((model) => model.tasks.includes(state.sel.task)).map((model) => ({
    id: model.id, disabled: !act.available("model", model.id),
    render: () => h("button", {type: "button", class: "option"},
      h("span", {class: "name"}, model.label),
      h("span", {class: `badge ${model.support}`}, SUPPORT_LABEL[model.support] || model.support),
      h("span", {class: "sub"}, model.note)),
  }));
  return step("2", "모델", null, radioGroup("모델", items, state.sel.model, (id) => act.pick({model: id}), "options"));
}

function algorithmStep(state, ctx, act) {
  return step("3", "누산 알고리즘", "가상 설계", ...designer(state, ctx, act));
}

function formatStep(state, ctx, act) {
  const algo = ctx.catalog.algorithms.find((item) => item.id === state.sel.mma.algorithm);
  const known = new Map(ctx.catalog.formats.map((format) => [format.id ?? null, format]));
  const formatItems = (algo?.formats || []).map((id) => {
    const format = known.get(id ?? null) || {id, label: id ?? "양자화 없음"};
    return {id: id ?? null, disabled: !act.available("format", id ?? null),
      render: () => h("button", {type: "button", title: format.note || ""}, format.label)};
  });
  const baselineItems = ctx.catalog.baselines.map((baseline) => ({
    id: baseline.id, disabled: !act.available("baseline", baseline.id),
    render: () => h("button", {type: "button", title: baseline.description}, baseline.label),
  }));
  const current = ctx.catalog.baselines.find((baseline) => baseline.id === state.sel.baseline);
  return step("4", "형식과 비교 기준", null,
    radioGroup("입력 형식", formatItems, state.sel.format, (id) => act.pick({format: id}), "chips"),
    radioGroup("비교 기준", baselineItems, state.sel.baseline, (id) => act.pick({baseline: id}), "seg"),
    current ? h("p", {class: "note"}, current.description) : null);
}

function recordedInputs(state, ctx) {
  const seen = new Map();
  for (const run of ctx.runs) {
    if (run.task !== state.sel.task || run.model !== state.sel.model || !run.input) continue;
    const key = run.input.prompt ?? run.input.image;
    if (key && !seen.has(key)) seen.set(key, run.input);
  }
  return [...seen.values()];
}

function promptInput(state, ctx, act) {
  const demo = ctx.mode === "demo";
  const samples = recordedInputs(state, ctx).map((input) => h("button", {
    type: "button", class: "sample", "aria-pressed": String(input.prompt === state.sel.prompt),
    onclick: () => act.pick({prompt: input.prompt}),
  }, input.prompt));
  const area = h("textarea", {class: "prompt", id: "prompt", "aria-label": "프롬프트", readonly: demo,
    oninput: (event) => act.edit({prompt: event.target.value})});
  area.value = state.sel.prompt || "";
  const tokens = h("input", {type: "range", id: "max-tokens", min: "8", max: "64", step: "8",
    value: String(state.sel.maxTokens), disabled: demo,
    oninput: (event) => {
      act.edit({maxTokens: Number(event.target.value)});
      event.target.nextElementSibling.textContent = event.target.value;
    }});
  return [
    samples.length ? h("div", {class: "samples", role: "group", "aria-label": "기록된 프롬프트"}, samples) : null,
    area,
    h("label", {class: "field-row", for: "max-tokens"}, "생성 토큰", tokens,
      h("span", {class: "num"}, String(state.sel.maxTokens))),
  ];
}

function imageInput(state, ctx, act) {
  const demo = ctx.mode === "demo";
  const thumbs = recordedInputs(state, ctx).map((input) => h("button", {
    type: "button", class: "thumb", "aria-pressed": String(input.image === state.sel.image),
    "aria-label": input.image, onclick: () => act.pick({image: input.image}),
  }, h("img", {src: mediaURL(ctx, input.image), alt: "", loading: "lazy"})));
  if (state.sel.image?.startsWith("data:")) {
    thumbs.unshift(h("button", {type: "button", class: "thumb", "aria-pressed": "true", "aria-label": "올린 이미지"},
      h("img", {src: state.sel.image, alt: ""})));
  }
  const upload = demo ? null : h("label", {class: "field-row"}, "이미지 올리기",
    h("input", {type: "file", id: "upload", accept: "image/jpeg,image/png",
      onchange: (event) => act.upload(event.target.files[0])}));
  return [thumbs.length ? h("div", {class: "thumbs"}, thumbs) : null, upload];
}

function inputStep(state, ctx, act) {
  const llm = state.sel.task === "llm.generate";
  return step("5", "입력", llm ? "같은 프롬프트, greedy 생성" : "같은 이미지",
    ...(llm ? promptInput(state, ctx, act) : imageInput(state, ctx, act)));
}

function runControls(state, ctx, act) {
  if (ctx.mode === "demo") {
    const recorded = act.recorded();
    return h("div", null, h("p", {class: recorded ? "note" : "note warn"}, recorded
      ? "기록 재생 모드입니다. 파라미터를 바꾸면 기록된 설계점 중 가장 가까운 실행이 열립니다."
      : "이 조합은 기록된 실행이 없습니다. 라이브 서버(python -m app.server --device cuda)에서 실행할 수 있습니다."));
  }
  const ready = state.sel.task === "llm.generate" ? Boolean((state.sel.prompt || "").trim()) : Boolean(state.sel.image);
  return h("div", null,
    h("button", {type: "button", class: "run-btn", disabled: state.busy || !ready, onclick: () => act.run()},
      state.busy ? "실행 중…" : "비교 실행"),
    h("p", {class: "note"}, "GPU 하나에서 원본과 에뮬레이션을 차례로 실행합니다."));
}

function labRail(state, ctx, act) {
  const sources = ctx.catalog.models.map((model) => ({
    id: model.id,
    render: () => h("button", {type: "button", class: "option"}, h("span", {class: "name"}, model.label),
      h("span", {class: "sub"}, model.note)),
  }));
  const sizes = SIZES.map((size) => ({id: size.id, render: () => h("button", {type: "button"}, size.label)}));
  return [
    step("1", "작업", "브라우저 GPU", h("p", {class: "note"},
      "연산 실험실: FP8 행렬곱 하나를 설계한 산술로 실행합니다. LLM·이미지 모델 전체는 서버 GPU 모드에서 실행합니다.")),
    step("2", "피연산자", null, radioGroup("피연산자", sources, state.sel.model, (id) => act.pick({model: id}), "options")),
    step("3", "누산 알고리즘", "가상 설계 · CoFDA", ...designer(state, ctx, act)),
    step("4", "형식과 기준", null, h("p", {class: "note"},
      "입력 FP8 E4M3 (fn) · 출력 FP32 · 기준은 같은 피연산자를 FP64로 정확히 누산한 값입니다.")),
    state.sel.model === "random" ? step("5", "크기", "M × N × K",
      radioGroup("크기", sizes, state.lab.size, (id) => act.setLabSize(id), "seg")) : null,
    h("div", null, h("button", {type: "button", class: "run-btn", disabled: state.busy, onclick: () => act.run()},
      state.busy ? "실행 중…" : "브라우저 GPU에서 실행"),
    h("p", {class: "note"}, "이 컴퓨터의 GPU를 씁니다. 결과는 레퍼런스와 비트 단위로 대조합니다.")),
  ];
}

export function renderRail(root, state, ctx, act) {
  if (ctx.mode === "webgpu") {
    root.replaceChildren(...labRail(state, ctx, act).filter(Boolean));
    return;
  }
  root.replaceChildren(
    taskStep(state, ctx, act), modelStep(state, ctx, act), algorithmStep(state, ctx, act), formatStep(state, ctx, act),
    inputStep(state, ctx, act), runControls(state, ctx, act));
}
