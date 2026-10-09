// Reproduction details: environment capture, proof that emulation ran, the recipe, and how to read results.
import {copyText, fmt, h} from "./ui.js";

const RULES = [
  "칩 이름은 모델링하는 산술입니다. 그 GPU에서 측정했다는 뜻이 아니며, Hopper native WGMMA 비트 일치는 140/141로 실패 상태입니다.",
  "원본 대비 차이는 양자화와 누산을 함께 바꾼 결과입니다. 누산만 보려면 비교 기준을 '같은 형식 + FP64 누산'으로 바꿉니다.",
  "텍스트가 갈라진 뒤의 토큰은 서로 다른 문맥이라 위치별로 비교하지 않습니다. 분포 차이는 teacher-forced 지표로 봅니다.",
  "한 장·한 문장의 차이는 사례입니다. 품질 판단은 전체 평가(support_matrix.yaml)로 합니다.",
];

function row(key, value) {
  return [h("dt", null, key), h("dd", null, value ?? "—")];
}

export function renderEvidence(run) {
  const env = run.env || {};
  const versions = env.versions || {};
  const evidence = run.evidence || {};
  const source = env.git_sha ? `${fmt.short(env.git_sha)}${env.git_dirty ? " (수정된 작업 트리)" : ""}` : "—";
  const copy = h("button", {type: "button", class: "copy", onclick: (event) => copyText(event.target, run.recipe.yaml)}, "복사");
  return [
    h("section", {class: "panel"}, h("h3", null, "실행 환경"),
      h("dl", {class: "kv"},
        row("기록 시각 (UTC)", env.utc), row("호스트", env.hostname), row("TriCast 소스", source),
        env.src_sha256 ? row("src 트리 해시", fmt.short(env.src_sha256)) : null,
        row("GPU", (env.gpu_names || []).join(", ") || "없음"), row("CUDA · 드라이버", `${env.cuda ?? "—"} · ${env.gpu_driver ?? "—"}`),
        row("torch · triton", `${versions.torch ?? "—"} · ${versions.triton ?? "—"}`),
        row("transformers", versions.transformers), row("모델", `${env.model_id ?? run.request.model}${env.model_sha ? ` @ ${fmt.short(env.model_sha)}` : ""}`))),
    h("section", {class: "panel"}, h("h3", null, "에뮬레이션 실행 증거"),
      h("p", {class: "lead"}, "패치한 연산자 수와 실제 에뮬레이션 호출 수입니다. 호출 수가 0이면 에뮬레이션이 실행되지 않은 것입니다."),
      h("dl", {class: "kv"},
        row("backend", evidence.backend), row("패치한 Linear", String(evidence.patched_linear ?? "—")),
        row("패치한 Conv2d", String(evidence.patched_conv2d ?? "—")), row("에뮬레이션 호출", String(evidence.emulated_calls ?? "—")),
        row("시간 (기준 · 에뮬레이션)", `${fmt.sec(run.timing?.baseline_s)} · ${fmt.sec(run.timing?.emulated_s)} (처리량 벤치마크 아님)`))),
    h("section", {class: "panel"},
      h("div", {class: "toolbar", style: "justify-content: space-between"},
        h("h3", null, `레시피 ${run.recipe.bundled ? `(번들 ${run.recipe.bundled}와 같음)` : ""}`), copy),
      h("pre", {class: "code"}, run.recipe.yaml)),
    h("section", {class: "panel"}, h("h3", null, "읽는 법"), h("ul", {class: "rules"}, RULES.map((rule) => h("li", null, rule)))),
    h("details", {class: "panel"}, h("summary", null, "실행 결과 원문 (JSON)"),
      h("div", {class: "toolbar", style: "margin-top: 10px"}, h("button", {type: "button", class: "copy",
        onclick: (event) => copyText(event.target, JSON.stringify(run, null, 2))}, "복사"),
      h("span", null, "원본·에뮬레이션 출력, 지표, 환경 기록이 모두 들어 있습니다.")),
      h("pre", {class: "code", style: "margin-top: 10px; max-height: 420px"}, JSON.stringify(run, null, 2))),
  ];
}
