// LLM text generation: two channels of tokens, the shared prefix, and teacher-forced distribution shift.
import {fmt, h, s, withTip} from "./ui.js";

function visible(piece) {
  return piece.replace(/\n/g, "⏎").replace(/ /g, "␣") || "∅";
}

function tokenTip(token, index) {
  const chosen = Math.exp(token.logprob);
  return h("div", null,
    h("div", null, `${index + 1}번째 토큰 · 선택 확률 ${fmt.pct(chosen)}`),
    h("table", null, token.top.map((alt) => h("tr", null,
      h("td", null, visible(alt.piece)), h("td", {class: "muted"}, fmt.pct(alt.p))))));
}

function tokenSpans(tokens, split, side) {
  const spans = [];
  tokens.forEach((token, index) => {
    if (index === split) spans.push(h("span", {class: "split-mark", "aria-hidden": "true"}));
    let kind = "tok";
    if (split != null && index === split) kind += " split";
    else if (split != null && index > split) kind += side === "b" ? " after-b" : " after-a";
    const span = h("span", {class: kind, tabindex: index === split ? "0" : null}, token.piece);
    spans.push(withTip(span, () => tokenTip(token, index)));
  });
  return spans;
}

function channel(tag, label, aside, tokens, split, side, extra) {
  return h("section", {class: "channel", "aria-label": label},
    h("div", {class: "channel-head"}, h("span", {class: `ch-tag ${side}`}, tag), h("span", {class: "label"}, label),
      extra, h("span", {class: "aside"}, aside)),
    h("div", {class: "channel-body"}, h("div", {class: "tokens"}, tokenSpans(tokens, split, side))));
}

function ruler(length, split) {
  const cells = Array.from({length}, (_, i) => {
    if (split == null || i < split) return h("span", {title: `${i + 1}: 같은 토큰`});
    if (i === split) return h("span", {class: "split", title: `${i + 1}: 첫 분기`});
    return h("span", {class: "na", title: `${i + 1}: 분기 이후, 위치 비교 안 함`});
  });
  return h("div", {class: "ruler"},
    h("div", {class: "ruler-cells", role: "img",
      "aria-label": split == null ? "생성한 토큰이 모두 같습니다" : `${split + 1}번째 토큰에서 처음 갈라집니다`}, cells),
    h("div", {class: "ruler-legend"},
      h("span", null, h("i", {style: "background: var(--ch-b)"}), "같은 토큰"),
      h("span", null, h("i", {style: "background: var(--diff)"}), "첫 분기"),
      h("span", null, h("i", {style: "background: repeating-linear-gradient(135deg, var(--rule) 0 3px, transparent 3px 6px)"}),
        "분기 이후 (서로 다른 문맥이라 위치 비교 안 함)")));
}

function readout(key, value, sub, changed) {
  return h("div", {class: "readout"}, h("span", {class: "k"}, key),
    h("span", {class: changed ? "v changed" : "v"}, value), h("span", {class: "s"}, sub));
}

function klChart(run) {
  const tf = run.metrics.teacher_forced;
  const tokens = run.baseline.tokens;
  const floor = 1e-6;
  const top = Math.max(1, tf.kl_max || 0);
  const span = Math.log10(top) - Math.log10(floor);
  const bar = 12;
  const gap = 3;
  const left = 44;
  const height = 150;
  const plot = height - 26;
  const width = Math.max(560, left + tf.positions * (bar + gap) + 8);
  const y = (value) => plot - ((Math.log10(Math.max(value, floor)) - Math.log10(floor)) / span) * (plot - 8);
  const svg = s("svg", {class: "chart", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": `위치별 KL, 평균 ${fmt.kl(tf.kl_mean)}, 최대 ${fmt.kl(tf.kl_max)} nats`});
  // The maximum gets its own label when it exceeds 1; the "1" label is dropped if the two would overlap.
  const ticks = [1e-6, 1e-4, 1e-2, 1, top > 1 ? top : null]
    .filter((v) => v != null && !(v === 1 && top > 1 && y(1) - y(top) < 12));
  for (const tick of ticks) {
    svg.append(s("line", {class: "axis", x1: left - 4, x2: width, y1: y(tick), y2: y(tick)}),
      s("text", {x: left - 8, y: y(tick) + 3.5, "text-anchor": "end"}, tick >= 1 ? tick.toFixed(tick > 1 ? 1 : 0)
        : `1e${Math.round(Math.log10(tick))}`));
  }
  tf.kl.forEach((value, i) => {
    const x = left + i * (bar + gap);
    const rect = s("rect", {class: tf.top1[i] ? "bar" : "bar miss", x, y: y(value), width: bar,
      height: Math.max(1, plot - y(value)), rx: 1.5});
    withTip(rect, () => h("div", null,
      h("div", null, `${i + 1}번째 위치 · 기준(A) 토큰 ${visible(tokens[i]?.piece ?? "")}`),
      h("div", null, `KL ${fmt.kl(value)} nats · top-1 ${tf.top1[i] ? "같음" : "다름"}`)));
    svg.append(rect);
    if (tf.positions <= 64 && (i + 1) % 8 === 0) {
      svg.append(s("text", {x: x + bar / 2, y: height - 8, "text-anchor": "middle"}, String(i + 1)));
    }
  });
  return h("div", {class: "chart-wrap"}, svg);
}

export function renderLLM(run, chipGlyph) {
  const split = run.metrics.first_divergence;
  const a = run.baseline.tokens;
  const b = run.emulated.tokens;
  const tf = run.metrics.teacher_forced;
  const length = Math.max(a.length, b.length);
  return [
    h("p", {class: "prompt-line"}, run.request.input.prompt),
    h("div", {class: "channels"},
      channel("A", run.baseline.label, `${a.length}토큰`, a, split, "a"),
      channel("B", run.emulated.label, `${b.length}토큰`, b, split, "b", chipGlyph)),
    ruler(length, split),
    h("div", {class: "readouts"},
      readout("첫 분기", split == null ? "없음" : `${split + 1}번째`, `생성 ${length}토큰 중`, split != null),
      readout("같은 앞부분", `${run.metrics.prefix_match}토큰`, "두 채널이 똑같이 생성한 길이"),
      readout("top-1 일치", fmt.pct(tf.top1_agreement), `기준(A) 토큰을 차례로 넣었을 때 ${tf.positions}개 위치`,
        tf.top1_agreement < 1),
      readout("KL 평균", fmt.kl(tf.kl_mean), `최대 ${fmt.kl(tf.kl_max)} nats`)),
    h("section", {class: "panel"},
      h("h3", null, "같은 문맥에서의 다음 토큰 분포 차이"),
      h("p", {class: "lead"}, "기준(A)이 생성한 토큰을 에뮬레이션 모델에 하나씩 차례로 넣어(생성과 같은 방식), 위치마다 다음 토큰 분포의 "
        + "KL(A‖B)을 로그 눈금으로 그렸습니다. 주황 막대는 가장 확률 높은 토큰이 달라진 위치입니다. 텍스트가 갈라진 뒤에도 이 비교는 같은 문맥에서 이루어집니다."),
      klChart(run)),
  ];
}
