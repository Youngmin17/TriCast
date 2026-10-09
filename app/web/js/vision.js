// Vision tasks: detection boxes on two channels with linked highlighting, and top-5 classification.
import {mediaURL} from "./api.js";
import {fmt, h, s} from "./ui.js";

const IOU_STABLE = 0.9;
const CONF_STABLE = 0.05;

function readout(key, value, sub, changed) {
  return h("div", {class: "readout"}, h("span", {class: "k"}, key),
    h("span", {class: changed ? "v changed" : "v"}, value), h("span", {class: "s"}, sub));
}

function pairing(metrics) {
  const toB = new Map();
  const toA = new Map();
  for (const pair of metrics.pairs) {
    toB.set(pair.baseline, pair);
    toA.set(pair.emulated, pair);
  }
  return {toB, toA};
}

function changedPair(pair, a, b) {
  return pair.iou < IOU_STABLE || Math.abs(a.conf - b.conf) >= CONF_STABLE;
}

function frame(ctx, output, side, links, diffOnly, onHover) {
  const {width, height} = output.image;
  const svg = s("svg", {viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "xMidYMid meet"});
  const fontSize = Math.max(12, Math.round(width / 48));
  output.boxes.forEach((box, index) => {
    const pair = links.get(index);
    const other = pair ? links.other(pair) : null;
    const changed = !pair || changedPair(pair, side === "a" ? box : other, side === "a" ? other : box);
    if (diffOnly && !changed) return;
    const [x1, y1, x2, y2] = box.xyxy;
    const kind = pair ? side : "only";
    const group = s("g", {dataset: {side, index: String(index), pair: pair ? String(links.key(pair)) : ""}});
    group.append(
      s("rect", {class: `box ${kind}`, x: x1, y: y1, width: x2 - x1, height: y2 - y1}),
      s("text", {class: `box-label ${kind}`, x: x1 + 3, y: Math.max(fontSize, y1 - 4), "font-size": fontSize},
        `${box.cls} ${box.conf.toFixed(2)}`));
    group.addEventListener("mouseenter", () => onHover(group.dataset.pair, side, index));
    group.addEventListener("mouseleave", () => onHover(null));
    svg.append(group);
  });
  return h("div", {class: "frame"}, h("img", {src: mediaURL(ctx, output.image.media), alt: "입력 이미지", width, height}), svg);
}

function highlight(root, pairKey, side, index) {
  for (const group of root.querySelectorAll(".frame g")) {
    const rect = group.firstChild;
    const same = pairKey ? group.dataset.pair === pairKey
      : group.dataset.side === side && group.dataset.index === String(index);
    rect.classList.toggle("hot", Boolean(pairKey || side) && same);
    rect.classList.toggle("dim", Boolean(pairKey || side) && !same);
  }
}

function diffTable(run) {
  const a = run.baseline.boxes;
  const b = run.emulated.boxes;
  const {toB, toA} = pairing(run.metrics);
  const rows = run.metrics.pairs.map((pair) => {
    const x = a[pair.baseline];
    const y = b[pair.emulated];
    return h("tr", null, h("td", null, x.cls), h("td", {class: "mono"}, x.conf.toFixed(3)),
      h("td", {class: "mono"}, y.conf.toFixed(3)), h("td", {class: "mono"}, (y.conf - x.conf).toFixed(3)),
      h("td", {class: "mono"}, pair.iou.toFixed(3)));
  });
  a.forEach((box, i) => toB.has(i) || rows.push(h("tr", {class: "only"}, h("td", null, `${box.cls} (A에만)`),
    h("td", {class: "mono"}, box.conf.toFixed(3)), h("td", null, "—"), h("td", null, "—"), h("td", null, "—"))));
  b.forEach((box, i) => toA.has(i) || rows.push(h("tr", {class: "only"}, h("td", null, `${box.cls} (에뮬레이션에만)`),
    h("td", null, "—"), h("td", {class: "mono"}, box.conf.toFixed(3)), h("td", null, "—"), h("td", null, "—"))));
  return h("div", {class: "table-wrap"}, h("table", {class: "diff"},
    h("thead", null, h("tr", null, ["클래스", "A 신뢰도", "B 신뢰도", "Δ", "IoU"].map((t) => h("th", null, t)))),
    h("tbody", null, rows)));
}

export function renderDetect(run, ctx, state, act, chipGlyph) {
  const m = run.metrics;
  const {toB, toA} = pairing(m);
  const linksA = {get: (i) => toB.get(i), other: (p) => run.emulated.boxes[p.emulated], key: (p) => p.baseline};
  const linksB = {get: (i) => toA.get(i), other: (p) => run.baseline.boxes[p.baseline], key: (p) => p.baseline};
  const channels = h("div", {class: "channels"});
  const hover = (pairKey, side, index) => highlight(channels, pairKey, pairKey ? null : side, index);
  channels.append(
    h("section", {class: "channel", "aria-label": run.baseline.label},
      h("div", {class: "channel-head"}, h("span", {class: "ch-tag a"}, "A"), h("span", {class: "label"}, run.baseline.label),
        h("span", {class: "aside"}, `${run.baseline.boxes.length}개`)),
      h("div", {class: "channel-body"}, frame(ctx, run.baseline, "a", linksA, state.diffOnly, hover))),
    h("section", {class: "channel", "aria-label": run.emulated.label},
      h("div", {class: "channel-head"}, h("span", {class: "ch-tag b"}, "B"), h("span", {class: "label"}, run.emulated.label),
        chipGlyph, h("span", {class: "aside"}, `${run.emulated.boxes.length}개`)),
      h("div", {class: "channel-body"}, frame(ctx, run.emulated, "b", linksB, state.diffOnly, hover))));
  const toggle = h("input", {type: "checkbox", id: "diff-only", checked: state.diffOnly,
    onchange: (event) => act.setDiffOnly(event.target.checked)});
  return [
    h("div", {class: "toolbar"},
      h("label", {class: "toggle", for: "diff-only", title: `한쪽에만 있거나, IoU < ${IOU_STABLE} 또는 신뢰도 차이 ≥ ${CONF_STABLE}인 상자`},
        toggle, "차이 나는 상자만"),
      h("span", null, "상자에 마우스를 올리면 다른 채널의 짝이 함께 강조됩니다. 주황 점선은 한쪽에만 있는 상자입니다.")),
    channels,
    h("div", {class: "readouts"},
      readout("짝지은 상자", String(m.matched), "같은 클래스 · IoU ≥ 0.5"),
      readout("A에만", String(m.baseline_only), "B(에뮬레이션)에서 사라짐", m.baseline_only > 0),
      readout("B에만", String(m.emulated_only), "B에서 새로 생김", m.emulated_only > 0),
      readout("평균 IoU", fmt.num(m.mean_iou), "짝지은 상자 기준"),
      readout("평균 |Δ신뢰도|", fmt.num(m.mean_abs_conf_delta), "짝지은 상자 기준")),
    h("section", {class: "panel"}, h("h3", null, "상자별 비교"), diffTable(run)),
  ];
}

function topList(output, other, side) {
  const rankInOther = new Map(other.top.map((item, i) => [item.class_id, i]));
  return h("ol", {class: `toplist ${side}`}, output.top.map((item, i) => {
    const moved = rankInOther.get(item.class_id) !== i;
    return h("li", {class: moved ? "moved" : null},
      h("span", {class: "rank"}, String(i + 1)),
      h("span", {class: "lbl"}, h("span", null, item.label),
        h("span", {class: "bar", style: `width: ${Math.max(2, item.p * 100)}%`})),
      h("span", {class: "p"}, fmt.pct(item.p)));
  }));
}

export function renderClassify(run, ctx, chipGlyph) {
  const m = run.metrics;
  const media = run.baseline.image.media;
  const card = h("figure", {class: "input-card"},
    h("img", {src: mediaURL(ctx, media), alt: "입력 이미지", width: run.baseline.image.width, height: run.baseline.image.height}),
    h("figcaption", null, `입력 이미지 · ${media}`));
  return [
    h("div", {class: "classify-wrap"}, card, h("div", {class: "channels"},
      h("section", {class: "channel"}, h("div", {class: "channel-head"}, h("span", {class: "ch-tag a"}, "A"),
        h("span", {class: "label"}, run.baseline.label)), h("div", {class: "channel-body"}, topList(run.baseline, run.emulated, "a"))),
      h("section", {class: "channel"}, h("div", {class: "channel-head"}, h("span", {class: "ch-tag b"}, "B"),
        h("span", {class: "label"}, run.emulated.label), chipGlyph),
      h("div", {class: "channel-body"}, topList(run.emulated, run.baseline, "b"))))),
    h("div", {class: "readouts"},
      readout("top-1", m.top1_same ? "같음" : "다름", `${run.baseline.top[0].label} → ${run.emulated.top[0].label}`, !m.top1_same),
      readout("top-5 겹침", `${m.top5_overlap}/5`, "같은 클래스 수", m.top5_overlap < 5),
      readout("KL", fmt.kl(m.kl), "A‖B, nats"),
      readout("top-1 확률", `${fmt.pct(m.baseline_top1_p)} → ${fmt.pct(m.emulated_top1_p)}`, "A → B")),
  ];
}
