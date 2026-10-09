// Dataflow sketch of the designed accumulation: what is aligned together, where C joins, where FP32 enters.
import {s} from "./ui.js";

const W = 312;
const H = 92;

function box(x, y, w, label, strong) {
  return [
    s("rect", {x, y, width: w, height: 24, rx: 4, class: strong ? "fl-box strong" : "fl-box"}),
    s("text", {x: x + w / 2, y: y + 16, "text-anchor": "middle", class: "fl-text"}, label),
  ];
}

function arrow(x1, y1, x2, y2) {
  return s("path", {d: `M${x1} ${y1} L${x2} ${y2}`, class: "fl-line", "marker-end": "url(#fl-head)"});
}

function products(x, y, count, label) {
  const cells = [];
  const shown = Math.min(count, 8);
  for (let i = 0; i < shown; i += 1) cells.push(s("rect", {x: x + i * 8, y, width: 6, height: 24, rx: 1, class: "fl-cell"}));
  cells.push(s("text", {x: x + shown * 4 - 1, y: y + 38, "text-anchor": "middle", class: "fl-note"}, label));
  return cells;
}

function frame() {
  return s("svg", {class: "flow", viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img"},
    s("defs", null, s("marker", {id: "fl-head", viewBox: "0 0 6 6", refX: "5", refY: "3", markerWidth: "6",
      markerHeight: "6", orient: "auto-start-reverse"}, s("path", {d: "M0 0 L6 3 L0 6 z", class: "fl-headfill"}))));
}

export function flow(mma) {
  const svg = frame();
  const add = (...nodes) => svg.append(...nodes.flat());
  if (mma.algorithm === "cofda") {
    const promote = mma.promote_interval > 0;
    add(products(4, 14, mma.chunk_size, `곱 ${mma.chunk_size}개`));
    if (mma.c_mode === "decoupled") {
      add(arrow(70, 26, 92, 26), box(92, 14, 58, `FDA F${mma.f_bits}`, true), arrow(150, 26, 170, 26),
        s("text", {x: 160, y: 18, "text-anchor": "middle", class: "fl-note"}, "P"),
        box(170, 14, 62, `FDA F2=${mma.f2_bits}`), arrow(232, 26, 252, 26), box(252, 14, 40, "C"),
        s("path", {d: "M272 38 L272 52 L201 52 L201 40", class: "fl-line", "marker-end": "url(#fl-head)"}),
        s("text", {x: 236, y: 64, "text-anchor": "middle", class: "fl-note"}, "C는 묶음 합과 따로 결합"));
    } else {
      add(arrow(70, 26, 100, 26), box(100, 14, 70, `FDA F${mma.f_bits}`, true), arrow(170, 26, 196, 26), box(196, 14, 40, "C"),
        s("path", {d: "M216 38 L216 52 L135 52 L135 40", class: "fl-line", "marker-end": "url(#fl-head)"}),
        s("text", {x: 176, y: 64, "text-anchor": "middle", class: "fl-note"}, "C도 같은 정렬·절단에 참여"));
    }
    if (promote) {
      add(s("rect", {x: 2, y: 4, width: W - 4, height: 66, rx: 6, class: "fl-group"}),
        s("text", {x: W - 6, y: 84, "text-anchor": "end", class: "fl-note"},
          `${mma.promote_interval}개마다 부분합을 FP32 FMA로 합류, 부분합은 0에서 다시 시작`));
    } else {
      add(s("text", {x: 4, y: 84, class: "fl-note"}, `K를 ${mma.chunk_size}개씩 나눠 반복 · FP32 레지스터 C를 다음 묶음에 전달`));
    }
    return svg;
  }
  if (mma.algorithm === "gdfs") {
    const groups = Math.max(1, Math.round(mma.k_tile / mma.group_size));
    add(products(4, 6, mma.group_size, `그룹 ${mma.group_size}개 × ${groups}`), arrow(70, 18, 92, 18),
      box(92, 6, 62, `정렬·합 G${mma.g_bits}`), arrow(154, 18, 178, 18), box(178, 6, 62, `FDA F${mma.f_bits}`, true),
      arrow(240, 18, 258, 18), box(258, 6, 40, "C"),
      s("path", {d: "M278 30 L278 44 L209 44 L209 32", class: "fl-line", "marker-end": "url(#fl-head)"}),
      s("text", {x: 4, y: 70, class: "fl-note"}, `그룹 합은 고정소수점으로 두고, 타일(K ${mma.k_tile})마다 FDA 한 번`),
      s("text", {x: 4, y: 84, class: "fl-note"}, "그룹마다 FP32로 정규화하지 않음"));
    return svg;
  }
  if (mma.algorithm === "fp32_fma") {
    add(products(4, 14, 1, "곱 1개"), arrow(16, 26, 60, 26), box(60, 14, 84, "FMA (FP32)", true), arrow(144, 26, 170, 26),
      box(170, 14, 40, "acc"), s("path", {d: "M190 38 L190 52 L102 52 L102 40", class: "fl-line", "marker-end": "url(#fl-head)"}),
      s("text", {x: 4, y: 84, class: "fl-note"}, "곱마다 IEEE FP32 반올림 1회 (CUDA core SGEMM과 같은 순서)"));
    return svg;
  }
  add(products(4, 14, 1, "곱 1개"), arrow(16, 26, 60, 26), box(60, 14, 84, "FMA (FP64)", true), arrow(144, 26, 170, 26),
    box(170, 14, 54, "FP32로"), s("text", {x: 4, y: 84, class: "fl-note"}, "binary64로 누산 후 FP32로 한 번 반올림 (수치 기준)"));
  return svg;
}
