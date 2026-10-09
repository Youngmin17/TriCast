// The datapath glyph: one cell per significand bit of the accumulator after alignment.
// Filled = bits the chip keeps (F), faded = bits beyond the FP32 fraction, outlined = bits it truncates.
// The tick marks the 23-bit FP32 fraction.
import {s} from "./ui.js";

const CELLS = 28;
const FP32_FRACTION = 23;

export function keptBits(mma) {
  if (mma.algorithm === "cofda" || mma.algorithm === "gdfs") return mma.f_bits;
  if (mma.algorithm === "fp32_fma") return FP32_FRACTION;
  return Infinity;
}

export function datapath(mma, width = 168) {
  const height = 12;
  const cell = width / CELLS;
  const kept = keptBits(mma);
  const svg = s("svg", {class: "datapath", width, height: height + 6, viewBox: `0 0 ${width} ${height + 6}`,
    "aria-hidden": "true"});
  for (let i = 0; i < CELLS; i += 1) {
    const kind = i < Math.min(kept, FP32_FRACTION) ? "kept" : i < kept ? "wide" : "cut";
    svg.append(s("rect", {class: kind, x: i * cell + 0.7, y: 3, width: cell - 1.8, height, rx: 1}));
  }
  const edge = FP32_FRACTION * cell - 0.25;
  svg.append(s("line", {class: "edge", x1: edge, x2: edge, y1: 0, y2: height + 6}));
  return svg;
}

export function mmaSummary(mma) {
  if (mma.algorithm === "fp64") return "FP64 정확 누산";
  if (mma.algorithm === "fp32_fma") return "FP32 FMA (IEEE)";
  if (mma.algorithm === "int_exact") return "정수 정확 누산";
  const parts = [mma.algorithm === "gdfs" ? "GDFS" : "CoFDA", `F${mma.f_bits}`];
  if (mma.algorithm === "gdfs") parts.push(`G${mma.g_bits}`, `GS${mma.group_size}`);
  else parts.push(`CS${mma.chunk_size}`, mma.c_mode === "decoupled" ? `decoupled F2=${mma.f2_bits}` : "fused");
  if (mma.promote_interval) parts.push(`FP32 승격 ${mma.promote_interval}`);
  return parts.join(" · ");
}

export function keptLabel(mma) {
  const kept = keptBits(mma);
  if (!Number.isFinite(kept)) return "정렬 절단 없음";
  if (mma.algorithm === "gdfs") {
    return `그룹 안에서 ${mma.g_bits}비트로 정렬해 합하고, 그룹 결과를 F=${kept}비트로 다시 정렬 (FP32 가수는 23비트)`;
  }
  return `정렬 뒤 가수 ${kept}비트 유지 (FP32 가수는 23비트)`;
}
