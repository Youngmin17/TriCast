// DOM helpers, number formatting and the shared tooltip.
const SVG_NS = "http://www.w3.org/2000/svg";

function assign(el, props) {
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") el.setAttribute("class", value);
    else if (key === "text") el.textContent = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
}

function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : String(kid));
  }
  return el;
}

export function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  assign(el, props);
  return append(el, kids);
}

export function s(tag, props, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  assign(el, props);
  return append(el, kids);
}

export const fmt = {
  pct: (x, digits = 1) => (x == null ? "—" : `${(x * 100).toFixed(digits)}%`),
  num: (x, digits = 3) => (x == null ? "—" : Number(x).toFixed(digits)),
  kl: (x) => {
    if (x == null) return "—";
    if (x === 0) return "0";
    return Math.abs(x) >= 0.01 ? x.toFixed(3) : x.toExponential(1);
  },
  sec: (x) => (x == null ? "—" : `${Number(x).toFixed(1)}초`),
  short: (sha) => (sha ? String(sha).slice(0, 7) : "—"),
};

const tip = () => document.getElementById("tip");

export function showTip(target, content) {
  const el = tip();
  el.replaceChildren(content);
  el.hidden = false;
  const box = target.getBoundingClientRect();
  const width = el.offsetWidth;
  const height = el.offsetHeight;
  const left = Math.min(Math.max(8, box.left + box.width / 2 - width / 2), window.innerWidth - width - 8);
  const above = box.top - height - 8;
  el.style.left = `${left}px`;
  el.style.top = `${above > 8 ? above : box.bottom + 8}px`;
}

export function hideTip() {
  tip().hidden = true;
}

// Attach a tooltip that follows hover and keyboard focus.
export function withTip(el, build) {
  const open = () => showTip(el, build());
  el.addEventListener("mouseenter", open);
  el.addEventListener("focus", open);
  el.addEventListener("mouseleave", hideTip);
  el.addEventListener("blur", hideTip);
  return el;
}

// A group of mutually exclusive buttons (role=radio) with arrow-key movement.
export function radioGroup(label, items, current, onPick, className) {
  const group = h("div", {class: className, role: "radiogroup", "aria-label": label});
  for (const item of items) {
    const button = item.render();
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(item.id === current));
    button.tabIndex = item.id === current ? 0 : -1;
    if (item.disabled) button.disabled = true;
    button.addEventListener("click", () => onPick(item.id));
    group.append(button);
  }
  group.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
    const buttons = [...group.querySelectorAll("[role=radio]:not(:disabled)")];
    const index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    event.preventDefault();
    const step = event.key === "ArrowDown" || event.key === "ArrowRight" ? 1 : -1;
    buttons[(index + step + buttons.length) % buttons.length].click();
  });
  return group;
}

export async function copyText(button, text) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "복사됨";
  } catch {
    button.textContent = "복사하지 못했습니다. 직접 선택하세요";
  }
  setTimeout(() => (button.textContent = "복사"), 1600);
}
