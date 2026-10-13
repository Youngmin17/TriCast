// Quantizer demo: pick one quantization method per Linear layer of Qwen3-0.6B on a kind × block grid, set each
// method's options and the calibration data, and get the TriCast recipe. The recipe part is a line-for-line port of
// app/quantizer_recipe.py (same rule order, same YAML bytes). demo/quantizer/presets.json
// (app/export_quantizer_demo.py) supplies the method catalog, the option tables, the presets and the recorded runs
// that belong to them.
import {h, hideTip, radioGroup, showTip, withTip} from "./ui.js";

const SCHEMA = "tricast.quantizer_demo/1";
const DATA = "demo/quantizer/presets.json";
const EXPORT = "python -m app.export_quantizer_demo --runs runs/<실행> "
  + "--native runs/<실행>/ppl_base/native.json --out app/web/demo/quantizer";
const NEW_NAME = "my_recipe";
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

// ---------- recipe: port of app/quantizer_recipe.py ----------
// Only the algorithm is ported. The tables (method fragments, KV presets, kinds, ...) come from the Python
// module through presets.json "generator", so a change there reaches the page with the next export.
let KINDS = [];
let PARENT = {};
let BLOCKS = 0;
let METHODS = {};   // key order = YAML key order, as in the Python dicts
let CALIBRATED = new Set();
let KV = {};
let DESCRIPTION = "";
let OPTIONS = {};          // option id -> {type, choices | min/max, default, when}
let METHOD_OPTIONS = {};   // method id -> option ids, in the order they are written
let KV_OPTIONS = {};       // KV preset -> {field: spec}
let CALIBRATION = {};      // calibration key -> spec (key order = YAML order)
let COMPLETE = {};         // transform / weight_algo -> every field and its default, in dataclass order
let CALIBRATING_OBSERVERS = new Set();
let FLOAT_MIN = 0;

// presets.json writes mappings as {"items": [[key, value], ...]} so their key order survives sort_keys.
const ordered = (value) => (Array.isArray(value) ? value.map(ordered)
  : value && typeof value === "object" ? Object.fromEntries(value.items.map(([k, v]) => [k, ordered(v)])) : value);

function useGenerator(generator) {
  KINDS = [...generator.kinds];
  PARENT = ordered(generator.parent);
  BLOCKS = generator.blocks;
  METHODS = ordered(generator.methods);
  CALIBRATED = new Set(generator.calibrated);
  KV = ordered(generator.kv);
  DESCRIPTION = generator.description;
  OPTIONS = ordered(generator.options);
  METHOD_OPTIONS = ordered(generator.method_options);
  KV_OPTIONS = ordered(generator.kv_options);
  CALIBRATION = ordered(generator.calibration);
  COMPLETE = ordered(generator.complete);
  CALIBRATING_OBSERVERS = new Set(generator.calibrating_observers);
  FLOAT_MIN = generator.float_min;
}

const has = (table, key) => typeof key === "string" && Object.hasOwn(table, key);
const isDict = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
// Canonical JSON (sorted keys), the stand-in for Python's json.dumps(..., sort_keys=True) as a lookup key.
const canon = (value) => JSON.stringify(value, (key, item) => (isDict(item)
  ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) : item));

const layerName = (block, kind) => `model.layers.${block}.${PARENT[kind]}.${kind}`;

function blocksText(blocks) {
  const parts = [];
  let start = null;
  let prev = null;
  for (const block of [...blocks].sort((a, b) => a - b)) {
    if (start === null) start = prev = block;
    else if (block === prev + 1) prev = block;
    else {
      parts.push(start === prev ? String(start) : `${start}-${prev}`);
      start = prev = block;
    }
  }
  if (start !== null) parts.push(start === prev ? String(start) : `${start}-${prev}`);
  return parts.join(",");
}

function expandBlocks(text) {
  const out = new Set();
  for (const part of text.split(",")) {
    const cut = part.indexOf("-");
    const lo = cut < 0 ? part : part.slice(0, cut);
    const hi = cut < 0 ? "" : part.slice(cut + 1);
    for (let block = Number(lo); block <= Number(hi || lo); block += 1) out.add(block);
  }
  return out;
}

// ---------- options (port of check_value ... _kv_fields) ----------
const pyList = (values) => `[${values.map((v) => (typeof v === "string" ? `'${v}'` : String(v))).join(", ")}]`;

// The value if spec (an OPTIONS / KV_OPTIONS / CALIBRATION entry) allows it; numbers that are integral print as
// ints in both languages.
function checkValue(where, spec, value) {
  if (spec.type === "choice") {
    if (!spec.choices.some((choice) => choice === value)) throw new Error(`${where}: ${value} is not one of ${pyList(spec.choices)}`);
    return value;
  }
  if (spec.type === "bool") {
    if (typeof value !== "boolean") throw new Error(`${where}: expected true or false, got ${value}`);
    return value;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || (spec.type === "int" && !Number.isInteger(value))) {
    throw new Error(`${where}: expected ${spec.type === "int" ? "an integer" : "a number"}, got ${value}`);
  }
  if (spec.type === "float" && value !== 0 && Math.abs(value) < FLOAT_MIN) {
    throw new Error(`${where}: use 0 or a magnitude of at least ${FLOAT_MIN}, got ${value}`);
  }
  if (value < spec.min || (value === spec.min && spec.min_open) || value > spec.max || (value === spec.max && spec.max_open)) {
    throw new Error(`${where}: ${value} is outside ${spec.min_open ? "(" : "["}${spec.min}, ${spec.max}${spec.max_open ? ")" : "]"}`);
  }
  if (spec.multiple && value % spec.multiple) throw new Error(`${where}: ${value} is not a multiple of ${spec.multiple}`);
  return value;
}

// Whether an option takes effect given the other chosen options (unset ones are at their default).
function applies(option, chosen) {
  const when = OPTIONS[option].when;
  if (!when) return true;
  const [parent, values] = when;
  const value = applies(parent, chosen) && Object.hasOwn(chosen, parent) ? chosen[parent] : OPTIONS[parent].default;
  return values.includes(value);
}

// The checked options of one method that differ from their defaults; one that does not apply must be at its default.
function methodOptions(method, options) {
  const given = {...(options || {})};
  const allowed = METHOD_OPTIONS[method];
  const unknown = Object.keys(given).filter((option) => !allowed.includes(option)).sort();
  if (unknown.length) throw new Error(`${method}: unknown options ${pyList(unknown)}`);
  const chosen = {};
  for (const option of allowed) if (Object.hasOwn(given, option)) chosen[option] = checkValue(`${method}.${option}`, OPTIONS[option], given[option]);
  const out = {};
  for (const [option, value] of Object.entries(chosen)) {
    if (value === OPTIONS[option].default) continue;
    if (!applies(option, chosen)) throw new Error(`${method}.${option}: applies only when ${OPTIONS[option].when[0]} is one of ${pyList(OPTIONS[option].when[1])}`);
    out[option] = value;
  }
  return out;
}

function checkOptions(options) {
  if (options == null) return {};
  if (!isDict(options)) throw new Error("options must map method ids to {option id: value}");
  const unknown = Object.keys(options).filter((method) => !has(METHODS, method)).sort();
  if (unknown.length) throw new Error(`options for unknown methods ${pyList(unknown)}`);
  return Object.fromEntries(Object.entries(options).map(([method, value]) => [method, methodOptions(method, value)]));
}

function needsCalibration(method, options) {
  const chosen = methodOptions(method, options);
  return CALIBRATED.has(method) || CALIBRATING_OBSERVERS.has(chosen["activation.observer.kind"]);
}

function complete(key, value) {
  const fields = typeof value === "string" ? {kind: value} : value;
  return {kind: fields.kind, ...Object.fromEntries(Object.entries(COMPLETE[key]).map(([name, d]) => [name, Object.hasOwn(fields, name) ? fields[name] : d]))};
}

const modified = (chosen) => new Set(Object.keys(chosen).map((option) => option.split(".")[0]).filter((key) => has(COMPLETE, key)));

// METHODS[method] with the options written in: operand options inside the scheme mapping (mma_input last), transform /
// weight_algo as complete mappings.
function variantOf(method, chosen, completeKeys) {
  const base = METHODS[method];
  if (base === null) return null;
  const fragment = structuredClone(base);
  for (const key of new Set([...completeKeys, ...modified(chosen)])) fragment[key] = complete(key, fragment[key]);
  for (const [option, value] of Object.entries(chosen)) {
    const [target, ...path] = option.split(".");
    if (has(COMPLETE, target)) {
      fragment[target][path[0]] = value;
      continue;
    }
    const operand = fragment[target];
    const mapping = typeof operand === "string" ? {scheme: operand} : {...operand};
    const tail = Object.hasOwn(mapping, "mma_input") ? {mma_input: mapping.mma_input} : {};
    delete mapping.mma_input;
    if (path[0] === "observer" && path[1] === "kind") mapping.observer = {kind: value};
    else if (path.length === 2) (mapping[path[0]] ??= {})[path[1]] = value;
    else mapping[path[0]] = value;
    fragment[target] = {...mapping, ...tail};
  }
  return fragment;
}

function variant(method, options) {
  const chosen = methodOptions(method, options);
  return variantOf(method, chosen, modified(chosen));
}

function fragments(def, options) {
  const checked = checkOptions(options);
  const completeKeys = modified(checked[def] ?? {});
  return Object.fromEntries(Object.keys(METHODS).map((method) => [method, variantOf(method, checked[method] ?? {}, completeKeys)]));
}

function checkCalibration(calibration) {
  const given = {...(calibration || {})};
  const unknown = Object.keys(given).filter((key) => !has(CALIBRATION, key)).sort();
  if (unknown.length) throw new Error(`calibration: unknown keys ${pyList(unknown)}`);
  return Object.fromEntries(Object.entries(CALIBRATION).map(([key, spec]) => [key,
    Object.hasOwn(given, key) ? checkValue(`calibration.${key}`, spec, given[key]) : spec.default]));
}

function kvFields(kv, kvOptions) {
  if (!has(KV, kv)) throw new Error(`unknown KV preset ${kv}; known: ${Object.keys(KV).sort().join(", ")}`);
  const given = {...(kvOptions || {})};
  const unknown = Object.keys(given).filter((key) => !has(KV_OPTIONS[kv], key)).sort();
  if (unknown.length) throw new Error(`kv ${kv}: unknown options ${pyList(unknown)}`);
  if (KV[kv] === null) return null;
  const fields = {...KV[kv]};
  for (const [key, spec] of Object.entries(KV_OPTIONS[kv])) {
    if (!Object.hasOwn(given, key)) continue;
    const value = checkValue(`kv.${key}`, spec, given[key]);
    if (value !== spec.default) fields[key] = value;
  }
  return fields;
}

function rule(selector, method, table) {
  const fields = table[method];
  return fields === null ? {...selector, skip: true} : {...selector, ...fields};
}

// Minimal first-match rules that reproduce the grid on top of the default: single cells, then whole blocks,
// then whole kinds.
function overridesFor(def, grid, table = METHODS) {
  const covered = new Map();
  const at = (b, k) => `${b},${k}`;
  const blockRules = new Map();
  grid.forEach((row, block) => {
    if (new Set(row).size === 1 && row[0] !== def) {
      if (!blockRules.has(row[0])) blockRules.set(row[0], []);
      blockRules.get(row[0]).push(block);
      for (let k = 0; k < KINDS.length; k += 1) covered.set(at(block, k), row[0]);
    }
  });
  const kindRules = [];
  for (let k = 0; k < KINDS.length; k += 1) {
    const rest = new Set();
    for (let b = 0; b < grid.length; b += 1) if (!covered.has(at(b, k))) rest.add(grid[b][k]);
    if (rest.size !== 1) continue;
    const method = [...rest][0];
    if (method === def) continue;
    kindRules.push([k, method]);
    for (let b = 0; b < grid.length; b += 1) if (!covered.has(at(b, k))) covered.set(at(b, k), method);
  }
  const rules = [];
  for (let b = 0; b < grid.length; b += 1) {
    for (let k = 0; k < KINDS.length; k += 1) {
      if (grid[b][k] !== (covered.get(at(b, k)) ?? def)) rules.push(rule({match: layerName(b, KINDS[k])}, grid[b][k], table));
    }
  }
  for (const [method, blocks] of [...blockRules].sort((x, y) => Math.min(...x[1]) - Math.min(...y[1]))) {
    rules.push(rule({layers: blocksText(blocks)}, method, table));
  }
  for (const [k, method] of kindRules) rules.push(rule({match: `*.${PARENT[KINDS[k]]}.${KINDS[k]}`}, method, table));
  return rules;
}

// Replay first-match override resolution to the method each layer gets.
function resolve(def, rules, blocks = BLOCKS, table = METHODS) {
  const byFields = new Map(Object.entries(table).filter(([, fields]) => fields).map(([m, fields]) => [canon(fields), m]));
  if (byFields.size !== Object.values(table).filter(Boolean).length) throw new Error("two methods have the same recipe fields");
  const grid = [];
  for (let block = 0; block < blocks; block += 1) {
    const row = [];
    for (const kind of KINDS) {
      const name = layerName(block, kind);
      let method = def;
      for (const item of rules) {
        let hit;
        if ("match" in item) {
          const pattern = item.match;
          hit = name === pattern || (pattern.startsWith("*.") && name.endsWith(pattern.slice(1)));
        } else hit = expandBlocks(item.layers).has(block);
        if (hit) {
          const fields = Object.fromEntries(Object.entries(item).filter(([key]) => key !== "match" && key !== "layers"));
          method = fields.skip ? "bf16" : byFields.get(canon(fields));
          break;
        }
      }
      row.push(method);
    }
    grid.push(row);
  }
  return grid;
}

// extra = {options, calibration, kv_options}: the keyword arguments of the Python build().
function build(name, def, grid, kv = "none", description = DESCRIPTION, {options = null, calibration = null, kv_options: kvOptions = null} = {}) {
  if (!has(METHODS, def) || def === "bf16") throw new Error(`default method must be a quantization method, got ${def}`);
  if (!Array.isArray(grid) || grid.length !== BLOCKS || grid.some((row) => !Array.isArray(row) || row.length !== KINDS.length)) {
    throw new Error(`grid must be ${BLOCKS} blocks × ${KINDS.length} kinds`);
  }
  const unknown = [...new Set(grid.flat())].filter((m) => !has(METHODS, m)).sort();
  if (unknown.length) throw new Error(`unknown methods ${unknown.join(", ")}`);
  const kvPart = kvFields(kv, kvOptions);
  const checked = checkOptions(options);
  const settings = checkCalibration(calibration);
  const table = fragments(def, checked);
  const rules = overridesFor(def, grid, table);
  if (canon(resolve(def, rules, BLOCKS, table)) !== canon(grid)) throw new Error("generated rules do not reproduce the grid");
  const used = new Set([def, ...grid.flat()]);
  const recipe = {name, description, defaults: {...(table[def] || {})}};
  if (rules.length) recipe.overrides = rules;
  if (kvPart !== null) recipe.kv = kvPart;
  recipe.exclude = ["lm_head"];
  const changed = new Set(Object.keys(settings).filter((key) => settings[key] !== CALIBRATION[key].default));
  if (changed.size || [...used].some((m) => needsCalibration(m, checked[m]))) {
    recipe.calibration = Object.fromEntries(Object.entries(settings).filter(([key]) => changed.has(key) || key === "seed"));
  }
  return recipe;
}

// Python's str.splitlines (no trailing empty piece).
function splitlines(text) {
  const lines = text.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/u);
  if (lines.length && lines.at(-1) === "" && text.length) lines.pop();
  return text.length ? lines : [];
}

const RESERVED = ["null", "true", "false", "none", "yes", "no", "on", "off"];
const ALPHA = /^\p{L}$/u;
const ALNUM = /^[\p{L}\p{N}]$/u;

function scalar(item) {
  if (item === null || item === undefined) return "null";
  if (typeof item === "boolean") return item ? "true" : "false";
  // = Python repr: recipe numbers are ints or floats that are 0 or ≥ FLOAT_MIN and < 1e16 (no exponent), and
  // integral floats arrive here as ints (checkValue)
  if (typeof item === "number") return String(item);
  if (Array.isArray(item)) return `[${item.map(scalar).join(", ")}]`;
  if (isDict(item)) return "{}";
  const text = String(item);
  const chars = [...text];
  const plain = chars.length > 0 && ALPHA.test(chars[0]) && chars.every((c) => ALNUM.test(c) || "_-.".includes(c))
    && !RESERVED.includes(text);
  return plain || text === "none" ? text : JSON.stringify(text);  // = json.dumps(text, ensure_ascii=False)
}

// Deterministic YAML for the recipe subset (mappings, lists of mappings, scalars).
function toYaml(value, indent = 0) {
  const pad = " ".repeat(indent);
  const lines = [];
  for (const [key, item] of Object.entries(value)) {
    if (isDict(item) && Object.keys(item).length) {
      lines.push(`${pad}${key}:`);
      lines.push(toYaml(item, indent + 2));
    } else if (Array.isArray(item) && item.length && item.every(isDict)) {
      lines.push(`${pad}${key}:`);
      for (const entry of item) {
        const body = splitlines(toYaml(entry, indent + 4));
        lines.push(`${pad}  - ${body[0].replace(/^\s+/u, "")}`);
        lines.push(...body.slice(1));
      }
    } else lines.push(`${pad}${key}: ${scalar(item)}`);
  }
  return lines.join("\n");
}

const uniform = (method) => Array.from({length: BLOCKS}, () => KINDS.map(() => method));
const copyGrid = (grid) => grid.map((row) => [...row]);

// ---------- page state ----------
const root = document.getElementById("qz");
let data = null;       // presets.json
let byId = null;       // method id -> catalog entry
let palette = [];      // catalog order; digits 1-9 then 0 apply these
let trusted = true;    // every preset's YAML here equals the exporter's Python YAML
// options: method -> {option id: value} as the controls last held them (every value passed checkValue; one that
// does not apply stays stored and comes back when it applies again); calib: calibration key -> value;
// kvOptions: KV preset -> {field: value}. merge: the control whose typing is coalesced into one undo step.
const state = {name: NEW_NAME, nameAuto: true, loaded: null, def: "nvfp4", grid: [], kv: "none",
  options: {}, calib: {}, kvOptions: {}, sel: new Set(), anchor: null, focus: [0, 0], history: [], note: "", merge: null};
const ui = {invalid: new Map()};  // invalid: control id -> reason, while a control holds a value build() refuses
const key = (b, k) => b * KINDS.length + k;
const posOf = (cell) => [Number(cell.dataset.b), Number(cell.dataset.k)];

// What build() gets from a state: per method the options that apply and differ from their default, the changed
// calibration values and the current KV preset's changed fields.
function normalizeOptions(method, raw = {}) {
  const out = {};
  for (const option of METHOD_OPTIONS[method] ?? []) {
    if (Object.hasOwn(raw, option) && raw[option] !== OPTIONS[option].default && applies(option, raw)) out[option] = raw[option];
  }
  return out;
}
const pageOptions = (s = state) => Object.fromEntries(Object.keys(METHODS).map((m) => [m, normalizeOptions(m, s.options[m])])
  .filter(([, chosen]) => Object.keys(chosen).length));
const pageCalibration = (s = state) => Object.fromEntries(Object.entries(s.calib).filter(([k, v]) => v !== CALIBRATION[k].default));
const pageKvOptions = (s = state) => Object.fromEntries(Object.entries(s.kvOptions[s.kv] ?? {})
  .filter(([k, v]) => v !== KV_OPTIONS[s.kv]?.[k]?.default));
const extra = (s = state) => ({options: pageOptions(s), calibration: pageCalibration(s), kv_options: pageKvOptions(s)});
const presetState = (p) => ({def: p.default, grid: p.grid, kv: p.kv, options: {}, calib: {}, kvOptions: {}});

const snapshot = () => ({def: state.def, grid: copyGrid(state.grid), kv: state.kv, name: state.name,
  nameAuto: state.nameAuto, loaded: state.loaded, options: structuredClone(state.options), calib: {...state.calib},
  kvOptions: structuredClone(state.kvOptions)});
const same = (a, b) => a.def === b.def && a.kv === b.kv && canon(a.grid) === canon(b.grid) && canon(extra(a)) === canon(extra(b));
const matchedPreset = () => data.presets.find((p) => same(presetState(p), state)) ?? null;

function autoName() {
  if (!state.nameAuto) return;
  const loaded = data.presets.find((p) => p.id === state.loaded);
  state.name = loaded && same(presetState(loaded), state) ? loaded.id : NEW_NAME;
}

// Every change to the recipe goes through here so that ⌘/Ctrl+Z can take it back. Changes with the same merge key
// in a row (typing into one number field) are one undo step.
function commit(change, note, merge = null) {
  const before = snapshot();
  change();
  autoName();
  const after = snapshot();
  if (same(before, after) && before.name === after.name && before.loaded === after.loaded
      && canon([before.options, before.calib, before.kvOptions]) === canon([after.options, after.calib, after.kvOptions])) {
    update();
    return false;
  }
  if (!merge || state.merge !== merge) {
    state.history.push(before);
    if (state.history.length > 200) state.history.shift();
  }
  state.merge = merge;
  state.note = note;
  update();
  return true;
}

function undo() {
  const previous = state.history.pop();
  if (!previous) {
    announce("되돌릴 변경이 없습니다");
    return;
  }
  Object.assign(state, {...previous, grid: copyGrid(previous.grid)});
  state.merge = null;
  state.note = "마지막 변경을 되돌렸습니다";
  update();
}

// A preset or a new recipe is a whole recipe: options, calibration and KV fields go back to their defaults.
function loadPreset(preset) {
  commit(() => {
    Object.assign(state, {def: preset.default, grid: copyGrid(preset.grid), kv: preset.kv, loaded: preset.id,
      nameAuto: true, options: {}, calib: {}, kvOptions: {}});
    state.sel.clear();
  }, `프리셋 ${preset.id} 를 불러왔습니다`);
}

function newRecipe() {
  commit(() => {
    Object.assign(state, {def: "nvfp4", grid: uniform("nvfp4"), kv: "none", loaded: null, nameAuto: true,
      options: {}, calib: {}, kvOptions: {}});
    state.sel.clear();
  }, "새 레시피: 모든 레이어가 NVFP4 입니다");
}

// Cells that showed the old default follow the new one; cells set to another method keep it.
function setDefault(method) {
  const old = state.def;
  commit(() => {
    state.def = method;
    state.grid = state.grid.map((row) => row.map((m) => (m === old ? method : m)));
  }, `기본 방법을 ${byId[method].label} 로 바꿨습니다`);
}

function apply(method, cells) {
  if (!cells.length) {
    announce("먼저 칸을 고르세요");
    return;
  }
  commit(() => {
    for (const c of cells) state.grid[Math.floor(c / KINDS.length)][c % KINDS.length] = method;
  }, `${cells.length}칸에 ${byId[method].label} 적용`);
}

const rowCells = (k) => Array.from({length: BLOCKS}, (_, b) => key(b, k));
const colCells = (b) => KINDS.map((_, k) => key(b, k));
function rect([b0, k0], [b1, k1]) {
  const out = [];
  for (let b = Math.min(b0, b1); b <= Math.max(b0, b1); b += 1) {
    for (let k = Math.min(k0, k1); k <= Math.max(k0, k1); k += 1) out.push(key(b, k));
  }
  return out;
}

function select(cells, base = new Set()) {
  state.sel = new Set([...base, ...cells]);
  updateGrid();
}

function clearSelection() {
  if (!state.sel.size) return;
  state.sel.clear();
  state.note = "선택을 해제했습니다";
  updateGrid();
}

// ---------- formatting ----------
const fixed = (v, d) => (v == null ? "—" : Number(v).toFixed(d));
const signed = (v, d) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`);
const count = (n) => Number(n).toLocaleString("ko-KR");
const millions = (n) => `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
const minutes = (sec) => (sec == null ? "—" : sec < 90 ? `${Math.round(sec)}초` : `${(sec / 60).toFixed(1)}분`);
const token = (v) => (v === "inf" ? "∞" : v === "-inf" ? "−∞" : v === "nan" ? "NaN" : null);
const code = (method) => h("span", {class: "qz-code", "data-m": method, "aria-hidden": "true"}, byId[method].code);

function announce(text) {
  state.note = text;
  if (ui.status) ui.status.textContent = text;
  if (ui.live) ui.live.textContent = text;
}

async function copy(button, text) {
  const label = button.dataset.label;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "복사됨";
  } catch {
    button.textContent = "복사하지 못했습니다 · 직접 선택하세요";
  }
  setTimeout(() => (button.textContent = label), 1600);
}

const copyButton = (label, text) => {
  const button = h("button", {type: "button", class: "copy", "data-label": label}, label);
  button.addEventListener("click", () => copy(button, typeof text === "function" ? text() : text));
  return button;
};

// ---------- config column ----------
const step = (no, title, hint, ...body) => h("section", {class: "panel qz-step", "aria-labelledby": `qz-step-${no}`},
  h("div", {class: "step-head"}, h("span", {class: "step-no"}, no), h("h2", {class: "step-title", id: `qz-step-${no}`}, title),
    hint ? h("span", {class: "step-hint"}, hint) : null), ...body);

function presetStep() {
  ui.presets = data.presets.map((preset) => {
    const recorded = preset.run || preset.report;
    const button = h("button", {type: "button", class: "qz-preset", "aria-pressed": "false", "data-preset": preset.id,
      onclick: () => loadPreset(preset)},
    h("span", {class: "qz-preset-label"}, preset.label),
    h("span", {class: "qz-preset-sub"}, h("span", {class: recorded ? "qz-dot on" : "qz-dot"}), preset.id,
      recorded ? " · 기록 있음" : " · 기록 없음"));
    return button;
  });
  const fresh = h("button", {type: "button", class: "qz-preset qz-new", "data-preset": "new", onclick: newRecipe},
    h("span", {class: "qz-preset-label"}, "새 레시피"), h("span", {class: "qz-preset-sub"}, "모든 레이어 NVFP4 에서 시작"));
  return step("1", "시작점", "프리셋을 고르면 아래 설정이 모두 바뀝니다",
    h("div", {class: "qz-presets", role: "group", "aria-label": "시작점"}, ui.presets, fresh));
}

// ---------- option controls ----------
// One control per method option, calibration key and KV field. A typed value build() would refuse never reaches the
// state: the control says why in one line, download and server runs wait, and leaving the control restores the last
// good value. A control whose option does not apply is disabled with its reason, or hidden when it is a sub-field.
const fields = [];

function valueProblem(spec, value) {
  if (spec.type === "choice" || spec.type === "bool") return null;
  if (typeof value !== "number" || !Number.isFinite(value) || (spec.type === "int" && !Number.isInteger(value))) {
    return spec.type === "int" ? "정수를 넣으세요" : "수를 넣으세요";
  }
  if (spec.type === "float" && value !== 0 && Math.abs(value) < FLOAT_MIN) return `0 이거나 ${FLOAT_MIN} 이상이어야 합니다`;
  if (value < spec.min || (value === spec.min && spec.min_open) || value > spec.max || (value === spec.max && spec.max_open)) {
    const end = spec.type === "int" ? "의 정수여야" : spec.max_open ? "이어야" : "여야";
    return `${spec.min}${spec.min_open ? " 초과" : " 이상"} ${spec.max}${spec.max_open ? " 미만" : " 이하"}${end} 합니다`;
  }
  if (spec.multiple && value % spec.multiple) return `${spec.multiple}의 배수여야 합니다`;
  return null;
}

function control({id, spec, label, words = null, value, set, applicable = () => true, unavailable = null}) {
  const fid = `qz-f-${id.replaceAll(".", "-")}`;
  const numeric = spec.type === "int" || spec.type === "float";
  const input = spec.type === "choice"
    ? h("select", {id: fid, class: "qz-select"}, spec.choices.map((c) => h("option", {value: String(c)}, words?.[String(c)] ?? String(c))))
    : spec.type === "bool" ? h("input", {type: "checkbox", id: fid})
      : h("input", {type: "number", id: fid, class: "qz-num", min: String(spec.min), max: String(spec.max),
        step: spec.type === "int" ? String(spec.multiple ?? 1) : "any", inputmode: spec.type === "int" ? "numeric" : "decimal"});
  const why = h("p", {class: "qz-why", id: `${fid}-why`});
  input.setAttribute("aria-describedby", why.id);
  input.dataset.field = id;
  const row = spec.type === "bool"
    ? h("div", {class: "qz-field qz-check", "data-row": id}, h("label", {for: fid}, input, h("span", null, label)), why)
    : h("div", {class: "qz-field", "data-row": id}, h("label", {for: fid}, label), input, why);
  const entry = {id, spec, input, row, why, value, applicable, unavailable};
  const read = () => (spec.type === "choice" ? spec.choices.find((c) => String(c) === input.value)
    : spec.type === "bool" ? input.checked : input.value.trim() === "" ? Number.NaN : Number(input.value));
  input.addEventListener(numeric ? "input" : "change", () => {
    const next = read();
    const problem = valueProblem(spec, next);
    if (problem) ui.invalid.set(id, `${label}: ${problem}`);
    else ui.invalid.delete(id);
    if (problem) {
      paint(entry);
      updateBlocked();
    } else set(next, numeric ? id : null);
  });
  input.addEventListener("blur", () => {
    state.merge = null;
    if (ui.invalid.delete(id)) {
      announce(`${label}: 올바르지 않은 값이라 마지막 값으로 되돌렸습니다`);
      updateBlocked();
    }
    paint(entry);
  });
  fields.push(entry);
  return entry;
}

function paint(entry) {
  const on = entry.applicable();
  const problem = ui.invalid.get(entry.id);
  entry.row.hidden = !on && !entry.unavailable;
  entry.row.classList.toggle("off", !on);
  entry.input.disabled = !on;
  const shown = on ? entry.value() : entry.spec.default;
  if (entry.spec.type === "bool") entry.input.checked = shown;
  else if (!(problem || (document.activeElement === entry.input && entry.spec.type !== "choice"))) entry.input.value = String(shown);
  entry.input.setAttribute("aria-invalid", String(Boolean(problem)));
  entry.why.textContent = problem ? problem.slice(problem.indexOf(": ") + 2) : on ? "" : entry.unavailable ?? "";
  entry.why.classList.toggle("warn", Boolean(problem));
}

// The "옵션" disclosure under a method card (and the KV step): collapsed until opened.
function disclosure(id, label, panel) {
  const toggle = h("button", {type: "button", class: "qz-opt-toggle", "aria-expanded": "false", "aria-controls": id, "data-options": label},
    h("span", {class: "qz-chev", "aria-hidden": "true"}), h("span", null, "옵션"), h("span", {class: "qz-opt-n", "data-optn": label}));
  panel.id = id;
  panel.hidden = true;
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    panel.hidden = !open;
  });
  return toggle;
}

const optDot = (owner) => h("span", {class: "qz-optdot", "data-optdot": owner, hidden: true, title: "옵션을 바꿨습니다"},
  h("span", {class: "sr-only"}, "옵션 바꿈"));
const OPTION_GROUPS = [["weight.", "가중치"], ["activation.", "활성"], ["transform.", "변환 (보정 데이터로 맞춤)"], ["weight_algo.", "GPTQ"]];

function optionsPanel(method) {
  const m = byId[method];
  const entries = METHOD_OPTIONS[method].map((option) => {
    const info = data.option_info[option];
    return control({id: `${method}.${option}`, spec: OPTIONS[option], label: info.label, words: info.choices,
      unavailable: info.unavailable, value: () => state.options[method]?.[option] ?? OPTIONS[option].default,
      applicable: () => applies(option, state.options[method] ?? {}),
      set: (value, merge) => commit(() => ((state.options[method] ??= {})[option] = value),
        `${m.label} · ${info.label}: ${info.choices?.[String(value)] ?? value}`, merge)});
  });
  const groups = OPTION_GROUPS.map(([prefix, title]) => {
    const rows = entries.filter((entry) => entry.id.startsWith(`${method}.${prefix}`)).map((entry) => entry.row);
    // inside the group the label drops the group's own word ("가중치 원소 반올림" -> "원소 반올림"); the legend names it
    const word = `${title.split(" ")[0]} `;
    for (const row of rows) for (const label of row.querySelectorAll("label, label > span")) {
      if (label.firstChild?.nodeType === Node.TEXT_NODE && label.firstChild.data.startsWith(word)) label.firstChild.data = label.firstChild.data.slice(word.length);
    }
    return rows.length ? h("fieldset", {class: "qz-optgroup"}, h("legend", null, title), rows) : null;
  });
  const reset = h("button", {type: "button", class: "qz-quick qz-opt-reset", "data-reset": method,
    onclick: () => commit(() => delete state.options[method], `${m.label} 옵션을 기본값으로 되돌렸습니다`)}, "기본값으로");
  return h("div", {class: "qz-opt-panel", role: "group", "aria-label": `${m.label} 옵션`},
    m.fixed ? h("p", {class: "qz-fixed"}, h("b", null, "고정 "), m.fixed) : null, groups,
    h("p", {class: "qz-opt-foot"}, "이 방법을 쓰는 모든 레이어에 적용", reset));
}

function methodCard(method) {
  const m = byId[method];
  return h("button", {type: "button", class: "option qz-card", "data-default": method},
    h("span", {class: "qz-card-head"}, code(method), h("span", {class: "name"}, m.label), optDot(method)),
    h("span", {class: "qz-bits"}, m.wa_bits),
    h("span", {class: "sub"}, m.description),
    h("span", {class: "qz-tags"}, h("span", {class: "badge qz-calib", "data-calib": method, hidden: !m.calibration}, "보정 필요"),
      m.note ? h("span", {class: "badge qz-new-format"}, m.note) : null));
}

// The radio cards stay in the radiogroup; each is wrapped with its "옵션" disclosure and panel.
function defaultStep() {
  const items = palette.filter((m) => !byId[m].skip).map((m) => ({id: m, render: () => methodCard(m)}));
  ui.defaults = radioGroup("기본 방법", items, state.def, (m) => (m === state.def ? null : setDefault(m)), "qz-cards");
  for (const card of [...ui.defaults.querySelectorAll("[role=radio]")]) {
    const method = card.dataset.default;
    const panel = optionsPanel(method);
    const wrap = h("div", {class: "qz-method", "data-method": method});
    card.replaceWith(wrap);
    wrap.append(card, disclosure(`qz-opt-${method}`, method, panel), panel);
  }
  return step("2", "기본 방법 · 방법별 옵션", "카드 = 따로 정하지 않은 레이어의 방법 · 옵션 = 그 방법을 쓰는 모든 레이어", ui.defaults);
}

// Global, next to the summary that says whether the recipe needs it.
function calibPanel() {
  ui.calibFields = Object.entries(CALIBRATION).map(([name, spec]) => {
    const info = data.calibration_info[name];
    return control({id: `calibration.${name}`, spec, label: info.label, words: info.choices,
      value: () => state.calib[name] ?? spec.default,
      set: (value, merge) => commit(() => (state.calib[name] = value), `보정 ${info.label}: ${info.choices?.[value] ?? value}`, merge)});
  });
  ui.calibStatus = h("p", {class: "qz-calib-status"});
  ui.calibReset = h("button", {type: "button", class: "qz-quick", "data-reset": "calibration",
    onclick: () => commit(() => (state.calib = {}), "보정 설정을 기본값으로 되돌렸습니다")}, "기본값으로");
  return h("section", {class: "panel qz-calibpanel", "aria-labelledby": "qz-calib-title"},
    h("div", {class: "qz-panel-head"}, h("h2", {id: "qz-calib-title"}, "보정 데이터"), ui.calibReset),
    h("p", {class: "qz-cap"}, "GPTQ · AWQ · SmoothQuant · 정적 observer 가 train split 에서 뽑는 창"),
    h("div", {class: "qz-calib-grid", role: "group", "aria-label": "보정 데이터"}, ui.calibFields.map((entry) => entry.row)),
    ui.calibStatus);
}

function gridStep() {
  ui.cells = [];
  const head = [h("div", {role: "columnheader", class: "qz-corner"}, h("span", {class: "qz-axis"}, "블록 →"))];
  for (let b = 0; b < BLOCKS; b += 1) {
    head.push(h("div", {role: "columnheader", class: "qz-colhead"},
      h("button", {type: "button", tabindex: "-1", "data-col": b, title: `블록 ${b} 의 7개 Linear 를 고릅니다 (Shift: 추가)`,
        "aria-label": `블록 ${b} 전체 선택`, onclick: (event) => pickGroup(colCells(b), event.shiftKey, [b, 0], `블록 ${b}`)}, b)));
  }
  const rows = KINDS.map((kind, k) => {
    const cells = Array.from({length: BLOCKS}, (_, b) => {
      const cell = h("div", {role: "gridcell", class: `qz-cell${k && PARENT[kind] !== PARENT[KINDS[k - 1]] ? " qz-mlp" : ""}`, tabindex: "-1",
        "data-b": b, "data-k": k, "aria-selected": "false"});
      const tip = () => (drag ? null : showTip(cell, cellTip(b, k)));
      cell.addEventListener("mouseenter", tip);
      cell.addEventListener("focus", () => {
        state.focus = [b, k];
        tip();
      });
      cell.addEventListener("mouseleave", hideTip);
      cell.addEventListener("blur", hideTip);
      ui.cells[key(b, k)] = cell;
      return cell;
    });
    return h("div", {role: "row", class: "qz-row"},
      h("div", {role: "rowheader", class: `qz-rowhead${k && PARENT[kind] !== PARENT[KINDS[k - 1]] ? " qz-mlp" : ""}`},
        h("button", {type: "button", tabindex: "-1", "data-row": k, title: `모든 블록의 ${kind} 를 고릅니다 (Shift: 추가)`,
          "aria-label": `${kind} 전체 선택`, onclick: (event) => pickGroup(rowCells(k), event.shiftKey, [0, k], `모든 ${kind}`)},
        h("span", {class: "qz-parent"}, PARENT[kind] === "mlp" ? "mlp" : "attn"), kind)), cells);
  });
  ui.grid = h("div", {class: "qz-grid", role: "grid", "aria-label": "Linear 종류 × 디코더 블록. 화살표로 이동, Space 로 선택",
    "aria-rowcount": KINDS.length + 1, "aria-colcount": BLOCKS + 1, onkeydown: onGridKey, onpointerdown: onPointerDown},
  h("div", {role: "row", class: "qz-row"}, head), rows);
  ui.apply = palette.map((m, i) => h("button", {type: "button", class: "qz-apply", "data-apply": m,
    title: `${byId[m].label} (${i < 9 ? i + 1 : 0} 키)`, onclick: () => apply(m, [...state.sel])},
  h("kbd", null, i < 9 ? i + 1 : 0), code(m), h("span", null, byId[m].skip ? "BF16 유지" : byId[m].label)));
  ui.count = h("span", {class: "qz-count"});
  ui.status = h("span", {class: "qz-status"});
  const quick = (label, title, action) => h("button", {type: "button", class: "qz-quick", title, "data-quick": label, onclick: action}, label);
  ui.undo = quick("실행 취소", "마지막 변경을 되돌립니다 (⌘Z / Ctrl+Z)", undo);
  return step("3", "레이어별 방법", "칸 하나가 Linear 레이어 하나",
    h("div", {class: "qz-palette", role: "group", "aria-label": "선택에 적용"}, h("span", {class: "qz-label"}, "선택에 적용"), ui.apply),
    h("div", {class: "qz-quickrow"},
      quick("첫·마지막 블록 BF16", "블록 0 과 27 의 모든 Linear 를 BF16 으로 둡니다", () => apply("bf16", [...colCells(0), ...colCells(BLOCKS - 1)])),
      ...["o_proj", "down_proj"].map((kind) => quick(`모든 ${kind}`, `모든 블록의 ${kind} 를 고릅니다`,
        () => pickGroup(rowCells(KINDS.indexOf(kind)), false, [0, KINDS.indexOf(kind)], `모든 ${kind}`))),
      quick("선택 해제", "선택을 비웁니다 (Esc)", clearSelection),
      quick("기본값으로 되돌리기", "모든 레이어가 기본 방법을 쓰게 합니다", () => commit(() => (state.grid = uniform(state.def)),
        "모든 레이어를 기본 방법으로 되돌렸습니다")),
      ui.undo, h("span", {class: "qz-tally"}, ui.count, ui.status)),
    h("div", {class: "qz-gridwrap", tabindex: "-1"}, ui.grid),
    h("p", {class: "qz-help"}, "칸을 누르거나 끌어서 사각형으로 고르고, 블록 번호·종류 이름을 누르면 그 열·행 전체를 고릅니다. "
      + "Shift+클릭은 선택에 더합니다. 키보드: 화살표로 이동, Shift+화살표로 넓히기, Space 로 고르기, 1–9·0 으로 적용, "
      + "Esc 로 해제, ⌘/Ctrl+Z 로 되돌리기."),
    h("div", {class: "qz-legend", role: "group", "aria-label": "범례"}, palette.map((m) => h("span", {class: "qz-key"}, code(m),
      h("span", null, byId[m].label), optDot(m), h("span", {class: "qz-key-n", "data-count": m}))),
    h("span", {class: "qz-key"}, h("span", {class: "qz-code qz-selected-sample", "aria-hidden": "true"}), "선택한 칸")));
}

// Stored weight bits per element with the method's options (a GPTQ group size changes the scale overhead).
function weightBits(method, kind, opts = pageOptions()) {
  const [element, scale, per] = byId[method].bits;
  return element + scale / (opts[method]?.["weight.group_size"] ?? per ?? kind.in_features);
}

function cellTip(b, k) {
  const kind = data.kinds[k];
  const m = byId[state.grid[b][k]];
  const opts = pageOptions();
  const bits = weightBits(m.id, kind, opts);
  const changed = Object.keys(opts[m.id] ?? {}).length;
  return h("div", null, h("b", null, layerName(b, kind.id)), h("br"),
    `${m.label}${changed ? ` (옵션 ${changed}개 바꿈)` : ""} · ${m.wa_bits} · 가중치 ${bits.toFixed(bits % 1 ? 3 : 0)}비트`, h("br"),
    h("span", {class: "muted"}, `${kind.in_features} × ${kind.out_features} · ${millions(kind.params)} 파라미터`
      + (state.grid[b][k] === state.def ? " · 기본 방법" : "")));
}

function pickGroup(cells, add, anchor, label) {
  state.anchor = anchor;
  select(cells, add ? state.sel : new Set());
  announce(`${label} 선택 · 모두 ${state.sel.size}칸`);
}

let drag = null;
function onPointerDown(event) {
  const cell = event.target.closest(".qz-cell");
  if (!cell || event.button !== 0) return;
  event.preventDefault();
  hideTip();
  const at = posOf(cell);
  const extend = event.shiftKey && state.anchor;
  drag = {base: extend ? new Set(state.sel) : new Set(), anchor: extend ? state.anchor : at, last: key(...at)};
  state.anchor = drag.anchor;
  state.focus = at;
  select(rect(drag.anchor, at), drag.base);
  focusCell();
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp, {once: true});
}

function onPointerMove(event) {
  const cell = document.elementFromPoint(event.clientX, event.clientY)?.closest?.(".qz-cell");
  if (!drag || !cell) return;
  const at = posOf(cell);
  if (key(...at) === drag.last) return;
  drag.last = key(...at);
  state.focus = at;
  select(rect(drag.anchor, at), drag.base);
  focusCell();
}

function onPointerUp() {
  window.removeEventListener("pointermove", onPointerMove);
  if (drag) announce(`${state.sel.size}칸 선택`);
  drag = null;
}

function focusCell() {
  const [b, k] = state.focus;
  for (const cell of ui.cells) cell.tabIndex = -1;
  const cell = ui.cells[key(b, k)];
  cell.tabIndex = 0;
  if (document.activeElement !== cell) cell.focus({preventScroll: false});
}

function onGridKey(event) {
  const moves = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  if (!event.target.classList.contains("qz-cell")) return;
  if (moves[event.key]) {
    event.preventDefault();
    const [db, dk] = moves[event.key];
    const [b, k] = state.focus;
    state.focus = [Math.min(BLOCKS - 1, Math.max(0, b + db)), Math.min(KINDS.length - 1, Math.max(0, k + dk))];
    if (event.shiftKey) {
      state.anchor ??= [b, k];
      select(rect(state.anchor, state.focus));
    } else updateGrid();
    focusCell();
  } else if (event.key === " ") {
    event.preventDefault();
    const c = key(...state.focus);
    if (state.sel.has(c)) state.sel.delete(c);
    else state.sel.add(c);
    state.anchor = [...state.focus];
    updateGrid();
    announce(`${state.sel.size}칸 선택`);
  }
}

// Digits apply the n-th method anywhere in the grid step (to the selection, or to the focused cell when nothing
// is selected); ⌘/Ctrl+Z and Esc work everywhere except while typing.
function onKey(event) {
  const typing = event.target.closest?.("input, textarea, select");
  if ((event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "z" && !typing) {
    event.preventDefault();
    undo();
  } else if (event.key === "Escape" && !typing) {
    clearSelection();
  } else if (/^[0-9]$/.test(event.key) && !event.metaKey && !event.ctrlKey && !event.altKey && !typing
      && event.target.closest?.(".qz-grid-step")) {
    const method = palette[event.key === "0" ? 9 : Number(event.key) - 1];
    if (!method) return;
    event.preventDefault();
    apply(method, state.sel.size ? [...state.sel] : [key(...state.focus)]);
  }
}

function kvStep() {
  const items = data.kv.map((item) => ({id: item.id, render: () => h("button", {type: "button", "data-kv": item.id,
    title: item.description}, item.label, h("small", null, item.short))}));
  ui.kv = radioGroup("KV 캐시", items, state.kv, (id) => commit(() => (state.kv = id),
    `KV 캐시: ${data.kv.find((item) => item.id === id).label}`), "seg");
  ui.kvNote = h("p", {class: "note"});
  const entries = Object.entries(KV_OPTIONS).flatMap(([preset, specs]) => Object.entries(specs).map(([name, spec]) => control({
    id: `kv.${preset}.${name}`, spec, label: data.kv_option_info[name].label,
    value: () => state.kvOptions[preset]?.[name] ?? spec.default, applicable: () => state.kv === preset,
    set: (value, merge) => commit(() => ((state.kvOptions[preset] ??= {})[name] = value), `KV ${name}: ${value}`, merge)})));
  const panel = h("div", {class: "qz-opt-panel qz-kv-panel", role: "group", "aria-label": "KV 캐시 옵션"},
    h("p", {class: "qz-fixed"}, "KIVI: 32의 배수 (키 채널 그룹 32), 기본 128 · FP8: 기본 0 = 모든 토큰을 저장 (vLLM)"),
    entries.map((entry) => entry.row));
  ui.kvToggle = disclosure("qz-opt-kv", "kv", panel);
  return step("4", "KV 캐시", "모드 fakequant", ui.kv, h("div", {class: "qz-kv-opts"}, ui.kvToggle, optDot("kv")), panel, ui.kvNote);
}

function nameStep() {
  ui.name = h("input", {type: "text", id: "qz-name", class: "qz-name", spellcheck: "false", autocomplete: "off",
    "aria-describedby": "qz-name-help qz-name-error", value: state.name});
  ui.name.addEventListener("input", () => {
    state.name = ui.name.value;
    state.nameAuto = false;
    updateResults();
    updateNameError();
  });
  ui.nameError = h("p", {class: "note warn", id: "qz-name-error", role: "status"});
  return step("5", "레시피 이름", null,
    h("label", {class: "qz-name-row", for: "qz-name"}, h("span", null, "name"), ui.name),
    h("p", {class: "note", id: "qz-name-help"}, "YAML 의 name 이자 파일 이름(<이름>.yaml)입니다. 프리셋을 그대로 쓰면 프리셋 id 가 이름이 됩니다."),
    ui.nameError);
}

function updateNameError() {
  const text = state.name;
  ui.nameError.textContent = !text ? "이름을 입력하세요. 결과 파일 이름으로도 쓰입니다."
    : SAFE_NAME.test(text) ? "" : "영문·숫자로 시작하고 영문·숫자·_ - . 만 쓰세요. 결과 파일 이름으로 쓰입니다.";
  const ok = !ui.nameError.textContent;
  ui.name.setAttribute("aria-invalid", String(!ok));
  updateBlocked();
}

// Download, copy and server runs wait while the name is unsafe or a control holds a value build() would refuse.
function updateBlocked() {
  const ok = !ui.nameError.textContent && !ui.invalid.size;
  ui.download.disabled = !ok;
  ui.yamlCopy.disabled = !ok;
  for (const button of ui.commandCopies) button.disabled = !ok;
  const first = ui.invalid.values().next().value;
  ui.yamlNote.textContent = first ? `고칠 입력이 있습니다 — ${first}. YAML 은 마지막 올바른 값으로 만든 것입니다.` : "";
  updateGpu();
}

// ---------- results column ----------
function summaryPanel() {
  ui.bar = h("div", {class: "qz-bar", role: "img"});
  ui.barKeys = h("div", {class: "qz-barkeys"});
  const readout = (id, label) => {
    const v = h("span", {class: "v"});
    const sub = h("span", {class: "s"});
    ui[id] = {v, sub};
    return h("div", {class: "readout", "data-readout": id}, h("span", {class: "k"}, label), v, sub);
  };
  return h("section", {class: "panel qz-summary", "aria-labelledby": "qz-sum-title"},
    h("h2", {id: "qz-sum-title"}, "요약"),
    h("p", {class: "qz-cap"}, "디코더 Linear 196개, 방법별 개수"),
    ui.bar, ui.barKeys,
    h("div", {class: "readouts qz-readouts"}, readout("bits", "가중치 평균 비트"), readout("bf16", "BF16 로 남는 파라미터"),
      readout("calib", "보정"), readout("rules", "override 규칙")));
}

function yamlPanel() {
  ui.yaml = h("pre", {class: "code qz-yaml", tabindex: "0", "aria-label": "레시피 YAML"});
  ui.download = h("button", {type: "button", class: "copy", onclick: download}, "내려받기 (.yaml)");
  ui.yamlCopy = copyButton("YAML 복사", () => `${currentYaml()}\n`);
  ui.yamlNote = h("p", {class: "note warn", role: "status"});
  return h("section", {class: "panel qz-yamlpanel", "aria-labelledby": "qz-yaml-title"},
    h("div", {class: "qz-panel-head"}, h("h2", {id: "qz-yaml-title"}, "레시피 YAML"),
      h("span", {class: "qz-actions"}, ui.yamlCopy, ui.download)),
    ui.yamlNote, ui.yaml);
}

const benchTasks = () => data.benchmarks.map((item) => item.task);

function commands() {
  const file = `${state.name}.yaml`;
  return [
    ["레시피 확인", `tricast recipe-check ${file}`],
    ["레이어별 오차 리포트", `tricast report --model Qwen/Qwen3-0.6B --recipe ${file}`],
    ["WikiText-2 PPL", `tricast ppl --model Qwen/Qwen3-0.6B --recipe ${file}`],
    ["lm-eval 5과제 (0-shot, 과제당 앞 300문서)",
      `tricast eval --model Qwen/Qwen3-0.6B --recipe ${file} --tasks ${benchTasks().join(",")} --limit 300`],
    ["벤치 설정으로 BF16 과 함께 (설정 파일은 아래 버튼)", `tricast run ${state.name}_bench.yaml`],
    ["레이어 지도에 추가", "python -m app.export_layers runs/<run>/report --out app/web/demo/layers"],
  ];
}

// The `tricast run` config of the recorded lm-eval runs (runs/20261013_quantizer_w3/lmeval_A.yaml), for this recipe.
function benchConfig() {
  return `${[`# lm-eval E2E for ${state.name}: 0-shot, first 300 docs per task, with the BF16 baseline (one GPU).`,
    "model: Qwen/Qwen3-0.6B", "dtype: bfloat16", "device: cuda", "seed: 42", "native_baseline: true", "tasks:",
    "  lm_eval:", `    tasks: [${benchTasks().join(", ")}]`, "    batch_size: 16", "    limit: 300", "recipes:",
    `  - ${state.name}.yaml`, `output_dir: runs/${state.name}_bench`].join("\n")}\n`;
}

// The same recipe from Python: only existing tricast functions (checked by test_quantizer_runner.py).
function apiSnippet() {
  return ["import transformers",
    "from tricast import load_recipe, patch_model, calibrate",
    "from tricast.eval import perplexity, evaluate",
    'tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")',
    'model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="bfloat16").cuda().eval()',
    `recipe = load_recipe("${state.name}.yaml"); patch_model(model, recipe)`,
    'if recipe.needs_calibration: calibrate(model, recipe, tokenizer=tok, device="cuda")',
    `print(perplexity(model, tok, device="cuda")["ppl"], evaluate(model, tok, [${benchTasks().map((t) => `"${t}"`).join(", ")}], `
      + 'limit=300)["results"])'].join("\n");
}

function commandPanel() {
  ui.commandCodes = [];
  ui.commandCopies = [];
  const rows = commands().map(([label], i) => {
    const text = h("code", {class: "qz-cmd"});
    const button = copyButton("복사", () => commands()[i][1]);
    button.setAttribute("aria-label", `${label} 명령 복사`);
    ui.commandCodes.push(text);
    ui.commandCopies.push(button);
    return h("li", null, h("span", {class: "qz-cmd-label"}, label), text, button);
  });
  ui.calibNote = h("p", {class: "note"});
  ui.benchDownload = h("button", {type: "button", class: "copy", onclick: () => save(`${state.name}_bench.yaml`, benchConfig())},
    "벤치 설정 내려받기 (.yaml)");
  ui.api = h("pre", {class: "code qz-api", tabindex: "0", "aria-label": "Python API 예시"});
  const apiCopy = copyButton("코드 복사", () => `${apiSnippet()}\n`);
  ui.commandCopies.push(ui.benchDownload, apiCopy);
  return h("section", {class: "panel qz-commands", "aria-labelledby": "qz-cmd-title"},
    h("h2", {id: "qz-cmd-title"}, "실행 명령"),
    h("p", {class: "qz-cap"}, "내려받은 YAML 이 있는 저장소 루트에서 실행합니다. 웹·CLI·Python 모두 같은 YAML 을 씁니다."),
    h("ol", {class: "qz-cmds"}, rows), h("div", {class: "qz-actions"}, ui.benchDownload), ui.calibNote,
    h("div", {class: "qz-panel-head"}, h("h3", {class: "qz-sub"}, "Python API"), apiCopy), ui.api);
}

function recordPanel() {
  ui.record = h("div", {class: "qz-record-body"});
  return h("section", {class: "panel qz-record", "aria-labelledby": "qz-rec-title"},
    h("h2", {id: "qz-rec-title"}, "기록된 결과"), ui.record);
}

function benchPanel() {
  ui.bench = h("div", {class: "qz-bench-body"});
  return h("section", {class: "panel qz-bench", "aria-labelledby": "qz-bench-title"},
    h("h2", {id: "qz-bench-title"}, "벤치마크 (E2E)"), ui.bench);
}

const currentRecipe = () => build(state.name, state.def, state.grid, state.kv, DESCRIPTION, extra());

function currentYaml() {
  return toYaml(currentRecipe());
}

function download() {
  save(`${state.name}.yaml`, `${currentYaml()}\n`);
}

function stats(opts) {
  const counts = new Map(palette.map((m) => [m, 0]));
  let params = 0;
  let bits = 0;
  let bf16 = 0;
  state.grid.forEach((row) => row.forEach((m, k) => {
    const kind = data.kinds[k];
    counts.set(m, counts.get(m) + 1);
    params += kind.params;
    bits += kind.params * weightBits(m, kind, opts);
    if (m === "bf16") bf16 += kind.params;
  }));
  return {counts, params, bits: bits / params, bf16, used: new Set([state.def, ...state.grid.flat()])};
}

// Methods in use that need calibration (options included) and the calibration settings build() writes.
function calibration(opts = pageOptions()) {
  const used = new Set([state.def, ...state.grid.flat()]);
  return {methods: palette.filter((m) => used.has(m) && needsCalibration(m, opts[m])), settings: checkCalibration(pageCalibration())};
}

function updateSummary(recipe, opts) {
  const st = stats(opts);
  const runs = palette.filter((m) => st.counts.get(m));
  ui.bar.replaceChildren(...runs.map((m) => withTip(h("span", {class: "qz-seg", "data-m": m, tabindex: "0",
    style: `flex-grow: ${st.counts.get(m)}`}), () => h("div", null, h("b", null, byId[m].label), h("br"),
    `${st.counts.get(m)}개 레이어 (${(st.counts.get(m) / (BLOCKS * KINDS.length) * 100).toFixed(1)}%)`))));
  ui.bar.setAttribute("aria-label", `방법별 레이어 수: ${runs.map((m) => `${byId[m].label} ${st.counts.get(m)}개`).join(", ")}`);
  ui.barKeys.replaceChildren(...runs.map((m) => h("span", {class: "qz-key"}, code(m), h("span", null, byId[m].label),
    h("b", {class: "num"}, st.counts.get(m)))));
  ui.bits.v.textContent = st.bits.toFixed(2);
  ui.bits.sub.textContent = "파라미터 가중 · 스케일 포함";
  ui.bf16.v.textContent = `${(st.bf16 / st.params * 100).toFixed(1)}%`;
  ui.bf16.sub.textContent = `${millions(st.bf16)} / ${millions(st.params)} (디코더 Linear)`;
  const {methods: calibrated, settings} = calibration(opts);
  const tokens = `${count(settings.samples)} × ${count(settings.seqlen)} = ${count(settings.samples * settings.seqlen)} 토큰`;
  ui.calib.v.textContent = calibrated.length ? "보정 필요" : "없음";
  ui.calib.sub.textContent = calibrated.length ? `${tokens} · ${calibrated.map((m) => byId[m].label).join(", ")}` : "보정 데이터 없이 실행";
  ui.calib.v.closest(".readout").classList.toggle("qz-need", calibrated.length > 0);
  ui.rules.v.textContent = `${recipe.overrides?.length ?? 0}개`;
  ui.rules.sub.textContent = "칸 → 블록 → 종류 순, 먼저 맞는 규칙";
  for (const el of ui.grid.closest(".qz-step").querySelectorAll("[data-count]")) {
    const n = st.counts.get(el.dataset.count);
    el.textContent = n;
    el.closest(".qz-key").classList.toggle("off", !n);
  }
  return calibrated;
}

function sqnrMap(report, grid) {
  const values = report.sqnr_db;
  const finite = values.filter((v) => typeof v === "number");
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const bin = (v) => (hi > lo ? Math.min(7, Math.floor(((hi - v) / (hi - lo)) * 8)) : 0);  // 0 = best
  const cells = [];
  KINDS.forEach((kind, k) => {
    for (let b = 0; b < BLOCKS; b += 1) {
      const v = values[b * KINDS.length + k];
      const method = grid[b][k];
      const cls = typeof v === "number" ? `qz-s${bin(v)}` : token(v) ? "qz-s0" : "qz-na";
      const text = typeof v === "number" ? `${v.toFixed(2)} dB` : token(v) ?? (method === "bf16" ? "BF16 유지 · 측정 안 함" : "값 없음");
      const cell = h("span", {class: `qz-mini ${cls}`, tabindex: "-1"});
      withTip(cell, () => h("div", null, h("b", null, layerName(b, kind)), h("br"), `${byId[method].label} · 출력 SQNR ${text}`));
      cells.push(cell);
    }
  });
  return h("figure", {class: "qz-minimap"},
    h("div", {class: "qz-minigrid", role: "img", "aria-label": `레이어별 출력 SQNR, ${fixed(lo, 1)}–${fixed(hi, 1)} dB`}, cells),
    h("figcaption", {class: "qz-scale"}, h("span", null, "좋음 ", h("b", null, `${fixed(hi, 1)} dB`)),
      h("span", {class: "qz-ramp", "aria-hidden": "true"}), h("span", null, h("b", null, `${fixed(lo, 1)} dB`), " 나쁨"),
      h("span", {class: "qz-key"}, h("i", {class: "qz-mini qz-na"}), "BF16 유지 (측정 안 함)")));
}

function stat(label, value, sub) {
  return h("div", {class: "readout"}, h("span", {class: "k"}, label), h("span", {class: "v"}, value),
    sub ? h("span", {class: "s"}, sub) : null);
}

function save(file, text) {
  const blob = new Blob([text], {type: "application/yaml"});
  const url = URL.createObjectURL(blob);
  const link = h("a", {href: url, download: file});
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  announce(`${file} 을 내려받았습니다`);
}

const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1));
const dirty = (env) => (env.git_dirty === false ? "작업 트리 깨끗함" : env.git_dirty ? "작업 트리 수정 있음" : "작업 트리 알 수 없음");
const gpus = (env) => [...new Set(env.gpu_names ?? [])].join(", ") || "—";

// Δ against the BF16 run only when both measured the same windows; never interpolated.
function pplDelta(run) {
  if (run.ppl_delta_pct != null) return run.ppl_delta_pct;
  const native = data.native;
  return native && native.dataset_fingerprint === run.dataset_fingerprint && native.n_windows === run.n_windows
    ? (run.ppl / native.ppl - 1) * 100 : null;
}

// replaceChildren would print a null as the text "null"; optional parts are null here, so drop them.
const fill = (el, ...nodes) => el.replaceChildren(...nodes.flat(Infinity).filter((node) => node != null && node !== false));

function rejectedNote(reason) {
  return reason ? h("p", {class: "note warn qz-rejected", title: reason}, "기록 파일이 있지만 현재 프리셋과 맞지 않아 보여 주지 않습니다: ",
    reason.length > 160 ? `${reason.slice(0, 160)}…` : reason) : null;
}

// PPL + report readouts for a recorded preset or a finished live job.
function resultBody({run, report, grid, rejected = []}) {
  const native = data.native;
  const env = run?.env ?? report?.env;
  const body = [];
  if (run) {
    const delta = pplDelta(run);
    body.push(h("div", {class: "readouts qz-readouts"},
      stat("WikiText-2 test PPL", fixed(run.ppl, 2), `${count(run.n_windows)}개 창 · ${count(run.n_tokens)} 토큰`),
      stat("BF16 원본 PPL", native ? fixed(native.ppl, 2) : "—", native ? "같은 창, 양자화 없음" : "BF16 기록 없음"),
      stat("Δ vs BF16", delta == null ? "—" : `${signed(delta, 1)}%`, delta == null ? "같은 창의 BF16 기록 없음" : "PPL 상대 변화")));
  } else body.push(h("p", {class: "qz-empty"}, "WikiText-2 PPL 은 아직 기록이 없습니다."), rejected[0]);
  if (env) {
    const sha = run?.recipe_hash ?? report.recipe_hash;
    body.push(h("dl", {class: "kv qz-env"},
      h("dt", null, "GPU"), h("dd", null, gpus(env)),
      h("dt", null, "commit"), h("dd", null, `${String(env.git_sha ?? "").slice(0, 7) || "—"} · ${dirty(env)}`),
      run ? [h("dt", null, "실행 시간"), h("dd", null, minutes(run.wall_time_s))] : null,
      h("dt", null, "레시피 SHA"), h("dd", null, h("b", null, String(sha).slice(0, 12)), " = 기록된 실행의 recipe_hash")));
  }
  if (report) {
    body.push(h("h3", {class: "qz-sub"}, "레이어별 출력 SQNR"), sqnrMap(report, grid),
      h("p", {class: "qz-cap"}, `리포트 창 ${report.samples} × ${report.seqlen} 토큰 (train) · logits KL ${fixed(report.logits_kl, 3)}`
        + ` · top-1 일치 ${report.top1_agreement == null ? "—" : `${(report.top1_agreement * 100).toFixed(1)}%`}`));
  } else body.push(h("p", {class: "qz-empty"}, "레이어별 리포트는 아직 기록이 없습니다."), rejected[1]);
  return body;
}

function benchBody(lm) {
  const native = data.lmeval_native;
  const base = (task) => native?.tasks.find((item) => item.task === task) ?? null;
  const delta = (item) => {
    if (item.delta_pts != null) return item.delta_pts;
    const b = base(item.task);
    return b && b.n === item.n && native.limit === lm.limit ? (item.value - b.value) * 100 : null;
  };
  const rows = lm.tasks.map((item) => {
    const b = base(item.task);
    const d = delta(item);
    return h("tr", {"data-task": item.task},
      h("th", {scope: "row"}, item.label, h("small", null, item.metric)),
      h("td", {class: "num v"}, pct(item.value)),
      h("td", {class: "num"}, b ? pct(b.value) : "—"),
      h("td", {class: `num${d != null && d < 0 ? " neg" : ""}`}, d == null ? "—" : signed(d, 1)),
      h("td", {class: "num muted"}, item.stderr == null ? "—" : `±${(item.stderr * 100).toFixed(1)}`),
      h("td", {class: "num muted"}, item.n ?? "—"),
      h("td", {class: "qz-bcell", "aria-hidden": "true"}, h("span", {class: "qz-bbar"},
        h("i", {style: `width: ${(item.value * 100).toFixed(1)}%`}),
        b ? h("b", {style: `left: ${(b.value * 100).toFixed(1)}%`}) : null)));
  });
  const all = native && lm.tasks.length === native.tasks.length;
  const macroDelta = lm.macro_delta_pts ?? (all && lm.tasks.every((item) => delta(item) != null)
    ? (lm.macro - native.macro) * 100 : null);
  rows.push(h("tr", {class: "qz-macro"}, h("th", {scope: "row"}, "매크로 평균", h("small", null, `${lm.tasks.length}과제`)),
    h("td", {class: "num v"}, pct(lm.macro)), h("td", {class: "num"}, all ? pct(native.macro) : "—"),
    h("td", {class: `num${macroDelta != null && macroDelta < 0 ? " neg" : ""}`}, macroDelta == null ? "—" : signed(macroDelta, 1)),
    h("td", null), h("td", null), h("td", null)));
  const env = lm.env;
  return [h("div", {class: "table-wrap"}, h("table", {class: "diff qz-btable"},
    h("thead", null, h("tr", null, ["과제", "이 레시피", "BF16", "Δ (pt)", "±stderr", "n", ""].map((t, i) =>
      h("th", {scope: "col", class: i && i < 6 ? "num" : null}, t)))), h("tbody", null, rows))),
    h("p", {class: "qz-cap"}, "점수는 % (acc_norm, WinoGrande 는 acc). 막대는 이 레시피, 세로 눈금은 BF16."),
    h("dl", {class: "kv qz-env"}, h("dt", null, "GPU"), h("dd", null, gpus(env)),
      h("dt", null, "lm-eval"), h("dd", null, `${lm.lm_eval_version ?? "—"} · 0-shot · limit ${lm.limit ?? "—"} · batch ${lm.batch_size ?? "—"}`),
      h("dt", null, "commit"), h("dd", null, `${String(env.git_sha ?? "").slice(0, 7) || "—"} · ${dirty(env)}`),
      h("dt", null, "BF16"), h("dd", null, native ? `${native.source}` : "기록 없음 (Δ 를 계산하지 않습니다)"))];
}

function updateRecord() {
  const preset = matchedPreset();
  const recorded = data.presets.filter((p) => p.run || p.report);
  const jump = recorded.length ? h("p", {class: "qz-cap"}, "기록이 있는 프리셋: ", recorded.map((p) => h("button", {type: "button",
    class: "qz-link", onclick: () => loadPreset(p)}, p.id))) : null;
  if (!trusted) {
    const warn = () => h("p", {class: "note warn", role: "alert"}, "이 페이지가 만든 프리셋 YAML 이 내보낸 Python 기준과 다릅니다. "
      + "기록을 보여 주지 않습니다. 페이지와 presets.json 을 같은 버전으로 맞추세요: ", h("code", null, EXPORT));
    fill(ui.record, warn());
    fill(ui.bench, warn());
    return;
  }
  const rejected = preset ? [rejectedNote(preset.run_rejected), rejectedNote(preset.report_rejected)] : [];
  if (preset?.lmeval) fill(ui.bench, ...benchBody(preset.lmeval));
  else {
    const benched = data.presets.filter((p) => p.lmeval);
    fill(ui.bench, h("p", {class: "qz-empty"}, "기록 없음 — 아래 명령으로 실행"), preset ? rejectedNote(preset.lmeval_rejected) : null,
      benched.length ? h("p", {class: "qz-cap"}, "lm-eval 기록이 있는 프리셋: ", benched.map((p) => h("button", {type: "button",
        class: "qz-link", onclick: () => loadPreset(p)}, p.id))) : null);
  }
  if (!preset || !(preset.run || preset.report)) {
    fill(ui.record, h("p", {class: "qz-empty"}, preset ? `이 조합은 프리셋 ${preset.id} 와 같지만 아직 기록이 없습니다 — `
      + "아래 명령으로 실행하면 레이어 지도에 추가됩니다" : "이 조합은 아직 기록이 없습니다 — 아래 명령으로 실행하면 레이어 지도에 추가됩니다"),
    ...rejected, jump);
    return;
  }
  const body = [h("p", {class: "qz-match"}, "지금 조합 = 프리셋 ", h("b", null, preset.id), ` · ${preset.label}`),
    ...resultBody({run: preset.run, report: preset.report, grid: preset.grid, rejected})];
  if (state.name !== preset.id) {
    const rename = h("button", {type: "button", class: "qz-link", onclick: () => {
      state.name = preset.id;
      state.nameAuto = false;
      update();
    }}, `이름을 ${preset.id} 로`);
    body.push(h("p", {class: "note"}, `SHA 는 이름까지 포함합니다. 지금 이름(${state.name})의 YAML 은 name 줄만 다릅니다. `, rename));
  }
  body.push(h("a", {class: "qz-maplink", href: "layers.html"}, "레이어 지도 열기 →"));
  fill(ui.record, ...body);
}

// ---------- run on the server GPU (live mode) ----------
const GPU_TASKS = [["ppl", "WikiText-2 PPL"], ["report", "레이어별 리포트"], ["arc_easy", "ARC-Easy"],
  ["arc_challenge", "ARC-Challenge"], ["piqa", "PIQA"], ["winogrande", "WinoGrande"], ["hellaswag", "HellaSwag"]];
let server = {live: false, reason: "서버 상태를 확인하는 중입니다."};
let job = null;  // {name, grid, tasks, id, record, error}

async function probeServer() {
  try {
    const response = await fetch("api/catalog", {headers: {accept: "application/json"}});
    if (!response.ok) throw new Error(String(response.status));
    const catalog = await response.json();
    if (catalog.mode !== "live") {
      server = {live: false, reason: "데모 서버라 GPU 가 없습니다. GPU 노드에서 python -m app.server --device cuda 로 띄운 서버에서 쓰세요."};
    } else if (catalog.device?.kind !== "cuda") {
      server = {live: false, reason: "이 서버는 CUDA 없이 떠 있습니다. GPU 노드에서 python -m app.server --device cuda 로 다시 띄우세요."};
    } else server = {live: true, reason: `${catalog.device.name ?? "CUDA"} 에서 실행합니다. 작업은 서버 큐에서 차례로 돕니다.`};
  } catch {
    server = {live: false, reason: "정적 파일 서버에서 열었습니다. 서버 GPU 실행은 GPU 노드에서 python -m app.server --device cuda 로 띄운 서버에서만 됩니다."};
  }
  updateGpu();
}

function gpuPanel() {
  ui.gpuTasks = GPU_TASKS.map(([id, label]) => h("label", {class: "qz-task"},
    h("input", {type: "checkbox", "data-task": id, checked: id === "ppl" || id === "report"}), label));
  ui.gpuLimit = h("input", {type: "number", class: "qz-name qz-limit", min: "1", max: "300", value: "300", id: "qz-limit"});
  ui.gpuRun = h("button", {type: "button", class: "run-btn qz-gpu-run", onclick: submitJob}, "서버 GPU에서 실행");
  ui.gpuNote = h("p", {class: "note", id: "qz-gpu-note"});
  ui.gpuJob = h("div", {class: "qz-job", "aria-live": "polite"});
  ui.gpuRun.setAttribute("aria-describedby", "qz-gpu-note");
  return h("section", {class: "panel qz-gpu", "aria-labelledby": "qz-gpu-title"},
    h("h2", {id: "qz-gpu-title"}, "서버 GPU에서 실행"),
    h("p", {class: "qz-cap"}, "지금 레시피 YAML 을 서버로 보내 tricast 로 잽니다. 결과는 위 패널과 같은 모양으로 나옵니다."),
    h("div", {class: "qz-tasks", role: "group", "aria-label": "잴 항목"}, ui.gpuTasks),
    h("label", {class: "qz-name-row", for: "qz-limit"}, h("span", null, "lm-eval 과제당 문서"), ui.gpuLimit, h("span", null, "개 (1–300)")),
    ui.gpuRun, ui.gpuNote, ui.gpuJob);
}

function updateGpu() {
  if (!ui.gpuRun) return;
  const nameOk = SAFE_NAME.test(state.name);
  const busy = job && !job.error && !["done", "error"].includes(job.record?.status);
  ui.gpuRun.disabled = !server.live || !nameOk || busy || ui.invalid.size > 0;
  ui.gpuNote.textContent = !server.live ? server.reason : !nameOk ? "레시피 이름을 먼저 고치세요 (5 레시피 이름)."
    : ui.invalid.size ? "고칠 입력이 있습니다. 옵션·보정 칸의 안내를 보세요."
    : busy ? "실행 중입니다. 끝나면 다시 실행할 수 있습니다." : server.reason;
  ui.gpuNote.classList.toggle("warn", !server.live);
}

async function submitJob() {
  const tasks = ui.gpuTasks.map((label) => label.querySelector("input")).filter((input) => input.checked).map((input) => input.dataset.task);
  job = {name: state.name, grid: copyGrid(state.grid), tasks, id: null, record: null, error: null};
  renderJob();
  try {
    const response = await fetch("api/quantizer/jobs", {method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({name: state.name, yaml: currentYaml(), tasks, limit: Number(ui.gpuLimit.value)})});
    const answer = await response.json();
    if (!response.ok) throw new Error(answer.error?.message ?? `요청이 ${response.status} 로 끝났습니다`);
    job.id = answer.id;
    await pollJob();
  } catch (error) {
    job.error = error.message;
    renderJob();
  }
}

async function pollJob() {
  const response = await fetch(`api/quantizer/jobs/${job.id}`, {headers: {accept: "application/json"}});
  const record = await response.json();
  if (!response.ok) throw new Error(record.error?.message ?? `상태 요청이 ${response.status} 로 끝났습니다`);
  job.record = record;
  renderJob();
  if (!["done", "error"].includes(record.status)) setTimeout(() => pollJob().catch((error) => {
    job.error = error.message;
    renderJob();
  }), 1500);
}

function jobStages(tasks) {
  const lm = tasks.some((task) => !["ppl", "report"].includes(task));
  return ["대기", "모델 불러오기", tasks.includes("report") ? "레이어별 리포트" : null, tasks.some((t) => t !== "report") ? "패치" : null,
    tasks.includes("ppl") ? "WikiText-2 PPL" : null, lm ? "lm-eval" : null, "완료"].filter(Boolean);
}

function renderJob() {
  updateGpu();
  if (!job) return;
  const record = job.record;
  const status = job.error ? "error" : record?.status ?? "queued";
  const stages = jobStages(job.tasks);
  const at = status === "done" ? stages.length - 1 : status === "queued" ? 0 : Math.max(1, stages.indexOf(record?.stage === "보정" ? "패치" : record?.stage));
  const parts = [h("div", {class: "pipeline", role: "list", "aria-label": "진행 단계"}, stages.map((stage, i) =>
    h("span", {role: "listitem", class: i < at ? "done" : i === at && status !== "error" ? "now" : null}, stage)))];
  if (status === "running" || status === "queued") {
    parts.push(h("div", {class: "qz-progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
      "aria-valuenow": String(Math.round((record?.progress ?? 0) * 100))}, h("i", {style: `width: ${((record?.progress ?? 0) * 100).toFixed(0)}%`})),
    h("p", {class: "qz-cap"}, status === "queued" ? "서버 큐에서 차례를 기다립니다." : `${record.stage ?? "실행"} 중 · ${job.name}`));
  }
  if (status === "error") {
    parts.push(h("p", {class: "note warn", role: "alert"}, `실행하지 못했습니다: ${job.error ?? record?.error?.message ?? "알 수 없는 오류"}`));
  }
  if (status === "done") {
    const results = record.results ?? {};
    parts.push(h("p", {class: "qz-match"}, "서버 실행 결과 · ", h("b", null, job.name), ` · job ${record.id.slice(0, 8)}`),
      ...(results.run || results.report ? resultBody({run: results.run, report: results.report, grid: job.grid}) : []),
      ...(results.lmeval ? [h("h3", {class: "qz-sub"}, "벤치마크 (E2E)"), ...benchBody(results.lmeval)] : []));
    const command = record.evidence?.layer_map_command;
    if (command) {
      parts.push(h("p", {class: "qz-cap"}, "레이어 지도에 더하려면 저장소 루트에서:"),
        h("div", {class: "qz-cmds"}, h("li", null, h("code", {class: "qz-cmd"}, command), copyButton("복사", command))));
    }
  }
  fill(ui.gpuJob, ...parts);
}

function updateResults() {
  let recipe;
  try {
    recipe = currentRecipe();
  } catch (error) {
    ui.yaml.textContent = `레시피를 만들지 못했습니다: ${error.message}`;
    return;
  }
  ui.yaml.textContent = toYaml(recipe);
  const opts = pageOptions();
  const calibrated = updateSummary(recipe, opts);
  commands().forEach(([, text], i) => (ui.commandCodes[i].textContent = text));
  ui.api.textContent = apiSnippet();
  const {settings} = calibration(opts);
  const source = `${data.calibration_info.dataset.choices[settings.dataset]} train 에서 ${count(settings.samples)}개 창 × `
    + `${count(settings.seqlen)} 토큰 = ${count(settings.samples * settings.seqlen)} 토큰, seed ${settings.seed}`
    + (settings.sequential ? ", 순차 보정" : "");
  ui.calibNote.textContent = calibrated.length ? `보정이 필요한 레시피입니다. 보정 데이터: ${source} (YAML calibration). `
    + "CLI 에서는 --calib-samples 등으로 덮어쓸 수 있습니다." : "";
  ui.calibStatus.textContent = calibrated.length ? `보정 필요 — ${calibrated.map((m) => byId[m].label).join(", ")} · ${source}`
    : Object.keys(pageCalibration()).length ? "지금 레시피는 보정이 필요 없습니다. 바꾼 값은 YAML calibration 에 적히지만 쓰이지 않습니다."
      : "지금 레시피는 보정이 필요 없습니다. 보정이 필요한 방법을 쓰면 이 값으로 calibration 이 자동으로 적힙니다.";
  ui.calibStatus.classList.toggle("qz-need", calibrated.length > 0);
  updateRecord();
  updateGpu();
}

// Option controls, the changed-option dots and counts, and the calibration badges follow the state.
function updateOptions() {
  const opts = pageOptions();
  for (const entry of fields) paint(entry);
  for (const method of palette) {
    const n = Object.keys(opts[method] ?? {}).length;
    for (const dot of root.querySelectorAll(`[data-optdot="${method}"]`)) dot.hidden = !n;
    for (const label of root.querySelectorAll(`[data-optn="${method}"]`)) label.textContent = n ? ` · ${n}개 바꿈` : "";
    const badge = root.querySelector(`[data-calib="${method}"]`);
    if (badge) badge.hidden = !needsCalibration(method, opts[method]);
  }
  const kvChanged = Object.keys(pageKvOptions()).length;
  ui.kvToggle.hidden = !Object.keys(KV_OPTIONS[state.kv] ?? {}).length;
  if (ui.kvToggle.hidden && ui.kvToggle.getAttribute("aria-expanded") === "true") ui.kvToggle.click();
  root.querySelector('[data-optdot="kv"]').hidden = !kvChanged;
  root.querySelector('[data-optn="kv"]').textContent = kvChanged ? " · 바꿈" : "";
}

function updateGrid() {
  const [fb, fk] = state.focus;
  ui.cells.forEach((cell, c) => {
    const b = Math.floor(c / KINDS.length);
    const k = c % KINDS.length;
    const m = state.grid[b][k];
    if (cell.dataset.m !== m) {
      cell.dataset.m = m;
      cell.textContent = byId[m].code;
    }
    cell.setAttribute("aria-selected", String(state.sel.has(c)));
    cell.setAttribute("aria-label", `블록 ${b} ${KINDS[k]}: ${byId[m].label}`);
    cell.tabIndex = b === fb && k === fk ? 0 : -1;
  });
  const n = state.sel.size;
  ui.count.textContent = n ? `선택 ${n}칸` : "선택 없음";
  for (const button of ui.apply) button.disabled = !n;
  ui.status.textContent = state.note;
}

function update() {
  const matched = matchedPreset();
  ui.presets.forEach((button, i) => button.setAttribute("aria-pressed", String(data.presets[i] === matched)));
  for (const button of ui.defaults.querySelectorAll("[role=radio]")) {
    const on = button.dataset.default === state.def;
    button.setAttribute("aria-checked", String(on));
    button.tabIndex = on ? 0 : -1;
  }
  for (const button of ui.kv.querySelectorAll("[role=radio]")) {
    const on = button.dataset.kv === state.kv;
    button.setAttribute("aria-checked", String(on));
    button.tabIndex = on ? 0 : -1;
  }
  ui.kvNote.textContent = data.kv.find((item) => item.id === state.kv).description;
  if (document.activeElement !== ui.name) ui.name.value = state.name;
  ui.undo.disabled = !state.history.length;
  updateOptions();
  updateGrid();
  updateResults();
  updateNameError();
  if (ui.live) ui.live.textContent = state.note;
}

function render() {
  const gridSection = gridStep();
  gridSection.classList.add("qz-grid-step");
  ui.live = h("p", {class: "sr-only", "aria-live": "polite"});
  root.replaceChildren(h("div", {class: "qz-layout"},
    h("div", {class: "qz-config", "aria-label": "레시피 구성"}, presetStep(), defaultStep(), gridSection, kvStep(), nameStep()),
    h("aside", {class: "qz-results", "aria-label": "결과"}, summaryPanel(), calibPanel(), recordPanel(), benchPanel(), yamlPanel(),
      commandPanel(), gpuPanel())), ui.live);
  update();
}

// ---------- boot ----------
function stateView() {
  return {name: state.name, default: state.def, grid: copyGrid(state.grid), kv: state.kv, ...extra(),
    needs_calibration: calibration().methods.length > 0, invalid: [...ui.invalid.keys()],
    selection: [...state.sel].sort((a, b) => a - b).map((c) => [Math.floor(c / KINDS.length), c % KINDS.length]),
    matched: matchedPreset()?.id ?? null, history: state.history.length, trusted, yaml: (() => {
      try {
        return currentYaml();
      } catch {
        return null;
      }
    })()};
}

// options / calibration / kv_options as build() takes them (kv_options for the resulting KV preset).
function setState({name, default: def, grid, kv, options, calibration: calib, kv_options: kvOptions} = {}) {
  commit(() => {
    if (def !== undefined) state.def = def;
    if (grid !== undefined) state.grid = copyGrid(grid);
    if (kv !== undefined) state.kv = kv;
    if (name !== undefined) Object.assign(state, {name, nameAuto: false});
    if (options !== undefined) state.options = structuredClone(options);
    if (calib !== undefined) state.calib = {...calib};
    if (kvOptions !== undefined) state.kvOptions = {...state.kvOptions, [state.kv]: {...kvOptions}};
  }, "상태를 바꿨습니다");
  return stateView();
}

function failure(title, message, command) {
  const parts = [h("h2", null, title), h("p", null, message)];
  if (command) parts.push(h("pre", {class: "code"}, command), copyButton("명령 복사", command));
  root.replaceChildren(h("section", {class: "state", role: "alert"}, parts));
}

async function boot() {
  window.__quantizer = {build, toYaml, setState, getState: stateView, normalizeOptions, variant, needsCalibration};
  let response;
  try {
    response = await fetch(DATA, {headers: {accept: "application/json"}});
  } catch (error) {
    failure("프리셋 데이터를 읽지 못했습니다", `${DATA}: ${error.message}`);
    return;
  }
  if (response.status === 404) {
    failure("Quantizer 데이터가 아직 없습니다", "프리셋과 방법 목록(presets.json)을 내보내면 이 화면이 열립니다. 저장소 루트에서 실행하세요.", EXPORT);
    return;
  }
  try {
    if (!response.ok) throw new Error(`${DATA} 요청이 ${response.status} 로 끝났습니다`);
    data = await response.json();
    if (data.schema !== SCHEMA) throw new Error(`presets.json 형식이 ${data.schema ?? "알 수 없음"} 입니다. ${SCHEMA} 형식으로 다시 내보내세요.`);
  } catch (error) {
    failure("Quantizer 데이터를 열지 못했습니다", error.message, EXPORT);
    return;
  }
  useGenerator(data.generator);
  state.grid = uniform(state.def);
  byId = Object.fromEntries(data.methods.map((m) => [m.id, m]));
  palette = data.methods.map((m) => m.id);
  trusted = data.presets.every((p) => toYaml(build(p.id, p.default, p.grid, p.kv)) === p.yaml)
    && palette.length === Object.keys(METHODS).length && palette.every((m) => has(METHODS, m));
  render();
  document.addEventListener("keydown", onKey);
  probeServer();
}

if (location.protocol !== "file:") boot();
