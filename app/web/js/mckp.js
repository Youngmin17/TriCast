// The exact multiple-choice knapsack of `python -m app.autotune solve` for the Auto-tune page (no DOM). Same
// integer units (costs = params x stored bits from WEIGHT_BITS, exact as BigInt; each layer's costs above its
// cheapest option, which every layer pays anyway, divided by the GCD of every nonzero such cost),
// same float64 sums row by row, same ties: the smallest total cost, then, deciding rows from the last one, the
// earliest option of the row. scripts/quantizer/autotune_check.mjs compares every choice with the Python solver.

const MAX_DP_CELLS = 2 ** 28;  // app/autotune.MAX_DP_CELLS

export class SolveError extends Error {}

function gcd(a, b) {
  while (b) [a, b] = [b, a % b];
  return a < 0n ? -a : a;
}

function lcm(a, b) {
  return a / gcd(a, b) * b;
}

// Stored bits per weight as a fraction [numerator, denominator]: element + scale / (per or in_features).
export function weightBits([element, scale, per], inFeatures) {
  const den = BigInt(per ?? inFeatures);
  return [BigInt(element) * den + BigInt(scale), den];
}

// A budget in bits per weight written as a decimal string or "p/q", as an exact fraction.
export function parseBits(text) {
  const value = String(text).trim();
  const ratio = /^(\d+)\s*\/\s*(\d+)$/.exec(value);
  if (ratio) {
    if (BigInt(ratio[2]) === 0n) throw new SolveError(`budget ${value}: zero denominator`);
    return [BigInt(ratio[1]), BigInt(ratio[2])];
  }
  const decimal = /^(\d+)(?:\.(\d+))?$/.exec(value);
  if (!decimal) throw new SolveError(`budget ${value}: write a number such as 3.15 or a fraction such as 403/128`);
  const digits = decimal[2] ?? "";
  return [BigInt(decimal[1] + digits), 10n ** BigInt(digits.length)];
}

// The per-layer options and their exact costs: options[i] = method indices the table has a value for (table order),
// costs[i][j] = params x bits of option j scaled by `scale` (the LCM of every bits denominator) into a BigInt.
export function problem(table) {
  const {layers, methods, weight_bits: weightTable} = table;
  const bits = layers.map((layer) => methods.map((m) => weightBits(weightTable[m], layer.in_features)));
  let scale = 1n;
  for (const row of bits) for (const [, den] of row) scale = lcm(scale, den);
  const options = layers.map((layer) => methods.map((_, j) => j).filter((j) => layer.value[j] !== null));
  const costs = layers.map((layer, i) => options[i].map((j) => {
    const [num, den] = bits[i][j];
    return BigInt(layer.params) * num * (scale / den);
  }));
  const params = layers.reduce((sum, layer) => sum + BigInt(layer.params), 0n);
  return {options, costs, scale, params};
}

// Solve ``table`` (bundle form: layers[i].value[j] = the solver's value of method j, null = not tabled) within
// `budget` = [p, q] bits per weight. Returns the chosen method index per layer, the value sum and the exact cost.
export function solve(table, budget) {
  const {options, costs, scale, params} = problem(table);
  const [p, q] = budget;
  if (q <= 0n || p <= 0n) throw new SolveError("the budget must be positive");
  // budget in cost units: p / q x params x scale (kept as the fraction num / q)
  const num = p * params * scale;
  const cheapest = costs.reduce((sum, row) => sum + row.reduce((a, b) => (b < a ? b : a)), 0n);
  if (cheapest * q > num) {
    const bitsCheapest = Number(cheapest * 1000000n / (params * scale)) / 1e6;
    throw new SolveError(`infeasible budget: ${Number(p * 1000000n / q) / 1e6} bits is below the cheapest ` +
                         `assignment of these methods, ${bitsCheapest} bits`);
  }
  const lowest = costs.map((row) => row.reduce((a, b) => (b < a ? b : a)));
  const excess = costs.map((row, i) => row.map((c) => c - lowest[i]));
  let step = 0n;
  for (const row of excess) for (const c of row) if (c) step = gcd(step, c);
  if (step === 0n) step = 1n;
  const units = excess.map((row) => row.map((c) => Number(c / step)));
  let capacity = (num - q * cheapest) / (q * step);  // floor of what the budget leaves above the cheapest
  const dearest = units.reduce((sum, row) => sum + Math.max(...row), 0);
  if (capacity > BigInt(dearest)) capacity = BigInt(dearest);
  const cap = Number(capacity);
  const rows = units.length;
  if (rows * (cap + 1) > MAX_DP_CELLS) throw new SolveError(`${rows} rows x ${cap + 1} budget units is too large`);
  let dp = new Float64Array(cap + 1).fill(Infinity);
  dp[0] = 0;
  const picks = new Int8Array(rows * (cap + 1)).fill(-1);
  for (let i = 0; i < rows; i++) {
    const next = new Float64Array(cap + 1).fill(Infinity);
    const values = options[i].map((j) => table.layers[i].value[j]);
    const base = i * (cap + 1);
    for (let k = 0; k < units[i].length; k++) {
      const u = units[i][k];
      if (u > cap) continue;
      const v = values[k];
      for (let t = 0; t + u <= cap; t++) {
        const candidate = dp[t] + v;
        if (candidate < next[t + u]) {
          next[t + u] = candidate;
          picks[base + t + u] = k;
        }
      }
    }
    dp = next;
  }
  let left = 0;
  for (let t = 1; t <= cap; t++) if (dp[t] < dp[left]) left = t;  // the first minimum
  if (!Number.isFinite(dp[left])) throw new SolveError("no choice fits the budget");
  const choice = new Array(rows);
  for (let i = rows - 1; i >= 0; i--) {
    const k = picks[i * (cap + 1) + left];
    choice[i] = options[i][k];
    left -= units[i][k];
  }
  let value = 0;
  let cost = 0n;
  for (let i = 0; i < rows; i++) {
    const k = options[i].indexOf(choice[i]);
    value += table.layers[i].value[choice[i]];
    cost += costs[i][k];
  }
  return {choice, value, cost, scale, params, step, capacity: cap};
}

// Average stored bits of a choice as a float, and the method counts in table order.
export function summary(table, solution) {
  const counts = table.methods.map((_, j) => solution.choice.filter((c) => c === j).length);
  const bits = Number(solution.cost * 1000000000n / (solution.params * solution.scale)) / 1e9;
  return {bits, counts};
}
