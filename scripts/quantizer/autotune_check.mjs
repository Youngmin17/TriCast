// Checks for the Auto-tune page's solver (app/web/js/mckp.js, Node >= 22, no browser): for every budget, the JS
// choice is the Python solver's (`python -m app.autotune solve`) layer by layer, with the same value sum (float64
// equal) and the same exact bits; a budget below the cheapest assignment is refused by both.
// Prints PASS/FAIL per budget and AUTOTUNE_PARITY n/m.
//
// usage (from the repo root, python = an env that runs app.autotune solve):
//   node scripts/quantizer/autotune_check.mjs <out_dir> <python> <table.json> [budget ...]
import {execFileSync} from "node:child_process";
import {mkdirSync, readFileSync} from "node:fs";
import {pathToFileURL} from "node:url";
import {resolve} from "node:path";

const [, , outDir, python, tablePath, ...given] = process.argv;
const budgets = given.length ? given : ["4", "4.2", "4.25", "133/32", "9/2", "4.75", "5.09", "6", "8", "16"];
mkdirSync(outDir, {recursive: true});
const repo = resolve(".");
const py = (args) => execFileSync(python, args, {cwd: repo, encoding: "utf-8", maxBuffer: 1 << 28,
  env: {...process.env, PYTHONIOENCODING: "utf-8", PYTHONPATH: `${repo}/src:${repo}`}, stdio: ["ignore", "pipe", "pipe"]});
const {solve, parseBits, summary, SolveError} = await import(pathToFileURL(`${repo}/app/web/js/mckp.js`).href);

py(["-m", "app.export_autotune_demo", "--table", tablePath, "--out", `${outDir}/bundle`]);
const {table} = JSON.parse(readFileSync(`${outDir}/bundle/bundle.json`, "utf-8"));
const results = [];
for (const [index, budget] of budgets.entries()) {
  let pyResult;
  try {
    py(["-m", "app.autotune", "solve", "--table", tablePath, "--budget-bits", budget, "--out", `${outDir}/py_${index}`]);
    pyResult = JSON.parse(readFileSync(`${outDir}/py_${index}/solution.json`, "utf-8"));
  } catch (error) {
    pyResult = {refused: String(error.stderr ?? error.message).trim().split("\n").pop()};
  }
  let jsResult;
  const started = performance.now();
  try {
    jsResult = solve(table, parseBits(budget));
  } catch (error) {
    if (!(error instanceof SolveError)) throw error;
    jsResult = {refused: error.message};
  }
  const ms = performance.now() - started;
  let pass;
  let detail;
  if (pyResult.refused || jsResult.refused) {
    pass = Boolean(pyResult.refused && jsResult.refused);
    detail = `python: ${pyResult.refused ?? "solved"} | js: ${jsResult.refused ?? "solved"}`;
  } else {
    const pyMethods = pyResult.layers.map((layer) => layer.method);
    const jsMethods = jsResult.choice.map((j) => table.methods[j]);
    const differ = pyMethods.filter((method, i) => method !== jsMethods[i]).length;
    const {bits} = summary(table, jsResult);
    const bitsEqual = Math.abs(bits - pyResult.achieved.avg_bits) < 1e-9;
    pass = differ === 0 && jsResult.value === pyResult.layers.reduce((sum, layer) => sum + layer.value, 0) && bitsEqual;
    detail = `layers differing ${differ}/${pyMethods.length}, value js ${jsResult.value} py-sum ` +
             `${pyResult.layers.reduce((sum, layer) => sum + layer.value, 0)}, bits js ${bits.toFixed(6)} py ` +
             `${pyResult.achieved.avg_bits.toFixed(6)}, js ${ms.toFixed(0)} ms (${jsResult.capacity + 1} units)`;
  }
  results.push(pass);
  console.log(`${pass ? "PASS" : "FAIL"} budget ${budget}: ${detail}`);
}
console.log(`AUTOTUNE_PARITY ${results.filter(Boolean).length}/${results.length}`);
process.exit(results.every(Boolean) ? 0 : 1);
