// The exact knapsack (js/mckp.js) off the main thread for the tuning workspace (js/tune.js): a solve takes seconds
// for 196 layers, and the page stays responsive meanwhile. {id, table, budget} → {id, solution, seconds} or
// {id, error}; the solution's BigInts cross as they are (structured clone).
import {parseBits, solve} from "./mckp.js";

self.addEventListener("message", ({data}) => {
  const {id, table, budget} = data;
  const started = performance.now();
  try {
    const solution = solve(table, parseBits(budget));
    self.postMessage({id, solution, seconds: (performance.now() - started) / 1000});
  } catch (error) {
    self.postMessage({id, error: error.message});
  }
});
