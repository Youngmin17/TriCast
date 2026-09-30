"""One markdown table from several `tricast run` output directories.

    python scripts/e2e/summarize.py runs/e2e/qwen3_0.6b_ppl_a runs/e2e/qwen3_0.6b_ppl_b
    python scripts/e2e/summarize.py runs/e2e/qwen3_0.6b_lmeval_a runs/e2e/qwen3_0.6b_lmeval_b

Each directory holds one JSON record per recipe (see tricast.eval.runner). Perplexity runs give
the perplexity of every complete run next to the unpatched model (`native`), the relative change
and the wall time; lm-eval runs give each task's headline metrics with lm-eval's standard error.
Failed runs are listed with their error, followed by the environment of the runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Headline metrics per lm-eval task, in table order.
LM_EVAL_METRICS = {"hellaswag": ("acc_norm",), "coqa": ("em", "f1"), "arc_easy": ("acc_norm",),
                   "arc_challenge": ("acc_norm",), "piqa": ("acc_norm",), "winogrande": ("acc",)}


def load(directories: list[str]) -> list[dict]:
    records = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            if path.name != "env.json":
                records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def unique(records: list[dict]) -> list[dict]:
    """Native first, then each recipe once (the first directory that has it)."""
    seen, result = set(), []
    for record in sorted(records, key=lambda r: r["recipe"]["name"] != "native"):
        if record["recipe"]["name"] not in seen:
            seen.add(record["recipe"]["name"])
            result.append(record)
    return result


def ppl_table(records: list[dict]) -> None:
    natives = {r["metrics"]["ppl"]["ppl"] for r in records
               if r["recipe"]["name"] == "native" and r["status"] == "complete"}
    if len(natives) != 1:
        raise SystemExit(f"expected one native perplexity across directories, got {sorted(natives)}")
    native = natives.pop()
    print("| Recipe | PPL | vs native | Wall time |")
    print("| --- | ---: | ---: | ---: |")
    for record in unique(records):
        name = record["recipe"]["name"]
        if record["status"] != "complete":
            print(f"| {name} | failed: {record.get('error', '')[:80]} | | |")
            continue
        ppl = record["metrics"]["ppl"]["ppl"]
        delta = "" if name == "native" else f"{100 * (ppl / native - 1):+.2f}%"
        print(f"| {name} | {ppl:.4f} | {delta} | {record['wall_time_s'] / 60:.1f} min |")
    ppl = next(r for r in records if r["status"] == "complete")["metrics"]["ppl"]
    print(f"\n{ppl['n_windows']} windows, {ppl['n_tokens']} scored tokens, dataset fingerprint "
          f"{ppl['dataset_fingerprint']}.")


def lm_eval_table(records: list[dict]) -> None:
    complete = [r for r in records if r["status"] == "complete"]
    tasks = [task for task in LM_EVAL_METRICS if task in complete[0]["metrics"]["lm_eval"]["results"]]
    columns = [(task, metric) for task in tasks for metric in LM_EVAL_METRICS[task]]
    print("| Recipe | " + " | ".join(f"{task} {metric}" for task, metric in columns)
          + " | Batch size | Wall time |")
    print("| --- |" + " ---: |" * len(columns) + " ---: | ---: |")
    for record in unique(records):
        name = record["recipe"]["name"]
        if record["status"] != "complete":
            print(f"| {name} | failed: {record.get('error', '')[:80]} |" + " |" * (len(columns) + 1))
            continue
        lm_eval = record["metrics"]["lm_eval"]
        cells = []
        for task, metric in columns:
            result = lm_eval["results"][task]
            cells.append(f"{result[f'{metric},none']:.4f} ± {result[f'{metric}_stderr,none']:.4f}")
        used = (lm_eval.get("batch_size") or {}).get("used", lm_eval["config"]["batch_size"])
        print(f"| {name} | " + " | ".join(cells) + f" | {used} | {record['wall_time_s'] / 60:.1f} min |")
    samples = complete[0]["metrics"]["lm_eval"]["n-samples"]
    print("\n" + ", ".join(f"{task}: {samples[task]['effective']} of {samples[task]['original']} items"
                           for task in tasks) + ".")


def main() -> None:
    records = load(sys.argv[1:])
    if not any(r["status"] == "complete" for r in records):
        raise SystemExit("no complete run in these directories")
    if any("lm_eval" in (r.get("metrics") or {}) for r in records):
        lm_eval_table(records)
    else:
        ppl_table(records)
    envs = [r["env"] for r in records if r["status"] == "complete"]
    env = envs[0]
    shas = sorted({f"{e['git_sha'][:7]}{' (dirty)' if e['git_dirty'] else ''}" for e in envs})
    gpus = sorted({name for e in envs for name in e["gpu_names"]})
    print(f"Model {env['model_id']}@{env['model_sha'][:12]}, git {', '.join(shas)}, {', '.join(gpus)}, "
          f"CUDA {env['cuda']}, torch {env['versions']['torch']}, triton {env['versions']['triton']}, "
          f"transformers {env['versions']['transformers']}, lm_eval {env['versions'].get('lm_eval')}.")


if __name__ == "__main__":
    main()
