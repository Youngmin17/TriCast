"""One markdown table from several `tricast run` output directories.

    python scripts/e2e/summarize.py runs/e2e/qwen3_0.6b_ppl_a runs/e2e/qwen3_0.6b_ppl_b

Each directory holds one JSON record per recipe (see tricast.eval.runner). The table lists the
perplexity of every complete run next to the unpatched model (`native`), the relative change, the
wall time, and the environment the runs share; failed runs are listed with their error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(directories: list[str]) -> list[dict]:
    records = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            if path.name != "env.json":
                records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def main() -> None:
    records = sorted(load(sys.argv[1:]), key=lambda r: r["recipe"]["name"] != "native")
    natives ={r["metrics"]["ppl"]["ppl"] for r in records
               if r["recipe"]["name"] == "native" and r["status"] == "complete"}
    if len(natives) != 1:
        raise SystemExit(f"expected one native perplexity across directories, got {sorted(natives)}")
    native = natives.pop()
    print("| Recipe | PPL | vs native | Wall time |")
    print("| --- | ---: | ---: | ---: |")
    seen = set()
    for record in records:
        name = record["recipe"]["name"]
        if name in seen:
            continue
        seen.add(name)
        if record["status"] != "complete":
            print(f"| {name} | failed: {record.get('error', '')[:80]} | | |")
            continue
        ppl = record["metrics"]["ppl"]["ppl"]
        delta = "" if name == "native" else f"{100 * (ppl / native - 1):+.2f}%"
        print(f"| {name} | {ppl:.4f} | {delta} | {record['wall_time_s'] / 60:.1f} min |")
    env = records[0]["env"]
    ppl = records[0]["metrics"]["ppl"]
    print(f"\n{ppl['n_windows']} windows, {ppl['n_tokens']} scored tokens, dataset fingerprint "
          f"{ppl['dataset_fingerprint']}. Model {env['model_id']}@{env['model_sha'][:12]}, "
          f"git {env['git_sha'][:12]} (dirty={env['git_dirty']}), {', '.join(env['gpu_names'])}, "
          f"CUDA {env['cuda']}, torch {env['versions']['torch']}, triton {env['versions']['triton']}, "
          f"transformers {env['versions']['transformers']}.")


if __name__ == "__main__":
    main()
