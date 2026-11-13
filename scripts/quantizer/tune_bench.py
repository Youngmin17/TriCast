"""Benchmarks beyond WikiText-2 for the tuning app's offline comparison of the existing way (one method on
every layer) and ours (the Auto-tune assignment) at the same size (app/web/tune.html, a model with a
benchmarks file).

    # 1 the page's own recipe file for every setup of a bundle (one method on every layer, Auto-tune),
    #   checked byte for byte against the file the bundle's PPL was measured with (recipe_file_sha256)
    python scripts/quantizer/tune_bench.py recipes --bundle app/web/demo/autotune/<model>/bundle.json \
        --out DIR
    # 2 on a GPU node (tricast env): per recipe file, a fresh model prepared as tune_outputs.py record
    #   prepares it (patch, calibration), then WikiText-2 and C4 perplexity and the lm-eval tasks of
    #   BENCHMARKS, every document's result kept for paired differences; --native also records the
    #   unquantized model (BF16)
    python scripts/quantizer/tune_bench.py record --model M --revision R --out BENCH [--native] \
        [--calib-cache DIR] DIR/*.yaml
    # 3 here: one JSON for the page — setups keyed by recipe-file SHA-256, and for each Auto-tune size the
    #   best one-method setup within that size on each benchmark, with the paired difference and its
    #   standard error
    python scripts/quantizer/tune_bench.py export --bench BENCH --bundle …/bundle.json --out …/benchmarks.json

``record`` prints ``BENCH_PART`` per benchmark and ends with ``BENCH_DONE ok=<n> failed=<k>``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import tune_outputs as to  # noqa: E402  (the same model preparation as the answers)

SCHEMA = "tricast.tune_bench/1"
# kind ppl: lower is better, no per-document results; kind acc: an lm-eval task (a group's score is lm-eval's
# size-weighted mean over its subtasks), higher is better, ``metric``/``filter`` name the lm-eval result key
BENCHMARKS: tuple[dict[str, Any], ...] = (
    {"id": "wikitext2", "label": "WikiText-2", "group": "언어 모델링", "kind": "ppl",
     "what": "위키백과 글의 다음 토큰 (test 전체, 2048 토큰 창)"},
    {"id": "c4", "label": "C4", "group": "언어 모델링", "kind": "ppl",
     "what": "웹 문서의 다음 토큰 (validation 앞 1,100 문서, 2048 토큰 창 256개 — GPTQ 규칙)"},
    {"id": "lambada_openai", "label": "LAMBADA", "group": "언어 모델링", "kind": "acc", "metric": "acc",
     "num_fewshot": 0, "limit": 1000, "what": "긴 문맥 끝의 마지막 단어 맞히기"},
    {"id": "arc_easy", "label": "ARC-Easy", "group": "상식 · 추론", "kind": "acc", "metric": "acc_norm",
     "num_fewshot": 0, "limit": 1000, "what": "초등 과학 객관식"},
    {"id": "arc_challenge", "label": "ARC-Challenge", "group": "상식 · 추론", "kind": "acc",
     "metric": "acc_norm", "num_fewshot": 0, "limit": None, "what": "어려운 초등 과학 객관식"},
    {"id": "hellaswag", "label": "HellaSwag", "group": "상식 · 추론", "kind": "acc", "metric": "acc_norm",
     "num_fewshot": 0, "limit": 1000, "what": "이어질 장면 고르기 (4지선다)"},
    {"id": "piqa", "label": "PIQA", "group": "상식 · 추론", "kind": "acc", "metric": "acc_norm",
     "num_fewshot": 0, "limit": 1000, "what": "물리 상식 (2지선다)"},
    {"id": "winogrande", "label": "WinoGrande", "group": "상식 · 추론", "kind": "acc", "metric": "acc",
     "num_fewshot": 0, "limit": None, "what": "대명사가 가리키는 것 (2지선다)"},
    # lm-eval's limit is per subtask: 20 from each of the 57 subjects
    {"id": "mmlu", "label": "MMLU", "group": "지식", "kind": "acc", "metric": "acc", "num_fewshot": 0,
     "limit": 20, "what": "57개 분야 4지선다 (분야마다 앞 20문항)"},
)
PPL_SEQLEN = 2048
# C4 as tricast.eval.ppl's built-in "c4" (GPTQ get_c4_new): the first 1100 documents of validation shard 0
# joined with one space, at most 256 windows. The built-in loader streams the shard through datasets, which
# under datasets 4.x finds no offline cache for it; the shard file itself is in the hub cache, so it is read
# directly at a pinned revision. The joined text is byte-identical to the stream's (SHA-256 d7b24fe9…,
# 2,410,955 characters, checked 2026-11-12 on the cluster); every record stores the text's SHA-256.
C4_REPO, C4_REVISION = "allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_FILE = "en/c4-validation.00000-of-00008.json.gz"
C4_DOCS, C4_WINDOWS = 1100, 256
C4_TEXT_SHA256 = "d7b24fe935a0504319ac236f317b0e5b255800af5f7491dc1d8d58c2b48a52d5"
ENV_KEYS = ("gpu_names", "git_sha", "git_dirty", "model_sha", "versions", "utc", "hostname")


def recipes(bundle_path: Path, out: Path) -> None:
    """``<id>.yaml`` for every baseline and solution of the bundle, through the page's own builder; refuses a
    file whose SHA-256 is not the one the bundle's PPL was measured with."""
    from app import autotune_recipe as ar
    from app import quantizer_recipe as qr

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    blocks = bundle["table"]["blocks"]
    out.mkdir(parents=True, exist_ok=True)
    rows = [(row, qr.uniform(row["method"], blocks)) for row in bundle["baselines"]]
    rows += [(row, row["grid"]) for row in bundle["solutions"] if (row.get("kv") or "none") == "none"]
    for row, grid in rows:
        recipe = ar.grid_recipe(bundle, grid)
        if recipe["sha256"] != row.get("recipe_file_sha256"):
            raise SystemExit(
                f"{row['id']}: the page builds {recipe['sha256'][:12]}, the bundle measured "
                f"{str(row.get('recipe_file_sha256'))[:12]}"
            )
        target = out / f"{row['id']}.yaml"
        target.write_text(recipe["yaml"], encoding="utf-8")
        print(f"{target} sha256={recipe['sha256']}")


def c4_text() -> str:
    """The joined C4 evaluation text (see C4_FILE); refuses any other text."""
    import gzip

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(C4_REPO, C4_FILE, repo_type="dataset", revision=C4_REVISION)
    docs: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            docs.append(json.loads(line)["text"])
            if len(docs) == C4_DOCS:
                break
    text = " ".join(docs)
    if to._sha256(text.encode("utf-8")) != C4_TEXT_SHA256:
        raise ValueError("the C4 validation documents differ from the checked ones (C4_TEXT_SHA256)")
    return text


def _leaves(result: dict[str, Any], task: str) -> list[str]:
    """The leaf tasks whose documents make up ``task`` (a group's subtasks, recursively; sorted)."""
    children = (result.get("group_subtasks") or {}).get(task)
    if not children:
        return [task]
    return sorted(leaf for child in children for leaf in _leaves(result, child))


def _docs(samples: dict[str, list[dict[str, Any]]], leaves: list[str], metric: str, filt: str) -> str:
    """Each document's 0/1 result, leaves in order and documents by ``doc_id``."""
    out = []
    for leaf in leaves:
        rows = sorted(
            (s for s in samples[leaf] if s.get("filter", "none") == filt), key=lambda s: s["doc_id"]
        )
        for sample in rows:
            value = sample[metric]
            if value not in (0, 1):
                raise ValueError(f"{leaf} doc {sample['doc_id']}: {metric}={value!r} is not 0 or 1")
            out.append("1" if value else "0")
    return "".join(out)


def _lmeval(
    model: Any, tokenizer: Any, specs: list[dict[str, Any]], limit: int | None, batch_size: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The lm-eval tasks of ``specs`` (one few-shot setting) on the prepared model, with per-document
    results."""
    import torch

    from tricast.eval.lmeval import evaluate

    setting = {(spec["num_fewshot"], spec.get("limit")) for spec in specs}
    if len(setting) != 1:
        raise ValueError("one lm-eval call takes one few-shot setting and one document limit")
    shots, task_limit = setting.pop()
    if limit is not None:  # a timing check: fewer documents still
        task_limit = limit if task_limit is None else min(limit, task_limit)
    torch.manual_seed(42)  # lm-eval's seeds come from torch.initial_seed()
    result = evaluate(
        model,
        tokenizer,
        [spec["id"] for spec in specs],
        num_fewshot=shots,
        limit=task_limit,
        batch_size=batch_size,
        log_samples=True,
        bootstrap_iters=0,  # no forked bootstrap (tricast.eval.lmeval.evaluate); SEs from the documents
    )
    out = {}
    for spec in specs:
        task, metric, filt = spec["id"], spec["metric"], spec.get("filter", "none")
        scores = result["results"][task]
        leaves = _leaves(result, task)
        docs = _docs(result["samples"], leaves, metric, filt)
        value = scores[f"{metric},{filt}"]
        # firing check: the stored documents are the ones lm-eval scored (a group's score is its
        # size-weighted mean)
        if not docs or abs(docs.count("1") / len(docs) - value) > 1e-9:
            raise ValueError(
                f"{task}: {len(docs)} documents average {docs.count('1') / max(len(docs), 1)}, "
                f"lm-eval reports {value}"
            )
        out[task] = {
            "metric": metric,
            "filter": filt,
            "value": value,
            "stderr": mean_stderr(docs),
            "n": len(docs),
            "n_total": sum(int(result["n-samples"][leaf]["original"]) for leaf in leaves),
            "limit": task_limit,
            "num_fewshot": result["n-shot"].get(leaves[0]),
            "versions": {leaf: result["versions"].get(leaf) for leaf in leaves},
            "fingerprints": {leaf: result["dataset_fingerprints"].get(leaf) for leaf in leaves},
            "docs": docs,
        }
    meta = {"batch_size": result.get("batch_size"), "limit": task_limit}
    return out, meta


def _record_one(
    model_id: str,
    revision: str,
    path: Path | None,
    device: str,
    limit: int | None,
    batch_size: int,
    cache: Path | None,
) -> dict[str, Any]:
    import importlib.metadata

    import torch

    from tricast.eval import perplexity

    started = time.time()
    model, tokenizer, row, env = to.prepare(model_id, revision, path, device, cache)
    seconds = {"prepare": round(time.time() - started, 1)}
    row.update(
        schema=SCHEMA,
        limit=limit,
        recorder_sha256=to._sha256(Path(__file__).read_bytes()),
        lm_eval_version=importlib.metadata.version("lm_eval"),
        results={},
    )
    windows = None if limit is None else 8
    for spec in BENCHMARKS:
        if spec["kind"] != "ppl":
            continue
        tick = time.time()
        if spec["id"] == "c4":
            text = c4_text()
            ppl = perplexity(model, tokenizer, texts=[text], device=device, seqlen=PPL_SEQLEN,
                             max_windows=min(C4_WINDOWS, windows or C4_WINDOWS))
            source = {"split": "validation", "text_sha256": to._sha256(text.encode("utf-8"))}
        else:
            ppl = perplexity(model, tokenizer, dataset="wikitext2", split="test", device=device,
                             seqlen=PPL_SEQLEN, max_windows=windows)
            source = {"split": "test"}
        row["results"][spec["id"]] = {
            "value": ppl["ppl"],
            "seqlen": PPL_SEQLEN,
            **source,
            **{key: ppl.get(key) for key in ("n_tokens", "n_windows", "dataset_fingerprint")},
        }
        seconds[spec["id"]] = round(time.time() - tick, 1)
        print(
            f"BENCH_PART {row['name']} {spec['id']} ppl={ppl['ppl']:.4f} {seconds[spec['id']]}s", flush=True
        )
    calls: dict[str, list[dict[str, Any]]] = {}  # one lm-eval call per few-shot setting and document limit
    for spec in BENCHMARKS:
        if spec["kind"] == "acc":
            calls.setdefault(f"{spec['num_fewshot']}shot_limit{spec.get('limit')}", []).append(spec)
    row["lm_eval"] = {}
    for key, specs in calls.items():
        tick = time.time()
        results, meta = _lmeval(model, tokenizer, specs, limit, batch_size)
        row["results"].update(results)
        row["lm_eval"][key] = meta
        seconds[f"lm_eval_{key}"] = round(time.time() - tick, 1)
        for task, item in results.items():
            print(
                f"BENCH_PART {row['name']} {task} {item['metric']}={item['value']:.4f} n={item['n']}",
                flush=True,
            )
        print(f"BENCH_PART {row['name']} lm_eval_{key} {seconds[f'lm_eval_{key}']}s", flush=True)
    row["seconds"] = seconds
    row["env"] = env
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def record(
    model_id: str,
    revision: str,
    paths: list[Path],
    native: bool,
    out: Path,
    device: str,
    limit: int | None,
    batch_size: int,
    cache: Path | None,
) -> int:
    out.mkdir(parents=True, exist_ok=True)
    jobs: list[Path | None] = ([None] if native else []) + list(paths)
    ok = failed = 0
    for path in jobs:
        name = "native" if path is None else (path.stem if path.name != "recipe.yaml" else path.parent.name)
        target = out / f"{name}.json"
        if target.is_file():
            print(f"SKIP {name} (recorded)", flush=True)
            ok += 1
            continue
        started = time.time()
        try:
            row = _record_one(model_id, revision, path, device, limit, batch_size, cache)
            row["seconds"]["total"] = round(time.time() - started, 1)
            target.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"OK {name} {row['seconds']['total']}s", flush=True)
            ok += 1
        except Exception:  # one failing recipe does not stop the others; its reason is in the log
            traceback.print_exc()
            print(f"FAILED {name}", flush=True)
            failed += 1
    print(f"BENCH_DONE ok={ok} failed={failed}", flush=True)
    return 1 if failed else 0


# ---------- export ----------
def mean_stderr(docs: str) -> float:
    """lm-eval's ``mean_stderr`` of 0/1 results: sample standard deviation / √n."""
    n = len(docs)
    mean = docs.count("1") / n
    return math.sqrt(mean * (1 - mean) * n / (n - 1) / n) if n > 1 else 0.0


def paired(a: str, b: str) -> tuple[float, float]:
    """Mean of a − b over the same documents and its standard error (sample SD of the differences / √n)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired results need the same documents")
    d = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    n = len(d)
    mean = sum(d) / n
    var = (sum(v * v for v in d) - n * mean * mean) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(max(var, 0.0) / n)


def verdict(spec: dict[str, Any], diff: float, se: float | None) -> str | None:
    """better / same / worse for ours against the existing way: beyond 2 standard errors of the paired
    difference (accuracy); a perplexity has no per-document results, so it gets no verdict."""
    if spec["kind"] != "acc" or se is None:
        return None
    if diff > 2 * se:
        return "better"
    if diff < -2 * se:
        return "worse"
    return "same"


def _best(
    spec: dict[str, Any], rows: list[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """The best one-method row on ``spec`` (ties: fewer bits, then id)."""
    sign = 1 if spec["kind"] == "acc" else -1
    return max(
        rows,
        key=lambda row: (
            sign * records[row["recipe_file_sha256"]]["results"][spec["id"]]["value"],
            -row["bits"],
            row["id"],
        ),
    )


def _same_documents(row: dict[str, Any], native: dict[str, Any]) -> None:
    keys = ("n", "n_total", "limit", "num_fewshot", "versions", "fingerprints", "metric", "filter")
    for spec in BENCHMARKS:
        mine, base = row["results"][spec["id"]], native["results"][spec["id"]]
        keys_here = (keys if spec["kind"] == "acc"
                     else ("split", "n_tokens", "n_windows", "dataset_fingerprint", "text_sha256"))
        differ = [key for key in keys_here if mine.get(key) != base.get(key)]
        if differ:
            raise SystemExit(f"{row['name']}: {spec['id']} differs from the native record in {differ}")
    if row["lm_eval_version"] != native["lm_eval_version"] or row["lm_eval"] != native["lm_eval"]:
        raise SystemExit(f"{row['name']}: another lm-eval version or batch setting than the native record")


def export(bench: Path, bundle_path: Path, out: Path) -> None:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(bench.glob("*.json"))]
    native = next((row for row in rows if row["name"] == "native"), None)
    if native is None or native.get("limit") is not None:
        raise SystemExit(f"{bench}: no full native.json (record with --native, without --limit)")
    if native["model"] != bundle["table"]["meta"]["model_id"]:
        raise SystemExit(
            f"{bench}: records of {native['model']}, the bundle is {bundle['table']['meta']['model_id']}"
        )
    page = {
        row["recipe_file_sha256"]: (row, kind)
        for kind, key in (("uniform", "baselines"), ("auto", "solutions"))
        for row in bundle[key]
        if row.get("recipe_file_sha256")
    }
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["name"] == "native":
            continue
        if row.get("limit") is not None:
            raise SystemExit(f"{row['name']}: a --limit record (pipeline check only)")
        if row["file_sha256"] not in page:
            raise SystemExit(
                f"{row['name']}: recipe file {row['file_sha256'][:12]} is no setup of {bundle_path}"
            )
        loaded, base = row["env"].get("model_sha"), native["env"].get("model_sha")  # commits actually loaded
        if row["model"] != native["model"] or loaded is None or loaded != base:
            raise SystemExit(
                f"{row['name']}: a record of {row['model']}@{loaded}, not of {native['model']}@{base}"
            )
        _same_documents(row, native)
        records[row["file_sha256"]] = row
    keep = lambda row: {key: row["env"].get(key) for key in ENV_KEYS}  # noqa: E731
    values = lambda row: {spec["id"]: row["results"][spec["id"]]["value"] for spec in BENCHMARKS}  # noqa: E731
    stderr = lambda row: {spec["id"]: row["results"][spec["id"]].get("stderr") for spec in BENCHMARKS}  # noqa: E731
    setups = {
        sha: {
            "id": page[sha][0]["id"],
            "kind": page[sha][1],
            "bits": page[sha][0]["bits"],
            "values": values(row),
            "stderr": stderr(row),
            "patched_layers": row["patched_layers"],
            "calibration": row.get("calibration"),
            "seconds": row["seconds"].get("total"),
            "env": keep(row),
        }
        for sha, row in records.items()
    }
    accs = [spec for spec in BENCHMARKS if spec["kind"] == "acc"]
    compare = {}
    for solution in bundle["solutions"]:
        sha = solution.get("recipe_file_sha256")
        if sha not in records or solution.get("budget_bits") is None:
            continue
        limit = (
            _fraction(solution["budget_bits"]) + 1e-9
        )  # the page's uniformsWithin: the budget, not the bits
        within = [
            row
            for row in bundle["baselines"]
            if row.get("recipe_file_sha256") in records and row["bits"] <= limit
        ]
        if not within:
            continue
        ours = records[sha]
        table = []
        for spec in BENCHMARKS:
            best = _best(spec, within, records)
            theirs = records[best["recipe_file_sha256"]]
            a, b = ours["results"][spec["id"]], theirs["results"][spec["id"]]
            diff, se = (
                paired(a["docs"], b["docs"]) if spec["kind"] == "acc" else (a["value"] - b["value"], None)
            )
            table.append(
                {
                    "bench": spec["id"],
                    "ours": a["value"],
                    "theirs": b["value"],
                    "best": best["id"],
                    "diff": diff,
                    "se": se,
                    "verdict": verdict(spec, diff, se),
                }
            )
        # the accuracy average: against the one-method setup with the best average (one setup, not a
        # per-task mix)
        mean = lambda sha_: sum(records[sha_]["results"][s["id"]]["value"] for s in accs) / len(accs)  # noqa: E731
        best = max(within, key=lambda row: (mean(row["recipe_file_sha256"]), -row["bits"], row["id"]))
        pairs = [
            paired(
                ours["results"][s["id"]]["docs"],
                records[best["recipe_file_sha256"]]["results"][s["id"]]["docs"],
            )
            for s in accs
        ]
        diff = sum(d for d, _ in pairs) / len(pairs)
        se = math.sqrt(sum(s * s for _, s in pairs)) / len(pairs)  # tasks are separate documents
        macro = {
            "ours": mean(sha),
            "theirs": mean(best["recipe_file_sha256"]),
            "best": best["id"],
            "diff": diff,
            "se": se,
            "verdict": verdict({"kind": "acc"}, diff, se),
            "tasks": len(accs),
        }
        compare[sha] = {
            "id": solution["id"],
            "budget_bits": solution["budget_bits"],
            "uniforms": [row["id"] for row in within],
            "rows": table,
            "macro": macro,
        }
    lm_eval = native["lm_eval"]
    benchmarks = []
    for spec in BENCHMARKS:
        item = native["results"][spec["id"]]
        benchmarks.append(
            {key: spec.get(key) for key in ("id", "label", "group", "kind", "metric", "num_fewshot", "what")}
            | {"n": item.get("n") if spec["kind"] == "acc" else item.get("n_tokens"),
               "n_total": item.get("n_total"), "limit": item.get("limit")}
        )
    payload = {
        "schema": SCHEMA,
        "model": native["model"],
        "revision": native["revision"],
        "lm_eval_version": native["lm_eval_version"],
        "batch_size": {key: meta["batch_size"] for key, meta in lm_eval.items()},
        "benchmarks": benchmarks,
        "native": {"values": values(native), "stderr": stderr(native), "env": keep(native)},
        "setups": setups,
        "compare": compare,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"EXPORTED native + {len(setups)} setups, {len(compare)} sizes compared -> {out}")


def _fraction(text: str) -> float:
    from fractions import Fraction

    return float(Fraction(str(text)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("recipes")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("record")
    r.add_argument("--model", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--native", action="store_true")
    r.add_argument("--device", default="cuda")
    r.add_argument(
        "--limit", type=int, help="documents per lm-eval task and 8 PPL windows (timing check only)"
    )
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--calib-cache", type=Path, help="per-layer calibration cache directory (node-local disk)")
    r.add_argument("recipes", nargs="*", type=Path)
    e = sub.add_parser("export")
    e.add_argument("--bench", type=Path, required=True)
    e.add_argument("--bundle", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "recipes":
        recipes(args.bundle, args.out)
        return 0
    if args.command == "record":
        return record(
            args.model,
            args.revision,
            args.recipes,
            args.native,
            args.out,
            args.device,
            args.limit,
            args.batch_size,
            args.calib_cache,
        )
    export(args.bench, args.bundle, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
