"""Auto-tune v1 spike C2 analysis (docs/spikes/2026-10-13_autotune_mckp.md): predicted vs measured KL.

usage: PYTHONPATH=src:. python scripts/quantizer/autotune_c2.py <predict.json> <measure.json> <out.json>
         [--overlap MODEL_ID --revision SHA --seqlen 512 --windows 8 --seeds 42,43]

- Joins `python -m app.autotune predict` and `measure` outputs by recipe name (each name once per side).
- Distinctness: every measured recipe is resolved with tricast (Recipe.spec_for for all 196 Linear names
  + KV preset); recipes with identical resolved layers count once (the first by name order).
- C2: Spearman rho (average ranks for ties) of predicted kl_sum vs measured kl over the distinct recipes,
  n >= 10.
- Declared before the results (spike "4절 보충 2"), not part of C2's pass/fail:
  band rho  — rho over the distinct recipes with 4.25 <= avg bits <= 1831/320;
  pairs     — predicted order == measured order for (qd_nvfp4, auto_4.50), (qd_nvint4, auto_4.50),
              (qd_mxfp4_ends_bf16, auto_5.09).
- --overlap: token offsets of the WikiText-2 train windows for each seed (tricast.calibration._windows on
  the joined text) and whether any window of one seed overlaps a window of another.
Last line: AUTOTUNE_C2_DONE ...
"""
import argparse
import hashlib
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import app.quantizer_recipe as qr

BAND = (Fraction(17, 4), Fraction(1831, 320))
PAIRS = (("qd_nvfp4", "auto_4.50"), ("qd_nvint4", "auto_4.50"), ("qd_mxfp4_ends_bf16", "auto_5.09"))
THRESHOLD, MIN_N = 0.8, 10


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy)


def signature(recipe_path: str) -> str:
    from tricast.recipe import load_recipe

    recipe = load_recipe(Path(recipe_path))
    layers = []
    for block in range(qr.BLOCKS):
        for kind in qr.KINDS:
            spec = recipe.spec_for(qr.layer_name(block, kind))
            layers.append(None if spec is None else repr(spec))
    kv = None if recipe.kv is None else repr(recipe.kv)
    return hashlib.sha256(json.dumps([layers, kv]).encode()).hexdigest()


def overlap(model: str, revision: str, seqlen: int, windows: int, seeds: list[int]) -> dict:
    from transformers import AutoTokenizer

    from tricast.calibration import _dataset_texts

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    text = "\n\n".join(_dataset_texts("wikitext2", "train"))
    n_tokens = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
    starts = {}
    for seed in seeds:  # mirrors tricast.calibration._windows for a single 1 x N id row
        rng = random.Random(seed)
        starts[seed] = []
        for _ in range(windows):
            rng.randrange(1)
            starts[seed].append(rng.randrange(n_tokens - seqlen + 1))
    clashes = [(a, sa, b, sb) for i, a in enumerate(seeds) for b in seeds[i + 1:]
               for sa in starts[a] for sb in starts[b] if abs(sa - sb) < seqlen]
    return {"n_tokens": n_tokens, "starts": {str(k): v for k, v in starts.items()},
            "overlaps": [{"seed_a": a, "start_a": sa, "seed_b": b, "start_b": sb}
                         for a, sa, b, sb in clashes]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predict")
    parser.add_argument("measure")
    parser.add_argument("out")
    parser.add_argument("--overlap", metavar="MODEL_ID")
    parser.add_argument("--revision")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--seeds", default="42,43")
    args = parser.parse_args(argv)
    predicted = json.loads(Path(args.predict).read_text(encoding="utf-8"))["results"]
    measured = json.loads(Path(args.measure).read_text(encoding="utf-8"))["results"]
    for side, rows in (("predict", predicted), ("measure", measured)):
        names = [row["name"] for row in rows]
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise SystemExit(f"{side}: duplicate names {dup}")
    by_pred = {row["name"]: row for row in predicted}
    missing = sorted(row["name"] for row in measured if row["name"] not in by_pred)
    if missing:
        raise SystemExit(f"measured but not predicted: {missing}")
    seen: dict[str, str] = {}
    rows, duplicates = [], []
    for row in sorted(measured, key=lambda r: r["name"]):
        sig = signature(row["recipe"])
        if sig in seen:
            duplicates.append({"name": row["name"], "same_layers_as": seen[sig]})
            continue
        seen[sig] = row["name"]
        pred = by_pred[row["name"]]
        rows.append({"name": row["name"], "predicted": pred["kl_sum"], "measured": row["kl"],
                     "avg_bits": pred["avg_bits"], "kv": pred["kv"], "layers_sha256": sig})
    rho = spearman([r["predicted"] for r in rows], [r["measured"] for r in rows])
    band = [r for r in rows if BAND[0] <= Fraction(r["avg_bits"]).limit_denominator(10**6) <= BAND[1]]
    band_rho = (spearman([r["predicted"] for r in band], [r["measured"] for r in band])
                if len(band) >= 3 else None)
    by_name = {r["name"]: r for r in rows}
    pairs = []
    for a, b in PAIRS:
        if a in by_name and b in by_name:
            pa, pb, ma, mb = (by_name[a]["predicted"], by_name[b]["predicted"],
                              by_name[a]["measured"], by_name[b]["measured"])
            pairs.append({"pair": [a, b], "predicted_lower": a if pa < pb else b,
                          "measured_lower": a if ma < mb else b, "agree": (pa < pb) == (ma < mb)})
    result = {"c2": {"rho": rho, "n": len(rows), "threshold": THRESHOLD, "min_n": MIN_N,
                     "pass": rho >= THRESHOLD and len(rows) >= MIN_N},
              "band": {"bits": [str(BAND[0]), str(BAND[1])], "n": len(band), "rho": band_rho},
              "pairs": pairs, "duplicates": duplicates, "rows": rows}
    if args.overlap:
        result["windows"] = overlap(args.overlap, args.revision, args.seqlen, args.windows,
                                    [int(s) for s in args.seeds.split(",")])
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(f"C2 {r['name']:>20} predicted={r['predicted']:.6g} measured={r['measured']:.6g} "
              f"bits={r['avg_bits']:.4f}")
    agree = sum(p["agree"] for p in pairs)
    band_text = "n/a" if band_rho is None else f"{band_rho:.4f}"
    overlaps = len(result["windows"]["overlaps"]) if args.overlap else "n/a"
    print(f"AUTOTUNE_C2_DONE rho={rho:.4f} n={len(rows)} pass={result['c2']['pass']} band_n={len(band)} "
          f"band_rho={band_text} pairs_agree={agree}/{len(pairs)} duplicates={len(duplicates)} "
          f"overlaps={overlaps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
