"""deck_data.json for build_deck.js: every number on the midterm-2 intro slides, computed from the
Auto-tune bundle (app/web/demo/autotune/bundle.json, hash-checked by app/export_autotune_demo.py) and
the calibration-seed PPL records under runs/20261105_replan/ (git-ignored; paths kept in the deck
notes). Text that is not a measurement (targets, plans) is taken as written in docs/DEMO.md §3.
    python3 docs/slides/midterm2/make_data.py"""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = "app/web/demo/autotune/bundle.json"
SEEDS = ("runs/20261105_replan/seeds_turin", "runs/20261105_replan/seeds46_base",
         "runs/20261105_replan/seeds_x250", "runs/20261105_replan/seeds_x300")
XFER = "runs/20261105_replan/xfer"
HISTORY = "evals/results/history.csv"
TARGETS = {  # docs/DEMO.md §3 목표값 (2026-11-08 사용자 승인)
    "heur": "같은 비트 균일 최선 대비 BF16 대비 PPL 열화 40% 이상 감소 (모델 2 × 3 · 4비트대)",
    "ai": "휴리스틱보다 test PPL 이 낮고, 차이 > 휴리스틱 seed 간 범위, AI 측정 8건 이하",
    "xfer": "AI 가 찾은 옵션을 다른 예산의 휴리스틱 해에 그대로 적용해도 test PPL 이 낮음",
    "s2": "한 모델에서 30개 이상 배정을 같은 조건으로 측정 (인터뷰 07: 30개 이상 중 6개)",
    "s3": "배정 확정 1주 이내 (인터뷰 3개월), 그중 도구 시간 1일 이내",
    "s4": "시연의 모든 배정을 레시피 파일만으로: 0회",
    "s5": "화면 측정값 100% 에 GPU · 트리 SHA · 레시피 해시",
    "evals": "지어낸 값 · 인젝션 · 환각 인용 · 예산 위반 0건",
    "basis": "judge 와 사람 라벨 일치율 ≥ 80%, 기준선을 같은 커밋 3회 변동 안에서 재현",
    "s1": "동료 80% 이상이 README 만으로 예산 → 배정 → 레시피 → AI 근거 (대상 사용자의 대리)",
    "model": "Llama-3.2-1B 에서 같은 명령 · 같은 화면",
}


def seed_ppl() -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for folder in SEEDS:
        for path in sorted((ROOT / folder).glob("seed*/*.json")):
            if path.name == "env.json":
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            seed = int(re.fullmatch(r"seed(\d+)", path.parent.name).group(1))
            assert record["calibration_config"]["seed"] == seed, path
            out.setdefault(path.stem, {})[seed] = record["metrics"]["ppl"]["ppl"]
    return out


def sign_flip_p(diffs: list[float]) -> tuple[int, int]:
    """Exact one-sided sign-flip test of mean(diffs) < 0: (count, 2^n)."""
    observed = sum(diffs)
    count = sum(sum(s * abs(d) for s, d in zip(signs, diffs, strict=True)) <= observed + 1e-12
                for signs in itertools.product((1, -1), repeat=len(diffs)))
    return count, 2 ** len(diffs)


def transfer(bundle: dict) -> list[dict]:
    """The knapsack solutions with the AI best's options (make_xfer.py), next to the same ones without."""
    heur = {s["id"]: s for s in bundle["solutions"]}
    rows = []
    for part in sorted((ROOT / XFER).glob("out_*")):
        kl_path = part / "kl_seed43.json"
        if not kl_path.is_file():
            continue
        kl = {r["name"]: r["kl"] for r in json.loads(kl_path.read_text(encoding="utf-8"))["results"]}
        for path in sorted((part / "ppl").glob("xfer_*.json")):
            base = heur[path.stem.removeprefix("xfer_")]
            ppl = json.loads(path.read_text(encoding="utf-8"))["metrics"]["ppl"]["ppl"]
            rows.append({"id": base["id"], "bits": base["bits"], "kl": kl[path.stem], "ppl": ppl,
                         "heur_kl": base["kl"], "heur_ppl": base["ppl"]})
    return sorted(rows, key=lambda r: r["bits"])


def evals_rows() -> tuple[str, str]:
    """The evals rows from the baseline runs in history.csv (change "baseline k/3 ...")."""
    import csv

    rows = [row for row in csv.DictReader((ROOT / HISTORY).open(encoding="utf-8"))
            if row["change"].startswith("baseline ") and row["mode"] == "live"]
    if not rows:
        return "기준선 미측정", "기준선 미측정 · judge 라벨 미작성"
    rule = max(int(row[name] or 0) for row in rows for name in ("A2", "A6", "B2", "B3", "B5"))
    spread = {name: max(float(r[name]) for r in rows) - min(float(r[name]) for r in rows)
              for name in ("A1", "A3", "A4", "A5", "A7", "B1") if all(r[name] for r in rows)}
    code = max(value for name, value in spread.items() if name != "B1")
    labels = ROOT / "evals" / "judge_labels.jsonl"
    return (f"기준선 {len(rows)}회 ({rows[0]['run_id'][:10]}): 최대 {rule}건",
            f"같은 커밋 {len(rows)}회 변동: 코드 지표 {code:.2f}, B1 ±{spread['B1']:.2f}; judge 라벨 "
            + ("작성됨" if labels.is_file() else "미작성 (사람) → judge 지표 판정 보류"))


def main() -> None:
    bundle = json.loads((ROOT / BUNDLE).read_text(encoding="utf-8"))
    session, native = bundle["session"], bundle["native"]["ppl"]
    start = next(s for s in bundle["solutions"] if s["id"] == session["start"])
    best = session["best"]
    seeds = seed_ppl()
    heur, ai = seeds[session["start"]], seeds[best["name"]]
    common = sorted(set(heur) & set(ai))
    diffs = [ai[s] - heur[s] for s in common]
    mean_h = sum(heur[s] for s in common) / len(common)
    mean_a = sum(ai[s] for s in common) / len(common)
    count, total = sign_flip_p(diffs)
    spread = max(heur[s] for s in common) - min(heur[s] for s in common)
    cost = sum((r.get("llm") or {}).get("cost_usd") or 0 for r in session["rounds"])
    pct, wins, gap = 100 * (mean_a / mean_h - 1), sum(d < 0 for d in diffs), abs(mean_a - mean_h)

    def drop(solution_id: str) -> str:
        """BF16 대비 test PPL 열화: 같은 비트 이하 균일 최선 → 휴리스틱 해."""
        sol = next(s for s in bundle["solutions"] if s["id"] == solution_id)
        uni = min((b for b in bundle["baselines"] if b["bits"] <= sol["bits"] + 1e-3 and b.get("ppl")),
                  key=lambda b: b["ppl"])
        a, b = uni["ppl"] - native, sol["ppl"] - native
        return f"{sol['bits']:.2f}비트 {a:.2f} → {b:.2f} ({100 * (b / a - 1):.0f}%)"

    rows = bundle["solutions"] + bundle["baselines"] + bundle.get("transfer", [])
    kl_n = sum(1 for r in rows if r.get("kl_record")) + sum(
        1 for r in session["rounds"] for p in r["proposals"] if p.get("kl_id"))
    ppl_n = sum(1 for r in rows if r.get("ppl_record")) + sum(
        1 for r in session["rounds"] for p in r["proposals"] if p.get("ppl_id"))
    measured = sum(1 for r in session["rounds"] for p in r["proposals"] if p.get("kl_id"))
    rounds = session["rounds"]
    found = next(r["round"] for r in rounds if any(p["name"] == best["name"] for p in r["proposals"]))
    to_best = sum(1 for r in rounds if r["round"] <= found for p in r["proposals"] if p.get("kl_id"))
    xfer = transfer(bundle)
    goals = [
        {"kind": "정량", "metric": "같은 크기 품질 (휴리스틱)", "target": TARGETS["heur"],
         "now": f"{drop('iso_3.15')}, {drop('iso_4.16')} — 모델 1개"},
        {"kind": "정량", "metric": "AI 의 추가 이득", "key": True, "target": TARGETS["ai"],
         "now": f"seed {common[0]}–{common[-1]}: {mean_h:.2f} → {mean_a:.2f} ({pct:.2f}%, "
                f"{wins}/{len(diffs)}), 차이 {gap:.2f} > 범위 {spread:.2f}, p = {count}/{total}, "
                f"최선까지 측정 {to_best}건 (총 {measured}건)"},
    ]
    robust = []
    for r in xfer:  # seed 42 is the bundle's record; other seeds from the seed runs
        h = {42: r["heur_ppl"], **seeds.get(r["id"], {})}
        x = {42: r["ppl"], **seeds.get(f"xfer_{r['id']}", {})}
        both = sorted(set(h) & set(x))
        if len(both) > 1:
            n_low = sum(x[k] < h[k] for k in both)
            mean = 100 * (sum(x[k] for k in both) / sum(h[k] for k in both) - 1)
            gap = abs(sum(x[k] - h[k] for k in both) / len(both))
            spread_h = max(h[k] for k in both) - min(h[k] for k in both)
            p_flip = "/".join(map(str, sign_flip_p([x[k] - h[k] for k in both])))
            r["seeds"] = {"n": len(both), "lower": n_low, "mean_pct": mean, "gap": gap,
                          "heur_range": spread_h, "p": p_flip}
            r["seeds"].update(heur_mean=sum(h[k] for k in both) / len(both),
                              mean=sum(x[k] for k in both) / len(both))
            robust.append(f"{r['bits']:.2f}b {mean:+.1f}%")
    if xfer:
        once = [r for r in xfer if "seeds" not in r]
        each = ", ".join(f"{r['bits']:.2f}b {100 * (r['ppl'] / r['heur_ppl'] - 1):+.1f}%" for r in once)
        tested = [r["seeds"] for r in xfer if "seeds" in r]
        head = (f"{' · '.join(robust)} (seed {tested[0]['lower']}/{tested[0]['n']} 씩, p = {tested[0]['p']})"
                if tested and all(t["lower"] == t["n"] for t in tested) else ", ".join(robust))
        goals.append({"kind": "정량", "metric": "AI 발견의 전이", "target": TARGETS["xfer"],
                      "now": "; ".join(filter(None, [head, f"seed 42 한 번: {each}" if once else ""]))})
    rules, basis = evals_rows()
    rest = (
        ("정량", "S2 조합 커버리지 (대리)", "s2", f"{kl_n} 배정 (균일 · 배낭 · AI 세션 · 옵션 전이)"),
        ("정량", "S3 배정 확정 시간", "s3",
         "도구 시간: 표 5.7시간 + 배낭 초 + AI 라운드 약 11분; 사람 시간 미측정"),
        ("정량", "S4 코드 재작성", "s4", "0회 (레시피 파일만 바꿈)"),
        ("정량", "S5 근거 동반률", "s5", f"100% (KL {kl_n} · PPL {ppl_n} 대조 통과)"),
        ("정량", "evals 절대 규칙", "evals", rules),
        ("정량", "evals 근거화", "basis", basis),
        ("정성", "S1 동료 사용", "s1", "11/11–11/18 예정"),
        ("정성", "모델 확장", "model", "예정"),
    )
    goals += [{"kind": k, "metric": m, "target": TARGETS[t], "now": n} for k, m, t, n in rest]
    data = {
        "model": bundle["table"]["meta"]["model_id"].split("/")[-1],
        "session": {"bits_label": f"{start['bits']:.2f}", "start_kl": f"{start['kl']:.4f}",
                    "best_kl": f"{best['kl']:.4f}",
                    "kl_change": f"{100 * (best['kl'] / start['kl'] - 1):.1f}%", "n_seeds": len(common),
                    "ppl_pct": f"{pct:.2f}%", "wins": f"{wins}/{len(diffs)}", "p": f"{count}/{total}",
                    "llm_cost": f"{cost:.2f}"},
        "goals": goals, "transfer": xfer,
        "transfer_line": ", ".join(
            f"{r['bits']:.2f}비트 PPL {r['seeds']['heur_mean']:.1f} → {r['seeds']['mean']:.1f}"
            for r in xfer if "seeds" in r) + " (보정 seed 5개 평균, 5개 모두 낮음)",
        "sources": {"bundle": BUNDLE, "seeds": " · ".join(SEEDS), "transfer": XFER},
    }
    out = Path(__file__).with_name("deck_data.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"DECK_DATA_DONE out={out.relative_to(ROOT)} goals={len(goals)} transfer={len(xfer)}")


if __name__ == "__main__":
    main()
