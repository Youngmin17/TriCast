"""Judge validation for the tune evals (evals/EVAL_DESIGN.md §3): a labeling sheet that shows people exactly
what the judge saw, and the agreement of the judge with their labels.

    python evals/judge_agreement.py sheet evals/results/<run_id>.json      # -> <run_id>_labeling_full.md
    python evals/judge_agreement.py agree evals/results/<run_id>.json [evals/judge_labels.jsonl]

``sheet`` writes, for every judged unit of the run (a request case, or one accepted proposal of an agent
case), the judge's whole input — including the retrieved documents the runner's own sheet leaves out — and
the criteria to label, but no judge verdict or reason. People label it before they open the result file.

``agree`` reads ``judge_labels.jsonl`` lines ``{"run_id", "id", "unit", "labeler", "<criterion>":
true|false, ...}`` (``unit`` is ``request`` for request cases, the proposal name for agent cases) and
compares each labeled criterion with the judge's verdict on that unit: agreement per criterion and overall,
and the direction of each disagreement (judge passes what people fail = lenient, the reverse = strict).
Labels of another run, of a unit the run did not judge, or of a criterion the judge left null are counted
as unmatched, never as agreement.
Prints JUDGE_AGREEMENT <overall> (or NO_LABELS) and writes <run_id>_agreement.json. Stdlib only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CRITERIA = ("q1", "q2", "c1", "c2", "c3", "c4")
TARGET = 0.8  # §3: below this the judge prompt's conditions are revised and measured again


def units(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every judged unit of a result file: case id, unit, the criteria it requires, the input and verdict."""
    out = []
    for case in result["cases"]:
        for entry in case.get("judge") or []:
            out.append({"id": case["id"], "category": case["category"], "unit": entry["unit"],
                        "required": entry["required"], "input": entry["input"],
                        "verdict": entry.get("verdict")})
    return out


def sheet(result: dict[str, Any]) -> str:
    run = result["run"]["run_id"]
    lines = [f"# 라벨링 시트 (judge 입력 전체) — 실행 {run}", "",
             "judge 판정은 이 시트에 없다. 다 매기기 전에 결과 파일(`evals/results/*.json`)을 열지 않는다.",
             "기준의 뜻은 `evals/tune_judge_prompt.md`, 라벨은 `evals/judge_labels.jsonl` 에 한 줄씩:",
             f'`{{"run_id": "{run}", "id": "<케이스>", "unit": "<아래 unit>", '
             '"labeler": "<이름 · 이름 합의>", "<기준>": true|false, ...}}`', ""]
    for item in units(result):
        lines += [f"## {item['id']} · unit `{item['unit']}` ({item['category']})", "",
                  "```json", json.dumps(item["input"], ensure_ascii=False, indent=1), "```", "",
                  "| 기준 | 통과? (true/false) | 메모 |", "|---|---|---|"]
        lines += [f"| {name} | | |" for name in item["required"]] + [""]
    return "\n".join(lines)


def agree(result: dict[str, Any], labels: list[dict[str, Any]]) -> dict[str, Any]:
    run = result["run"]["run_id"]
    judged = {(item["id"], item["unit"]): item for item in units(result)}
    per = {name: {"n": 0, "agree": 0, "lenient": 0, "strict": 0} for name in CRITERIA}
    unmatched = []
    for label in labels:
        item = judged.get((label.get("id"), label.get("unit")))
        if label.get("run_id") != run or item is None:
            unmatched.append({"id": label.get("id"), "unit": label.get("unit"),
                              "why": "no such judged unit in this run"})
            continue
        for name in CRITERIA:
            if name not in label:
                continue
            verdict = ((item["verdict"] or {}).get(name) or {}).get("pass")
            if not isinstance(label[name], bool) or not isinstance(verdict, bool):
                unmatched.append({"id": item["id"], "unit": item["unit"], "criterion": name,
                                  "why": "label or judge verdict is not true/false"})
                continue
            row = per[name]
            row["n"] += 1
            row["agree"] += verdict == label[name]
            row["lenient"] += verdict and not label[name]
            row["strict"] += label[name] and not verdict
    total = sum(row["n"] for row in per.values())
    agreed = sum(row["agree"] for row in per.values())
    for row in per.values():
        row["rate"] = row["agree"] / row["n"] if row["n"] else None
    overall = agreed / total if total else None
    return {"run_id": run, "labels": len(labels), "judgments": total, "overall": overall, "target": TARGET,
            "met": None if overall is None else overall >= TARGET,
            "criteria": {name: row for name, row in per.items() if row["n"]}, "unmatched": unmatched}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("sheet", "agree"):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    path = Path(argv[1])
    result = json.loads(path.read_text(encoding="utf-8"))
    if argv[0] == "sheet":
        out = path.with_name(f"{result['run']['run_id']}_labeling_full.md")
        out.write_text(sheet(result) + "\n", encoding="utf-8")
        print(f"LABELING_SHEET {out} units={len(units(result))}")
        return 0
    labels_path = Path(argv[2]) if len(argv) > 2 else path.parent.parent / "judge_labels.jsonl"
    if not labels_path.is_file():
        print(f"NO_LABELS {labels_path}: people label the sheet first (EVAL_DESIGN.md §3)")
        return 1
    text = labels_path.read_text(encoding="utf-8")
    labels = [json.loads(line) for line in text.splitlines() if line.strip()]
    report = agree(result, labels)
    out = path.with_name(f"{result['run']['run_id']}_agreement.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shown = "n/a" if report["overall"] is None else f"{report['overall']:.3f}"
    print(f"JUDGE_AGREEMENT {shown} judgments={report['judgments']} target>={TARGET} met={report['met']} "
          f"unmatched={len(report['unmatched'])} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
