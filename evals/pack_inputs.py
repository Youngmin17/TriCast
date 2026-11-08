"""Pack what the tune evals runner reads into the repository, so a clone can re-grade the kept replies
(``--mode stored``) without ``runs/`` (git-ignored; the full ledger is 73 MB):

- ``evals/fixtures/tune_inputs/table.json``: the Auto-tune table the eval set's sessions were run on;
- ``evals/fixtures/tune_inputs/s3150/``: the demo tune session (its rounds, proposals and analyses), its
  ``loop.json`` pointed at these packed files and without the cluster settings;
- ``evals/fixtures/tune_inputs/ledger.jsonl``: the ledger records any of them can reach — every record id
  written in the session or in the kept replies (``evals/fixtures/tune/``), and the measured-recipe records
  (``recipe_hash``) of the session's proposals — in the ledger's order.

    python evals/pack_inputs.py --table runs/.../table.json --ledger runs/ledger.jsonl \\
        --session runs/.../s3150

Prints PACK_DONE with the record counts. Re-run it after a live run adds kept replies.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "fixtures" / "tune_inputs"
RECORD = re.compile(r"E[0-9a-f]{12}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.table, OUT / "table.json")
    target = OUT / args.session.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(args.session, target)
    # the session's loop.json points inside the repository (so `app.server --agent` works from a clone) and
    # carries no cluster settings; the cluster round script is not packed
    loop = json.loads((target / "loop.json").read_text(encoding="utf-8"))
    loop.pop("remote", None)
    base = OUT.relative_to(ROOT)
    loop.update(session=str(base / target.name), table=str(base / "table.json"),
                ledger=str(base / "ledger.jsonl"))
    (target / "loop.json").write_text(json.dumps(loop, indent=1) + "\n", encoding="utf-8")
    (target / "run_round.sh").unlink(missing_ok=True)
    texts = [path.read_text(encoding="utf-8") for path in target.rglob("*.json")]
    kept_replies = ROOT / "evals" / "fixtures" / "tune"
    texts += [path.read_text(encoding="utf-8") for path in kept_replies.rglob("*.json")]
    ids = {match for text in texts for match in RECORD.findall(text)}
    hashes = {json.loads(path.read_text(encoding="utf-8"))["recipe_sha256"]
              for path in (target / "proposals").glob("*/proposal.json")}
    kept, total = [], 0
    with args.ledger.open(encoding="utf-8") as handle:
        for line in handle:
            total += 1
            record = json.loads(line)
            if record["id"] in ids or record.get("recipe_hash") in hashes:
                kept.append(line)
    (OUT / "ledger.jsonl").write_text("".join(kept), encoding="utf-8")
    print(f"PACK_DONE out={OUT.relative_to(ROOT)} records={len(kept)}/{total} ids_written={len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
