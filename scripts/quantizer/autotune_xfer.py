"""Option transfer: knapsack solutions with a tune session's method options applied and nothing else changed
(same cells, same bits), so the options an agent found at one budget can be measured at the others.

    PYTHONPATH=src:. python3 scripts/quantizer/autotune_xfer.py --solutions <run> --ids auto_2.50 auto_3.00 \\
        --options-from <session>/proposals/<best> --out <dir>

Writes <out>/xfer_<id>/{grid.json, recipe.yaml}. The recipe is built and written the way the tune loop
writes a proposal (``app.tune_agent._recipe``, ``qr.to_yaml`` + newline), and grid.json carries ``base``
(the solution's name, its id in the bundle) and ``options_from`` for ``app.export_autotune_demo --transfer``.
Check: a solution whose grid is the options-from grid must give that proposal's recipe byte for byte, the name
line aside (exit 1 otherwise). Prints XFER_DONE."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app import quantizer_recipe as qr
from app import tune_agent as ta


def recipe_text(name: str, spec: dict) -> str:
    """The recipe file the tune loop writes for a proposal (``tune_agent.propose``)."""
    return qr.to_yaml(ta._recipe(name, spec)) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--solutions", type=Path, required=True, help="run directory with <id>/grid.json")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--options-from", type=Path, required=True, help="a session proposal directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source = ta.load_grid(args.options_from / "grid.json")
    if not source["options"]:
        print(f"error: {args.options_from} has no options", file=sys.stderr)
        return 2
    origin = args.options_from.name
    checked = None
    for sid in args.ids:
        path = args.solutions / sid / "grid.json"
        spec, base = ta.load_grid(path), json.loads(path.read_text(encoding="utf-8")).get("name", sid)
        if spec["options"]:
            print(f"error: {sid} already has options", file=sys.stderr)
            return 2
        spec["options"] = source["options"]
        name = f"xfer_{sid}"
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        text = recipe_text(name, spec)
        record = {"name": name, "base": base, "options_from": origin, **spec}
        (folder / "grid.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        (folder / "recipe.yaml").write_text(text, encoding="utf-8")
        if spec["grid"] == source["grid"]:
            own = (args.options_from / "recipe.yaml").read_text(encoding="utf-8")
            checked = text.replace(name, origin) == own
            print(f"XFER_CHECK {sid} + options == {origin} (name aside): {checked}")
    print(f"XFER_DONE out={args.out} ids={len(args.ids)} options_from={origin}")
    return 0 if checked is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
