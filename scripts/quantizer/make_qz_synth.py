"""Quantizer page data with SYNTHETIC recorded runs (made-up PPL, SQNR and lm-eval numbers from
app/tests/quantizer_records.py) so qz_check.mjs can drive the recorded-result panels. Never copy the output
into app/web.

usage: PYTHONPATH=<repo> python scripts/quantizer/make_qz_synth.py <runs_out_dir> <bundle_out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from app import export_quantizer_demo as ex
from app.tests import quantizer_records as rec


def main() -> int:
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    rec.synthetic_runs(runs)
    return ex.main(
        ["--runs", str(runs), "--native", str(rec.NATIVE), "--lmeval", str(runs / "lm"), "--out", str(out)]
    )


if __name__ == "__main__":
    raise SystemExit(main())
