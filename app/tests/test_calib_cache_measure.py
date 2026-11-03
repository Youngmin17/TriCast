"""autotune measure --calib-cache and the tune round script: a second measurement of the same calibrated
recipe loads every layer from tricast calibrate's result cache, runs no calibration forward, and measures the
same KL bit for bit as a run without the cache. Tiny Llama on CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app import autotune as at
from app import quantizer_recipe as qr
from app import tune_agent as ta
from app.tests import test_autotune_v2 as v2
from app.tests.test_autotune_v2 import LLAMA_ARGS, SETTINGS, load, run, spies, tiny_llama

calibrated = v2.calibrated  # the module-scoped fixture: a tiny Llama with its calibrated table


def test_a_cached_calibration_measures_the_same_kl_with_no_forward(calibrated: tuple[Path, dict, Any],
                                                                   tmp_path: Path) -> None:
    root, _, _ = calibrated
    grid = qr.uniform("bf16")
    for block in range(2):
        for kind in ("o_proj", "down_proj"):
            grid[block][qr.KINDS.index(kind)] = "gptq_int4"
    at.write_grid(tmp_path / "gptq", "gptq_od", "gptq_int4", grid, "none", SETTINGS)
    recipe = str(tmp_path / "gptq" / "recipe.yaml")
    common = ["measure", "--model", str(root / "model"), "--text", str(root / "text.txt"), *LLAMA_ARGS,
              "--seed", "42", "--calibrate"]
    results = {}
    for name, extra in (("plain", []), ("fill", ["--calib-cache", str(tmp_path / "cache")]),
                        ("hit", ["--calib-cache", str(tmp_path / "cache")])):
        with spies():
            rc, _, err = run([*common, *extra, "--out", str(tmp_path / f"{name}.json"), recipe],
                             model_factory=tiny_llama)
        assert rc == 0, err
        results[name] = load(tmp_path / f"{name}.json")["results"][0]
    plain, fill, hit = results["plain"], results["fill"], results["hit"]
    assert "cache" not in plain["calibration"]
    assert plain["kl"] == fill["kl"] == hit["kl"]  # bit for bit (JSON floats round-trip exactly)
    assert fill["calibration"]["cache"]["hits"] == 0 and len(fill["calibration"]["cache"]["misses"]) == 4
    assert hit["calibration"]["cache"]["hits"] == 4 and hit["calibration"]["cache"]["misses"] == []
    assert hit["calibration"]["full_model_forwards"] == 0 < fill["calibration"]["full_model_forwards"]
    assert plain["calibration"]["dataset_fingerprint"] == hit["calibration"]["dataset_fingerprint"]


def test_the_round_script_passes_the_cache_only_when_set(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from app.tests.test_tune_loop import _session

    _session(tmp_path)
    script = ta.round_script(tmp_path / "s", ["start"], 1).read_text()
    assert '${CALIB_CACHE:+--calib-cache "$CALIB_CACHE"}' in script
