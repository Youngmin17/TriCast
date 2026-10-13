"""Re-derive the Quantizer test fixtures and check the exporter's PINS: for every preset, tricast's normalised
recipe (what a runner records as ``recipe``) goes to app/tests/data/quantizer/recipes/<id>.recipe.json, and
the PINS line (SHA-256 of the preset YAML, tricast load_recipe(...).sha256) is printed and asserted equal to
app/export_quantizer_demo.PINS. Run it when a preset changes; it ends with QZ_FIXTURES_OK.

usage: cd <repo> && PYTHONPATH=src:. python scripts/quantizer/make_qz_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr

from tricast.recipe import load_recipe

OUT = Path(qr.__file__).resolve().parent / "tests" / "data" / "quantizer" / "recipes"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for preset_id in qr.PRESETS:
        text = qr.to_yaml(qr.preset(preset_id))
        recipe = load_recipe(qr.preset(preset_id))
        from_yaml = load_recipe(yaml.safe_load(text))
        yaml_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f'    "{preset_id}": ("{yaml_digest}",\n{" " * (len(preset_id) + 9)}"{recipe.sha256}"),')
        assert recipe.sha256 == from_yaml.sha256, preset_id
        assert ex.PINS[preset_id] == (yaml_digest, recipe.sha256), (
            f"{preset_id}: PINS is stale, paste the line above"
        )
        assert ex.recipe_sha256(recipe.to_dict()) == recipe.sha256, preset_id
        path = OUT / f"{preset_id}.recipe.json"
        path.write_text(json.dumps(recipe.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(path.name, recipe.sha256[:12])
    print("QZ_FIXTURES_OK")


if __name__ == "__main__":
    main()
