"""tricast acceptance of page-generated recipes (input: qz_accept_input.json written by qz_check.mjs).

For every page YAML: it parses to exactly app.quantizer_recipe.build(...), tricast load_recipe accepts it,
each of the 196 decoder Linear layers gets (spec_for) the spec its method has when used alone as the defaults
(None for BF16) — with the entry's options, the method-variant used alone — and the KV block is the preset's
(with the entry's kv_options). Entries with options also check that the calibration settings reach
Recipe.calibration_options. Also re-derives the exporter's pinned preset SHA-256s.

usage: cd <repo> && PYTHONPATH=src:. python scripts/quantizer/qz_tricast_accept.py
<out_dir>/qz_accept_input.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from app import export_quantizer_demo as ex
from app import quantizer_recipe as qr

from tricast.recipe import load_recipe


def main() -> int:
    entries = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    alone = {m: load_recipe({"name": m, "defaults": dict(f)}).defaults for m, f in qr.METHODS.items() if f}
    kv_alone = {
        kv: load_recipe({"name": kv, "defaults": dict(qr.METHODS["nvfp4"]), "kv": dict(spec)}).to_dict()["kv"]
        for kv, spec in qr.KV.items()
        if spec
    }
    layers = variants = 0
    methods: set[str] = set()
    for entry in entries:
        data = yaml.safe_load(entry["yaml"])
        extra = {key: entry.get(key) for key in ("options", "calibration", "kv_options")}
        assert data == qr.build(entry["name"], entry["default"], entry["grid"], entry["kv"], **extra), entry[
            "name"
        ]
        recipe = load_recipe(data)
        options = entry.get("options") or {}
        own = {
            m: load_recipe({"name": m, "defaults": qr.variant(m, o)}).defaults
            for m, o in options.items()
            if qr.METHODS[m] is not None
        }
        variants += len(own)
        for b in range(qr.BLOCKS):
            for k, kind in enumerate(qr.KINDS):
                method = entry["grid"][b][k]
                spec = recipe.spec_for(
                    qr.layer_name(b, kind), num_layers=qr.BLOCKS, decoder=("model.layers", qr.BLOCKS)
                )
                want = None if method == "bf16" else own.get(method, alone[method])
                assert spec == want, (entry["name"], b, kind, method)
                layers += 1
                methods.add(method)
        kv = kv_alone.get(entry["kv"])
        if kv is not None and entry.get("kv_options"):
            kv = load_recipe(
                {
                    "name": "kv",
                    "defaults": dict(qr.METHODS["nvfp4"]),
                    "kv": {**qr.KV[entry["kv"]], **entry["kv_options"]},
                }
            ).to_dict()["kv"]
        assert recipe.to_dict()["kv"] == kv, (entry["name"], entry["kv"])
        if entry.get("calibration") and "calibration" in data:
            settings = qr.check_calibration(entry["calibration"])
            assert all(recipe.calibration_options[key] == value for key, value in settings.items()), entry[
                "name"
            ]
    for preset_id in qr.PRESETS:
        assert load_recipe(yaml.safe_load(qr.to_yaml(qr.preset(preset_id)))).sha256 == ex.preset_sha256(
            preset_id
        )
    print(
        f"recipes {len(entries)}, layers checked {layers}, methods seen {len(methods)}/{len(qr.METHODS)}, "
        f"method variants {variants}, kv seen {sorted({e['kv'] for e in entries})}, "
        f"preset pins {len(qr.PRESETS)} ok"
    )
    print("QZ_TRICAST_ACCEPT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
