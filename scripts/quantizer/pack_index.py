"""The low-bit packages the tuning app's step 5 shows (app/web/demo/autotune/packages.json), from what the
cluster wrote: a language model's package directory (``python -m tricast.pack build``, its manifest.json and
the ``.run_codes.json`` beside it) and a detector's ONNX files (``scripts/quantizer/yolo_tune.py onnx``).

    python scripts/quantizer/pack_index.py --out app/web/demo/autotune/packages.json \
        --llm runs/…/packages/q06b_act_damp05_r3 --llm runs/…/packages/q8b_auto_3.00 \
        --onnx yolo11n=runs/…/onnx --onnx-copy app/web/demo/autotune/packages/yolo11n

A language model's entry names its model and recipe file SHA-256 (the page attaches it only to that model's
setup with that very file); the
package itself stays on the cluster (hundreds of MB to GB). A detector's ONNX files with integer weights are
small (about 1.5 MB) and are copied next to the page, where step 5 links them; the original model's FP32 file
keeps its record only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

SCHEMA = "tricast.packages/1"
PPL_LIMIT = 0.001  # the package's PPL against TriCast's, relative (the same 0.1% as AC4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def llm_entry(package: Path) -> dict[str, Any]:
    """The page's entry for one package directory (its manifest.json and quant_config.json; model.safetensors
    is checked too when it is here). Refused unless every layer decoded to TriCast's weight, the files agree
    with each other and the package's PPL is within PPL_LIMIT of TriCast's."""
    from app.autotune_export import MODEL_FILES

    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((package / "quant_config.json").read_text(encoding="utf-8"))
    checks = manifest["checks"]
    decoded = checks["decoded_equals_tricast_weight"]
    if decoded["mismatched"]:
        raise SystemExit(f"{package}: {len(decoded['mismatched'])} layers do not decode to TriCast's weight")
    layers = len(config["layers"])
    if not decoded["layers"] == manifest["quantized_layers"] == layers:
        raise SystemExit(f"{package}: the decode check covered {decoded['layers']} of {layers} layers")
    recipe_sha = hashlib.sha256(config["recipe"]["yaml"].encode("utf-8")).hexdigest()
    if not recipe_sha == config["recipe"]["sha256"] == manifest["recipe_sha256"]:
        raise SystemExit(
            f"{package}: the recipe text, quant_config.json and manifest.json name other recipes"
        )
    for name in ("quant_config.json", "model.safetensors"):
        if (package / name).is_file() and _sha256(package / name) != manifest["files"][name]["sha256"]:
            raise SystemExit(f"{package}: {name} is not the file manifest.json describes")
    tricast, packed = checks.get("ppl_tricast"), checks.get("ppl_package")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (tricast, packed)):
        raise SystemExit(f"{package}: no measured PPL of TriCast's model and of the package")
    gap = packed / tricast - 1
    if not abs(gap) <= PPL_LIMIT:
        raise SystemExit(f"{package}: package PPL vs TriCast {gap:+.4%} (needs to be within {PPL_LIMIT:.1%})")
    reasons = sorted(set(manifest["fallback_layers"].values()))
    codes = package.with_name(package.name + ".run_codes.json")
    codes_run: dict[str, Any] = {}
    if codes.is_file() and codes.stat().st_size:
        # the run's JSON; a captured stdout has the PACK_RUN_DONE line after it
        codes_run = json.JSONDecoder().raw_decode(codes.read_text(encoding="utf-8").lstrip())[0]
        if Path(codes_run.get("package", "")).name != package.name:
            raise SystemExit(f"{codes}: a run of another package ({codes_run.get('package')})")
    original = MODEL_FILES.get(manifest["model"], {}).get("file_bytes")
    entry = {
        "model": manifest["model"],
        "revision": manifest["revision"],
        "dir": package.name,
        "recipe_sha256": manifest["recipe_sha256"],
        "files": [
            {"name": name, "bytes": item["bytes"], "sha256": item["sha256"]}
            for name, item in manifest["files"].items()
        ]
        + [{"name": "manifest.json", "bytes": (package / "manifest.json").stat().st_size, "sha256": None}],
        "bytes": manifest["bytes"] + (package / "manifest.json").stat().st_size,
        "weights_bytes": manifest["weights_bytes"],
        "original_bytes": original,
        "tied": sorted(config.get("tied") or {}),
        "layers": manifest["quantized_layers"],
        "kinds": manifest["layer_kinds"],
        "fallback": {"count": len(manifest["fallback_layers"]), "reasons": reasons},
        "decoded_equal": {"layers": decoded["layers"], "mismatched": 0},
        "ppl_tricast": checks.get("ppl_tricast"),
        "ppl_package": checks.get("ppl_package"),
        "ppl_relative_difference": gap,
        "ppl_codes": codes_run.get("ppl"),
        "gpu": checks.get("gpu"),
        "model_sha": checks.get("model_sha"),
        "gpu_memory_loaded_bytes": codes_run.get("gpu_memory_loaded_bytes"),
        "gpu_memory_peak_eval_bytes": codes_run.get("gpu_memory_peak_eval_bytes"),
        "seconds": checks.get("seconds"),
    }
    return entry


def onnx_entries(model: str, work: Path, copy: Path | None) -> dict[str, Any]:
    out = {}
    for path in sorted(work.glob("onnx_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        onnx = work / record["file"]
        digest = hashlib.sha256(onnx.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise SystemExit(f"{onnx}: SHA-256 differs from {path.name}")
        # a setup with no quantized conv (the original model, FP32) keeps its record; its file is not copied
        shipped = copy is not None and bool(record["stored"])
        if shipped:
            copy.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(onnx, copy / record["file"])
        kinds: dict[str, int] = {}
        for item in record["stored"].values():
            kinds[item["dtype"]] = kinds.get(item["dtype"], 0) + 1
        out[record["setup"]] = {
            "file": f"packages/{model}/{record['file']}" if shipped else None,
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "kinds": kinds,
            "metrics": record["metrics"],
            "pytorch_metrics": record["pytorch_metrics"],
            "raw_max_abs_diff": max(record["raw_max_abs_diff"]),
            "demo_boxes": record["demo_boxes"],
            "images": record["evaluation_images"],
            "env": {k: record["env"].get(k) for k in ("onnx", "onnxruntime", "host", "git_sha", "utc")},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--llm", type=Path, action="append", default=[])
    parser.add_argument("--onnx", action="append", default=[], help="<model>=<onnx work dir>")
    parser.add_argument("--onnx-copy", type=Path, help="where the ONNX files go next to the page")
    args = parser.parse_args()
    llm = [llm_entry(path) for path in args.llm]
    keys = [(e["model"], e["recipe_sha256"]) for e in llm]
    if len(set(keys)) != len(keys):
        raise SystemExit("two packages of the same model and recipe file")
    onnx = {}
    for spec in args.onnx:
        model, work = spec.split("=", 1)
        onnx[model] = onnx_entries(model, Path(work), args.onnx_copy)
    args.out.write_text(
        json.dumps({"schema": SCHEMA, "llm": llm, "onnx": onnx}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"PACK_INDEX llm={len(llm)} onnx={sum(len(v) for v in onnx.values())} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
