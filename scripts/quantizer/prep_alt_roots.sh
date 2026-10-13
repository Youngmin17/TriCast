#!/usr/bin/env bash
# (Re)build the alternate static roots the CDP checks serve: page files are symlinks into app/web, so they always
# test the current code; only the demo data differs.
#   <alt>/empty   no demo bundles at all            (layer map and Quantizer empty states)
#   <alt>/bad     bundles with a wrong schema        (error states)
#   <alt>/synth   Quantizer data exported from SYNTHETIC run records (made-up numbers, tests only)
# usage: bash scripts/quantizer/prep_alt_roots.sh <repo> <alt_dir> <python>   (python: pyyaml; no torch needed)
set -euo pipefail
REPO=$1; ALT=$2; PY=$3
HERE=$(cd "$(dirname "$0")" && pwd)
WEB="$REPO/app/web"
rm -rf "$ALT/empty" "$ALT/bad" "$ALT/synth" "$ALT/synth_runs"
for root in empty bad synth; do
  mkdir -p "$ALT/$root"
  for item in index.html layers.html layers.css quantizer.html quantizer.css styles.css js assets; do
    ln -s "$WEB/$item" "$ALT/$root/$item"
  done
done
mkdir -p "$ALT/bad/demo/layers" "$ALT/bad/demo/quantizer"
echo '{"schema": "tricast.layer_map/0", "reports": []}' > "$ALT/bad/demo/layers/index.json"
echo '{"schema": "tricast.quantizer_demo/0"}' > "$ALT/bad/demo/quantizer/presets.json"
mkdir -p "$ALT/synth/demo"
ln -s "$WEB/demo/layers" "$ALT/synth/demo/layers"
(cd "$REPO" && PYTHONPATH="$REPO" "$PY" "$HERE/make_qz_synth.py" "$ALT/synth_runs" "$ALT/synth/demo/quantizer")
echo "ALT_ROOTS_READY $ALT"
