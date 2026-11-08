# tune round 4: whole-recipe KL of replan_r4 on the session's validation windows.
# S = this session directory on the cluster; T = a clean TriCast tree; PY = its python.
S=${S:?set S to the session directory}; T=${T:?set T to the TriCast tree}; PY=${PY:-python}
cd "$T" || exit 3
"$PY" -m app.autotune measure --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --device cuda --split validation --windows 8 --seqlen 512 --seed 43 --calibrate ${CALIB_CACHE:+--calib-cache "$CALIB_CACHE"} --out "$S/rounds/4/kl.json" "$S/proposals/replan_r4/recipe.yaml" || { rc=$?; echo "TUNE_ROUND_FAIL round=4 rc=$rc"; exit "$rc"; }
echo "TUNE_ROUND_DONE round=4"
