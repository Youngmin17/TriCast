# tune round 2: whole-recipe KL of opt_act_order_r2, opt_damp_low_r2, opt_damp_high_r2, g256_mlp_upgrades_r2 on the session's validation windows.
# S = this session directory on the cluster; T = a clean TriCast tree; PY = its python.
S=${S:?set S to the session directory}; T=${T:?set T to the TriCast tree}; PY=${PY:-python}
cd "$T" || exit 3
"$PY" -m app.autotune measure --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --device cuda --split validation --windows 8 --seqlen 512 --seed 43 --calibrate ${CALIB_CACHE:+--calib-cache "$CALIB_CACHE"} --out "$S/rounds/2/kl.json" "$S/proposals/opt_act_order_r2/recipe.yaml" "$S/proposals/opt_damp_low_r2/recipe.yaml" "$S/proposals/opt_damp_high_r2/recipe.yaml" "$S/proposals/g256_mlp_upgrades_r2/recipe.yaml" || { rc=$?; echo "TUNE_ROUND_FAIL round=2 rc=$rc"; exit "$rc"; }
echo "TUNE_ROUND_DONE round=2"
