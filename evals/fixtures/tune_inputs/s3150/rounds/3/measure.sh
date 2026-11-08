# tune round 3: whole-recipe KL of act_damp05_r3, act_damp10_r3, act_damp025_r3, int4_g256_fund_b14down_r3 on the session's validation windows.
# S = this session directory on the cluster; T = a clean TriCast tree; PY = its python.
S=${S:?set S to the session directory}; T=${T:?set T to the TriCast tree}; PY=${PY:-python}
cd "$T" || exit 3
"$PY" -m app.autotune measure --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --device cuda --split validation --windows 8 --seqlen 512 --seed 43 --calibrate ${CALIB_CACHE:+--calib-cache "$CALIB_CACHE"} --out "$S/rounds/3/kl.json" "$S/proposals/act_damp05_r3/recipe.yaml" "$S/proposals/act_damp10_r3/recipe.yaml" "$S/proposals/act_damp025_r3/recipe.yaml" "$S/proposals/int4_g256_fund_b14down_r3/recipe.yaml" || { rc=$?; echo "TUNE_ROUND_FAIL round=3 rc=$rc"; exit "$rc"; }
echo "TUNE_ROUND_DONE round=3"
