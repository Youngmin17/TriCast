#!/bin/bash
# GPU wrapper for the FP4 vectors, launched through gpu_cap.sh:
#   bash /scratch/uceeeee/bin/gpu_cap.sh run nadpe_oracle -- bash run_gen_fp4.sh gen|verify <run_dir>
# Terminal marker: ORACLE_FP4_GEN_DONE rc=<rc> / ORACLE_FP4_VERIFY_DONE rc=<rc>  (no `set -u`)
WORK="${NADPE_ORACLE_WORK:-/scratch/uceeeee/tricast/nadpe_oracle}"
PY="${NADPE_ORACLE_PY:-/scratch/uceeeee/conda_envs/tricast/bin/python}"
MODE="${1:-gen}"
RUN_DIR="$2"
MARK="ORACLE_FP4_$(echo "$MODE" | tr '[:lower:]' '[:upper:]')_DONE"
[ -n "$RUN_DIR" ] || { echo "$MARK rc=98 (no run dir)"; exit 98; }
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$WORK" || { echo "$MARK rc=97 (no workdir)"; exit 97; }
echo "RUN_START mode=fp4-$MODE utc=$(date -u +%FT%TZ) host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PY" gen_vectors_fp4.py "$MODE" --run-dir "$RUN_DIR"
rc=$?
echo "$MARK rc=$rc utc=$(date -u +%FT%TZ)"
exit $rc
