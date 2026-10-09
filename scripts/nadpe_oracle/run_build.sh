#!/bin/bash
# Build wrapper (CPU only). Usage: bash run_build.sh fp8|fp4 > log 2>&1
# Terminal marker: ORACLE_BUILD_DONE rc=<rc> target=<target>
# (no `set -u`: torch/conda activation helpers reference unset vars)
WORK="${NADPE_ORACLE_WORK:-/scratch/uceeeee/tricast/nadpe_oracle}"
PY="${NADPE_ORACLE_PY:-/scratch/uceeeee/conda_envs/tricast/bin/python}"
TARGET="${1:-fp8}"
export PYTHONNOUSERSITE=1
# /usr/local/bin/ninja on geneva is a broken python wrapper; use the native
# binary from `pip install --target $WORK/_tools ninja` instead.
export PATH="$WORK/_tools/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8
cd "$WORK" || { echo "ORACLE_BUILD_DONE rc=97 target=$TARGET (no workdir)"; exit 97; }
echo "BUILD_START target=$TARGET utc=$(date -u +%FT%TZ) host=$(hostname -s)"
echo "ninja=$(command -v ninja) $(ninja --version)"
nvcc --version | tail -2
"$PY" build.py --target "$TARGET"
rc=$?
echo "ORACLE_BUILD_DONE rc=$rc target=$TARGET utc=$(date -u +%FT%TZ)"
exit $rc
