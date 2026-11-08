# Written by app.tune_loop: S, T, R are set by the launch command.
export PY=/scratch/uceeeee/conda_envs/tricast/bin/python
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/uceeeee/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_CBWR=COMPATIBLE
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/scratch/uceeeee/tricast/cache/triton
export CUDA_CACHE_PATH=/scratch/uceeeee/tricast/cache/cuda
export TORCHINDUCTOR_CACHE_DIR=/scratch/uceeeee/tricast/cache/inductor
export PYTHONPATH="$T/src:$T":/scratch/uceeeee/tricast/pydeps_qm
echo "== TUNE_PREFLIGHT host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES head=$(cd "$T" && git rev-parse HEAD) dirty=$(cd "$T" && git status --short | wc -l) round=$R"
bash "$S/rounds/$R/measure.sh"
