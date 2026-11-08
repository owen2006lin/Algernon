# Sourced by setup_env.sh and the sbatch script. Keeps every large file on scratch.
export QC_SCRATCH=${QC_SCRATCH:-/scratch/zt1/project/cmsc416/user/$USER/quad_check}
export QC_CODE=${QC_CODE:-$HOME/Algernon/experiments/quad_check}
export QC_OUT=${QC_OUT:-$QC_SCRATCH/outputs/full}
export HF_HOME=$QC_SCRATCH/hf
export UV_CACHE_DIR=$QC_SCRATCH/uv-cache
export UV_PYTHON_INSTALL_DIR=$QC_SCRATCH/uv-python
export UV_PROJECT_ENVIRONMENT=$QC_SCRATCH/venv
export PATH=$HOME/.local/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$QC_SCRATCH/logs" "$QC_OUT" "$HF_HOME"
