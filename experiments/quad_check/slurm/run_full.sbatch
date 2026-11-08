#!/bin/bash
# All GPU stages in one job (queue waits dominate, so one job beats a chain of five).
# Each stage is skipped if its output already exists, so a resubmit resumes.
#SBATCH -J quad_check
#SBATCH -A cmsc416-class
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/zt1/project/cmsc416/user/%u/quad_check/logs/%x-%j.out
set -euo pipefail
source "$HOME/Algernon/experiments/quad_check/slurm/env.sh"
cd "$QC_CODE"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # everything was pre-downloaded by setup_env.sh

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# QC_VENV selects a prebuilt venv (e.g. the cu126 one for V100s); default is the uv project env
if [ -n "${QC_VENV:-}" ]; then PY="$QC_VENV/bin/python"; else PY="uv run --frozen --offline python"; fi
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable (driver too old for this torch build?)'; x=torch.randn(64,64,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(), float((x@x).sum()))"

stage() { echo; echo "=== $1  $(date +%T) ==="; }
CFG=${QC_CFG:-configs/full.yaml}
echo "config: $CFG  out: $QC_OUT"
[ -f "$QC_OUT/data/check_new.jsonl" ] || { stage prompts; $PY -m quad_check.prompts --config $CFG; }
[ -d "$QC_OUT/adapters/final" ]       || { stage train;   $PY -m quad_check.train_pilot --config $CFG; }
[ -f "$QC_OUT/rollouts_old.pt" ]      || { stage rollouts; $PY -m quad_check.rollouts --config $CFG; }
[ -f "$QC_OUT/check_final.json" ]     || { stage check-final; $PY -m quad_check.check --config $CFG --adapters final; }
[ -f "$QC_OUT/eval.json" ]            || { stage eval;    $PY -m quad_check.eval_acc --config $CFG; }
[ -f "$QC_OUT/check_step_75.json" ]   || { stage check-ckpts; $PY -m quad_check.check --config $CFG --adapters step_25 step_50 step_75; }
stage analyze
$PY -m quad_check.analyze --config $CFG
echo "all done $(date +%T)"
