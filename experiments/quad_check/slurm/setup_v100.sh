#!/bin/bash
# Login node, once (after setup_env.sh): a second venv with torch 2.14.1+cu126 for the V100 node.
# CUDA 13 builds (the default torch 2.15+cu132) no longer support Volta (sm_70).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$QC_CODE"
V=$QC_SCRATCH/venv-v100
uv venv --python 3.14 "$V"
uv pip install --python "$V/bin/python" "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$V/bin/python" -e . "torch==2.14.1" --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
"$V/bin/python" -c "import torch, transformers, peft; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), transformers.__version__, peft.__version__)"
