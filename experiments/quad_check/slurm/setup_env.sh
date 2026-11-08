#!/bin/bash
# Run once on a Zaratan LOGIN node: installs uv, builds the venv on scratch,
# downloads the model + datasets, and builds the prompt sets (CPU only, ~2 min).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$QC_CODE"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
uv run --frozen python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
uv run --frozen python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-0.5B-Instruct'))"
uv run --frozen python -m quad_check.prompts --config configs/full.yaml
echo "setup done; outputs in $QC_OUT"
