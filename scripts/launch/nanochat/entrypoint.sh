#!/usr/bin/env bash
set -euo pipefail

export NANOCHAT_BASE_DIR=/workspace/data

echo "==> Downloading TinyStories..."
python scripts/data/prepare_tiny_dataset.py

echo "==> Training BPE tokenizer..."
cd nanochat
NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR \
  python -m scripts.tok_train --max-chars 50000000
cd /workspace

echo "==> Starting nanochat training..."
python scripts/launch/nanochat/run_local.py \
  scripts/launch/nanochat/config/local_linux_gpu.yaml
