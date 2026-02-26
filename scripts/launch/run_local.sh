#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$REPO_ROOT/configs/local_mac_tiny.yaml}"

# Load env vars (WANDB_BASE_URL, WANDB_API_KEY)
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

export WANDB_BASE_URL
export WANDB_API_KEY

# Point nanochat at repo's data directory so it reads data/base_data/*.parquet
# and writes checkpoints to data/base_checkpoints/
export NANOCHAT_BASE_DIR="$REPO_ROOT/data"

# Convert flat YAML config (key: value) → CLI flags (--key=value).
# Uses only stdlib so no pyyaml dependency is needed.
TRAIN_ARGS=$("$REPO_ROOT/.venv/bin/python" - "$CONFIG" <<'EOF'
import sys, re
flags = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, _, val = line.partition(':')
        val = val.strip()
        if val:
            flags.append(f"--{key.strip().replace('_', '-')}={val}")
print(" ".join(flags))
EOF)

cd "$REPO_ROOT/nanochat"
# Use plain python (not torchrun): MPS is single-process; device_type comes from config.
"$REPO_ROOT/.venv/bin/python" -m scripts.base_train $TRAIN_ARGS
