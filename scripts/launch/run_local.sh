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

# If the ml-netprof agent is running on localhost:9100, scrape its Prometheus
# endpoint as W&B custom metrics during the training run.
# Start the agent first: make agent-build-darwin && cd agent && ./bin/agent-darwin &
#
# wandb.setup() must be called in the same Python process before wandb.init() —
# WANDB__ env vars are parsed by Python but not reliably forwarded to wandb-core.
# We write a temp wrapper that calls setup() then runs the training module in-process.
if curl -sf http://localhost:9100/healthz >/dev/null 2>&1; then
    _WRAPPER=$(mktemp /tmp/ml_run_XXXXXX.py)
    trap "rm -f '$_WRAPPER'" EXIT INT TERM
    cat > "$_WRAPPER" << 'PYEOF'
import sys, os, wandb, runpy
sys.path.insert(0, os.getcwd())  # make nanochat/scripts importable (cwd = nanochat/)
wandb.setup(settings=wandb.Settings(
    x_stats_open_metrics_endpoints={"agent": "http://localhost:9100/metrics"},
))
runpy.run_module("scripts.base_train", run_name="__main__", alter_sys=True)
PYEOF
    # Use plain python (not torchrun): MPS is single-process; device_type comes from config.
    "$REPO_ROOT/.venv/bin/python" "$_WRAPPER" $TRAIN_ARGS
else
    # Use plain python (not torchrun): MPS is single-process; device_type comes from config.
    "$REPO_ROOT/.venv/bin/python" -m scripts.base_train $TRAIN_ARGS
fi
