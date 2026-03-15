#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_multinode.sh — Launch nanochat training across multiple GPU nodes via SSH
#
# Prerequisites on every node:
#   - This repo cloned to the same path (REPO_DIR below)
#   - Python venv set up (make setup)
#   - Data prepared  (make prepare-data && make tokenizer)
#   - .env populated with W&B credentials
#   - SSH access from this machine (the one running this script)
#
# Usage:
#   ./scripts/launch/nanochat/run_multinode.sh
#
# Override defaults with environment variables:
#   NODES="10.0.0.10 10.0.0.11" GPUS_PER_NODE=4 ./scripts/launch/nanochat/run_multinode.sh
# ============================================================================

# ---------- configuration (edit these or set via env) ----------

# Space-separated list of node IPs/hostnames. First node is the master.
NODES="${NODES:-node0 node1}"

# GPUs per node (must be the same on every node)
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"

# Repo path on the remote nodes
REPO_DIR="${REPO_DIR:-~/ml-infra-profiler}"

# Config file (relative to repo root)
CONFIG="${CONFIG:-scripts/launch/nanochat/config/multi_node_2x2.yaml}"

# Port for PyTorch rendezvous (pick any free port)
MASTER_PORT="${MASTER_PORT:-29500}"

# SSH user (empty = current user)
SSH_USER="${SSH_USER:-}"

# Extra flags passed through to nanochat (after --)
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# ---------- derived values ----------

read -ra NODE_ARRAY <<< "$NODES"
NNODES=${#NODE_ARRAY[@]}
MASTER_ADDR="${NODE_ARRAY[0]}"

SSH_PREFIX=""
if [[ -n "$SSH_USER" ]]; then
    SSH_PREFIX="${SSH_USER}@"
fi

echo "=== Multi-node nanochat training ==="
echo "  Nodes:          ${NODE_ARRAY[*]}"
echo "  GPUs per node:  $GPUS_PER_NODE"
echo "  Total GPUs:     $((NNODES * GPUS_PER_NODE))"
echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
echo "  Config:         $CONFIG"
echo ""

# ---------- build the torchrun command ----------

TORCHRUN_CMD="cd ${REPO_DIR} && \
${REPO_DIR}/.venv/bin/torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --node-rank=RANK_PLACEHOLDER \
    scripts/launch/nanochat/train_wrapper.py \
    --config ${REPO_DIR}/${CONFIG}"

if [[ -n "$EXTRA_FLAGS" ]]; then
    TORCHRUN_CMD="${TORCHRUN_CMD} -- ${EXTRA_FLAGS}"
fi

# ---------- launch on each node ----------

PIDS=()

for i in "${!NODE_ARRAY[@]}"; do
    NODE="${NODE_ARRAY[$i]}"
    CMD="${TORCHRUN_CMD//RANK_PLACEHOLDER/$i}"

    echo "[node $i] $NODE — launching (rank $i/$NNODES)"

    if [[ "$i" -eq 0 && -z "$SSH_USER" && ("$NODE" == "localhost" || "$NODE" == "127.0.0.1") ]]; then
        # Master node is local — run directly (useful for debugging)
        bash -c "$CMD" &
    else
        # Remote node — launch via SSH.
        # -f: go to background after auth
        # -o StrictHostKeyChecking=no: skip host key prompt on first connect
        ssh -f -o StrictHostKeyChecking=no "${SSH_PREFIX}${NODE}" "$CMD" &
    fi
    PIDS+=($!)
done

echo ""
echo "All nodes launched. Waiting for training to finish..."
echo "(Ctrl-C to abort — you may need to manually kill torchrun on remote nodes)"
echo ""

# ---------- wait for all SSH sessions ----------

EXIT_CODE=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "[node $i] ${NODE_ARRAY[$i]} — exited with error"
        EXIT_CODE=1
    fi
done

if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "Training complete on all nodes."
else
    echo "Training finished with errors on one or more nodes."
    exit 1
fi
