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
#
# On cloud VMs (Azure, GCP, etc.) the public IPs used for SSH are often
# different from the private IPs that nodes use to talk to each other.
# Set INTERNAL_IPS for the torchrun rendezvous/NCCL traffic:
#   NODES="<public0> <public1>" INTERNAL_IPS="<private0> <private1>" ./scripts/launch/nanochat/run_multinode.sh
# ============================================================================

# ---------- configuration (edit these or set via env) ----------

# Space-separated list of node IPs/hostnames (used for SSH). First node is the master.
NODES="${NODES:-node0 node1}"

# Internal/private IPs for torchrun rendezvous and NCCL traffic.
# If unset, NODES are used for both SSH and torchrun (fine when nodes are on the same LAN).
INTERNAL_IPS="${INTERNAL_IPS:-}"

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

# Path to SSH private key (empty = use ssh-agent / default key)
SSH_KEY="${SSH_KEY:-}"

# NCCL transport: "ib" for InfiniBand, "tcp" for TCP sockets (default).
# Use "tcp" if IB verbs fail (e.g. ibv_create_qp errors on Azure VMs without
# SR-IOV InfiniBand). Use "ib" only when VMs have working IB connectivity.
NCCL_TRANSPORT="${NCCL_TRANSPORT:-tcp}"

# Extra flags passed through to nanochat (after --)
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# ---------- derived values ----------

read -ra NODE_ARRAY <<< "$NODES"
NNODES=${#NODE_ARRAY[@]}

# If INTERNAL_IPS is set, use those for torchrun; otherwise fall back to NODES.
if [[ -n "$INTERNAL_IPS" ]]; then
    read -ra INTERNAL_ARRAY <<< "$INTERNAL_IPS"
    if [[ ${#INTERNAL_ARRAY[@]} -ne $NNODES ]]; then
        echo "ERROR: INTERNAL_IPS has ${#INTERNAL_ARRAY[@]} entries but NODES has $NNODES" >&2
        exit 1
    fi
else
    INTERNAL_ARRAY=("${NODE_ARRAY[@]}")
fi
MASTER_ADDR="${INTERNAL_ARRAY[0]}"

SSH_PREFIX=""
if [[ -n "$SSH_USER" ]]; then
    SSH_PREFIX="${SSH_USER}@"
fi

SSH_OPTS=(-o StrictHostKeyChecking=no)
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS+=(-i "$SSH_KEY")
fi

LOG_DIR="logs/multinode_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Multi-node nanochat training ==="
echo "  SSH nodes:      ${NODE_ARRAY[*]}"
echo "  Internal IPs:   ${INTERNAL_ARRAY[*]}"
echo "  GPUs per node:  $GPUS_PER_NODE"
echo "  Total GPUs:     $((NNODES * GPUS_PER_NODE))"
echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
echo "  Config:         $CONFIG"
echo "  Logs:           $LOG_DIR/"
echo ""

# ---------- build the torchrun command ----------

# Tell NCCL which transport to use and which network interface carries the
# internal IPs (eth0 on most Azure VMs). Without this, NCCL may pick an IB
# interface that isn't wired between the VMs.
NCCL_ENV=""
if [[ "$NCCL_TRANSPORT" == "tcp" ]]; then
    NCCL_ENV="NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"
fi

TORCHRUN_CMD="cd ${REPO_DIR} && ${NCCL_ENV} \
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

# ---------- cleanup on exit (Ctrl+C or failure) ----------

cleanup() {
    echo ""
    echo "Shutting down all nodes..."
    # Kill local SSH/bash background jobs
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Kill remote torchrun processes
    for NODE in "${NODE_ARRAY[@]}"; do
        ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "${SSH_PREFIX}${NODE}" \
            "pkill -f torchrun; pkill -f base_train" 2>/dev/null || true
    done
    echo "Done. Logs are in $LOG_DIR/"
}
trap cleanup EXIT

# ---------- launch on each node ----------

PIDS=()

for i in "${!NODE_ARRAY[@]}"; do
    NODE="${NODE_ARRAY[$i]}"
    CMD="${TORCHRUN_CMD//RANK_PLACEHOLDER/$i}"
    LOG_FILE="$LOG_DIR/node_${i}.log"

    echo "[node $i] $NODE — launching (rank $i/$NNODES) → $LOG_FILE"

    if [[ "$i" -eq 0 && -z "$SSH_USER" && ("$NODE" == "localhost" || "$NODE" == "127.0.0.1") ]]; then
        bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    else
        ssh "${SSH_OPTS[@]}" "${SSH_PREFIX}${NODE}" "$CMD" > "$LOG_FILE" 2>&1 &
    fi
    PIDS+=($!)
done

echo ""
echo "All nodes launched. Tailing node 0 log (Ctrl+C to stop all nodes)..."
echo ""

# Tail node 0 in the background so the user sees live progress
tail -f "$LOG_DIR/node_0.log" 2>/dev/null &
TAIL_PID=$!

# ---------- wait for all SSH sessions ----------

EXIT_CODE=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo ""
        echo "[node $i] ${NODE_ARRAY[$i]} — exited with error (see $LOG_DIR/node_${i}.log)"
        EXIT_CODE=1
    fi
done

kill "$TAIL_PID" 2>/dev/null || true

if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo ""
    echo "Training complete on all nodes."
else
    echo ""
    echo "Training finished with errors. Check logs:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  node $i: $LOG_DIR/node_${i}.log"
    done
    exit 1
fi
