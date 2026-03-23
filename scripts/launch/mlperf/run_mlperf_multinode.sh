#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_multinode.sh — Launch MLPerf Llama3.1 8B training across multiple
# GPU nodes via SSH + Docker (NeMo container).
#
# Mirrors run_multinode.sh structure exactly; key difference is that each node
# runs the NeMo container via `docker run --network=host` rather than torchrun
# in a local venv.
#
# Prerequisites on every node:
#   - Docker installed and NeMo image pulled (make pull-nemo)
#   - C4 data at DATA_DIR (default /data/c4) — run make prepare-mlperf-data first
#   - .env populated with WANDB_API_KEY, WANDB_BASE_URL, WANDB_ENTITY
#   - ml-netprof-agent running as systemd service (for Prometheus scraping)
#   - SSH access from this machine
#
# Usage:
#   ./scripts/launch/mlperf/run_mlperf_multinode.sh
#
# Override defaults with environment variables:
#   NODES="10.0.0.10 10.0.0.11" INTERNAL_IPS="10.0.0.10 10.0.0.11" \
#   GPUS_PER_NODE=2 SSH_KEY=~/gpu-key.pem SSH_USER=azureuser \
#   make run-mlperf
# ============================================================================

# ---------- configuration (edit these or set via env) ----------

# Space-separated list of node IPs/hostnames (used for SSH). First node is master.
NODES="${NODES:-node0 node1}"

# Internal/private IPs for NCCL traffic and NeMo master address.
# If unset, NODES are used for both SSH and NCCL (fine on same LAN).
INTERNAL_IPS="${INTERNAL_IPS:-}"

# GPUs per node (must be the same on every node)
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"

# Repo path on the remote nodes
REPO_DIR="${REPO_DIR:-~/ml-infra-profiler}"

# NeMo container image
NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:24.12-rc0}"

# Config file (relative to repo root), mounted into container
CONFIG="${CONFIG:-scripts/launch/mlperf/config/llama3_8b_4gpu.yaml}"

# Data directory on the nodes (must exist and contain C4 mmap files)
DATA_DIR="${DATA_DIR:-/data/c4}"

# Results directory on the nodes (created if absent)
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results/mlperf}"

# Port for NCCL rendezvous
MASTER_PORT="${MASTER_PORT:-29500}"

# SSH user (empty = current user)
SSH_USER="${SSH_USER:-}"

# Path to SSH private key (empty = use ssh-agent / default key)
SSH_KEY="${SSH_KEY:-}"

# NCCL transport: "tcp" (default) or "ib" for InfiniBand.
# Use "tcp" on Azure VMs without SR-IOV IB; use "ib" only when IB is confirmed working.
NCCL_TRANSPORT="${NCCL_TRANSPORT:-tcp}"

# ---------- derived values ----------

read -ra NODE_ARRAY <<< "$NODES"
NNODES=${#NODE_ARRAY[@]}

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

LOG_DIR="logs/mlperf_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== MLPerf Llama3.1 8B multi-node training ==="
echo "  SSH nodes:      ${NODE_ARRAY[*]}"
echo "  Internal IPs:   ${INTERNAL_ARRAY[*]}"
echo "  GPUs per node:  $GPUS_PER_NODE"
echo "  Total GPUs:     $((NNODES * GPUS_PER_NODE))"
echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
echo "  NeMo image:     $NEMO_IMAGE"
echo "  Config:         $CONFIG"
echo "  Data:           $DATA_DIR"
echo "  Results:        $RESULTS_DIR"
echo "  Logs:           $LOG_DIR/"
echo ""

# ---------- build the docker run command ----------

NCCL_ENV_FLAGS=""
if [[ "$NCCL_TRANSPORT" == "tcp" ]]; then
    NCCL_ENV_FLAGS="-e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=eth0"
fi

# GPU device string: "device=0,1" for 2 GPUs, "device=0,1,2,3" for 4, etc.
GPU_INDICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

DOCKER_CMD="mkdir -p ${RESULTS_DIR} && docker run --rm \
    --gpus '\"device=${GPU_INDICES}\"' \
    --network=host \
    --shm-size=64g \
    -v ${DATA_DIR}:/data/c4:ro \
    -v ${RESULTS_DIR}:/results \
    -v ${REPO_DIR}/.env:/workspace/.env:ro \
    -v ${REPO_DIR}/scripts/launch/mlperf/nemo_entrypoint.sh:/workspace/nemo_entrypoint.sh:ro \
    -v ${REPO_DIR}/${CONFIG}:/workspace/config.yaml:ro \
    -e MASTER_ADDR=${MASTER_ADDR} \
    -e MASTER_PORT=${MASTER_PORT} \
    -e NNODES=${NNODES} \
    -e GPUS_PER_NODE=${GPUS_PER_NODE} \
    -e NODE_RANK=RANK_PLACEHOLDER \
    ${NCCL_ENV_FLAGS} \
    --env-file ${REPO_DIR}/.env \
    ${NEMO_IMAGE} \
    bash /workspace/nemo_entrypoint.sh"

# ---------- cleanup on exit (Ctrl+C or failure) ----------

cleanup() {
    echo ""
    echo "Shutting down all nodes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for NODE in "${NODE_ARRAY[@]}"; do
        ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "${SSH_PREFIX}${NODE}" \
            "docker ps -q --filter ancestor=${NEMO_IMAGE} | xargs -r docker stop" 2>/dev/null || true
    done
    echo "Done. Logs are in $LOG_DIR/"
}
trap cleanup EXIT

# ---------- launch on each node ----------

PIDS=()

for i in "${!NODE_ARRAY[@]}"; do
    NODE="${NODE_ARRAY[$i]}"
    CMD="${DOCKER_CMD//RANK_PLACEHOLDER/$i}"
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
    echo "MLPerf training complete on all nodes."
    echo "Run 'make verify-mlperf' to check val perplexity target (≤ 3.3)."
else
    echo ""
    echo "Training finished with errors. Check logs:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  node $i: $LOG_DIR/node_${i}.log"
    done
    exit 1
fi
