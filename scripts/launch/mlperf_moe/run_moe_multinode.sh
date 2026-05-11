#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_moe_multinode.sh — Launch the MoE FSDP training run across multiple
# GPU nodes via SSH + Docker (using the existing ml-netprof/diloco image,
# which has PyTorch 2.4 + transformers 4.45.2 baked in).
#
# Mirrors scripts/launch/mlperf/run_mlperf_multinode.sh, but with --entrypoint
# overridden to torchrun and the new train_moe.py script bind-mounted in.
#
# Prerequisites on every node:
#   - Docker installed; ml-netprof/diloco:latest pulled (or buildable)
#   - ml-netprof-agent running as systemd service
#   - SSH access from this machine
#
# Usage (typical):
#   NODES="20.29.43.19 172.212.226.225" \
#   INTERNAL_IPS="172.21.0.4 172.21.0.5" \
#   GPUS_PER_NODE=2 SSH_KEY=~/.ssh/gpu-ib_key.pem SSH_USER=azureuser \
#   MAX_STEPS=5 FAKE_DATA=1 RUN_TAG=smoke \
#   bash scripts/launch/mlperf_moe/run_moe_multinode.sh
# ============================================================================

# ---------- configuration ----------

NODES="${NODES:-}"
INTERNAL_IPS="${INTERNAL_IPS:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
SSH_USER="${SSH_USER:-azureuser}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/gpu-ib_key.pem}"

REPO_DIR="${REPO_DIR:-\$HOME/ml-infra-profiler}"
IMAGE="${IMAGE:-ml-netprof/diloco:latest}"

# Training knobs
MAX_STEPS="${MAX_STEPS:-2000}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-8}"
MODEL_SIZE="${MODEL_SIZE:-moe-3b}"
LR="${LR:-3e-4}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-5}"
FAKE_DATA="${FAKE_DATA:-0}"
RUN_TAG="${RUN_TAG:-run}"

MASTER_PORT="${MASTER_PORT:-29501}"

# NCCL transport: "tcp" (default — works on Azure A100s without SR-IOV IB) or "ib"
NCCL_TRANSPORT="${NCCL_TRANSPORT:-tcp}"

# ---------- validate ----------

if [[ -z "$NODES" ]]; then
    echo "ERROR: NODES env var is required (space-separated public IPs)" >&2
    exit 2
fi

read -ra NODE_ARRAY <<< "$NODES"
NNODES=${#NODE_ARRAY[@]}

if [[ -n "$INTERNAL_IPS" ]]; then
    read -ra INTERNAL_ARRAY <<< "$INTERNAL_IPS"
    if [[ ${#INTERNAL_ARRAY[@]} -ne $NNODES ]]; then
        echo "ERROR: INTERNAL_IPS count != NODES count" >&2
        exit 2
    fi
else
    INTERNAL_ARRAY=("${NODE_ARRAY[@]}")
fi
MASTER_ADDR="${INTERNAL_ARRAY[0]}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30)
if [[ -n "$SSH_KEY" ]]; then
    SSH_OPTS+=(-i "$SSH_KEY")
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/mlperf_moe_${RUN_TAG}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== MoE FSDP multi-node training ==="
echo "  Nodes (SSH):    ${NODE_ARRAY[*]}"
echo "  Internal IPs:   ${INTERNAL_ARRAY[*]}"
echo "  Master:         $MASTER_ADDR:$MASTER_PORT"
echo "  GPUs per node:  $GPUS_PER_NODE  (total: $((NNODES * GPUS_PER_NODE)))"
echo "  Image:          $IMAGE"
echo "  Model size:     $MODEL_SIZE"
echo "  Seq length:     $SEQ_LENGTH"
echo "  Per-dev batch:  $PER_DEVICE_BATCH_SIZE"
echo "  Total batch:    $TOTAL_BATCH_SIZE"
echo "  Max steps:      $MAX_STEPS"
echo "  Fake data:      $FAKE_DATA"
echo "  NCCL transport: $NCCL_TRANSPORT"
echo "  Log dir:        $LOG_DIR/"
echo ""

# ---------- build the docker run command ----------

NCCL_ENV_FLAGS="-e NCCL_DEBUG=WARN -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if [[ "$NCCL_TRANSPORT" == "tcp" ]]; then
    NCCL_ENV_FLAGS="$NCCL_ENV_FLAGS -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=eth0 -e NCCL_P2P_DISABLE=0"
elif [[ "$NCCL_TRANSPORT" == "ib" ]]; then
    NCCL_ENV_FLAGS="$NCCL_ENV_FLAGS -e NCCL_IB_DISABLE=0 -e NCCL_NET_GDR_LEVEL=PHB"
fi

GPU_INDICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

FAKE_DATA_FLAG=""
if [[ "$FAKE_DATA" == "1" ]]; then
    FAKE_DATA_FLAG="--fake-data"
fi

# Build the torchrun command (passed as args to the container — its entrypoint is torchrun)
# Each node will run its own torchrun with its own --node-rank.
TORCHRUN_ARGS_PREFIX="--nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

TRAIN_ARGS="/workspace/train_moe.py \
    --model-size ${MODEL_SIZE} \
    --seq-length ${SEQ_LENGTH} \
    --per-device-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --total-batch-size ${TOTAL_BATCH_SIZE} \
    --total-steps ${TOTAL_STEPS} \
    --max-steps ${MAX_STEPS} \
    --warmup-steps ${WARMUP_STEPS} \
    --lr ${LR} \
    --log-every-n-steps ${LOG_EVERY_N_STEPS} \
    --metrics-csv /results/metrics.csv \
    ${FAKE_DATA_FLAG}"

# ---------- cleanup on exit ----------

cleanup() {
    echo ""
    echo "Stopping all nodes..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for NODE in "${NODE_ARRAY[@]}"; do
        ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 "${SSH_USER}@${NODE}" \
            "docker ps -q --filter ancestor=${IMAGE} | xargs -r docker stop -t 5" \
            2>/dev/null || true
    done
    echo "Done. Logs: $LOG_DIR/"
}
trap cleanup EXIT

# ---------- launch on each node ----------

PIDS=()

for i in "${!NODE_ARRAY[@]}"; do
    NODE="${NODE_ARRAY[$i]}"
    LOG_FILE="$LOG_DIR/node_${i}.log"

    REMOTE_RESULTS_DIR="${REPO_DIR}/results/mlperf_moe_${RUN_TAG}_${TIMESTAMP}"

    REMOTE_CMD="
        set -e
        cd ${REPO_DIR}
        mkdir -p ${REMOTE_RESULTS_DIR}
        # Make sure the train script is up-to-date on the node (it lives under scripts/launch/mlperf_moe/).
        ls scripts/launch/mlperf_moe/train_moe.py >/dev/null
        docker run --rm \\
            --gpus '\"device=${GPU_INDICES}\"' \\
            --network=host \\
            --ipc=host \\
            --shm-size=16g \\
            --entrypoint torchrun \\
            -v ${REPO_DIR}/scripts/launch/mlperf_moe/train_moe.py:/workspace/train_moe.py:ro \\
            -v ${REMOTE_RESULTS_DIR}:/results \\
            -e HF_HOME=/results/hf_cache \\
            -e HF_HUB_DISABLE_TELEMETRY=1 \\
            ${NCCL_ENV_FLAGS} \\
            ${IMAGE} \\
            ${TORCHRUN_ARGS_PREFIX} \\
            --node-rank=${i} \\
            ${TRAIN_ARGS}
    "

    echo "[node $i] $NODE — launching (rank $i/$NNODES) -> $LOG_FILE"
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${NODE}" "$REMOTE_CMD" >"$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All nodes launched. Tailing node 0 log (Ctrl+C stops all)..."
echo ""
tail -f "$LOG_DIR/node_0.log" &
TAIL_PID=$!

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
    echo "MoE training complete on all nodes."
else
    echo ""
    echo "Training finished with errors. Check logs:"
    for i in "${!NODE_ARRAY[@]}"; do
        echo "  node $i: $LOG_DIR/node_${i}.log"
    done
fi
exit $EXIT_CODE
