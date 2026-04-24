#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_t2i.sh — Launch MLPerf T2I (Flux.1-schnell) training on 2 nodes.
#
# Launches torchrun on both nodes via SSH, each running 2 GPUs. Logs are
# written to logs/mlperf_t2i_<timestamp>/node_{0,1}.log.
#
# Prerequisites:
#   - make setup-mlperf-t2i          (.venv-t2i with diffusers + accelerate)
#   - make prepare-mlperf-t2i-data   (CC12M + COCO + Flux.1-schnell weights)
#   - .env populated with NODES, INTERNAL_IPS, SSH_KEY, SSH_USER
#
# Usage:
#   source .env && \
#   NODES="$NODES" INTERNAL_IPS="$INTERNAL_IPS" GPUS_PER_NODE=2 \
#   SSH_KEY=$SSH_KEY SSH_USER=$SSH_USER make run-mlperf-t2i
# ============================================================================

NODES="${NODES:?Set NODES='<pub0> <pub1>'}"
INTERNAL_IPS="${INTERNAL_IPS:?Set INTERNAL_IPS='<priv0> <priv1>'}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
SSH_KEY="${SSH_KEY:?Set SSH_KEY=path/to/key.pem}"
SSH_USER="${SSH_USER:-azureuser}"
NCCL_TRANSPORT="${NCCL_TRANSPORT:-tcp}"
CONFIG="${CONFIG:-scripts/launch/mlperf_t2i/config/t2i_4gpu.yaml}"

# Parse node lists
read -ra NODE_LIST <<< "$NODES"
read -ra IP_LIST <<< "$INTERNAL_IPS"
NNODES=${#NODE_LIST[@]}
MASTER_IP="${IP_LIST[0]}"
MASTER_PORT=29500

LOG_DIR="logs/mlperf_t2i_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== MLPerf Text-to-Image (Flux.1-schnell) ==="
echo "  Nodes:       ${NODE_LIST[*]}"
echo "  Internal:    ${IP_LIST[*]}"
echo "  GPUs/node:   $GPUS_PER_NODE"
echo "  Master:      $MASTER_IP:$MASTER_PORT"
echo "  NCCL:        $NCCL_TRANSPORT"
echo "  Config:      $CONFIG"
echo "  Log dir:     $LOG_DIR"
echo ""

SSH_OPTS="-o StrictHostKeyChecking=no -i $SSH_KEY"

# Launch on each node
for i in $(seq 0 $((NNODES - 1))); do
    NODE="${NODE_LIST[$i]}"
    NODE_RANK=$i

    echo "Starting node $i ($NODE) ..."

    ssh $SSH_OPTS "${SSH_USER}@${NODE}" bash -s <<REMOTE_EOF &
set -euo pipefail

cd ~/ml-infra-profiler
source .venv-t2i/bin/activate

export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=$([[ "$NCCL_TRANSPORT" == "tcp" ]] && echo 1 || echo 0)

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    scripts/launch/mlperf_t2i/benchmark_runner.py \
    --config $CONFIG

REMOTE_EOF

    # Capture PID for log tailing
    PIDS[$i]=$!
    echo "  Node $i PID: ${PIDS[$i]}"

    # Redirect output to log file
    # (SSH stdout/stderr goes to log)
done

# Wait and capture logs
for i in $(seq 0 $((NNODES - 1))); do
    wait ${PIDS[$i]} 2>&1 | tee "$LOG_DIR/node_${i}.log" || true
done

echo ""
echo "Training complete. Logs: $LOG_DIR/"
echo "Run 'make verify-mlperf-t2i' to check convergence."
