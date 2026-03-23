#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# nemo_entrypoint.sh — Runs inside the NeMo container on each training node.
#
# Environment variables injected by run_mlperf_multinode.sh via docker run:
#   MASTER_ADDR        — private IP of rank-0 node
#   MASTER_PORT        — rendezvous port (default 29500)
#   NNODES             — total number of nodes
#   GPUS_PER_NODE      — GPUs on this node
#   NODE_RANK          — this node's rank (0-based)
#   WANDB_API_KEY      — from .env (bind-mounted)
#   WANDB_BASE_URL     — from .env
#   WANDB_ENTITY       — from .env
# ============================================================================

# Load .env if present (bind-mounted at /workspace/.env)
if [[ -f /workspace/.env ]]; then
    set -a
    # shellcheck disable=SC1091
    source /workspace/.env
    set +a
fi

echo "=== NeMo entrypoint ==="
echo "  Node rank:    ${NODE_RANK}"
echo "  Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Nodes:        ${NNODES}"
echo "  GPUs/node:    ${GPUS_PER_NODE}"
echo "  W&B base URL: ${WANDB_BASE_URL:-<not set>}"
echo ""

# Verify data files exist before launching
TRAIN_BIN=/data/c4/llama3_1_8b_preprocessed_c4_dataset/c4-train.en_6_text_document.bin
VAL_BIN=/data/c4/llama3_1_8b_preprocessed_c4_dataset/c4-validation-91205-samples.en_text_document.bin

if [[ ! -f "${TRAIN_BIN}" ]]; then
    echo "ERROR: ${TRAIN_BIN} not found." >&2
    echo "  Run 'make prepare-mlperf-data' on this node first." >&2
    exit 1
fi

if [[ ! -f "${VAL_BIN}" ]]; then
    echo "ERROR: ${VAL_BIN} not found." >&2
    exit 1
fi

echo "Data files verified."
echo ""

# Launch via torchrun for multi-node distributed training.
# --network=host on the docker run ensures NCCL can reach the master node directly.
exec torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path /workspace \
    --config-name config.yaml \
    trainer.num_nodes="${NNODES}" \
    trainer.devices="${GPUS_PER_NODE}"
