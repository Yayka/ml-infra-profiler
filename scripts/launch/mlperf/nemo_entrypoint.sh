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
if [[ ! -f /data/c4/c4_train_text_document.bin ]]; then
    echo "ERROR: /data/c4/c4_train_text_document.bin not found." >&2
    echo "  Run 'make prepare-mlperf-data' on this node first." >&2
    echo "  See scripts/data/prepare_c4_mlperf.sh for details." >&2
    exit 1
fi

if [[ ! -f /data/c4/c4_val_text_document.bin ]]; then
    echo "ERROR: /data/c4/c4_val_text_document.bin not found." >&2
    exit 1
fi

echo "Data files verified."
echo ""

# NeMo uses torchrun internally via the megatron_gpt_pretraining.py launcher.
# Pass distributed args via the trainer overrides.
exec python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path /workspace \
    --config-name config.yaml \
    trainer.num_nodes="${NNODES}" \
    trainer.devices="${GPUS_PER_NODE}" \
    +trainer.strategy.find_unused_parameters=false \
    cluster_type=bcm \
    +cluster.node_rank="${NODE_RANK}" \
    +cluster.master_address="${MASTER_ADDR}" \
    +cluster.master_port="${MASTER_PORT}"
