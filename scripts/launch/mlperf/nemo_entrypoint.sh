#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# nemo_entrypoint.sh — Runs inside the NeMo container on each training node.
#
# Uses NeMo's built-in megatron_llama_config as the Hydra base config and
# passes all customisations as command-line overrides. This avoids OmegaConf
# struct errors that occur when providing a standalone config that is missing
# required fields.
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

# Use NeMo's built-in megatron_llama_config as the Hydra base, then override
# with Llama3.1 8B architecture values and our 4-GPU scaling parameters.
exec torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-name megatron_llama_config \
    \
    trainer.devices="${GPUS_PER_NODE}" \
    trainer.num_nodes="${NNODES}" \
    trainer.precision=bf16 \
    trainer.max_steps=400 \
    trainer.val_check_interval=100 \
    trainer.log_every_n_steps=10 \
    trainer.gradient_clip_val=1.0 \
    trainer.enable_checkpointing=false \
    +trainer.num_sanity_val_steps=0 \
    \
    model.micro_batch_size=1 \
    model.global_batch_size=2048 \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=1 \
    \
    model.encoder_seq_length=8192 \
    model.max_position_embeddings=8192 \
    model.num_layers=32 \
    model.hidden_size=4096 \
    model.ffn_hidden_size=14336 \
    model.num_attention_heads=32 \
    model.num_query_groups=8 \
    model.position_embedding_type=rope \
    model.rotary_percentage=1.0 \
    model.normalization=rmsnorm \
    model.bias=false \
    model.activation=fast-swiglu \
    model.share_embeddings_and_output_weights=false \
    model.hidden_dropout=0.0 \
    model.attention_dropout=0.0 \
    \
    model.activations_checkpoint_method=block \
    model.activations_checkpoint_num_layers=1 \
    \
    model.tokenizer.library=huggingface \
    model.tokenizer.type=/data/c4/llama3_1_8b_tokenizer \
    model.tokenizer.model=/data/c4/llama3_1_8b_tokenizer/original/tokenizer.model \
    \
    model.data.data_impl=mmap \
    model.data.seq_length=8192 \
    model.data.skip_warmup=true \
    model.data.num_workers=2 \
    model.data.dataloader_type=single \
    '+model.data.data_prefix={train:[1.0,/data/c4/llama3_1_8b_preprocessed_c4_dataset/c4-train.en_6_text_document],validation:[/data/c4/llama3_1_8b_preprocessed_c4_dataset/c4-validation-91205-samples.en_text_document],test:[/data/c4/llama3_1_8b_preprocessed_c4_dataset/c4-validation-91205-samples.en_text_document]}' \
    \
    model.optim.name=distributed_fused_adam \
    model.optim.lr=3.0e-4 \
    model.optim.weight_decay=0.1 \
    'model.optim.betas=[0.9,0.95]' \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=50 \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=3.0e-5 \
    \
    exp_manager.exp_dir=/results/mlperf_llama3_8b \
    exp_manager.name=mlperf-llama3-8b-4gpu \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.project=ml-netprof \
    exp_manager.wandb_logger_kwargs.name=mlperf-llama3-8b-4gpu \
    exp_manager.resume_if_exists=false
