#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_server.sh — Start vLLM OpenAI API server for MLPerf Llama2-70B.
#
# Runs vLLM's built-in OpenAI-compatible HTTP server inside the existing
# Docker image. Blocks until killed (Ctrl-C). Node B can then drive
# LoadGen against this server via: SERVER_URL=http://<this-node-ip>:8000
#   make run-mlperf-llama2-client
#
# Prerequisites:
#   - Docker image built: make build-mlperf-llama2
#   - Model downloaded: make prepare-mlperf-llama2-data
#
# Usage:
#   bash scripts/launch/mlperf_llama2/run_server.sh
# ============================================================================

# Load .env
if [[ -f ".env" ]]; then
    set -a && . ./.env && set +a
fi

IMAGE="${LLAMA2_IMAGE:-ml-netprof/mlperf-llama2:latest}"
CONFIG="scripts/launch/mlperf_llama2/config/llama2_70b_server.yaml"
SERVER_PORT="${SERVER_PORT:-8000}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# Parse a YAML key from a config file (simple key: value only, no nesting)
yaml_get() {
    local file="$1" key="$2"
    grep -E "^${key}:" "$file" | head -1 | awk '{print $2}'
}

MODEL_PATH=$(yaml_get "$CONFIG" model_path)
TP_SIZE=$(yaml_get "$CONFIG" tensor_parallel_size)
DTYPE=$(yaml_get "$CONFIG" dtype)

# Resolve relative paths against repo root; leave absolute paths as-is
[[ "$MODEL_PATH" != /* ]] && MODEL_PATH="$(pwd)/${MODEL_PATH}"

MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

echo "=== MLPerf Llama2-70B — vLLM API Server ==="
echo "  Image:   $IMAGE"
echo "  Model:   $MODEL_PATH"
echo "  Name:    $MODEL_NAME"
echo "  TP size: $TP_SIZE"
echo "  Dtype:   $DTYPE"
echo "  Port:    $SERVER_PORT"
echo ""
echo "Server will be available at http://0.0.0.0:${SERVER_PORT}"
echo "Press Ctrl-C to stop."
echo ""

docker run --rm \
    --gpus all \
    --network=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v "${MODEL_PATH}:/data/model:ro" \
    "$IMAGE" \
    python -m vllm.entrypoints.openai.api_server \
        --model /data/model \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --dtype "${DTYPE}" \
        --port "${SERVER_PORT}" \
        --host 0.0.0.0
