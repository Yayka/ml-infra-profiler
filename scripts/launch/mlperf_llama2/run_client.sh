#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_client.sh — Run MLPerf LoadGen Llama2-70B benchmark against a remote
#                 vLLM server (Node A).
#
# Runs LoadGen + SUTHTTPClient(Server) in Docker on Node B. No GPUs needed —
# the container only drives HTTP traffic and processes tokenization.
#
# Prerequisites:
#   - Docker image built: make build-mlperf-llama2
#   - Data downloaded: make prepare-mlperf-llama2-data
#   - Node A serving: make run-mlperf-llama2
#   - SERVER_URL env var set to Node A's URL
#
# Usage:
#   SERVER_URL=http://<node-a-ip>:8000 make run-mlperf-llama2-client
#   # or directly:
#   SERVER_URL=http://10.0.0.10:8000 bash scripts/launch/mlperf_llama2/run_client.sh
# ============================================================================

# Load .env
if [[ -f ".env" ]]; then
    set -a && . ./.env && set +a
fi

if [[ -z "${SERVER_URL:-}" ]]; then
    echo "ERROR: SERVER_URL is required. Set it to the address of the vLLM server." >&2
    echo "  Example: SERVER_URL=http://10.0.0.10:8000 make run-mlperf-llama2-client" >&2
    exit 1
fi

IMAGE="${LLAMA2_IMAGE:-ml-netprof/mlperf-llama2:latest}"
CONFIG="scripts/launch/mlperf_llama2/config/llama2_70b_server.yaml"
LOG_DIR="logs/mlperf_llama2_$(date +%Y%m%d_%H%M%S)_client"

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
DATASET_PATH=$(yaml_get "$CONFIG" dataset_path)
BATCH_SIZE=$(yaml_get "$CONFIG" batch_size)
DTYPE=$(yaml_get "$CONFIG" dtype)
TOTAL_SAMPLES=$(yaml_get "$CONFIG" total_sample_count)
MAX_OUTPUT_TOKENS=$(yaml_get "$CONFIG" max_output_tokens)
MIN_DURATION=$(yaml_get "$CONFIG" min_duration_ms)
MAX_QUERY_COUNT=$(yaml_get "$CONFIG" max_query_count)
TARGET_QPS=$(yaml_get "$CONFIG" target_qps)

# Resolve relative paths against repo root; leave absolute paths as-is
[[ "$MODEL_PATH"   != /* ]] && MODEL_PATH="$(pwd)/${MODEL_PATH}"
[[ "$DATASET_PATH" != /* ]] && DATASET_PATH="$(pwd)/${DATASET_PATH}"

MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

mkdir -p "$LOG_DIR"

echo "=== MLPerf Llama2-70B — LoadGen Client ==="
echo "  Image:           $IMAGE"
echo "  Server URL:      $SERVER_URL"
echo "  Model name:      $MODEL_NAME"
echo "  Model path:      $MODEL_PATH"
echo "  Dataset:         $DATASET_PATH"
echo "  Dtype:           $DTYPE"
echo "  Total samples:   $TOTAL_SAMPLES"
echo "  Target QPS:      $TARGET_QPS"
echo "  Min duration ms: $MIN_DURATION"
echo "  Log dir:         $LOG_DIR/"
echo ""

# Write user.conf with target_qps
cat > "${LOG_DIR}/user.conf" <<EOF
*.Server.target_qps = ${TARGET_QPS}
*.Server.target_duration = ${MIN_DURATION}
*.Server.min_duration = ${MIN_DURATION}
*.Server.min_query_count = 100
*.Server.max_query_count = 200
EOF

# Mount the directory containing the dataset file to avoid Docker creating the
# bind-mount target as a directory (classic Docker quirk with file bind mounts).
DATASET_DIR="$(dirname "${DATASET_PATH}")"
DATASET_FILE="$(basename "${DATASET_PATH}")"

echo "Starting LoadGen client run..."
docker run --rm \
    --network=host \
    -v "${DATASET_DIR}:/data/dataset:ro" \
    -v "${MODEL_PATH}:/data/model:ro" \
    -v "$(pwd)/${LOG_DIR}:/output" \
    -v "$(pwd)/scripts/launch/mlperf_llama2/SUT_VLLM_patched.py:/mlperf_inference/language/llama2-70b/SUT_VLLM.py:ro" \
    -v "$(pwd)/scripts/launch/mlperf_llama2/main_patched.py:/mlperf_inference/language/llama2-70b/main.py:ro" \
    "$IMAGE" \
    python main.py \
        --scenario Server \
        --vllm \
        --api-server "${SERVER_URL}" \
        --api-model-name "${MODEL_NAME}" \
        --model-path /data/model \
        --dataset-path "/data/dataset/${DATASET_FILE}" \
        --dtype "${DTYPE}" \
        --batch-size "${BATCH_SIZE}" \
        --total-sample-count "${TOTAL_SAMPLES}" \
        --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
        --max-query-count "${MAX_QUERY_COUNT}" \
        --output-log-dir /output \
        --user-conf /output/user.conf \
    2>&1 | tee "${LOG_DIR}/run_client.log"

echo "Client run complete. Logs: ${LOG_DIR}/"
