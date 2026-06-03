#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_client.sh — Run MLPerf LoadGen Llama3.1-8B benchmark against a remote
#                 vLLM server (Node A).
#
# Runs LoadGen + SUTHTTPClient(Server) in Docker on Node B. No GPUs needed —
# the container only drives HTTP traffic and processes tokenization.
#
# Node B does NOT need the full ~16GB model weights — only the tokenizer
# (~a few MB). Copy tokenizer files from Node A once:
#
#   mkdir -p /data/llama3-tokenizer
#   scp azureuser@<node-a-ip>:/data/mlperf_inference/models/Meta-Llama-3.1-8B-Instruct/tokenizer*.{model,json} \
#       /data/llama3-tokenizer/
#   scp azureuser@<node-a-ip>:/data/mlperf_inference/models/Meta-Llama-3.1-8B-Instruct/special_tokens_map.json \
#       /data/llama3-tokenizer/
#
# Then run:
#   TOKENIZER_PATH=/data/llama3-tokenizer \
#   SERVER_URL=http://<node-a-ip>:8000 make run-mlperf-inference-client
#
# If TOKENIZER_PATH is not set, falls back to MODEL_PATH from config
# (requires the full model to be present on Node B).
# ============================================================================

# Load .env
if [[ -f ".env" ]]; then
    set -a && . ./.env && set +a
fi

if [[ -z "${SERVER_URL:-}" ]]; then
    echo "ERROR: SERVER_URL is required. Set it to the address of the vLLM server." >&2
    echo "  Example: SERVER_URL=http://10.0.0.10:8000 make run-mlperf-inference-client" >&2
    exit 1
fi

IMAGE="${INFERENCE_IMAGE:-ml-netprof/mlperf-inference:latest}"
CONFIG="scripts/launch/mlperf_inference/config/llama3_8b_server.yaml"
LOG_DIR="logs/mlperf_inference_$(date +%Y%m%d_%H%M%S)_client"

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

# TOKENIZER_PATH: directory containing just the tokenizer files (~a few MB).
# Node B does not need the full ~16GB model weights — copy only the tokenizer
# from Node A. If not set, falls back to MODEL_PATH (requires full model).
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
[[ "$TOKENIZER_PATH" != /* ]] && TOKENIZER_PATH="$(pwd)/${TOKENIZER_PATH}"

mkdir -p "$LOG_DIR"

echo "=== MLPerf Inference — Llama3.1-8B LoadGen Client ==="
echo "  Image:           $IMAGE"
echo "  Server URL:      $SERVER_URL"
echo "  Model name:      $MODEL_NAME"
echo "  Tokenizer path:  $TOKENIZER_PATH"
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

echo "  Dataset dir:     $DATASET_DIR"
echo "  Dataset file:    $DATASET_FILE"

if [[ ! -e "${DATASET_PATH}" ]]; then
    echo "ERROR: dataset not found on this host: ${DATASET_PATH}" >&2
    echo "  Make sure the dataset is accessible on Node B at the path in ${CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "ERROR: dataset path exists but is not a regular file: ${DATASET_PATH}" >&2
    echo "  ls output:" >&2
    ls -la "${DATASET_DIR}/" >&2
    exit 1
fi

echo "Starting LoadGen client run..."
docker run --rm \
    --network=host \
    -v "${DATASET_DIR}:/data/dataset:ro" \
    -v "${TOKENIZER_PATH}:/data/tokenizer:ro" \
    -v "$(pwd)/${LOG_DIR}:/output" \
    -v "$(pwd)/scripts/launch/mlperf_llama2/SUT_VLLM_patched.py:/mlperf_inference/language/llama3.1-405b/SUT_VLLM.py:ro" \
    -v "$(pwd)/scripts/launch/mlperf_llama2/main_patched.py:/mlperf_inference/language/llama3.1-405b/main.py:ro" \
    "$IMAGE" \
    python main.py \
        --scenario Server \
        --vllm \
        --api-server "${SERVER_URL}" \
        --api-model-name "${MODEL_NAME}" \
        --model-path /data/tokenizer \
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
