#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_llama2.sh — Run MLPerf Inference v5.0 Llama2-70B Server benchmark.
#
# Runs LoadGen Server performance mode inside Docker using vLLM as the
# inference backend. Single-node, tensor_parallel_size=2 across both A100 80GB.
# Llama2-70B (~140GB fp16) fits across two A100 80GB with TP=2.
#
# Prerequisites:
#   - Docker image built: make build-mlperf-llama2
#   - Data downloaded: make prepare-mlperf-llama2-data
#   - .env populated with HF_TOKEN
#
# Usage:
#   bash scripts/launch/mlperf_llama2/run_mlperf_llama2.sh
# ============================================================================

# Load .env
if [[ -f ".env" ]]; then
    set -a && . ./.env && set +a
fi

IMAGE="${LLAMA2_IMAGE:-ml-netprof/mlperf-llama2:latest}"
CONFIG="scripts/launch/mlperf_llama2/config/llama2_70b_server.yaml"
LOG_DIR="logs/mlperf_llama2_$(date +%Y%m%d_%H%M%S)_server"

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

# Resolve relative paths against repo root; leave absolute paths as-is
[[ "$MODEL_PATH"   != /* ]] && MODEL_PATH="$(pwd)/${MODEL_PATH}"
[[ "$DATASET_PATH" != /* ]] && DATASET_PATH="$(pwd)/${DATASET_PATH}"

TP_SIZE=$(yaml_get "$CONFIG" tensor_parallel_size)
BATCH_SIZE=$(yaml_get "$CONFIG" batch_size)
DTYPE=$(yaml_get "$CONFIG" dtype)
TOTAL_SAMPLES=$(yaml_get "$CONFIG" total_sample_count)
MAX_OUTPUT_TOKENS=$(yaml_get "$CONFIG" max_output_tokens)
MIN_DURATION=$(yaml_get "$CONFIG" min_duration_ms)
MAX_QUERY_COUNT=$(yaml_get "$CONFIG" max_query_count)
TARGET_QPS=$(yaml_get "$CONFIG" target_qps)

mkdir -p "$LOG_DIR"

echo "=== MLPerf Inference v5.0 — Llama2-70B Server ==="
echo "  Image:           $IMAGE"
echo "  Model:           $MODEL_PATH"
echo "  Dataset:         $DATASET_PATH"
echo "  TP size:         $TP_SIZE"
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

echo "Starting performance run..."
docker run --rm \
    --gpus all \
    --network=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v "${MODEL_PATH}:/data/model:ro" \
    -v "${DATASET_PATH}:/data/dataset/open_orca.pkl:ro" \
    -v "$(pwd)/${LOG_DIR}:/output" \
    -v "$(pwd)/scripts/launch/mlperf_llama2/SUT_VLLM_patched.py:/mlperf_inference/language/llama2-70b/SUT_VLLM.py:ro" \
    -v "$(pwd)/scripts/launch/mlperf_llama2/main_patched.py:/mlperf_inference/language/llama2-70b/main.py:ro" \
    "$IMAGE" \
    python main.py \
        --scenario Server \
        --vllm \
        --tensor-parallel-size "${TP_SIZE}" \
        --model-path /data/model \
        --dataset-path /data/dataset/open_orca.pkl \
        --dtype "${DTYPE}" \
        --batch-size "${BATCH_SIZE}" \
        --total-sample-count "${TOTAL_SAMPLES}" \
        --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
        --max-query-count "${MAX_QUERY_COUNT}" \
        --output-log-dir /output \
        --user-conf /output/user.conf \
    2>&1 | tee "${LOG_DIR}/run_performance.log"

echo "Performance run complete. Logs: ${LOG_DIR}/"
