#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_inference.sh — Run MLPerf Inference v5.0 Llama3.1-8B benchmark.
#
# Runs LoadGen performance + accuracy modes inside Docker using vLLM as the
# inference backend. Supports Offline (throughput) and Server (latency) scenarios.
#
# Single-node, tensor_parallel_size=2 across both GPUs on the node.
# Llama3.1-8B fits on one A100 80GB; no multi-node needed for inference.
#
# Prerequisites:
#   - Docker image built: make build-mlperf-inference
#   - Data downloaded: make prepare-mlperf-inference-data
#   - .env populated with HF_TOKEN
#
# Usage:
#   bash scripts/launch/mlperf_inference/run_mlperf_inference.sh
#
# Run a single scenario:
#   SCENARIO=offline bash scripts/launch/mlperf_inference/run_mlperf_inference.sh
#   SCENARIO=server  bash scripts/launch/mlperf_inference/run_mlperf_inference.sh
# ============================================================================

# Load .env
if [[ -f ".env" ]]; then
    set -a && . ./.env && set +a
fi

SCENARIO="${SCENARIO:-all}"
IMAGE="${INFERENCE_IMAGE:-ml-netprof/mlperf-inference:latest}"
LOG_BASE="logs/mlperf_inference_$(date +%Y%m%d_%H%M%S)"

# Parse a YAML key from a config file (simple key: value only, no nesting)
yaml_get() {
    local file="$1" key="$2"
    grep -E "^${key}:" "$file" | head -1 | awk '{print $2}'
}

run_scenario() {
    local scenario="$1"
    local CONFIG="scripts/launch/mlperf_inference/config/llama3_8b_${scenario}.yaml"

    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: config not found: $CONFIG" >&2
        exit 1
    fi

    local MODEL_PATH
    local DATASET_PATH
    local TP_SIZE
    local BATCH_SIZE
    local DTYPE
    local TOTAL_SAMPLES
    local MAX_OUTPUT_TOKENS
    local MIN_DURATION
    local MAX_QUERY_COUNT

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

    local LOG_DIR="${LOG_BASE}_${scenario}"
    mkdir -p "$LOG_DIR"

    echo "=== MLPerf Inference v5.0 — Llama3.1-8B ${scenario^} ==="
    echo "  Image:           $IMAGE"
    echo "  Model:           $MODEL_PATH"
    echo "  Dataset:         $DATASET_PATH"
    echo "  TP size:         $TP_SIZE"
    echo "  Dtype:           $DTYPE"
    echo "  Total samples:   $TOTAL_SAMPLES"
    echo "  Min duration ms: $MIN_DURATION"
    echo "  Log dir:         $LOG_DIR/"
    echo ""

    # Scenario flag for LoadGen
    local SCENARIO_FLAG
    case "$scenario" in
        offline) SCENARIO_FLAG="Offline" ;;
        server)  SCENARIO_FLAG="Server" ;;
        *) echo "ERROR: unknown scenario: $scenario" >&2; exit 1 ;;
    esac

    # Server-specific: write user.conf with target_qps
    local USER_CONF_ARGS=""
    if [[ "$scenario" == "server" ]]; then
        local TARGET_QPS
        TARGET_QPS=$(yaml_get "$CONFIG" target_qps)
        cat > "${LOG_DIR}/user.conf" <<EOF
*.Server.target_qps = ${TARGET_QPS}
*.Server.target_duration = $(yaml_get "$CONFIG" min_duration_ms)
*.Server.min_duration = $(yaml_get "$CONFIG" min_duration_ms)
*.Server.min_query_count = 100
*.Server.max_query_count = 200
*.Offline.min_duration = 600000
*.Offline.min_query_count = 2000
EOF
        USER_CONF_ARGS="--user-conf /output/user.conf"
    fi

    # ---------- Performance run ----------

    echo "[${scenario}] Starting performance run..."
    docker run --rm \
        --gpus all \
        --network=host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -v "${MODEL_PATH}:/data/model:ro" \
        -v "${DATASET_PATH}:/data/dataset/cnn_eval.json:ro" \
        -v "$(pwd)/${LOG_DIR}:/output" \
        -v "$(pwd)/scripts/launch/mlperf_inference/SUT_VLLM_patched.py:/mlperf_inference/language/llama3.1-405b/SUT_VLLM.py:ro" \
        -v "$(pwd)/scripts/launch/mlperf_inference/main_patched.py:/mlperf_inference/language/llama3.1-405b/main.py:ro" \
        "$IMAGE" \
        python main.py \
            --scenario "${SCENARIO_FLAG}" \
            --vllm \
            --tensor-parallel-size "${TP_SIZE}" \
            --model-path /data/model \
            --dataset-path /data/dataset/cnn_eval.json \
            --dtype "${DTYPE}" \
            --batch-size "${BATCH_SIZE}" \
            --total-sample-count "${TOTAL_SAMPLES}" \
            --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
            --max-query-count "${MAX_QUERY_COUNT:-0}" \
            --output-log-dir /output \
            ${USER_CONF_ARGS} \
        2>&1 | tee "${LOG_DIR}/run_performance.log"

    echo "[${scenario}] Performance run complete."

    # ---------- Accuracy run (skipped if PERF_ONLY=1) ----------

    if [[ "${PERF_ONLY:-0}" == "1" ]]; then
        echo "[${scenario}] Skipping accuracy run (PERF_ONLY=1)."
        return
    fi

    echo "[${scenario}] Starting accuracy run..."
    docker run --rm \
        --gpus all \
        --network=host \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -v "${MODEL_PATH}:/data/model:ro" \
        -v "${DATASET_PATH}:/data/dataset/cnn_eval.json:ro" \
        -v "$(pwd)/${LOG_DIR}:/output" \
        -v "$(pwd)/scripts/launch/mlperf_inference/SUT_VLLM_patched.py:/mlperf_inference/language/llama3.1-405b/SUT_VLLM.py:ro" \
        -v "$(pwd)/scripts/launch/mlperf_inference/main_patched.py:/mlperf_inference/language/llama3.1-405b/main.py:ro" \
        "$IMAGE" \
        bash -c "python main.py \
            --scenario ${SCENARIO_FLAG} \
            --vllm \
            --accuracy \
            --tensor-parallel-size ${TP_SIZE} \
            --model-path /data/model \
            --dataset-path /data/dataset/cnn_eval.json \
            --dtype ${DTYPE} \
            --batch-size ${BATCH_SIZE} \
            --total-sample-count ${TOTAL_SAMPLES} \
            --max-output-tokens ${MAX_OUTPUT_TOKENS} \
            --output-log-dir /output \
            ${USER_CONF_ARGS} \
            && python evaluate-accuracy.py \
            --checkpoint-path /data/model \
            --mlperf-accuracy-file /output/mlperf_log_accuracy.json \
            --dataset-file /data/dataset/cnn_eval.json \
            | tee /output/accuracy_result.txt" \
        2>&1 | tee "${LOG_DIR}/run_accuracy.log"

    echo "[${scenario}] Accuracy run complete."

    # ---------- Generate submission tree ----------

    echo "[${scenario}] Building submission tree..."
    python3 scripts/launch/mlperf_inference/generate_submission.py \
        --log-dir "${LOG_DIR}" \
        --scenario "${scenario}" \
        --output-dir results/mlperf_inference_v5.0

    echo "[${scenario}] Done. Logs: ${LOG_DIR}/"
    echo ""
}

# ---------- Main ----------

case "$SCENARIO" in
    all)
        run_scenario offline
        run_scenario server
        echo "Both scenarios complete. Run 'make verify-mlperf-inference' to check ROUGE targets."
        ;;
    offline|server)
        run_scenario "$SCENARIO"
        echo "Run 'make verify-mlperf-inference' to check ROUGE targets."
        ;;
    *)
        echo "ERROR: SCENARIO must be 'offline', 'server', or 'all'. Got: $SCENARIO" >&2
        exit 1
        ;;
esac
