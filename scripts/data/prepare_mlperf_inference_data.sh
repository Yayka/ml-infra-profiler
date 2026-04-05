#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_mlperf_inference_data.sh — Download CNN/DM dataset and
# Llama3.1-8B-Instruct model for MLPerf Inference v5.0.
#
# Downloads:
#   - CNN/DailyMail eval dataset (~500 MB) from MLCommons R2 storage
#     → data/mlperf_inference/dataset/cnn_eval.json (13,368 samples)
#   - Meta-Llama-3.1-8B-Instruct model via HuggingFace (~15 GB, gated)
#     → data/mlperf_inference/models/Meta-Llama-3.1-8B-Instruct/
#
# Prerequisites:
#   - HF_TOKEN set in .env (model is gated — accept license at HuggingFace first)
#   - huggingface_hub installed: pip install huggingface_hub
#
# Usage:
#   bash scripts/data/prepare_mlperf_inference_data.sh
#
# Override data directory:
#   DATA_DIR=/mnt/data bash scripts/data/prepare_mlperf_inference_data.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-data/mlperf_inference}"
DATASET_DIR="${DATA_DIR}/dataset"
MODEL_DIR="${DATA_DIR}/models/Meta-Llama-3.1-8B-Instruct"

# Load .env for HF_TOKEN
if [[ -f ".env" ]]; then
    set -a && . ./.env && set +a
fi

HF_TOKEN="${HF_TOKEN:-}"

echo "=== MLPerf Inference v5.0 — data preparation ==="
echo "  Dataset dir: ${DATASET_DIR}"
echo "  Model dir:   ${MODEL_DIR}"
echo ""

# ---------- Step 1: CNN/DM dataset ----------

mkdir -p "${DATASET_DIR}"
cd "${DATASET_DIR}"

echo "Step 1/2: Downloading CNN/DailyMail eval dataset (~500 MB)..."
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri

cd - > /dev/null

if [[ ! -f "${DATASET_DIR}/cnn_eval.json" ]]; then
    echo "ERROR: cnn_eval.json not found after download in ${DATASET_DIR}." >&2
    echo "  Check the download output above for errors." >&2
    exit 1
fi

SAMPLE_COUNT=$(python3 -c "import json; d=json.load(open('${DATASET_DIR}/cnn_eval.json')); print(len(d))")
echo "  OK: ${DATASET_DIR}/cnn_eval.json  (${SAMPLE_COUNT} samples)"

# ---------- Step 2: Llama 3.1 8B Instruct model ----------

echo ""
echo "Step 2/2: Downloading Meta-Llama-3.1-8B-Instruct (~15 GB)..."

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN is required — the model is gated." >&2
    echo "  1. Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct" >&2
    echo "  2. Generate a token at https://huggingface.co/settings/tokens" >&2
    echo "  3. Add HF_TOKEN=hf_... to your .env and re-run." >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}"

huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --local-dir "${MODEL_DIR}" \
    --token "${HF_TOKEN}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: Model download incomplete — config.json not found in ${MODEL_DIR}." >&2
    exit 1
fi

echo "  OK: ${MODEL_DIR}/config.json"

# ---------- Done ----------

echo ""
echo "Data preparation complete."
echo "  Dataset: ${DATASET_DIR}/cnn_eval.json"
echo "  Model:   ${MODEL_DIR}/"
echo ""
echo "Next steps:"
echo "  make build-mlperf-inference"
echo "  make run-mlperf-inference"
