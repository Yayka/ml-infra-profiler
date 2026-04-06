#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_mlperf_inference_data.sh — Download CNN/DM dataset and
# Llama3.1-8B-Instruct model for MLPerf Inference v5.0.
#
# Downloads:
#   - CNN/DailyMail eval dataset (~500 MB) from MLCommons R2 storage
#     → data/mlperf_inference/dataset/cnn_eval.json (13,368 samples)
#   - Meta-Llama-3.1-8B-Instruct model via MLCommons R2 (default, no auth)
#     or via HuggingFace (set MODEL_DOWNLOAD_PATH=hf, requires HF_TOKEN)
#
# Model download paths:
#   Path A (default — no auth): MLCommons R2 via mlcr automation
#     mlcr get,ml-model,llama3,_mlc,_8b,_r2-downloader
#   Path B (HuggingFace — gated): set MODEL_DOWNLOAD_PATH=hf
#     requires HF_TOKEN in .env and license accepted at huggingface.co
#
# Usage:
#   bash scripts/data/prepare_mlperf_inference_data.sh
#
#   MODEL_DOWNLOAD_PATH=hf bash scripts/data/prepare_mlperf_inference_data.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-data/mlperf_inference}"
DATASET_DIR="${DATA_DIR}/dataset"
MODEL_DIR="${DATA_DIR}/models/Meta-Llama-3.1-8B-Instruct"
MODEL_DOWNLOAD_PATH="${MODEL_DOWNLOAD_PATH:-mlc}"

# Load .env (strip Makefile ?= syntax if present)
if [[ -f ".env" ]]; then
    # Replace `KEY?=VALUE` → `KEY=VALUE` before sourcing
    set -a && eval "$(sed 's/?=/=/' .env | grep -v '^#' | grep -v '^$')" && set +a
fi

# Strip quotes from HF_TOKEN if present
HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN="${HF_TOKEN//\"/}"
HF_TOKEN="${HF_TOKEN//\'/}"

echo "=== MLPerf Inference v5.0 — data preparation ==="
echo "  Dataset dir:   ${DATASET_DIR}"
echo "  Model dir:     ${MODEL_DIR}"
echo "  Model path:    ${MODEL_DOWNLOAD_PATH}"
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

# ---------- Step 2: Model ----------

echo ""
echo "Step 2/2: Downloading Meta-Llama-3.1-8B-Instruct model..."

mkdir -p "${MODEL_DIR}"

if [[ "${MODEL_DOWNLOAD_PATH}" == "mlc" ]]; then
    # ---------- Path A: MLCommons R2 via mlcr (no auth required) ----------
    echo "  Path A: MLCommons R2 downloader (no HuggingFace account needed)"

    if ! command -v mlcr &>/dev/null; then
        echo "  Installing mlcflow (provides mlcr)..."
        pip install --quiet mlcflow
    fi

    mlcr get,ml-model,llama3,_mlc,_8b,_r2-downloader \
        --outdirname="${MODEL_DIR}" \
        -j

    if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
        echo "ERROR: Model download incomplete — config.json not found in ${MODEL_DIR}." >&2
        echo "  Try: MODEL_DOWNLOAD_PATH=hf bash scripts/data/prepare_mlperf_inference_data.sh" >&2
        exit 1
    fi

elif [[ "${MODEL_DOWNLOAD_PATH}" == "hf" ]]; then
    # ---------- Path B: HuggingFace (gated — requires HF_TOKEN) ----------
    echo "  Path B: HuggingFace (gated model)"

    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: HF_TOKEN is required for HuggingFace download." >&2
        echo "  1. Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct" >&2
        echo "  2. Generate a token at https://huggingface.co/settings/tokens" >&2
        echo "  3. Add HF_TOKEN=hf_... to your .env (no ?= prefix) and re-run." >&2
        exit 1
    fi

    if ! command -v huggingface-cli &>/dev/null; then
        pip install --quiet huggingface_hub
    fi

    huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
        --local-dir "${MODEL_DIR}" \
        --token "${HF_TOKEN}"

    if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
        echo "ERROR: Model download incomplete — config.json not found in ${MODEL_DIR}." >&2
        exit 1
    fi

else
    echo "ERROR: MODEL_DOWNLOAD_PATH must be 'mlc' or 'hf', got '${MODEL_DOWNLOAD_PATH}'" >&2
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
