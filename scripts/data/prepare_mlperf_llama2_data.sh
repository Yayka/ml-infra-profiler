#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_mlperf_llama2_data.sh — Download OpenOrca dataset and
# Llama-2-70b-chat-hf model for MLPerf Inference v5.0 Llama2-70B.
#
# Downloads:
#   - OpenOrca eval dataset (~4 GB) from MLCommons R2 storage
#     → data/mlperf_llama2/dataset/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
#   - Llama-2-70b-chat-hf model via HuggingFace (~140 GB, requires HF_TOKEN)
#
# Usage:
#   bash scripts/data/prepare_mlperf_llama2_data.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-data/mlperf_llama2}"
DATASET_DIR="${DATA_DIR}/dataset"
MODEL_DIR="${DATA_DIR}/models/Llama-2-70b-chat-hf"

# Load .env (strip Makefile ?= syntax if present)
if [[ -f ".env" ]]; then
    set -a && eval "$(sed 's/?=/=/' .env | grep -v '^#' | grep -v '^$')" && set +a
fi

# Strip quotes from HF_TOKEN if present
HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN="${HF_TOKEN//\"/}"
HF_TOKEN="${HF_TOKEN//\'/}"

echo "=== MLPerf Inference v5.0 — Llama2-70B data preparation ==="
echo "  Dataset dir:   ${DATASET_DIR}"
echo "  Model dir:     ${MODEL_DIR}"
echo ""

# ---------- Step 1: OpenOrca dataset (preprocessed pickle) ----------

DATASET_PKL="open_orca_gpt4_tokenized_llama.sampled_24576.pkl"

mkdir -p "${DATASET_DIR}"

echo "Step 1/2: Downloading preprocessed OpenOrca dataset (~4 GB)..."

pushd "${DATASET_DIR}" > /dev/null
bash <(curl -fsSL https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://raw.githubusercontent.com/mlcommons/r2-infra/main/inference/metadata/llama2-70b-dataset-24576.uri
popd > /dev/null

if [[ ! -f "${DATASET_DIR}/${DATASET_PKL}" ]]; then
    echo "ERROR: ${DATASET_PKL} not found after download in ${DATASET_DIR}." >&2
    echo "  The dataset file may have a slightly different name. Check ${DATASET_DIR}/." >&2
    ls "${DATASET_DIR}/" || true
    exit 1
fi

echo "  OK: ${DATASET_DIR}/${DATASET_PKL}"

# ---------- Step 2: Model ----------

echo ""
echo "Step 2/2: Downloading Llama-2-70b-chat-hf model (~140 GB)..."

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN is required for HuggingFace download." >&2
    echo "  1. Accept the license at https://huggingface.co/meta-llama/Llama-2-70b-chat-hf" >&2
    echo "  2. Generate a token at https://huggingface.co/settings/tokens" >&2
    echo "  3. Add HF_TOKEN=hf_... to your .env and re-run." >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
PIP="${PIP:-$(command -v pip3 || command -v pip)}"

if ! "$PYTHON" -c "import huggingface_hub" &>/dev/null; then
    "$PIP" install --quiet huggingface_hub
fi

"$PYTHON" - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="meta-llama/Llama-2-70b-chat-hf",
    local_dir="${MODEL_DIR}",
    token="${HF_TOKEN}",
)
PYEOF

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: Model download incomplete — config.json not found in ${MODEL_DIR}." >&2
    exit 1
fi

echo "  OK: ${MODEL_DIR}/config.json"

# ---------- Done ----------

echo ""
echo "Data preparation complete."
echo "  Dataset: ${DATASET_DIR}/${DATASET_PKL}"
echo "  Model:   ${MODEL_DIR}/"
echo ""
echo "Next steps:"
echo "  make build-mlperf-llama2"
echo "  make run-mlperf-llama2"
