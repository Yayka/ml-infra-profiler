#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_mlperf_sdxl_data.sh — Download COCO 2014 val annotations and
# SDXL-base-1.0 model weights for the MLPerf SDXL T2I benchmark.
#
# Downloads:
#   - COCO 2014 val annotations (~240 MB) → DATA_DIR/annotations/
#     Reuses existing download at /data/mlperf_t2i/annotations/ if present.
#   - stabilityai/stable-diffusion-xl-base-1.0 via HuggingFace (~6.5 GB)
#     SDXL is NOT gated — no HF_TOKEN required.
#
# Usage:
#   bash scripts/data/prepare_mlperf_sdxl_data.sh
#   DATA_DIR=/data/mlperf_sdxl bash scripts/data/prepare_mlperf_sdxl_data.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-/data/mlperf_sdxl}"
ANNOTATIONS_DIR="${DATA_DIR}/annotations"
MODEL_DIR="${DATA_DIR}/models/sdxl-base-1.0"

# COCO 2014 val annotations (same file used by the T2I training benchmark)
COCO_ANNOTATIONS_REUSE_PATH="/data/mlperf_t2i/annotations/captions_val2014.json"
COCO_ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

echo "=== MLPerf SDXL data preparation ==="
echo "  Annotations dir : ${ANNOTATIONS_DIR}"
echo "  Model dir       : ${MODEL_DIR}"
echo ""

# ---------- Step 1: COCO 2014 val annotations ----------

mkdir -p "${ANNOTATIONS_DIR}"

if [[ -f "${ANNOTATIONS_DIR}/captions_val2014.json" ]]; then
    echo "Step 1/2: COCO annotations already present — skipping download."
elif [[ -f "${COCO_ANNOTATIONS_REUSE_PATH}" ]]; then
    echo "Step 1/2: Reusing existing COCO annotations from mlperf_t2i..."
    ln -sf "${COCO_ANNOTATIONS_REUSE_PATH}" "${ANNOTATIONS_DIR}/captions_val2014.json"
    echo "  Symlink created: ${ANNOTATIONS_DIR}/captions_val2014.json"
else
    echo "Step 1/2: Downloading COCO 2014 val annotations (~240 MB)..."
    TMP_ZIP="$(mktemp /tmp/coco_annotations_XXXXXX.zip)"
    curl -fL --progress-bar "${COCO_ANNOTATIONS_URL}" -o "${TMP_ZIP}"
    unzip -j "${TMP_ZIP}" "annotations/captions_val2014.json" -d "${ANNOTATIONS_DIR}"
    rm -f "${TMP_ZIP}"
    echo "  OK: ${ANNOTATIONS_DIR}/captions_val2014.json"
fi

if [[ ! -f "${ANNOTATIONS_DIR}/captions_val2014.json" ]]; then
    echo "ERROR: captions_val2014.json not found after download." >&2
    exit 1
fi

# ---------- Step 2: SDXL-base-1.0 model weights ----------

if [[ -f "${MODEL_DIR}/model_index.json" ]]; then
    echo "Step 2/2: SDXL model already present — skipping download."
else
    echo "Step 2/2: Downloading SDXL-base-1.0 model (~6.5 GB, no auth required)..."
    mkdir -p "${MODEL_DIR}"

    PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
    PIP="${PIP:-$(command -v pip3 || command -v pip)}"

    if ! "$PYTHON" -c "import huggingface_hub" &>/dev/null; then
        "$PIP" install --quiet huggingface_hub
    fi

    "$PYTHON" - <<PYEOF
from huggingface_hub import snapshot_download
print("Downloading stabilityai/stable-diffusion-xl-base-1.0 ...")
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="${MODEL_DIR}",
    ignore_patterns=["*.ckpt"],  # skip large ckpt, use safetensors only
)
print("Done.")
PYEOF

    if [[ ! -f "${MODEL_DIR}/model_index.json" ]]; then
        echo "ERROR: Model download incomplete — model_index.json not found in ${MODEL_DIR}." >&2
        exit 1
    fi
    echo "  OK: ${MODEL_DIR}/model_index.json"
fi

# ---------- Done ----------

echo ""
echo "Data preparation complete."
echo "  Annotations : ${ANNOTATIONS_DIR}/captions_val2014.json"
echo "  Model       : ${MODEL_DIR}/"
echo ""
echo "Next steps:"
echo "  make setup-mlperf-sdxl      # install Python deps"
echo "  SERVER_IP=<gpu-node> make run-mlperf-sdxl"
