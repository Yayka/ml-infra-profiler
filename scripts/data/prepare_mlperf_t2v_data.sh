#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_mlperf_t2v_data.sh — Download COCO 2014 val annotations and
# Wan2.2-T2V-A14B model weights for the MLPerf T2V benchmark.
#
# Downloads:
#   - COCO 2014 val annotations (~240 MB) — reuses existing download if present
#   - Wan-AI/Wan2.2-T2V-A14B via HuggingFace
#     NOTE: Check if model is gated at https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
#     If gated: add HF_TOKEN to .env and accept the license on HuggingFace.
#
# Usage:
#   bash scripts/data/prepare_mlperf_t2v_data.sh
#   DATA_DIR=/data/mlperf_t2v bash scripts/data/prepare_mlperf_t2v_data.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-/data/mlperf_t2v}"
ANNOTATIONS_DIR="${DATA_DIR}/annotations"
MODEL_DIR="${DATA_DIR}/models/Wan2.2-T2V-A14B-Diffusers"
# Use the Diffusers variant — the native Wan format is NOT compatible with diffusers WanPipeline
MODEL_REPO="${MODEL_REPO:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"

# Reuse COCO annotations from other benchmarks if available
COCO_REUSE_PATHS=(
    "/data/mlperf_sdxl/annotations/captions_val2014.json"
    "/data/mlperf_t2i/annotations/captions_val2014.json"
)
COCO_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

if [[ -f ".env" ]]; then
    set -a && eval "$(sed 's/?=/=/' .env | grep -v '^#' | grep -v '^$')" && set +a
fi
HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN="${HF_TOKEN//\"/}"
HF_TOKEN="${HF_TOKEN//\'/}"

echo "=== MLPerf Wan2.2-T2V data preparation ==="
echo "  Annotations dir : ${ANNOTATIONS_DIR}"
echo "  Model dir       : ${MODEL_DIR}"
echo "  Model repo      : ${MODEL_REPO}"
echo ""

# ---------- Step 1: COCO 2014 val annotations ----------

mkdir -p "${ANNOTATIONS_DIR}"

if [[ -f "${ANNOTATIONS_DIR}/captions_val2014.json" ]]; then
    echo "Step 1/2: COCO annotations already present — skipping."
else
    REUSED=false
    for REUSE_PATH in "${COCO_REUSE_PATHS[@]}"; do
        if [[ -f "${REUSE_PATH}" ]]; then
            echo "Step 1/2: Reusing existing COCO annotations from ${REUSE_PATH}..."
            ln -sf "${REUSE_PATH}" "${ANNOTATIONS_DIR}/captions_val2014.json"
            echo "  Symlink: ${ANNOTATIONS_DIR}/captions_val2014.json"
            REUSED=true
            break
        fi
    done

    if [[ "${REUSED}" == false ]]; then
        echo "Step 1/2: Downloading COCO 2014 val annotations (~240 MB)..."
        TMP_ZIP="$(mktemp /tmp/coco_XXXXXX.zip)"
        curl -fL --progress-bar "${COCO_URL}" -o "${TMP_ZIP}"
        unzip -j "${TMP_ZIP}" "annotations/captions_val2014.json" -d "${ANNOTATIONS_DIR}"
        rm -f "${TMP_ZIP}"
    fi
fi

[[ -f "${ANNOTATIONS_DIR}/captions_val2014.json" ]] || { echo "ERROR: annotations missing" >&2; exit 1; }

# ---------- Step 2: Wan2.2-T2V-A14B model ----------

if [[ -n "$(ls -A ${MODEL_DIR} 2>/dev/null)" ]]; then
    echo "Step 2/2: Model already present — skipping download."
else
    echo "Step 2/2: Downloading ${MODEL_REPO} (~60-80 GB)..."
    mkdir -p "${MODEL_DIR}"

    PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
    PIP="${PIP:-$(command -v pip3 || command -v pip)}"

    if ! "$PYTHON" -c "import huggingface_hub" &>/dev/null; then
        "$PIP" install --quiet huggingface_hub
    fi

    "$PYTHON" - <<PYEOF
from huggingface_hub import snapshot_download
import os

token = "${HF_TOKEN}" or None
print(f"Downloading ${MODEL_REPO} → ${MODEL_DIR}")
print("This may take 30-60 minutes depending on network speed...")

try:
    snapshot_download(
        repo_id="${MODEL_REPO}",
        local_dir="${MODEL_DIR}",
        token=token,
    )
    print("Download complete.")
except Exception as e:
    if "401" in str(e) or "gated" in str(e).lower():
        print()
        print("ERROR: Model is gated — authentication required.")
        print("  1. Visit https://huggingface.co/${MODEL_REPO}")
        print("  2. Accept the license agreement")
        print("  3. Generate a token at https://huggingface.co/settings/tokens")
        print("  4. Add HF_TOKEN=hf_... to your .env and re-run")
        raise SystemExit(1)
    raise
PYEOF

    if [[ -z "$(ls -A ${MODEL_DIR} 2>/dev/null)" ]]; then
        echo "ERROR: Model download incomplete — directory is empty." >&2
        exit 1
    fi
fi

# ---------- Done ----------

echo ""
echo "Data preparation complete."
echo "  Annotations : ${ANNOTATIONS_DIR}/captions_val2014.json"
echo "  Model       : ${MODEL_DIR}/"
echo ""
echo "Next steps:"
echo "  make setup-mlperf-t2v"
echo "  SERVER_IP=<gpu-node> make run-mlperf-t2v"
