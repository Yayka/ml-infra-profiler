#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_mlperf_tiny_data.sh — Download data for MLPerf Tiny IC benchmark.
#
# Downloads:
#   1. ResNet v1 int8 TFLite model (~1 MB) from MLCommons tiny repo
#      → data/mlperf_tiny/models/resnet_v1_int8.tflite
#   2. CIFAR-10 test set (~170 MB) from Toronto mirror
#      → data/mlperf_tiny/ic/cifar10_test.npz  (10,000 images + labels)
#
# Environment variables:
#   DATA_DIR  — destination root (default: data/mlperf_tiny)
#
# Usage:
#   bash scripts/data/prepare_mlperf_tiny_data.sh
#   DATA_DIR=/data/mlperf_tiny bash scripts/data/prepare_mlperf_tiny_data.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-data/mlperf_tiny}"
MODEL_DIR="${DATA_DIR}/models"
IC_DIR="${DATA_DIR}/ic"

mkdir -p "$MODEL_DIR" "$IC_DIR"

echo "=== MLPerf Tiny data preparation ==="
echo "  DATA_DIR:   $DATA_DIR"
echo ""

# ---------- 1. ResNet v1 int8 TFLite model ----------

MODEL_PATH="${MODEL_DIR}/resnet_v1_int8.tflite"
# MLCommons tiny repo — ResNet int8 quantized (32x32 input, CIFAR-10, ~85% top-1)
MODEL_URLS=(
    "https://raw.githubusercontent.com/mlcommons/tiny/master/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite"
    "https://raw.githubusercontent.com/mlcommons/tiny/master/benchmark/training/image_classification/trained_models/pretrainedResnet.tflite"
)

if [[ -f "$MODEL_PATH" ]]; then
    SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    echo "  SKIP: model already present at $MODEL_PATH  (${SIZE})"
else
    echo "Step 1/3: Downloading ResNet int8 TFLite model..."
    DOWNLOADED=0
    for URL in "${MODEL_URLS[@]}"; do
        echo "  Trying: $URL"
        if curl -fsSL --retry 3 --retry-delay 2 -o "$MODEL_PATH" "$URL"; then
            SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
            echo "  OK: $MODEL_PATH  (${SIZE})"
            DOWNLOADED=1
            break
        else
            echo "  Failed (404 or network error), trying next..."
            rm -f "$MODEL_PATH"
        fi
    done
    if [[ "$DOWNLOADED" -eq 0 ]]; then
        echo "ERROR: All model download URLs failed." >&2
        echo "  Manual download:" >&2
        echo "    curl -o $MODEL_PATH ${MODEL_URLS[0]}" >&2
        exit 1
    fi
fi

echo ""

# ---------- 2. CIFAR-10 test set ----------

CIFAR_NPZ="${IC_DIR}/cifar10_test.npz"
CIFAR_TAR="${IC_DIR}/cifar-10-python.tar.gz"
CIFAR_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

if [[ -f "$CIFAR_NPZ" ]]; then
    SIZE=$(du -sh "$CIFAR_NPZ" | cut -f1)
    echo "  SKIP: CIFAR-10 test set already present at $CIFAR_NPZ  (${SIZE})"
else
    echo "Step 2/3: Downloading CIFAR-10 (~170 MB)..."
    if curl -fL --retry 3 --retry-delay 5 -o "$CIFAR_TAR" "$CIFAR_URL"; then
        SIZE=$(du -sh "$CIFAR_TAR" | cut -f1)
        echo "  Downloaded: $CIFAR_TAR  (${SIZE})"
    else
        echo "ERROR: CIFAR-10 download failed from $CIFAR_URL" >&2
        exit 1
    fi

    echo ""
    echo "Step 3/3: Extracting CIFAR-10 test batch → cifar10_test.npz..."
    CIFAR_TAR="$CIFAR_TAR" CIFAR_NPZ="$CIFAR_NPZ" python3 - <<'PYEOF'
import os, pickle, tarfile, numpy as np

tar_path = os.environ["CIFAR_TAR"]
out_path = os.environ["CIFAR_NPZ"]

with tarfile.open(tar_path, "r:gz") as tar:
    member = next(m for m in tar.getmembers() if "test_batch" in m.name)
    f = tar.extractfile(member)
    batch = pickle.load(f, encoding="bytes")

images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC uint8
labels = np.array(batch[b"labels"], dtype=np.int32)

np.savez_compressed(out_path, images=images, labels=labels)
print(f"  Saved {len(labels)} images + labels → {out_path}")
PYEOF

    rm -f "$CIFAR_TAR"
    echo "  Removed tar archive."
fi

# ---------- validation ----------

echo ""
echo "=== Validation ==="
ERRORS=0

for f in "$MODEL_PATH" "$CIFAR_NPZ"; do
    if [[ ! -f "$f" ]]; then
        echo "  MISSING: $f"
        ERRORS=1
    else
        SIZE=$(du -sh "$f" | cut -f1)
        echo "  OK: $f  (${SIZE})"
    fi
done

if [[ "$ERRORS" -ne 0 ]]; then
    echo ""
    echo "ERROR: Some files are missing. Re-run to retry downloads." >&2
    exit 1
fi

echo ""
echo "MLPerf Tiny data preparation complete."
echo ""
echo "Next steps:"
echo "  make run-mlperf-tiny       # GPU (A100 with TFLite GPU delegate)"
echo "  make run-mlperf-tiny-cpu   # CPU (local dev)"
