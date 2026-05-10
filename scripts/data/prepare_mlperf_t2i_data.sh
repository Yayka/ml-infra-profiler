#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_mlperf_t2i_data.sh — Download CC12M subset, COCO-2014 val, and
# Flux.1-schnell model weights for the MLPerf T2I benchmark.
#
# DATA_DIR defaults to /data/mlperf_t2i on GPU nodes (large datasets).
# HF_TOKEN required for gated model downloads.
#
# Usage:
#   HF_TOKEN=<token> bash scripts/data/prepare_mlperf_t2i_data.sh
#   DATA_DIR=/mnt/data/mlperf_t2i HF_TOKEN=<token> bash scripts/data/prepare_mlperf_t2i_data.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-/data/mlperf_t2i}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN for downloading gated models from HuggingFace}"
T2I_VENV="${T2I_VENV:-/data/.venv-t2i}"
PYTHON="${T2I_VENV}/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv python not found at $PYTHON"
    echo "Run: make setup-mlperf-t2i"
    exit 1
fi

echo "=== MLPerf T2I Data Preparation ==="
echo "  Data dir: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR/cc12m_256" "$DATA_DIR/coco2014_256" "$DATA_DIR/models"

# --- 1. Download CC12M subset (training data) ---
echo "[1/3] Downloading CC12M subset (~1.1M images, 256x256)..."
if [[ -d "$DATA_DIR/cc12m_256" ]] && [[ $(find "$DATA_DIR/cc12m_256" -name "*.jpg" 2>/dev/null | head -1) ]]; then
    echo "  CC12M data already exists, skipping."
else
    echo "  Downloading CC12M via img2dataset..."
    echo "  NOTE: This requires img2dataset (pip install img2dataset) and ~200GB disk space."
    echo "  If using MLCommons pre-processed data, copy to $DATA_DIR/cc12m_256/ manually."
    echo ""
    echo "  To download manually:"
    echo "    1. Get CC12M TSV from https://github.com/google-research-datasets/conceptual-12m"
    echo "    2. Filter to MLPerf subset (1,099,776 samples)"
    echo "    3. Resize to 256x256 and save to $DATA_DIR/cc12m_256/"
    echo ""
    echo "  For MLCommons R2 bucket download (recommended):"
    echo "    rclone copy mlc-r2:mlcommons-training-wg-public/text_to_image/cc12m_256 $DATA_DIR/cc12m_256/"
fi

# --- 2. Download COCO-2014 validation (evaluation data) ---
echo ""
echo "[2/3] Downloading COCO-2014 validation set (29,696 images, 256x256)..."
if [[ -d "$DATA_DIR/coco2014_256" ]] && [[ $(find "$DATA_DIR/coco2014_256" -name "*.jpg" 2>/dev/null | head -1) ]]; then
    echo "  COCO-2014 val data already exists, skipping."
else
    COCO_ZIP="$DATA_DIR/val2014.zip"
    if [[ ! -f "$COCO_ZIP" ]]; then
        echo "  Downloading COCO-2014 val images..."
        wget -q --show-progress -O "$COCO_ZIP" \
            "http://images.cocodataset.org/zips/val2014.zip"
    fi
    echo "  Extracting..."
    unzip -q -o "$COCO_ZIP" -d "$DATA_DIR/coco2014_raw/"
    rm -f "$COCO_ZIP"

    echo "  Resizing to 256x256..."
    $PYTHON -c "
import os, glob
from PIL import Image
from pathlib import Path

src = '$DATA_DIR/coco2014_raw/val2014'
dst = '$DATA_DIR/coco2014_256'
os.makedirs(dst, exist_ok=True)

images = sorted(glob.glob(os.path.join(src, '*.jpg')))
print(f'  Resizing {len(images)} images...')
for i, img_path in enumerate(images):
    img = Image.open(img_path).convert('RGB').resize((256, 256), Image.BICUBIC)
    img.save(os.path.join(dst, Path(img_path).name))
    if (i + 1) % 5000 == 0:
        print(f'    {i + 1}/{len(images)}')
print(f'  Done: {len(images)} images in {dst}')
"
    echo "  Downloading COCO-2014 captions..."
    wget -q --show-progress -O "$DATA_DIR/annotations.zip" \
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    unzip -q -o "$DATA_DIR/annotations.zip" -d "$DATA_DIR/"
    rm -f "$DATA_DIR/annotations.zip"

    echo "  Generating caption sidecar files..."
    $PYTHON -c "
import json, os
from pathlib import Path

ann_file = '$DATA_DIR/annotations/captions_val2014.json'
dst = '$DATA_DIR/coco2014_256'

with open(ann_file) as f:
    data = json.load(f)

# Build image_id -> first caption map
captions = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in captions:
        captions[img_id] = ann['caption']

# Build image_id -> filename map
id_to_file = {img['id']: img['file_name'] for img in data['images']}

count = 0
for img_id, caption in captions.items():
    filename = id_to_file.get(img_id)
    if filename:
        txt_path = os.path.join(dst, Path(filename).stem + '.txt')
        with open(txt_path, 'w') as f:
            f.write(caption)
        count += 1

print(f'  Written {count} caption files to {dst}')
"
fi

# --- 3. Download Flux.1-schnell model weights ---
echo ""
echo "[3/3] Downloading Flux.1-schnell model weights..."
if [[ -d "$DATA_DIR/models/flux1-schnell" ]] && [[ -f "$DATA_DIR/models/flux1-schnell/config.json" ]]; then
    echo "  Flux.1-schnell weights already exist, skipping."
else
    echo "  Downloading via huggingface_hub..."
    $PYTHON -c "
from huggingface_hub import snapshot_download
import os

os.environ['HF_TOKEN'] = '$HF_TOKEN'
snapshot_download(
    'black-forest-labs/FLUX.1-schnell',
    local_dir='$DATA_DIR/models/flux1-schnell',
    token='$HF_TOKEN',
)
print('  Flux.1-schnell downloaded successfully.')
"
fi

echo ""
echo "=== Data preparation complete ==="
echo "  CC12M train:      $DATA_DIR/cc12m_256/"
echo "  COCO-2014 val:    $DATA_DIR/coco2014_256/"
echo "  Flux.1-schnell:   $DATA_DIR/models/flux1-schnell/"
echo ""
echo "Next: make run-mlperf-t2i"
