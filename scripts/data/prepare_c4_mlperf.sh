#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# prepare_c4_mlperf.sh — Prepare C4 v3.0.1 data for MLPerf Llama3.1 8B.
#
# Two paths:
#
#   Path A (preferred — no HF gating):
#     Download pre-tokenized Megatron mmap files from MLCommons R2 storage.
#     Training: c4-train.en_<N>_text_document.{bin,idx} (8 shards, 768-1023)
#     Validation: c4-validation-91205-samples.en_text_document.{bin,idx}
#     Total size: ~80GB. These are ready to use directly by NeMo.
#     Uses the MLCommons R2 downloader (no account required).
#
#   Path B (self-tokenize):
#     Download raw C4 from HuggingFace (~300GB) and tokenize using the
#     Llama 3.1 tokenizer. Requires HF_TOKEN for the gated tokenizer repo.
#     Uses preprocess_data.py from inside the NeMo container.
#
# Environment variables:
#   DATA_DIR      — destination directory (default: /data/c4)
#   NEMO_IMAGE    — NeMo container image (default: nvcr.io/nvidia/nemo:24.12-rc0)
#   HF_TOKEN      — HuggingFace token (required for Path B only)
#   TOKENIZE_PATH — set to "B" to use Path B; default is Path A instructions
#
# Usage (Path A — copy pre-tokenized files manually first):
#   DATA_DIR=/data/c4 bash scripts/data/prepare_c4_mlperf.sh
#
# Usage (Path B — self-tokenize):
#   HF_TOKEN=hf_... DATA_DIR=/data/c4 TOKENIZE_PATH=B \
#     bash scripts/data/prepare_c4_mlperf.sh
# ============================================================================

DATA_DIR="${DATA_DIR:-/data/c4}"
NEMO_IMAGE="${NEMO_IMAGE:-nvcr.io/nvidia/nemo:24.12-rc0}"
TOKENIZE_PATH="${TOKENIZE_PATH:-A}"
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$DATA_DIR"

echo "=== C4 MLPerf data preparation ==="
echo "  DATA_DIR:     $DATA_DIR"
echo "  NEMO_IMAGE:   $NEMO_IMAGE"
echo "  Path:         $TOKENIZE_PATH"
echo ""

if [[ "$TOKENIZE_PATH" == "A" ]]; then
    # ---------- Path A: download via MLCommons R2 downloader ----------
    echo "Path A selected: downloading pre-tokenized C4 from MLCommons R2 storage."
    echo "  Destination: ${DATA_DIR}"
    echo "  Expected size: ~80 GB total"
    echo ""

    mkdir -p "${DATA_DIR}"
    cd "${DATA_DIR}"

    echo "Step 1/3: Downloading dataset (~80 GB)..."
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        -d llama3_1_8b_preprocessed_c4_dataset \
        https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

    echo ""
    echo "Step 2/3: Downloading tokenizer..."
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        -d llama3_1_8b_tokenizer \
        https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

    echo ""
    echo "Step 3/3: Verifying downloaded files..."
    echo "Files in ${DATA_DIR}:"
    find "${DATA_DIR}" -name "*.bin" -o -name "*.idx" | sort | while read -r f; do
        SIZE=$(du -sh "$f" | cut -f1)
        echo "  OK: $f  (${SIZE})"
    done

    BIN_COUNT=$(find "${DATA_DIR}" -name "*.bin" | wc -l)
    if [[ "$BIN_COUNT" -eq 0 ]]; then
        echo ""
        echo "ERROR: No .bin files found after download. Check download logs above."
        exit 1
    fi

    echo ""
    echo "Download complete. ${BIN_COUNT} .bin files found."

elif [[ "$TOKENIZE_PATH" == "B" ]]; then
    # ---------- Path B: self-tokenize via NeMo container ----------
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: HF_TOKEN is required for Path B (Llama 3.1 tokenizer is gated)." >&2
        echo "  1. Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B" >&2
        echo "  2. Generate a token at https://huggingface.co/settings/tokens" >&2
        echo "  3. Re-run: HF_TOKEN=hf_... TOKENIZE_PATH=B bash scripts/data/prepare_c4_mlperf.sh" >&2
        exit 1
    fi

    echo "Path B selected: downloading raw C4 and tokenizing inside NeMo container."
    echo "WARNING: Raw C4 download is ~305GB. This will take several hours."
    echo ""

    RAW_DIR="${DATA_DIR}/raw"
    mkdir -p "$RAW_DIR"

    echo "Step 1/3: Downloading C4 train split from HuggingFace..."
    docker run --rm \
        -v "${DATA_DIR}:/data/c4" \
        -e HF_TOKEN="${HF_TOKEN}" \
        "${NEMO_IMAGE}" \
        python3 -c "
from datasets import load_dataset
import os, json

token = os.environ['HF_TOKEN']
print('Downloading C4 train split (this is large)...')
ds = load_dataset('allenai/c4', 'en', split='train', streaming=False,
                  cache_dir='/data/c4/raw', token=token)
out_path = '/data/c4/raw/c4_train.jsonl'
print(f'Writing to {out_path}...')
with open(out_path, 'w') as f:
    for row in ds:
        f.write(json.dumps({'text': row['text']}) + '\n')
print('Train download complete.')
"

    echo ""
    echo "Step 2/3: Downloading C4 validation split..."
    docker run --rm \
        -v "${DATA_DIR}:/data/c4" \
        -e HF_TOKEN="${HF_TOKEN}" \
        "${NEMO_IMAGE}" \
        python3 -c "
from datasets import load_dataset
import os, json

token = os.environ['HF_TOKEN']
print('Downloading C4 validation split...')
ds = load_dataset('allenai/c4', 'en', split='validation', streaming=False,
                  cache_dir='/data/c4/raw', token=token)
out_path = '/data/c4/raw/c4_val.jsonl'
with open(out_path, 'w') as f:
    for row in ds:
        f.write(json.dumps({'text': row['text']}) + '\n')
print('Validation download complete.')
"

    echo ""
    echo "Step 3/3: Tokenizing with Llama 3.1 tokenizer (preprocess_data.py)..."
    for SPLIT in train val; do
        OUTPUT_PREFIX="/data/c4/c4_${SPLIT}_text_document"
        echo "  Tokenizing ${SPLIT} split → ${OUTPUT_PREFIX}.{bin,idx}"
        docker run --rm \
            -v "${DATA_DIR}:/data/c4" \
            -e HF_TOKEN="${HF_TOKEN}" \
            "${NEMO_IMAGE}" \
            python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
                --input="/data/c4/raw/c4_${SPLIT}.jsonl" \
                --json-keys=text \
                --tokenizer-library=huggingface \
                --tokenizer-type=meta-llama/Meta-Llama-3.1-8B \
                --tokenizer-auth-token="${HF_TOKEN}" \
                --output-prefix="${OUTPUT_PREFIX}" \
                --append-eod \
                --workers=8
    done

    echo ""
    echo "Tokenization complete. Verifying output files..."
    MISSING=0
    for f in c4_train_text_document.bin c4_train_text_document.idx \
              c4_val_text_document.bin c4_val_text_document.idx; do
        if [[ ! -f "${DATA_DIR}/${f}" ]]; then
            echo "  MISSING: ${DATA_DIR}/${f}"
            MISSING=1
        else
            SIZE=$(du -sh "${DATA_DIR}/${f}" | cut -f1)
            echo "  OK:      ${DATA_DIR}/${f}  (${SIZE})"
        fi
    done

    if [[ "$MISSING" -eq 1 ]]; then
        echo "ERROR: Tokenization did not produce all expected files." >&2
        exit 1
    fi

else
    echo "ERROR: TOKENIZE_PATH must be 'A' or 'B', got '${TOKENIZE_PATH}'" >&2
    exit 1
fi

echo ""
echo "C4 data preparation complete. Files are in: ${DATA_DIR}"
echo ""
echo "Next steps:"
echo "  1. Copy files to all nodes (or mount shared NFS at ${DATA_DIR})"
echo "  2. Run: make run-mlperf"
