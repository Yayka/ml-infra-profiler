#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# verify_run.sh — Verify MLPerf T2I convergence from result file.
#
# Reads results/mlperf_t2i/result.txt and checks that the reported validation
# loss meets the MLPerf target (<= 0.586).
#
# No API calls — works fully offline.
#
# Usage:
#   bash scripts/launch/mlperf_t2i/verify_run.sh
# ============================================================================

RESULT_FILE="${RESULT_FILE:-results/mlperf_t2i/result.txt}"
TARGET=0.586

if [[ ! -f "$RESULT_FILE" ]]; then
    echo "ERROR: result file not found: $RESULT_FILE" >&2
    echo "  Run 'make run-mlperf-t2i' first." >&2
    exit 1
fi

echo "=== MLPerf Text-to-Image — Convergence Verification ==="
echo "  File: $RESULT_FILE"
echo ""

# Extract "Val loss" line
VAL_LOSS_LINE=$(grep -i "^Val loss" "$RESULT_FILE" || true)
if [[ -z "$VAL_LOSS_LINE" ]]; then
    echo "ERROR: Could not find 'Val loss' line in $RESULT_FILE" >&2
    echo "File contents:" >&2
    cat "$RESULT_FILE" >&2
    exit 1
fi

echo "$VAL_LOSS_LINE"

# Parse the numeric value
VAL_LOSS=$(echo "$VAL_LOSS_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)

if [[ -z "$VAL_LOSS" ]]; then
    echo "ERROR: Could not parse val_loss value from: $VAL_LOSS_LINE" >&2
    exit 1
fi

echo ""
echo "  Val loss : ${VAL_LOSS}"
echo "  Target   : ${TARGET}"
echo ""

# Compare using Python (bc not always available on all distros)
PASS=$(python3 -c "print('yes' if float('${VAL_LOSS}') <= float('${TARGET}') else 'no')")

if [[ "$PASS" == "yes" ]]; then
    echo "  Result: PASS (${VAL_LOSS} <= ${TARGET})"
    exit 0
else
    echo "  Result: FAIL (${VAL_LOSS} > ${TARGET})"
    exit 1
fi
