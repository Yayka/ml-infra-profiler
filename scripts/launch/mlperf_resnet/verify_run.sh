#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# verify_run.sh — Verify MLPerf ResNet-50 convergence from result file.
#
# Reads results/mlperf_resnet/result.txt and checks that the reported
# top-1 accuracy meets the MLPerf target (>= 75.9%).
#
# No API calls — works fully offline.
#
# Usage:
#   bash scripts/launch/mlperf_resnet/verify_run.sh
# ============================================================================

RESULT_FILE="results/mlperf_resnet/result.txt"
TARGET=75.9

if [[ ! -f "$RESULT_FILE" ]]; then
    echo "ERROR: result file not found: $RESULT_FILE" >&2
    echo "  Run 'make run-mlperf-resnet' first." >&2
    exit 1
fi

echo "=== MLPerf ResNet-50 — Convergence Verification ==="
echo "  File: $RESULT_FILE"
echo ""

# Extract "Top-1 Accuracy : XX.XX%" line
ACCURACY_LINE=$(grep -i "^Top-1 Accuracy" "$RESULT_FILE" || true)
if [[ -z "$ACCURACY_LINE" ]]; then
    echo "ERROR: Could not find 'Top-1 Accuracy' line in $RESULT_FILE" >&2
    echo "File contents:" >&2
    cat "$RESULT_FILE" >&2
    exit 1
fi

echo "$ACCURACY_LINE"

# Parse the numeric value (e.g. "76.12")
ACCURACY=$(echo "$ACCURACY_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)

if [[ -z "$ACCURACY" ]]; then
    echo "ERROR: Could not parse accuracy value from: $ACCURACY_LINE" >&2
    exit 1
fi

echo ""
echo "  Top-1 accuracy : ${ACCURACY}%"
echo "  Target         : ${TARGET}%"
echo ""

# Compare using Python (bc not always available on all distros)
PASS=$(python3 -c "print('yes' if float('${ACCURACY}') >= float('${TARGET}') else 'no')")

if [[ "$PASS" == "yes" ]]; then
    echo "  Result: PASS (${ACCURACY}% >= ${TARGET}%)"
    exit 0
else
    echo "  Result: FAIL (${ACCURACY}% < ${TARGET}%)"
    exit 1
fi
