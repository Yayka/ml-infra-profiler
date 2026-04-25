#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# verify_inference_run.sh — Verify MLPerf Inference v5.0 Llama3.1-8B ROUGE scores.
#
# Reads accuracy results from the submission tree and checks that ROUGE scores
# meet 99% of the reference targets (Open Division, BF16).
#
# Targets (99% of reference):
#   ROUGE-1 >= 38.78
#   ROUGE-2 >= 15.91
#   ROUGE-L >= 24.50
#
# No API calls — works fully offline.
#
# Usage:
#   bash scripts/launch/mlperf_inference/verify_inference_run.sh
# ============================================================================

RESULT_DIR="results/mlperf_inference_v5.0/open/ml-infra-profiler/results/llama3.1-8b"

# Targets (99% of reference)
TARGET_ROUGE1=38.78
TARGET_ROUGE2=15.91
TARGET_ROUGEL=24.50

# Check at least the offline accuracy result exists
OFFLINE_RESULT="${RESULT_DIR}/offline/accuracy/result.txt"

if [[ ! -f "$OFFLINE_RESULT" ]]; then
    echo "ERROR: accuracy result not found: $OFFLINE_RESULT" >&2
    echo "  Run 'make run-mlperf-inference' first." >&2
    exit 1
fi

echo "=== MLPerf Inference v5.0 — Llama3.1-8B ROUGE Verification ==="
echo "  File: $OFFLINE_RESULT"
echo ""

# Extract ROUGE scores from accuracy result file.
# Expected format (from evaluate-accuracy.py):
#   {'rouge1': 42.56, 'rouge2': 17.82, 'rougeL': 26.34, ...}
# or:
#   ROUGE-1: 42.56
#   ROUGE-2: 17.82
#   ROUGE-L: 26.34

extract_rouge() {
    local file="$1" metric="$2"
    # Try JSON-style dict first
    python3 -c "
import re, sys
content = open('${file}').read()
# Try dict-style: 'rouge1': 42.56
m = re.search(r\"['\\\"]${metric}['\\\"]\\s*:\\s*([0-9]+\\.?[0-9]*)\", content, re.IGNORECASE)
if m:
    print(m.group(1))
    sys.exit(0)
# Try label-style: ROUGE-1: 42.56 or rouge1 42.56
m = re.search(r'${metric}[^0-9]*([0-9]+\\.?[0-9]*)', content, re.IGNORECASE)
if m:
    print(m.group(1))
    sys.exit(0)
print('')
" 2>/dev/null || echo ""
}

ROUGE1=$(extract_rouge "$OFFLINE_RESULT" "rouge1")
ROUGE2=$(extract_rouge "$OFFLINE_RESULT" "rouge2")
ROUGEL=$(extract_rouge "$OFFLINE_RESULT" "rougeL")

if [[ -z "$ROUGE1" || -z "$ROUGE2" || -z "$ROUGEL" ]]; then
    echo "ERROR: Could not parse ROUGE scores from $OFFLINE_RESULT" >&2
    echo "File contents:" >&2
    cat "$OFFLINE_RESULT" >&2
    exit 1
fi

echo "  ROUGE-1: ${ROUGE1}  (target >= ${TARGET_ROUGE1})"
echo "  ROUGE-2: ${ROUGE2}  (target >= ${TARGET_ROUGE2})"
echo "  ROUGE-L: ${ROUGEL}  (target >= ${TARGET_ROUGEL})"
echo ""

PASS=$(python3 -c "
r1 = float('${ROUGE1}')
r2 = float('${ROUGE2}')
rl = float('${ROUGEL}')
t1 = float('${TARGET_ROUGE1}')
t2 = float('${TARGET_ROUGE2}')
tl = float('${TARGET_ROUGEL}')
fails = []
if r1 < t1: fails.append(f'ROUGE-1 {r1} < {t1}')
if r2 < t2: fails.append(f'ROUGE-2 {r2} < {t2}')
if rl < tl: fails.append(f'ROUGE-L {rl} < {tl}')
if fails:
    print('FAIL: ' + ', '.join(fails))
else:
    print('PASS')
")

if [[ "$PASS" == "PASS" ]]; then
    echo "  Result: PASS — all ROUGE scores meet 99% of reference targets."
    exit 0
else
    echo "  Result: ${PASS}"
    exit 1
fi
