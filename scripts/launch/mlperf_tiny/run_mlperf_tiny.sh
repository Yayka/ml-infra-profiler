#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_tiny.sh — Run MLPerf Tiny IC benchmark (TFLite, Single Stream).
#
# Runs the inference benchmark on a single node using TFLite, then generates
# the MLPerf Tiny v1.1 submission tree.
#
# Prerequisites:
#   - make setup-mlperf-tiny      (.venv-tiny with tensorflow 2.14)
#   - make prepare-mlperf-tiny-data  (CIFAR-10 + ResNet TFLite model)
#
# Usage:
#   bash scripts/launch/mlperf_tiny/run_mlperf_tiny.sh
#
# Override via env:
#   BENCHMARK=ic DELEGATE=cpu bash scripts/launch/mlperf_tiny/run_mlperf_tiny.sh
# ============================================================================

BENCHMARK="${BENCHMARK:-ic}"
DELEGATE="${DELEGATE:-gpu}"

CONFIG="scripts/launch/mlperf_tiny/config/${BENCHMARK}_resnet_${DELEGATE}.yaml"

LOG_DIR="logs/mlperf_tiny_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

OUTPUT_DIR="results/mlperf_tiny_v1.1"

echo "=== MLPerf Tiny IC Benchmark ==="
echo "  Benchmark:  $BENCHMARK"
echo "  Delegate:   $DELEGATE"
echo "  Config:     $CONFIG"
echo "  Log dir:    $LOG_DIR"
echo "  Output:     $OUTPUT_DIR"
echo ""

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    echo "  Available configs:" >&2
    ls scripts/launch/mlperf_tiny/config/ >&2
    exit 1
fi

# Activate the dedicated TFLite venv (separate from main .venv — TF 2.14 conflicts with torch)
VENV=".venv-tiny"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: $VENV not found. Run: make setup-mlperf-tiny" >&2
    exit 1
fi
source "${VENV}/bin/activate"

# ---------- performance + accuracy phases ----------

python scripts/launch/mlperf_tiny/benchmark_runner.py \
    --config "$CONFIG" \
    --log-dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/run.log"

BENCH_EXIT=${PIPESTATUS[0]}

echo ""

# ---------- generate submission tree ----------

python scripts/launch/mlperf_tiny/generate_submission.py \
    --log-dir "$LOG_DIR" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Submission tree: $OUTPUT_DIR"
echo "Run logs:        $LOG_DIR/"
echo ""

if [[ "$BENCH_EXIT" -ne 0 ]]; then
    echo "WARNING: Benchmark exited with code $BENCH_EXIT (accuracy below target or error)."
    echo "  Check $LOG_DIR/run.log for details."
    exit "$BENCH_EXIT"
fi

echo "Run 'make verify-mlperf-tiny' to confirm accuracy >= 85%."
