#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_resnet.sh — Run MLPerf ResNet-50 v1.5 training benchmark.
#
# Trains ResNet-50 v1.5 on ImageNet using TF2 with MirroredStrategy,
# then writes results to the submission directory.
#
# Prerequisites:
#   - make setup-mlperf-resnet        (.venv-resnet with tensorflow 2.14)
#   - make prepare-mlperf-resnet-data (ImageNet TFRecords in data/imagenet/tfrecord/)
#
# Usage:
#   bash scripts/launch/mlperf_resnet/run_mlperf_resnet.sh
#
# Override config via env:
#   CONFIG=scripts/launch/mlperf_resnet/config/resnet_4gpu.yaml \
#     bash scripts/launch/mlperf_resnet/run_mlperf_resnet.sh
# ============================================================================

CONFIG="${CONFIG:-scripts/launch/mlperf_resnet/config/resnet_4gpu.yaml}"

LOG_DIR="logs/mlperf_resnet_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== MLPerf ResNet-50 v1.5 Training Benchmark ==="
echo "  Config:   $CONFIG"
echo "  Log dir:  $LOG_DIR"
echo ""

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG" >&2
    echo "  Available configs:" >&2
    ls scripts/launch/mlperf_resnet/config/ >&2
    exit 1
fi

# Activate the dedicated TF venv (separate from main .venv — TF 2.14 conflicts with torch)
VENV=".venv-resnet"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: $VENV not found. Run: make setup-mlperf-resnet" >&2
    exit 1
fi
source "${VENV}/bin/activate"

# ---------- run training ----------

python scripts/launch/mlperf_resnet/benchmark_runner.py \
    --config "$CONFIG" \
    --log-dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/run.log"

BENCH_EXIT=${PIPESTATUS[0]}

echo ""
echo "Results:  results/mlperf_resnet/"
echo "Run logs: $LOG_DIR/"
echo ""

if [[ "$BENCH_EXIT" -ne 0 ]]; then
    echo "WARNING: Benchmark exited with code $BENCH_EXIT (accuracy below target or error)."
    echo "  Check $LOG_DIR/run.log for details."
    exit "$BENCH_EXIT"
fi

echo "Run 'make verify-mlperf-resnet' to confirm top-1 accuracy >= 75.9%."
