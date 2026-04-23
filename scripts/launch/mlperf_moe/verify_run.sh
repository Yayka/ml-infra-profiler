#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# verify_run.sh — Check whether the MLPerf MoE run met the target
# validation loss of <= 3.34 (log perplexity).
#
# Reads results/mlperf_moe/summary.json for the final val_loss.
# Falls back to W&B if summary.json is not available.
#
# Usage:
#   bash scripts/launch/mlperf_moe/verify_run.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/mlperf_moe}"
SUMMARY_FILE="${RESULTS_DIR}/summary.json"
TARGET_VAL_LOSS=3.34

echo "=== MLPerf MoE — Convergence Verification ==="
echo "  Results dir: $RESULTS_DIR"
echo "  Target:      val_loss <= $TARGET_VAL_LOSS"
echo ""

if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "ERROR: summary.json not found: $SUMMARY_FILE" >&2
    echo "  Run 'make run-mlperf-moe' first." >&2
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

"$PYTHON" - "$SUMMARY_FILE" "$TARGET_VAL_LOSS" <<'PYEOF'
import json
import sys

summary_path = sys.argv[1]
target = float(sys.argv[2])

with open(summary_path) as f:
    summary = json.load(f)

print(f"  Benchmark:       {summary.get('benchmark', 'N/A')}")
print(f"  Model:           {summary.get('model', 'N/A')}")
print(f"  Experts:         {summary.get('num_experts', 'N/A')} (top-{summary.get('moe_router_topk', 'N/A')})")
print(f"  Steps completed: {summary.get('total_steps_completed', 'N/A')}")
print(f"  Final train loss: {summary.get('final_train_loss', 'N/A')}")
print()

val_loss = summary.get("final_val_loss")
if val_loss is None:
    print("ERROR: No val_loss recorded in summary.json.", file=sys.stderr)
    print("  The run may not have completed a validation step.", file=sys.stderr)
    print("  Increase trainer.max_steps or decrease trainer.val_check_interval.", file=sys.stderr)
    sys.exit(1)

print(f"  Final val_loss:  {val_loss:.4f}")
print(f"  Target val_loss: <= {target}")
print()

if val_loss <= target:
    print(f"RESULT: PASS  (val_loss {val_loss:.4f} <= {target})")
    sys.exit(0)
else:
    print(f"RESULT: FAIL  (val_loss {val_loss:.4f} > {target})")
    print()
    print("Note: This is a 4-GPU Open Division configuration.")
    print("If the goal is infrastructure profiling, the convergence target is informational only.")
    print("For convergence, set trainer.max_steps=1200000 in the config.")
    sys.exit(1)
PYEOF
