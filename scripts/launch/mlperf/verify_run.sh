#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# verify_run.sh — Check whether the MLPerf Llama3.1 8B run met the target
# validation perplexity of ≤ 3.3 (i.e., val_loss ≤ ln(3.3) ≈ 1.1939).
#
# Reads the final val_loss from W&B using the Python SDK.
# Requires WANDB_API_KEY and WANDB_BASE_URL in .env (or environment).
#
# Usage:
#   ./scripts/launch/mlperf/verify_run.sh
#   WANDB_RUN_PATH=entity/project/run_id ./scripts/launch/mlperf/verify_run.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load .env
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

WANDB_PROJECT="${WANDB_PROJECT:-ml-netprof}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mlperf-llama3-8b-4gpu}"
# Override with full path entity/project/run_id if known
WANDB_RUN_PATH="${WANDB_RUN_PATH:-}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

"$PYTHON" - <<'PYEOF'
import os
import sys
import math

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb", file=sys.stderr)
    sys.exit(1)

api = wandb.Api(
    api_key=os.environ.get("WANDB_API_KEY"),
    overrides={"base_url": os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")},
)

run_path = os.environ.get("WANDB_RUN_PATH", "")
if run_path:
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"ERROR: Could not fetch run '{run_path}': {e}", file=sys.stderr)
        sys.exit(1)
else:
    project = os.environ.get("WANDB_PROJECT", "ml-netprof")
    entity = os.environ.get("WANDB_ENTITY", "")
    run_name = os.environ.get("WANDB_RUN_NAME", "mlperf-llama3-8b-4gpu")
    path = f"{entity}/{project}" if entity else project
    try:
        runs = api.runs(path, filters={"display_name": run_name}, order="-created_at")
        runs = list(runs)
    except Exception as e:
        print(f"ERROR: Could not query runs in '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    if not runs:
        print(f"ERROR: No run named '{run_name}' found in project '{path}'.", file=sys.stderr)
        print("  Set WANDB_RUN_PATH=entity/project/run_id to target a specific run.", file=sys.stderr)
        sys.exit(1)
    run = runs[0]
    print(f"Found run: {run.name} (id: {run.id}, state: {run.state})")

# Fetch final val_loss — NeMo logs it as 'val_loss' in the run summary
val_loss = run.summary.get("val_loss")
if val_loss is None:
    # Try history for the last logged val_loss
    history = run.scan_history(keys=["val_loss"])
    rows = list(history)
    if rows:
        val_loss = rows[-1].get("val_loss")

if val_loss is None:
    print("ERROR: 'val_loss' metric not found in run summary or history.", file=sys.stderr)
    print("  The run may still be in progress or may not have completed a validation step.", file=sys.stderr)
    sys.exit(1)

TARGET_PERPLEXITY = 3.3
TARGET_LOSS = math.log(TARGET_PERPLEXITY)

val_perplexity = math.exp(val_loss)
print(f"\nFinal val_loss:        {val_loss:.4f}")
print(f"Final val_perplexity:  {val_perplexity:.4f}")
print(f"Target perplexity:     ≤ {TARGET_PERPLEXITY}")
print(f"Target val_loss:       ≤ {TARGET_LOSS:.4f}")
print()

if val_perplexity <= TARGET_PERPLEXITY:
    print(f"RESULT: PASS  (val_perplexity {val_perplexity:.4f} ≤ {TARGET_PERPLEXITY})")
    sys.exit(0)
else:
    print(f"RESULT: FAIL  (val_perplexity {val_perplexity:.4f} > {TARGET_PERPLEXITY})")
    print()
    print("Note: This is a 4-GPU Open Division configuration (not closed-division reference).")
    print("If the goal is infrastructure profiling, perplexity target is informational only.")
    sys.exit(1)
PYEOF
