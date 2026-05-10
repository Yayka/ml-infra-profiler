#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_sdxl.sh — Two-node MLPerf SDXL T2I inference benchmark launcher.
#
# Architecture:
#   CLIENT NODE (this script)          SERVER NODE
#   main.py (LoadGen) ─── HTTP ──►  server.py (FastAPI + SDXL)
#
# Required env vars:
#   SERVER_IP      — IP/hostname of the GPU server node
#   SSH_USER       — SSH username on server node (default: $USER)
#   SSH_KEY        — path to SSH private key (default: ~/.ssh/id_rsa)
#
# Optional:
#   SCENARIO       — SingleStream | Offline | all (default: all)
#   MAX_QUERY_COUNT — cap on queries (default: 100 for smoke test; 5000 for full)
#   SDXL_VENV      — path to Python venv (default: /data/.venv-sdxl)
#   CONFIG         — path to config YAML (default: scripts/launch/mlperf_sdxl/config/sdxl_2node.yaml)
#   SERVER_VENV    — venv path on server node (default: same as SDXL_VENV)
#   PERF_ONLY      — set to 1 to skip accuracy evaluation
#
# Usage:
#   source .env
#   SERVER_IP=10.0.0.10 SSH_USER=ubuntu make run-mlperf-sdxl
#   SERVER_IP=10.0.0.10 SCENARIO=SingleStream make run-mlperf-sdxl
# ============================================================================

SERVER_IP="${SERVER_IP:?ERROR: SERVER_IP must be set (IP/hostname of GPU server node)}"
SSH_USER="${SSH_USER:-$USER}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SCENARIO="${SCENARIO:-all}"
MAX_QUERY_COUNT="${MAX_QUERY_COUNT:-100}"
SDXL_VENV="${SDXL_VENV:-/data/.venv-sdxl}"
SERVER_VENV="${SERVER_VENV:-$SDXL_VENV}"
CONFIG="${CONFIG:-scripts/launch/mlperf_sdxl/config/sdxl_2node.yaml}"
SERVER_PORT="${SERVER_PORT:-8080}"
LOG_BASE="logs/mlperf_sdxl_$(date +%Y%m%d_%H%M%S)"
REPO_DIR="$(pwd)"
PERF_ONLY="${PERF_ONLY:-0}"

SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o BatchMode=yes"
SSH="ssh ${SSH_OPTS} ${SSH_USER}@${SERVER_IP}"

SERVER_SCRIPT="scripts/launch/mlperf_sdxl/server.py"
CLIENT_SCRIPT="scripts/launch/mlperf_sdxl/main.py"
ACCURACY_SCRIPT="scripts/launch/mlperf_sdxl/accuracy.py"

OUTPUT_DIR="$( grep 'output_dir' "${CONFIG}" | head -1 | awk '{print $2}' )"
OUTPUT_DIR="${OUTPUT_DIR:-/data/mlperf_sdxl/output}"
ANNOTATIONS="$( grep 'annotations_path' "${CONFIG}" | head -1 | awk '{print $2}' )"
ANNOTATIONS="${ANNOTATIONS:-/data/mlperf_sdxl/annotations/captions_val2014.json}"

mkdir -p "$LOG_BASE"

echo "=== MLPerf SDXL T2I Inference Benchmark ==="
echo "  Server node  : ${SSH_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "  Scenario     : ${SCENARIO}"
echo "  Max queries  : ${MAX_QUERY_COUNT}"
echo "  Config       : ${CONFIG}"
echo "  Log dir      : ${LOG_BASE}/"
echo ""

# ---------- 1. Start server on GPU node ----------

echo "[1/4] Starting SDXL server on ${SERVER_IP}..."
$SSH "bash -lc '
    cd $(basename ${REPO_DIR}) 2>/dev/null || cd ${REPO_DIR}
    nohup ${SERVER_VENV}/bin/python ${SERVER_SCRIPT} \
        --config ${CONFIG} \
        --host 0.0.0.0 --port ${SERVER_PORT} \
        > /tmp/sdxl_server.log 2>&1 &
    echo \$! > /tmp/sdxl_server.pid
    echo \"Server PID: \$(cat /tmp/sdxl_server.pid)\"
'"

# ---------- 2. Wait for server to be ready ----------

echo "[2/4] Waiting for server at http://${SERVER_IP}:${SERVER_PORT}/healthz ..."
DEADLINE=$(( $(date +%s) + 300 ))
until curl -sf "http://${SERVER_IP}:${SERVER_PORT}/healthz" > /dev/null 2>&1; do
    if [[ $(date +%s) -gt $DEADLINE ]]; then
        echo "ERROR: Server did not become ready within 5 minutes." >&2
        echo "Check server logs: ssh ${SSH_USER}@${SERVER_IP} 'cat /tmp/sdxl_server.log'" >&2
        exit 1
    fi
    echo "  ... waiting"
    sleep 10
done
echo "  Server is ready."

# ---------- 3. Run LoadGen scenarios ----------

run_scenario() {
    local scenario="$1"
    local log_dir="${LOG_BASE}/${scenario}"
    mkdir -p "$log_dir"

    echo ""
    echo "[3/4] Running ${scenario} scenario..."
    SERVER_IP="${SERVER_IP}" \
    "${SDXL_VENV}/bin/python" "${CLIENT_SCRIPT}" \
        --scenario "${scenario}" \
        --config "${CONFIG}" \
        --server-host "${SERVER_IP}" \
        --server-port "${SERVER_PORT}" \
        --max-query-count "${MAX_QUERY_COUNT}" \
        --output-log-dir "${log_dir}" \
        --skip-server-check \
        2>&1 | tee "${log_dir}/run_performance.log"

    echo "  ${scenario} complete. Logs: ${log_dir}/"
}

case "$SCENARIO" in
    all)
        run_scenario SingleStream
        run_scenario Offline
        ;;
    SingleStream|Offline)
        run_scenario "$SCENARIO"
        ;;
    *)
        echo "ERROR: SCENARIO must be 'SingleStream', 'Offline', or 'all'. Got: $SCENARIO" >&2
        exit 1
        ;;
esac

# ---------- 4. Stop server ----------

echo ""
echo "[4/4] Stopping server on ${SERVER_IP}..."
$SSH "bash -lc '
    if [[ -f /tmp/sdxl_server.pid ]]; then
        kill \$(cat /tmp/sdxl_server.pid) 2>/dev/null || true
        rm -f /tmp/sdxl_server.pid
        echo \"Server stopped.\"
    else
        echo \"No PID file found; server may have already exited.\"
    fi
'" || true

# ---------- 5. Accuracy evaluation ----------

if [[ "${PERF_ONLY}" != "1" ]]; then
    echo ""
    echo "[+] Running accuracy evaluation..."
    "${SDXL_VENV}/bin/python" "${ACCURACY_SCRIPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --annotations "${ANNOTATIONS}" \
        2>&1 | tee "${LOG_BASE}/accuracy_result.log"
fi

echo ""
echo "=== Benchmark complete ==="
echo "  Logs : ${LOG_BASE}/"
echo "  Run 'make verify-mlperf-sdxl' to check FID + CLIP targets."
