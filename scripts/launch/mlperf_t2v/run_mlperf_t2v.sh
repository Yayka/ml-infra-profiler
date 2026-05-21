#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_mlperf_t2v.sh — Two-node MLPerf Wan2.2-T2V-A14B inference benchmark.
#
# Architecture:
#   CLIENT NODE (this script)          SERVER NODE
#   main.py (LoadGen) ─── HTTP ──►  server.py (FastAPI + Wan2.2-T2V)
#
# Required env vars:
#   SERVER_IP      — IP/hostname of the GPU server node
#   SSH_USER       — SSH username (default: $USER)
#   SSH_KEY        — path to SSH private key (default: ~/.ssh/id_rsa)
#
# Optional:
#   SCENARIO       — SingleStream | Offline | all (default: all)
#   MAX_QUERY_COUNT — queries per scenario (default: 5000)
#   T2V_VENV       — Python venv path (default: /data/.venv-t2v)
#   CONFIG         — config YAML path
#   PERF_ONLY      — set to 1 to skip accuracy evaluation
#
# Usage:
#   SERVER_IP=10.0.0.10 SSH_USER=azureuser make run-mlperf-t2v
# ============================================================================

SERVER_IP="${SERVER_IP:?ERROR: SERVER_IP must be set}"
SSH_USER="${SSH_USER:-$USER}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SCENARIO="${SCENARIO:-all}"
MAX_QUERY_COUNT="${MAX_QUERY_COUNT:-5000}"
T2V_VENV="${T2V_VENV:-/data/.venv-t2v}"
SERVER_VENV="${SERVER_VENV:-$T2V_VENV}"
CONFIG="${CONFIG:-scripts/launch/mlperf_t2v/config/wan_t2v_2node.yaml}"
SERVER_PORT="${SERVER_PORT:-8080}"
LOG_BASE="logs/mlperf_t2v_$(date +%Y%m%d_%H%M%S)"
PERF_ONLY="${PERF_ONLY:-0}"

SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o BatchMode=yes"
SSH="ssh ${SSH_OPTS} ${SSH_USER}@${SERVER_IP}"

OUTPUT_DIR="$( grep 'output_dir' "${CONFIG}" | head -1 | awk '{print $2}' )"
OUTPUT_DIR="${OUTPUT_DIR:-/data/mlperf_t2v/output}"
ANNOTATIONS="$( grep 'annotations_path' "${CONFIG}" | head -1 | awk '{print $2}' )"
ANNOTATIONS="${ANNOTATIONS:-/data/mlperf_t2v/annotations/captions_val2014.json}"

mkdir -p "$LOG_BASE"

echo "=== MLPerf Wan2.2-T2V-A14B Inference Benchmark ==="
echo "  Server node  : ${SSH_USER}@${SERVER_IP}:${SERVER_PORT}"
echo "  Scenario     : ${SCENARIO}"
echo "  Max queries  : ${MAX_QUERY_COUNT}"
echo "  Config       : ${CONFIG}"
echo "  Log dir      : ${LOG_BASE}/"
echo ""

# ---------- 1. Start server on GPU node ----------

echo "[1/4] Starting Wan T2V server on ${SERVER_IP}..."
echo "      (model load ~5-10 min for 14B params — be patient)"
$SSH "bash -lc '
    cd \$(basename $(pwd)) 2>/dev/null || true
    nohup ${SERVER_VENV}/bin/python scripts/launch/mlperf_t2v/server.py \
        --config ${CONFIG} \
        --host 0.0.0.0 --port ${SERVER_PORT} \
        > /tmp/t2v_server.log 2>&1 &
    echo \$! > /tmp/t2v_server.pid
    echo \"Server PID: \$(cat /tmp/t2v_server.pid)\"
'"

# ---------- 2. Wait for server (longer timeout — 14B model is slow to load) ----------

echo "[2/4] Waiting for server at http://${SERVER_IP}:${SERVER_PORT}/healthz ..."
DEADLINE=$(( $(date +%s) + 900 ))  # 15 min
until curl -sf "http://${SERVER_IP}:${SERVER_PORT}/healthz" > /dev/null 2>&1; do
    if [[ $(date +%s) -gt $DEADLINE ]]; then
        echo "ERROR: Server did not become ready within 15 minutes." >&2
        echo "  ssh ${SSH_USER}@${SERVER_IP} 'cat /tmp/t2v_server.log'" >&2
        exit 1
    fi
    echo "  ... waiting ($(( DEADLINE - $(date +%s) ))s remaining)"
    sleep 15
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
    "${T2V_VENV}/bin/python" "scripts/launch/mlperf_t2v/main.py" \
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
        echo "ERROR: SCENARIO must be 'SingleStream', 'Offline', or 'all'" >&2
        exit 1
        ;;
esac

# ---------- 4. Stop server ----------

echo ""
echo "[4/4] Stopping server on ${SERVER_IP}..."
$SSH "bash -lc '
    if [[ -f /tmp/t2v_server.pid ]]; then
        kill \$(cat /tmp/t2v_server.pid) 2>/dev/null || true
        rm -f /tmp/t2v_server.pid
        echo \"Server stopped.\"
    fi
'" || true

# ---------- 5. Accuracy ----------

if [[ "${PERF_ONLY}" != "1" ]]; then
    echo ""
    echo "[+] Running accuracy evaluation..."
    "${T2V_VENV}/bin/python" "scripts/launch/mlperf_t2v/accuracy.py" \
        --output-dir "${OUTPUT_DIR}" \
        --annotations "${ANNOTATIONS}" \
        2>&1 | tee "${LOG_BASE}/accuracy_result.log"
fi

echo ""
echo "=== Benchmark complete ==="
echo "  Logs : ${LOG_BASE}/"
echo "  Run 'make verify-mlperf-t2v' to check CLIP target."
