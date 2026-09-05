#!/bin/bash
# ==============================================================================
# start.sh - All-in-One Launch Script for Blake2bCudaMiner
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/bin/b2bcudaminer"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
PROXY_SCRIPT="${SCRIPT_DIR}/solo_stratum_proxy.py"

PROXY_PID=""
MINER_PID=""

cleanup() {
    echo ""
    echo "🛑 Shutting down Blake2bCudaMiner & Stratum Proxy..."
    if [ -n "${MINER_PID}" ]; then
        kill "${MINER_PID}" 2>/dev/null || true
    fi
    if [ -n "${PROXY_PID}" ]; then
        kill "${PROXY_PID}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit 0
}

trap cleanup EXIT INT TERM

# Build if binary is missing
if [ ! -f "${BIN_PATH}" ]; then
    echo "⚠️ Miner binary not found. Compiling project..."
    make -C "${SCRIPT_DIR}" all
fi

# If solo_stratum_proxy.py exists locally, manage its lifecycle
if [ -f "${PROXY_SCRIPT}" ]; then
    if [ ! -f "${CONFIG_FILE}" ]; then
        if [ -f "${SCRIPT_DIR}/config.example.json" ]; then
            echo "⚠️ config.json not found! Please copy config.example.json to config.json and fill in your credentials."
            exit 1
        fi
    fi
    # Enforce secure read/write permissions for current user only
    chmod 600 "${CONFIG_FILE}" 2>/dev/null || true

    if ! pgrep -f "solo_stratum_proxy.py" > /dev/null 2>&1; then
        echo "Starting local Solo Stratum Proxy..."
        python3 "${PROXY_SCRIPT}" "${CONFIG_FILE}" &
        PROXY_PID=$!
        sleep 1.5
    else
        echo "Solo Stratum Proxy is already active on port 3333."
    fi
fi

# Start Blake2bCudaMiner
echo "Starting Blake2bCudaMiner..."
"${BIN_PATH}" -o stratum+tcp://127.0.0.1:3333 -u miner -p x "$@" &
MINER_PID=$!
wait "${MINER_PID}"
