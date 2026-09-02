#!/bin/bash
# ==============================================================================
# start.sh - All-in-One Launch Script for Blake2bCudaMiner
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/bin/b2bcudaminer"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
PROXY_SCRIPT="${SCRIPT_DIR}/solo_stratum_proxy.py"

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

    if ! pgrep -f "solo_stratum_proxy.py" > /dev/null 2>&1; then
        echo "Starting local Solo Stratum Proxy..."
        python3 "${PROXY_SCRIPT}" "${CONFIG_FILE}" &
        PROXY_PID=$!
        trap "kill ${PROXY_PID} 2>/dev/null || true" EXIT INT TERM
        sleep 1.5
    else
        echo "Solo Stratum Proxy is already active on port 3333."
    fi
fi

# Start Blake2bCudaMiner
echo "Starting Blake2bCudaMiner..."
exec "${BIN_PATH}" -o stratum+tcp://127.0.0.1:3333 -u miner -p x "$@"
