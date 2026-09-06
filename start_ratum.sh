#!/usr/bin/env bash
# ==============================================================================
# start_ratum.sh - Automated Launcher for Ratum-Gateway & Blake2bCudaMiner
# ==============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -d "$HOME/.cargo/bin" ] && [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi
GATEWAY_DIR="${RATUM_DIR:-$SCRIPT_DIR/ratum}"
if [ ! -d "$GATEWAY_DIR" ] && [ -d "$HOME/ratum" ]; then
    GATEWAY_DIR="$HOME/ratum"
elif [ ! -d "$GATEWAY_DIR" ] && [ -d "$HOME/test/ratum" ]; then
    GATEWAY_DIR="$HOME/test/ratum"
fi
GATEWAY_CONFIG="$GATEWAY_DIR/datum_gateway_config.json"
if [ ! -f "$GATEWAY_CONFIG" ] && [ -f "$SCRIPT_DIR/datum_gateway_config.json" ]; then
    GATEWAY_CONFIG="$SCRIPT_DIR/datum_gateway_config.json"
fi
GATEWAY_BIN="$GATEWAY_DIR/target/release/ratum-gateway"
MINER_BIN="$SCRIPT_DIR/bin/b2bcudaminer"

# 1. Parameter / Address detection (auto-extract from gateway config if not provided)
ADDRESS="${1:-${MINER_ADDRESS:-}}"
if [ -z "$ADDRESS" ] && [ -f "$GATEWAY_CONFIG" ]; then
    CFG_ADDR=$(grep -oP '"pool_address"\s*:\s*"\K[^"]+' "$GATEWAY_CONFIG" || true)
    if [ -n "$CFG_ADDR" ] && [ "$CFG_ADDR" != "YOUR_BITCOIN_PAYOUT_ADDRESS" ]; then
        ADDRESS="${CFG_ADDR}.miner"
    fi
fi

if [ -z "$ADDRESS" ]; then
    echo "================================================================="
    echo " ⚡ Blake2bCudaMiner: Ratum Pool All-in-One Launcher"
    echo "================================================================="
    echo "Usage: $0 [BITCOIN_ADDRESS.WORKER] [EXTRA_MINER_OPTIONS]"
    echo ""
    echo "Example:"
    echo "  $0 bc1q...worker1"
    echo "  $0 bc1q...worker1 -d 0 -b 512"
    echo "================================================================="
    exit 1
fi

# 2. Check if miner binary exists, build if needed
if [ ! -f "$MINER_BIN" ]; then
    echo "  • Building Blake2bCudaMiner..."
    make -C "$SCRIPT_DIR" miner
fi

# 3. Detect environment & Windows Node IP (WSL only)
IS_WSL=0
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=1
fi

# 4. Synchronize Gateway Config with current Windows host IP (only in WSL)
if [ "$IS_WSL" -eq 1 ] && [ -f "$GATEWAY_CONFIG" ]; then
    HOST_IP=$(ip route show default 2>/dev/null | awk '{print $3}')
    if [ -n "$HOST_IP" ]; then
        sed -i -E "s|(\"rpcurl\"[[:space:]]*:[[:space:]]*\"http://)[^:]+(:38332\")|\1$HOST_IP\2|" "$GATEWAY_CONFIG"
        echo "  • WSL detected: Knots Node IP updated -> http://$HOST_IP:38332"
    fi
fi

# 5. Check gateway status or launch in background
GATEWAY_PID=$(pgrep -f "ratum-gateway.*datum_gateway_config.json" || true)
STARTED_GATEWAY=0

if [ -z "$GATEWAY_PID" ]; then
    if [ ! -f "$GATEWAY_BIN" ]; then
        echo "  [ERROR] ratum-gateway binary not found at: $GATEWAY_BIN"
        echo "  Please compile it first: cd $GATEWAY_DIR && cargo build --release --bin ratum-gateway"
        echo "  (If cargo is missing: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)"
        exit 1
    fi
    echo "  • Starting ratum-gateway in background..."
    (cd "$GATEWAY_DIR" && "$GATEWAY_BIN" -c "$GATEWAY_CONFIG" > "$GATEWAY_DIR/gateway.log" 2>&1) &
    GATEWAY_PID=$!
    STARTED_GATEWAY=1
    sleep 2
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
        echo "  [ERROR] Failed to start ratum-gateway. See log: $GATEWAY_DIR/gateway.log"
        exit 1
    fi
    echo "  • ratum-gateway running (PID $GATEWAY_PID, Port 3334)"
else
    echo "  • ratum-gateway already running (PID $GATEWAY_PID)"
fi

# 6. Clean shutdown handler (Ctrl+C)
cleanup() {
    echo ""
    echo "  • Shutting down miner..."
    if [ "$STARTED_GATEWAY" -eq 1 ] && [ -n "$GATEWAY_PID" ]; then
        echo "  • Stopping background gateway (PID $GATEWAY_PID)..."
        kill "$GATEWAY_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 7. Launch Miner
echo "  • Launching Blake2bCudaMiner for $ADDRESS..."
if [ "$1" = "$ADDRESS" ]; then
    shift || true
fi
"$MINER_BIN" -o stratum+tcp://127.0.0.1:3334 -u "$ADDRESS" -p x "$@"
