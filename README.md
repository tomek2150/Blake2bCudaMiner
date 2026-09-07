> [!NOTE] 
> all previous versions before v1.3.2 had a serious bug, please upgrade to the newest version!!!

> [!NOTE] 
> There are no hidden fees programmed into the code. What you mine belongs to you. I spent several days coding this miner. I would appreciate a donation if you get some rewards with it :-)  
> the power is with us bitcoiners ;-)
>
> bc1qq39udmr430qft85r0hvngcuchzc2xujuym0w67 (sha256 chain)  
> bc1q3kqkcx9vdnnrg94d4yze3ds2vm722rr3xrk3f0 (blake2b chain)

# ⚡ Blake2bCudaMiner v2.0.0

High-efficiency, lightweight CUDA GPU miner for **Bitcoin Knots (Blake2b PoW)**, supporting both **DATUM Pool Mining** (e.g. [pool.iohzrd.tech](https://pool.iohzrd.tech/)) via Ratum and **Solo Mining** via an integrated Stratum proxy.

---

## 📋 System Prerequisites

| Component | Pure Native Linux | Windows 10/11 with WSL2 |
| :--- | :--- | :--- |
| **OS Environment** | Ubuntu 22.04 / 24.04 LTS or Debian 12 | Windows 10/11 running WSL2 (Ubuntu) |
| **NVIDIA Driver** | Latest NVIDIA Linux Driver (`>= 535`) | Standard NVIDIA Windows Driver (paravirtualized in WSL) |
| **CUDA Toolkit** | CUDA 12+ (`nvidia-cuda-toolkit`) | CUDA 12+ installed inside WSL2 |
| **Compilers** | GCC/G++ (C++17), NVCC, Rust/Cargo | GCC/G++ (C++17), NVCC, Rust/Cargo inside WSL2 |
| **Libraries** | OpenSSL (`libssl-dev`), Python 3 | OpenSSL (`libssl-dev`), Python 3 inside WSL2 |

### 1. Install System Packages & Toolchains
Run inside your Linux / WSL2 terminal:
```bash
sudo apt update
sudo apt install -y build-essential nvidia-cuda-toolkit libssl-dev pkg-config python3 git curl
```

Install official Rust & Cargo compiler (required for `ratum-gateway`):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

---

## ⚙️ Step 0: Bitcoin Knots Node Setup (`bitcoin.conf`)

Both pool mining and solo mining communicate with a local **Bitcoin Knots** node via JSON-RPC on port `38332`.

### Option A: Pure Native Linux (Node on same Linux machine)
Add to `~/.bitcoin/bitcoin.conf`:
```ini
server=1
daemon=1
txindex=1

# Dedicated Blake2b ports (avoids collision with standard Bitcoin Core)
port=8334
rpcport=38332

# RPC credentials
rpcuser=YOUR_RPC_USER
rpcpassword=YOUR_RPC_PASSWORD

# Local loopback binding
rpcallowip=127.0.0.1
rpcbind=127.0.0.1
```

### Option B: Windows + WSL2 Setup (Node on Windows, Miner in WSL2)
1. Add to `bitcoin.conf` on Windows (e.g. `%APPDATA%\Bitcoin\bitcoin.conf`):
   ```ini
   server=1
   txindex=1
   port=8334
   rpcport=38332
   rpcuser=YOUR_RPC_USER
   rpcpassword=YOUR_RPC_PASSWORD

   # Allow local loopback and WSL2 virtual network subnet
   rpcallowip=127.0.0.1
   rpcallowip=172.16.0.0/12
   rpcbind=0.0.0.0
   ```
2. Allow RPC traffic through Windows Firewall (open **PowerShell as Administrator** on Windows):
   ```powershell
   New-NetFirewallRule -DisplayName "Bitcoin Knots RPC for WSL" -Direction Inbound -LocalPort 38332 -Protocol TCP -Action Allow -RemoteAddress 172.16.0.0/12
   ```
3. Restart your Bitcoin Knots node.

---

## 🔨 Step 1: Clone and Build Blake2bCudaMiner

```bash
git clone https://github.com/tomek2150/Blake2bCudaMiner.git
cd Blake2bCudaMiner
make all
```

---

## 🏊 Mining Mode 1: DATUM Pool Mining

Connects to the DATUM mining pool ([pool.iohzrd.tech](https://pool.iohzrd.tech/)) via the local `ratum-gateway`.

### 1. Build Ratum Gateway
Inside your `Blake2bCudaMiner` directory:
```bash
git clone https://github.com/iohzrd/ratum.git ratum
cd ratum
cargo build --release --bin ratum-gateway
cd ..
```
*(If you see `cargo: command not found`, run `source "$HOME/.cargo/env"`).*

### 2. Configure Gateway
Copy the template configuration into the miner directory and configure your credentials:
```bash
cp datum_gateway_config.example.json datum_gateway_config.json
chmod 600 datum_gateway_config.json
nano datum_gateway_config.json
```
Fill in your node RPC credentials, payout address, and pool parameters (works for all pool addresses, this is just an example):
```json
{
  "bitcoind": {
    "rpcurl": "http://127.0.0.1:38332",
    "rpcuser": "YOUR_RPC_USER",
    "rpcpassword": "YOUR_RPC_PASSWORD",
    "work_update_seconds": 30
  },
  "stratum": {
    "listen_addr": "127.0.0.1",
    "listen_port": 3334,
    "vardiff_min": 64
  },
  "mining": {
    "pool_address": "YOUR_BITCOIN_PAYOUT_ADDRESS",
    "coinbase_tag_primary": "Blake2bMiner",
    "coinbase_tag_secondary": "your_miner_name"
  },
  "datum": {
    "pool_host": "pool.iohzrd.tech",
    "pool_port": 28915,
    "pool_pubkey": "POOL_PUBLIC_KEY",
    "pool_pass_workers": true,
    "pool_pass_full_users": true,
    "pooled_mining_only": true
  }
}
```

#### Configuration Field Reference:
| Field | Description | Default / Recommended |
| :--- | :--- | :--- |
| `bitcoind.rpcurl` | Bitcoin Knots RPC address (*Auto-updated to Windows host IP in WSL2 by `start_ratum.sh`*) | `http://127.0.0.1:38332` |
| `bitcoind.rpcuser` / `rpcpassword` | RPC credentials configured in your `bitcoin.conf` | `YOUR_RPC_USER` / `YOUR_RPC_PASSWORD` |
| `bitcoind.work_update_seconds` | Interval (seconds) to poll Bitcoin Knots for new block templates | `30` |
| `stratum.listen_addr` / `listen_port` | Local Stratum endpoint where the GPU miner connects | `127.0.0.1:3334` |
| `stratum.vardiff_min` | Minimum stratum share difficulty | `64` |
| `mining.pool_address` | Your Bitcoin Knots payout address (`bc1q...`) | `YOUR_BITCOIN_PAYOUT_ADDRESS` |
| `mining.coinbase_tag_primary` | Primary tag inscribed into the coinbase transaction | `"Blake2bMiner"` |
| `mining.coinbase_tag_secondary` | Secondary miner identifier / signature | `"your_miner_name"` |
| `datum.pool_host` / `pool_port` | Remote DATUM pool endpoint | `pool.iohzrd.tech:28915` |
| `datum.pool_pubkey` | Public key of the DATUM pool operator | Provided by pool dashboard / operator |
| `datum.pool_pass_workers` | Forward individual worker names (`address.worker`) to pool dashboard | `true` |
| `datum.pool_pass_full_users` | Pass full user strings to pool | `true` |
| `datum.pooled_mining_only` | Enforce pooled mining payouts | `true` |

> [!SECURITY]
> **Protect your RPC credentials!**
> Because configuration files contain your node's RPC password, keep permissions restricted:
> ```bash
> chmod 600 datum_gateway_config.json
> ```

### 3. Launch Pool Mining
```bash
chmod +x start_ratum.sh
./start_ratum.sh
```
*(Optionally specify a custom address / worker tag: `./start_ratum.sh YOUR_ADDRESS.worker1`)*

The script automatically:
* Auto-detects the Windows host IP when running in WSL2 and keeps `rpcurl` synchronized.
* Starts `ratum-gateway` in the background if not already active.
* Automatically extracts your payout address from `datum_gateway_config.json`.
* Launches the GPU miner with full shader saturation on your NVIDIA GPU (~6.88+ GH/s on RTX 5070 Ti).
* Gracefully shuts down both the miner and background gateway upon `Ctrl+C`.

### 4. Dashboards & Live Monitoring
* **Miner Terminal:** Displays live hashrates and accepted shares (`Shares: X/X`).
* **Local Gateway Web Dashboard:** Open [http://localhost:8000](http://localhost:8000) in your browser for real-time worker connections, difficulty, and share statistics.
* **DATUM Pool Dashboard:** Monitor your payout address and hashrate at [https://pool.iohzrd.tech/](https://pool.iohzrd.tech/).

---

## ⛏️ Mining Mode 2: Solo Mining (Direct Node)

Mines directly against your own Bitcoin Knots node via the included lightweight Stratum proxy (`solo_stratum_proxy.py`).

### 1. Configure Solo Credentials
```bash
cp config.example.json config.json
chmod 600 config.json
nano config.json
```
```json
{
  "algo": "blake2b",
  "url": "http://127.0.0.1:38332",
  "user": "YOUR_RPC_USER",
  "pass": "YOUR_RPC_PASSWORD",
  "coinbase-addr": "YOUR_BITCOIN_PAYOUT_ADDRESS",
  "coinbase-sig": "your-miner-tag"
}
```

### 2. Launch Solo Mining
```bash
chmod +x start.sh
./start.sh
```
`start.sh` automatically manages `solo_stratum_proxy.py`, routes to the node (with WSL auto-IP detection), and launches the miner. When a valid block is found, it is submitted directly to your node via `submitblock`.

---

## ⚙️ Advanced CLI Execution

If you already run a Stratum instance or wish to invoke the binary directly:

```bash
./bin/b2bcudaminer -o stratum+tcp://127.0.0.1:3334 -u <wallet_address>.worker1 -p x -b 512 -d 0
```

### Command-Line Arguments:
* `-o, --url <url>` : Stratum server address (Default: `127.0.0.1:3333`)
* `-u, --user <username>` : Stratum username or payout address (Default: `miner`)
* `-p, --pass <password>` : Stratum password (Default: `x`)
* `-d, --device <id>` : CUDA device ID to use (Default: `0`)
* `-b, --block-size <n>` : Threads per CUDA block: `[64, 128, 256, 512]` (Default: `512`)
* `-h, --help` : Display interactive ASCII help screen

---

## 🧪 Verification & Benchmarks

Verify consensus accuracy and benchmark shader throughput on your hardware:

```bash
# 1. Automated Unit Test Suite (100 nonces GPU vs CPU + Live Mainnet Block 967420 verification)
./bin/test_correctness

# 2. Block-Size Sweeper & Multi-Stream Pipelining Benchmark
./bin/benchmark
```

---

## 📁 Repository Structure

```text
Blake2bCudaMiner/
├── include/
│   ├── blake2b_cuda.cuh       # Hardware funnel shifts, zero-folding macros & midstate types
│   ├── blake2b_host.h         # Host header for CPU precomputation
│   └── stratum_client.h       # Lightweight Stratum v1 TCP client (Pool & Solo)
├── src/
│   ├── blake2b_host.cpp       # CPU midstate precomputation & reference hashing
│   ├── blake2b_kernel.cu      # Optimized CUDA search kernel (Constant Memory broadcast)
│   ├── stratum_client.cpp     # Asynchronous Stratum client (PoT & Profile 0 support)
│   └── main.cpp               # Standalone CLI miner executable
├── tests/
│   ├── test_correctness.cu    # Automated unit test suite (100 nonces + Block 967420)
│   └── benchmark.cu           # Throughput, block-size tuning & stream benchmark
├── docs/
│   └── architecture.md        # Technical architecture, PTX lop3, pipeline diagrams
├── config.example.json        # Template configuration for Solo Mining
├── datum_gateway_config.example.json # Template configuration for Ratum Pool Mining
├── start.sh                   # All-in-one launcher for Solo Mining
├── start_ratum.sh             # All-in-one launcher for DATUM Pool Mining
├── solo_stratum_proxy.py      # Standalone Solo Stratum Proxy (GBT bridge)
├── Makefile                   # Multi-architecture NVCC & G++ build system
├── README.md                  # Project documentation
├── LICENSE                    # GNU General Public License v3.0 (GPL-3.0)
└── .gitignore                 # Repository exclusions
```

---

## 🔬 Technical Architecture & Deep-Dive Optimizations

For low-level algorithmic details, hardware funnel shifts, PTX `lop3.lut` logic fusion, and zero-folding diagrams, see our dedicated [Technical Architecture & Optimizations Documentation](docs/architecture.md).

---

## 📄 License
Licensed under the **GNU General Public License v3.0 (GPL-3.0)**.
