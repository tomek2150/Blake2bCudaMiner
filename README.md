> [!NOTE] 
> There are no hidden fees programmed into the code. What you mine belongs to you. I spent several days coding this miner, so I would appreciate a donation if you use it :-)  
> the power is with us bitcoiners ;-)
>
> bc1qq39udmr430qft85r0hvngcuchzc2xujuym0w67 (sha256 chain)  
> bc1q3kqkcx9vdnnrg94d4yze3ds2vm722rr3xrk3f0 (blake2b chain)



# ⚡ Blake2bCudaMiner v2.0.0

High-efficiency, lightweight CUDA GPU miner for **Bitcoin Knots (Blake2b PoW)**, supporting both **DATUM Pool Mining** ([pool.iohzrd.tech](https://pool.iohzrd.tech/)) via Ratum and **Solo Mining** via Stratum Proxy.

---

## 🚀 Quickstart: Pool Mining (DATUM Pool)

### 1. Install Prerequisites (Ubuntu / Debian / WSL2)
Install system packages:
```bash
sudo apt update
sudo apt install -y build-essential nvidia-cuda-toolkit libssl-dev pkg-config git curl
```

Install official Rust & Cargo compiler (required to build `ratum-gateway`):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

> [!NOTE]
> * **Bitcoin Knots Node:** Ensure your local Bitcoin Knots node is running with RPC enabled (`rpcport=38332`, `server=1`).
> * **Native Linux (Bare Metal):** Ensure the proprietary NVIDIA graphics driver is installed (`nvidia-smi` displays your GPU). Inside WSL2, CUDA utilizes the Windows NVIDIA driver automatically.

### 2. Build Blake2bCudaMiner
```bash
git clone https://github.com/tomek2150/Blake2bCudaMiner.git
cd Blake2bCudaMiner
make all
```

### 3. Setup Ratum Gateway
Inside your `Blake2bCudaMiner` directory, clone and build `ratum-gateway`:
```bash
git clone https://github.com/iohzrd/ratum.git ratum
cd ratum
cargo build --release --bin ratum-gateway
cd ..
```
*(If you encounter `cargo: command not found`, run: `source "$HOME/.cargo/env"` or install Rust via step 1).*

Copy the example configuration to `datum_gateway_config.json` and enter your node RPC credentials and payout address:
```bash
cp datum_gateway_config.example.json datum_gateway_config.json
chmod 600 datum_gateway_config.json
nano datum_gateway_config.json
```

Example configuration (`datum_gateway_config.json`):
```json
{
  "bitcoind": {
    "rpcurl": "http://127.0.0.1:38332",
    "rpcuser": "your_rpc_user",
    "rpcpassword": "your_rpc_password"
  },
  "stratum": {
    "listen_addr": "127.0.0.1",
    "listen_port": 3334,
    "vardiff_min": 64
  },
  "mining": {
    "pool_address": "YOUR_BITCOIN_PAYOUT_ADDRESS"
  },
  "datum": {
    "pool_host": "pool.iohzrd.tech",
    "pool_port": 28915
  }
}
```

> [!SECURITY]
> **Protect your RPC credentials!**
> Because `datum_gateway_config.json` contains your node's RPC password, restrict file permissions so only your user account can read it:
> ```bash
> chmod 600 datum_gateway_config.json
> ```

### 4. Start Mining
Make the launch script executable and start mining:
```bash
chmod +x start_ratum.sh
./start_ratum.sh
```
*(Optionally pass a custom address / worker name: `./start_ratum.sh YOUR_BITCOIN_ADDRESS.WORKER`)*

The script automatically:
* Detects the Windows host IP of your Bitcoin Knots node in WSL2.
* Keeps `rpcurl` in `datum_gateway_config.json` synchronized.
* Starts `ratum-gateway` in the background if not already running.
* Launches the GPU miner with full shader saturation on your NVIDIA GPU (~6.8+ GH/s on RTX 5070 Ti).
* Cleanly terminates background processes upon `Ctrl+C`.

---

## ⛏️ Solo Mining (Direct against your Node)

If you prefer solo mining directly to your local Bitcoin Knots node without a pool:
1. Configure `config.json` with your node RPC credentials:
   ```bash
   cp config.example.json config.json
   chmod 600 config.json
   nano config.json
   ```
2. Start the solo miner with the integrated stratum proxy:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

---

## 🧪 Verification & Unit Tests

Verify 100% mathematical consensus accuracy directly on your hardware:
```bash
./bin/test_correctness
```
* Runs 100 nonce verification tests (GPU vs. CPU reference).
* Validates live Bitcoin Knots Mainnet Block 967420 bit-for-bit.

---

## 🔬 Technical Architecture & Deep-Dive Optimizations

For low-level algorithmic details, hardware funnel shifts, PTX `lop3.lut` logic fusion, and zero-folding diagrams, see our dedicated [Technical Architecture & Optimizations Documentation](docs/architecture.md).

---

## 📄 License
Licensed under the **GNU General Public License v3.0 (GPL-3.0)**.
