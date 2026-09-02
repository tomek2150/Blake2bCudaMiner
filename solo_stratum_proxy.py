#!/usr/bin/env python3
"""
Solo Stratum Proxy for Bitcoin Knots (Blake2b PoW)
Bridges between Bitcoin Knots RPC (getblocktemplate) and Blake2bCudaMiner (Stratum v1 protocol).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SoloProxy")


def get_wsl_host_ip() -> str:
    """Detects Windows Host IP in WSL from /proc/net/route."""
    try:
        with open("/proc/net/route", "r") as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) >= 3 and fields[1] == "00000000":
                    ip_int = int(fields[2], 16)
                    ip_bytes = ip_int.to_bytes(4, byteorder="little")
                    return ".".join(map(str, ip_bytes))
    except Exception:
        pass
    return "127.0.0.1"


def double_sha256(data: bytes) -> bytes:
    """Computes SHA256(SHA256(data))."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def blake2b_256(data: bytes) -> bytes:
    """Computes Blake2b-256 for Block Header PoW."""
    return hashlib.blake2b(data, digest_size=32).digest()


def compact_to_target(bits: int) -> int:
    """Converts nBits (compact format) to 256-bit target integer."""
    exponent = bits >> 24
    mantissa = bits & 0x007FFFFF
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))
    return target


def varint(n: int) -> bytes:
    """Encodes an integer as a Bitcoin variable-length integer (VarInt)."""
    if n < 0xFD:
        return struct.pack("<B", n)
    elif n <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", n)
    elif n <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", n)
    else:
        return b"\xff" + struct.pack("<Q", n)


def serialize_script(script: bytes) -> bytes:
    """Serializes a script with its VarInt length prefix."""
    return varint(len(script)) + script


def address_to_script_pubkey(address: str) -> bytes:
    """Converts a Bitcoin Base58 or Bech32 address into its ScriptPubKey."""
    # SegWit / Bech32 (bc1...)
    if address.lower().startswith(("bc1", "tb1", "bcrt1")):
        charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
        hrp, data_part = address.lower().split("1", 1)
        data = [charset.find(c) for c in data_part[:-6]]
        version = data[0]
        acc = 0
        bits = 0
        ret = []
        for val in data[1:]:
            acc = (acc << 5) | val
            bits += 5
            while bits >= 8:
                bits -= 8
                ret.append((acc >> bits) & 0xFF)
        prog = bytes(ret)
        if version == 0:
            return bytes([0x00, len(prog)]) + prog
        else:
            return bytes([0x50 + version, len(prog)]) + prog

    # Legacy Base58
    b58_digits = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = 0
    for c in address:
        n = n * 58 + b58_digits.index(c)
    decoded = n.to_bytes(25, byteorder="big")
    version_byte = decoded[0]
    h160 = decoded[1:21]

    if version_byte in (0x00, 0x6F):  # P2PKH (Mainnet / Testnet)
        return b"\x76\xa9\x14" + h160 + b"\x88\xac"
    elif version_byte in (0x05, 0xC4):  # P2SH (Mainnet / Testnet)
        return b"\xa9\x14" + h160 + b"\x87"
    return b"\x76\xa9\x14" + h160 + b"\x88\xac"


def encode_bip34_height(height: int) -> bytes:
    """Encodes block height for BIP34 coinbase scriptSig."""
    if height == 0:
        return b"\x00"
    elif 1 <= height <= 16:
        return bytes([0x50 + height])
    raw = height.to_bytes((height.bit_length() + 7) // 8, byteorder="little")
    return bytes([len(raw)]) + raw


def swap32_hex(hex_str: str) -> str:
    """Swaps byte order of 32-bit words for Stratum compatibility."""
    raw = bytes.fromhex(hex_str)
    result = bytearray()
    for i in range(0, len(raw), 4):
        result.extend(raw[i : i + 4][::-1])
    return result.hex()


def compute_merkle_branches(txids: List[bytes]) -> List[str]:
    """Calculates Merkle branches for Stratum mining.notify."""
    branches = []
    current_hashes = list(txids)
    while len(current_hashes) > 1:
        if len(current_hashes) % 2 != 0:
            current_hashes.append(current_hashes[-1])
        branches.append(current_hashes[1].hex())
        next_hashes = []
        for i in range(0, len(current_hashes), 2):
            combined = current_hashes[i] + current_hashes[i + 1]
            next_hashes.append(double_sha256(combined))
        current_hashes = next_hashes
    return branches


class StratumJob:
    """Encapsulates a single Stratum job derived from a GBT template."""

    def __init__(
        self,
        job_id: str,
        template: Dict[str, Any],
        coinbase_address: str,
        extranonce1: str,
        extranonce2_size: int = 4,
    ):
        self.job_id = job_id
        self.template = template
        self.coinbase_address = coinbase_address
        self.extranonce1 = extranonce1
        self.extranonce2_size = extranonce2_size

        self.height = template["height"]
        self.version = template["version"]
        self.previousblockhash = template["previousblockhash"]
        self.curtime = template["curtime"]
        self.bits = int(template["bits"], 16)
        self.target = (
            int(template["target"], 16)
            if "target" in template
            else compact_to_target(self.bits)
        )

        self.raw_transactions = [
            bytes.fromhex(tx["data"]) for tx in template.get("transactions", [])
        ]
        self.tx_hashes = [double_sha256(tx) for tx in self.raw_transactions]

        self._build_coinbase_parts()
        self.merkle_branches = compute_merkle_branches([b"\x00" * 32] + self.tx_hashes)

    def _build_coinbase_parts(self) -> None:
        """Constructs coinb1 and coinb2 parts."""
        script_prefix = encode_bip34_height(self.height) + b"/SoloBlake2bCudaMiner/"
        total_scriptsig_len = (
            len(script_prefix)
            + (len(self.extranonce1) // 2)
            + self.extranonce2_size
        )

        tx_in = (
            struct.pack("<I", 1)
            + varint(1)
            + b"\x00" * 32
            + struct.pack("<I", 0xFFFFFFFF)
            + varint(total_scriptsig_len)
            + script_prefix
        )
        self.coinb1 = tx_in.hex()

        sequence = struct.pack("<I", 0xFFFFFFFF)
        coinbase_val = self.template["coinbasevalue"]
        payout_script = address_to_script_pubkey(self.coinbase_address)
        tx_out_payout = struct.pack("<Q", coinbase_val) + serialize_script(payout_script)

        witness_commitment = self.template.get("default_witness_commitment")
        if witness_commitment:
            tx_outs = (
                varint(2)
                + tx_out_payout
                + struct.pack("<Q", 0)
                + serialize_script(bytes.fromhex(witness_commitment))
            )
        else:
            tx_outs = varint(1) + tx_out_payout

        self.coinb2 = (sequence + tx_outs + struct.pack("<I", 0)).hex()

    def get_notify_params(self, clean_jobs: bool = True) -> List[Any]:
        """Returns parameters for mining.notify."""
        prevhash_swapped = swap32_hex(self.previousblockhash)
        version_hex = struct.pack(">I", self.version).hex()
        nbits_hex = struct.pack(">I", self.bits).hex()
        ntime_hex = struct.pack(">I", self.curtime).hex()

        return [
            self.job_id,
            prevhash_swapped,
            self.coinb1,
            self.coinb2,
            self.merkle_branches,
            version_hex,
            nbits_hex,
            ntime_hex,
            clean_jobs,
        ]

    def build_full_block(self, extranonce2: str, ntime_hex: str, nonce_hex: str) -> Tuple[bytes, bytes]:
        """Reconstructs the full 80-byte header and serialized block."""
        coinbase_bytes = (
            bytes.fromhex(self.coinb1)
            + bytes.fromhex(self.extranonce1)
            + bytes.fromhex(extranonce2)
            + bytes.fromhex(self.coinb2)
        )
        coinbase_hash = double_sha256(coinbase_bytes)

        merkle_root = coinbase_hash
        for branch_hex in self.merkle_branches:
            branch_bytes = bytes.fromhex(branch_hex)
            merkle_root = double_sha256(merkle_root + branch_bytes)

        if len(ntime_hex) == 8:
            ntime_bytes = struct.pack("<I", int(ntime_hex, 16))
        else:
            ntime_bytes = bytes.fromhex(ntime_hex)

        nonce_val = int(nonce_hex, 16)
        nonce_bytes = struct.pack("<I", nonce_val)

        header_bytes = (
            struct.pack("<I", self.version)
            + bytes.fromhex(self.previousblockhash)[::-1]
            + merkle_root
            + ntime_bytes
            + struct.pack("<I", self.bits)
            + nonce_bytes
        )

        header_pow_hash = blake2b_256(header_bytes)[::-1]

        total_tx_count = 1 + len(self.raw_transactions)
        block_bytes = (
            header_bytes
            + varint(total_tx_count)
            + coinbase_bytes
            + b"".join(self.raw_transactions)
        )
        return block_bytes, header_pow_hash


class SoloStratumServer:
    """Async Stratum Server bridging to Bitcoin Knots GBT."""

    def __init__(self, config_path: str, host: str = "0.0.0.0", port: int = 3333):
        self.config_path = Path(config_path)
        self.host = host
        self.port = port
        self.rpc_url = ""
        self.rpc_user = ""
        self.rpc_pass = ""
        self.coinbase_address = ""

        self._load_config()

        self.clients: List[asyncio.StreamWriter] = []
        self.current_job: Optional[StratumJob] = None
        self.job_counter = 0
        self.extranonce1 = "00000001"
        self.extranonce2_size = 4
        self.running = True

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_url = data["url"]
        self.rpc_user = data["user"]
        self.rpc_pass = data["pass"]
        self.coinbase_address = data.get("coinbase-addr") or data.get("coinbase_addr")

        wsl_host_ip = get_wsl_host_ip()
        if ("127.0.0.1" in raw_url or "localhost" in raw_url) and wsl_host_ip != "127.0.0.1":
            self.rpc_url = raw_url.replace("127.0.0.1", wsl_host_ip).replace("localhost", wsl_host_ip)
            logger.info("WSL detected: Auto-routed Node URL to Host IP -> %s", self.rpc_url)
        else:
            self.rpc_url = raw_url

    def _rpc_call(self, method: str, params: List[Any]) -> Any:
        payload = json.dumps({"jsonrpc": "1.0", "id": "soloproxy", "method": method, "params": params}).encode("utf-8")
        req = urllib.request.Request(self.rpc_url, data=payload, headers={"Content-Type": "application/json"})
        auth_string = f"{self.rpc_user}:{self.rpc_pass}"
        auth_bytes = base64.b64encode(auth_string.encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {auth_bytes}")

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                if result.get("error"):
                    logger.error("RPC Error in %s: %s", method, result["error"])
                    return None
                return result.get("result")
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            logger.error("HTTP Error %d in %s: %s", e.code, method, err_msg)
            return None
        except Exception as e:
            logger.error("Connection Error in %s: %s", method, e)
            return None

    async def poll_node_loop(self) -> None:
        last_block_hash = ""
        gbt_params = [{"rules": ["segwit", "blake2b"], "capabilities": ["coinbasevalue", "longpoll"]}]

        while self.running:
            try:
                template = await asyncio.to_thread(self._rpc_call, "getblocktemplate", gbt_params)
                if template and template.get("previousblockhash") != last_block_hash:
                    last_block_hash = template["previousblockhash"]
                    self.job_counter += 1
                    self.current_job = StratumJob(
                        str(self.job_counter),
                        template,
                        self.coinbase_address,
                        self.extranonce1,
                        self.extranonce2_size,
                    )
                    logger.info(
                        "New Block Template! Height: %d | Target: %s | Txs: %d",
                        self.current_job.height,
                        hex(self.current_job.target)[:16] + "...",
                        len(self.current_job.raw_transactions),
                    )
                    await self.broadcast_job(self.current_job, clean_jobs=True)
            except Exception as e:
                logger.error("Error in GBT polling: %s", e)

            await asyncio.sleep(1.0)

    async def broadcast_job(self, job: StratumJob, clean_jobs: bool = True) -> None:
        msg = json.dumps({"id": None, "method": "mining.notify", "params": job.get_notify_params(clean_jobs)}) + "\n"
        encoded = msg.encode("utf-8")
        for writer in list(self.clients):
            try:
                writer.write(encoded)
                await writer.drain()
            except Exception:
                if writer in self.clients:
                    self.clients.remove(writer)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        self.clients.append(writer)

        try:
            while self.running:
                line = await reader.readline()
                if not line:
                    break
                raw_str = line.decode("utf-8").strip()
                if not raw_str:
                    continue

                msg = json.loads(raw_str)
                method = msg.get("method")
                msg_id = msg.get("id")
                params = msg.get("params", [])

                if method == "mining.subscribe":
                    resp = {
                        "id": msg_id,
                        "result": [
                            [["mining.set_difficulty", "sub1"], ["mining.notify", "sub2"]],
                            self.extranonce1,
                            self.extranonce2_size,
                        ],
                        "error": None,
                    }
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()

                elif method == "mining.authorize":
                    resp = {"id": msg_id, "result": True, "error": None}
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))

                    # Stratum difficulty: 16.0
                    diff_msg = json.dumps({"id": None, "method": "mining.set_difficulty", "params": [16.0]}) + "\n"
                    writer.write(diff_msg.encode("utf-8"))
                    await writer.drain()

                    if self.current_job:
                        notify_msg = json.dumps({"id": None, "method": "mining.notify", "params": self.current_job.get_notify_params(True)}) + "\n"
                        writer.write(notify_msg.encode("utf-8"))
                        await writer.drain()

                elif method == "mining.submit":
                    resp = {"id": msg_id, "result": True, "error": None}
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()

                    if self.current_job and len(params) >= 5:
                        _, _, xn2, ntime, nonce = params[:5]
                        raw_block, block_hash = self.current_job.build_full_block(xn2, ntime, nonce)
                        hash_int = int.from_bytes(block_hash, byteorder="big")

                        if hash_int <= self.current_job.target:
                            logger.info("🎉 VALID BLOCK FOUND! Hash: %s", block_hash.hex())
                            sub_res = await asyncio.to_thread(self._rpc_call, "submitblock", [raw_block.hex()])
                            if sub_res is None or sub_res == "":
                                logger.info("🚀 BLOCK ACCEPTED BY BITCOIN NODE!")

                elif method == "mining.extranonce.subscribe":
                    resp = {"id": msg_id, "result": True, "error": None}
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()

        except Exception as e:
            logger.warning("Connection closed for miner %s: %s", addr, e)
        finally:
            if writer in self.clients:
                self.clients.remove(writer)
            writer.close()
            await writer.wait_closed()

    async def start(self) -> None:
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info("Solo Stratum Proxy running on %s:%d", self.host, self.port)
        logger.info("Connect Blake2bCudaMiner via: stratum+tcp://127.0.0.1:%d", self.port)

        async with server:
            await asyncio.gather(server.serve_forever(), self.poll_node_loop())


def main() -> None:
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "config.json")
    proxy = SoloStratumServer(config_path=config_file)
    asyncio.run(proxy.start())


if __name__ == "__main__":
    main()
