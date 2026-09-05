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
        return bytes([0x50 + height, 0x00])
    raw = bytearray()
    val = height
    while val > 0:
        raw.append(val & 0xFF)
        val >>= 8
    if raw[-1] & 0x80:
        raw.append(0)
    return bytes([len(raw)]) + bytes(raw)


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


def tagged_sha256(tag: str, data: bytes) -> bytes:
    """Core's TaggedHash writer finished with GetSHA256() (single SHA256)."""
    t = hashlib.sha256(tag.encode("utf-8")).digest()
    return hashlib.sha256(t + t + data).digest()


class HeaderV2:
    """Bitcoin Knots 164-Byte Block Header v2 consensus implementation."""

    def __init__(
        self,
        nVersion: int,
        hashPrevBlock: bytes,
        hashMerkleRoot: bytes,
        nTime: int,
        nBits: int,
        nNonce: int = 0,
        m_nonce2: int = 0,
        m_nonce3: int = 0,
        m_extranonce: bytes = b"\x00" * 16,
        m_time_offset: int = 0,
        m_txcount: int = 0,
        m_flags: int = 0,
        m_xor_key_mask_clear_bits: int = 0,
        m_xor_key: bytes = b"\x00" * 16,
        m_height: int = 0,
        m_mm_rhs: bytes = b"\x00" * 32,
    ):
        self.nVersion = nVersion
        self.hashPrevBlock = hashPrevBlock
        self.hashMerkleRoot = hashMerkleRoot
        self.nTime = nTime
        self.nBits = nBits
        self.nNonce = nNonce
        self.m_nonce2 = m_nonce2
        self.m_nonce3 = m_nonce3
        self.m_extranonce = m_extranonce
        self.m_time_offset = m_time_offset
        self.m_txcount = m_txcount
        self.m_flags = m_flags
        self.m_xor_key_mask_clear_bits = m_xor_key_mask_clear_bits
        self.m_xor_key = m_xor_key
        self.m_height = m_height
        self.m_mm_rhs = m_mm_rhs

    def complete_version(self) -> int:
        return 0x80000000 | (self.nVersion & ~0x80000000)

    def time_on_wire(self) -> int:
        if not (self.m_flags & 4):
            return self.nTime
        return (self.nTime - self.m_time_offset) & 0xFFFFFFFF

    def serialize(self) -> bytes:
        b = (
            struct.pack("<I", self.complete_version())
            + self.hashPrevBlock
            + self.hashMerkleRoot
            + struct.pack("<I", self.time_on_wire())
            + struct.pack("<I", self.nBits)
            + struct.pack("<I", self.nNonce)
            + struct.pack("<I", self.m_nonce2)
            + struct.pack("<I", self.m_nonce3)
            + self.m_extranonce
            + struct.pack("<I", self.m_time_offset)
            + struct.pack("<H", self.m_txcount & 0xFFFF)
            + bytes([self.m_flags, self.m_xor_key_mask_clear_bits])
            + self.m_xor_key
            + struct.pack("<i", self.m_height)
            + self.m_mm_rhs
        )
        assert len(b) == 164, f"Invalid serialized header length: {len(b)}"
        return b

    def xor_key_hash(self) -> bytes:
        return tagged_sha256("Bitcoin block hash PoW XOR key", self.m_xor_key)

    def xor_key_mask(self) -> bytes:
        if self.m_xor_key == b"\x00" * 16:
            return b"\x00" * 32
        mask = bytearray(tagged_sha256("Bitcoin block hash PoW XOR mask", self.m_xor_key))
        clear = self.m_xor_key_mask_clear_bits
        for i in range(clear // 8):
            mask[i] = 0
        mask[clear // 8] &= 0xFF >> (clear % 8)
        return bytes(mask)

    def h1(self) -> bytes:
        data = (
            struct.pack("<I", self.complete_version())
            + self.hashPrevBlock[::-1]
            + struct.pack("<i", self.m_height)
            + self.hashMerkleRoot
            + struct.pack("<I", self.time_on_wire())
            + b"\x00"
            + struct.pack("<I", self.nBits)
            + struct.pack("<I", self.m_txcount)
            + bytes([self.m_flags, self.m_xor_key_mask_clear_bits])
            + self.xor_key_hash()
        )
        assert len(data) == 119
        return tagged_sha256("Bitcoin block header 1", data)

    def h2(self) -> bytes:
        return tagged_sha256("Merge-mining hook", self.h1() + b"\x00" * 32 + self.m_mm_rhs)

    def sv1_hash(self) -> bytes:
        payload = struct.pack("<I", 0) + self.h2() + self.m_extranonce
        assert len(payload) == 52
        return blake2b_256(payload)

    def asic_message(self) -> bytes:
        grind = struct.pack("<I", self.nNonce) + struct.pack("<I", self.m_nonce2)
        hidden = bytearray(tagged_sha256("Bitcoin prevblock header, hashed", self.hashPrevBlock[::-1]))
        hidden[0:6] = b"\x00" * 6
        msg = (bytes(hidden) + grind + struct.pack("<I", self.m_time_offset)
               + struct.pack("<I", self.m_nonce3) + self.sv1_hash())
        assert len(msg) == 80
        return msg

    def powhash(self) -> bytes:
        raw = blake2b_256(self.asic_message())
        mask = self.xor_key_mask()
        return bytes(raw[i] ^ mask[i] for i in range(32))

    def block_hash_hex(self) -> str:
        return self.powhash().hex()


class StratumJob:
    """Encapsulates a single Stratum job derived from a GBT template."""

    def __init__(
        self,
        job_id: str,
        template: Dict[str, Any],
        coinbase_address: str,
        extranonce1: str,
        extranonce2_size: int = 4,
        coinbase_sig: str = "",
    ):
        self.job_id = job_id
        self.template = template
        self.coinbase_address = coinbase_address
        self.extranonce1 = extranonce1
        self.extranonce2_size = extranonce2_size
        self.coinbase_sig = coinbase_sig

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

        DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
        self.network_difficulty = DIFF1_TARGET / self.target
        self.stratum_difficulty = min(16.0, self.network_difficulty)

        self.raw_transactions = [
            bytes.fromhex(tx["data"]) for tx in template.get("transactions", [])
        ]
        self.tx_hashes = [
            bytes.fromhex(tx["txid"])[::-1]
            if "txid" in tx
            else double_sha256(bytes.fromhex(tx["data"]))
            for tx in template.get("transactions", [])
        ]

        self._build_coinbase_parts()
        self.merkle_branches = compute_merkle_branches([b"\x00" * 32] + self.tx_hashes)

        # Precompute initial Merkle Root (for extranonce2 = 0)
        coinbase_bytes = (
            bytes.fromhex(self.coinb1)
            + bytes.fromhex(self.extranonce1)
            + (b"\x00" * self.extranonce2_size)
            + bytes.fromhex(self.coinb2)
        )
        coinbase_hash = double_sha256(coinbase_bytes)
        merkle_root = coinbase_hash
        for branch_hex in self.merkle_branches:
            merkle_root = double_sha256(merkle_root + bytes.fromhex(branch_hex))

        extranonce_raw = (bytes.fromhex(self.extranonce1) + (b"\x00" * self.extranonce2_size)).ljust(16, b"\x00")[:16]

        self.header_v2 = HeaderV2(
            nVersion=self.version & ~0x80000000,
            hashPrevBlock=bytes.fromhex(self.previousblockhash)[::-1],
            hashMerkleRoot=merkle_root,
            nTime=self.curtime,
            nBits=self.bits,
            nNonce=0,
            m_extranonce=extranonce_raw,
            m_txcount=1 + len(self.raw_transactions),
            m_height=self.height,
        )
        self.work_msg_80 = self.header_v2.asic_message()

    def _build_coinbase_parts(self) -> None:
        """Constructs coinb1 and coinb2 parts."""
        coinbaseaux = self.template.get("coinbaseaux", {})
        headline_hex = coinbaseaux.get("blake2b_headline", "")
        headline_raw = bytes.fromhex(headline_hex) if headline_hex else b""
        headline_push = bytes([len(headline_raw)]) + headline_raw if headline_raw else b""

        sig_raw = self.coinbase_sig.encode("utf-8")[:32] if self.coinbase_sig else b""
        sig_push = bytes([len(sig_raw)]) + sig_raw if sig_raw else b""

        script_prefix = encode_bip34_height(self.height) + headline_push + sig_push
        total_scriptsig_len = (
            len(script_prefix)
            + (len(self.extranonce1) // 2)
            + self.extranonce2_size
        )

        tx_in = (
            struct.pack("<I", 2)
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
        """Returns parameters for mining.notify with 80-byte Profile 0 ASIC template."""
        version_hex = struct.pack(">I", self.version).hex()
        nbits_hex = struct.pack(">I", self.bits).hex()
        ntime_hex = struct.pack(">I", self.curtime).hex()

        return [
            self.job_id,
            self.work_msg_80.hex(),  # 160 hex characters = 80-byte Profile 0 ASIC message
            self.coinb1,
            self.coinb2,
            self.merkle_branches,
            version_hex,
            nbits_hex,
            ntime_hex,
            clean_jobs,
        ]

    def build_full_block(self, extranonce2: str, ntime_hex: str, nonce_hex: str) -> Tuple[bytes, bytes]:
        """Reconstructs the full 164-byte Header v2 and serialized block for Bitcoin Knots."""
        coinbase_legacy = (
            bytes.fromhex(self.coinb1)
            + bytes.fromhex(self.extranonce1)
            + (b"\x00" * self.extranonce2_size)
            + bytes.fromhex(self.coinb2)
        )
        coinbase_hash = double_sha256(coinbase_legacy)

        merkle_root = coinbase_hash
        for branch_hex in self.merkle_branches:
            merkle_root = double_sha256(merkle_root + bytes.fromhex(branch_hex))

        nonce_val = int(nonce_hex, 16)
        nonce2_val = int(extranonce2, 16) if extranonce2 else 0
        extranonce_raw = (bytes.fromhex(self.extranonce1) + (b"\x00" * self.extranonce2_size)).ljust(16, b"\x00")[:16]

        self.header_v2.hashMerkleRoot = merkle_root
        self.header_v2.nNonce = nonce_val
        self.header_v2.m_nonce2 = nonce2_val
        self.header_v2.m_extranonce = extranonce_raw

        header_164 = self.header_v2.serialize()
        block_hash = self.header_v2.powhash()

        if self.template.get("default_witness_commitment"):
            cb_body = coinbase_legacy[4:-4]
            coinbase_for_block = (
                struct.pack("<I", 2)
                + b"\x00\x01"
                + cb_body
                + b"\x01\x20" + (b"\x00" * 32)
                + struct.pack("<I", 0)
            )
        else:
            coinbase_for_block = coinbase_legacy

        total_tx_count = 1 + len(self.raw_transactions)
        block_bytes = (
            header_164
            + varint(total_tx_count)
            + coinbase_for_block
            + b"".join(self.raw_transactions)
        )
        return block_bytes, block_hash


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
        self.coinbase_sig = ""

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
        self.coinbase_sig = data.get("coinbase-sig") or data.get("coinbase_sig") or ""

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
        last_tx_count = -1
        last_job_time = 0.0
        gbt_params = [{"rules": ["segwit", "blake2b"], "capabilities": ["coinbasevalue", "longpoll"]}]

        while self.running:
            try:
                template = await asyncio.to_thread(self._rpc_call, "getblocktemplate", gbt_params)
                now = time.time()
                if template:
                    prev_hash = template.get("previousblockhash", "")
                    tx_count = len(template.get("transactions", []))
                    is_new_block = (prev_hash != last_block_hash)
                    is_refresh = (not is_new_block and (tx_count != last_tx_count or (now - last_job_time) >= 15.0))

                    if is_new_block or is_refresh:
                        last_block_hash = prev_hash
                        last_tx_count = tx_count
                        last_job_time = now
                        self.job_counter += 1
                        self.current_job = StratumJob(
                            str(self.job_counter),
                            template,
                            self.coinbase_address,
                            self.extranonce1,
                            self.extranonce2_size,
                            coinbase_sig=self.coinbase_sig,
                        )
                        logger.info(
                            "%s Height: %d | Target: %s | Txs: %d",
                            "New Block Template!" if is_new_block else "Refreshed Template:",
                            self.current_job.height,
                            hex(self.current_job.target)[:16] + "...",
                            len(self.current_job.raw_transactions),
                        )
                        await self.broadcast_job(self.current_job, clean_jobs=is_new_block)
            except Exception as e:
                logger.error("Error in GBT polling: %s", e)

            await asyncio.sleep(1.0)

    async def broadcast_job(self, job: StratumJob, clean_jobs: bool = True) -> None:
        diff_msg = json.dumps({"id": None, "method": "mining.set_difficulty", "params": [job.stratum_difficulty]}) + "\n"
        msg = json.dumps({"id": None, "method": "mining.notify", "params": job.get_notify_params(clean_jobs)}) + "\n"
        encoded = (diff_msg + msg).encode("utf-8")
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

                    diff_val = self.current_job.stratum_difficulty if self.current_job else 16.0
                    diff_msg = json.dumps({"id": None, "method": "mining.set_difficulty", "params": [diff_val]}) + "\n"
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
                                logger.info("🚀 BLOCK ACCEPTED BY BITCOIN NODE! Height: %d", self.current_job.height)
                            else:
                                logger.error("⚠️ Node rejected block: %s", sub_res)

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
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 3333
    proxy = SoloStratumServer(config_path=config_file, port=port)
    asyncio.run(proxy.start())


if __name__ == "__main__":
    main()
