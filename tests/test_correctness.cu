/**
 * @file test_correctness.cu
 * @brief Automated unit test suite verifying 100% byte accuracy of the optimized Blake2b kernel against CPU reference.
 */

#include "blake2b_cuda.cuh"
#include "blake2b_host.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cassert>

extern "C" cudaError_t blake2b_set_midstate_cuda(const blake2b_midstate_t* host_midstate);
extern "C" cudaError_t blake2b_compute_single_hash_gpu(uint64_t full_nonce, uint64_t out_h[4]);

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " 🧪 AUTOMATED CORRECTNESS TEST (100 Nonces GPU vs CPU)" << std::endl;
    std::cout << "============================================================" << std::endl;

    uint8_t header[80];
    for (int i = 0; i < 80; ++i) header[i] = (uint8_t)(i * 5 + 17);
    uint32_t test_nbits = 0x1e00ffff;

    // Precompute midstate on CPU and transfer to GPU
    blake2b_midstate_t midstate;
    blake2b_precompute_midstate(header, test_nbits, &midstate);
    blake2b_set_midstate_cuda(&midstate);

    int passed = 0;
    const int total_tests = 100;

    for (uint32_t nonce = 1000; nonce < 1000 + total_tests; ++nonce) {
        uint32_t nonce2 = 0x12340000 + (nonce * 7);
        // Prepare CPU header (Bitcoin Knots Profile 0: Nonce at bytes 32..35, Nonce2 at bytes 36..39)
        uint8_t h_copy[80];
        std::memcpy(h_copy, header, 80);
        h_copy[32] = (uint8_t)nonce;
        h_copy[33] = (uint8_t)(nonce >> 8);
        h_copy[34] = (uint8_t)(nonce >> 16);
        h_copy[35] = (uint8_t)(nonce >> 24);
        h_copy[36] = (uint8_t)nonce2;
        h_copy[37] = (uint8_t)(nonce2 >> 8);
        h_copy[38] = (uint8_t)(nonce2 >> 16);
        h_copy[39] = (uint8_t)(nonce2 >> 24);

        // 1. CPU Reference Hash
        uint8_t cpu_hash[32];
        blake2b_256_cpu_reference(h_copy, cpu_hash);

        // 2. GPU Hash
        uint64_t full_nonce = ((uint64_t)nonce2 << 32) | nonce;
        uint64_t gpu_h[4];
        blake2b_compute_single_hash_gpu(full_nonce, gpu_h);

        uint8_t gpu_hash[32];
        std::memcpy(gpu_hash, gpu_h, 32);

        if (std::memcmp(cpu_hash, gpu_hash, 32) == 0) {
            passed++;
        } else {
            std::cerr << "  ❌ ERROR at nonce " << nonce << "!" << std::endl;
            return 1;
        }
    }

    std::cout << "  • Tested nonces:    " << total_tests << std::endl;
    std::cout << "  • Passed tests:     " << passed << " / " << total_tests << std::endl;

    // Test Vector: Bitcoin Knots Block 967420
    std::cout << "\n 📦 VERIFYING LIVE BITCOIN KNOTS BLOCK 967420..." << std::endl;
    const char* b967420_hex = "000000000000dfd46bfcea946a738250ba3e381a94e0478536ca95ccd9012684fff0985040735a1500000000a99e9a6afdf7ad5c2490beffa266ce42a90f065eabb96dbcf28f06743df358395fa8c80e";
    uint8_t b967420_msg[80];
    for (int i = 0; i < 80; ++i) {
        unsigned int byte_val;
        sscanf(b967420_hex + i * 2, "%02x", &byte_val);
        b967420_msg[i] = (uint8_t)byte_val;
    }

    blake2b_midstate_t b_mid;
    blake2b_precompute_midstate(b967420_msg, 0x193c2d40, &b_mid);
    blake2b_set_midstate_cuda(&b_mid);

    uint32_t b_nonce = 1352200447; // 0x5098f0ff
    uint32_t b_nonce2 = 358249280; // 0x155a7340
    uint64_t b_full_nonce = ((uint64_t)b_nonce2 << 32) | b_nonce;
    uint64_t b_gpu_h[4];
    blake2b_compute_single_hash_gpu(b_full_nonce, b_gpu_h);

    uint8_t b_gpu_bytes[32];
    std::memcpy(b_gpu_bytes, b_gpu_h, 32);

    const char* exp_hash = "000000000000002207d392c9a6cdb2f58918ce26beebfa19def8efe55c021c3f";
    char got_hash[65];
    for (int i = 0; i < 32; ++i) {
        sprintf(got_hash + i * 2, "%02x", b_gpu_bytes[i]);
    }
    got_hash[64] = '\0';

    std::cout << "  • Block 967420 Hash: " << got_hash << std::endl;
    if (std::strcmp(got_hash, exp_hash) == 0) {
        std::cout << "  ✅ BLOCK 967420 MATCHES NETWORK 100%!" << std::endl;
    } else {
        std::cerr << "  ❌ BLOCK 967420 MISMATCH! Expected: " << exp_hash << std::endl;
        return 1;
    }

    std::cout << "  ✅ ALL TESTS PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
