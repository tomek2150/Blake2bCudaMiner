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
extern "C" cudaError_t blake2b_compute_single_hash_gpu(uint32_t nonce, uint64_t out_h[4]);

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
        // Prepare CPU header
        uint8_t h_copy[80];
        std::memcpy(h_copy, header, 80);
        h_copy[72] = (uint8_t)test_nbits;
        h_copy[73] = (uint8_t)(test_nbits >> 8);
        h_copy[74] = (uint8_t)(test_nbits >> 16);
        h_copy[75] = (uint8_t)(test_nbits >> 24);
        h_copy[76] = (uint8_t)nonce;
        h_copy[77] = (uint8_t)(nonce >> 8);
        h_copy[78] = (uint8_t)(nonce >> 16);
        h_copy[79] = (uint8_t)(nonce >> 24);

        // 1. CPU Reference Hash
        uint8_t cpu_hash[32];
        blake2b_256_cpu_reference(h_copy, cpu_hash);

        // 2. GPU Hash
        uint64_t gpu_h[4];
        blake2b_compute_single_hash_gpu(nonce, gpu_h);

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
    std::cout << "  ✅ ALL 100/100 TESTS PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
