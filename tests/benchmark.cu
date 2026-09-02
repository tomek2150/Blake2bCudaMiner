/**
 * @file benchmark.cu
 * @brief Benchmark & parameter tuning (block-size sweeper) for Blake2bCudaMiner.
 */

#include "blake2b_cuda.cuh"
#include "blake2b_host.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>

extern "C" cudaError_t blake2b_set_midstate_cuda(const blake2b_midstate_t* host_midstate);
extern "C" cudaError_t blake2b_launch_kernel(
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint64_t target_diff,
    uint32_t* d_found_nonces,
    uint32_t* d_found_count,
    uint32_t block_size,
    cudaStream_t stream
);

void run_benchmark_for_block_size(uint32_t block_size, uint32_t* d_found, uint32_t* d_cnt) {
    const uint32_t batch_size = 64 * 1024 * 1024; // 67,108,864 nonces
    const int iterations = 5;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    blake2b_launch_kernel(0, 1024 * 1024, 0x0000000000000000ULL, d_found, d_cnt, block_size, 0);
    cudaDeviceSynchronize();

    float total_ms = 0.0f;
    float min_ms = 99999.0f;

    for (int i = 0; i < iterations; ++i) {
        uint32_t start_nonce = i * batch_size;
        cudaEventRecord(start);
        blake2b_launch_kernel(start_nonce, batch_size, 0x0000000000000000ULL, d_found, d_cnt, block_size, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }

    double best_hashrate = ((double)batch_size / (min_ms / 1000.0)) / 1e6;
    double avg_hashrate = ((double)batch_size / ((total_ms / iterations) / 1000.0)) / 1e6;

    std::cout << "  • Block size " << std::setw(3) << block_size << ": "
              << "Best: " << std::fixed << std::setprecision(2) << min_ms << " ms (" << best_hashrate / 1000.0 << " GH/s)"
              << " | Avg: " << avg_hashrate / 1000.0 << " GH/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " 🚀 BLAKE2BCUDAMINER - BLOCK-SIZE & GRID TUNING" << std::endl;
    std::cout << "============================================================" << std::endl;

    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "  • GPU:                   " << prop.name << std::endl;
    std::cout << "  • SMs (Multiprocessors): " << prop.multiProcessorCount << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    uint8_t test_header[80];
    for (int i = 0; i < 80; ++i) test_header[i] = (uint8_t)(i * 3 + 7);
    uint32_t test_nbits = 0x1e00ffff;

    blake2b_midstate_t midstate;
    blake2b_precompute_midstate(test_header, test_nbits, &midstate);
    blake2b_set_midstate_cuda(&midstate);

    uint32_t* d_found_nonces;
    uint32_t* d_found_count;
    cudaMalloc(&d_found_nonces, 16 * sizeof(uint32_t));
    cudaMalloc(&d_found_count, sizeof(uint32_t));
    cudaMemset(d_found_count, 0, sizeof(uint32_t));

    std::cout << " ⚡ Testing thread block sizes (threads/block)..." << std::endl;
    std::vector<uint32_t> block_sizes = {64, 128, 256, 512};
    for (uint32_t bs : block_sizes) {
        run_benchmark_for_block_size(bs, d_found_nonces, d_found_count);
    }

    std::cout << "============================================================" << std::endl;

    cudaFree(d_found_nonces);
    cudaFree(d_found_count);
    return 0;
}
